from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from extensions.crawl.state import CrawlState
from extensions.io.output_paths import ensure_output_root
from extensions.utils.versioning import safe_version_metadata

from .db_migration import rebuild_db_to_current_schema
from .json_io import read_json_file, validate_json_bytes, write_json_file
from .paths import company_base, first_existing, url_index_candidates
from .types import EnforceMaxPagesReport, EnforcedCompany
from .url_index import (
    resolve_markdown_and_html_paths_from_url_index_entry,
    url_index_meta_patch_max_pages,
)

VERSION_META_KEY = "version_metadata"


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _read_text_best_effort(path: Path, *, max_bytes: int = 512_000) -> str:
    b = path.read_bytes()
    if len(b) > max_bytes:
        b = b[:max_bytes]
    return b.decode("utf-8", errors="replace")


_TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
_META_RE = re.compile(r"(?is)<meta\s+[^>]*>")
_NAME_RE = re.compile(r'(?is)\bname\s*=\s*["\']([^"\']+)["\']')
_CONTENT_RE = re.compile(r'(?is)\bcontent\s*=\s*["\']([^"\']*)["\']')


def _build_head_doc_from_html(html_text: str) -> str:
    """
    Deterministic, dependency-free head extraction:
      - title repeated 3x
      - description repeated 2x
      - keywords once
      - then all meta contents concatenated
    """
    parts: List[str] = []

    m = _TITLE_RE.search(html_text)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        if title:
            parts.append((title + " ") * 3)

    meta_kv: Dict[str, str] = {}
    for mm in _META_RE.finditer(html_text):
        tag = mm.group(0)
        nm = _NAME_RE.search(tag)
        cm = _CONTENT_RE.search(tag)
        if not (nm and cm):
            continue
        k = nm.group(1).strip().lower()
        v = re.sub(r"\s+", " ", cm.group(1)).strip()
        if not k or not v:
            continue
        meta_kv[k] = v

    desc = meta_kv.get("description", "")
    if desc:
        parts.append((desc + " ") * 2)

    kw = meta_kv.get("keywords", "")
    if kw:
        parts.append(kw)

    if meta_kv:
        parts.append(" ".join([v for v in meta_kv.values() if v]))

    return " ".join(parts).strip()


def _move_to_hidden_subdir(src: Path, *, hidden_dir: Path) -> Path:
    hidden_dir.mkdir(parents=True, exist_ok=True)
    dst = hidden_dir / src.name
    if dst.exists():
        raise RuntimeError(f"destination collision: {dst}")
    src.rename(dst)
    return dst


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None or isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).strip()
    if not s:
        return default
    return int(float(s))


async def enforce_max_pages_async(
    *,
    out_dir: Path,
    db_path: Path,
    max_pages: Optional[int],
    selector: str = "crawl_finished_true",
    apply_only_if_crawl_finished: bool = True,
    write_global_state: bool = True,
    dry_run: bool = False,
    max_examples: int = 50,
    concurrency: int = 32,
) -> EnforceMaxPagesReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    rebuild_db_to_current_schema(actual_db_path)

    conc = max(1, int(concurrency))

    version_meta = safe_version_metadata(
        component="enforce_max_pages", start_path=Path(__file__)
    )
    if not isinstance(version_meta, dict):
        version_meta = {"component": "enforce_max_pages", "available": False}

    from extensions.filter.dual_bm25 import DualBM25Config, DualBM25Scorer

    # NOTE: language is expected to already be set by caller (CLI) via default_language_factory.set_language(...)
    from configs.language import default_language_factory

    positive_query = default_language_factory.default_product_bm25_query()
    negative_query = " ".join(sorted(set(default_language_factory.exclude_tokens())))
    stopwords = default_language_factory.stopwords()

    scorer_cfg = DualBM25Config(
        threshold=None,
        alpha=0.7,
        k1=1.2,
        b=0.75,
        avgdl=1000,
    )

    state = CrawlState(db_path=actual_db_path)
    db_lock = asyncio.Lock()
    try:
        wrote = False

        # Counters and examples
        scanned = 0
        applied = 0
        skipped = 0
        total_candidates = 0
        total_kept = 0
        total_overflow = 0

        company_reports: List[EnforcedCompany] = []
        lock = asyncio.Lock()

        # Select companies once
        if selector == "crawl_finished_true":
            rows = await state._query_all(
                "SELECT company_id, crawl_finished, max_pages FROM companies",
                tuple(),
            )
            selected = [dict(r) for r in rows if int(r["crawl_finished"] or 0) == 1]
        elif selector == "all":
            selected = [
                dict(r)
                for r in await state._query_all(
                    "SELECT company_id, crawl_finished, max_pages FROM companies",
                    tuple(),
                )
            ]
        else:
            raise ValueError(f"unknown selector: {selector!r}")

        scanned = len(selected)

        q: asyncio.Queue[dict] = asyncio.Queue()
        for r in selected:
            q.put_nowait(r)

        async def process_company(r: dict) -> None:
            nonlocal applied, skipped, total_candidates, total_kept, total_overflow

            cid = str(r["company_id"])
            crawl_finished = int(r["crawl_finished"] or 0)

            if apply_only_if_crawl_finished and crawl_finished != 1:
                async with lock:
                    skipped += 1
                    if len(company_reports) < max_examples:
                        company_reports.append(
                            EnforcedCompany(
                                company_id=cid,
                                max_pages=int(max_pages or 0),
                                candidates=0,
                                kept=0,
                                overflow=0,
                                moved_md=0,
                                moved_html=0,
                                skipped_reason="crawl_not_finished",
                                example_overflow_urls=[],
                            )
                        )
                return

            db_cap = _safe_int(r.get("max_pages"))
            cap = int(max_pages) if max_pages is not None else int(db_cap)
            if cap <= 0:
                async with lock:
                    skipped += 1
                    if len(company_reports) < max_examples:
                        company_reports.append(
                            EnforcedCompany(
                                company_id=cid,
                                max_pages=int(cap),
                                candidates=0,
                                kept=0,
                                overflow=0,
                                moved_md=0,
                                moved_html=0,
                                skipped_reason="max_pages_not_set",
                                example_overflow_urls=[],
                            )
                        )
                return

            base = company_base(out_dir, cid)
            md_dir = base / "markdown"
            html_dir = base / "html"

            idx_path = first_existing(url_index_candidates(out_dir, cid))
            if idx_path is None:
                async with lock:
                    skipped += 1
                    if len(company_reports) < max_examples:
                        company_reports.append(
                            EnforcedCompany(
                                company_id=cid,
                                max_pages=int(cap),
                                candidates=0,
                                kept=0,
                                overflow=0,
                                moved_md=0,
                                moved_html=0,
                                skipped_reason="missing_url_index",
                                example_overflow_urls=[],
                            )
                        )
                return

            ok, reason = validate_json_bytes(idx_path)
            if not ok:
                async with lock:
                    skipped += 1
                    if len(company_reports) < max_examples:
                        company_reports.append(
                            EnforcedCompany(
                                company_id=cid,
                                max_pages=int(cap),
                                candidates=0,
                                kept=0,
                                overflow=0,
                                moved_md=0,
                                moved_html=0,
                                skipped_reason=f"url_index_invalid:{reason}",
                                example_overflow_urls=[],
                            )
                        )
                return

            idx = read_json_file(idx_path)
            if not isinstance(idx, dict) or not idx:
                async with lock:
                    skipped += 1
                    if len(company_reports) < max_examples:
                        company_reports.append(
                            EnforcedCompany(
                                company_id=cid,
                                max_pages=int(cap),
                                candidates=0,
                                kept=0,
                                overflow=0,
                                moved_md=0,
                                moved_html=0,
                                skipped_reason="url_index_empty_dict",
                                example_overflow_urls=[],
                            )
                        )
                return

            url_entries: List[Tuple[str, Dict[str, Any]]] = []
            for k, v in idx.items():
                if str(k).startswith("__"):
                    continue
                if not isinstance(v, Mapping):
                    continue
                url_entries.append((str(k), dict(v)))

            doc_index: Dict[str, str] = {}
            candidates: List[Tuple[str, float, Optional[Path], Optional[Path]]] = []

            for url, ent in url_entries:
                md_p, html_p = resolve_markdown_and_html_paths_from_url_index_entry(
                    md_dir=md_dir,
                    html_dir=html_dir,
                    entry=ent,
                )

                md_exists = bool(md_p is not None and md_p.exists() and md_p.is_file())
                html_exists = bool(
                    html_p is not None and html_p.exists() and html_p.is_file()
                )
                if not (md_exists or html_exists):
                    continue

                if html_exists and html_p is not None:
                    html_txt = _read_text_best_effort(html_p)
                    head_doc = _build_head_doc_from_html(html_txt)
                    if head_doc:
                        doc_index[url] = head_doc

                candidates.append(
                    (
                        url,
                        0.0,
                        md_p if md_exists else None,
                        html_p if html_exists else None,
                    )
                )

            if not candidates:
                async with lock:
                    skipped += 1
                    if len(company_reports) < max_examples:
                        company_reports.append(
                            EnforcedCompany(
                                company_id=cid,
                                max_pages=int(cap),
                                candidates=0,
                                kept=0,
                                overflow=0,
                                moved_md=0,
                                moved_html=0,
                                skipped_reason="no_candidates_with_artifacts",
                                example_overflow_urls=[],
                            )
                        )
                return

            scorer = DualBM25Scorer(
                positive_query=positive_query,
                negative_query=negative_query,
                stopwords=stopwords,
                cfg=scorer_cfg,
                doc_index=doc_index,
                weight=1.0,
            )

            scored: List[Tuple[str, float, Optional[Path], Optional[Path]]] = []
            for url, _, md_p, html_p in candidates:
                s = float(scorer.score(url))
                scored.append((url, s, md_p, html_p))

            scored.sort(key=lambda t: (-t[1], t[0]))

            hard_hit = len(scored) > cap
            kept = min(len(scored), cap)
            overflow = max(0, len(scored) - cap)

            overflow_rows = scored[cap:]
            moved_md = 0
            moved_html = 0

            if not dry_run:
                md_hidden = md_dir / ".md"
                html_hidden = html_dir / ".html"

                for _, _, md_p, html_p in overflow_rows:
                    if md_p is not None and md_p.exists():
                        _move_to_hidden_subdir(md_p, hidden_dir=md_hidden)
                        moved_md += 1
                    if html_p is not None and html_p.exists():
                        _move_to_hidden_subdir(html_p, hidden_dir=html_hidden)
                        moved_html += 1

                meta_raw = idx.get("__meta__")
                meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}

                overflow_set = set([u for (u, _, _, _) in overflow_rows])

                suppressed_count = 0
                for url in overflow_set:
                    ent2 = dict(idx.get(url) or {})
                    ent2["suppressed"] = True
                    ent2["suppressed_reason"] = "max_pages"
                    idx[url] = ent2
                    suppressed_count += 1

                kept_md_done = 0
                for _, _, md_p, _ in scored[:cap]:
                    if md_p is not None and md_p.exists():
                        kept_md_done += 1

                idx["__meta__"] = url_index_meta_patch_max_pages(
                    meta=meta,
                    company_id=cid,
                    kept=kept,
                    suppressed=suppressed_count,
                    hard_max_pages=cap,
                    hard_hit=hard_hit,
                    version_meta=version_meta,
                    markdown_saved=int(kept_md_done),
                )

                write_json_file(idx_path, idx, pretty=False)

                async with db_lock:
                    await state._exec(
                        """
                        UPDATE companies
                        SET
                            urls_total=?,
                            urls_markdown_done=?,
                            updated_at=?
                        WHERE company_id=?
                        """,
                        (int(kept), int(kept_md_done), _now_iso(), cid),
                    )

            async with lock:
                applied += 1
                total_candidates += len(scored)
                total_kept += kept
                total_overflow += overflow

                if len(company_reports) < max_examples:
                    company_reports.append(
                        EnforcedCompany(
                            company_id=cid,
                            max_pages=int(cap),
                            candidates=int(len(scored)),
                            kept=int(kept),
                            overflow=int(overflow),
                            moved_md=int(moved_md) if not dry_run else 0,
                            moved_html=int(moved_html) if not dry_run else 0,
                            skipped_reason=None,
                            example_overflow_urls=[
                                u for (u, _, _, _) in overflow_rows[:10]
                            ],
                        )
                    )

        async def worker() -> None:
            while True:
                try:
                    r = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await process_company(r)
                q.task_done()

        workers = [
            asyncio.create_task(worker()) for _ in range(min(conc, scanned or 1))
        ]
        await asyncio.gather(*workers)

        if write_global_state and not dry_run:
            async with db_lock:
                await state.write_global_state_from_db_only(pretty=False)
            wrote = True

        return EnforceMaxPagesReport(
            out_dir=str(out_dir),
            db_path=str(actual_db_path),
            dry_run=bool(dry_run),
            selector=str(selector),
            scanned_companies=int(scanned),
            applied_companies=int(applied),
            skipped_companies=int(skipped),
            total_candidates=int(total_candidates),
            total_kept=int(total_kept),
            total_overflow=int(total_overflow),
            wrote_global_state=bool(wrote),
            companies=company_reports,
        )
    finally:
        state.close()


def enforce_max_pages(
    *,
    out_dir: Path,
    db_path: Path,
    max_pages: Optional[int],
    selector: str = "crawl_finished_true",
    apply_only_if_crawl_finished: bool = True,
    write_global_state: bool = True,
    dry_run: bool = False,
    max_examples: int = 50,
    concurrency: int = 32,
) -> EnforceMaxPagesReport:
    return asyncio.run(
        enforce_max_pages_async(
            out_dir=out_dir,
            db_path=db_path,
            max_pages=max_pages,
            selector=selector,
            apply_only_if_crawl_finished=apply_only_if_crawl_finished,
            write_global_state=write_global_state,
            dry_run=dry_run,
            max_examples=max_examples,
            concurrency=concurrency,
        )
    )
