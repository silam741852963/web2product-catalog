from __future__ import annotations

import csv
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from concurrent.futures import ProcessPoolExecutor
import resource

from extensions.io.output_paths import ensure_output_root, ensure_company_dirs
from extensions.nice_match.sentence_split import split_into_sentences
from extensions.nice_match.nice_loader import load_nice_rows
from extensions.nice_match.nice_token_normalizer import build_tokens
from extensions.nice_match.matcher import build_pattern_index, match_sentence_hits


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[nice-match {ts}] {msg}", flush=True)


def _iter_company_ids_from_csv(company_file: Path, *, id_col: str) -> Iterable[str]:
    with company_file.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {company_file}")
        if id_col not in r.fieldnames:
            raise RuntimeError(
                f"CSV missing id column '{id_col}'. Available: {r.fieldnames}"
            )
        for row in r:
            v = (row.get(id_col) or "").strip()
            if v == "":
                continue
            yield v


_COOKIE_MARKERS = {
    "consent",
    "preferences",
    "privacy",
    "policy",
    "settings",
    "accept",
    "decline",
    "analytics",
    "tracking",
    "gdpr",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _cookie_boilerplate_flag(sentence_text: str) -> bool:
    s = (sentence_text or "").lower()
    words = set(_WORD_RE.findall(s))
    if "cookie" not in words and "cookies" not in words:
        return False
    for m in _COOKIE_MARKERS:
        if m in words:
            return True
    return False


def _word_count(s: str) -> int:
    return len(_WORD_RE.findall((s or "").lower()))


# ---- markdown cleaning (local, deterministic, no dependency on md_concat internals)

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_markdown_to_text(md: str) -> str:
    s = md or ""
    s = _CODE_FENCE_RE.sub(" ", s)
    s = _MD_IMAGE_RE.sub(" ", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = _INLINE_CODE_RE.sub(r"\1", s)
    s = re.sub(r"^\s{0,3}#+\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s{0,3}[-*+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s{0,3}\d+\.\s+", "", s, flags=re.MULTILINE)
    s = _HTML_TAG_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


@dataclass(frozen=True, slots=True)
class _MarkdownPage:
    url: str
    text: str


def _load_url_index_map_tolerant(url_index_path: Path) -> dict[str, str]:
    """
    Tolerant loader:
      - returns {} if missing/unreadable/invalid
      - maps markdown filename -> url when possible
    """
    try:
        raw = url_index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    out: dict[str, str] = {}
    for k, v in data.items():
        if k == "__meta__":
            continue
        if not isinstance(v, dict):
            continue
        md_path = v.get("markdown_path")
        url = v.get("url")
        if not isinstance(md_path, str) or md_path.strip() == "":
            continue
        if not isinstance(url, str) or url.strip() == "":
            continue
        fname = Path(md_path).name.strip()
        if fname == "":
            continue
        # Deterministic: first wins; do not raise on conflicts
        if fname not in out:
            out[fname] = url.strip()
    return out


def _iter_company_markdown_pages_tolerant(bvd_id: str) -> list[_MarkdownPage]:
    """
    Tolerant page iterator:
      - resolves company folder using ensure_company_dirs (same sanitize strategy as crawl)
      - reads *.md even if url_index.json missing or does not contain the file
      - url fallback is "" when unknown (deterministic)
      - skips unreadable markdown files
    """
    dirs = ensure_company_dirs(bvd_id)
    md_dir = Path(dirs["markdown"]).resolve()
    meta_dir = Path(dirs["metadata"]).resolve()
    url_index_path = (meta_dir / "url_index.json").resolve()

    if not md_dir.exists():
        raise FileNotFoundError(f"Missing markdown directory: {md_dir}")

    md_files = sorted(md_dir.glob("*.md"))
    if len(md_files) == 0:
        raise FileNotFoundError(f"No markdown files found under: {md_dir}")

    md_to_url = _load_url_index_map_tolerant(url_index_path)

    pages: list[_MarkdownPage] = []
    for p in md_files:
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            continue

        text = _clean_markdown_to_text(raw)
        if text == "":
            continue

        url = md_to_url.get(p.name, "")
        pages.append(_MarkdownPage(url=url, text=text))

    if len(pages) == 0:
        raise FileNotFoundError(
            f"All markdown files were empty/unreadable under: {md_dir}"
        )

    return pages


@dataclass(frozen=True, slots=True)
class SentenceRecord:
    sentence_id: int  # resets per firm (1-based)
    bvd_id: str
    url: str
    sentence_text: str
    sentence_word_count: int
    sentence_char_count: int
    is_cookie_boilerplate: bool


@dataclass(frozen=True, slots=True)
class _ChunkResult:
    part_sentences_csv: str
    part_matches_csv: str
    firms_scraped: int
    firms_seen: int
    matched_rows: int
    sentences_per_firm: list[tuple[str, int]]  # (bvd_id, sentence_count)
    skipped_firms: int
    skipped_pages: int
    skipped_md_files: int


def _chunk_part_path(base: Path, *, chunk_idx: int, num_chunks: int) -> Path:
    suffix = f".part-{chunk_idx:05d}-of-{num_chunks:05d}"
    return base.with_name(base.name + suffix)


def _merge_csv_parts(final_path: Path, part_paths: list[Path]) -> None:
    if len(part_paths) == 0:
        raise RuntimeError("No part paths provided for merge.")
    for p in part_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing part file for merge: {p}")

    final_path.parent.mkdir(parents=True, exist_ok=True)

    with final_path.open("w", encoding="utf-8", newline="") as out_f:
        wrote_header = False
        for p in part_paths:
            with p.open("r", encoding="utf-8", newline="") as in_f:
                for line_idx, line in enumerate(in_f):
                    if line_idx == 0:
                        if not wrote_header:
                            out_f.write(line)
                            wrote_header = True
                        continue
                    out_f.write(line)

    if not wrote_header:
        raise RuntimeError(f"Merge produced no header: final_path={final_path}")


def _delete_paths(paths: list[Path]) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Cannot delete missing file: {p}")
        if not p.is_file():
            raise RuntimeError(f"Cannot delete non-file path: {p}")
        p.unlink()


def _split_contiguous(ids: Sequence[str], *, workers: int) -> list[list[str]]:
    n = len(ids)
    if workers < 1:
        raise ValueError("workers must be >= 1")

    if n == 0:
        return [[] for _ in range(workers)]

    base = n // workers
    rem = n % workers

    chunks: list[list[str]] = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        end = start + size
        chunks.append(list(ids[start:end]))
        start = end

    if start != n:
        raise RuntimeError("Internal inconsistency in chunk splitting.")
    return chunks


def _max_workers_by_nofile() -> tuple[int, int]:
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    reserve_fds = 256
    fds_per_worker = 8
    if soft <= reserve_fds:
        return 0, int(soft)
    max_workers = (soft - reserve_fds) // fds_per_worker
    if max_workers < 1:
        return 0, int(soft)
    return int(max_workers), int(soft)


def _run_chunk_worker(
    *,
    chunk_idx: int,
    num_chunks: int,
    bvd_ids: list[str],
    nice_xlsx: str,
    nice_sheet: int | str,
    part_matches_csv: str,
    part_sentences_csv: str,
    minimal: bool,
    missing_policy: str,
) -> _ChunkResult:
    if missing_policy not in ("error", "skip"):
        raise ValueError("missing_policy must be 'error' or 'skip'")

    part_matches_path = Path(part_matches_csv).expanduser().resolve()
    part_sentences_path = Path(part_sentences_csv).expanduser().resolve()
    part_matches_path.parent.mkdir(parents=True, exist_ok=True)
    part_sentences_path.parent.mkdir(parents=True, exist_ok=True)

    nice_rows = load_nice_rows(nice_xlsx, sheet=nice_sheet)
    tokens = build_tokens(nice_rows)
    index = build_pattern_index(tokens)

    sentence_headers = [
        "bvd_id",
        "sentence_id",
        "url",
        "sentence_text",
        "sentence_word_count",
        "sentence_char_count",
        "is_cookie_boilerplate",
    ]

    if minimal:
        match_headers = ["bvd_id", "sentence_id", "token_id"]
    else:
        match_headers = [
            "bvd_id",
            "sentence_id",
            "token_id",
            "Cl.",
            "Prop. No./№",
            "X",
            "Action EN",
            "matched_alias",
            "aliases",
            "note",
            "match_count",
            "match_spans",
            "match_lengths",
            "matched_chars_total",
        ]

    firms_seen = 0
    firms_scraped = 0
    matched_rows = 0
    sentences_per_firm: list[tuple[str, int]] = []

    skipped_firms = 0
    skipped_pages = 0
    skipped_md_files = 0

    with (
        part_sentences_path.open("w", encoding="utf-8", newline="") as sf,
        part_matches_path.open("w", encoding="utf-8", newline="") as mf,
    ):
        sw = csv.DictWriter(sf, fieldnames=sentence_headers)
        mw = csv.DictWriter(mf, fieldnames=match_headers)
        sw.writeheader()
        mw.writeheader()

        for bvd_id in bvd_ids:
            firms_seen += 1

            try:
                pages = _iter_company_markdown_pages_tolerant(bvd_id)
            except FileNotFoundError:
                if missing_policy == "error":
                    raise
                skipped_firms += 1
                continue
            except Exception:
                # Deterministic "maximize processing": skip firm on unexpected errors
                skipped_firms += 1
                continue

            firms_scraped += 1
            local_sentence_id = 1
            firm_sentence_count = 0

            sentences: list[SentenceRecord] = []

            for page in pages:
                try:
                    sents = split_into_sentences(page.text)
                except Exception:
                    skipped_pages += 1
                    continue

                for sent in sents:
                    sent2 = (sent or "").strip()
                    if sent2 == "":
                        continue

                    wc = _word_count(sent2)
                    cc = len(sent2)
                    is_cookie = _cookie_boilerplate_flag(sent2)

                    rec = SentenceRecord(
                        sentence_id=local_sentence_id,
                        bvd_id=bvd_id,
                        url=page.url,
                        sentence_text=sent2,
                        sentence_word_count=wc,
                        sentence_char_count=cc,
                        is_cookie_boilerplate=is_cookie,
                    )
                    sentences.append(rec)

                    sw.writerow(
                        {
                            "bvd_id": rec.bvd_id,
                            "sentence_id": rec.sentence_id,
                            "url": rec.url,
                            "sentence_text": rec.sentence_text,
                            "sentence_word_count": rec.sentence_word_count,
                            "sentence_char_count": rec.sentence_char_count,
                            "is_cookie_boilerplate": int(rec.is_cookie_boilerplate),
                        }
                    )

                    local_sentence_id += 1
                    firm_sentence_count += 1

            # If firm had no sentences, still record as 0 (deterministic)
            sentences_per_firm.append((bvd_id, firm_sentence_count))

            for s in sentences:
                if s.is_cookie_boilerplate:
                    continue

                try:
                    hits = match_sentence_hits(
                        sentence_text=s.sentence_text, index=index
                    )
                except Exception:
                    # Deterministic skip on matcher error for this sentence
                    continue

                for h in hits:
                    t = h.token

                    if minimal:
                        mw.writerow(
                            {
                                "bvd_id": s.bvd_id,
                                "sentence_id": s.sentence_id,
                                "token_id": t.token_id,
                            }
                        )
                        matched_rows += 1
                        continue

                    spans = h.spans
                    lengths = [end - start for start, end in spans]
                    matched_chars_total = sum(lengths)
                    span_str = "|".join(f"{start}:{end}" for start, end in spans)
                    len_str = "|".join(str(x) for x in lengths)

                    mw.writerow(
                        {
                            "bvd_id": s.bvd_id,
                            "sentence_id": s.sentence_id,
                            "token_id": t.token_id,
                            "Cl.": t.cl,
                            "Prop. No./№": t.prop_no,
                            "X": t.x,
                            "Action EN": t.action_en,
                            "matched_alias": h.matched_alias,
                            "aliases": " | ".join(t.aliases),
                            "note": t.note,
                            "match_count": len(spans),
                            "match_spans": span_str,
                            "match_lengths": len_str,
                            "matched_chars_total": matched_chars_total,
                        }
                    )
                    matched_rows += 1

    return _ChunkResult(
        part_sentences_csv=str(part_sentences_path),
        part_matches_csv=str(part_matches_path),
        firms_scraped=firms_scraped,
        firms_seen=firms_seen,
        matched_rows=matched_rows,
        sentences_per_firm=sentences_per_firm,
        skipped_firms=skipped_firms,
        skipped_pages=skipped_pages,
        skipped_md_files=skipped_md_files,
    )


def run_nice_sentence_matching(
    *,
    out_dir: str,
    company_file: str,
    company_id_col: str,
    nice_xlsx: str,
    nice_sheet: int | str,
    output_csv: str,
    sentences_csv: str,
    stats_json: str,
    minimal: bool,
    missing_policy: str,
    limit: Optional[int],
    workers: int,
) -> None:
    if missing_policy not in ("error", "skip"):
        raise ValueError("--missing-policy must be 'error' or 'skip'")
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    ensure_output_root(out_dir)

    company_path = Path(company_file).expanduser().resolve()
    if not company_path.exists():
        raise FileNotFoundError(f"--company-file not found: {company_path}")

    out_csv_path = Path(output_csv).expanduser().resolve()
    sent_csv_path = Path(sentences_csv).expanduser().resolve()
    stats_json_path = Path(stats_json).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sent_csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)

    ids = list(_iter_company_ids_from_csv(company_path, id_col=company_id_col))
    if limit is not None:
        ids = ids[:limit]

    n_ids = len(ids)
    _log(f"Companies loaded: {n_ids} (limit={limit})")
    _log(f"Workers requested: {workers}")

    if n_ids == 0:
        stats = {
            "firms_seen": 0,
            "firms_scraped": 0,
            "matched_rows": 0,
            "sentences_per_firm": {},
            "mean_sentences_per_firm": None,
            "median_sentences_per_firm": None,
            "skipped_firms": 0,
        }
        stats_json_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _log("Done. firms_scraped=0 firms_seen=0 matched_rows=0 (0.00%)")
        return

    if workers > n_ids:
        raise ValueError(
            f"--workers ({workers}) must be <= number of companies loaded ({n_ids})."
        )

    max_by_nofile, soft_nofile = _max_workers_by_nofile()
    if max_by_nofile == 0:
        raise RuntimeError(
            "RLIMIT_NOFILE is too low to safely spawn worker processes. "
            f"ulimit -n (soft) = {soft_nofile}."
        )
    if workers > max_by_nofile:
        raise ValueError(
            f"--workers ({workers}) exceeds max allowed by RLIMIT_NOFILE ({max_by_nofile}). "
            f"ulimit -n (soft) = {soft_nofile}. Reduce --workers or increase ulimit -n."
        )

    if workers == 1:
        r = _run_chunk_worker(
            chunk_idx=0,
            num_chunks=1,
            bvd_ids=ids,
            nice_xlsx=nice_xlsx,
            nice_sheet=nice_sheet,
            part_matches_csv=str(out_csv_path),
            part_sentences_csv=str(sent_csv_path),
            minimal=minimal,
            missing_policy=missing_policy,
        )

        firms_scraped = r.firms_scraped
        firms_seen = r.firms_seen
        matched_rows = r.matched_rows
        sentences_pairs = r.sentences_per_firm
        skipped_firms = r.skipped_firms

    else:
        chunks = _split_contiguous(ids, workers=workers)

        part_matches: list[Path] = []
        part_sentences: list[Path] = []
        results: list[Optional[_ChunkResult]] = [None] * workers

        for i in range(workers):
            pm = _chunk_part_path(out_csv_path, chunk_idx=i, num_chunks=workers)
            ps = _chunk_part_path(sent_csv_path, chunk_idx=i, num_chunks=workers)
            part_matches.append(pm)
            part_sentences.append(ps)

        _log(
            f"Launching {workers} workers (contiguous chunks, deterministic merge order)."
        )

        completed_seen = 0

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = []
            for i in range(workers):
                futs.append(
                    ex.submit(
                        _run_chunk_worker,
                        chunk_idx=i,
                        num_chunks=workers,
                        bvd_ids=chunks[i],
                        nice_xlsx=nice_xlsx,
                        nice_sheet=nice_sheet,
                        part_matches_csv=str(part_matches[i]),
                        part_sentences_csv=str(part_sentences[i]),
                        minimal=minimal,
                        missing_policy=missing_policy,
                    )
                )

            for i, fut in enumerate(futs):
                res = fut.result()
                results[i] = res

                completed_seen += res.firms_seen
                pct = (completed_seen / n_ids) * 100.0
                _log(
                    f"Progress: {completed_seen}/{n_ids} ({pct:.2f}%) chunks_done={i + 1}/{workers}"
                )

        firms_scraped = 0
        firms_seen = 0
        matched_rows = 0
        sentences_pairs: list[tuple[str, int]] = []
        skipped_firms = 0

        for r in results:
            if r is None:
                raise RuntimeError("Internal inconsistency: missing chunk result.")
            firms_scraped += r.firms_scraped
            firms_seen += r.firms_seen
            matched_rows += r.matched_rows
            sentences_pairs.extend(r.sentences_per_firm)
            skipped_firms += r.skipped_firms

        _log("Merging part CSVs (sentences, then matches).")
        _merge_csv_parts(sent_csv_path, part_sentences)
        _merge_csv_parts(out_csv_path, part_matches)

        _log("Deleting part CSVs after successful merge.")
        _delete_paths(part_sentences)
        _delete_paths(part_matches)

    sentences_map: dict[str, int] = {}
    for bvd_id, n_sent in sentences_pairs:
        # deterministic first-wins if duplicated in input list
        if bvd_id not in sentences_map:
            sentences_map[bvd_id] = int(n_sent)

    counts = list(sentences_map.values())

    if len(counts) == 0:
        mean_v = None
        median_v = None
    else:
        mean_v = float(statistics.mean(counts))
        median_v = float(statistics.median(counts))

    stats = {
        "firms_seen": firms_seen,
        "firms_scraped": firms_scraped,
        "matched_rows": matched_rows,
        "sentences_per_firm": sentences_map,
        "mean_sentences_per_firm": mean_v,
        "median_sentences_per_firm": median_v,
        "skipped_firms": skipped_firms,
    }

    stats_json_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pct_done = (firms_seen / n_ids) * 100.0
    _log(
        f"Done. firms_scraped={firms_scraped} firms_seen={firms_seen} "
        f"matched_rows={matched_rows} ({pct_done:.2f}%) skipped_firms={skipped_firms}"
    )
