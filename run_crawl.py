from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterable

# Crawl4AI
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, RateLimiter
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

# Local modules
from components.csv_loader import load_companies_from_csv, CompanyInput
from components.md_generator import build_default_md_generator, should_save_markdown
from components.llm_extractor import build_llm_extraction_strategy
from components.url_seeder import seed_urls

from extensions.checkpoint import CheckpointManager
from extensions.logging import LoggingExtension
from extensions.output_paths import ensure_company_dirs, save_stage_output
from extensions.filtering import (
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
)

META_PATH_NAME = "crawl_meta.json"
URL_INDEX_NAME = "url_index.json"  # NEW: local manifest to resume without seeding


# ----------------------------
# CLI parsing
# ----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seed → filter → crawl (HTML/MD) → local LLM over saved Markdown"
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Input CSV file path")
    src.add_argument("--csv-dir", type=Path, help="Directory containing one or more CSV files")

    p.add_argument(
        "--pipeline",
        type=str,
        default="html,markdown,llm",
        help="Comma-separated stages from {seed,html,markdown,llm}. Examples: 'seed', 'html,markdown', 'markdown,llm', 'html,markdown,llm'.",
    )

    p.add_argument("--limit", type=int, default=None, help="Optional limit of rows")
    p.add_argument("--max-slots", type=int, default=20, help="Global max concurrent crawl sessions across companies")

    p.add_argument(
        "--seeding-source",
        type=str,
        default="sitemap+cc",
        choices=["sitemap", "cc", "sitemap+cc"],
        help="Where to seed URLs from",
    )

    p.add_argument("--include", type=str, default="", help="Comma-separated URL globs to KEEP (in addition to defaults)")
    p.add_argument("--exclude", type=str, default="", help="Comma-separated URL globs to DROP (in addition to defaults)")
    p.add_argument("--query", type=str, default=None, help="BM25 query for relevance scoring in seeding (optional)")
    p.add_argument("--score-threshold", type=float, default=None, help="Min BM25 score to keep (if --query set)")
    p.add_argument("--force-seeder-cache", action="store_true", help="Force seeder to bypass its cache (fresh discovery)")
    p.add_argument("--bypass-local", action="store_true", help="Skip local HTML/MD reuse (force remote fetch)")
    p.add_argument("--respect-crawl-date", action="store_true", help="Skip URLs older than last crawl date recorded")

    p.add_argument("--live-check", action="store_true", help="Enable live HTTP status check during seeding (default OFF)")

    # Brand / language / externals
    p.add_argument("--discover-brands", action="store_true", default=True, help="Enable brand discovery (default ON)")
    p.add_argument("--no-discover-brands", dest="discover_brands", action="store_false", help="Disable brand discovery")
    p.add_argument("--drop-universal-externals", action="store_true", default=True, help="Drop universal externals (default ON)")
    p.add_argument("--lang-primary", type=str, default="en", help="Primary language to keep")
    p.add_argument("--lang-accept-en-regions", type=str, default="us,gb,ca,au,nz,ie,sg", help="Comma-separated English regions")
    p.add_argument("--lang-strict-cctld", action="store_true", help="Treat ccTLDs as hard language gates")

    p.add_argument("--max-urls", type=int, default=-1, help="Max URLs from seeding per root (-1 = unlimited)")
    p.add_argument("--company-max-pages", type=int, default=-1, help="Company-wide page cap (-1 = unlimited)")
    p.add_argument("--hits-per-sec", type=int, default=50, help="Seeder rate limit")

    # Markdown generator knobs
    p.add_argument("--md-min-words", type=int, default=80, help="Min words for saving markdown")
    p.add_argument("--md-threshold", type=float, default=0.48, help="PruningContentFilter threshold")
    p.add_argument("--md-threshold-type", choices=["dynamic", "fixed"], default="dynamic", help="Pruning threshold type")
    p.add_argument("--md-min-block-words", type=int, default=100, help="PruningContentFilter min_word_threshold")
    p.add_argument("--md-content-source", choices=["cleaned_html", "fit_html", "raw_html"], default="cleaned_html", help="HTML variant for MD")
    p.add_argument("--md-ignore-links", action="store_true", help="Strip links in markdown")
    p.add_argument("--md-ignore-images", action="store_true", help="Strip images in markdown")
    p.add_argument("--md-body-width", type=int, default=0, help="Markdown wrap width (0 = no wrap)")

    # LLM
    p.add_argument("--presence-only", action="store_true", help="LLM presence classification instead of schema extraction")

    # Logging
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console/file log level")
    return p.parse_args()


# ----------------------------
# Small helpers
# ----------------------------

def _parse_pipeline(args: argparse.Namespace) -> List[str]:
    stages = [s.strip().lower() for s in (args.pipeline or "").split(",") if s.strip()]
    allowed = {
        ("seed",), ("html",), ("markdown",), ("llm",),
        ("html", "markdown"),
        ("markdown", "llm"),
        ("html", "markdown", "llm"),
        ("seed", "html"), ("seed", "markdown"), ("seed", "llm"),
        ("seed", "html", "markdown"), ("seed", "markdown", "llm"),
        ("seed", "html", "markdown", "llm"),
    }
    if tuple(stages) not in allowed:
        raise SystemExit(
            f"Invalid --pipeline '{args.pipeline}'. "
            "Use combinations of: seed, html, markdown, llm"
        )
    return stages

def _url_hash(url: str) -> str:
    return sha1(url.encode()).hexdigest()[:8]

def _ext_for_stage(stage: str) -> str:
    return {"html": ".html", "markdown": ".md", "llm": ".json"}[stage]

def _stage_dir_key(stage: str) -> str:
    return {"html": "html", "markdown": "markdown", "llm": "json"}[stage]

def _find_existing_artifact(hojin_id: str, url: str, stage: str) -> Optional[Path]:
    dirs = ensure_company_dirs(hojin_id)
    d = dirs[_stage_dir_key(stage)]
    h = _url_hash(url)
    ext = _ext_for_stage(stage)
    for p in d.glob(f"*{h}{ext}"):
        return p
    return None

def _prefer_local_html(hojin_id: str, url: str) -> Optional[Path]:
    return _find_existing_artifact(hojin_id, url, "html")

def _prefer_local_md(hojin_id: str, url: str) -> Optional[Path]:
    return _find_existing_artifact(hojin_id, url, "markdown")

def _meta_path(hojin_id: str) -> Path:
    return ensure_company_dirs(hojin_id)["checkpoints"] / META_PATH_NAME

def _url_index_path(hojin_id: str) -> Path:  # NEW
    return ensure_company_dirs(hojin_id)["checkpoints"] / URL_INDEX_NAME

def load_url_index(hojin_id: str) -> Dict[str, Any]:  # NEW
    p = _url_index_path(hojin_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def upsert_url_index(  # NEW
    hojin_id: str,
    url: str,
    *,
    html_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
) -> None:
    idx = load_url_index(hojin_id)
    ent = idx.get(url, {})
    if html_path:
        ent["html_path"] = str(html_path)
        ent["html_saved_at"] = datetime.now(timezone.utc).isoformat()
    if markdown_path:
        ent["markdown_path"] = str(markdown_path)
        ent["markdown_saved_at"] = datetime.now(timezone.utc).isoformat()
    if json_path:
        ent["json_path"] = str(json_path)
        ent["json_saved_at"] = datetime.now(timezone.utc).isoformat()
    idx[url] = ent
    p = _url_index_path(hojin_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(idx, indent=2), encoding="utf-8")

def read_last_crawl_date(hojin_id: str) -> Optional[datetime]:
    p = _meta_path(hojin_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        v = data.get("last_crawled_at")
        return datetime.fromisoformat(v) if v else None
    except Exception:
        return None

def write_last_crawl_date(hojin_id: str, dt: Optional[datetime] = None) -> None:
    p = _meta_path(hojin_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"last_crawled_at": (dt or datetime.now(timezone.utc)).isoformat()}, indent=2), encoding="utf-8")

def filter_by_last_crawl_date(url_items: List[Dict[str, Any]], last_crawl: datetime) -> List[Dict[str, Any]]:
    def _parse(u: Dict[str, Any]) -> Optional[datetime]:
        for k in ("lastmod", "last_modified", "sitemap_lastmod", "lastModified"):
            v = u.get(k)
            if not v:
                continue
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except Exception:
                pass
        return None
    out = []
    for u in url_items:
        lm = _parse(u)
        if lm is None or lm > last_crawl:
            out.append(u)
    return out

def _build_md_generator_from_args(args: argparse.Namespace):
    return build_default_md_generator(
        threshold=args.md_threshold,
        threshold_type=args.md_threshold_type,
        min_word_threshold=args.md_min_block_words,
        body_width=args.md_body_width,
        ignore_links=args.md_ignore_links,
        ignore_images=args.md_ignore_images,
        content_source=args.md_content_source,
        min_meaningful_words=args.md_min_words,
    )

def _mk_md_config(args: argparse.Namespace) -> CrawlerRunConfig:
    # Local HTML → MD pass (non-streaming for stable order)
    return CrawlerRunConfig(
        markdown_generator=_build_md_generator_from_args(args),
        cache_mode=CacheMode.BYPASS,
        stream=False,
    )

def _mk_llm_config_for_markdown_input(args: argparse.Namespace) -> CrawlerRunConfig:
    # Local MD → LLM pass (non-streaming for stable order)
    extraction = build_llm_extraction_strategy(presence_only=args.presence_only)
    return CrawlerRunConfig(
        extraction_strategy=extraction,
        cache_mode=CacheMode.BYPASS,
        stream=False,
    )

def _mk_remote_config_for_pipeline(pipeline: List[str], args: argparse.Namespace) -> CrawlerRunConfig:
    # Remote pass never does extraction; we only want HTML/MD here.
    need_md = ("markdown" in pipeline or "llm" in pipeline)
    md_gen = _build_md_generator_from_args(args) if need_md else None
    return CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=md_gen,
        extraction_strategy=None,
        stream=True,
    )

def _make_dispatcher(max_concurrency: int) -> MemoryAdaptiveDispatcher:
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=85.0,
        check_interval=1.0,
        max_session_permit=max_concurrency,
        rate_limiter=RateLimiter(base_delay=(0.5, 1.2), max_delay=20.0, max_retries=2),
        monitor=None,
    )

def _aggregate_seed_by_root(items: Iterable[Dict[str, Any]], base_root: str) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    roots: List[str] = []
    for it in items:
        r = str(it.get("seed_root") or "").strip()
        if not r:
            continue
        counts[r] = counts.get(r, 0) + 1
        roots.append(r)
    unique_roots = sorted(set(roots))
    brand_roots = [r for r in unique_roots if r != base_root]
    return {
        "seed_counts_by_root": counts,
        "seed_roots": unique_roots,
        "seed_brand_roots": brand_roots,
        "seed_brand_count": len(brand_roots),
    }


# ----------------------------
# Save artifacts
# ----------------------------

def _save_result_for_pipeline(
    hojin_id: str,
    url: str,
    pipeline: List[str],
    result,
    logger: logging.Logger,
    md_min_words: int,
) -> Dict[str, Any]:
    """
    Save artifacts and upsert url_index.json so later 'llm' runs can skip seeding.
    """
    stats = {"saved_html": False, "saved_md": False, "saved_json": False, "md_suppressed": False}

    html = getattr(result, "html", None) or getattr(result, "cleaned_html", None)
    md_obj = getattr(result, "markdown", None)
    raw_md = getattr(md_obj, "raw_markdown", None) if md_obj is not None else None
    fit_md = getattr(md_obj, "fit_markdown", None) if md_obj is not None else None
    extracted = getattr(result, "extracted_content", None)

    html_path = md_path = json_path = None

    if html and "html" in pipeline:
        save_stage_output(hojin_id, url, html, stage="html")
        stats["saved_html"] = True
        html_path = _find_existing_artifact(hojin_id, url, "html")

    if ("markdown" in pipeline or "llm" in pipeline):
        chosen_md = (fit_md or raw_md or "").strip()
        if chosen_md:
            ok, _ = should_save_markdown(chosen_md, min_meaningful_words=md_min_words, url=url)
            if ok:
                save_stage_output(hojin_id, url, chosen_md, stage="markdown")
                stats["saved_md"] = True
                md_path = _find_existing_artifact(hojin_id, url, "markdown")
            else:
                stats["md_suppressed"] = True

    if "llm" in pipeline and extracted:
        save_stage_output(hojin_id, url, extracted, stage="json")
        stats["saved_json"] = True
        json_path = _find_existing_artifact(hojin_id, url, "llm")

    # NEW: persist index after saves
    try:
        upsert_url_index(hojin_id, url, html_path=html_path, markdown_path=md_path, json_path=json_path)
    except Exception as e:
        logger.debug("[%s] url_index upsert failed for %s: %s", hojin_id, url, e)

    return stats


# ----------------------------
# Company crawl
# ----------------------------

async def _crawl_company(
    company: CompanyInput,
    *,
    pipeline: List[str],
    seeding_source: str,
    max_concurrency_for_company: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    respect_crawl_date: bool,
    bypass_local: bool,
    force_seeder_cache: bool,
    bm25_query: Optional[str],
    bm25_score_threshold: Optional[float],
    max_urls: int,
    company_max_pages: int,
    hits_per_sec: Optional[int],
    cp_mgr: CheckpointManager,
    log_ext: LoggingExtension,
    discover_brands: bool,
    drop_universal_externals: bool,
    lang_primary: str,
    lang_accept_en_regions: set[str],
    lang_strict_cctld: bool,
    args: argparse.Namespace,
) -> None:
    hojin_id = company.hojin_id
    logger = log_ext.get_company_logger(hojin_id)
    ensure_company_dirs(hojin_id)

    await cp_mgr.append_company(hojin_id)
    cp = await cp_mgr.get(hojin_id)
    await cp.mark_start(stage="|".join(pipeline), total_urls=0)
    logger.info("[%s] Start: %s (%s) pipeline=%s", hojin_id, company.company_name, company.url, pipeline)

    # =========
    # 1) Seeding (smart short-circuit)
    # =========
    seeded_items: List[Dict[str, Any]] = []
    seed_stats: Dict[str, Any] = {}

    # We can skip seeding if:
    #   - pipeline does NOT explicitly include 'seed' and 'html' (we don't need discovery for LLM over existing MD)
    #   - AND 'llm' is in the pipeline
    #   - AND we have an existing url_index.json with at least one entry
    pipeline_set = set(pipeline)
    try_index_first = ("llm" in pipeline_set) and ("seed" not in pipeline_set) and ("html" not in pipeline_set)

    used_index_for_urls = False
    url_index: Dict[str, Any] = {}

    if try_index_first:
        url_index = load_url_index(hojin_id)
        if url_index:
            used_index_for_urls = True
            seeded_items = [{"url": u, "status": "valid"} for u in url_index.keys()]
            seed_stats = {
                "discovered_total": len(seeded_items),
                "filtered_total": len(seeded_items),
                "seed_roots": [company.url],
                "seed_brand_count": 0,
                "resume_mode": "url_index",
            }
            logger.info("[%s] Resume without seeding: %d URL(s) from url_index.json", hojin_id, len(seeded_items))

    if not used_index_for_urls:
        # Fallback to normal seeding
        seeded_items, seed_stats = await seed_urls(
            company.url,
            source=seeding_source,
            include=include_patterns,
            exclude=exclude_patterns,
            query=bm25_query,
            score_threshold=bm25_score_threshold,
            live_check=args.live_check,
            force=force_seeder_cache,
            max_urls=max_urls,
            company_max_pages=company_max_pages,
            hits_per_sec=hits_per_sec,
            drop_universal_externals=drop_universal_externals,
            lang_primary=lang_primary,
            lang_accept_en_regions=lang_accept_en_regions,
            lang_strict_cctld=lang_strict_cctld,
            discover_brands=discover_brands,
        )

    cp.data.setdefault("seeding", {}).update(seed_stats)
    await cp.save()

    logger.info(
        "[%s] Seeding: discovered=%s, filtered=%s, roots=%d, brands=%d",
        hojin_id,
        seed_stats.get("discovered_total", 0),
        seed_stats.get("filtered_total", 0),
        len(seed_stats.get("seed_roots", [])),
        seed_stats.get("seed_brand_count", 0),
    )

    if pipeline == ["seed"]:
        agg = _aggregate_seed_by_root(seeded_items, base_root=company.url)
        cp.data.update({"urls_total": int(seed_stats.get("filtered_total", 0)), "seed_source": seeding_source})
        cp.data.update(agg)
        await cp.save()
        await cp.mark_finished()
        write_last_crawl_date(hojin_id)
        logger.info("[%s] SEED stage complete.", hojin_id)
        return

    # 2) Optionally respect last crawl date (skip when resuming from index because
    #    we’re not hitting network again—everything is local)
    seeded = list(seeded_items)
    if respect_crawl_date and not used_index_for_urls:
        last_dt = read_last_crawl_date(hojin_id)
        if last_dt:
            before = len(seeded)
            seeded = filter_by_last_crawl_date(seeded, last_dt)
            logger.info("[%s] Respect last crawl (%s): kept %d/%d", hojin_id, last_dt.isoformat(), len(seeded), before)

    # Map to URL list
    url_to_item: Dict[str, Dict[str, Any]] = {}
    seeded_urls: List[str] = []
    for u in seeded:
        url = u.get("final_url") or u.get("url")
        if url:
            url_to_item[url] = u
            seeded_urls.append(url)

    cp.data["urls_total"] = len(seeded_urls)
    await cp.save()

    if not seeded_urls:
        await cp.mark_finished()
        write_last_crawl_date(hojin_id)
        logger.info("[%s] No URLs after seeding/resume. DONE.", hojin_id)
        return

    llm_in_pipeline = ("llm" in pipeline_set)

    # =========
    # 3) Partition work
    # =========
    local_html_needed: List[Tuple[str, Path]] = []
    remote_candidates: List[Dict[str, Any]] = []
    completion_stage = pipeline[-1]

    for url in seeded_urls:
        # artifact-based resume for final stage
        if _find_existing_artifact(hojin_id, url, stage=completion_stage):
            await cp.mark_url_done(url)
            continue

        html_local = None if (bypass_local or ("markdown" not in pipeline_set and not llm_in_pipeline)) else _prefer_local_html(hojin_id, url)
        if html_local:
            local_html_needed.append((url, html_local))
        else:
            remote_candidates.append(url_to_item[url])

    # Save counters
    saved_html = saved_md = saved_json = md_suppressed = 0

    # =========
    # 4) Local HTML → Markdown (non-streaming)
    # =========
    if local_html_needed:
        async with AsyncWebCrawler() as local_crawler:
            file_urls = [f"file://{p.resolve()}" for _, p in local_html_needed]
            md_cfg = _mk_md_config(args)
            results = await local_crawler.arun_many(file_urls, config=md_cfg)

            # Map file:// to original URL
            file2orig = {f"file://{p.resolve()}": u for (u, p) in local_html_needed}
            for r in (results or []):
                file_url = getattr(r, "url", None)
                url = file2orig.get(file_url)
                if not url:
                    continue
                if getattr(r, "success", False):
                    s = _save_result_for_pipeline(hojin_id, url, ["markdown"], r, logger, args.md_min_words)
                    saved_html += int(s["saved_html"])
                    saved_md += int(s["saved_md"])
                    md_suppressed += int(s["md_suppressed"])
                    if not llm_in_pipeline:
                        await cp.mark_url_done(url)
                else:
                    err = getattr(r, "error_message", "local-md-fail")
                    logger.warning("[%s] Local HTML→MD failed: %s err=%s", hojin_id, url, err)
                    await cp.mark_url_failed(url, f"local-md-error: {err}")

    # =========
    # 5) Remote crawl (stream) for remaining → save HTML/MD only
    # =========
    if remote_candidates:
        logger.info("[%s] Remote crawl (HTML/MD only): %d URL(s)", hojin_id, len(remote_candidates))
        remote_cfg = _mk_remote_config_for_pipeline(pipeline, args)
        dispatcher = _make_dispatcher(max_concurrency_for_company)
        remote_urls = [u.get("final_url") or u.get("url") for u in remote_candidates if (u.get("final_url") or u.get("url"))]

        async with AsyncWebCrawler() as crawler:
            stream = await crawler.arun_many(urls=remote_urls, config=remote_cfg, dispatcher=dispatcher)
            if hasattr(stream, "__aiter__"):
                async for r in stream:
                    url = getattr(r, "url", None)
                    if not url:
                        continue
                    if getattr(r, "success", False):
                        phases: List[str] = []
                        if "html" in pipeline_set: phases.append("html")
                        if ("markdown" in pipeline_set) or llm_in_pipeline: phases.append("markdown")
                        s = _save_result_for_pipeline(hojin_id, url, phases, r, logger, args.md_min_words)
                        saved_html += int(s["saved_html"])
                        saved_md += int(s["saved_md"])
                        md_suppressed += int(s["md_suppressed"])
                        if not llm_in_pipeline:
                            await cp.mark_url_done(url)
                    else:
                        err = getattr(r, "error_message", "unknown-error")
                        code = getattr(r, "status_code", None)
                        logger.warning("[%s] Failed: %s status=%s err=%s", hojin_id, url, code, err)
                        await cp.mark_url_failed(url, f"http-error: {code} {err}")
            else:
                for r in (stream or []):
                    url = getattr(r, "url", None)
                    if not url:
                        continue
                    if getattr(r, "success", False):
                        phases: List[str] = []
                        if "html" in pipeline_set: phases.append("html")
                        if ("markdown" in pipeline_set) or llm_in_pipeline: phases.append("markdown")
                        s = _save_result_for_pipeline(hojin_id, url, phases, r, logger, args.md_min_words)
                        saved_html += int(s["saved_html"])
                        saved_md += int(s["saved_md"])
                        md_suppressed += int(s["md_suppressed"])
                        if not llm_in_pipeline:
                            await cp.mark_url_done(url)
                    else:
                        err = getattr(r, "error_message", "unknown-error")
                        code = getattr(r, "status_code", None)
                        logger.warning("[%s] Failed: %s status=%s err=%s", hojin_id, url, code, err)
                        await cp.mark_url_failed(url, f"http-error: {code} {err}")

    # =========
    # 6) LLM PHASE (local over saved Markdown, non-streaming)
    # =========
    if llm_in_pipeline:
        llm_targets: List[Tuple[str, Path]] = []

        # If we resumed from url_index, use it to locate markdowns quickly
        if used_index_for_urls and url_index:
            for url in seeded_urls:
                ent = url_index.get(url, {})
                md_path = ent.get("markdown_path")
                if md_path:
                    p = Path(md_path)
                    if p.exists():
                        llm_targets.append((url, p))

        # Fallback: normal resolution via _prefer_local_md
        if not llm_targets:
            for url in seeded_urls:
                md_path = _prefer_local_md(hojin_id, url)
                if md_path:
                    llm_targets.append((url, md_path))

        logger.info("[%s] LLM local pass over Markdown: %d URL(s)", hojin_id, len(llm_targets))

        async with AsyncWebCrawler() as local_crawler:
            raw_inputs: List[str] = []
            ordered_urls: List[str] = []
            for url, md_path in llm_targets:
                text = md_path.read_text(encoding="utf-8", errors="ignore")
                ok, _ = should_save_markdown(text, min_meaningful_words=args.md_min_words, url=url)
                if not ok:
                    md_suppressed += 1
                    await cp.mark_url_done(url)
                    continue
                raw_inputs.append(f"raw:{text}")
                ordered_urls.append(url)

            if raw_inputs:
                llm_cfg = _mk_llm_config_for_markdown_input(args)
                results = await local_crawler.arun_many(raw_inputs, config=llm_cfg)
                for i, r in enumerate(results or []):
                    url = ordered_urls[i]
                    if getattr(r, "success", False):
                        s = _save_result_for_pipeline(hojin_id, url, ["llm"], r, logger, args.md_min_words)
                        saved_json += int(s["saved_json"])
                        await cp.mark_url_done(url)
                    else:
                        err = getattr(r, "error_message", "local-llm-fail")
                        logger.warning("[%s] Local MD→LLM failed: %s err=%s", hojin_id, url, err)
                        await cp.mark_url_failed(url, f"local-llm-error: {err}")

        # Mark any leftover URLs (no markdown) as done to avoid reprocessing in-session
        for url in seeded_urls:
            if _find_existing_artifact(hojin_id, url, stage="llm"):
                continue
            if not _prefer_local_md(hojin_id, url):
                await cp.mark_url_done(url)

    # Finish
    cp.data.setdefault("saves", {}).update({
        "saved_html_total": saved_html,
        "saved_md_total": saved_md,
        "saved_json_total": saved_json,
        "md_suppressed_total": md_suppressed,
    })
    await cp.save()

    await cp.mark_finished()
    write_last_crawl_date(hojin_id)
    logger.info(
        "[%s] DONE: processed=%s/%s failed=%s | saved_html=%d saved_md=%d saved_json=%d md_suppressed=%d",
        hojin_id,
        cp.data.get("urls_done"), cp.data.get("urls_total"), cp.data.get("urls_failed"),
        saved_html, saved_md, saved_json, md_suppressed,
    )


# ----------------------------
# Concurrency helpers
# ----------------------------

def _per_company_slots(n_companies: int, max_slots: int) -> int:
    if n_companies <= 1:
        return max_slots
    return max(2, max_slots // min(n_companies, max_slots))


# ----------------------------
# Main async
# ----------------------------

async def main_async() -> None:
    args = _parse_args()

    # Logging
    level = getattr(logging, args.log_level)
    log_ext = LoggingExtension(global_level=level, per_company_level=level)
    root_logger = logging.getLogger("run_crawl")
    root_logger.setLevel(level)

    pipeline = _parse_pipeline(args)
    root_logger.info("Pipeline=%s", pipeline)
    root_logger.info("Seeder live-check: %s", "ON" if args.live_check else "OFF")
    root_logger.info(
        "MD config: min_words=%d threshold=%.2f thresh_type=%s min_block_words=%d src=%s ignore_links=%s ignore_images=%s",
        args.md_min_words, args.md_threshold, args.md_threshold_type, args.md_min_block_words,
        args.md_content_source, bool(args.md_ignore_links), bool(args.md_ignore_images),
    )
    root_logger.info("LLM config: presence_only=%s", args.presence_only)

    # Load input companies (single CSV or directory)
    if getattr(args, "csv_dir", None):
        csv_dir: Path = args.csv_dir
        if not csv_dir.exists() or not csv_dir.is_dir():
            root_logger.error("--csv-dir is not a directory: %s", csv_dir)
            log_ext.close()
            return
        files = sorted(p for p in csv_dir.rglob("*.csv") if p.is_file())
        if not files:
            root_logger.error("No CSV files found under: %s", csv_dir)
            log_ext.close()
            return
        root_logger.info("Discovered %d CSV file(s) in %s", len(files), csv_dir)
        companies: List[CompanyInput] = []
        for f in files:
            companies.extend(load_companies_from_csv(f, limit=None))
        if args.limit is not None and args.limit > 0:
            companies = companies[:args.limit]
            root_logger.info("Applied global limit: %d", args.limit)
    else:
        companies = load_companies_from_csv(args.csv, limit=args.limit)

    if not companies:
        root_logger.error("No valid rows in CSV input. Exiting.")
        log_ext.close()
        return

    # Patterns
    include_patterns = DEFAULT_INCLUDE_PATTERNS + ([p.strip() for p in args.include.split(",") if p.strip()] if args.include else [])
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + ([p.strip() for p in args.exclude.split(",") if p.strip()] if args.exclude else [])

    # Language regions
    lang_regions = set(x.strip().lower() for x in args.lang_accept_en_regions.split(",") if x.strip())

    n = len(companies)
    per_company = _per_company_slots(n, args.max_slots)
    root_logger.info("Loaded %d companies | global slots=%d → per-company=%d", n, args.max_slots, per_company)
    root_logger.debug("Include patterns: %s", include_patterns)
    root_logger.debug("Exclude patterns: %s", exclude_patterns)

    # Checkpoints
    cp_mgr = CheckpointManager()
    await cp_mgr.mark_global_start()

    sem = asyncio.Semaphore(min(n, args.max_slots))  # bound parallel companies

    async def _runner(c: CompanyInput):
        async with sem:
            token = log_ext.set_company_context(c.hojin_id)
            try:
                await _crawl_company(
                    c,
                    pipeline=pipeline,
                    seeding_source=args.seeding_source,
                    max_concurrency_for_company=per_company,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    respect_crawl_date=args.respect_crawl_date,
                    bypass_local=args.bypass_local,
                    force_seeder_cache=args.force_seeder_cache,
                    bm25_query=args.query,
                    bm25_score_threshold=args.score_threshold,
                    max_urls=args.max_urls,
                    company_max_pages=args.company_max_pages,
                    hits_per_sec=args.hits_per_sec,
                    cp_mgr=cp_mgr,
                    log_ext=log_ext,
                    discover_brands=args.discover_brands,
                    drop_universal_externals=args.drop_universal_externals,
                    lang_primary=args.lang_primary.lower(),
                    lang_accept_en_regions=lang_regions,
                    lang_strict_cctld=args.lang_strict_cctld,
                    args=args,
                )
            finally:
                log_ext.reset_company_context(token)

    tasks = [asyncio.create_task(_runner(c)) for c in companies]
    try:
        await asyncio.gather(*tasks)
    finally:
        summary = await cp_mgr.summary()
        root_logger.info("Session summary:")
        for hid, s in summary.items():
            root_logger.info("  %s: %s/%s done, %s failed, ratio=%.2f%%, finished=%s",
                             hid, s["done"], s["total"], s["failed"], 100.0 * s["ratio"], s["finished"])
        log_ext.close()


# ----------------------------
# Entrypoint
# ----------------------------

def main() -> None:
    asyncio.run(main_async())

if __name__ == "__main__":
    main()