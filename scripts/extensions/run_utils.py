from __future__ import annotations

import argparse
import errno
import json
import time
import logging
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import deque
import re
from urllib.parse import urlparse


from crawl4ai import CacheMode, CrawlerRunConfig, RateLimiter
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

from components.llm_extractor import build_llm_extraction_strategy
from components.md_generator import build_default_md_generator

from configs import language_settings as lang_cfg
from configs.crawler_settings import crawler_base_cfg

from extensions.filtering import (
    DEFAULT_EXCLUDE_PATTERNS as FALLBACK_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS as FALLBACK_INCLUDE_PATTERNS,
)

from extensions.output_paths import ensure_company_dirs

logger = logging.getLogger(__name__)

class _NullMetrics:
    def incr(self, name: str, value: float = 1.0, **labels: Any) -> None: ...
    def observe(self, name: str, value: float, **labels: Any) -> None: ...
    def set(self, name: str, value: float, **labels: Any) -> None: ...
    def time(self, name: str, **labels: Any):
        from contextlib import contextmanager
        @contextmanager
        def _cm():
            yield
        return _cm()
metrics = _NullMetrics()  # type: ignore

__all__ = [
    # language/pattern helpers
    "get_default_include_patterns",
    "get_default_exclude_patterns",
    "parse_patterns_csv",
    "parse_lang_accept_regions",
    # cli & pipeline
    "build_parser",
    "parse_pipeline",
    "make_dispatcher",
    # crawler stage configs
    "mk_md_config",
    "mk_llm_config_for_markdown_input",  # kept for compatibility alias
    "mk_llm_config",
    "mk_remote_config_for_pipeline",      # back-compat no-op alias
    # url seeder kwarg builder
    "build_seeder_kwargs",
    # misc utilities used elsewhere
    "aggregate_seed_by_root",
    "classify_failure",
    "find_existing_artifact",
    "prefer_local_html",
    "prefer_local_md",
    "load_url_index",
    "upsert_url_index",
    "write_url_index_seed_only",
    "read_last_crawl_date",
    "write_last_crawl_date",
    "filter_by_last_crawl_date",
    "company_log_tail_has_error",
    "recommend_company_timeouts",
    "is_llm_extracted_empty",
]

# -----------------------
# Error scan defaults
# -----------------------
_ERROR_DEFAULT_PATTERNS = (
    r"\b(traceback)\b",
    r"\b(exception|error)\b",
    r"\btimeout\b",
    r"\bsoft[_\s-]?timeout\b",
    r"\btransport\b",
    r"i/o operation on closed pipe",
    r"too many open files|\bemfile\b",
    r"\bnet::|dns|name_not_resolved|connection reset",
)

META_PATH_NAME = "crawl_meta.json"
URL_INDEX_NAME = "url_index.json"

# -----------------------
# Language-sensitive getters
# -----------------------
def get_default_include_patterns() -> List[str]:
    try:
        pats = lang_cfg.get("DEFAULT_INCLUDE_PATTERNS", None)
        if pats and isinstance(pats, (list, tuple)):
            return list(pats)
    except Exception:
        logger.debug("[run_utils] language_settings.get(DEFAULT_INCLUDE_PATTERNS) failed; using fallback.")
    return list(FALLBACK_INCLUDE_PATTERNS)

def get_default_exclude_patterns() -> List[str]:
    try:
        pats = lang_cfg.get("DEFAULT_EXCLUDE_PATTERNS", None)
        if pats and isinstance(pats, (list, tuple)):
            return list(pats)
    except Exception:
        logger.debug("[run_utils] language_settings.get(DEFAULT_EXCLUDE_PATTERNS) failed; using fallback.")
    return list(FALLBACK_EXCLUDE_PATTERNS)

def parse_patterns_csv(raw: Optional[str]) -> List[str]:
    """
    Parse a comma/semicolon separated pattern list, ignore empties/whitespace.
    Accepts '*' wildcards. Returns [] if nothing provided.
    """
    if not raw:
        return []
    parts: List[str] = []
    for tok in re.split(r"[;,]", raw):
        t = (tok or "").strip()
        if t:
            parts.append(t)
    return parts

def parse_lang_accept_regions(raw: Optional[str]) -> set[str]:
    if not raw:
        return set()
    return { t.strip().lower() for t in re.split(r"[,\s]+", raw) if t.strip() }

# -----------------------
# CLI + utility functions
# -----------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seed → filter → fetch (Markdown + reference HTML) → local LLM over saved Markdown"
    )

    # --- Input sources (multi-format via source_loader) ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--source",
        type=Path,
        help=(
            "Input data file path (supported: .csv, .tsv, .xlsx, .xls, .json, .jsonl, .ndjson, "
            ".parquet, ".replace('"', '\\"') + ".feather, .dta, .sas7bdat, .sav)"
        ),
    )
    src.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing one or more supported data files (scanned recursively).",
    )
    p.add_argument(
        "--source-pattern",
        type=str,
        default="*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav",
        help="Comma-separated glob(s) to scan inside --source-dir.",
    )

    # --- Pipeline stages ---
    p.add_argument("--pipeline", type=str, default="markdown,llm",
                   help=("Comma-separated stages from {seed,markdown,llm}"))

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-slots", type=int, default=20)

    grp = p.add_argument_group("Slot allocator")
    grp.add_argument("--slot-cap-per-company", type=int, default=16,
                     help="Hard ceiling of concurrent slots for any single company (default: 16).")
    grp.add_argument("--slot-min-per-company", type=int, default=2,
                     help="Floor of concurrent slots for any single company (default: 2).")
    grp.add_argument("--slot-tail-frac", type=float, default=0.75,
                     help="Start boosting when finished/total >= this fraction (default: 0.75).")
    grp.add_argument("--slot-tail-cap", type=int, default=0,
                     help="Optional higher cap applied only in the tail; 0 to reuse the per-company cap.")

    # --- Seeding / filtering ---
    p.add_argument("--seeding-source", type=str, default="cc+sitemap",
                   choices=["sitemap", "cc", "sitemap+cc", "cc+sitemap"])

    p.add_argument("--include", type=str, default="",
                   help="CSV of include URL patterns (overrides language defaults).")
    p.add_argument("--exclude", type=str, default="",
                   help="CSV of exclude URL patterns (overrides language defaults).")

    p.add_argument("--query", type=str, default=None)
    p.add_argument("--score-threshold", type=float, default=None)
    p.add_argument("--force-seeder-cache", action="store_true")
    p.add_argument("--bypass-local", action="store_true")
    p.add_argument("--respect-crawl-date", action="store_true")
    p.add_argument("--live-check", action="store_true")

    p.add_argument("--require-presence", action="store_true",
                   help="Only process Markdown/LLM for URLs that have has_offering==1 and presence_checked in url_index.json")

    # --- Brand / language / externals ---
    p.add_argument("--lang", type=str, default="en",
                   help="2-letter language code to use for language-sensitive constants (e.g. en, ja, vi)")


    # Dataset externals list (full universe), distinct from the current --source batch
    egrp = p.add_mutually_exclusive_group()
    egrp.add_argument("--use-externals-list", dest="use_externals_list", action="store_true", default=True,
                    help="If set, build an externals list of company hosts from a separate full dataset "
                        "and use it to block cross-company externals only in brand discovery and url_index.")
    egrp.add_argument("--no-use-externals-list", dest="use_externals_list", action="store_false")
    p.add_argument("--externals-list-source", type=Path,
                help="Full dataset file (supported formats) whose company URLs form the externals list.")
    p.add_argument("--externals-list-dir", type=Path,
                help="Directory containing one or more supported data files for the externals list.")
    p.add_argument("--externals-list-pattern", type=str,
                default="*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav",
                help="Comma-separated glob(s) to scan inside --externals-list-dir.")

    # Mutually exclusive toggles for brand discovery (default: True)
    dgrp = p.add_mutually_exclusive_group()
    dgrp.add_argument("--discover-brands", dest="discover_brands", action="store_true", default=True)
    dgrp.add_argument("--no-discover-brands", dest="discover_brands", action="store_false")

    # Mutually exclusive toggles for dropping universal externals (default: True)
    ugr = p.add_mutually_exclusive_group()
    ugr.add_argument("--drop-universal-externals", dest="drop_universal_externals", action="store_true", default=True)
    ugr.add_argument("--no-drop-universal-externals", dest="drop_universal_externals", action="store_false")

    p.add_argument("--lang-primary", type=str, default="en")
    p.add_argument("--lang-accept-en-regions", type=str, default="us,gb,ca,au,nz,ie,sg")
    p.add_argument("--lang-strict-cctld", action="store_true")

    p.add_argument("--max-urls", type=int, default=-1)
    p.add_argument("--company-max-pages", type=int, default=-1)
    p.add_argument("--hits-per-sec", type=int, default=50)

    # --- Dual BM25 (for seeder) ---
    p.add_argument("--use-dual-bm25", dest="use_dual_bm25", action="store_true", default=True)
    p.add_argument("--no-use-dual-bm25", dest="use_dual_bm25", action="store_false")
    p.add_argument("--dual-alpha", type=float, default=0.5)

    # --- Markdown generator knobs ---
    p.add_argument("--md-min-words", type=int, default=5)
    p.add_argument("--md-threshold", type=float, default=0.12)
    p.add_argument("--md-threshold-type", choices=["dynamic", "fixed"], default="fixed")
    p.add_argument("--md-min-block-words", type=int, default=30)
    p.add_argument("--md-content-source", choices=["cleaned_html", "fit_html", "raw_html"], default="fit_html")
    p.add_argument("--md-ignore-links", action="store_true")
    p.add_argument("--md-ignore-images", action="store_true")
    p.add_argument("--md-body-width", type=int, default=0)
    p.add_argument("--md-cookie-max-frac", type=float, default=0.15)

    # Mutually exclusive toggles for MD structure gating (default: True)
    mgrp = p.add_mutually_exclusive_group()
    mgrp.add_argument("--md-require-structure", dest="md_require_structure", action="store_true", default=True)
    mgrp.add_argument("--md-no-require-structure", dest="md_require_structure", action="store_false")

    # --- Page timeouts + remote batch knobs ---
    p.add_argument("--page-timeout-ms", type=int, default=120000,
                   help="Per-page navigation/processing timeout in milliseconds (default: 120000).")
    p.add_argument("--remote-batch-size", type=int, default=64,
                   help="How many URLs to send in one remote fetch batch (default: 64).")

    # --- LLM ---
    p.add_argument("--presence-only", action="store_true")
    p.add_argument("--llm-per-run-timeout", type=int, default=900,
                help="Overall LLM run timeout in seconds (default 900).")
    p.add_argument("--llm-per-item-timeout", type=int, default=180,
                help="Per-item streaming timeout in seconds when iterating the LLM stream (default 180).")

    # --- Logging ---
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--max-open-company-logs", type=int, default=128)
    p.add_argument("--enable-session-log", action="store_true")

    # --- Robustness ---
    p.add_argument("--company-timeout", type=int, default=1800)
    p.add_argument("--stall-interval", type=int, default=30)
    p.add_argument("--stall-timeout", type=int, default=1800)

    # --- Failure classification knobs ---
    p.add_argument("--treat-timeouts-as-transport", action="store_true")
    p.add_argument("--retry-transport", type=int, default=2)
    p.add_argument("--retry-soft-timeout", type=int, default=1)
    return p

def parse_pipeline(pipeline_arg: str) -> List[str]:
    stages = [s.strip().lower() for s in (pipeline_arg or "").split(",") if s.strip()]
    allowed = {
        ("seed",), ("markdown",), ("llm",),
        ("seed", "markdown"), ("markdown", "llm"),
        ("seed", "llm"), ("seed", "markdown", "llm"),
    }
    if tuple(stages) not in allowed:
        raise SystemExit(
            f"Invalid --pipeline '{pipeline_arg}'. Use combinations of: seed, markdown, llm"
        )
    return stages

# -----------------------
# Artifact / index helpers
# -----------------------
def _url_hash(url: str) -> str:
    return sha1(url.encode()).hexdigest()[:8]

def _ext_for_stage(stage: str) -> str:
    return {"html": ".html", "markdown": ".md", "llm": ".json"}[stage]

def _stage_dir_key(stage: str) -> str:
    return {"html": "html", "markdown": "markdown", "llm": "json"}[stage]

def find_existing_artifact(bvdid: str, url: str, stage: str) -> Optional[Path]:
    d = ensure_company_dirs(bvdid)[_stage_dir_key(stage)]
    h = _url_hash(url)
    ext = _ext_for_stage(stage)
    for p in d.glob(f"*{h}{ext}"):
        return p
    return None

def prefer_local_html(bvdid: str, url: str) -> Optional[Path]:
    return find_existing_artifact(bvdid, url, "html")

def prefer_local_md(bvdid: str, url: str) -> Optional[Path]:
    return find_existing_artifact(bvdid, url, "markdown")

def _meta_path(bvdid: str) -> Path:
    return ensure_company_dirs(bvdid)["checkpoints"] / META_PATH_NAME

def _url_index_path(bvdid: str) -> Path:
    return ensure_company_dirs(bvdid)["checkpoints"] / URL_INDEX_NAME

def load_url_index(bvdid: str) -> Dict[str, Any]:
    p = _url_index_path(bvdid)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _retry_emfile(fn: Callable[[], None], attempts: int = 6, delay: float = 0.15) -> None:
    for i in range(attempts):
        try:
            fn()
            return
        except OSError as e:
            if e.errno == errno.EMFILE or "Too many open files" in str(e):
                time.sleep(delay * (2 ** i))
                continue
            raise

def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    def _do():
        tmp.write_text(data, encoding=encoding)
        tmp.replace(path)
    _retry_emfile(_do)

def _atomic_write_json(path: Path, obj: Any) -> None:
    content = json.dumps(obj, indent=2, ensure_ascii=False)
    _atomic_write_text(path, content, "utf-8")

# ---------------------------------------------------------------------
# URL index status precedence + writers (UPDATED)
# ---------------------------------------------------------------------

# Higher number wins. Special handling for `filtered_*` and `presence_checked`.
_STATUS_PRECEDENCE: Dict[str, int] = {
    "queued": 10,
    # filtered_* gets 12 via _status_rank (below)
    "seeded": 20,
    "presence_checked": 25,     # meta; shouldn't override html/markdown/llm
    "html_saved": 30,
    "markdown_retry": 31,
    "markdown_suppressed": 32,
    "markdown_saved": 40,
    "llm_extracted_empty": 50,
    "llm_extracted": 60,
    "_other_": 35,
}

def _status_rank(s: Optional[str]) -> int:
    if not s:
        return -1
    if isinstance(s, str) and s.startswith("filtered_"):
        return 12
    return _STATUS_PRECEDENCE.get(s, _STATUS_PRECEDENCE["_other_"])

def _should_update_status(old: Optional[str], new: Optional[str]) -> bool:
    if not new:
        return False
    if new == "presence_checked":
        # Only lift to presence_checked if we're at most seeded.
        return old in (None, "queued", "seeded")
    if isinstance(new, str) and new.startswith("filtered_"):
        # filtered_* never overrides seeded or beyond
        return old in (None, "queued")
    return _status_rank(new) >= _status_rank(old)

def upsert_url_index(
    bvdid: str,
    url: str,
    *,
    html_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    has_offering: Optional[int] = None,
    status: Optional[str] = None,
    presence_checked: Optional[bool] = None,
    llm_extracted_empty: Optional[bool] = None,
    markdown_words: Optional[int] = None,
    score: Optional[float] = None,
    skip_status_update: bool = False,
    **extra: Any,
) -> None:
    """
    Upsert a single URL entry in outputs/{bvdid}/checkpoints/url_index.json

    Status handling:
      - If `skip_status_update=True`, status won’t change at all.
      - Otherwise, we only upgrade status using precedence rules (never downgrade).
      - Saving an artifact auto-lifts status to the corresponding stage if that’s an upgrade.
    """
    idx = load_url_index(bvdid)
    ent: Dict[str, Any] = dict(idx.get(url, {}))
    now = datetime.now(timezone.utc).isoformat()

    # Preserve earliest discovery timestamp
    ent.setdefault("discovered_at", ent.get("discovered_at", now))

    # Score (rounded)
    if score is not None:
        try:
            ent["score"] = round(float(score), 6)
        except Exception:
            ent["score"] = score

    # Current status snapshot for precedence checks
    current_status: Optional[str] = ent.get("status")

    # --- Artifact paths (idempotent) with auto-lift proposals -----------
    proposed_status: Optional[str] = status  # start with explicit status if provided

    if html_path is not None:
        new_html = str(html_path)
        if ent.get("html_path") != new_html:
            ent["html_path"] = new_html
            ent["html_saved_at"] = now
        if not skip_status_update and proposed_status is None:
            proposed_status = "html_saved"

    if markdown_path is not None:
        new_md = str(markdown_path)
        if ent.get("markdown_path") != new_md:
            ent["markdown_path"] = new_md
            ent["markdown_saved_at"] = now
        if markdown_words is not None:
            try:
                ent["markdown_words"] = int(markdown_words)
            except Exception:
                ent["markdown_words"] = 0
        if not skip_status_update and proposed_status is None:
            proposed_status = "markdown_saved"

    if json_path is not None:
        new_json = str(json_path)
        if ent.get("json_path") != new_json:
            ent["json_path"] = new_json
            ent["json_saved_at"] = now
        if not skip_status_update and proposed_status is None:
            proposed_status = "llm_extracted_empty" if llm_extracted_empty else "llm_extracted"

    # --- Presence / LLM flags -------------------------------------------
    if has_offering is not None:
        try:
            ent["has_offering"] = int(has_offering)
        except Exception:
            ent["has_offering"] = 0
        ent["has_offering_at"] = now

    if presence_checked is not None:
        ent["presence_checked"] = bool(presence_checked)
        ent["presence_checked_at"] = now
        if not skip_status_update and proposed_status is None and presence_checked:
            proposed_status = "presence_checked"

    if llm_extracted_empty is not None:
        ent["llm_extracted_empty"] = bool(llm_extracted_empty)
        ent["llm_extracted_empty_at"] = now
        if not skip_status_update and proposed_status is None and llm_extracted_empty:
            proposed_status = "llm_extracted_empty"

    # Optional reason
    if "reason" in extra and isinstance(extra["reason"], str):
        ent["reason"] = extra["reason"]

    # --- Apply status with precedence (this is the key fix) -------------
    if not skip_status_update and proposed_status:
        if _should_update_status(current_status, proposed_status):
            ent["status"] = proposed_status
    elif "status" not in ent and proposed_status:
        # If there is no prior status at all, set it.
        ent["status"] = proposed_status

    # Update write timestamp
    ent["updated_at"] = now

    idx[url] = ent
    _atomic_write_json(_url_index_path(bvdid), idx)
    metrics.incr("url_index.upsert", bvdid=bvdid)

def write_url_index_seed_only(bvdid: str, seeded_urls: List[str]) -> int:
    """
    Ensure all URLs are present with at least status=seeded, but never
    downgrade an existing higher status.
    """
    idx = load_url_index(bvdid)
    now = datetime.now(timezone.utc).isoformat()
    for u in seeded_urls:
        if not u:
            continue
        ent: Dict[str, Any] = dict(idx.get(u, {}))
        ent.setdefault("discovered_at", now)
        cur = ent.get("status")
        # Lift to 'seeded' only if that's an upgrade by precedence
        if _should_update_status(cur, "seeded"):
            ent["status"] = "seeded"
        ent["updated_at"] = now
        idx[u] = ent
    _atomic_write_json(_url_index_path(bvdid), idx)
    metrics.set("url_index.size", float(len(idx)), bvdid=bvdid)
    return len(idx)

# -----------------------
# Crawl date helpers
# -----------------------
def read_last_crawl_date(bvdid: str) -> Optional[datetime]:
    p = _meta_path(bvdid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        v = data.get("last_crawled_at")
        return datetime.fromisoformat(v) if v else None
    except Exception:
        return None

def write_last_crawl_date(bvdid: str, dt: Optional[datetime] = None) -> None:
    meta_p = _meta_path(bvdid)
    new_ts = (dt or datetime.now(timezone.utc)).isoformat()
    payload: Dict[str, Any] = {"last_crawled_at": new_ts}
    if meta_p.exists():
        try:
            existing = json.loads(meta_p.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing["last_crawled_at"] = new_ts
                payload = existing
        except Exception:
            pass
    _atomic_write_json(meta_p, payload)
    metrics.set("company.last_crawl_epoch", time.time(), bvdid=bvdid)

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

# -----------------------
# MD / LLM run configs
# -----------------------
def build_md_generator_from_args(args: argparse.Namespace):
    return build_default_md_generator(
        threshold=args.md_threshold,
        threshold_type=args.md_threshold_type,
        min_word_threshold=args.md_min_block_words,
        body_width=args.md_body_width,
        ignore_links=args.md_ignore_links,
        ignore_images=args.md_ignore_images,
        content_source=args.md_content_source,
        min_meaningful_words=args.md_min_words,
        interstitial_max_share=0.60,
        interstitial_min_hits=2,
        cookie_max_fraction=getattr(args, "md_cookie_max_frac", 0.15),
        require_structure=bool(getattr(args, "md_require_structure", True)),
    )

def mk_md_config(args: argparse.Namespace) -> CrawlerRunConfig:
    return crawler_base_cfg.clone(
        markdown_generator=build_md_generator_from_args(args),
        cache_mode=CacheMode.BYPASS,
        stream=False,
        page_timeout=int(getattr(args, "page_timeout_ms", 120000))  # 120s default
    )

# Back-compat alias (kept so external calls don't break)
def mk_llm_config_for_markdown_input(args: argparse.Namespace) -> CrawlerRunConfig:
    return mk_llm_config(args)

def mk_llm_config(args: argparse.Namespace) -> CrawlerRunConfig:
    extraction = build_llm_extraction_strategy(presence_only=bool(getattr(args, "presence_only", False)))
    logger.debug("[run_utils] mk_llm_config_for_markdown_input: extraction preview=%s", {
        "provider": getattr(extraction.llm_config, "provider", None) if hasattr(extraction, "llm_config") else None,
        "base_url": getattr(extraction.llm_config, "base_url", None) if hasattr(extraction, "llm_config") else None,
        "schema_type": getattr(extraction, "schema", None),
        "input_format": getattr(extraction, "input_format", None),
    })
    return crawler_base_cfg.clone(
        extraction_strategy=extraction,
        cache_mode=CacheMode.BYPASS,
        stream=True,
    )

# No-op, kept for compatibility with older code paths
def mk_remote_config_for_pipeline(*args, **kwargs) -> None:
    return None

def make_dispatcher(max_concurrency: int) -> MemoryAdaptiveDispatcher:
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=85.0,
        check_interval=1.0,
        max_session_permit=max_concurrency,
        rate_limiter=RateLimiter(base_delay=(0.5, 1.2), max_delay=20.0, max_retries=2),
        monitor=None,
    )

# -----------------------
# Discovery helpers (seed stats)
# -----------------------
def _normalize_seed_root(url: str) -> str:
    """
    Normalize any seed_root URL to a canonical site root:

        "https://www.example.com/foo/bar"  -> "https://example.com/"
        "example.com/path"                 -> "https://example.com/"
        "//example.com"                    -> "https://example.com/"

    Used for seed aggregation so that all subpages under the same host
    collapse into a single root entry.
    """
    s = (url or "").strip()
    if not s:
        return ""

    # Ensure we have a scheme so urlparse works consistently
    if s.startswith("//"):
        s = "https:" + s
    elif "://" not in s:
        s = "https://" + s.lstrip("/")

    try:
        pu = urlparse(s)
    except Exception:
        return ""

    host = (pu.hostname or "").lower().strip(".")
    if host.startswith("www.") and len(host) > 4:
        host = host[4:]
    if not host:
        return ""

    # We intentionally normalize scheme to https for aggregation
    return f"https://{host}/"

def aggregate_seed_by_root(items: Iterable[Dict[str, Any]], base_root: str) -> Dict[str, Any]:
    """
    Aggregate seed statistics by canonical site root, not by individual page.

    Example:
        seed_root values:
            https://marleycoffee.com/
            https://marleycoffee.com/where-to-buy-usa/
            https://www.marleycoffee.com/where-to-buy-canada/

        -> all collapse to:
            https://marleycoffee.com/

    Returned structure:
        {
          "seed_counts_by_root": {
              "https://mother-parkers.com/": 50,
              "https://marleycoffee.com/": 3,
          },
          "seed_roots": [
              "https://marleycoffee.com/",
              "https://mother-parkers.com/"
          ],
          "seed_brand_roots": [
              # everything except the normalized base_root
          ],
          "seed_brand_count": <len(seed_brand_roots)>
        }
    """
    counts: Dict[str, int] = {}
    roots: List[str] = []

    norm_base = _normalize_seed_root(base_root)

    for it in items:
        raw_root = str(it.get("seed_root") or "").strip()
        if not raw_root:
            continue
        r = _normalize_seed_root(raw_root)
        if not r:
            continue
        counts[r] = counts.get(r, 0) + 1
        roots.append(r)

    unique_roots = sorted(set(roots))
    brand_roots = [r for r in unique_roots if r != norm_base]

    return {
        "seed_counts_by_root": counts,
        "seed_roots": unique_roots,
        "seed_brand_roots": brand_roots,
        "seed_brand_count": len(brand_roots),
    }

# -----------------------
# Mode-specific kwargs builders
# -----------------------
def _resolve_patterns_from_args(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    inc = parse_patterns_csv(getattr(args, "include", "")) or get_default_include_patterns()
    exc = parse_patterns_csv(getattr(args, "exclude", "")) or get_default_exclude_patterns()
    return inc, exc

def _resolve_lang_regions_from_args(args: argparse.Namespace) -> set[str]:
    return parse_lang_accept_regions(getattr(args, "lang_accept_en_regions", ""))

def build_seeder_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build kwargs for url_seeder.seed_urls() / discover_and_crawl().
    """
    include, exclude = _resolve_patterns_from_args(args)
    return {
        "source": getattr(args, "seeding_source", "cc"),
        "include": include,
        "exclude": exclude,
        "query": getattr(args, "query", None),
        "score_threshold": getattr(args, "score_threshold", None),
        "pattern": "*",
        "extract_head": True,
        "live_check": bool(getattr(args, "live_check", False)),
        "max_urls": int(getattr(args, "max_urls", -1)),
        "company_max_pages": int(getattr(args, "company_max_pages", -1)),
        "concurrency": int(getattr(args, "max_slots", 20)),
        "hits_per_sec": int(getattr(args, "hits_per_sec", 50)),
        "force": bool(getattr(args, "force_seeder_cache", False)),
        "verbose": logger.isEnabledFor(logging.DEBUG),
        "drop_universal_externals": bool(getattr(args, "drop_universal_externals", True)),
        "lang_primary": getattr(args, "lang_primary", "en"),
        "lang_accept_en_regions": _resolve_lang_regions_from_args(args),
        "lang_strict_cctld": bool(getattr(args, "lang_strict_cctld", False)),
        "discover_brands": bool(getattr(args, "discover_brands", True)),
        "auto_product_query": True,
        "product_signal_threshold": 0.30,
        "default_score_threshold_if_query": 0.25,
        "use_dual_bm25": bool(getattr(args, "use_dual_bm25", True)),
        "dual_alpha": float(getattr(args, "dual_alpha", 0.5)),
    }

# -----------------------
# Failure classification
# -----------------------
def classify_failure(err_msg: str | None, status_code: Optional[int], *, treat_timeouts_as_transport: bool) -> str:
    s = (err_msg or "").lower()

    if "download is starting" in s or "download-start" in s or "download started" in s:
        return "download"

    if "page.goto" in s and "timeout" in s:
        return "transport" if treat_timeouts_as_transport else "soft_timeout"

    if "net::" in s:
        # IMPORTANT: HTTP response code failure (Playwright) – do NOT retry like transport
        if "err_http_response_code_failure" in s:
            return "other"
        if "timeout" in s or "err_connection_timed_out" in s or "err_timed_out" in s:
            return "transport" if treat_timeouts_as_transport else "soft_timeout"
        return "transport"

    transport_indicators = (
        "name_not_resolved",
        "name not resolved",
        "dnserr",
        "dns",
        "eai_again",
        "socket.gaierror",
        "gaierror",
        "connection reset",
        "connection refused",
        "refused",
        "ecoff",
        "econnrefused",
        "econnreset",
        "connection aborted",
        "no route to host",
        "network is unreachable",
        "host unreachable",
        "host is down",
        "temporary failure in name resolution",
        "tls handshake",
        "ssl error",
        "ssl.sslerror",
        "ssl.sslcertverificationerror",
        "certificat",
    )
    for tok in transport_indicators:
        if tok in s:
            return "transport"

    if status_code is not None:
        try:
            sc = int(status_code)
            if sc == 408:
                return "transport" if treat_timeouts_as_transport else "soft_timeout"
            return "other"
        except Exception:
            pass

    if "timeout" in s or "timed out" in s:
        return "transport" if treat_timeouts_as_transport else "soft_timeout"

    if "navigation failed" in s or "navigation timeout" in s:
        return "transport" if treat_timeouts_as_transport else "soft_timeout"

    return "other"

# -----------------------
# Misc
# -----------------------
def per_company_slots(n_companies: int, max_slots: int) -> int:
    # Backward-compat helper (no longer used by run_crawl.py)
    if n_companies <= 1:
        return max_slots
    return max(2, max_slots // min(n_companies, max_slots))

def company_log_tail_has_error(
    log_path: Path,
    *,
    window_lines: int = 250,
    patterns: Iterable[str] = _ERROR_DEFAULT_PATTERNS,
) -> tuple[bool, str]:
    try:
        if not log_path or not log_path.exists():
            return (False, "")
        dq: deque[str] = deque(maxlen=max(10, int(window_lines)))
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                dq.append(line)
        blob = "\n".join(dq).lower()
        for pat in patterns:
            if re.search(pat, blob, flags=re.IGNORECASE):
                return (True, pat)
    except Exception:
        return (False, "")
    return (False, "")

def recommend_company_timeouts(
    urls_total: int,
    *,
    base_timeout: int = 1800,
    per_url_seconds: float = 0.9,
    startup_buffer: int = 300,
    max_timeout: int = 8 * 3600,
) -> dict[str, int]:
    urls_total = max(0, int(urls_total))
    est = int(startup_buffer + urls_total * per_url_seconds)
    recommended = min(max(base_timeout, est), max_timeout)
    stall = max(1800, recommended // 2)
    return {"company_timeout": recommended, "stall_timeout": stall}

def is_llm_extracted_empty(extracted: Any) -> bool:
    try:
        obj = extracted
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                return False
        if isinstance(obj, dict):
            if "offerings" in obj:
                return isinstance(obj.get("offerings"), list) and len(obj.get("offerings")) == 0
            return False
        if isinstance(obj, list):
            if not obj:
                return True
            all_empty = True
            for item in obj:
                if not isinstance(item, dict):
                    all_empty = False
                    break
                off = item.get("offerings", None)
                if not isinstance(off, list) or len(off) > 0:
                    all_empty = False
                    break
            return all_empty
    except Exception:
        return False
    return False