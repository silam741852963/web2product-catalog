from __future__ import annotations

import argparse
import errno
import json
import time
import logging
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from collections import deque
import re

from crawl4ai import CacheMode, CrawlerRunConfig, RateLimiter
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

from components.llm_extractor import build_llm_extraction_strategy
from components.md_generator import build_default_md_generator
from config import language_settings as langcfg

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

# Re-export patterns so run_crawl can import from here
__all__ = [
    "get_default_include_patterns",
    "get_default_exclude_patterns",
    "build_parser",
    "parse_pipeline",
    "make_dispatcher",
    "mk_md_config",
    "mk_llm_config_for_markdown_input",
    "mk_llm_config",
    "mk_remote_config_for_pipeline",
    "aggregate_seed_by_root",
    "classify_failure",
    "per_company_slots",
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

# Backwards-compatible alias names for external code that expects those constants:
# (They are functions below; importers should call them. Kept only for annotation clarity.)
# DEFAULT_INCLUDE_PATTERNS / DEFAULT_EXCLUDE_PATTERNS must be acquired via getters.

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
    """Return the active language spec's DEFAULT_INCLUDE_PATTERNS (fallback to code-level defaults)."""
    try:
        pats = langcfg.get("DEFAULT_INCLUDE_PATTERNS", None)
        if pats and isinstance(pats, (list, tuple)):
            return list(pats)
    except Exception:
        logger.debug("[run_utils] language_settings.get(DEFAULT_INCLUDE_PATTERNS) failed; using fallback.")
    return list(FALLBACK_INCLUDE_PATTERNS)

def get_default_exclude_patterns() -> List[str]:
    """Return the active language spec's DEFAULT_EXCLUDE_PATTERNS (fallback to code-level defaults)."""
    try:
        pats = langcfg.get("DEFAULT_EXCLUDE_PATTERNS", None)
        if pats and isinstance(pats, (list, tuple)):
            return list(pats)
    except Exception:
        logger.debug("[run_utils] language_settings.get(DEFAULT_EXCLUDE_PATTERNS) failed; using fallback.")
    return list(FALLBACK_EXCLUDE_PATTERNS)

# -----------------------
# CLI + utility functions (unchanged behavior)
# -----------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seed → filter → fetch (Markdown + reference HTML) → local LLM over saved Markdown"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Input CSV file path")
    src.add_argument("--csv-dir", type=Path, help="Directory containing one or more CSV files")

    p.add_argument("--pipeline", type=str, default="markdown,llm",
                   help=("Comma-separated stages from {seed,markdown,llm}"))

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-slots", type=int, default=20)

    p.add_argument("--seeding-source", type=str, default="sitemap+cc",
                   choices=["sitemap", "cc", "sitemap+cc"])

    p.add_argument("--include", type=str, default="")
    p.add_argument("--exclude", type=str, default="")
    p.add_argument("--query", type=str, default=None)
    p.add_argument("--score-threshold", type=float, default=None)
    p.add_argument("--force-seeder-cache", action="store_true")
    p.add_argument("--bypass-local", action="store_true")
    p.add_argument("--respect-crawl-date", action="store_true")
    p.add_argument("--live-check", action="store_true")

    p.add_argument("--require-presence", action="store_true",
                   help="Only process Markdown/LLM for URLs that have has_offering==1 and presence_checked in url_index.json")

    # Brand / language / externals
    p.add_argument("--lang", type=str, default="en", help="2-letter language code to use for language-sensitive constants (e.g. en, ja, vi)")
    p.add_argument("--discover-brands", action="store_true", default=True)
    p.add_argument("--no-discover-brands", dest="discover_brands", action="store_false")
    p.add_argument("--drop-universal-externals", action="store_true", default=True)
    p.add_argument("--lang-primary", type=str, default="en")
    p.add_argument("--lang-accept-en-regions", type=str, default="us,gb,ca,au,nz,ie,sg")
    p.add_argument("--lang-strict-cctld", action="store_true")

    p.add_argument("--max-urls", type=int, default=-1)
    p.add_argument("--company-max-pages", type=int, default=-1)
    p.add_argument("--hits-per-sec", type=int, default=50)

    # Dual BM25 controls (for url_seeder)
    p.add_argument("--use-dual-bm25", dest="use_dual_bm25", action="store_true", default=True)
    p.add_argument("--no-use-dual-bm25", dest="use_dual_bm25", action="store_false")
    p.add_argument("--dual-alpha", type=float, default=0.65)

    # Markdown generator knobs
    p.add_argument("--md-min-words", type=int, default=5)
    p.add_argument("--md-threshold", type=float, default=0.48)
    p.add_argument("--md-threshold-type", choices=["dynamic", "fixed"], default="dynamic")
    p.add_argument("--md-min-block-words", type=int, default=5)
    p.add_argument("--md-content-source", choices=["cleaned_html", "fit_html", "raw_html"], default="fit_html")
    p.add_argument("--md-ignore-links", action="store_true")
    p.add_argument("--md-ignore-images", action="store_true")
    p.add_argument("--md-body-width", type=int, default=0)
    p.add_argument("--md-cookie-max-frac", type=float, default=0.15)
    p.add_argument("--md-require-structure", action="store_true", default=True)

    # LLM
    p.add_argument("--presence-only", action="store_true")
    p.add_argument("--llm-per-run-timeout", type=int, default=900,
                help="Overall LLM run timeout in seconds (default 900).")
    p.add_argument("--llm-per-item-timeout", type=int, default=180,
                help="Per-item streaming timeout in seconds when iterating the LLM stream (default 180).")


    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--max-open-company-logs", type=int, default=128)
    p.add_argument("--enable-session-log", action="store_true")

    # Robustness
    p.add_argument("--company-timeout", type=int, default=1800)
    p.add_argument("--stall-interval", type=int, default=30)
    p.add_argument("--stall-timeout", type=int, default=1800)

    # Failure classification knobs
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

def upsert_url_index(
    bvdid: str,
    url: str,
    * ,
    html_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    has_offering: Optional[int] = None,
    status: Optional[str] = None,
    presence_checked: Optional[bool] = None,
    llm_extracted_empty: Optional[bool] = None,
    markdown_words: Optional[int] = None,
) -> None:
    idx = load_url_index(bvdid)
    ent = idx.get(url, {})
    now = datetime.now(timezone.utc).isoformat()
    if html_path:
        ent["html_path"] = str(html_path)
        ent["html_saved_at"] = now
        status = "html_saved"
    if markdown_path:
        ent["markdown_path"] = str(markdown_path)
        ent["markdown_saved_at"] = now
        status = "markdown_saved"
    if json_path:
        ent["json_path"] = str(json_path)
        ent["json_saved_at"] = now
        status = "llm_extracted"
    if has_offering is not None:
        try:
            ent["has_offering"] = int(has_offering)
            ent["has_offering_at"] = now
        except Exception:
            ent["has_offering"] = 0
            ent["has_offering_at"] = now
    if presence_checked:
        ent["presence_checked"] = True
        ent["presence_checked_at"] = now
        status = "presence_checked"
    if llm_extracted_empty is not None:
        ent["llm_extracted_empty"] = bool(llm_extracted_empty)
        ent["llm_extracted_empty_at"] = now
        if llm_extracted_empty:
            status = "llm_extracted_empty"

    if markdown_words is not None:
        try:
            ent["markdown_words"] = int(markdown_words)
        except Exception:
            ent["markdown_words"] = 0

    ent["status"] = status

    ent.setdefault("discovered_at", now)
    idx[url] = ent
    _atomic_write_json(_url_index_path(bvdid), idx)
    metrics.incr("url_index.upsert", bvdid=bvdid)

def write_url_index_seed_only(bvdid: str, seeded_urls: List[str]) -> int:
    idx = load_url_index(bvdid)
    now = datetime.now(timezone.utc).isoformat()
    for u in seeded_urls:
        if not u:
            continue
        ent = idx.get(u, {})
        ent.setdefault("status", "seeded")
        ent.setdefault("discovered_at", now)
        idx[u] = ent
    _atomic_write_json(_url_index_path(bvdid), idx)
    metrics.set("url_index.size", float(len(idx)), bvdid=bvdid)
    return len(idx)

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
    return CrawlerRunConfig(
        markdown_generator=build_md_generator_from_args(args),
        cache_mode=CacheMode.BYPASS,
        stream=False,
    )

def mk_llm_config(args: argparse.Namespace) -> CrawlerRunConfig:
    extraction = build_llm_extraction_strategy(presence_only=bool(getattr(args, "presence_only", False)))
    logger.debug("[run_utils] mk_llm_config_for_markdown_input: extraction preview=%s", {
        "provider": getattr(extraction.llm_config, "provider", None) if hasattr(extraction, "llm_config") else None,
        "base_url": getattr(extraction.llm_config, "base_url", None) if hasattr(extraction, "llm_config") else None,
        "schema_type": getattr(extraction, "schema", None),
        "input_format": getattr(extraction, "input_format", None),
    })
    return CrawlerRunConfig(
        extraction_strategy=extraction,
        cache_mode=CacheMode.BYPASS,
        stream=True,
    )

def make_dispatcher(max_concurrency: int) -> MemoryAdaptiveDispatcher:
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=85.0,
        check_interval=1.0,
        max_session_permit=max_concurrency,
        rate_limiter=RateLimiter(base_delay=(0.5, 1.2), max_delay=20.0, max_retries=2),
        monitor=None,
    )

def aggregate_seed_by_root(items: Iterable[Dict[str, Any]], base_root: str) -> Dict[str, Any]:
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

def classify_failure(err_msg: str | None, status_code: Optional[int], *, treat_timeouts_as_transport: bool) -> str:
    s = (err_msg or "").lower()
    if "download is starting" in s or "download is starting" in (err_msg or ""):
        return "download"
    if ("net::" in s) or ("name_not_resolved" in s) or ("dns" in s) or ("tcp" in s) or ("connection reset" in s):
        return "transport"
    if ("page.goto" in s and "timeout" in s):
        return "transport" if treat_timeouts_as_transport else "soft_timeout"
    if status_code is None and ("timeout" in s):
        return "transport" if treat_timeouts_as_transport else "soft_timeout"
    return "other"

def per_company_slots(n_companies: int, max_slots: int) -> int:
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
    base_timeout: int = 1800,        # 30 min
    per_url_seconds: float = 0.9,    # ~0.9s per URL on average end-to-end
    startup_buffer: int = 300,       # +5 min boot/seed margin
    max_timeout: int = 8 * 3600,     # 8 hours cap
) -> dict[str, int]:
    urls_total = max(0, int(urls_total))
    est = int(startup_buffer + urls_total * per_url_seconds)
    recommended = min(max(base_timeout, est), max_timeout)
    stall = max(1800, recommended // 2)
    return {"company_timeout": recommended, "stall_timeout": stall}

# ---- Helper: detect "llm_extracted_empty" from extracted content ----
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