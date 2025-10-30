from __future__ import annotations

import os
import sys
import logging
from typing import List, Optional, Dict, Any, Iterable, Set, Tuple
from urllib.parse import urlparse

from crawl4ai import (
    AsyncUrlSeeder,
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    RateLimiter,
    SeedingConfig,
)
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

# CrawlerMonitor is optional and varies by platform/version.
try:
    from crawl4ai import CrawlerMonitor  # type: ignore
except Exception:  # pragma: no cover
    CrawlerMonitor = None  # type: ignore

# Filters & helpers (centralized universal externals live here)
from extensions.filtering import (
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
    filter_seeded_urls,
    apply_url_filters,
    make_basic_filter_chain,
    make_keyword_scorer,
    is_universal_external,
)

from extensions.brand_discovery import discover_brand_sites
from components.md_generator import build_default_md_generator
from components.llm_extractor import build_llm_extraction_strategy

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product-first defaults (used when caller doesn't provide a query)
# ---------------------------------------------------------------------------

# A compact, language-agnostic “product/service intent” query.
# Tuned to capture product/service/solution/platform/offerings + commercial CTAs.
DEFAULT_PRODUCT_QUERY = (
    "product products service services solution solutions platform platforms "
    "offering offerings catalog catalogue pricing plans quote buy purchase "
    "request demo contact sales datasheet brochure specification "
    "industr manufactur equipment technology"
)

# CTA / commercial intent hints from <head> metadata or URL
CTA_KEYWORDS = {
    "pricing", "plans", "plan", "quote", "get a quote", "contact sales", "request demo",
    "book a demo", "buy", "purchase", "subscribe", "trial", "free trial",
    "datasheet", "brochure", "specification", "download", "get started", "get in touch"
}

# URL path tokens that typically indicate product-ish sections
PRODUCTISH_URL_TOKENS = {
    "product", "products", "service", "services", "solution", "solutions",
    "platform", "platforms", "capabilities", "offerings",
    "pricing", "catalog", "catalogue", "what-we-do", "technology",
    "industr", "manufactur", "equipment"
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_monitor() -> object | None:
    """Windows/termios-safe monitor construction."""
    if os.name == "nt" or sys.platform.startswith("win"):
        return None
    try:
        import termios  # noqa: F401
    except Exception:
        return None
    if CrawlerMonitor is None:
        return None

    for kwargs in ({"max_visible_rows": 15}, {}):
        try:
            return CrawlerMonitor(**kwargs)  # type: ignore[arg-type]
        except TypeError:
            continue
        except Exception:
            break
    return None


def _make_dispatcher(max_concurrency: int) -> MemoryAdaptiveDispatcher:
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=85.0,
        check_interval=1.0,
        max_session_permit=max_concurrency,
        rate_limiter=RateLimiter(base_delay=(0.5, 1.2), max_delay=20.0, max_retries=2),
        monitor=_build_monitor(),  # ok to pass None
    )


def _stage_run_config(stage: str) -> CrawlerRunConfig:
    """Convenience: stream results; attach MD/LLM only where needed."""
    md_gen = None
    extraction = None

    if stage in ("markdown", "llm"):
        md_gen = build_default_md_generator()
    if stage == "llm":
        extraction = build_llm_extraction_strategy()

    return CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=md_gen,
        extraction_strategy=extraction,
        stream=True,
    )

# ---------------------------------------------------------------------------
# Brand roots
# ---------------------------------------------------------------------------

def _normalize_brand_sites(items: Iterable[Any], *, require_valid_status: bool = True) -> List[str]:
    """
    Accept plain string URLs or dicts returned by discover_brand_sites().
    Prefer 'root' in dicts; fall back to typical url-like keys. Optionally enforce status.
    """
    out: List[str] = []
    seen: Set[str] = set()
    for it in items or []:
        url = None
        status_ok = True
        if isinstance(it, str):
            s = it.strip()
            if s.startswith("http://") or s.startswith("https://"):
                url = s
        elif isinstance(it, dict):
            url = (it.get("root")
                   or it.get("url")
                   or it.get("homepage")
                   or it.get("href")
                   or it.get("link")
                   or "")
            url = str(url).strip()
            if require_valid_status:
                st = str(it.get("status") or "unknown").lower()
                status_ok = (st in {"valid", "unknown"})
        else:
            s = str(it).strip()
            if s.startswith("http://") or s.startswith("https://"):
                url = s

        if not url or not status_ok:
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


async def _collect_roots(
    base_url: str,
    *,
    discover_brands: bool = True,
) -> List[str]:
    roots: Set[str] = set()
    roots.add(base_url)

    if discover_brands and not is_universal_external(base_url):
        try:
            log.info("[url_seeder] Running brand discovery for %s", base_url)
            brand_sites_raw = await discover_brand_sites(base_url)
            brand_sites = _normalize_brand_sites(brand_sites_raw, require_valid_status=True)
            brand_sites = [u for u in brand_sites if not is_universal_external(u)]
            if brand_sites:
                log.info("[url_seeder] Brand discovery: %d candidate site(s)", len(brand_sites))
                for b in brand_sites:
                    log.info("[url_seeder]   brand root: %s", b)
                roots.update(brand_sites)
            else:
                log.info("[url_seeder] Brand discovery: none found.")
        except Exception as e:
            log.exception("[url_seeder] Brand discovery failed for %s: %s", base_url, e)

    return list(roots)

# ---------------------------------------------------------------------------
# Product-signal heuristics (URL + head metadata)
# ---------------------------------------------------------------------------

def _url_has_productish_tokens(u: str) -> bool:
    p = u.lower()
    return any(tok in p for tok in PRODUCTISH_URL_TOKENS)

def _head_meta_hits(head: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Return (#productish_hits, #cta_hits) from <head> data.
    Looks at title, meta.description, keywords, og:* and twitter:* snippets.
    """
    if not head or not isinstance(head, dict):
        return (0, 0)

    fields: List[str] = []
    title = head.get("title")
    if title:
        fields.append(str(title))

    meta = head.get("meta") or {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, str):
                fields.append(v)
    # JSON-LD small scan: look for offer/price/brand/sku-ish cues
    for node in head.get("jsonld", []) or []:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str):
                    fields.append(v)

    blob = " ".join(fields).lower()

    productish_hits = 0
    # very light signals
    for t in ("product", "service", "solution", "platform", "catalog", "catalogue",
              "specification", "datasheet", "features", "capabilities"):
        if t in blob:
            productish_hits += 1

    cta_hits = 0
    for c in CTA_KEYWORDS:
        if c in blob:
            cta_hits += 1

    return (productish_hits, cta_hits)

def _product_signal_score(item: Dict[str, Any]) -> float:
    """0..1 score from URL tokens + head metadata (if available)."""
    url = str(item.get("final_url") or item.get("url") or "").strip()
    head = item.get("head_data") if isinstance(item.get("head_data"), dict) else None

    score = 0.0
    if url and _url_has_productish_tokens(url):
        score += 0.4

    p_hits, c_hits = _head_meta_hits(head)
    # cap tiny contributions to keep conservative
    score += min(p_hits, 3) * 0.1
    score += min(c_hits, 2) * 0.2

    return min(score, 1.0)

# ---------------------------------------------------------------------------
# Seeding with stepwise prefilter, product-signal guard, and patterns/lang filter
# ---------------------------------------------------------------------------

def _aggregate_roots(items: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    per_root: Dict[str, int] = {}
    roots: List[str] = []
    for it in items:
        r = str(it.get("seed_root") or "").strip()
        if not r:
            continue
        per_root[r] = per_root.get(r, 0) + 1
        roots.append(r)
    return per_root, sorted(set(roots))


def _prefilter_stepwise(
    items: List[Dict[str, Any]],
    *,
    live_check: bool,
    min_score: Optional[float],
    drop_universal_externals: bool,
    min_product_signal: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Apply status/external/score/product-signal filters step-by-step so we can count drops.
    Returns (prefiltered_items, drop_counts).
    """
    drops = {
        "dropped_status": 0,
        "dropped_external": 0,
        "dropped_score": 0,
        "dropped_product_signal": 0,
    }

    cur = list(items)

    # 1) Status (only if live_check=True; otherwise don't drop on status)
    if live_check:
        after_status = filter_seeded_urls(
            cur,
            require_status="valid",
            min_score=None,
            drop_universal_externals=False,
        )
        drops["dropped_status"] = len(cur) - len(after_status)
        cur = after_status

    # 2) Universal externals
    if drop_universal_externals:
        after_ext = filter_seeded_urls(
            cur,
            require_status=None,
            min_score=None,
            drop_universal_externals=True,
        )
        drops["dropped_external"] = len(cur) - len(after_ext)
        cur = after_ext

    # 3) Score (BM25 or seeder relevance score)
    if min_score is not None:
        after_score = filter_seeded_urls(
            cur,
            require_status=None,
            min_score=min_score,
            drop_universal_externals=False,
        )
        drops["dropped_score"] = len(cur) - len(after_score)
        cur = after_score

    # 4) Product-signal / CTA guard (light but effective)
    if min_product_signal is not None and min_product_signal > 0:
        kept: List[Dict[str, Any]] = []
        dropped = 0
        for it in cur:
            s = _product_signal_score(it)
            if s >= min_product_signal:
                kept.append(it)
            else:
                dropped += 1
        drops["dropped_product_signal"] = dropped
        cur = kept

    return cur, drops


async def seed_urls(
    base_url: str,
    *,
    source: str = "sitemap",   # "sitemap" | "cc" | "sitemap+cc"
    include: Optional[List[str]] = None,  # glob includes
    exclude: Optional[List[str]] = None,  # glob excludes
    query: Optional[str] = None,          # BM25 topic
    score_threshold: Optional[float] = None,
    pattern: str = "*",
    extract_head: bool = True,
    live_check: bool = False,             # DEFAULT: OFF (user-requested)
    max_urls: int = -1,                   # per-root limit (applied by SeedingConfig)
    company_max_pages: int = -1,          # cap across ALL roots for this company
    concurrency: int = 10,
    hits_per_sec: Optional[int] = 5,
    force: bool = False,
    verbose: bool = False,
    drop_universal_externals: bool = True,
    lang_primary: str = "en",
    lang_accept_en_regions: Optional[set[str]] = None,
    lang_strict_cctld: bool = False,
    discover_brands: bool = True,        # run brand discovery by default
    # Product-first knobs
    auto_product_query: bool = True,
    product_signal_threshold: Optional[float] = 0.3,  # 0..1; set None/0 to disable
    default_score_threshold_if_query: float = 0.25,   # used when we auto-inject query
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover URLs via Crawl4AI AsyncUrlSeeder (with optional live-check), then prefilter
    step-by-step (status → externals → score → product-signal/CTA) and finally apply
    patterns/language filters.

    Returns:
      (filtered_items, stats_dict)
    """
    if is_universal_external(base_url):
        log.debug("[url_seeder] Skip universal external base: %s", base_url)
        return [], {
            "discovered_total": 0,
            "live_check_enabled": bool(live_check),
            "live_checked_total": 0,
            "prefilter_kept": 0,
            "prefilter_dropped_status": 0,
            "prefilter_dropped_external": 0,
            "prefilter_dropped_score": 0,
            "prefilter_dropped_product_signal": 0,
            "filtered_total": 0,
            "filtered_dropped_patterns_lang": 0,
            "seed_roots": [base_url],
            "seed_brand_roots": [],
            "seed_brand_count": 0,
            "per_root_discovered": {},
            "company_cap": company_max_pages if (company_max_pages and company_max_pages > 0) else None,
            "company_cap_applied": False,
            "seeding_source": source,
            "auto_query_used": False,
        }

    # Auto-inject a product-first query if none provided
    auto_query_used = False
    eff_query = query
    eff_score_threshold = score_threshold
    if auto_product_query and not query:
        eff_query = DEFAULT_PRODUCT_QUERY
        auto_query_used = True
        if eff_score_threshold is None:
            eff_score_threshold = default_score_threshold_if_query

    roots = await _collect_roots(base_url, discover_brands=discover_brands)
    log.info(
        "[url_seeder] Seeding roots (%d) [live-check %s]: %s",
        len(roots),
        "ON" if live_check else "OFF",
        ", ".join(roots),
    )

    scoring_method = "bm25" if eff_query else None
    base_cfg = SeedingConfig(
        source=source,
        pattern=pattern,
        extract_head=extract_head,          # pass-through (caller can disable)
        live_check=live_check,              # honor user flag
        max_urls=max_urls,                  # per-root cap
        concurrency=concurrency,
        hits_per_sec=hits_per_sec,
        force=force,
        verbose=verbose,
        query=eff_query,
        scoring_method=scoring_method,
        score_threshold=eff_score_threshold,
        filter_nonsense_urls=True,
    )

    grand_total = 0
    cap_enabled = company_max_pages is not None and company_max_pages > 0
    all_discovered: List[Dict[str, Any]] = []

    async with AsyncUrlSeeder() as seeder:
        for root in roots:
            if cap_enabled and grand_total >= company_max_pages:
                log.info(
                    "[url_seeder] Company cap reached (cap=%d) before root '%s'; stopping.",
                    company_max_pages, root
                )
                break

            # Per-root effective limit honors the remaining company budget if smaller
            effective_max = base_cfg.max_urls
            if cap_enabled:
                remaining = company_max_pages - grand_total
                if effective_max is None or effective_max < 0:
                    effective_max = remaining
                else:
                    effective_max = max(0, min(effective_max, remaining))

            cfg = SeedingConfig(
                source=base_cfg.source,
                pattern=base_cfg.pattern,
                extract_head=base_cfg.extract_head,
                live_check=base_cfg.live_check,
                max_urls=effective_max,
                concurrency=base_cfg.concurrency,
                hits_per_sec=base_cfg.hits_per_sec,
                force=base_cfg.force,
                verbose=base_cfg.verbose,
                query=base_cfg.query,
                scoring_method=base_cfg.scoring_method,
                score_threshold=base_cfg.score_threshold,
                filter_nonsense_urls=base_cfg.filter_nonsense_urls,
            )

            log.info(
                "[url_seeder] Seeding '%s' via %s (live-check %s, per-root max=%s, company cap=%s, used=%d, auto_query=%s)",
                root, source,
                "ON" if live_check else "OFF",
                str(effective_max if effective_max is not None else -1),
                str(company_max_pages if cap_enabled else "disabled"),
                grand_total,
                auto_query_used,
            )

            try:
                raw = await seeder.urls(root, cfg)
                for r in raw:
                    r.setdefault("seed_root", root)
                all_discovered.extend(raw)
                grand_total += len(raw)
                log.info("[url_seeder]   -> %d URL(s) discovered; company_total=%d",
                         len(raw), grand_total)
            except Exception as e:
                log.exception("[url_seeder] Seeding failed for %s: %s", root, e)

            if cap_enabled and grand_total >= company_max_pages:
                log.info(
                    "[url_seeder] Company cap reached after root '%s' (cap=%d).",
                    root, company_max_pages
                )
                break

    # Nothing found
    per_root, unique_roots = _aggregate_roots(all_discovered)
    brand_roots = [r for r in unique_roots if r != base_url]

    if not all_discovered:
        log.info("[url_seeder] No URLs discovered from any root.")
        stats = {
            "discovered_total": 0,
            "live_check_enabled": bool(live_check),
            "live_checked_total": 0 if not live_check else 0,
            "prefilter_kept": 0,
            "prefilter_dropped_status": 0,
            "prefilter_dropped_external": 0,
            "prefilter_dropped_score": 0,
            "prefilter_dropped_product_signal": 0,
            "filtered_total": 0,
            "filtered_dropped_patterns_lang": 0,
            "seed_roots": unique_roots or [base_url],
            "seed_brand_roots": brand_roots,
            "seed_brand_count": len(brand_roots),
            "per_root_discovered": per_root,
            "company_cap": company_max_pages if (company_max_pages and company_max_pages > 0) else None,
            "company_cap_applied": False,
            "seeding_source": source,
            "auto_query_used": auto_query_used,
        }
        return [], stats

    # Stepwise prefilter (status/external/score/product-signal) so we can count drops
    prefiltered, drop_counts = _prefilter_stepwise(
        all_discovered,
        live_check=live_check,
        min_score=eff_score_threshold,
        drop_universal_externals=drop_universal_externals,
        min_product_signal=product_signal_threshold if extract_head else None,  # head-based guard needs head_data
    )

    # Patterns/language filter (black-box; we only know net drop)
    filtered = apply_url_filters(
        prefiltered,
        include=include or DEFAULT_INCLUDE_PATTERNS,
        exclude=exclude or DEFAULT_EXCLUDE_PATTERNS,
        drop_universal_externals=drop_universal_externals,
        lang_primary=lang_primary,
        lang_accept_en_regions=lang_accept_en_regions,
        lang_strict_cctld=lang_strict_cctld,
        include_overrides_language=False,   # includes do NOT override language drops
        sort_by_priority=True,              # ensure stronger negative/host filters win
    )
    patterns_lang_dropped = len(prefiltered) - len(filtered)

    # Company cap after filtering (final clamp)
    cap_applied = False
    if company_max_pages is not None and company_max_pages > 0 and len(filtered) > company_max_pages:
        filtered = filtered[:company_max_pages]
        cap_applied = True
        log.info("[url_seeder] Company cap applied after filtering: final=%d", len(filtered))

    stats = {
        "discovered_total": len(all_discovered),
        "live_check_enabled": bool(live_check),
        "live_checked_total": len(all_discovered) if live_check else 0,
        "prefilter_kept": len(prefiltered),
        "prefilter_dropped_status": int(drop_counts.get("dropped_status", 0)),
        "prefilter_dropped_external": int(drop_counts.get("dropped_external", 0)),
        "prefilter_dropped_score": int(drop_counts.get("dropped_score", 0)),
        "prefilter_dropped_product_signal": int(drop_counts.get("dropped_product_signal", 0)),
        "filtered_total": len(filtered),
        "filtered_dropped_patterns_lang": patterns_lang_dropped,
        "seed_roots": unique_roots,
        "seed_brand_roots": brand_roots,
        "seed_brand_count": len(brand_roots),
        "per_root_discovered": per_root,
        "company_cap": company_max_pages if (company_max_pages and company_max_pages > 0) else None,
        "company_cap_applied": cap_applied,
        "seeding_source": source,
        "auto_query_used": auto_query_used,
    }

    return filtered, stats

# ---------------------------------------------------------------------------
# Crawl seeded URLs (or deep-crawl fallback)
# ---------------------------------------------------------------------------

async def crawl_seeded(
    seeded_urls: Iterable[Dict[str, Any]],
    *,
    stage: str,
    max_concurrency: int = 10,
) -> List[Any]:
    urls = [u.get("final_url") or u.get("url") for u in seeded_urls if (u.get("final_url") or u.get("url"))]
    if not urls:
        return []

    run_cfg = _stage_run_config(stage)
    dispatcher = _make_dispatcher(max_concurrency)

    results = []
    async with AsyncWebCrawler() as crawler:
        async for r in await crawler.arun_many(urls=urls, config=run_cfg, dispatcher=dispatcher):
            results.append(r)
    return results


async def deep_crawl_fallback(
    base_url: str,
    *,
    stage: str,
    max_concurrency: int = 6,
    keywords: Optional[List[str]] = None,
    max_depth: int = 2,
    max_pages: int = 200,
    company_max_pages: int = -1,  # honor company cap in fallback too
) -> List[Any]:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()

    if is_universal_external(base_url):
        log.debug("[url_seeder] Fallback seed is universal external, skip: %s", base_url)
        return []

    # Clamp fallback pages to company cap (if provided)
    effective_max_pages = max_pages
    if company_max_pages is not None and company_max_pages > 0:
        effective_max_pages = min(max_pages, company_max_pages)

    scorer = make_keyword_scorer(keywords or [], weight=0.7)
    filter_chain = make_basic_filter_chain(
        allowed_domains=[host],
        patterns=None,
        content_types=["text/html"],
    )

    run_cfg = _stage_run_config(stage)
    run_cfg.deep_crawl_strategy = BestFirstCrawlingStrategy(  # type: ignore[attr-defined]
        max_depth=max_depth,
        include_external=False,
        url_scorer=scorer,
        filter_chain=filter_chain,
        max_pages=effective_max_pages,
    )

    dispatcher = _make_dispatcher(max_concurrency)

    results = []
    async with AsyncWebCrawler() as crawler:
        async for r in await crawler.arun(base_url, config=run_cfg, dispatcher=dispatcher):
            results.append(r)
    return results

# ---------------------------------------------------------------------------
# One-shot pipeline helper
# ---------------------------------------------------------------------------

async def discover_and_crawl(
    base_url: str,
    *,
    stage: str,
    max_concurrency: int = 10,
    seeding_source: str = "sitemap",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    query: Optional[str] = None,
    score_threshold: Optional[float] = None,
    pattern: str = "*",
    extract_head: bool = False,
    live_check: bool = False,             # default OFF
    max_urls: int = -1,                   # per-root cap
    company_max_pages: int = -1,          # per-company cap (across all roots)
    concurrency: int = 10,
    hits_per_sec: Optional[int] = 5,
    force: bool = False,
    verbose: bool = False,
    drop_universal_externals: bool = True,
    lang_primary: str = "en",
    lang_accept_en_regions: Optional[set[str]] = None,
    lang_strict_cctld: bool = False,
    fallback_keywords: Optional[List[str]] = None,
    discover_brands: bool = True,
    # product-first knobs (passed through)
    auto_product_query: bool = True,
    product_signal_threshold: Optional[float] = 0.3,
    default_score_threshold_if_query: float = 0.25,
) -> List[Any]:
    filtered, _stats = await seed_urls(
        base_url,
        source=seeding_source,
        include=include,
        exclude=exclude,
        query=query,
        score_threshold=score_threshold,
        pattern=pattern,
        extract_head=extract_head,
        live_check=live_check,
        max_urls=max_urls,                    # per-root
        company_max_pages=company_max_pages,  # per-company cap
        concurrency=concurrency,
        hits_per_sec=hits_per_sec,
        force=force,
        verbose=verbose,
        drop_universal_externals=drop_universal_externals,
        lang_primary=lang_primary,
        lang_accept_en_regions=lang_accept_en_regions,
        lang_strict_cctld=lang_strict_cctld,
        discover_brands=discover_brands,
        auto_product_query=auto_product_query,
        product_signal_threshold=product_signal_threshold,
        default_score_threshold_if_query=default_score_threshold_if_query,
    )

    if not filtered:
        log.info("[url_seeder] No seeds found → deep-crawl fallback")
        return await deep_crawl_fallback(
            base_url,
            stage=stage,
            max_concurrency=max_concurrency,
            keywords=fallback_keywords or [],
            company_max_pages=company_max_pages,
        )

    log.info("[url_seeder] Crawling %d seeded URLs", len(filtered))
    return await crawl_seeded(filtered, stage=stage, max_concurrency=max_concurrency)
