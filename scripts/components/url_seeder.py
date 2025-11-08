from __future__ import annotations

import os
import sys
import asyncio
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

try:
    from crawl4ai import CrawlerMonitor  # type: ignore
except Exception:  # pragma: no cover
    CrawlerMonitor = None  # type: ignore

from config import language_settings as langcfg

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
from components.md_generator import build_default_md_generator, evaluate_markdown
from components.llm_extractor import build_llm_extraction_strategy

from extensions.dual_bm25 import DualBM25Combiner, DualBM25Config, build_default_negative_query
from extensions.page_interaction import (
    PageInteractionPolicy,
    make_interaction_plan,
)

class _NullMetrics:
        def incr(self, name: str, value: float = 1.0, **labels: Any) -> None: ...
        def observe(self, name: str, value: float, **labels: Any) -> None: ...
        def set(self, name: str, value: float, **labels: Any) -> None: ...

metrics = _NullMetrics()  # type: ignore

log = logging.getLogger(__name__)

# Use language_settings for the default product query (language-aware)
DEFAULT_PRODUCT_QUERY = langcfg.default_product_bm25_query()

# CTA keywords: attempt to read from language settings, otherwise fallback to English set
CTA_KEYWORDS = set(langcfg.get("CTA_KEYWORDS"))

# PRODUCT-ish URL tokens — get from language settings (SMART include tokens)
PRODUCTISH_URL_TOKENS = set(langcfg.get("SMART_INCLUDE_TOKENS"))

_DEBUG_SAMPLE_TOP = 10
_DEBUG_SAMPLE_BOTTOM = 3

def _build_monitor() -> object | None:
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
        monitor=_build_monitor(),
    )

def _stage_run_config(stage: str) -> CrawlerRunConfig:
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

def _with_scheme_variants(url_or_host: str) -> List[str]:
    s = (url_or_host or "").strip()
    if not s:
        return []
    lower = s.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return [s]
    while s.startswith("/"):
        s = s[1:]
    return [f"https://{s}", f"http://{s}"]

def _same_netloc(a: str, b: str) -> bool:
    return (urlparse(a).netloc or "").lower() == (urlparse(b).netloc or "").lower()

def _normalize_brand_sites(items: Iterable[Any], *, require_valid_status: bool = True) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for it in items or []:
        raw = None
        status_ok = True
        if isinstance(it, str):
            raw = it.strip()
        elif isinstance(it, dict):
            raw = (it.get("root") or it.get("url") or it.get("homepage") or it.get("href") or it.get("link") or "")
            raw = str(raw).strip()
            if require_valid_status:
                st = str(it.get("status") or "unknown").lower()
                status_ok = (st in {"valid","unknown"})
        else:
            raw = str(it).strip()
        if not raw or not status_ok:
            continue
        for v in _with_scheme_variants(raw):
            if v and v not in seen:
                seen.add(v); out.append(v)
    return out

async def _collect_roots(base_url: str, *, discover_brands: bool = True) -> List[str]:
    roots: List[str] = []
    seen: Set[str] = set()
    for v in _with_scheme_variants(base_url):
        if v and v not in seen:
            seen.add(v); roots.append(v)
    preferred = roots[0] if roots else None
    if discover_brands and preferred and not is_universal_external(preferred):
        try:
            log.info("[url_seeder] Brand discovery for %s", preferred)
            brand_sites_raw = await discover_brand_sites(preferred)
            brand_sites = _normalize_brand_sites(brand_sites_raw, require_valid_status=True)
            brand_sites = [u for u in brand_sites if not is_universal_external(u)]
            for b in brand_sites:
                if b not in seen:
                    seen.add(b); roots.append(b)
            if len(roots) > 1:
                log.info("[url_seeder]   %d brand root(s)", len(roots) - 1)
            if log.isEnabledFor(logging.DEBUG):
                log.debug("[url_seeder]   Brand roots discovered: %s", ", ".join(roots[1:]))
        except Exception as e:
            log.exception("[url_seeder] Brand discovery failed: %s", e)
    return roots

def _url_has_productish_tokens(u: str) -> bool:
    p = u.lower()
    return any(tok in p for tok in PRODUCTISH_URL_TOKENS)

def _head_meta_hits(head: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if not head or not isinstance(head, dict):
        return (0, 0)
    fields: List[str] = []
    title = head.get("title");  fields.append(str(title)) if title else None
    meta = head.get("meta") or {}
    if isinstance(meta, dict):
        for v in meta.values():
            if isinstance(v, str):
                fields.append(v)
    for node in head.get("jsonld", []) or []:
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, str):
                    fields.append(v)
    blob = " ".join(fields).lower()
    prod = sum(k in blob for k in (
        "product","service","solution","platform","catalog","catalogue","specification","datasheet",
        "features","capabilities"
    ))
    cta = sum(c in blob for c in CTA_KEYWORDS)
    return (prod, cta)

def _product_signal_score(item: Dict[str, Any]) -> float:
    url = str(item.get("final_url") or item.get("url") or "").strip()
    head = item.get("head_data") if isinstance(item.get("head_data"), dict) else None
    score = 0.0
    if url and _url_has_productish_tokens(url):
        score += 0.4
    p_hits, c_hits = _head_meta_hits(head)
    score += min(p_hits, 3) * 0.1
    score += min(c_hits, 2) * 0.2
    return min(score, 1.0)

def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    out: List[str] = []
    cur = []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur)); cur = []
    if cur: out.append("".join(cur))
    return out

def _bm25_lite(text: str, query: str) -> float:
    if not text or not query:
        return 0.0
    tt = _tok(text)
    if not tt:
        return 0.0
    q = [t for t in _tok(query) if t]
    if not q:
        return 0.0
    tf: Dict[str, int] = {}
    for t in tt:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for t in q:
        f = tf.get(t, 0)
        if f:
            score += (f ** 0.5)
    return score

def _doc_text_for_item(it: Dict[str, Any]) -> str:
    parts: List[str] = []
    u = it.get("final_url") or it.get("url") or ""
    parts.append(str(u))
    h = it.get("head_data") or {}
    t = h.get("title")
    if t: parts.append(str(t))
    meta = h.get("meta") or {}
    if isinstance(meta, dict):
        for v in meta.values():
            if isinstance(v, str): parts.append(v)
    for node in h.get("jsonld", []) or []:
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, str): parts.append(v)
    return " ".join(parts)

def _compute_dual_scores(items: List[Dict[str, Any]], pos_query: Optional[str], alpha: float) -> None:
    comb = DualBM25Combiner(_bm25_lite, DualBM25Config(alpha=alpha))
    neg_q = build_default_negative_query()
    for it in items:
        doc = _doc_text_for_item(it)
        it["dual_bm25"] = comb.score(doc, pos_query=pos_query, neg_query=neg_q)

def _aggregate_roots(items: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    per_root: Dict[str, int] = {}
    roots: List[str] = []
    for it in items:
        r = str(it.get("seed_root") or "").strip()
        if r:
            per_root[r] = per_root.get(r, 0) + 1
            roots.append(r)
    return per_root, sorted(set(roots))

def _dedupe_seeded(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = str(it.get("final_url") or it.get("url") or "").strip()
        if key and key not in seen:
            seen.add(key); out.append(it)
    return out

def _prefilter_stepwise(
    items: List[Dict[str, Any]],
    *,
    live_check: bool,
    min_score: Optional[float],
    score_field: str,
    drop_universal_externals: bool,
    min_product_signal: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    drops = {"dropped_status": 0, "dropped_external": 0, "dropped_score": 0, "dropped_product_signal": 0}
    cur = list(items)

    if live_check:
        before = len(cur)
        after = filter_seeded_urls(cur, require_status="valid", min_score=None, drop_universal_externals=False)
        drops["dropped_status"] = before - len(after)
        cur = after
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[url_seeder.prefilter] live_check: %d -> %d (dropped=%d)",
                      before, len(cur), drops["dropped_status"])

    before = len(cur)
    if drop_universal_externals:
        after = filter_seeded_urls(cur, require_status=None, min_score=None, drop_universal_externals=True)
        drops["dropped_external"] = before - len(after)
        cur = after
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[url_seeder.prefilter] externals: %d -> %d (dropped=%d)",
                      before, len(cur), drops["dropped_external"])

    if min_score is not None:
        before = len(cur)
        kept: List[Dict[str, Any]] = []
        for it in cur:
            s = float(it.get("dual_bm25") if score_field == "dual_bm25" else it.get("score") or 0.0)
            if s >= float(min_score):
                kept.append(it)
        drops["dropped_score"] = before - len(kept)
        cur = kept
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[url_seeder.prefilter] score('%s'>=%.3f): %d -> %d (dropped=%d)",
                      score_field, float(min_score), before, len(cur), drops["dropped_score"])

    if min_product_signal is not None and min_product_signal > 0:
        before = len(cur)
        kept: List[Dict[str, Any]] = []
        dropped = 0
        for it in cur:
            if _product_signal_score(it) >= min_product_signal:
                kept.append(it)
            else:
                dropped += 1
        drops["dropped_product_signal"] = dropped
        cur = kept
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[url_seeder.prefilter] product_signal(>=%.2f): %d -> %d (dropped=%d)",
                      float(min_product_signal), before, len(cur), dropped)

    return cur, drops

async def seed_urls(
    base_url: str,
    *,
    source: str = "sitemap",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    query: Optional[str] = None,
    score_threshold: Optional[float] = None,
    pattern: str = "*",
    extract_head: bool = True,
    live_check: bool = False,
    max_urls: int = -1,
    company_max_pages: int = -1,
    concurrency: int = 10,
    hits_per_sec: Optional[int] = 5,
    force: bool = False,
    verbose: bool = False,
    drop_universal_externals: bool = True,
    lang_primary: str = "en",
    lang_accept_en_regions: Optional[set[str]] = None,
    lang_strict_cctld: bool = False,
    discover_brands: bool = True,
    auto_product_query: bool = True,
    product_signal_threshold: Optional[float] = 0.3,
    default_score_threshold_if_query: float = 0.25,
    use_dual_bm25: bool = True,
    dual_alpha: float = 0.65,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "[url_seeder] begin base=%s src=%s query=%s thresh=%s auto_q=%s dual=%s alpha=%.2f extract_head=%s live=%s include=%d exclude=%d "
            "max_urls=%d company_cap=%d lang_primary=%s en_regions=%s strict_cctld=%s",
            base_url, source, (query[:64] + "…") if query and len(query) > 64 else query,
            score_threshold, auto_product_query, use_dual_bm25, float(dual_alpha),
            bool(extract_head), bool(live_check), len(include or []), len(exclude or []),
            int(max_urls), int(company_max_pages), lang_primary, sorted(list(lang_accept_en_regions or set())),
            bool(lang_strict_cctld),
        )

    base_variants = _with_scheme_variants(base_url)
    if not base_variants:
        return [], {
            "discovered_total": 0,
            "live_check_enabled": bool(live_check),
            "filtered_total": 0,
            "seed_roots": [],
            "seed_brand_roots": [],
            "seed_brand_count": 0,
            "per_root_discovered": {},
            "company_cap": company_max_pages if (company_max_pages and company_max_pages > 0) else None,
            "company_cap_applied": False,
            "seeding_source": source,
            "auto_query_used": False,
        }

    preferred = base_variants[0]
    if is_universal_external(preferred):
        return [], {
            "discovered_total": 0,
            "live_check_enabled": bool(live_check),
            "filtered_total": 0,
            "seed_roots": base_variants,
            "seed_brand_roots": [],
            "seed_brand_count": 0,
            "per_root_discovered": {},
            "company_cap": company_max_pages if (company_max_pages and company_max_pages > 0) else None,
            "company_cap_applied": False,
            "seeding_source": source,
            "auto_query_used": False,
        }

    auto_query_used = False
    eff_query = query
    eff_threshold = score_threshold
    if auto_product_query and not query:
        eff_query = DEFAULT_PRODUCT_QUERY
        auto_query_used = True
        if eff_threshold is None:
            eff_threshold = default_score_threshold_if_query
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[url_seeder] effective query: %s | eff_threshold=%s | auto_query_used=%s",
                  (eff_query[:120] + "…") if eff_query and len(eff_query) > 120 else eff_query,
                  eff_threshold, auto_query_used)

    roots = await _collect_roots(preferred, discover_brands=discover_brands)
    for v in base_variants:
        if v not in roots:
            roots.insert(0, v)

    log.info("[url_seeder] Roots(%d) [live-check %s]: %s",
             len(roots), "ON" if live_check else "OFF", ", ".join(roots))
    
    scoring_method = "bm25" if eff_query else None
    base_cfg = SeedingConfig(
        source=source,
        pattern=pattern,
        extract_head=extract_head,
        live_check=live_check,
        max_urls=max_urls,
        concurrency=concurrency,
        hits_per_sec=hits_per_sec,
        force=force,
        verbose=verbose,
        query=eff_query,
        scoring_method=scoring_method,
        score_threshold=None if use_dual_bm25 else eff_threshold,
        filter_nonsense_urls=True,
    )

    if log.isEnabledFor(logging.DEBUG):
        log.debug("[url_seeder] base_cfg=%s", base_cfg)

    grand_total = 0
    cap_enabled = company_max_pages is not None and company_max_pages > 0
    all_discovered: List[Dict[str, Any]] = []

    async with AsyncUrlSeeder() as seeder:
        for root in roots:
            if cap_enabled and grand_total >= company_max_pages:
                log.info("[url_seeder] Cap reached (cap=%d); stop before %s", company_max_pages, root)
                break

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

            if log.isEnabledFor(logging.DEBUG):
                log.debug("[url_seeder] seeding %s with cfg=%s", root, cfg)

            try:
                raw = await seeder.urls(root, cfg)
                for r in raw:
                    r.setdefault("seed_root", root)
                all_discovered.extend(raw)
                grand_total += len(raw)
                log.info("[url_seeder] %s -> %d URL(s); total=%d", root, len(raw), grand_total)

                if log.isEnabledFor(logging.DEBUG) and raw:
                    sample = raw[:_DEBUG_SAMPLE_TOP]
                    log.debug("[url_seeder]   sample(%d) first=%s", len(sample),
                              [ (str(x.get("status") or "?"), round(float(x.get("score") or 0.0), 4), x.get("final_url") or x.get("url"))
                                for x in sample ])
            except Exception as e:
                log.exception("[url_seeder] Seeding failed for %s: %s", root, e)

            if cap_enabled and grand_total >= company_max_pages:
                log.info("[url_seeder] Cap reached after %s (cap=%d)", root, company_max_pages)
                break

    all_discovered = _dedupe_seeded(all_discovered)
    per_root, unique_roots = _aggregate_roots(all_discovered)
    brand_roots = [r for r in unique_roots if not _same_netloc(r, preferred)]

    if log.isEnabledFor(logging.DEBUG):
        log.debug("[url_seeder] discovered_total=%d (unique after dedupe)", len(all_discovered))
        if all_discovered:
            log.debug("[url_seeder] discovered sample=%s",
                      [(x.get("final_url") or x.get("url")) for x in all_discovered[:_DEBUG_SAMPLE_TOP]])

    if not all_discovered:
        return [], {
            "discovered_total": 0,
            "live_check_enabled": bool(live_check),
            "prefilter_kept": 0,
            "prefilter_dropped_status": 0,
            "prefilter_dropped_external": 0,
            "prefilter_dropped_score": 0,
            "prefilter_dropped_product_signal": 0,
            "filtered_total": 0,
            "filtered_dropped_patterns_lang": 0,
            "seed_roots": unique_roots or base_variants,
            "seed_brand_roots": brand_roots,
            "seed_brand_count": len(brand_roots),
            "per_root_discovered": per_root,
            "company_cap": company_max_pages if (company_max_pages and company_max_pages > 0) else None,
            "company_cap_applied": False,
            "seeding_source": source,
            "auto_query_used": auto_query_used,
            "dual_used": bool(use_dual_bm25),
        }

    if use_dual_bm25:
        _compute_dual_scores(all_discovered, pos_query=eff_query, alpha=float(dual_alpha))
        if log.isEnabledFor(logging.DEBUG):
            scores = [float(it.get("dual_bm25") or 0.0) for it in all_discovered]
            if scores:
                mn = min(scores); mx = max(scores); avg = sum(scores) / len(scores)
                log.debug("[url_seeder.dual] alpha=%.2f n=%d min=%.4f avg=%.4f max=%.4f",
                          float(dual_alpha), len(scores), mn, avg, mx)
                try:
                    ranked = sorted(all_discovered, key=lambda x: float(x.get("dual_bm25") or 0.0), reverse=True)
                    top = [ (round(float(x.get("dual_bm25") or 0.0), 4), x.get("final_url") or x.get("url")) for x in ranked[:_DEBUG_SAMPLE_TOP] ]
                    bot = [ (round(float(x.get("dual_bm25") or 0.0), 4), x.get("final_url") or x.get("url")) for x in ranked[-_DEBUG_SAMPLE_BOTTOM:] ]
                    log.debug("[url_seeder.dual] top=%s", top)
                    log.debug("[url_seeder.dual] bottom=%s", bot)
                except Exception:
                    pass

    prefiltered, drop_counts = _prefilter_stepwise(
        all_discovered,
        live_check=live_check,
        min_score=eff_threshold,
        score_field="dual_bm25" if use_dual_bm25 else "seeder_score",
        drop_universal_externals=drop_universal_externals,
        min_product_signal=(product_signal_threshold if extract_head else None),
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[url_seeder.prefilter] kept=%d drops=%s", len(prefiltered), drop_counts)

    filtered = apply_url_filters(
        prefiltered,
        include=include or DEFAULT_INCLUDE_PATTERNS,
        exclude=exclude or DEFAULT_EXCLUDE_PATTERNS,
        drop_universal_externals=drop_universal_externals,
        lang_primary=lang_primary,
        lang_accept_en_regions=lang_accept_en_regions,
        lang_strict_cctld=lang_strict_cctld,
        include_overrides_language=False,
        sort_by_priority=True,
    )
    patterns_lang_dropped = len(prefiltered) - len(filtered)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[url_seeder.filters] patterns/lang dropped=%d (from %d → %d)",
                  patterns_lang_dropped, len(prefiltered), len(filtered))
        if filtered:
            log.debug("[url_seeder.filters] kept sample=%s",
                      [(x.get("final_url") or x.get("url")) for x in filtered[:_DEBUG_SAMPLE_TOP]])

    cap_applied = False
    if company_max_pages is not None and company_max_pages > 0 and len(filtered) > company_max_pages:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[url_seeder.cap] applying company cap %d on %d items", company_max_pages, len(filtered))
        filtered = filtered[:company_max_pages]; cap_applied = True

    stats = {
        "discovered_total": len(all_discovered),
        "live_check_enabled": bool(live_check),
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
        "dual_used": bool(use_dual_bm25),
        "dual_alpha": float(dual_alpha) if use_dual_bm25 else None,
    }
    return filtered, stats

# ---------------------------------------------------------------------------
# Crawl seeded (Markdown stage uses interaction + AUTO JS-only retry on retry/suppress)
# ---------------------------------------------------------------------------

async def crawl_seeded(
    seeded_urls: Iterable[Dict[str, Any]],
    *,
    stage: str,
    max_concurrency: int = 10,
) -> List[Any]:
    urls = [str(u.get("final_url") or u.get("url") or "").strip() for u in seeded_urls]
    urls = [u for u in urls if u]
    if not urls:
        return []

    # For LLM stage we can keep old batch behavior
    if stage != "markdown":
        run_cfg = _stage_run_config(stage)
        dispatcher = _make_dispatcher(max_concurrency)
        results: List[Any] = []
        async with AsyncWebCrawler() as crawler:
            try:
                async for r in await crawler.arun_many(urls=urls, config=run_cfg, dispatcher=dispatcher):
                    results.append(r)
            except Exception as e:
                log.exception("[url_seeder] Crawler stream error: %s", e)
        return results

    # Markdown stage uses interactive plan + quality gate
    policy = PageInteractionPolicy(
        enable_cookie_playbook=True,
        max_in_session_retries=1,
        virtual_scroll=False,  # keep conservative by default; can be enabled by caller if desired
    )
    md_gen = build_default_md_generator()

    sem = asyncio.Semaphore(max_concurrency)
    out_results: List[Any] = []

    async with AsyncWebCrawler() as crawler:
        async def _one(url: str):
            async with sem:
                try:
                    init_cfg, retry_cfgs = make_interaction_plan(url, policy)
                    # attach md generator
                    init_cfg.markdown_generator = md_gen
                    for rc in retry_cfgs:
                        rc.markdown_generator = md_gen

                    # initial
                    r = await crawler.arun(url=url, config=init_cfg)
                    if not getattr(r, "success", False):
                        out_results.append(r)
                        return

                    md_obj = getattr(r, "markdown", None)
                    text = (getattr(md_obj, "fit_markdown", None) or "").strip()
                    action, reason, _stats = evaluate_markdown(
                        text, url=url, allow_retry=True, generator_ignores_links=True
                    )

                    # If gate suggests a retry OR suppressed due to cookie-dominant content,
                    # run a JS-only retry pass (if available) to try to remove cookie overlays.
                    if action in ("retry", "suppress") and retry_cfgs:
                        try:
                            rr = await crawler.arun(url=url, config=retry_cfgs[0])
                            out_results.append(rr)
                            metrics.incr("seeder.interaction.retry_used", reason=reason)
                            return
                        except Exception as e:
                            log.exception("[url_seeder] retry pass failed for %s: %s", url, e)
                            # fall-through to append original result
                    out_results.append(r)
                except Exception as e:
                    log.warning("[url_seeder] interaction error for %s: %s", url, e)

        tasks = [asyncio.create_task(_one(u)) for u in urls]
        await asyncio.gather(*tasks, return_exceptions=False)

    return out_results

# ---------------------------------------------------------------------------
# Deep crawl fallback + discover_and_crawl
# ---------------------------------------------------------------------------

async def deep_crawl_fallback(
    base_url: str,
    *,
    stage: str,
    max_concurrency: int = 6,
    keywords: Optional[List[str]] = None,
    max_depth: int = 2,
    max_pages: int = 200,
    company_max_pages: int = -1,
) -> List[Any]:
    variants = _with_scheme_variants(base_url)
    start_url = variants[0] if variants else base_url
    if is_universal_external(start_url):
        return []
    effective_max_pages = min(max_pages, company_max_pages) if (company_max_pages and company_max_pages > 0) else max_pages
    host = (urlparse(start_url).hostname or "").lower()
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
    results: List[Any] = []
    async with AsyncWebCrawler() as crawler:
        async for r in await crawler.arun(start_url, config=run_cfg, dispatcher=dispatcher):
            results.append(r)
    return results

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
    live_check: bool = False,
    max_urls: int = -1,
    company_max_pages: int = -1,
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
    auto_product_query: bool = True,
    product_signal_threshold: Optional[float] = 0.3,
    default_score_threshold_if_query: float = 0.25,
    use_dual_bm25: bool = True,
    dual_alpha: float = 0.65,
) -> List[Any]:
    filtered, _ = await seed_urls(
        base_url,
        source=seeding_source,
        include=include,
        exclude=exclude,
        query=query,
        score_threshold=score_threshold,
        pattern=pattern,
        extract_head=extract_head,
        live_check=live_check,
        max_urls=max_urls,
        company_max_pages=company_max_pages,
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
        use_dual_bm25=use_dual_bm25,
        dual_alpha=dual_alpha,
    )

    if not filtered:
        log.info("[url_seeder] No seeds → deep-crawl fallback")
        return await deep_crawl_fallback(
            base_url,
            stage=stage,
            max_concurrency=max_concurrency,
            keywords=fallback_keywords or [],
            company_max_pages=company_max_pages,
        )

    log.info("[url_seeder] Crawling %d seeded URLs", len(filtered))
    return await crawl_seeded(filtered, stage=stage, max_concurrency=max_concurrency)