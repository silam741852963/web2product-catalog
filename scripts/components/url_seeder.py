from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Set, Tuple
from urllib.parse import urlparse, urlunparse, urljoin

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

from configs import language_settings as lang_cfg
from configs.browser_settings import browser_cfg
from configs.crawler_settings import crawler_base_cfg

from extensions.dataset_externals import DatasetExternals
from extensions.filtering import (
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
    apply_url_filters,
    same_registrable_domain,
    set_dataset_externals,
    set_universal_external_runtime_exclusions,
    should_block_external_for_dataset,
)

from extensions.brand_discovery import discover_brand_sites
from extensions.dual_bm25 import DualBM25Combiner, DualBM25Config, build_default_negative_query
from extensions.sitemap_hunter import discover_from_sitemaps
from extensions.url_index import discover_and_write_url_index
from extensions.run_utils import upsert_url_index

log = logging.getLogger(__name__)

_EXTERNALS_INSTALLED_ONCE = False

# =====================================================================
# ExternalsUniverse: central loader/registrar for dataset externals
# =====================================================================

@dataclass
class ExternalsUniverse:
    """Holds dataset externals and installs runtime policies in filtering.py."""
    hosts: Set[str]
    registrable_domains: Set[str]
    allowed_brand_hosts: Set[str]

    @classmethod
    def from_sources(
        cls,
        *,
        source: Optional[Path] = None,
        source_dir: Optional[Path] = None,
        pattern: str = "*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav",
        limit: Optional[int] = None,
        allowed_brand_hosts: Optional[Iterable[str]] = None,
    ) -> "ExternalsUniverse":
        ds = DatasetExternals.from_sources(source=source, source_dir=source_dir, pattern=pattern, limit=limit)
        return cls(
            hosts=set(ds.hosts or set()),
            registrable_domains=set(ds.registrable_domains or set()),
            allowed_brand_hosts=set(allowed_brand_hosts or ()),
        )

    @classmethod
    def from_companies(
        cls,
        companies: Iterable,
        allowed_brand_hosts: Optional[Iterable[str]] = None,
    ) -> "ExternalsUniverse":
        ds = DatasetExternals.from_companies(companies)
        return cls(
            hosts=set(ds.hosts or set()),
            registrable_domains=set(ds.registrable_domains or set()),
            allowed_brand_hosts=set(allowed_brand_hosts or ()),
        )

    def install(self) -> None:
            """
            Register dataset externals in filtering module and demote any overlaps
            from UNIVERSAL_EXTERNALS at runtime so they aren't auto-dropped.
            """
            global _EXTERNALS_INSTALLED_ONCE
            
            if _EXTERNALS_INSTALLED_ONCE:
                log.debug("[externals] Dataset externals already installed; skipping.")
                return

            # Keep the behavior the same: always (re)register the sets
            set_dataset_externals(self.registrable_domains or self.hosts)
            set_universal_external_runtime_exclusions(self.registrable_domains or self.hosts)

            # But only log at INFO the first time
            if not _EXTERNALS_INSTALLED_ONCE:
                log.info(
                    "[externals] Installed dataset externals: hosts=%d registrable=%d, allowed_brand_hosts=%d",
                    len(self.hosts), len(self.registrable_domains), len(self.allowed_brand_hosts),
                )
                _EXTERNALS_INSTALLED_ONCE = True
            else:
                # Optional: keep a low-noise debug trace if you want
                log.debug(
                    "[externals] Dataset externals already installed; "
                    "skipping repeated INFO log (hosts=%d registrable=%d)",
                    len(self.hosts), len(self.registrable_domains),
                )


# -------------------------
# Helpers (HTTPS-only)
# -------------------------
def _with_https_only(url_or_host: str) -> List[str]:
    s = (url_or_host or "").strip()
    if not s:
        return []
    s = s.lstrip("/")
    if s.startswith("https://"):
        return [s]
    if s.startswith("http://"):
        return [f"https://{s[len('http://'):] }"]
    return [f"https://{s}"]


def _domain_variants(host: str) -> List[str]:
    h = (host or "").lower().strip().rstrip(".")
    if not h:
        return []
    base = h[4:] if h.startswith("www.") else h
    return sorted({h, base, f"www.{base}"})


def _ensure_https_root(url_or_host: str) -> str:
    s = (url_or_host or "").strip()
    if not s:
        return "https://"
    if "://" not in s:
        s = f"https://{s}"
    p = urlparse(s)
    return urlunparse(("https", (p.netloc or p.path), "/", "", "", ""))


def _apex_root(u: str) -> str:
    p = urlparse(_ensure_https_root(u))
    host = (p.netloc or "").lower().strip().rstrip(".")
    base = host[4:] if host.startswith("www.") else host
    return f"https://{base}/"


def _www_root(u: str) -> str:
    p = urlparse(_ensure_https_root(u))
    host = (p.netloc or "").lower().strip().rstrip(".")
    if host.startswith("www."):
        return f"https://{host}/"
    return f"https://www.{host}/"


def _normalized_root_key(root: str) -> str:
    """
    Normalize any root/base URL for per_root_discovered stats:
      - https scheme
      - apex host (no 'www.')
      - trailing slash
    e.g.  https://mother-parkers.com,
          https://mother-parkers.com/,
          https://www.mother-parkers.com
      -> https://mother-parkers.com/
    """
    return _apex_root(root)


def _normalize_abs_https(link: str, base: str) -> str:
    if not link:
        return ""
    if link.startswith(("mailto:", "javascript:", "tel:", "data:")):
        return ""
    absu = urljoin(base, link)
    p = urlparse(absu)
    if not p.netloc:
        return ""
    if p.scheme == "http":
        absu = urlunparse(("https", p.netloc, p.path, p.params, p.query, p.fragment))
    if "#" in absu:
        absu = absu.split("#", 1)[0]
    return absu


def _dedupe(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for u in seq:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# -------------------------
# Very light BM25 scoring utilities
# -------------------------
def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
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


def _dual_score(doc: str, comb: DualBM25Combiner, pos_q: Optional[str], neg_q: Optional[str]) -> float:
    return comb.score(doc, pos_query=pos_q, neg_query=neg_q)


# -------------------------
# Roots (+ optional brands on homepage)
# -------------------------
async def _collect_roots_https(
    base_url: str,
    *,
    discover_brands: bool,
    allowed_brand_hosts: Optional[Set[str]] = None,
) -> List[str]:
    roots: List[str] = []
    for v in _with_https_only(base_url):
        if v not in roots:
            roots.append(v)

    preferred = roots[0] if roots else None
    if preferred:
        host = (urlparse(preferred).hostname or "").lower()
        for host_variant in _domain_variants(host):
            for v in _with_https_only(host_variant):
                if v and v not in roots:
                    roots.append(v)

    if discover_brands and preferred:
        try:
            log.info("[url_seeder] Brand discovery for %s", preferred)
            raw = await discover_brand_sites(preferred)
            for u in (raw or []):
                for v in _with_https_only(str(u).strip()):
                    # Guard: do not add dataset externals as brand roots unless explicitly allowed
                    if should_block_external_for_dataset(preferred, v, allowed_brand_hosts=allowed_brand_hosts or set()):
                        log.debug("[url_seeder] skip brand root due to dataset externals policy: %s", v)
                        continue
                    if v and v not in roots:
                        roots.append(v)
            if len(roots) > 1:
                log.info("[url_seeder]   %d brand root(s)", len(roots) - 1)
        except Exception as e:
            log.debug("[url_seeder] brand discovery failed: %s", e)

    return roots


# -------------------------
# Crawl4AI homepage link discovery
# -------------------------
def _stage_run_config_for_discovery() -> CrawlerRunConfig:
    return crawler_base_cfg.clone(
        cache_mode=CacheMode.ENABLED,
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=False,
    )


async def _discover_with_crawl4ai(root: str) -> List[str]:
    run_cfg = _stage_run_config_for_discovery()
    homepage = _ensure_https_root(root)

    try:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            r = await crawler.arun(url=homepage, config=run_cfg)
    except Exception as e:
        log.debug("[url_seeder] Crawl4AI fetch failed for %s: %s", homepage, e)
        return []

    html = getattr(r, "html", None) or getattr(r, "cleaned_html", None) or ""
    if not html:
        return []

    try:
        from lxml import html as LH
        doc = LH.fromstring(html)
        hrefs = [el.get("href") for el in doc.findall(".//a") if el.get("href")]
    except Exception as e:
        log.debug("[url_seeder] lxml parse failed for %s: %s", homepage, e)
        hrefs = []

    abs_urls = []
    for h in hrefs:
        u = _normalize_abs_https(h, homepage)
        if u:
            abs_urls.append(u)

    abs_urls = [u for u in _dedupe(abs_urls) if same_registrable_domain(homepage, u)]
    return abs_urls


# -------------------------
# Fallback wrapper (parallel variants + timing)
# -------------------------
async def _sitemap_fallback_all_variants(root: str, *, timeout_s: float = 45.0) -> List[str]:
    start = time.perf_counter()
    canon = _ensure_https_root(root)
    apex = _apex_root(root)
    www = _www_root(root)

    variants = []
    for v in (canon, apex, www):
        if v not in variants:
            variants.append(v)

    log.info("[url_seeder] [sitemaps] start root=%s variants=%s", root, ", ".join(variants))

    async def _one(v: str):
        try:
            return await discover_from_sitemaps(v, timeout_s=timeout_s)
        except Exception as e:
            log.debug("[url_seeder] [sitemaps] error for %s: %s", v, e)
            return []

    results = await asyncio.gather(*(_one(v) for v in variants))
    merged = []
    for lst in results:
        merged.extend(lst or [])

    merged = [u for u in _dedupe(merged) if same_registrable_domain(canon, u)]

    elapsed = (time.perf_counter() - start)
    log.info("[url_seeder] [sitemaps] done root=%s urls=%d (%.2fs)", root, len(merged), elapsed)
    return merged


# -------------------------
# Brand detection heuristic (within a domain)
# -------------------------
def _looks_like_brand_page(u: str, root: str) -> bool:
    if not same_registrable_domain(root, u):
        return False
    path = (urlparse(u).path or "").lower()
    tokens = ("brand", "brands", "our-brands", "brand-portfolio", "portfolio/brands")
    return any(t in path for t in tokens)


# -------------------------
# Brand-root seeding: seed discovered brand roots, then append their URLs
# -------------------------
async def _seed_brand_roots(
    brand_roots: List[str],
    *,
    preferred_root: str,
    allowed_brand_hosts: Optional[Set[str]],
) -> Dict[str, List[str]]:
    """
    For each brand root, run homepage link discovery + sitemap variants;
    return dict[root] -> list[urls] (all in-domain, deduped).
    Dataset externals are blocked unless explicitly allowed via allowed_brand_hosts.
    """
    out: Dict[str, List[str]] = {}

    async def _seed_one(root: str) -> Tuple[str, List[str]]:
        # Guard: drop if dataset-external relative to preferred
        if should_block_external_for_dataset(preferred_root, root, allowed_brand_hosts=allowed_brand_hosts or set()):
            log.debug("[url_seeder] block brand root (dataset externals): %s", root)
            return (root, [])

        try:
            c4 = await _discover_with_crawl4ai(root)
            sm = await _sitemap_fallback_all_variants(root, timeout_s=45.0)
            merged = _dedupe(list(c4 or []) + list(sm or []))
            canon = _ensure_https_root(root)
            merged = [u for u in merged if same_registrable_domain(canon, u)]
            return (root, merged)
        except Exception as e:
            log.debug("[url_seeder] brand seeding failed for %s: %s", root, e)
            return (root, [])

    results = await asyncio.gather(*(_seed_one(r) for r in _dedupe(brand_roots)))
    for root, urls in results:
        out[root] = urls
    return out


# -------------------------
# (Fallback-only) simple brand expansion: append brand roots without further seeding
# -------------------------
async def _expand_with_brand_pages_simple(
    kept: List[Dict[str, Any]],
    preferred_root: str,
    company_id: Optional[str],
    allowed_brand_hosts: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    if not kept:
        return kept
    brand_page_urls = [it["url"] for it in kept if _looks_like_brand_page(it["url"], preferred_root)]
    if not brand_page_urls:
        return kept

    extras: List[Dict[str, Any]] = []
    seen_existing = {it["url"] for it in kept}

    for bp in _dedupe(brand_page_urls):
        try:
            brand_roots = await discover_brand_sites(bp)
        except Exception as e:
            log.debug("[url_seeder] brand_discovery (fallback) failed on %s: %s", bp, e)
            brand_roots = []

        for root in (brand_roots or []):
            for v in _with_https_only(root):
                # Guard: block dataset externals unless allowed
                if should_block_external_for_dataset(preferred_root, v, allowed_brand_hosts=allowed_brand_hosts or set()):
                    log.debug("[url_seeder] skip fallback brand root (dataset externals): %s", v)
                    continue
                if v in seen_existing:
                    continue
                rec = {
                    "url": v,
                    "seed_root": (preferred_root if same_registrable_domain(preferred_root, v) else v),
                    "score": None,
                    "source": "brand_discovery_root",
                }
                extras.append(rec)
                seen_existing.add(v)
                if company_id:
                    try:
                        upsert_url_index(company_id, v, status="seeded", score=None)
                    except Exception:
                        pass
    if extras:
        log.info("[url_seeder] (fallback) brand roots appended: %d", len(extras))
        kept = kept + extras
    return kept


# -------------------------
# Seeding with SIMPLE fallback (+ final url_index with dataset-aware guards)
# -------------------------
async def seed_urls(
    base_url: str,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    query: Optional[str] = None,
    score_threshold: Optional[float] = 0.25,
    company_max_pages: int = -1,
    discover_brands: bool = True,
    lang_primary: str = "en",
    lang_accept_en_regions: Optional[set[str]] = None,
    lang_strict_cctld: bool = False,
    drop_universal_externals: bool = True,
    use_dual_bm25: bool = True,
    dual_alpha: float = 0.65,
    company_id: Optional[str] = None,
    company_name: str = "",
    url_index_fallback: bool = True,
    url_index_max_pages: int = 500,
    url_index_max_depth: int = 3,
    url_index_concurrency: int = 8,
    externals_universe: Optional[ExternalsUniverse] = None,
    **_ignore,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    # If an externals universe is provided, install its protections.
    if externals_universe:
        externals_universe.install()
    allowed_brand_hosts = (externals_universe.allowed_brand_hosts if externals_universe else set())

    include = include or DEFAULT_INCLUDE_PATTERNS
    exclude = exclude or DEFAULT_EXCLUDE_PATTERNS

    pos_q = query or lang_cfg.default_product_bm25_query()
    neg_q = build_default_negative_query()
    comb = DualBM25Combiner(_bm25_lite, DualBM25Config(alpha=float(dual_alpha))) if use_dual_bm25 else None

    roots = await _collect_roots_https(
        base_url,
        discover_brands=discover_brands,
        allowed_brand_hosts=allowed_brand_hosts,
    )
    if not roots:
        return [], {
            "seed_roots": [],
            "per_root_discovered": {},
            "discovered_total": 0,
            "kept_total": 0,
            "filtered_total": 0,
            "filtered_breakdown": {},
            "seeding_source": "none",
        }
    preferred = roots[0]
    log.info("[url_seeder] Roots(%d): %s", len(roots), ", ".join(roots))

    # Track discovered URLs per canonical root (apex, no www) as sets
    root_to_urls: Dict[str, Set[str]] = {}
    raw_pairs: List[Tuple[str, str]] = []
    used_fallback = False

    # 1) Primary seeding on each root (homepage links, then sitemaps)
    for r in roots:
        c4_urls = await _discover_with_crawl4ai(r)
        urls: List[str] = []
        if c4_urls:
            urls = _dedupe(c4_urls)
        else:
            log.info("[url_seeder] 0 URLs from Crawl4AI on %s → fallback to sitemap_hunter", r)
            sm_urls = await _sitemap_fallback_all_variants(r, timeout_s=45.0)
            used_fallback = used_fallback or bool(sm_urls)
            urls = _dedupe(sm_urls or [])

        if not urls:
            continue

        root_key = _normalized_root_key(r)
        bucket = root_to_urls.setdefault(root_key, set())
        for u in urls:
            bucket.add(u)
            raw_pairs.append((u, r))

    # 2) Brand-root discovery & seeding (PRE-FILTER)
    if raw_pairs:
        brand_page_urls: List[str] = []
        for u, seed_root in raw_pairs:
            if _looks_like_brand_page(u, seed_root):
                brand_page_urls.append(u)
        if brand_page_urls:
            log.info("[url_seeder] brand pages detected=%d — discovering external brand roots", len(brand_page_urls))
            brand_roots_set: Set[str] = set()

            async def _discover_from_page(bp: str) -> List[str]:
                try:
                    return await discover_brand_sites(bp)
                except Exception as e:
                    log.debug("[url_seeder] discover_brand_sites failed on %s: %s", bp, e)
                    return []

            discovered_lists = await asyncio.gather(*(_discover_from_page(bp) for bp in _dedupe(brand_page_urls)))
            for lst in discovered_lists:
                for root in (lst or []):
                    for v in _with_https_only(root):
                        # Guard with dataset externals policy here too
                        if should_block_external_for_dataset(preferred, v, allowed_brand_hosts=allowed_brand_hosts):
                            log.debug("[url_seeder] skip discovered brand root (dataset externals): %s", v)
                            continue
                        brand_roots_set.add(v)

            if brand_roots_set:
                log.info("[url_seeder] unique brand roots discovered=%d — seeding them", len(brand_roots_set))
                seeded_map = await _seed_brand_roots(
                    sorted(brand_roots_set),
                    preferred_root=preferred,
                    allowed_brand_hosts=allowed_brand_hosts,
                )
                for root, urls in seeded_map.items():
                    if not urls:
                        continue
                    root_key = _normalized_root_key(root)
                    bucket = root_to_urls.setdefault(root_key, set())
                    for u in urls:
                        bucket.add(u)
                        raw_pairs.append((u, root))
                log.info("[url_seeder] brand-root seeding appended urls=%d", sum(len(v) for v in seeded_map.values()))

    # --- FINAL FALLBACK: url_index (dataset aware) ---
    total_discovered = sum(len(v) for v in root_to_urls.values())
    if total_discovered == 0 and url_index_fallback:
        log.info("[url_seeder] discovery empty → final fallback to url_index")
        try:
            idx_res = await discover_and_write_url_index(
                company_id=company_id or (urlparse(preferred).hostname or "unknown"),
                company_name=company_name or "",
                base_url=preferred,
                include=include,
                exclude=exclude,
                lang_primary=lang_primary,
                accept_en_regions=lang_accept_en_regions,
                strict_cctld=lang_strict_cctld,
                drop_universal_externals=drop_universal_externals,
                max_pages=int(url_index_max_pages),
                max_depth=int(url_index_max_depth),
                per_host_page_cap=max(50, url_index_max_pages),
                dual_alpha=float(dual_alpha),
                pos_query=pos_q,
                neg_query=neg_q,
                score_threshold=score_threshold,
                score_threshold_on_seeds=True,
                concurrency=int(url_index_concurrency),
            )
            idx_urls: List[Dict[str, Any]] = idx_res.get("urls", []) or []

            kept = []
            seen_k: Set[str] = set()
            for it in idx_urls:
                u = str(it.get("url") or "").strip()
                if not u or u in seen_k:
                    continue
                seen_k.add(u)
                seed_root = it.get("seed_root") or (preferred if same_registrable_domain(preferred, u) else u)
                kept.append({
                    "url": u,
                    "seed_root": seed_root,
                    "score": it.get("score"),
                })

            kept = await _expand_with_brand_pages_simple(
                kept,
                preferred_root=preferred,
                company_id=company_id,
                allowed_brand_hosts=allowed_brand_hosts,
            )

            if company_max_pages and company_max_pages > 0 and len(kept) > company_max_pages:
                trimmed = kept[company_max_pages:]
                kept = kept[:company_max_pages]
                if company_id:
                    for it in trimmed:
                        try:
                            upsert_url_index(company_id, it["url"], status="filtered_company_cap", score=it.get("score"))
                        except Exception:
                            pass

            meta = idx_res.get("meta", {}) or {}
            idx_stats = dict(meta.get("stats", {}))
            idx_metrics = dict(meta.get("seeding_metrics", {}))

            root_key = _normalized_root_key(preferred)
            per_root_discovered = {
                root_key: idx_stats.get("total_urls", len(idx_urls))
            }

            stats = {
                "seed_roots": meta.get("seed_roots", [preferred]),
                "per_root_discovered": per_root_discovered,
                "discovered_total": idx_stats.get("total_urls", len(idx_urls)),
                "kept_total": len(kept),
                "filtered_total": idx_metrics.get("filtered_total", 0),
                "filtered_breakdown": idx_metrics.get("filtered_breakdown", {}),
                "dual_used": bool(use_dual_bm25),
                "dual_alpha": float(dual_alpha) if use_dual_bm25 else None,
                "pos_query_used": bool(pos_q),
                "seeding_source": "url_index",
            }
            return kept, stats

        except Exception as e:
            log.debug("[url_seeder] url_index fallback failed: %s", e)

    # ---- Normal path: dedupe → score → filter → (company cap)
    seen: Set[str] = set()
    merged: List[Tuple[str, str]] = []
    for u, root in raw_pairs:
        if u not in seen:
            seen.add(u)
            merged.append((u, root))

    filtered_breakdown: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []
    thr = float(score_threshold) if score_threshold is not None else None

    for u, seed_root in merged:
        s = _dual_score(u, comb, pos_q, neg_q) if comb else _bm25_lite(u, pos_q or "")
        if thr is not None and s < thr:
            filtered_breakdown["score_below_threshold"] = filtered_breakdown.get("score_below_threshold", 0) + 1
            if company_id:
                try:
                    upsert_url_index(company_id, u, status="filtered_score_below_threshold", score=s)
                except Exception:
                    pass
            continue
        kept.append({"url": u, "seed_root": seed_root, "score": s})
        if company_id:
            try:
                upsert_url_index(company_id, u, status="queued", score=s)
            except Exception:
                pass

    if kept:
        pre = [{"url": it["url"], "score_hint": it["score"]} for it in kept]
        kept_sorted, dropped = apply_url_filters(
            pre,
            include=include,
            exclude=exclude,
            drop_universal_externals=drop_universal_externals,
            lang_primary=lang_primary,
            lang_accept_en_regions=lang_accept_en_regions,
            lang_strict_cctld=lang_strict_cctld,
            include_overrides_language=False,
            sort_by_priority=True,
            base_url=preferred,
            return_reasons=True,
        )  # type: ignore

        if dropped:
            for d in (dropped or []):
                reason = d.get("reason") or "unknown"
                u = d.get("url")
                filtered_breakdown[reason] = filtered_breakdown.get(reason, 0) + 1
                if company_id and u:
                    try:
                        upsert_url_index(company_id, u, status=f"filtered_{reason}", score=None)
                    except Exception:
                        pass

        kmap = {it["url"]: it for it in kept}
        kept = []
        for k in kept_sorted:
            u = k["url"]
            src = kmap.get(u) or {"score": k.get("score_hint", 0.0)}
            seed_root = preferred if same_registrable_domain(preferred, u) else u
            kept.append({"url": u, "seed_root": seed_root, "score": float(src.get("score", 0.0))})

    if company_max_pages and company_max_pages > 0 and len(kept) > company_max_pages:
        trimmed = kept[company_max_pages:]
        kept = kept[:company_max_pages]
        filtered_breakdown["company_cap"] = filtered_breakdown.get("company_cap", 0) + len(trimmed)
        if company_id:
            for it in trimmed:
                try:
                    upsert_url_index(company_id, it["url"], status="filtered_company_cap", score=it.get("score"))
                except Exception:
                    pass

    # Build per_root_counts from normalized root_to_urls
    per_root_counts: Dict[str, int] = {
        root_key: len(urls) for root_key, urls in sorted(root_to_urls.items(), key=lambda kv: kv[0])
    }

    stats = {
        "seed_roots": roots,
        "per_root_discovered": per_root_counts,
        "discovered_total": sum(per_root_counts.values()),
        "kept_total": len(kept),
        "filtered_total": sum(filtered_breakdown.values()),
        "filtered_breakdown": filtered_breakdown,
        "dual_used": bool(use_dual_bm25),
        "dual_alpha": float(dual_alpha) if use_dual_bm25 else None,
        "pos_query_used": bool(pos_q),
        "seeding_source": ("crawl4ai+fallback_sitemaps" if used_fallback else "crawl4ai"),
    }
    return kept, stats