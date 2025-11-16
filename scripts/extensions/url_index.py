from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from collections import defaultdict

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

from configs import language_settings as lang_cfg
from configs.browser_settings import browser_cfg
from configs.crawler_settings import crawler_base_cfg

from extensions.filtering import (
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
    apply_url_filters,
    same_registrable_domain,
    should_block_external_for_dataset,
    is_in_dataset_externals,
)

from extensions.dual_bm25 import (
    DualBM25Combiner,
    DualBM25Config,
    build_default_negative_query,
)

from extensions.output_paths import ensure_company_dirs, sanitize_bvdid
from extensions.run_utils import upsert_url_index
from extensions.global_state import GlobalState
from extensions.logging import LoggingExtension
from extensions.checkpoints import CompanyCheckpoint
from extensions.connectivity_guard import ConnectivityGuard

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Dynamic "universal external" persistence (with async file-level lock)
# IMPORTANT: dataset externals are NEVER blacklisted by this mechanism.
# ---------------------------------------------------------------------

DYNAMIC_COUNTS_FILE: Path = Path("outputs") / "dynamic_universal_counts.json"
DYNAMIC_THRESHOLD: int = 1  # >1 distinct companies -> considered universal
_DYN_COUNTS_LOCK = asyncio.Lock()  # guards read-modify-write of the shared file

def _dyn_counts_path() -> Path:
    return DYNAMIC_COUNTS_FILE

def _load_dyn_counts() -> Dict[str, List[str]]:
    p = _dyn_counts_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_dyn_counts(d: Dict[str, List[str]]) -> None:
    p = _dyn_counts_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        log.exception("[url_index] Failed to save dynamic counts: %s", str(p))

async def _increment_hosts_for_company_async(
    company_id: str,
    hosts: Iterable[str]
) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Update dynamic-universal counts for the given company.
    Returns:
      - counts_snapshot: host -> distinct_company_count
      - newly_blacklisted: hosts newly considered 'dynamic universal'
      - skipped_dataset_hosts: hosts excluded because they belong to dataset externals
    """
    newly_blacklisted: List[str] = []
    skipped_dataset: List[str] = []
    host_set = {h for h in hosts if h}

    # Filter out dataset externals BEFORE mutating the counts file
    filtered_hosts: Set[str] = set()
    for h in host_set:
        if is_in_dataset_externals(h):
            skipped_dataset.append(h)
            log.debug("[url_index.dynamic] skip dataset external host from dynamic counting: %s", h)
            continue
        filtered_hosts.add(h)

    async with _DYN_COUNTS_LOCK:
        data = _load_dyn_counts()
        for h in filtered_hosts:
            lst = data.get(h, [])
            if company_id not in lst:
                lst.append(company_id)
                data[h] = lst
        _save_dyn_counts(data)

        counts_snapshot = {k: len(v) for k, v in data.items()}

        # Only consider non-dataset hosts for blacklisting
        for h, lst in data.items():
            if len(lst) > DYNAMIC_THRESHOLD and h in filtered_hosts:
                newly_blacklisted.append(h)

    if skipped_dataset:
        log.info("[url_index.dynamic] skipped dataset hosts from blacklisting: %d", len(skipped_dataset))
    return counts_snapshot, newly_blacklisted, skipped_dataset


# ---------------------------------------------------------------------
# Tiny BM25-lite for URL/anchor text (document = url + anchor)
# ---------------------------------------------------------------------

def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    out, cur = [], []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur)); cur = []
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


@dataclass
class FrontierItem:
    score: float
    url: str
    depth: int
    parent: Optional[str] = None
    seed_root: Optional[str] = None
    anchor: Optional[str] = None


_HREF_RX = re.compile(r"""<a\s+[^>]*href\s*=\s*(['"])(.*?)\1[^>]*>(.*?)</a>""", re.IGNORECASE | re.DOTALL)

def _extract_links_from_html(html: str, base_url: str) -> List[Tuple[str, str]]:
    """Fallback: parse anchors from literal HTML (not preferred if CrawlResult.links is available)."""
    out: List[Tuple[str, str]] = []
    for _, href, text in _HREF_RX.findall(html or ""):
        href = (href or "").strip()
        if not href:
            continue
        if href.startswith("#") or href.lower().startswith(("javascript:", "mailto:")):
            continue
        absu = urljoin(base_url, href)
        p = urlparse(absu)
        if p.scheme != "https":  # restrict to https like previous behavior
            continue
        atxt = re.sub(r"\s+", " ", (text or "").strip())
        out.append((absu, atxt))
    return out

def _with_scheme_variants(url_or_host: str) -> List[str]:
    s = (url_or_host or "").strip()
    if not s:
        return []
    if s.lower().startswith("https://"):
        return [s]
    while s.startswith("/"):
        s = s[1:]
    return [f"https://{s}"]


# ---------------------------------------------------------------------
# Crawl4AI helpers (Container → CrawlResult, structured links first)
# ---------------------------------------------------------------------

_HEX_ID_RX = re.compile(r"^[0-9a-f]{16,64}$", re.IGNORECASE)

def _unwrap_first_result(res: Any) -> Any:
    """Return the first CrawlResult from a CrawlResultContainer or the value itself if already a CrawlResult."""
    try:
        item = res[0]  # type: ignore[index]
        return item
    except Exception:
        return res

def _structured_links(result_obj: Any) -> List[Tuple[str, str]]:
    """Read links from CrawlResult.links per Crawl4AI docs."""
    links_out: List[Tuple[str, str]] = []
    rl: Dict[str, List[Dict[str, Any]]] = getattr(result_obj, "links", {}) or {}
    for bucket in ("internal", "external"):
        for it in rl.get(bucket, []) or []:
            href = (it.get("href") or "").strip()
            if not href:
                continue
            if href.startswith("#") or href.lower().startswith(("javascript:", "mailto:")):
                continue
            p = urlparse(href)
            if not p.scheme:
                continue
            if p.scheme != "https":
                continue
            txt = re.sub(r"\s+", " ", (it.get("text") or "").strip())
            links_out.append((href, txt))
    return links_out


# ---------------------------------------------------------------------
# Main discoverer (concurrent + connectivity-guarded)
# ---------------------------------------------------------------------

@dataclass
class GenConfig:
    base_url: str
    company_id: str
    company_name: str = ""
    output_dir: Optional[Path] = None
    max_pages: int = 8000
    max_depth: int = 3
    concurrency: int = 8              # number of worker tasks
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    lang_primary: str = "en"
    accept_en_regions: Optional[Set[str]] = None
    strict_cctld: bool = False
    drop_universal_externals: bool = True
    dual_alpha: float = 0.5
    pos_query: Optional[str] = None  # if None → default_product_bm25_query()
    neg_query: Optional[str] = None  # if None → build_default_negative_query()
    per_host_page_cap: int = 4000
    dynamic_counts_file: Optional[Path] = None  # if provided, override default file path
    score_threshold: Optional[float] = 0.25
    score_threshold_on_seeds: bool = True
    guard: Optional[ConnectivityGuard] = None

    def normalize(self) -> None:
        self.include = list(self.include or DEFAULT_INCLUDE_PATTERNS)
        self.exclude = list(self.exclude or DEFAULT_EXCLUDE_PATTERNS)
        if self.pos_query is None:
            self.pos_query = lang_cfg.default_product_bm25_query()
        if self.neg_query is None:
            self.neg_query = build_default_negative_query()
        if self.accept_en_regions is None:
            self.accept_en_regions = set(lang_cfg.get("ENGLISH_REGIONS"))
        if self.dynamic_counts_file:
            global DYNAMIC_COUNTS_FILE
            globals()["DYNAMIC_COUNTS_FILE"] = Path(self.dynamic_counts_file)


class URLIndexer:
    """Link discoverer using Crawl4AI result.links (preferred) with HTML regex fallback.
       ConnectivityGuard ensures polite backoff during *actual* outages.
       Dataset externals are respected across filtering and dynamic-universal detection.
    """

    def __init__(self, cfg: GenConfig) -> None:
        self.cfg = cfg
        self.cfg.normalize()

        self.visited: Set[str] = set()
        self.frontier: List[FrontierItem] = []
        self.seen_hosts_for_dynamic: Set[str] = set()
        self.host_visit_count: Dict[str, int] = defaultdict(int)

        self._frontier_lock = asyncio.Lock()
        self._visited_lock = asyncio.Lock()
        self._metrics_lock = asyncio.Lock()
        self._out_lock = asyncio.Lock()

        self.comb = DualBM25Combiner(_bm25_lite, DualBM25Config(alpha=float(self.cfg.dual_alpha)))

        self.metrics: Dict[str, int] = {
            "discovered_total": 0,
            "kept_total": 0,
            "filtered_total": 0,
        }
        self.filtered_breakdown: Dict[str, int] = {}

        self._out_urls: List[Dict[str, Any]] = []

        self._guard: Optional[ConnectivityGuard] = self.cfg.guard
        self._owns_guard: bool = False

    def _dual_score(self, url: str, anchor: str) -> float:
        doc = " ".join([url, anchor or ""])
        return self.comb.score(doc, pos_query=self.cfg.pos_query, neg_query=self.cfg.neg_query)

    async def _fetch_page(self, crawler: AsyncWebCrawler, url: str) -> Tuple[str, List[Tuple[str, str]]]:
        run_cfg = crawler_base_cfg.clone(
            cache_mode=CacheMode.ENABLED,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=False,
        )
        res = await crawler.arun(url=url, config=run_cfg)
        if res is None:
            return "", []

        r = _unwrap_first_result(res)
        links = _structured_links(r)
        html = getattr(r, "cleaned_html", None) or getattr(r, "html", None) or ""
        if isinstance(html, str) and _HEX_ID_RX.match(html or ""):
            html = ""
        if not links and html:
            links = _extract_links_from_html(html, getattr(r, "url", url) or url)
        return html, links

    async def _enqueue(self, url: str, depth: int, parent: Optional[str], seed_root: Optional[str], anchor: str) -> None:
        async with self._visited_lock:
            if url in self.visited:
                return

        s = self._dual_score(url, anchor)

        if self.cfg.score_threshold is not None:
            if depth == 0 and not self.cfg.score_threshold_on_seeds:
                pass
            elif s < float(self.cfg.score_threshold):
                try:
                    upsert_url_index(self.cfg.company_id, url, status="filtered_score_below_threshold", score=s)
                except Exception:
                    pass
                async with self._metrics_lock:
                    self.metrics["filtered_total"] += 1
                    self.filtered_breakdown["score_below_threshold"] = self.filtered_breakdown.get("score_below_threshold", 0) + 1
                return

        async with self._frontier_lock:
            self.frontier.append(FrontierItem(score=s, url=url, depth=depth, parent=parent, seed_root=seed_root, anchor=anchor))
        try:
            upsert_url_index(self.cfg.company_id, url, status="queued", score=s)
        except Exception:
            pass

    async def _pop_best(self) -> Optional[FrontierItem]:
        async with self._frontier_lock:
            if not self.frontier:
                return None
            best_i = max(range(len(self.frontier)), key=lambda i: self.frontier[i].score)
            return self.frontier.pop(best_i)

    async def _record_visit(self, url: str, score: float, preferred: str) -> Tuple[bool, int]:
        h = (urlparse(url).hostname or "").lower()
        async with self._visited_lock:
            if self.host_visit_count[h] >= max(1, int(self.cfg.per_host_page_cap)):
                return (False, self.host_visit_count[h])
            if url in self.visited:
                return (False, self.host_visit_count[h])
            self.visited.add(url)
            self.host_visit_count[h] += 1
            self.seen_hosts_for_dynamic.add(h)
        try:
            upsert_url_index(self.cfg.company_id, url, status="seeded", score=score)
        except Exception:
            pass
        return (True, self.host_visit_count[h])

    async def _add_out_url(self, node: FrontierItem, preferred: str) -> None:
        rec = {
            "url": node.url,
            "seed_root": node.seed_root or preferred,
            "parent": node.parent,
            "depth": node.depth,
            "score": round(float(node.score), 6),
            "stage": "seeded",
        }
        async with self._out_lock:
            self._out_urls.append(rec)

    async def _ensure_connectivity(self) -> None:
        if not self._guard:
            return
        try:
            if getattr(self._guard, "state", None) and self._guard.state() == "open":
                log.info("[url_index] [%s] connectivity OPEN; waiting…", self.cfg.company_id)
                await self._guard.wait_until_healthy()
                log.info("[url_index] [%s] connectivity restored; resuming.", self.cfg.company_id)
        except Exception:
            pass

    async def _worker(self, crawler: AsyncWebCrawler, preferred: str) -> None:
        while True:
            if len(self.visited) >= self.cfg.max_pages:
                return
            node = await self._pop_best()
            if node is None:
                return

            ok, _ = await self._record_visit(node.url, node.score, preferred)
            if not ok:
                try:
                    upsert_url_index(self.cfg.company_id, node.url, status="filtered_per_host_cap", score=node.score)
                except Exception:
                    pass
                async with self._metrics_lock:
                    self.metrics["filtered_total"] += 1
                    self.filtered_breakdown["per_host_cap"] = self.filtered_breakdown.get("per_host_cap", 0) + 1
                continue

            await self._add_out_url(node, preferred)

            if node.depth >= self.cfg.max_depth:
                continue

            await self._ensure_connectivity()

            try:
                html, links = await self._fetch_page(crawler, node.url)
                if self._guard:
                    self._guard.record_success()
            except Exception as e:
                log.debug("[url_index] [%s] fetch error %s: %s", self.cfg.company_id, node.url, e)
                if self._guard:
                    self._guard.record_transport_error()
                continue

            if not links:
                continue

            # First pass: score all candidates
            candidates_scored: List[Dict[str, Any]] = []
            score_by_url: Dict[str, float] = {}
            for cand, anchor in links:
                async with self._metrics_lock:
                    self.metrics["discovered_total"] += 1
                s = self._dual_score(cand, anchor or "")
                if self.cfg.score_threshold is not None and s < float(self.cfg.score_threshold):
                    try:
                        upsert_url_index(self.cfg.company_id, cand, status="filtered_score_below_threshold", score=s)
                    except Exception:
                        pass
                    async with self._metrics_lock:
                        self.metrics["filtered_total"] += 1
                        self.filtered_breakdown["score_below_threshold"] = self.filtered_breakdown.get("score_below_threshold", 0) + 1
                    continue
                candidates_scored.append({"url": cand, "anchor": anchor, "score_hint": s})
                score_by_url[cand] = s

            if not candidates_scored:
                continue

            # NEW: block externals that belong to the dataset (cross-company links)
            dataset_blocked: List[Dict[str, Any]] = []
            filtered_for_apply: List[Dict[str, Any]] = []
            for it in candidates_scored:
                u = it["url"]
                if should_block_external_for_dataset(node.url, u):
                    dataset_blocked.append(it)
                else:
                    filtered_for_apply.append(it)

            if dataset_blocked:
                async with self._metrics_lock:
                    self.metrics["filtered_total"] += len(dataset_blocked)
                    self.filtered_breakdown["dataset_external"] = self.filtered_breakdown.get("dataset_external", 0) + len(dataset_blocked)
                for it in dataset_blocked:
                    try:
                        upsert_url_index(self.cfg.company_id, it["url"], status="filtered_dataset_external", score=score_by_url.get(it["url"]))
                    except Exception:
                        pass

            if not filtered_for_apply:
                continue

            # Standard filtering (language, universal externals, pattern include/exclude, etc.)
            kept_sorted, dropped = apply_url_filters(
                filtered_for_apply,
                include=self.cfg.include,
                exclude=self.cfg.exclude,
                drop_universal_externals=self.cfg.drop_universal_externals,
                lang_primary=self.cfg.lang_primary,
                lang_accept_en_regions=self.cfg.accept_en_regions,
                lang_strict_cctld=self.cfg.strict_cctld,
                include_overrides_language=False,
                sort_by_priority=True,
                base_url=node.url,
                return_reasons=True,
            )  # type: ignore

            if dropped:
                async with self._metrics_lock:
                    self.metrics["filtered_total"] += len(dropped)
                for d in dropped:
                    reason = d.get("reason") or "unknown"
                    u = d.get("url")
                    if not u:
                        continue
                    try:
                        upsert_url_index(self.cfg.company_id, u, status=f"filtered_{reason}", score=score_by_url.get(u))
                    except Exception:
                        pass
                    async with self._metrics_lock:
                        self.filtered_breakdown[reason] = self.filtered_breakdown.get(reason, 0) + 1

            if kept_sorted:
                async with self._metrics_lock:
                    self.metrics["kept_total"] += len(kept_sorted)
                anchor_map = {it["url"]: it.get("anchor", "") for it in filtered_for_apply}
                for it in kept_sorted:
                    cand = it["url"]
                    anc = anchor_map.get(cand, "")
                    seed_for_child = (node.seed_root or preferred) if same_registrable_domain(preferred, cand) else cand
                    await self._enqueue(cand, depth=node.depth + 1, parent=node.url, seed_root=seed_for_child, anchor=anc)

    async def run(self) -> Dict[str, Any]:
        roots = _with_scheme_variants(self.cfg.base_url)
        if not roots:
            return {"meta": {"error": "invalid_base"}, "urls": []}
        preferred = roots[0]

        _LE = LoggingExtension() if 'LoggingExtension' in globals() and LoggingExtension else None
        comp_log = log
        _ctx_token = None
        if _LE:
            try:
                comp_log = _LE.get_company_logger(self.cfg.company_id)
                _ctx_token = _LE.set_company_context(self.cfg.company_id)
            except Exception:
                comp_log = log

        cp: Optional[CompanyCheckpoint] = None
        try:
            cp = CompanyCheckpoint(self.cfg.company_id, company_name=self.cfg.company_name)
            await cp.load()
            await cp.set_seeding_stats({"seed_roots": roots})
        except Exception:
            cp = None

        seed_roots: List[str] = []
        for u in roots:
            s = self._dual_score(u, "")
            try:
                upsert_url_index(self.cfg.company_id, u, status="queued", score=s)
            except Exception:
                pass
            async with self._frontier_lock:
                self.frontier.append(FrontierItem(score=s, url=u, depth=0, parent=None, seed_root=u, anchor=""))
            seed_roots.append(u)

        host_cap = max(1, int(self.cfg.per_host_page_cap))
        comp_log.debug("[url_index] [%s] start discovery: base=%s seed_roots=%d cap=%d workers=%d",
                       self.cfg.company_id, self.cfg.base_url, len(seed_roots), host_cap, int(self.cfg.concurrency))

        guard = self._guard
        if guard is None:
            guard = ConnectivityGuard(
                probe_endpoints=("1.1.1.1:443", "8.8.8.8:443", "9.9.9.9:443"),
                interval_s=10.0,
                trip_heartbeats=4,
                connect_timeout_s=2.0,
                base_cooloff_s=8.0,
                max_cooloff_s=180.0,
                backoff_factor=2.0,
                jitter_s=0.35,
                logger=comp_log,
            )
            self._guard = guard
            self._owns_guard = True
        await guard.start()

        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                workers = [asyncio.create_task(self._worker(crawler, preferred)) for _ in range(max(1, int(self.cfg.concurrency)))]
                await asyncio.gather(*workers)
        finally:
            if self._owns_guard and self._guard:
                with contextlib.suppress(Exception):
                    await self._guard.stop()

        counts_snapshot, newly_blacklisted, skipped_dataset = await _increment_hosts_for_company_async(
            self.cfg.company_id, self.seen_hosts_for_dynamic
        )

        meta = {
            "company_id": self.cfg.company_id,
            "company_name": self.cfg.company_name,
            "base_url": self.cfg.base_url,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed_roots": seed_roots,
            "stats": {
                "total_urls": len(self._out_urls),
                "unique_hosts": len({ (urlparse(x["url"]).hostname or "").lower() for x in self._out_urls }),
                "max_depth": self.cfg.max_depth,
                "max_pages": self.cfg.max_pages,
                "per_host_cap": self.cfg.per_host_page_cap,
            },
            "dynamic_universal": {
                "file": str(_dyn_counts_path()),
                "newly_blacklisted_hosts": newly_blacklisted,
                "skipped_dataset_hosts": skipped_dataset,  # NEW: visibility for debugging
                "counts_snapshot": counts_snapshot,
            },
            "scoring": {
                "dual_alpha": float(self.cfg.dual_alpha),
                "pos_query": self.cfg.pos_query,
                "neg_query_terms_count": len((self.cfg.neg_query or "").split()),
                "score_threshold": self.cfg.score_threshold,
                "score_threshold_on_seeds": bool(self.cfg.score_threshold_on_seeds),
            },
            "connectivity": {
                "open_events": self._guard.snapshot_open_events() if self._guard else 0,
                "final_state": self._guard.state() if self._guard else "unknown",
                "probe": {"targets": list(getattr(self._guard, "targets", [])) if self._guard else []},
            },
            "filters": {
                "include": self.cfg.include,
                "exclude": self.cfg.exclude,
                "lang_primary": self.cfg.lang_primary,
                "accept_en_regions": sorted(list(self.accept_en_regions or set())) if hasattr(self, "accept_en_regions") else [],
                "strict_cctld": bool(self.cfg.strict_cctld),
                "drop_universal_externals": bool(self.cfg.drop_universal_externals),
            },
            "seeding_metrics": {
                "discovered_total": self.metrics.get("discovered_total", 0),
                "kept_total": self.metrics.get("kept_total", 0),
                "filtered_total": self.metrics.get("filtered_total", 0),
                "filtered_breakdown": dict(self.filtered_breakdown),
                "seeded_count": len(self._out_urls),
            },
        }

        try:
            if cp is not None:
                await cp.set_seeding_stats(meta)
        except Exception as e:
            comp_log.debug("[url_index] [%s] failed writing seeding stats: %s", self.cfg.company_id, e)

        comp_log.info("[url_index] [%s] discovery done: urls=%d hosts=%d",
                      self.cfg.company_id, len(self._out_urls), meta["stats"]["unique_hosts"])

        if _LE and _ctx_token is not None:
            try:
                _LE.reset_company_context(_ctx_token)
            except Exception:
                pass

        return {"meta": meta, "urls": list(self._out_urls)}


# ---------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------

async def discover_and_write_url_index(
    *,
    company_id: str,
    company_name: str,
    base_url: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    lang_primary: str = "en",
    accept_en_regions: Optional[Set[str]] = None,
    strict_cctld: bool = False,
    drop_universal_externals: bool = True,
    max_pages: int = 8000,
    max_depth: int = 3,
    per_host_page_cap: int = 4000,
    dual_alpha: float = 0.5,
    pos_query: Optional[str] = None,
    neg_query: Optional[str] = None,
    dynamic_counts_file: Optional[Path] = None,
    state: Optional[GlobalState] = None,
    score_threshold: Optional[float] = 0.25,
    score_threshold_on_seeds: bool = True,
    concurrency: int = 8,
    guard: Optional[ConnectivityGuard] = None,
) -> Dict[str, Any]:
    safe_id = sanitize_bvdid(company_id)
    ensure_company_dirs(safe_id)

    st = state or GlobalState()

    try:
        row = await st.get_company(safe_id)
        resume_mode = (row or {}).get("resume_mode")
        if resume_mode == "url_index":
            log.info("[url_index] [%s] GlobalState indicates reuse 'url_index' — skipping discovery", safe_id)
            return {"meta": {"company_id": safe_id, "base_url": base_url, "skipped": True, "reason": "reuse:url_index"}, "seeded": 0, "urls": []}
        try:
            plan = await st.recommend_resume(safe_id, requested_pipeline=["seed"], force_seeder_cache=False, bypass_local=False)
            if getattr(plan, "skip_seeding_entirely", False):
                log.info("[url_index] [%s] Resume plan says to skip seeding entirely — skipping discovery", safe_id)
                return {"meta": {"company_id": safe_id, "base_url": base_url, "skipped": True, "reason": getattr(plan, "reason", "resume-plan")}, "seeded": 0, "urls": []}
        except Exception:
            pass
    except Exception:
        pass

    try:
        await st.upsert_company(safe_id, company_name, base_url, stage="seed", status="pending")
        await st.mark_in_progress(safe_id, stage="seed")
    except Exception:
        pass

    cfg = GenConfig(
        base_url=base_url,
        company_id=safe_id,
        company_name=company_name,
        include=include,
        exclude=exclude,
        lang_primary=lang_primary,
        accept_en_regions=accept_en_regions,
        strict_cctld=strict_cctld,
        drop_universal_externals=drop_universal_externals,
        max_pages=max_pages,
        max_depth=max_depth,
        per_host_page_cap=per_host_page_cap,
        dual_alpha=dual_alpha,
        pos_query=pos_query,
        neg_query=neg_query,
        dynamic_counts_file=dynamic_counts_file,
        score_threshold=score_threshold,
        score_threshold_on_seeds=score_threshold_on_seeds,
        concurrency=concurrency,
        guard=guard,
    )

    log.info("[url_index] [%s] starting discovery: %s (workers=%d)", safe_id, base_url, int(concurrency))
    indexer = URLIndexer(cfg)
    result = await indexer.run()
    items = result.get("urls", []) or []

    seed_count = 0
    for it in items:
        u = it.get("url")
        sc = it.get("score")
        if not u:
            continue
        try:
            upsert_url_index(safe_id, u, status="seeded", score=sc)
            seed_count += 1
        except Exception as e:
            log.debug("[url_index] [%s] upsert_url_index failed for %s: %s", safe_id, u, e)

    try:
        await st.record_url_index(safe_id, count=seed_count)
        await st.set_resume_mode(safe_id, "url_index")
        await st.mark_done(safe_id, stage="seed")
    except Exception:
        pass

    log.info("[url_index] [%s] wrote url_index: seeded=%d", safe_id, seed_count)
    return {"meta": result.get("meta", {}), "seeded": seed_count, "urls": items}