from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

# Crawl4AI
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

from configs import language_settings as lang_cfg
from extensions.filtering import (
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
    apply_url_filters,
    same_registrable_domain,
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
from extensions.checkpoint import CompanyCheckpoint


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Dynamic "universal external" persistence (compatible with prior use)
# ---------------------------------------------------------------------

DYNAMIC_COUNTS_FILE: Path = Path("outputs") / "dynamic_universal_counts.json"
DYNAMIC_THRESHOLD: int = 1  # >1 distinct companies -> considered universal

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

def _increment_hosts_for_company(company_id: str, hosts: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
    data = _load_dyn_counts()
    newly_blacklisted: List[str] = []
    for h in {h for h in hosts if h}:
        lst = data.get(h, [])
        if company_id not in lst:
            lst.append(company_id)
            data[h] = lst
            if len(lst) > DYNAMIC_THRESHOLD:
                newly_blacklisted.append(h)
    _save_dyn_counts(data)
    counts_snapshot = {k: len(v) for k, v in data.items()}
    return counts_snapshot, newly_blacklisted

def _is_host_dynamically_universal(host: str) -> bool:
    data = _load_dyn_counts()
    lst = data.get((host or "").lower(), [])
    return len(lst) > DYNAMIC_THRESHOLD

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

def _extract_links(html: str, base_url: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for _, href, text in _HREF_RX.findall(html or ""):
        href = href.strip()
        if not href:
            continue
        if href.startswith("#") or href.lower().startswith("javascript:") or href.lower().startswith("mailto:"):
            continue
        absu = urljoin(base_url, href)
        p = urlparse(absu)
        if p.scheme not in {"https"}:
            continue
        atxt = re.sub(r"\s+", " ", (text or "").strip())
        out.append((absu, atxt))
    return out

def _with_scheme_variants(url_or_host: str) -> List[str]:
    s = (url_or_host or "").strip()
    if not s:
        return []
    lower = s.lower()
    if lower.startswith("https://"):
        return [s]
    while s.startswith("/"):
        s = s[1:]
    return [f"https://{s}"]

def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()

# ---------------------------------------------------------------------
# Main discoverer
# ---------------------------------------------------------------------

@dataclass
class GenConfig:
    base_url: str
    company_id: str
    company_name: str = ""
    output_dir: Optional[Path] = None
    max_pages: int = 8000
    max_depth: int = 3
    concurrency: int = 8
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
    """Internal discoverer; records filtered reasons + scores into url_index with explicit statuses."""

    def __init__(self, cfg: GenConfig) -> None:
        self.cfg = cfg
        self.cfg.normalize()

        self.visited: Set[str] = set()
        self.frontier: List[FrontierItem] = []
        self.seen_hosts_for_dynamic: Set[str] = set()

        self.comb = DualBM25Combiner(_bm25_lite, DualBM25Config(alpha=float(self.cfg.dual_alpha)))

        # seeding metrics
        self.metrics: Dict[str, int] = {
            "discovered_total": 0,
            "kept_total": 0,
            "filtered_total": 0,
        }
        self.filtered_breakdown: Dict[str, int] = {}

    def _dual_score(self, url: str, anchor: str) -> float:
        doc = " ".join([url, anchor or ""])
        return self.comb.score(doc, pos_query=self.cfg.pos_query, neg_query=self.cfg.neg_query)

    def _is_blocked_external(self, url: str) -> bool:
        h = _host(url)
        if self.cfg.drop_universal_externals and _is_host_dynamically_universal(h):
            return True
        return False

    async def _fetch_html(self, crawler: AsyncWebCrawler, url: str) -> str:
        cfg = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=False,
        )
        res = await crawler.arun(url=url, config=cfg)
        if res is None:
            return ""
        for k in ("html", "raw_html", "content", "page_html"):
            try:
                v = getattr(res, k, None)
                if isinstance(v, str) and v.strip():
                    return v
            except Exception:
                pass
        try:
            if isinstance(res, dict):
                return str(res.get("html") or res.get("raw_html") or "")  # type: ignore
        except Exception:
            pass
        return ""

    def _enqueue(self, url: str, depth: int, parent: Optional[str], seed_root: Optional[str], anchor: str) -> None:
        # If we've already visited, don't enqueue again (status should have been written earlier)
        if url in self.visited:
            return

        s = self._dual_score(url, anchor)

        # Threshold: skip enqueue if below cutoff (always allow seed roots unless flagged)
        if self.cfg.score_threshold is not None:
            if depth == 0 and not self.cfg.score_threshold_on_seeds:
                pass  # let roots through unconditionally
            elif s < float(self.cfg.score_threshold):
                # Explicitly record filtered status here
                try:
                    upsert_url_index(self.cfg.company_id, url, status="filtered_score_below_threshold", score=s)
                except Exception:
                    pass
                return

        # Enqueue + write queued status
        self.frontier.append(FrontierItem(score=s, url=url, depth=depth, parent=parent, seed_root=seed_root, anchor=anchor))
        try:
            upsert_url_index(self.cfg.company_id, url, status="queued", score=s)
        except Exception:
            pass

    def _pop_best(self) -> Optional[FrontierItem]:
        if not self.frontier:
            return None
        best_i = max(range(len(self.frontier)), key=lambda i: self.frontier[i].score)
        return self.frontier.pop(best_i)

    async def run(self) -> Dict[str, Any]:
        roots = _with_scheme_variants(self.cfg.base_url)
        if not roots:
            return {"meta": {"error": "invalid_base"}, "urls": []}
        preferred = roots[0]
        root_host = (urlparse(preferred).hostname or self.cfg.company_id).lower()

        # persistent logger + checkpoint
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

        # seed frontier: score + queued status
        seed_roots: List[str] = []
        for u in roots:
            s = self._dual_score(u, "")
            try:
                upsert_url_index(self.cfg.company_id, u, status="queued", score=s)
            except Exception:
                pass
            self.frontier.append(FrontierItem(score=s, url=u, depth=0, parent=None, seed_root=u, anchor=""))
            seed_roots.append(u)

        host_cap = max(1, int(self.cfg.per_host_page_cap))
        out_urls: List[Dict[str, Any]] = []

        comp_log.debug("[url_index] [%s] start discovery: base=%s seed_roots=%d",
                       self.cfg.company_id, self.cfg.base_url, len(seed_roots))

        async with AsyncWebCrawler(config=None) as crawler:
            while self.frontier and len(self.visited) < self.cfg.max_pages:
                node = self._pop_best()
                if node is None:
                    break
                url, depth = node.url, node.depth
                if url in self.visited:
                    continue

                h = (urlparse(url).hostname or "").lower()
                if sum(1 for v in self.visited if (urlparse(v).hostname or "").lower() == h) >= host_cap:
                    # per-host limit -> record as filtered
                    try:
                        upsert_url_index(self.cfg.company_id, url, status="filtered_per_host_cap", score=node.score)
                    except Exception:
                        pass
                    self.metrics["filtered_total"] += 1
                    self.filtered_breakdown["per_host_cap"] = self.filtered_breakdown.get("per_host_cap", 0) + 1
                    continue

                # mark visited and write SEEEDED status immediately
                self.visited.add(url)
                self.seen_hosts_for_dynamic.add(h)
                try:
                    upsert_url_index(self.cfg.company_id, url, status="seeded", score=node.score)
                except Exception:
                    pass

                out_urls.append({
                    "url": url,
                    "seed_root": node.seed_root or preferred,
                    "parent": node.parent,
                    "depth": depth,
                    "score": round(float(node.score), 6),
                    "stage": "seeded",
                })

                if depth >= self.cfg.max_depth:
                    # Depth cap reached—don’t expand children. (Status already 'seeded'.)
                    continue

                # fetch & parse children
                try:
                    html = await self._fetch_html(crawler, url)
                except Exception as e:
                    comp_log.debug("[url_index] [%s] fetch error %s: %s", self.cfg.company_id, url, e)
                    # keep it 'seeded'; we simply won't expand children on fetch failure
                    continue

                links = _extract_links(html, url)
                if not links:
                    continue

                # Build candidate list with scores and anchors; count discoveries
                candidates_scored: List[Dict[str, Any]] = []
                score_by_url: Dict[str, float] = {}
                for cand, anchor in links:
                    self.metrics["discovered_total"] += 1
                    s = self._dual_score(cand, anchor or "")

                    # Threshold: drop early and record explicit filtered status
                    if self.cfg.score_threshold is not None and s < float(self.cfg.score_threshold):
                        try:
                            upsert_url_index(self.cfg.company_id, cand, status="filtered_score_below_threshold", score=s)
                        except Exception:
                            pass
                        self.metrics["filtered_total"] += 1
                        self.filtered_breakdown["score_below_threshold"] = self.filtered_breakdown.get("score_below_threshold", 0) + 1
                        continue

                    candidates_scored.append({"url": cand, "anchor": anchor, "score_hint": s})
                    score_by_url[cand] = s

                if not candidates_scored:
                    continue

                # apply filters with REASONS
                kept_sorted, dropped = apply_url_filters(
                    candidates_scored,
                    include=self.cfg.include,
                    exclude=self.cfg.exclude,
                    drop_universal_externals=self.cfg.drop_universal_externals,
                    lang_primary=self.cfg.lang_primary,
                    lang_accept_en_regions=self.cfg.accept_en_regions,
                    lang_strict_cctld=self.cfg.strict_cctld,
                    include_overrides_language=False,
                    sort_by_priority=True,
                    base_url=url,
                    return_reasons=True,
                )  # type: ignore

                # record drops with reason + score (explicit filtered_* statuses)
                if dropped:
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
                        self.filtered_breakdown[reason] = self.filtered_breakdown.get(reason, 0) + 1

                # enqueue children from kept list (write queued here)
                if kept_sorted:
                    self.metrics["kept_total"] += len(kept_sorted)
                    for it in kept_sorted:
                        cand = it["url"]
                        anc = ""
                        for u2, a2 in links:
                            if u2 == cand:
                                anc = a2
                                break
                        seed_for_child = (node.seed_root or preferred) if same_registrable_domain(preferred, cand) else cand
                        self._enqueue(cand, depth=depth + 1, parent=url, seed_root=seed_for_child, anchor=anc)

        counts_snapshot, newly_blacklisted = _increment_hosts_for_company(root_host, self.seen_hosts_for_dynamic)

        meta = {
            "company_id": self.cfg.company_id,
            "company_name": self.cfg.company_name,
            "base_url": self.cfg.base_url,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed_roots": seed_roots,
            "stats": {
                "total_urls": len(out_urls),
                "unique_hosts": len({ (urlparse(x["url"]).hostname or "").lower() for x in out_urls }),
                "max_depth": self.cfg.max_depth,
                "max_pages": self.cfg.max_pages,
                "per_host_cap": self.cfg.per_host_page_cap,
            },
            "dynamic_universal": {
                "file": str(_dyn_counts_path()),
                "newly_blacklisted_hosts": newly_blacklisted,
                "counts_snapshot": counts_snapshot,
            },
            "scoring": {
                "dual_alpha": float(self.cfg.dual_alpha),
                "pos_query": self.cfg.pos_query,
                "neg_query_terms_count": len((self.cfg.neg_query or "").split()),
                "score_threshold": self.cfg.score_threshold,
                "score_threshold_on_seeds": bool(self.cfg.score_threshold_on_seeds),
            },
            "filters": {
                "include": self.cfg.include,
                "exclude": self.cfg.exclude,
                "lang_primary": self.cfg.lang_primary,
                "accept_en_regions": sorted(list(self.cfg.accept_en_regions or set())),
                "strict_cctld": bool(self.cfg.strict_cctld),
                "drop_universal_externals": bool(self.cfg.drop_universal_externals),
            },
        }

        # persist seeding metrics into crawl_meta.json
        try:
            if cp is not None:
                seeding_meta = dict(meta)
                seeding_meta.setdefault("seeding_metrics", {})
                seeding_meta["seeding_metrics"].update({
                    "discovered_total": self.metrics.get("discovered_total", 0),
                    "kept_total": self.metrics.get("kept_total", 0),
                    "filtered_total": self.metrics.get("filtered_total", 0),
                    "filtered_breakdown": self.filtered_breakdown,
                    "seeded_count": len(out_urls),
                })
                await cp.set_seeding_stats(seeding_meta)
        except Exception as e:
            comp_log.debug("[url_index] [%s] failed writing seeding stats: %s", self.cfg.company_id, e)

        comp_log.info("[url_index] [%s] discovery done: urls=%d hosts=%d",
                      self.cfg.company_id, len(out_urls), meta["stats"]["unique_hosts"])

        # reset logging context if any
        if _LE and _ctx_token is not None:
            try:
                _LE.reset_company_context(_ctx_token)
            except Exception:
                pass

        return {"meta": meta, "urls": out_urls}

# ---------------------------------------------------------------------
# Public helper to discover + persist into url_index.json via upsert
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
    max_pages: int = 500,
    max_depth: int = 3,
    per_host_page_cap: int = 250,
    dual_alpha: float = 0.5,
    pos_query: Optional[str] = None,
    neg_query: Optional[str] = None,
    dynamic_counts_file: Optional[Path] = None,
    state: Optional[GlobalState] = None,
    # NEW: threshold knobs
    score_threshold: Optional[float] = 0.25,
    score_threshold_on_seeds: bool = True,
) -> Dict[str, Any]:
    """
    Runs discovery and writes entries into outputs/{bvdid}/checkpoints/url_index.json
    using upsert_url_index(..., status='seeded') per URL for visited pages,
    'queued' for enqueued pages, and 'filtered_*' for drops.
    """
    safe_id = sanitize_bvdid(company_id)
    ensure_company_dirs(safe_id)

    # Ensure a state instance
    st = state or GlobalState()

    # --- Reuse-index short-circuit via GlobalState ---
    try:
        row = await st.get_company(safe_id)
        resume_mode = (row or {}).get("resume_mode")
        if resume_mode == "url_index":
            log.info("[url_index] [%s] GlobalState indicates reuse 'url_index' — skipping discovery", safe_id)
            return {"meta": {"company_id": safe_id, "base_url": base_url, "skipped": True, "reason": "reuse:url_index"}, "seeded": 0}
        try:
            plan = await st.recommend_resume(safe_id, requested_pipeline=["seed"], force_seeder_cache=False, bypass_local=False)
            if getattr(plan, "skip_seeding_entirely", False):
                log.info("[url_index] [%s] Resume plan says to skip seeding entirely — skipping discovery", safe_id)
                return {"meta": {"company_id": safe_id, "base_url": base_url, "skipped": True, "reason": getattr(plan, "reason", "resume-plan")}, "seeded": 0}
        except Exception:
            pass
    except Exception:
        pass

    # record pending in global state
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
    )

    log.info("[url_index] [%s] starting discovery: %s", safe_id, base_url)
    indexer = URLIndexer(cfg)
    result = await indexer.run()
    items = result.get("urls", []) or []

    # Items were recorded during run(); ensure scores present and final status 'seeded' for visited
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
    return {"meta": result.get("meta", {}), "seeded": seed_count}