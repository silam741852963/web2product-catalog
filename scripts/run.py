from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.deep_crawling.filters import FilterChain

# Repo config factories
from configs.browser import default_browser_factory
from configs.crawler import default_crawler_factory
from configs.deep_crawl import (
    DFSDeepCrawlStrategyProvider,
    DeepCrawlStrategyFactory,
    DeepCrawlStrategyFactory as _DeepCrawlStrategyFactory,
    build_deep_strategy,
)
from configs.js_injection import (
    PageInteractionFactory,
    PageInteractionPolicy,
    default_page_interaction_factory,
)
from configs.language import default_language_factory
from configs.llm import (
    DEFAULT_FULL_INSTRUCTION,
    DEFAULT_PRESENCE_INSTRUCTION,
    LLMExtractionFactory,
    RemoteAPIProviderStrategy,
    default_ollama_provider_strategy,
)
from configs.md import default_md_factory

# Extensions
from extensions.adaptive_scheduling import (
    AdaptiveScheduler,
    AdaptiveSchedulingConfig,
    compute_retry_exit_code_from_store,
)
from extensions.connectivity_guard import ConnectivityGuard
from extensions.crawl_state import (
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_PENDING,
    get_crawl_state,
)
from extensions.dataset_external import build_dataset_externals
from extensions.dual_bm25 import (
    DualBM25Filter,
    DualBM25Scorer,
    build_dual_bm25_components,
)
from extensions.filtering import (
    FirstTimeURLFilter,
    HTMLContentFilter,
    LanguageAwareURLFilter,
    UniversalExternalFilter,
)
from extensions.llm_passes import (
    run_full_pass_for_company,
    run_presence_pass_for_company,
)
from extensions.load_source import CompanyInput, load_companies_from_source
from extensions.logging import LoggingExtension
from extensions import md_gating
from extensions import output_paths
from extensions.output_paths import ensure_company_dirs
from extensions.page_pipeline import ConcurrentPageResultProcessor, UrlIndexFlushConfig
from extensions.resource_monitor import ResourceMonitor, ResourceMonitorConfig
from extensions.retry_state import RetryStateStore

logger = logging.getLogger("deep_crawl_runner")

_HTTP_LANG_MAP: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
}

RETRY_EXIT_CODE = 17
_forced_exit_code: Optional[int] = None
_retry_store_instance: Optional[RetryStateStore] = None


class CriticalMemoryPressure(RuntimeError):
    """Kept for compatibility: other modules may raise/import this."""

    def __init__(self, message: str, severity: str = "emergency") -> None:
        super().__init__(message)
        self.severity = severity


class CrawlerFatalError(RuntimeError):
    """Only for irrecoverable driver/browser connection failures."""


class CrawlerTimeoutError(TimeoutError):
    """Recoverable crawl timeout (do not crash the whole run)."""

    def __init__(self, message: str, *, stage: str, company_id: str, url: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.company_id = company_id
        self.url = url


def _touch(touch_cb: Optional[Any], company_id: str) -> None:
    if touch_cb is None:
        return
    try:
        touch_cb(company_id)
    except Exception:
        return


async def _run_cleanup_even_if_cancelled(coro: Any) -> None:
    """
    Ensure cleanup runs even if the current task is cancelled.
    """
    t = asyncio.current_task()
    uncancel = getattr(t, "uncancel", None) if t is not None else None

    if callable(uncancel):
        try:
            uncancel()
            await coro
        finally:
            try:
                t.cancel()
            except Exception:
                pass
        return

    try:
        await asyncio.shield(coro)
    except asyncio.CancelledError:

        async def _bg() -> None:
            try:
                await coro
            except Exception:
                logger.exception("Background cleanup failed")

        asyncio.create_task(_bg(), name="bg-cleanup")
        raise


def _maybe_malloc_trim() -> None:
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        return


def _is_playwright_driver_disconnect(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    lowered = msg.lower()
    needles = [
        "connection closed while reading from the driver",
        "pipe closed by peer",
        "browsercontext.new_page: connection closed",
        "browser has been closed",
        "target page, context or browser has been closed",
        "playwright connection closed",
    ]
    return any(n in lowered for n in needles)


# ---------------------------------------------------------------------------
# Crawler pool
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CrawlerSlot:
    idx: int
    crawler: AsyncWebCrawler
    processed_companies: int = 0
    last_restart_ts: float = 0.0


class CrawlerPool:
    def __init__(
        self,
        *,
        browser_cfg: Any,
        size: int,
        recycle_after_companies: int,
    ) -> None:
        self._browser_cfg = browser_cfg
        self._size = max(1, int(size))
        self._recycle_after = max(0, int(recycle_after_companies))

        self._queue: asyncio.Queue[_CrawlerSlot] = asyncio.Queue()
        self._all: List[_CrawlerSlot] = []
        self._started = False
        self._closing = False

        self._force_restart_budget: int = 0
        self._force_restart_reason: str = "forced_recycle"

    @property
    def size(self) -> int:
        return self._size

    def free_slots_approx(self) -> int:
        try:
            return int(self._queue.qsize())
        except Exception:
            return 0

    def request_recycle(self, count: int, reason: str) -> None:
        if count <= 0:
            return
        self._force_restart_budget = min(
            self._size, self._force_restart_budget + int(count)
        )
        self._force_restart_reason = reason or "forced_recycle"

    async def recycle_idle_now(self, count: int, reason: str) -> int:
        restarted = 0
        n = min(self._size, max(0, int(count)))
        for _ in range(n):
            try:
                slot = self._queue.get_nowait()
            except Exception:
                break
            slot = await self._restart_slot(slot, reason=reason or "recycle_idle")
            try:
                self._queue.put_nowait(slot)
            except Exception:
                try:
                    await slot.crawler.__aexit__(None, None, None)
                except Exception:
                    pass
                break
            restarted += 1
        return restarted

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._closing = False

        created: List[_CrawlerSlot] = []
        try:
            for i in range(self._size):
                c = AsyncWebCrawler(config=self._browser_cfg)
                await c.__aenter__()
                slot = _CrawlerSlot(
                    idx=i, crawler=c, processed_companies=0, last_restart_ts=time.time()
                )
                created.append(slot)

            self._all = created
            for slot in created:
                self._queue.put_nowait(slot)

            logger.info(
                "[CrawlerPool] started size=%d recycle_after_companies=%d",
                self._size,
                self._recycle_after,
            )
        except Exception:
            for slot in created:
                try:
                    await slot.crawler.__aexit__(None, None, None)
                except Exception:
                    pass
            self._all = []
            self._started = False
            raise

    async def stop(self) -> None:
        self._closing = True

        drained: List[_CrawlerSlot] = []
        while True:
            try:
                slot = self._queue.get_nowait()
                drained.append(slot)
            except Exception:
                break

        seen: set[int] = set()
        for slot in drained + list(self._all):
            if slot.idx in seen:
                continue
            seen.add(slot.idx)
            try:
                await slot.crawler.__aexit__(None, None, None)
            except Exception:
                pass

        self._all = []
        self._started = False
        logger.info("[CrawlerPool] stopped")

    async def _restart_slot(self, slot: _CrawlerSlot, *, reason: str) -> _CrawlerSlot:
        try:
            await slot.crawler.__aexit__(None, None, None)
        except Exception:
            pass

        c = AsyncWebCrawler(config=self._browser_cfg)
        await c.__aenter__()
        slot.crawler = c
        slot.processed_companies = 0
        slot.last_restart_ts = time.time()
        logger.warning("[CrawlerPool] restarted slot=%d reason=%s", slot.idx, reason)
        return slot

    async def lease(self) -> "_CrawlerLease":
        slot = await self._queue.get()
        return _CrawlerLease(pool=self, slot=slot)

    async def _release(self, slot: _CrawlerSlot, *, fatal: bool, reason: str) -> None:
        if self._closing:
            try:
                await slot.crawler.__aexit__(None, None, None)
            except Exception:
                pass
            return

        if fatal:
            slot = await self._restart_slot(slot, reason=reason)
        else:
            if self._force_restart_budget > 0:
                self._force_restart_budget -= 1
                slot = await self._restart_slot(slot, reason=self._force_restart_reason)
                try:
                    self._queue.put_nowait(slot)
                except Exception:
                    try:
                        await slot.crawler.__aexit__(None, None, None)
                    except Exception:
                        pass
                return

            slot.processed_companies += 1
            if (
                self._recycle_after > 0
                and slot.processed_companies >= self._recycle_after
            ):
                slot = await self._restart_slot(slot, reason="periodic_recycle")

        try:
            self._queue.put_nowait(slot)
        except Exception:
            try:
                await slot.crawler.__aexit__(None, None, None)
            except Exception:
                pass


class _CrawlerLease:
    def __init__(self, *, pool: CrawlerPool, slot: _CrawlerSlot) -> None:
        self._pool = pool
        self._slot = slot
        self._fatal = False
        self._reason = "ok"

    @property
    def crawler(self) -> AsyncWebCrawler:
        return self._slot.crawler

    def mark_fatal(self, reason: str) -> None:
        self._fatal = True
        self._reason = reason or "fatal"

    async def __aenter__(self) -> AsyncWebCrawler:
        return self._slot.crawler

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            if isinstance(exc, CrawlerFatalError) or _is_playwright_driver_disconnect(
                exc
            ):
                self._fatal = True
                self._reason = "driver_disconnect"
        await self._pool._release(self._slot, fatal=self._fatal, reason=self._reason)


# ---------------------------------------------------------------------------
# Models + IO helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Company:
    company_id: str
    domain_url: str


def _companies_from_source(path: Path) -> List[Company]:
    inputs: List[CompanyInput] = load_companies_from_source(path)
    companies: List[Company] = [
        Company(company_id=ci.bvdid, domain_url=ci.url) for ci in inputs
    ]
    logger.info("Loaded %d companies from source %s", len(companies), path)
    return companies


def _crawl_meta_path(company_id: str) -> Path:
    dirs = ensure_company_dirs(company_id)
    meta_dir = dirs.get("metadata") or dirs.get("checkpoints")
    if meta_dir is None:
        meta_dir = Path(output_paths.OUTPUT_ROOT) / company_id / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
    return Path(meta_dir) / "crawl_meta.json"


def _write_crawl_meta(company: Company, snapshot: Any) -> None:
    path = _crawl_meta_path(company.company_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "company_id": company.company_id,
        "root_url": company.domain_url,
        "status": getattr(snapshot, "status", None),
        "urls_total": getattr(snapshot, "urls_total", None),
        "urls_markdown_done": getattr(snapshot, "urls_markdown_done", None),
        "urls_llm_done": getattr(snapshot, "urls_llm_done", None),
        "last_error": getattr(snapshot, "last_error", None),
        "last_crawled_at": datetime.now(timezone.utc).isoformat(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_in_progress_companies(out_dir: Path) -> List[str]:
    p = out_dir / "crawl_global_state.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse crawl_global_state.json at %s", p)
        return []
    ids = data.get("in_progress_companies") or []
    out: List[str] = []
    for x in ids:
        s = str(x).strip() if x is not None else ""
        if s:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Crawl helpers
# ---------------------------------------------------------------------------


def _build_filter_chain(
    company: Company,
    args: argparse.Namespace,
    dataset_externals: frozenset[str],
    bm25_filter: Optional[DualBM25Filter],
) -> FilterChain:
    first_time_filter = FirstTimeURLFilter()
    html_filter = HTMLContentFilter()
    lang_filter = LanguageAwareURLFilter(lang_code=args.lang)

    universal_filter = UniversalExternalFilter(
        dataset_externals=sorted(dataset_externals),
        name=f"UniversalExternalFilter[{company.company_id}]",
    )
    universal_filter.set_company_url(company.domain_url)

    filters = [html_filter, first_time_filter, universal_filter, lang_filter]
    if bm25_filter is not None:
        filters.append(bm25_filter)
    return FilterChain(filters)


def _tune_page_pipeline_params(
    args: argparse.Namespace, scheduler: Optional[AdaptiveScheduler]
) -> Dict[str, Any]:
    conc = int(getattr(args, "page_result_concurrency", 6))
    qmax = int(getattr(args, "page_queue_maxsize", 64))
    flush_every = int(getattr(args, "url_index_flush_every", 24))
    flush_interval = float(getattr(args, "url_index_flush_interval_sec", 0.5))
    idx_qmax = int(getattr(args, "url_index_queue_maxsize", 2048))

    used_raw = 0.0
    used_eff = 0.0
    ramp_raw = 0.0
    if scheduler is not None:
        try:
            snap = scheduler.get_state_snapshot()
            used_raw = float(snap.get("used_frac_raw", 0.0) or 0.0)
            used_eff = float(snap.get("used_frac", 0.0) or 0.0)
            ramp_raw = float(snap.get("ramp_raw_frac_per_sec", 0.0) or 0.0)
        except Exception:
            pass

    hot2 = (used_raw >= 0.90) or (used_eff >= 0.86) or (ramp_raw >= 0.020)
    hot1 = (used_raw >= 0.88) or (used_eff >= 0.83) or (ramp_raw >= 0.012)

    if hot2:
        conc = max(1, conc // 3)
        qmax = max(16, qmax // 2)
        idx_qmax = max(512, idx_qmax // 2)
        flush_every = max(8, flush_every // 2)
        flush_interval = max(0.2, flush_interval / 2.0)
    elif hot1:
        conc = max(1, conc // 2)
        qmax = max(32, qmax // 2)
        idx_qmax = max(1024, idx_qmax)
        flush_every = max(12, flush_every)
        flush_interval = max(0.3, flush_interval)

    return {
        "page_result_concurrency": conc,
        "page_queue_maxsize": qmax,
        "url_index_flush_every": flush_every,
        "url_index_flush_interval_sec": flush_interval,
        "url_index_queue_maxsize": idx_qmax,
    }


async def crawl_company(
    company: Company,
    *,
    crawler: AsyncWebCrawler,
    deep_strategy: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    root_urls: Optional[List[str]] = None,
    crawler_base_cfg: Any = None,
    page_policy: PageInteractionPolicy,
    page_interaction_factory: PageInteractionFactory,
    page_timeout_ms: Optional[int] = None,
    page_result_concurrency: int = 8,
    page_queue_maxsize: int = 64,
    url_index_flush_every: int = 64,
    url_index_flush_interval_sec: float = 1.0,
    url_index_queue_maxsize: int = 2048,
    touch_cb: Optional[Any] = None,
    arun_init_timeout_sec: float = 90.0,
    stream_no_yield_timeout_sec: float = 900.0,
    submit_timeout_sec: float = 30.0,
) -> None:
    logger.info(
        "Starting crawl for company_id=%s url=%s",
        company.company_id,
        company.domain_url,
    )

    ensure_company_dirs(company.company_id)
    start_urls: List[str] = list(root_urls) if root_urls else [company.domain_url]

    flush_cfg = UrlIndexFlushConfig(
        flush_every=max(1, int(url_index_flush_every)),
        flush_interval_sec=max(0.2, float(url_index_flush_interval_sec)),
        queue_maxsize=max(256, int(url_index_queue_maxsize)),
    )

    processor = ConcurrentPageResultProcessor(
        company=company,
        guard=guard,
        gating_cfg=gating_cfg,
        timeout_error_marker="",
        mark_company_timeout_cb=None,
        mark_company_memory_cb=None,
        concurrency=max(1, int(page_result_concurrency)),
        page_queue_maxsize=max(1, int(page_queue_maxsize)),
        url_index_flush_cfg=flush_cfg,
    )

    cancelled_exc: Optional[BaseException] = None
    await processor.start()
    _touch(touch_cb, company.company_id)

    try:
        for start_url in start_urls:
            _touch(touch_cb, company.company_id)

            clone_kwargs: Dict[str, Any] = {
                "cache_mode": CacheMode.BYPASS,
                "remove_overlay_elements": True,
                "deep_crawl_strategy": deep_strategy,
                "stream": True,
            }

            interaction_cfg = page_interaction_factory.base_config(
                url=start_url,
                policy=page_policy,
                js_only=False,
            )
            js_code = (
                "\n\n".join(interaction_cfg.js_code) if interaction_cfg.js_code else ""
            )
            clone_kwargs.update(
                js_code=js_code,
                js_only=interaction_cfg.js_only,
                wait_for=interaction_cfg.wait_for,
            )

            config = crawler_base_cfg.clone(**clone_kwargs)
            if page_timeout_ms is not None:
                try:
                    config.page_timeout = page_timeout_ms  # type: ignore[attr-defined]
                except Exception:
                    pass

            try:
                _touch(touch_cb, company.company_id)
                results_or_gen = await asyncio.wait_for(
                    crawler.arun(start_url, config=config),
                    timeout=float(arun_init_timeout_sec),
                )
                _touch(touch_cb, company.company_id)
            except asyncio.TimeoutError as e:
                raise CrawlerTimeoutError(
                    f"crawler.arun init timeout ({arun_init_timeout_sec:.1f}s) company_id={company.company_id} url={start_url}",
                    stage="arun_init",
                    company_id=company.company_id,
                    url=start_url,
                ) from e
            except Exception as e:
                if _is_playwright_driver_disconnect(e):
                    raise CrawlerFatalError(
                        f"Playwright driver disconnected at arun(start_url={start_url})"
                    ) from e
                raise

            if not isinstance(results_or_gen, list):
                agen = results_or_gen
                try:
                    while True:
                        try:
                            page_result = await asyncio.wait_for(
                                agen.__anext__(),
                                timeout=float(stream_no_yield_timeout_sec),
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as e:
                            raise CrawlerTimeoutError(
                                f"stream no-yield timeout ({stream_no_yield_timeout_sec:.1f}s) company_id={company.company_id} url={start_url}",
                                stage="stream_no_yield",
                                company_id=company.company_id,
                                url=start_url,
                            ) from e

                        _touch(touch_cb, company.company_id)

                        try:
                            await asyncio.wait_for(
                                processor.submit(page_result),
                                timeout=float(submit_timeout_sec),
                            )
                        except asyncio.TimeoutError as e:
                            raise CrawlerTimeoutError(
                                f"processor.submit timeout ({submit_timeout_sec:.1f}s) company_id={company.company_id} url={start_url}",
                                stage="submit",
                                company_id=company.company_id,
                                url=start_url,
                            ) from e

                        _touch(touch_cb, company.company_id)

                except asyncio.CancelledError as e:
                    cancelled_exc = e
                    abort = getattr(processor, "abort", None)
                    if callable(abort):
                        try:
                            maybe = abort()
                            if asyncio.iscoroutine(maybe):
                                await maybe
                        except Exception:
                            logger.exception(
                                "Error aborting page processor (company_id=%s)",
                                company.company_id,
                            )
                    raise
                except Exception as e:
                    abort = getattr(processor, "abort", None)
                    if callable(abort):
                        try:
                            maybe = abort()
                            if asyncio.iscoroutine(maybe):
                                await maybe
                        except Exception:
                            logger.exception(
                                "Error aborting page processor on error (company_id=%s)",
                                company.company_id,
                            )

                    if _is_playwright_driver_disconnect(e):
                        raise CrawlerFatalError(
                            "Playwright driver disconnected during streaming"
                        ) from e
                    raise
                finally:
                    aclose = getattr(agen, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception:
                            logger.exception(
                                "Error closing deep crawl generator (company_id=%s)",
                                company.company_id,
                            )

            else:
                for page_result in results_or_gen:
                    _touch(touch_cb, company.company_id)
                    try:
                        await asyncio.wait_for(
                            processor.submit(page_result),
                            timeout=float(submit_timeout_sec),
                        )
                    except asyncio.TimeoutError as e:
                        raise CrawlerTimeoutError(
                            f"processor.submit timeout ({submit_timeout_sec:.1f}s) company_id={company.company_id} url={start_url}",
                            stage="submit",
                            company_id=company.company_id,
                            url=start_url,
                        ) from e
                    _touch(touch_cb, company.company_id)

    finally:
        try:
            _touch(touch_cb, company.company_id)
            if cancelled_exc is not None:
                await _run_cleanup_even_if_cancelled(processor.finish())
            else:
                await processor.finish()
        finally:
            try:
                gc.collect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Company pipeline
# ---------------------------------------------------------------------------


def _should_skip_company(status: str, llm_mode: str) -> bool:
    """
    Defines what "DONE" means under each mode:
      - llm_mode == none: markdown_done OR llm_done means no more work
      - otherwise: must be llm_done
    """
    if llm_mode == "none":
        return status in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            "terminal_done",
        )
    return status in (COMPANY_STATUS_LLM_DONE, "terminal_done")


async def run_company_pipeline(
    company: Company,
    idx: int,
    total: int,
    *,
    logging_ext: LoggingExtension,
    state: Any,
    guard: ConnectivityGuard,
    gating_cfg: md_gating.MarkdownGatingConfig,
    crawler_pool: CrawlerPool,
    args: argparse.Namespace,
    dataset_externals: frozenset[str],
    url_scorer: Optional[DualBM25Scorer],
    bm25_filter: Optional[DualBM25Filter],
    run_id: Optional[str],
    presence_llm: Any,
    full_llm: Any,
    dfs_factory: Optional[_DeepCrawlStrategyFactory],
    crawler_base_cfg: Any,
    page_policy: PageInteractionPolicy,
    page_interaction_factory: PageInteractionFactory,
    page_timeout_ms: Optional[int],
    enable_malloc_trim: bool,
    retry_store: RetryStateStore,
    touch_cb: Optional[Any] = None,
    scheduler: Optional[AdaptiveScheduler] = None,
) -> bool:
    completed_ok = False
    token: Optional[Any] = None
    cancelled_exc: Optional[BaseException] = None
    finalize_in_progress_md: bool = bool(
        getattr(args, "finalize_in_progress_md", False)
    )

    company_logger = None

    try:
        _touch(touch_cb, company.company_id)

        try:
            snap = await state.get_company_snapshot(company.company_id)
            status = snap.status or COMPANY_STATUS_PENDING
        except Exception:
            status = COMPANY_STATUS_PENDING

        do_crawl = False
        resume_roots: Optional[List[str]] = None

        if status == COMPANY_STATUS_PENDING:
            do_crawl = True
        elif status == COMPANY_STATUS_MD_NOT_DONE:
            pending_md = await state.get_pending_urls_for_markdown(company.company_id)
            if pending_md:
                do_crawl = True
                resume_roots = pending_md
        elif status in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            COMPANY_STATUS_LLM_NOT_DONE,
            "terminal_done",
        ):
            do_crawl = False
        else:
            do_crawl = True

        will_run_llm_presence = args.llm_mode == "presence" and presence_llm is not None
        will_run_llm_full = args.llm_mode == "full" and full_llm is not None
        will_run_llm = (
            will_run_llm_presence or will_run_llm_full
        ) and not finalize_in_progress_md

        if not do_crawl and not will_run_llm:
            completed_ok = True
            return True

        token = logging_ext.set_company_context(company.company_id)
        company_logger = logging_ext.get_company_logger(company.company_id)
        company_logger.info(
            "=== [%d/%d] Company company_id=%s url=%s ===",
            idx,
            total,
            company.company_id,
            company.domain_url,
        )

        filter_chain = _build_filter_chain(
            company=company,
            args=args,
            dataset_externals=dataset_externals,
            bm25_filter=bm25_filter,
        )

        max_pages_for_strategy: Optional[int] = None
        if getattr(args, "max_pages", None) and args.max_pages > 0:
            max_pages_for_strategy = int(args.max_pages)

        deep_strategy = build_deep_strategy(
            strategy=args.strategy,
            filter_chain=filter_chain,
            url_scorer=url_scorer,
            dfs_factory=dfs_factory,
            max_pages=max_pages_for_strategy,
        )

        # -------------------- Crawl stage --------------------
        if do_crawl:
            await guard.wait_until_healthy()
            _touch(touch_cb, company.company_id)

            lease_timeout = float(getattr(args, "crawler_lease_timeout_sec", 120.0))
            arun_init_timeout_sec = float(getattr(args, "arun_init_timeout_sec", 90.0))
            stream_no_yield_timeout_sec = float(
                getattr(args, "stream_no_yield_timeout_sec", 900.0)
            )
            submit_timeout_sec = float(getattr(args, "submit_timeout_sec", 30.0))

            tuned = _tune_page_pipeline_params(args=args, scheduler=scheduler)

            max_attempts = 2
            last_err: Optional[BaseException] = None
            last_stage: str = "crawl"
            last_cls: str = "net"

            for attempt in range(max_attempts):
                try:
                    lease = await asyncio.wait_for(
                        crawler_pool.lease(), timeout=lease_timeout
                    )
                    async with lease as crawler:
                        _touch(touch_cb, company.company_id)
                        await crawl_company(
                            company=company,
                            crawler=crawler,
                            deep_strategy=deep_strategy,
                            guard=guard,
                            gating_cfg=gating_cfg,
                            root_urls=resume_roots,
                            crawler_base_cfg=crawler_base_cfg,
                            page_policy=page_policy,
                            page_interaction_factory=page_interaction_factory,
                            page_timeout_ms=page_timeout_ms,
                            page_result_concurrency=int(
                                tuned["page_result_concurrency"]
                            ),
                            page_queue_maxsize=int(tuned["page_queue_maxsize"]),
                            url_index_flush_every=int(tuned["url_index_flush_every"]),
                            url_index_flush_interval_sec=float(
                                tuned["url_index_flush_interval_sec"]
                            ),
                            url_index_queue_maxsize=int(
                                tuned["url_index_queue_maxsize"]
                            ),
                            touch_cb=touch_cb,
                            arun_init_timeout_sec=arun_init_timeout_sec,
                            stream_no_yield_timeout_sec=stream_no_yield_timeout_sec,
                            submit_timeout_sec=submit_timeout_sec,
                        )
                    last_err = None
                    break

                except asyncio.CancelledError:
                    raise
                except CriticalMemoryPressure:
                    raise
                except (asyncio.TimeoutError, CrawlerTimeoutError) as e:
                    last_err = e
                    last_stage = getattr(e, "stage", "timeout")
                    last_cls = "stall"
                    company_logger.error(
                        "[Timeout] company_id=%s attempt=%d/%d stage=%s error=%s",
                        company.company_id,
                        attempt + 1,
                        max_attempts,
                        last_stage,
                        str(e),
                    )
                    if attempt + 1 < max_attempts:
                        await asyncio.sleep(0.5)
                        continue
                except CrawlerFatalError as e:
                    last_err = e
                    last_stage = "crawler_fatal"
                    last_cls = "net"
                    company_logger.error(
                        "[CrawlerFatalError] company_id=%s attempt=%d/%d error=%s",
                        company.company_id,
                        attempt + 1,
                        max_attempts,
                        str(e),
                    )
                    if attempt + 1 < max_attempts:
                        await asyncio.sleep(0.5)
                        continue
                except Exception as e:
                    last_err = e
                    last_stage = "unhandled_crawl"
                    last_cls = "net"
                    company_logger.exception(
                        "[UnhandledCrawlError] company_id=%s attempt=%d/%d",
                        company.company_id,
                        attempt + 1,
                        max_attempts,
                    )
                    if attempt + 1 < max_attempts:
                        await asyncio.sleep(0.5)
                        continue

            if last_err is not None:
                try:
                    retry_store.mark_failure(
                        company.company_id,
                        cls=last_cls,
                        error=str(last_err),
                        stage=last_stage,
                        status_code=getattr(last_err, "status_code", None),
                        nxdomain_like=RetryStateStore.classify_unreachable_error(
                            str(last_err)
                        ),
                    )
                except Exception:
                    pass
                return False

            await state.recompute_company_from_index(
                company.company_id, name=None, root_url=company.domain_url
            )

        # -------------------- LLM stage --------------------
        if will_run_llm:
            _touch(touch_cb, company.company_id)

            if scheduler is not None:
                try:
                    await scheduler.suspend_stall_detection(
                        key=f"llm:{company.company_id}",
                        company_id=company.company_id,
                        reset_timers=True,
                        reason="llm_extraction",
                    )
                except Exception:
                    logger.exception(
                        "[AdaptiveScheduling] failed to suspend stall detection (company_id=%s)",
                        company.company_id,
                    )

            try:
                if args.llm_mode == "presence":
                    await run_presence_pass_for_company(
                        company, presence_strategy=presence_llm
                    )
                elif args.llm_mode == "full":
                    await run_full_pass_for_company(company, full_strategy=full_llm)
            except Exception as e:
                if company_logger is not None:
                    company_logger.exception(
                        "[LLMError] company_id=%s", company.company_id
                    )
                try:
                    retry_store.mark_failure(
                        company.company_id,
                        cls="net",
                        error=str(e),
                        stage=f"llm_{args.llm_mode}",
                        nxdomain_like=RetryStateStore.classify_unreachable_error(
                            str(e)
                        ),
                    )
                except Exception:
                    pass
                return False
            finally:
                if scheduler is not None:
                    try:
                        await scheduler.resume_stall_detection(
                            key=f"llm:{company.company_id}",
                            company_id=company.company_id,
                            reset_timers=True,
                            reason="llm_extraction_done",
                        )
                    except Exception:
                        logger.exception(
                            "[AdaptiveScheduling] failed to resume stall detection (company_id=%s)",
                            company.company_id,
                        )

        completed_ok = True
        return True

    except asyncio.CancelledError as e:
        cancelled_exc = e
        raise
    except CriticalMemoryPressure:
        raise
    except Exception as e:
        logger.exception(
            "Unhandled error while processing company_id=%s", company.company_id
        )
        try:
            retry_store.mark_failure(
                company.company_id,
                cls="net",
                error=str(e),
                stage="pipeline_unhandled",
                nxdomain_like=RetryStateStore.classify_unreachable_error(str(e)),
            )
        except Exception:
            pass
        return False
    finally:

        async def _finalize() -> None:
            try:
                snap_meta = await state.get_company_snapshot(company.company_id)
                _write_crawl_meta(company, snap_meta)
            except Exception:
                logger.exception(
                    "Failed to write crawl_meta for company_id=%s", company.company_id
                )

            try:
                if run_id is not None and completed_ok:
                    await state.mark_company_completed(run_id, company.company_id)
            except Exception:
                logger.exception(
                    "Failed to update run state for company_id=%s", company.company_id
                )

            try:
                if completed_ok:
                    retry_store.mark_success(
                        company.company_id, stage="completed", note="ok"
                    )
            except Exception:
                logger.exception(
                    "Failed to mark_success for company_id=%s",
                    company.company_id,
                )

            if token is not None:
                try:
                    logging_ext.reset_company_context(token)
                    logging_ext.close_company(company.company_id)
                except Exception:
                    logger.exception(
                        "Failed to close logging context for company_id=%s",
                        company.company_id,
                    )

            try:
                gc.collect()
            except Exception:
                pass

            if enable_malloc_trim:
                _maybe_malloc_trim()

        if cancelled_exc is not None:
            await _run_cleanup_even_if_cancelled(_finalize())
        else:
            try:
                await _finalize()
            except Exception:
                logger.exception(
                    "Error during finalize (company_id=%s)", company.company_id
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep crawl corporate websites (per company pipeline)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url", type=str, help="Root domain URL to crawl (single company)."
    )
    group.add_argument(
        "--company-file",
        type=str,
        help="Path to input file or directory with company data (CSV/TSV/Excel/JSON/Parquet).",
    )

    parser.add_argument("--company-id", type=str, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["bestfirst", "bfs_internal", "dfs"],
        default="bestfirst",
    )
    parser.add_argument(
        "--llm-mode", type=str, choices=["none", "presence", "full"], default="none"
    )
    parser.add_argument(
        "--llm-provider", type=str, choices=["ollama", "api"], default="ollama"
    )
    parser.add_argument("--llm-api-provider", type=str, default="openai/gpt-4o-mini")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    parser.add_argument("--dataset-file", type=str, default=None)

    parser.add_argument(
        "--company-concurrency",
        type=int,
        default=12,
        help="Hard upper bound on concurrent companies.",
    )
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--enable-resource-monitor", action="store_true")
    parser.add_argument("--page-timeout-ms", type=int, default=30000)

    parser.add_argument(
        "--retry-mode",
        type=str,
        choices=["all", "skip-retry", "only-retry"],
        default="all",
    )
    parser.add_argument("--enable-session-log", action="store_true")

    parser.add_argument(
        "--finalize-in-progress-md",
        action="store_true",
        help="Special mode: read crawl_global_state.json and force-mark markdown completion for in_progress companies (no LLM).",
    )

    parser.add_argument("--page-result-concurrency", type=int, default=6)
    parser.add_argument("--page-queue-maxsize", type=int, default=64)
    parser.add_argument("--url-index-flush-every", type=int, default=24)
    parser.add_argument("--url-index-flush-interval-sec", type=float, default=0.5)
    parser.add_argument("--url-index-queue-maxsize", type=int, default=2048)

    parser.add_argument(
        "--crawler-pool-size",
        type=int,
        default=8,
        help="Number of independent AsyncWebCrawler instances.",
    )
    parser.add_argument(
        "--crawler-recycle-after",
        type=int,
        default=20,
        help="Recycle a crawler instance after N companies.",
    )

    # NOTE: scheduling knobs are now *used by AdaptiveScheduler*, not run.py
    parser.add_argument(
        "--max-start-per-tick",
        type=int,
        default=3,
        help="Upper bound of new company tasks started per scheduler tick (anti-burst).",
    )
    parser.add_argument(
        "--idle-recycle-interval-sec",
        type=float,
        default=25.0,
        help="Minimum interval between recycling one idle crawler slot under memory pressure.",
    )
    parser.add_argument(
        "--idle-recycle-raw-frac",
        type=float,
        default=0.88,
        help="Recycle one idle crawler if used_frac_raw >= this threshold.",
    )
    parser.add_argument(
        "--idle-recycle-eff-frac",
        type=float,
        default=0.83,
        help="Recycle one idle crawler if used_frac_eff >= this threshold.",
    )

    parser.add_argument("--crawler-lease-timeout-sec", type=float, default=120.0)
    parser.add_argument("--arun-init-timeout-sec", type=float, default=90.0)
    parser.add_argument("--stream-no-yield-timeout-sec", type=float, default=900.0)
    parser.add_argument("--submit-timeout-sec", type=float, default=30.0)

    parser.add_argument(
        "--enable-malloc-trim",
        action="store_true",
        help="Call malloc_trim(0) after each company finalize (Linux/glibc only).",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    global _forced_exit_code, _retry_store_instance

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths.OUTPUT_ROOT = out_dir  # type: ignore[attr-defined]

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    logger.setLevel(log_level)

    if getattr(args, "finalize_in_progress_md", False) and args.llm_mode != "none":
        logger.warning("--finalize-in-progress-md enabled; forcing --llm-mode none.")
        args.llm_mode = "none"

    enable_session_log = getattr(args, "enable_session_log", False)
    logging_ext = LoggingExtension(
        global_level=log_level,
        per_company_level=log_level,
        max_open_company_logs=128,
        enable_session_log=enable_session_log,
        session_log_path=(out_dir / "session.log") if enable_session_log else None,
    )

    resource_monitor: Optional[ResourceMonitor] = None
    if getattr(args, "enable_resource_monitor", False):
        rm_config = ResourceMonitorConfig()
        resource_monitor = ResourceMonitor(
            output_path=out_dir / "resource_usage.json", config=rm_config
        )
        resource_monitor.start()

    default_language_factory.set_language(args.lang)

    presence_llm = None
    full_llm = None
    if args.llm_mode != "none":
        provider_strategy = (
            default_ollama_provider_strategy
            if args.llm_provider == "ollama"
            else RemoteAPIProviderStrategy(
                provider=args.llm_api_provider, api_token=None, base_url=None
            )
        )
        llm_factory = LLMExtractionFactory(
            provider_strategy=provider_strategy,
            default_full_instruction=DEFAULT_FULL_INSTRUCTION,
            default_presence_instruction=DEFAULT_PRESENCE_INSTRUCTION,
        )
        if args.llm_mode in ("presence", "full"):
            presence_llm = llm_factory.create(mode="presence")
        if args.llm_mode == "full":
            full_llm = llm_factory.create(mode="schema")

    guard = ConnectivityGuard()
    await guard.start()

    gating_cfg = md_gating.build_gating_config(
        min_meaningful_words=30,
        cookie_max_fraction=0.02,
        require_structure=True,
        interstitial_max_share=0.70,
        interstitial_min_hits=2,
    )
    markdown_generator = default_md_factory.create(
        min_meaningful_words=gating_cfg.min_meaningful_words,
        interstitial_max_share=gating_cfg.interstitial_max_share,
        interstitial_min_hits=gating_cfg.interstitial_min_hits,
        cookie_max_fraction=gating_cfg.cookie_max_fraction,
        require_structure=gating_cfg.require_structure,
    )
    page_timeout_ms: int = int(getattr(args, "page_timeout_ms", 30000))
    crawler_base_cfg = default_crawler_factory.create(
        markdown_generator=markdown_generator, page_timeout=page_timeout_ms
    )

    page_policy = PageInteractionPolicy(
        enable_cookie_playbook=True,
        cookie_retry_attempts=4,
        cookie_retry_interval_ms=400,
        enable_anti_interstitial=False,
        wait_timeout_ms=page_timeout_ms,
        delay_before_return_sec=1.2,
        min_content_chars=800,
        max_cookie_hits=3,
        virtual_scroll=False,
    )
    page_interaction_factory = default_page_interaction_factory

    # State DB anchored to out_dir
    state = get_crawl_state(db_path=out_dir / "crawl_state.sqlite3")

    inflight_by_cid: Dict[str, asyncio.Task] = {}
    cid_by_task: Dict[asyncio.Task, str] = {}

    def get_active_company_ids() -> List[str]:
        return [cid for cid, t in inflight_by_cid.items() if not t.done()]

    crawler_pool: Optional[CrawlerPool] = None

    retry_store: Optional[RetryStateStore] = None

    async def request_recycle_idle(count: int, reason: str) -> int:
        if crawler_pool is None:
            return 0
        try:
            return await crawler_pool.recycle_idle_now(count, reason=reason)
        except Exception:
            logger.exception("[CrawlerPool] idle recycle failed")
            return 0

    def request_cancel_companies(ids: Sequence[str]) -> None:
        nonlocal retry_store

        if crawler_pool is not None and ids:
            try:
                crawler_pool.request_recycle(len(ids), reason="mem_pressure_cancel")
                asyncio.create_task(
                    crawler_pool.recycle_idle_now(
                        min(len(ids), crawler_pool.size),
                        reason="mem_pressure_idle_recycle",
                    )
                )
            except Exception:
                logger.exception("[CrawlerPool] failed to schedule forced recycle")

        for cid in ids:
            t = inflight_by_cid.get(cid)
            if t is None or t.done():
                continue
            logger.error("[AdaptiveScheduling] cancelling company_id=%s", cid)
            if retry_store is not None:
                try:
                    retry_store.mark_failure(
                        cid,
                        cls="mem",
                        error="cancelled by AdaptiveScheduler due to memory pressure/inactivity",
                        stage="scheduler_cancel",
                    )
                except Exception:
                    pass
            t.cancel()

    scheduler: Optional[AdaptiveScheduler] = None
    touch_cb: Optional[Any] = None

    stop_event = asyncio.Event()
    restart_due_to_scheduler = False

    def _on_sigterm() -> None:
        stop_event.set()
        logger.error("[Signal] SIGTERM received; requesting graceful shutdown.")

    def _on_sigint() -> None:
        stop_event.set()
        logger.warning("[Signal] SIGINT received; requesting graceful shutdown.")

    async def _cancel_inflight_and_maybe_exit_hard(
        *, reason: str, mark_timeout_like: bool, hard_exit_code: Optional[int]
    ) -> None:
        nonlocal retry_store

        if inflight_by_cid:
            logger.error("%s (inflight=%d)", reason, len(inflight_by_cid))

        if mark_timeout_like and retry_store is not None:
            for cid in list(inflight_by_cid.keys()):
                try:
                    retry_store.mark_failure(
                        cid,
                        cls="stall",
                        error=reason,
                        stage="shutdown_cancel",
                    )
                except Exception:
                    pass

        for t in list(inflight_by_cid.values()):
            if not t.done():
                t.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(*inflight_by_cid.values(), return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.critical(
                "Inflight tasks did not cancel within timeout; forcing exit."
            )
            if hard_exit_code is not None:
                os._exit(int(hard_exit_code))  # noqa: S606
        finally:
            inflight_by_cid.clear()
            cid_by_task.clear()

    try:
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGTERM, _on_sigterm)
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
        except NotImplementedError:
            signal.signal(signal.SIGTERM, lambda *_: _on_sigterm())
            signal.signal(signal.SIGINT, lambda *_: _on_sigint())
    except Exception:
        pass

    try:
        # -------------------- load companies --------------------
        if args.company_file:
            companies = _companies_from_source(Path(args.company_file))
        else:
            url = args.url
            assert url is not None
            cid = args.company_id
            if not cid:
                parsed = urlparse(url)
                cid = (parsed.netloc or parsed.path or "company").replace(":", "_")
            companies = [Company(company_id=cid, domain_url=url)]

        finalize_in_progress_md: bool = bool(
            getattr(args, "finalize_in_progress_md", False)
        )
        if finalize_in_progress_md:
            inprog = _read_in_progress_companies(out_dir)
            if not inprog:
                logger.info(
                    "--finalize-in-progress-md: no in_progress_companies; exiting."
                )
                return
            inprog_set = set(inprog)
            before = len(companies)
            companies = [c for c in companies if c.company_id in inprog_set]
            logger.info(
                "--finalize-in-progress-md: filtered companies %d -> %d (in_progress=%d).",
                before,
                len(companies),
                len(inprog_set),
            )
            if not companies:
                logger.info(
                    "--finalize-in-progress-md: none of the in_progress IDs exist in input; exiting."
                )
                return

        companies_by_id: Dict[str, Company] = {c.company_id: c for c in companies}
        company_ids_all: List[str] = [c.company_id for c in companies]

        run_id = await state.start_run(
            pipeline="deep_crawl", version=None, args_hash=None
        )
        for c in companies:
            await state.upsert_company(c.company_id, name=None, root_url=c.domain_url)

        try:
            await state.recompute_global_state()
        except Exception:
            logger.exception("Failed to recompute global crawl state at startup")

        # -------------------- IMPORTANT FIX: filter DONE companies BEFORE queuing --------------------
        try:
            runnable_ids = await state.filter_runnable_company_ids(
                company_ids_all,
                llm_mode=str(getattr(args, "llm_mode", "none")),
                refresh_pending=False,  # keep this fast for ongoing runs; DB already has status
                chunk_size=500,
                concurrency=32,
            )
        except Exception:
            logger.exception(
                "Failed to pre-filter runnable companies; falling back to full list."
            )
            runnable_ids = list(company_ids_all)

        skipped = len(company_ids_all) - len(runnable_ids)
        logger.info(
            "Worklist filtering: total=%d runnable=%d skipped_done=%d llm_mode=%s",
            len(company_ids_all),
            len(runnable_ids),
            skipped,
            args.llm_mode,
        )

        # Keep run totals as the original input list (for audit), but scheduling uses runnable only.
        await state.update_run_totals(run_id, total_companies=len(companies))

        dataset_externals: frozenset[str] = build_dataset_externals(
            args=args, companies=companies
        )

        bm25_components = build_dual_bm25_components()
        url_scorer: Optional[DualBM25Scorer] = bm25_components["url_scorer"]
        bm25_filter: Optional[DualBM25Filter] = bm25_components["url_filter"]

        dfs_factory: Optional[DeepCrawlStrategyFactory] = None
        if args.strategy == "dfs":
            dfs_factory = DeepCrawlStrategyFactory(
                provider=DFSDeepCrawlStrategyProvider(
                    default_max_depth=3,
                    default_include_external=False,
                    default_score_threshold=None,
                )
            )

        http_lang = _HTTP_LANG_MAP.get(args.lang, f"{args.lang}-US")
        browser_cfg = default_browser_factory.create(
            lang=http_lang, add_common_cookies_for=[], headless=True
        )

        crawler_pool = CrawlerPool(
            browser_cfg=browser_cfg,
            size=int(getattr(args, "crawler_pool_size", 2)),
            recycle_after_companies=int(getattr(args, "crawler_recycle_after", 40)),
        )
        await crawler_pool.start()

        # -------------------- scheduler (owns retry + scheduling) --------------------
        cc = max(1, int(getattr(args, "company_concurrency", 16)))

        scheduler_cfg = AdaptiveSchedulingConfig(
            log_path=out_dir / "adaptive_scheduling_state.jsonl",
            heartbeat_path=out_dir / "heartbeat.json",
            initial_target=min(4, cc),
            max_target=cc,
            retry_base_dir=out_dir / "_retry",
            max_start_per_tick=int(getattr(args, "max_start_per_tick", 3)),
            idle_recycle_interval_sec=float(
                getattr(args, "idle_recycle_interval_sec", 25.0)
            ),
            idle_recycle_raw_frac=float(getattr(args, "idle_recycle_raw_frac", 0.88)),
            idle_recycle_eff_frac=float(getattr(args, "idle_recycle_eff_frac", 0.83)),
        )

        scheduler = AdaptiveScheduler(
            cfg=scheduler_cfg,
            get_active_company_ids=get_active_company_ids,
            request_cancel_companies=request_cancel_companies,
            request_recycle_idle=request_recycle_idle,
        )
        await scheduler.start()

        retry_store = scheduler.retry_store
        _retry_store_instance = retry_store  # for exit-code computation in main()

        touch_cb = scheduler.touch_company

        async def is_company_runnable(cid: str) -> bool:
            # Fast check using DB snapshot, with safe fallback.
            try:
                snap = await state.get_company_snapshot(cid, recompute=False)
                st = getattr(snap, "status", None) or COMPANY_STATUS_PENDING
                return not _should_skip_company(st, args.llm_mode)
            except Exception:
                return True

        # Clean retry-state entries that are already DONE (prevents restart loops)
        try:
            cleared = await scheduler.cleanup_completed_retry_ids(
                is_company_runnable=is_company_runnable,
                treat_non_runnable_as_done=True,
                stage="startup_cleanup",
            )
            if cleared:
                logger.info(
                    "Cleaned %d completed IDs from retry_state.json before starting.",
                    cleared,
                )
        except Exception:
            logger.exception("Failed cleaning completed retry IDs at startup")

        # Seed scheduler with pre-filtered runnable worklist
        await scheduler.set_worklist(
            runnable_ids,
            retry_mode=str(getattr(args, "retry_mode", "all")),
            is_company_runnable=is_company_runnable,
        )

        if scheduler.pending_total() <= 0 and not inflight_by_cid:
            logger.info("No companies eligible to run now; exiting.")
            return

        launched = 0
        total = max(1, scheduler.initial_total_hint())

        last_global_recompute_ts = 0.0
        global_recompute_interval_sec = 120.0

        enable_malloc_trim: bool = bool(getattr(args, "enable_malloc_trim", False))

        while scheduler.has_pending() or inflight_by_cid:
            if stop_event.is_set():
                if scheduler is not None and scheduler.restart_recommended:
                    restart_due_to_scheduler = True
                await _cancel_inflight_and_maybe_exit_hard(
                    reason="[Shutdown] stop requested; cancelling inflight.",
                    mark_timeout_like=True,
                    hard_exit_code=(
                        RETRY_EXIT_CODE if restart_due_to_scheduler else None
                    ),
                )
                break

            if scheduler is not None and scheduler.restart_recommended:
                restart_due_to_scheduler = True
                await _cancel_inflight_and_maybe_exit_hard(
                    reason="[AdaptiveScheduling] restart recommended; cancelling inflight and exiting loop.",
                    mark_timeout_like=False,
                    hard_exit_code=RETRY_EXIT_CODE,
                )
                break

            free_crawlers = crawler_pool.free_slots_approx() if crawler_pool else 0
            start_ids = await scheduler.plan_start_batch(free_crawlers=free_crawlers)

            for cid in start_ids:
                company = companies_by_id.get(cid)
                if company is None:
                    continue
                launched += 1
                task = asyncio.create_task(
                    run_company_pipeline(
                        company=company,
                        idx=launched,
                        total=total,
                        logging_ext=logging_ext,
                        state=state,
                        guard=guard,
                        gating_cfg=gating_cfg,
                        crawler_pool=crawler_pool,  # type: ignore[arg-type]
                        args=args,
                        dataset_externals=dataset_externals,
                        url_scorer=url_scorer,
                        bm25_filter=bm25_filter,
                        run_id=run_id,
                        presence_llm=presence_llm,
                        full_llm=full_llm,
                        dfs_factory=dfs_factory,
                        crawler_base_cfg=crawler_base_cfg,
                        page_policy=page_policy,
                        page_interaction_factory=page_interaction_factory,
                        page_timeout_ms=page_timeout_ms,
                        enable_malloc_trim=enable_malloc_trim,
                        retry_store=retry_store,  # type: ignore[arg-type]
                        touch_cb=touch_cb,
                        scheduler=scheduler,
                    ),
                    name=f"company-{cid}",
                )
                inflight_by_cid[cid] = task
                cid_by_task[task] = cid
                _touch(touch_cb, cid)

            if not inflight_by_cid:
                if scheduler.has_pending():
                    delay = scheduler.sleep_hint_sec()
                    await asyncio.sleep(max(0.25, float(delay)))
                    continue
                break

            done, _ = await asyncio.wait(
                list(inflight_by_cid.values()),
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if done:
                for t in done:
                    cid = cid_by_task.pop(t, None)
                    if cid is not None:
                        inflight_by_cid.pop(cid, None)

                    if t.cancelled():
                        continue

                    ok = False
                    try:
                        ok = bool(t.result())
                    except CriticalMemoryPressure as e:
                        if cid and retry_store is not None:
                            try:
                                retry_store.mark_failure(
                                    cid,
                                    cls="mem",
                                    error=str(e),
                                    stage="critical_memory_pressure",
                                )
                            except Exception:
                                pass
                        ok = False
                    except Exception as e:
                        logger.exception("Company task failed (company_id=%s)", cid)
                        if cid and retry_store is not None:
                            try:
                                retry_store.mark_failure(
                                    cid,
                                    cls="net",
                                    error=str(e),
                                    stage="task_exception",
                                    nxdomain_like=RetryStateStore.classify_unreachable_error(
                                        str(e)
                                    ),
                                )
                            except Exception:
                                pass
                        ok = False

                    if ok and scheduler is not None:
                        scheduler.register_company_completed()

            now = time.time()
            if now - last_global_recompute_ts >= global_recompute_interval_sec:
                try:
                    await state.recompute_global_state()
                except Exception:
                    logger.exception(
                        "Failed to recompute global crawl state (periodic)"
                    )
                else:
                    last_global_recompute_ts = now

        try:
            await state.recompute_global_state()
        except Exception:
            logger.exception("Failed to recompute global crawl state at shutdown")

        if restart_due_to_scheduler:
            _forced_exit_code = RETRY_EXIT_CODE

    finally:
        try:
            await guard.stop()
        except Exception:
            logger.exception("Error while stopping ConnectivityGuard")

        if crawler_pool is not None:
            try:
                await crawler_pool.stop()
            except Exception:
                logger.exception("Error while stopping CrawlerPool")

        try:
            logging_ext.close()
        except Exception:
            logger.exception("Error while closing LoggingExtension")

        if resource_monitor is not None:
            try:
                resource_monitor.stop()
            except Exception:
                logger.exception("Error while stopping ResourceMonitor")

        if scheduler is not None:
            try:
                await scheduler.stop()
            except Exception:
                logger.exception("Error while stopping AdaptiveScheduler")

        try:
            gc.collect()
        except Exception:
            pass


def main(argv: Optional[Iterable[str]] = None) -> None:
    global _forced_exit_code, _retry_store_instance

    args = parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt; shutting down.")
    finally:
        exit_code = 0
        if _retry_store_instance is not None:
            exit_code = compute_retry_exit_code_from_store(
                _retry_store_instance, RETRY_EXIT_CODE
            )

        if _forced_exit_code is not None:
            exit_code = int(_forced_exit_code)

        if exit_code != 0:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
