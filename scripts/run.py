from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.deep_crawling.filters import FilterChain

# Repo config factories
from configs.language import default_language_factory
from configs.md import default_md_factory
from configs.crawler import default_crawler_factory
from configs.deep_crawl import (
    DFSDeepCrawlStrategyProvider,
    DeepCrawlStrategyFactory,
    DeepCrawlStrategyFactory as _DeepCrawlStrategyFactory,
    build_deep_strategy,
)
from configs.browser import default_browser_factory
from configs.llm import (
    LLMExtractionFactory,
    RemoteAPIProviderStrategy,
    default_ollama_provider_strategy,
    DEFAULT_FULL_INSTRUCTION,
    DEFAULT_PRESENCE_INSTRUCTION,
)

# JS / page interaction
from configs.js_injection import (
    PageInteractionPolicy,
    PageInteractionFactory,
    default_page_interaction_factory,
)

# Extensions
from extensions.connectivity_guard import ConnectivityGuard
from extensions.dual_bm25 import (
    DualBM25Filter,
    DualBM25Scorer,
    build_dual_bm25_components,
)
from extensions.filtering import (
    UniversalExternalFilter,
    HTMLContentFilter,
    LanguageAwareURLFilter,
    FirstTimeURLFilter,
)
from extensions.load_source import load_companies_from_source, CompanyInput
from extensions.logging import LoggingExtension
from extensions import md_gating
from extensions import output_paths
from extensions.output_paths import ensure_company_dirs
from extensions.crawl_state import (
    get_crawl_state,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
)
from extensions.resource_monitor import ResourceMonitor, ResourceMonitorConfig
from extensions.retry_tracker import (
    RetryTracker,
    RetryTrackerConfig,
    set_retry_tracker,
    record_company_attempt,
    mark_company_timeout,
    mark_company_memory_pressure,
    mark_company_completed,
)
from extensions.adaptive_scheduling import (
    AdaptiveSchedulingConfig,
    AdaptiveScheduler,
)
from extensions.page_pipeline import ConcurrentPageResultProcessor, UrlIndexFlushConfig
from extensions.llm_passes import (
    run_presence_pass_for_company,
    run_full_pass_for_company,
)
from extensions.dataset_external import build_dataset_externals

logger = logging.getLogger("deep_crawl_runner")

_HTTP_LANG_MAP: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
}

RETRY_EXIT_CODE = 17
_retry_tracker_instance: Optional[RetryTracker] = None


class CriticalMemoryPressure(RuntimeError):
    """Kept for compatibility: other modules may raise/import this."""

    def __init__(self, message: str, severity: str = "emergency") -> None:
        super().__init__(message)
        self.severity = severity


# ---------------------------------------------------------------------------
# Cancellation-safe cleanup (small + reliable)
# ---------------------------------------------------------------------------


async def _run_cleanup_even_if_cancelled(coro: Any) -> None:
    """
    Best-effort: ensure cleanup runs even if current task is cancelled.
    We prefer uncancel() on Py3.11+, otherwise shield and re-raise.
    """
    t = asyncio.current_task()
    uncancel = getattr(t, "uncancel", None) if t is not None else None

    if callable(uncancel):
        try:
            uncancel()
            await coro
        finally:
            try:
                t.cancel()  # preserve cancellation semantics upstream
            except Exception:
                pass
        return

    try:
        await asyncio.shield(coro)
    except asyncio.CancelledError:
        # If we can't reliably block, spawn it and re-raise.
        async def _bg() -> None:
            try:
                await coro
            except Exception:
                logger.exception("Background cleanup failed")

        asyncio.create_task(_bg(), name="bg-cleanup")
        raise


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
    return meta_dir / "crawl_meta.json"


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


def _should_skip_company(status: str, llm_mode: str) -> bool:
    if llm_mode == "none":
        return status in (COMPANY_STATUS_MD_DONE, COMPANY_STATUS_LLM_DONE)
    return status == COMPANY_STATUS_LLM_DONE


async def crawl_company(
    company: Company,
    *,
    crawler: AsyncWebCrawler,
    deep_strategy: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    timeout_error_marker: str,
    root_urls: Optional[List[str]] = None,
    crawler_base_cfg: Any = None,
    page_policy: PageInteractionPolicy,
    page_interaction_factory: PageInteractionFactory,
    page_timeout_ms: Optional[int] = None,
    page_result_concurrency: int = 8,
    page_queue_maxsize: int = 0,
    url_index_flush_every: int = 64,
    url_index_flush_interval_sec: float = 1.0,
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
        queue_maxsize=8192,
    )

    processor = ConcurrentPageResultProcessor(
        company=company,
        guard=guard,
        gating_cfg=gating_cfg,
        timeout_error_marker=timeout_error_marker,
        mark_company_timeout_cb=mark_company_timeout,
        mark_company_memory_cb=mark_company_memory_pressure,
        concurrency=max(1, int(page_result_concurrency)),
        page_queue_maxsize=int(page_queue_maxsize),
        url_index_flush_cfg=flush_cfg,
    )

    cancelled_exc: Optional[BaseException] = None
    await processor.start()

    try:
        for start_url in start_urls:
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

            results_or_gen = await crawler.arun(start_url, config=config)

            if not isinstance(results_or_gen, list):
                agen = results_or_gen
                try:
                    async for page_result in agen:
                        await processor.submit(page_result)
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

                if cancelled_exc is not None:
                    raise cancelled_exc
            else:
                for page_result in results_or_gen:
                    await processor.submit(page_result)

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
                    "Error aborting page processor on cancel (company_id=%s)",
                    company.company_id,
                )
        raise
    finally:
        try:
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


async def run_company_pipeline(
    company: Company,
    idx: int,
    total: int,
    *,
    logging_ext: LoggingExtension,
    state: Any,
    guard: ConnectivityGuard,
    gating_cfg: md_gating.MarkdownGatingConfig,
    timeout_error_marker: str,
    crawler: AsyncWebCrawler,
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
) -> bool:
    completed_ok = False
    token: Optional[Any] = None
    cancelled_exc: Optional[BaseException] = None
    finalize_in_progress_md: bool = bool(
        getattr(args, "finalize_in_progress_md", False)
    )

    try:
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
            return True

        record_company_attempt(company.company_id)

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

        if do_crawl:
            await guard.wait_until_healthy()
            await crawl_company(
                company=company,
                crawler=crawler,
                deep_strategy=deep_strategy,
                guard=guard,
                gating_cfg=gating_cfg,
                timeout_error_marker=timeout_error_marker,
                root_urls=resume_roots,
                crawler_base_cfg=crawler_base_cfg,
                page_policy=page_policy,
                page_interaction_factory=page_interaction_factory,
                page_timeout_ms=page_timeout_ms,
                page_result_concurrency=int(
                    getattr(args, "page_result_concurrency", 8)
                ),
                page_queue_maxsize=int(getattr(args, "page_queue_maxsize", 0)),
                url_index_flush_every=int(getattr(args, "url_index_flush_every", 64)),
                url_index_flush_interval_sec=float(
                    getattr(args, "url_index_flush_interval_sec", 1.0)
                ),
            )
            await state.recompute_company_from_index(
                company.company_id,
                name=None,
                root_url=company.domain_url,
            )

        if will_run_llm:
            if args.llm_mode == "presence":
                await run_presence_pass_for_company(
                    company, presence_strategy=presence_llm
                )
            elif args.llm_mode == "full":
                await run_full_pass_for_company(company, full_strategy=full_llm)

        completed_ok = True
        return True

    except asyncio.CancelledError as e:
        cancelled_exc = e
        raise
    except CriticalMemoryPressure:
        raise
    except Exception:
        logger.exception(
            "Unhandled error while processing company_id=%s", company.company_id
        )
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
                    mark_company_completed(company.company_id)
            except Exception:
                logger.exception(
                    "Failed to mark_company_completed for company_id=%s",
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
        default=20,
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
    parser.add_argument("--page-queue-maxsize", type=int, default=0)
    parser.add_argument("--url-index-flush-every", type=int, default=24)
    parser.add_argument("--url-index-flush-interval-sec", type=float, default=0.5)

    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    global _retry_tracker_instance

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths.OUTPUT_ROOT = out_dir  # type: ignore[attr-defined]

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.setLevel(log_level)

    if getattr(args, "finalize_in_progress_md", False) and args.llm_mode != "none":
        logger.warning("--finalize-in-progress-md enabled; forcing --llm-mode none.")
        args.llm_mode = "none"

    # Retry tracker
    retry_cfg = RetryTrackerConfig(out_dir=out_dir)
    retry_tracker = RetryTracker(retry_cfg)
    _retry_tracker_instance = retry_tracker
    set_retry_tracker(retry_tracker)

    retry_mode: str = getattr(args, "retry_mode", "all")
    prev_retry_ids: set[str] = retry_tracker.get_previous_retry_ids()

    page_timeout_ms: int = int(getattr(args, "page_timeout_ms", 60000))
    timeout_error_marker = f"Timeout {page_timeout_ms}ms exceeded."

    # Logging extension
    enable_session_log = getattr(args, "enable_session_log", False)
    logging_ext = LoggingExtension(
        global_level=log_level,
        per_company_level=log_level,
        max_open_company_logs=128,
        enable_session_log=enable_session_log,
        session_log_path=(out_dir / "session.log") if enable_session_log else None,
    )

    # Optional resource monitor
    resource_monitor: Optional[ResourceMonitor] = None
    if getattr(args, "enable_resource_monitor", False):
        rm_config = ResourceMonitorConfig()
        resource_monitor = ResourceMonitor(
            output_path=out_dir / "resource_usage.json",
            config=rm_config,
        )
        resource_monitor.start()

    # Language + LLM setup
    default_language_factory.set_language(args.lang)

    presence_llm = None
    full_llm = None
    if args.llm_mode != "none":
        provider_strategy = (
            default_ollama_provider_strategy
            if args.llm_provider == "ollama"
            else RemoteAPIProviderStrategy(
                provider=args.llm_api_provider,
                api_token=None,
                base_url=None,
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

    # Connectivity guard
    guard = ConnectivityGuard()
    await guard.start()

    # Markdown gating + crawler base cfg
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
    crawler_base_cfg = default_crawler_factory.create(
        markdown_generator=markdown_generator,
        page_timeout=page_timeout_ms,
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

    state = get_crawl_state()

    # Inflight tracking (simple, explicit)
    inflight_by_cid: Dict[str, asyncio.Task] = {}
    cid_by_task: Dict[asyncio.Task, str] = {}

    def get_active_company_ids() -> List[str]:
        return [cid for cid, t in inflight_by_cid.items() if not t.done()]

    def request_cancel_companies(ids: Iterable[str]) -> None:
        for cid in ids:
            t = inflight_by_cid.get(cid)
            if t is None or t.done():
                continue
            logger.error("[AdaptiveScheduling] cancelling company_id=%s", cid)
            mark_company_memory_pressure(cid)
            t.cancel()

    scheduler: Optional[AdaptiveScheduler] = None

    try:
        # ---- Load companies
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

        # finalize-in-progress-md filter
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

        # retry-mode filter (only when not finalize)
        if not finalize_in_progress_md and prev_retry_ids:
            if retry_mode == "skip-retry":
                companies = [c for c in companies if c.company_id not in prev_retry_ids]
            elif retry_mode == "only-retry":
                companies = [c for c in companies if c.company_id in prev_retry_ids]

        # ---- State init
        run_id = await state.start_run(
            pipeline="deep_crawl", version=None, args_hash=None
        )
        for c in companies:
            await state.upsert_company(c.company_id, name=None, root_url=c.domain_url)

        try:
            await state.recompute_global_state()
        except Exception:
            logger.exception("Failed to recompute global crawl state at startup")

        # skip already-done companies (unless finalize)
        if not finalize_in_progress_md:
            kept: List[Company] = []
            for c in companies:
                try:
                    snap = await state.get_company_snapshot(c.company_id)
                    status = getattr(snap, "status", None) or COMPANY_STATUS_PENDING
                except Exception:
                    kept.append(c)
                    continue
                if not _should_skip_company(status, args.llm_mode):
                    kept.append(c)
            companies = kept

        if not companies:
            logger.info(
                "No companies require work for llm_mode=%s; exiting.", args.llm_mode
            )
            return

        await state.update_run_totals(run_id, total_companies=len(companies))
        retry_tracker.set_total_companies(len(companies))

        dataset_externals: frozenset[str] = build_dataset_externals(
            args=args, companies=companies
        )

        bm25_components = build_dual_bm25_components()
        url_scorer: Optional[DualBM25Scorer] = bm25_components["url_scorer"]
        bm25_filter: Optional[DualBM25Filter] = bm25_components["url_filter"]

        # dfs factory
        dfs_factory: Optional[DeepCrawlStrategyFactory] = None
        if args.strategy == "dfs":
            dfs_factory = DeepCrawlStrategyFactory(
                provider=DFSDeepCrawlStrategyProvider(
                    default_max_depth=3,
                    default_include_external=False,
                    default_score_threshold=None,
                )
            )

        # Browser config
        http_lang = _HTTP_LANG_MAP.get(args.lang, f"{args.lang}-US")
        browser_cfg = default_browser_factory.create(
            lang=http_lang,
            add_common_cookies_for=[],
            headless=True,
        )

        # ---- Adaptive scheduler (new simple one)
        scheduler_cfg = AdaptiveSchedulingConfig(
            log_path=out_dir / "adaptive_scheduling_state.jsonl",
            heartbeat_path=out_dir / "heartbeat.json",
            initial_target=min(
                4, max(1, int(getattr(args, "company_concurrency", 20)))
            ),
            max_target=max(1, int(getattr(args, "company_concurrency", 20))),
        )
        scheduler = AdaptiveScheduler(
            cfg=scheduler_cfg,
            get_active_company_ids=get_active_company_ids,
            request_cancel_companies=request_cancel_companies,
        )
        await scheduler.start()

        # ---- Work queue
        companies_by_id: Dict[str, Company] = {c.company_id: c for c in companies}
        ready: deque[str] = deque(c.company_id for c in companies)

        launched = 0
        total = len(companies)
        hard_max = max(1, int(getattr(args, "company_concurrency", 20)))

        # Keep global recompute, but don’t overdo it.
        last_global_recompute_ts = 0.0
        global_recompute_interval_sec = 120.0

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            while ready or inflight_by_cid:
                # If scheduler says restart is needed, stop fast and let wrapper retry.
                if scheduler is not None and scheduler.restart_recommended:
                    logger.error(
                        "[AdaptiveScheduling] restart recommended; cancelling inflight and exiting loop."
                    )
                    for t in list(inflight_by_cid.values()):
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(
                        *inflight_by_cid.values(), return_exceptions=True
                    )
                    inflight_by_cid.clear()
                    cid_by_task.clear()
                    break

                # Launch: bounded by hard_max AND scheduler admission.
                active = len(inflight_by_cid)
                capacity = max(0, hard_max - active)

                if capacity > 0 and ready:
                    want = len(ready)
                    slots = capacity
                    if scheduler is not None:
                        slots = min(
                            slots, await scheduler.admissible_slots(num_waiting=want)
                        )
                    to_start = min(slots, capacity, want)

                    for _ in range(to_start):
                        cid = ready.popleft()
                        company = companies_by_id[cid]
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
                                timeout_error_marker=timeout_error_marker,
                                crawler=crawler,
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
                            ),
                            name=f"company-{cid}",
                        )
                        inflight_by_cid[cid] = task
                        cid_by_task[task] = cid

                # If nothing inflight, just wait a bit and loop (scheduler may unblock).
                if not inflight_by_cid:
                    if ready:
                        await asyncio.sleep(0.5)
                        continue
                    break

                done, _ = await asyncio.wait(
                    list(inflight_by_cid.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for t in done:
                    cid = cid_by_task.pop(t, None)
                    if cid is not None:
                        inflight_by_cid.pop(cid, None)

                    if t.cancelled():
                        continue

                    ok = False
                    try:
                        ok = bool(t.result())
                    except CriticalMemoryPressure:
                        if cid:
                            mark_company_memory_pressure(cid)
                        ok = False
                    except Exception:
                        logger.exception("Company task failed (company_id=%s)", cid)
                        ok = False

                    if ok and scheduler is not None:
                        scheduler.register_company_completed()

                # periodic global recompute (cheap sanity)
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

    finally:
        try:
            await guard.stop()
        except Exception:
            logger.exception("Error while stopping ConnectivityGuard")

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
    args = parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt; shutting down.")
    finally:
        if _retry_tracker_instance is not None:
            exit_code = _retry_tracker_instance.finalize_and_exit_code(RETRY_EXIT_CODE)
        else:
            exit_code = 0
        if exit_code != 0:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
