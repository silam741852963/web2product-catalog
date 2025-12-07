from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
    DualBM25Config,
    DualBM25Filter,
    DualBM25Scorer,
    build_dual_bm25_components,
)
from extensions.filtering import (
    UniversalExternalFilter,
    HTMLContentFilter,
    LanguageAwareURLFilter,
)
from extensions.load_source import load_companies_from_source, CompanyInput
from extensions.logging import LoggingExtension
from extensions import md_gating
from extensions import output_paths
from extensions.output_paths import ensure_company_dirs, save_stage_output
from extensions.crawl_state import (
    get_crawl_state,
    load_url_index,
    upsert_url_index_entry,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
)
from extensions.resource_monitor import ResourceMonitor, ResourceMonitorConfig
from extensions.stall_guard import (
    StallGuard,
    StallGuardConfig,
    global_stall_watchdog,
)
from extensions.memory_guard import MemoryGuard, CriticalMemoryPressure
from extensions.retry_tracker import (
    RetryTracker,
    RetryTrackerConfig,
    set_retry_tracker,
    record_company_attempt,
    mark_company_timeout,
    mark_company_stalled,
    mark_company_memory_pressure,
    mark_company_completed,
)
from extensions.adaptive_scheduling import (
    AdaptiveSchedulingConfig,
    AdaptiveScheduler,
)
from extensions.page_pipeline import process_page_result
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

# Special exit code so outer wrapper knows there are companies to retry.
RETRY_EXIT_CODE = 17

# Keep a reference to the tracker for exit code handling.
_retry_tracker_instance: Optional[RetryTracker] = None


# ---------------------------------------------------------------------------
# Models and company loading
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


# ---------------------------------------------------------------------------
# crawl_meta helpers
# ---------------------------------------------------------------------------


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
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        try:
            f.flush()
        except Exception:
            pass
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Company level crawl
# ---------------------------------------------------------------------------


async def crawl_company(
    company: Company,
    *,
    crawler: AsyncWebCrawler,
    deep_strategy: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    timeout_error_marker: str,
    stall_guard: Optional[StallGuard] = None,
    root_urls: Optional[List[str]] = None,
    crawler_base_cfg: Any = None,
    page_policy: PageInteractionPolicy,
    page_interaction_factory: PageInteractionFactory,
    page_timeout_ms: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> None:
    """
    Run a deep crawl for a single company.

    The per company page limit (if any) is enforced by the deep
    crawl strategy via its max_pages parameter.
    """
    logger.info(
        "Starting crawl for company_id=%s url=%s",
        company.company_id,
        company.domain_url,
    )

    ensure_company_dirs(company.company_id)

    start_urls: List[str] = list(root_urls) if root_urls else [company.domain_url]

    for start_url in start_urls:
        logger.info(
            "Deep crawl: company_id=%s root=%s pending_roots=%d",
            company.company_id,
            start_url,
            len(start_urls),
        )

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

        if memory_guard is not None:
            memory_guard.check_host_only(
                company_id=company.company_id,
                url=start_url,
                mark_company_memory=mark_company_memory_pressure,
            )

        # Async streaming deep crawl
        if not isinstance(results_or_gen, list):
            agen = results_or_gen
            if hasattr(agen, "__aiter__"):
                agen = agen.__aiter__()

            while True:
                if memory_guard is not None:
                    memory_guard.check_host_only(
                        company_id=company.company_id,
                        url=start_url,
                        mark_company_memory=mark_company_memory_pressure,
                    )

                try:
                    page_result = await agen.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as e:
                    logger.exception(
                        "Error fetching next deep crawl result (company=%s): %s",
                        company.company_id,
                        e,
                    )
                    aclose = getattr(agen, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception:
                            pass
                    break

                try:
                    await process_page_result(
                        page_result=page_result,
                        company=company,
                        guard=guard,
                        gating_cfg=gating_cfg,
                        timeout_error_marker=timeout_error_marker,
                        stall_guard=stall_guard,
                        memory_guard=memory_guard,
                        mark_company_timeout_cb=mark_company_timeout,
                        mark_company_memory_cb=mark_company_memory_pressure,
                    )
                except CriticalMemoryPressure:
                    raise
                except Exception as e:
                    url = getattr(page_result, "url", None)
                    logger.exception(
                        "Error processing page %s (company=%s): %s",
                        url,
                        company.company_id,
                        e,
                    )

        # Non streaming list result
        else:
            for page_result in results_or_gen:
                if memory_guard is not None:
                    memory_guard.check_host_only(
                        company_id=company.company_id,
                        url=getattr(page_result, "url", start_url),
                        mark_company_memory=mark_company_memory_pressure,
                    )
                try:
                    await process_page_result(
                        page_result=page_result,
                        company=company,
                        guard=guard,
                        gating_cfg=gating_cfg,
                        timeout_error_marker=timeout_error_marker,
                        stall_guard=stall_guard,
                        memory_guard=memory_guard,
                        mark_company_timeout_cb=mark_company_timeout,
                        mark_company_memory_cb=mark_company_memory_pressure,
                    )
                except CriticalMemoryPressure:
                    raise
                except Exception as e:
                    url = getattr(page_result, "url", None)
                    logger.exception(
                        "Error processing page %s (company=%s): %s",
                        url,
                        company.company_id,
                        e,
                    )


# ---------------------------------------------------------------------------
# BM25 and dataset helpers
# ---------------------------------------------------------------------------


def _build_filter_chain(
    company: Company,
    args: argparse.Namespace,
    dataset_externals: List[str],
    bm25_filter: Optional[DualBM25Filter],
) -> FilterChain:
    html_filter = HTMLContentFilter()
    lang_filter = LanguageAwareURLFilter(lang_code=args.lang)
    universal_filter = UniversalExternalFilter(
        dataset_externals=dataset_externals,
        name=f"UniversalExternalFilter[{company.company_id}]",
    )
    universal_filter.set_company_url(company.domain_url)

    filters = [universal_filter, html_filter, lang_filter]
    if bm25_filter is not None:
        filters.append(bm25_filter)
    return FilterChain(filters)


def _should_skip_company(status: str, llm_mode: str) -> bool:
    if llm_mode == "none":
        return status in (COMPANY_STATUS_MD_DONE, COMPANY_STATUS_LLM_DONE)
    return status == COMPANY_STATUS_LLM_DONE


# ---------------------------------------------------------------------------
# Company pipeline runner
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
    dataset_externals: List[str],
    url_scorer: Optional[DualBM25Scorer],
    bm25_filter: Optional[DualBM25Filter],
    run_id: Optional[str],
    presence_llm: Any,
    full_llm: Any,
    stall_guard: Optional[StallGuard],
    memory_guard: Optional[MemoryGuard],
    dfs_factory: Optional[DeepCrawlStrategyFactory],
    crawler_base_cfg: Any,
    page_policy: PageInteractionPolicy,
    page_interaction_factory: PageInteractionFactory,
    page_timeout_ms: Optional[int],
) -> None:
    completed_ok = False
    token: Optional[Any] = None

    try:
        try:
            snap = await state.get_company_snapshot(company.company_id)
            status = snap.status or COMPANY_STATUS_PENDING
        except Exception as e:
            logger.exception(
                "Pre check: failed to get snapshot for company_id=%s: %s",
                company.company_id,
                e,
            )
            status = COMPANY_STATUS_PENDING

        do_crawl = False
        resume_roots: Optional[List[str]] = None

        state_log_tpl: Optional[str] = None
        state_log_args: tuple[Any, ...] = ()

        if status == COMPANY_STATUS_PENDING:
            do_crawl = True
            state_log_tpl = "State=PENDING -> fresh crawl for %s"
            state_log_args = (company.company_id,)
        elif status == COMPANY_STATUS_MD_NOT_DONE:
            pending_md = await state.get_pending_urls_for_markdown(company.company_id)
            if pending_md:
                do_crawl = True
                resume_roots = pending_md
                state_log_tpl = "State=MARKDOWN_NOT_DONE -> resuming crawl for %s from %d pending URLs"
                state_log_args = (company.company_id, len(pending_md))
            else:
                state_log_tpl = "State=MARKDOWN_NOT_DONE but no pending URLs for %s; treating as no crawl"
                state_log_args = (company.company_id,)
        elif status in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            COMPANY_STATUS_LLM_NOT_DONE,
        ):
            state_log_tpl = "State=%s -> skipping crawl for %s"
            state_log_args = (status, company.company_id)
        else:
            do_crawl = True
            state_log_tpl = "State=%s (unknown) -> treating as PENDING for %s"
            state_log_args = (status, company.company_id)

        will_run_llm_presence = args.llm_mode == "presence" and presence_llm is not None
        will_run_llm_full = args.llm_mode == "full" and full_llm is not None
        will_run_llm = will_run_llm_presence or will_run_llm_full

        if not do_crawl and not will_run_llm:
            logger.info(
                "Company %s has status=%s, no crawl or LLM work needed in this run -> skipping.",
                company.company_id,
                status,
            )
            return

        if not do_crawl and args.llm_mode == "none":
            logger.info(
                "Company %s: no crawl required and llm_mode=none -> skipping.",
                company.company_id,
            )
            return

        record_company_attempt(company.company_id)

        if stall_guard is not None:
            stall_guard.record_company_start(company.company_id)

        token = logging_ext.set_company_context(company.company_id)
        company_logger = logging_ext.get_company_logger(company.company_id)

        if state_log_tpl:
            company_logger.info(state_log_tpl, *state_log_args)

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

        # Wire --max-pages into the deep crawl strategy itself.
        max_pages_for_strategy: Optional[int] = None
        if getattr(args, "max_pages", None) is not None and args.max_pages > 0:
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
                stall_guard=stall_guard,
                root_urls=resume_roots,
                crawler_base_cfg=crawler_base_cfg,
                page_policy=page_policy,
                page_interaction_factory=page_interaction_factory,
                page_timeout_ms=page_timeout_ms,
                memory_guard=memory_guard,
            )
            await state.recompute_company_from_index(
                company.company_id,
                name=None,
                root_url=company.domain_url,
            )

        if args.llm_mode == "presence" and presence_llm is not None:
            await run_presence_pass_for_company(
                company,
                presence_strategy=presence_llm,
                stall_guard=stall_guard,
            )
        elif args.llm_mode == "full" and full_llm is not None:
            await run_full_pass_for_company(
                company,
                full_strategy=full_llm,
                stall_guard=stall_guard,
            )

        completed_ok = True

    except asyncio.CancelledError:
        logger.warning(
            "Company pipeline cancelled for company_id=%s",
            company.company_id,
        )
        raise

    except CriticalMemoryPressure:
        logger.error(
            "Critical memory pressure while processing company_id=%s; propagating to top level.",
            company.company_id,
        )
        raise

    except Exception as e:
        logger.exception(
            "Unhandled error while processing company_id=%s: %s",
            company.company_id,
            e,
        )

    finally:
        try:
            snap_meta = await state.get_company_snapshot(company.company_id)
            _write_crawl_meta(company, snap_meta)
        except Exception:
            logger.exception(
                "Failed to write crawl_meta for company_id=%s",
                company.company_id,
            )

        try:
            if run_id is not None and completed_ok:
                await state.mark_company_completed(run_id, company.company_id)
                await state.recompute_global_state()
        except Exception:
            logger.exception(
                "Failed to update run/global state for company_id=%s",
                company.company_id,
            )

        if completed_ok:
            mark_company_completed(company.company_id)

        if stall_guard is not None and completed_ok:
            stall_guard.record_company_completed(company.company_id)

        if token is not None:
            logging_ext.reset_company_context(token)
            logging_ext.close_company(company.company_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep crawl corporate websites (per company pipeline)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        type=str,
        help="Root domain URL to crawl (single company).",
    )
    group.add_argument(
        "--company-file",
        type=str,
        help=(
            "Path to input file or directory with company data. "
            "Supports CSV or TSV or Excel or JSON or Parquet via extensions.load_source."
        ),
    )

    parser.add_argument(
        "--company-id",
        type=str,
        default=None,
        help="Optional ID for single company mode; defaults to hostname.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Global language code for this run (default: en).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Base output directory (default: ./outputs).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["bestfirst", "bfs_internal", "dfs"],
        default="bestfirst",
        help="Deep crawl strategy selection.",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        choices=["none", "presence", "full"],
        default="none",
        help=(
            "LLM integration mode. "
            "'presence' = presence only classification (0 or 1). "
            "'full' = full extraction to product/ (one JSON per URL)."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["ollama", "api"],
        default="ollama",
        help="LLM backend to use when --llm-mode != none: 'ollama' or 'api'.",
    )
    parser.add_argument(
        "--llm-api-provider",
        type=str,
        default="openai/gpt-4o-mini",
        help="Provider string when --llm-provider=api, for example 'openai/gpt-4o-mini'.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Root log level (default: INFO).",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help=(
            "Optional dataset file used to derive dataset externals (host whitelist) "
            "for UniversalExternalFilter."
        ),
    )
    parser.add_argument(
        "--company-concurrency",
        type=int,
        default=12,
        help=(
            "Maximum number of companies to process concurrently. "
            "Acts as a hard upper bound for adaptive scheduling."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help=(
            "Maximum number of pages to crawl per company. "
            "Enforced via the deep crawl strategy."
        ),
    )
    parser.add_argument(
        "--enable-resource-monitor",
        action="store_true",
        help=(
            "Enable lightweight resource monitoring (CPU and RAM and network and IO) "
            "for the entire run and write outputs/resource_usage.json."
        ),
    )
    parser.add_argument(
        "--resource-monitor-interval",
        type=float,
        default=2.0,
        help="Sampling interval in seconds for the resource monitor (default: 2.0).",
    )
    parser.add_argument(
        "--page-timeout-ms",
        type=int,
        default=60000,
        help=(
            "Per page timeout in milliseconds for Playwright and Crawl4AI. "
            "This value is applied to CrawlerRunConfig.page_timeout, "
            "PageInteractionPolicy.wait_timeout_ms, StallGuardConfig.page_timeout_sec "
            "and timeout error detection."
        ),
    )
    parser.add_argument(
        "--retry-mode",
        type=str,
        choices=["all", "skip-retry", "only-retry"],
        default="all",
        help=(
            "How to treat existing retry_companies.json: "
            "'all' = ignore it and consider all eligible companies; "
            "'skip-retry' = skip companies listed in retry_companies.json; "
            "'only-retry' = process only companies listed there."
        ),
    )
    parser.add_argument(
        "--enable-hard-memory-guard",
        action="store_true",
        help=(
            "Enable hard stop behavior on critical host memory usage. "
            "When enabled, MemoryGuard will cancel running company tasks and "
            "cause the run to exit with the retry exit code when host memory "
            "crosses the configured hard limit."
        ),
    )
    parser.add_argument(
        "--adaptive-disable-cpu",
        action="store_true",
        help=(
            "Disable CPU based adaptive scheduling decisions. "
            "When set, only memory (if enabled) will influence scaling."
        ),
    )
    parser.add_argument(
        "--adaptive-disable-mem",
        action="store_true",
        help=(
            "Disable memory based adaptive scheduling decisions. "
            "When set, only CPU (if enabled) will influence scaling. "
            "If both CPU and memory are disabled, concurrency will rise "
            "to --company-concurrency and stay there."
        ),
    )
    parser.add_argument(
        "--enable-session-log",
        action="store_true",
        help="Enable writing a session.log file with global events. Default: disabled.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    global _retry_tracker_instance

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths.OUTPUT_ROOT = out_dir  # type: ignore[attr-defined]

    # Retry tracker
    retry_cfg = RetryTrackerConfig(out_dir=out_dir)
    retry_tracker = RetryTracker(retry_cfg)
    _retry_tracker_instance = retry_tracker
    set_retry_tracker(retry_tracker)

    retry_company_mode: str = getattr(args, "retry_mode", "all")
    prev_retry_ids: set[str] = retry_tracker.get_previous_retry_ids()

    if prev_retry_ids:
        logger.info(
            "Loaded %d companies from existing retry_companies.json (mode=%s)",
            len(prev_retry_ids),
            retry_company_mode,
        )
    else:
        logger.info(
            "No prior retry companies found (mode=%s)",
            retry_company_mode,
        )

    page_timeout_ms: int = int(getattr(args, "page_timeout_ms", 60000))
    timeout_error_marker = f"Timeout {page_timeout_ms}ms exceeded."

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    enable_session_log = getattr(args, "enable_session_log", False)

    logging_ext = LoggingExtension(
        global_level=log_level,
        per_company_level=log_level,
        max_open_company_logs=128,
        enable_session_log=enable_session_log,
        session_log_path=(out_dir / "session.log") if enable_session_log else None,
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.setLevel(log_level)

    resource_monitor: Optional[ResourceMonitor] = None
    if getattr(args, "enable_resource_monitor", False):
        rm_config = ResourceMonitorConfig()
        resource_monitor = ResourceMonitor(
            output_path=out_dir / "resource_usage.json",
            config=rm_config,
        )
        resource_monitor.start()

    default_language_factory.set_language(args.lang)

    presence_llm = None
    full_llm = None
    if args.llm_mode != "none":
        if args.llm_provider == "ollama":
            provider_strategy = default_ollama_provider_strategy
        else:
            provider_strategy = RemoteAPIProviderStrategy(
                provider=args.llm_api_provider,
                api_token=None,
                base_url=None,
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
    logger.info("ConnectivityGuard started with targets=%s", guard.targets)

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

    stall_cfg = StallGuardConfig(
        page_timeout_sec=page_timeout_ms / 1000.0,
    )
    stall_guard = StallGuard(config=stall_cfg)

    memory_guard: Optional[MemoryGuard] = None
    company_tasks: Dict[str, asyncio.Task] = {}

    if getattr(args, "enable_hard_memory_guard", False):
        memory_guard = MemoryGuard()
        logger.info(
            "Memory guard enabled with built in thresholds "
            "(soft kills company, hard kills company and clamps concurrency, "
            "emergency cancels everything and exits with retry code)."
        )
    else:
        logger.info(
            "Memory guard disabled (process will not auto stop on host memory; "
            "rely on OS or container limits)."
        )

    async def _handle_stall(snapshot: Any) -> None:
        company_id = getattr(snapshot, "company_id", None)
        idle_seconds = getattr(snapshot, "idle_seconds", None)

        if not company_id:
            logger.error(
                "StallGuard on_stall called without company_id in snapshot: %r",
                snapshot,
            )
            return

        mark_company_stalled(company_id)

        task = company_tasks.get(company_id)
        if task is None:
            logger.debug(
                "StallGuard: stall detected for company_id=%s but no active "
                "task found to cancel (likely already finished).",
                company_id,
            )
            return

        if task.done():
            logger.info(
                "StallGuard: stall detected for company_id=%s but task already finished",
                company_id,
            )
            return

        logger.error(
            "StallGuard: cancelling stalled company task company_id=%s idle=%.1fs",
            company_id,
            idle_seconds if idle_seconds is not None else -1.0,
        )
        task.cancel()

    if hasattr(stall_guard, "on_stall"):
        stall_guard.on_stall = _handle_stall  # type: ignore[attr-defined]
    await stall_guard.start()

    state = get_crawl_state()

    scheduler: Optional[AdaptiveScheduler] = None
    global_stall_task: Optional[asyncio.Task] = None

    try:
        # Load companies
        if args.company_file:
            companies: List[Company] = _companies_from_source(Path(args.company_file))
        else:
            url = args.url
            assert url is not None
            cid = args.company_id
            if not cid:
                parsed = urlparse(url)
                cid = (parsed.netloc or parsed.path or "company").replace(":", "_")
            companies = [Company(company_id=cid, domain_url=url)]

        total_input = len(companies)

        # Apply retry filter
        if retry_company_mode == "skip-retry" and prev_retry_ids:
            before = len(companies)
            companies = [c for c in companies if c.company_id not in prev_retry_ids]
            logger.info(
                "Retry mode=skip-retry: filtered companies from %d to %d using %d retry ids",
                before,
                len(companies),
                len(prev_retry_ids),
            )
        elif retry_company_mode == "only-retry" and prev_retry_ids:
            before = len(companies)
            companies = [c for c in companies if c.company_id in prev_retry_ids]
            logger.info(
                "Retry mode=only-retry: filtered companies from %d to %d using %d retry ids",
                before,
                len(companies),
                len(prev_retry_ids),
            )
        else:
            logger.info(
                "Retry mode=%s: no filtering applied to %d input companies",
                retry_company_mode,
                total_input,
            )

        run_id = await state.start_run(
            pipeline="deep_crawl",
            version=None,
            args_hash=None,
        )

        for c in companies:
            await state.upsert_company(
                c.company_id,
                name=None,
                root_url=c.domain_url,
            )

        try:
            await state.recompute_global_state()
        except Exception:
            logger.exception("Failed to recompute global crawl state at startup")

        # Pre check statuses and skip already completed companies.
        companies_to_run: List[Company] = []
        skipped_companies: List[Company] = []

        for c in companies:
            try:
                snap = await state.get_company_snapshot(c.company_id)
                status = getattr(snap, "status", None) or COMPANY_STATUS_PENDING
            except Exception as e:
                logger.exception(
                    "Pre check: failed to get snapshot for company_id=%s: %s",
                    c.company_id,
                    e,
                )
                companies_to_run.append(c)
                continue

            if _should_skip_company(status, args.llm_mode):
                skipped_companies.append(c)
            else:
                companies_to_run.append(c)

        if skipped_companies:
            logger.info(
                "Pre check: %d/%d companies already completed for llm_mode=%s; %d companies remain to process",
                len(skipped_companies),
                len(companies),
                args.llm_mode,
                len(companies_to_run),
            )
        else:
            logger.info(
                "Pre check: all %d companies require processing for llm_mode=%s",
                len(companies),
                args.llm_mode,
            )

        companies = companies_to_run
        total = len(companies)
        await state.update_run_totals(run_id, total_companies=total)
        retry_tracker.set_total_companies(total)

        if not companies:
            logger.info(
                "No companies require crawl or LLM work for llm_mode=%s; exiting run.",
                args.llm_mode,
            )
            return

        dataset_externals: List[str] = build_dataset_externals(
            companies=companies,
            dataset_file=args.dataset_file,
        )

        bm25_components = build_dual_bm25_components()
        url_scorer: Optional[DualBM25Scorer] = bm25_components["url_scorer"]
        bm25_filter: Optional[DualBM25Filter] = bm25_components["url_filter"]

        max_companies = max(1, int(args.company_concurrency))

        sched_cfg = AdaptiveSchedulingConfig(
            max_concurrency=max_companies,
            use_cpu=not getattr(args, "adaptive_disable_cpu", False),
            use_mem=not getattr(args, "adaptive_disable_mem", False),
        )
        scheduler = AdaptiveScheduler(cfg=sched_cfg)
        await scheduler.start()

        # DFS factory only needed when selected.
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
            lang=http_lang,
            add_common_cookies_for=[],
            headless=True,
        )

        abort_run = False

        # Configure memory guard callback if enabled.
        if memory_guard is not None:
            cfg = memory_guard.config

            def _on_critical_memory(company_id: str, severity: str) -> None:
                nonlocal abort_run

                if severity == "soft":
                    logger.warning(
                        "MemoryGuard soft limit event for company_id=%s; "
                        "clamping concurrency.",
                        company_id,
                    )
                elif severity == "hard":
                    logger.error(
                        "MemoryGuard hard limit hit by company_id=%s; "
                        "tightening concurrency.",
                        company_id,
                    )
                else:
                    logger.critical(
                        "MemoryGuard emergency limit hit by company_id=%s; "
                        "cancelling all tasks and aborting run.",
                        company_id,
                    )
                    abort_run = True

                if scheduler is not None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None:
                        loop.create_task(
                            scheduler.on_memory_pressure(severity=severity)
                        )

                if severity == "emergency":
                    for cid, t in list(company_tasks.items()):
                        if not t.done():
                            logger.error(
                                "MemoryGuard: emergency cancel company_id=%s",
                                cid,
                            )
                            mark_company_memory_pressure(cid)
                            t.cancel()

            cfg.on_critical = _on_critical_memory

        def _on_global_stall(idle: float) -> None:
            nonlocal abort_run
            abort_run = True

        total_companies = len(companies)

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            global_stall_task = asyncio.create_task(
                global_stall_watchdog(
                    stall_guard=stall_guard,
                    stall_cfg=stall_cfg,
                    company_tasks=company_tasks,
                    mark_company_stalled=mark_company_stalled,
                    on_global_stall=_on_global_stall,
                ),
                name="global-stall-watchdog",
            )

            inflight: set[asyncio.Task] = set()
            next_idx = 0

            while not abort_run and (next_idx < total_companies or inflight):
                active_tasks = len(inflight)
                started = False

                if scheduler is not None:
                    can_start, reason = await scheduler.can_start_new_company(
                        active_tasks=active_tasks
                    )
                else:
                    can_start = active_tasks < max_companies
                    reason = "no_scheduler"

                if can_start and next_idx < total_companies and not abort_run:
                    company = companies[next_idx]
                    next_idx += 1
                    idx_for_log = next_idx

                    async def _run_company(
                        company: Company = company,
                        offset: int = idx_for_log,
                    ) -> None:
                        await run_company_pipeline(
                            company=company,
                            idx=offset,
                            total=total_companies,
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
                            stall_guard=stall_guard,
                            memory_guard=memory_guard,
                            dfs_factory=dfs_factory,
                            crawler_base_cfg=crawler_base_cfg,
                            page_policy=page_policy,
                            page_interaction_factory=page_interaction_factory,
                            page_timeout_ms=page_timeout_ms,
                        )

                    logger.info(
                        "[Scheduler] starting company company_id=%s (idx=%d/%d, active_before=%d, reason=%s)",
                        company.company_id,
                        idx_for_log,
                        total_companies,
                        active_tasks,
                        reason,
                    )

                    task = asyncio.create_task(
                        _run_company(),
                        name=f"company-{company.company_id}",
                    )
                    company_tasks[company.company_id] = task
                    inflight.add(task)
                    if scheduler is not None:
                        scheduler.notify_admitted(
                            company_id=company.company_id,
                            reason=reason,
                        )
                    started = True
                else:
                    if not started and not abort_run and next_idx < total_companies:
                        logger.info(
                            "[Scheduler] not starting new company (active=%d, reason=%s, next_idx=%d/%d)",
                            active_tasks,
                            reason,
                            next_idx,
                            total_companies,
                        )

                if not inflight and next_idx >= total_companies:
                    break

                if inflight:
                    timeout = (
                        scheduler.sample_interval_sec if scheduler is not None else None
                    )

                    done, pending = await asyncio.wait(
                        inflight,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=timeout,
                    )
                    inflight = pending

                    for t in done:
                        for cid, ct in list(company_tasks.items()):
                            if ct is t:
                                company_tasks.pop(cid, None)
                                break

                        if t.cancelled():
                            continue

                        try:
                            exc = t.exception()
                        except asyncio.CancelledError:
                            continue

                        if isinstance(exc, CriticalMemoryPressure):
                            if not getattr(args, "enable_hard_memory_guard", False):
                                continue

                            severity = getattr(exc, "severity", "emergency")
                            if severity == "emergency":
                                logger.error(
                                    "Emergency memory pressure reported by one company; "
                                    "aborting remaining run so the wrapper can restart.",
                                )
                                abort_run = True
                                break
                            else:
                                logger.warning(
                                    "Non emergency memory pressure reported (severity=%s); "
                                    "run will continue with the same scheduling policy.",
                                    severity,
                                )
                    if abort_run:
                        break
                else:
                    if scheduler is not None:
                        await asyncio.sleep(scheduler.sample_interval_sec)
                    else:
                        await asyncio.sleep(0.5)

            if abort_run:
                for t in inflight:
                    if not t.done():
                        t.cancel()
                inflight.clear()

            if global_stall_task is not None:
                global_stall_task.cancel()
                try:
                    await global_stall_task
                except asyncio.CancelledError:
                    pass
                global_stall_task = None

    finally:
        if global_stall_task is not None:
            global_stall_task.cancel()
            try:
                await global_stall_task
            except asyncio.CancelledError:
                pass

        try:
            await guard.stop()
        except Exception:
            logger.exception("Error while stopping ConnectivityGuard")

        try:
            await stall_guard.stop()
        except Exception:
            logger.exception("Error while stopping StallGuard")

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


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt (Ctrl+C); shutting down gracefully.")
    finally:
        if _retry_tracker_instance is not None:
            exit_code = _retry_tracker_instance.finalize_and_exit_code(RETRY_EXIT_CODE)
        else:
            exit_code = 0

        if exit_code != 0:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
