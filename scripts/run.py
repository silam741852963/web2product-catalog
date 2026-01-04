from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
import signal
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence
from urllib.parse import urlparse

from crawl4ai import CacheMode

# Repo config factories
from configs.browser import default_browser_factory
from configs.crawler import default_crawler_factory
from configs.deep_crawl import (
    DFSDeepCrawlStrategyProvider,
    DeepCrawlStrategyFactory,
    DeepCrawlStrategyFactory as DeepCrawlStrategyFactoryType,
    build_deep_strategy,
)
from configs.js_injection import (
    PageInteractionFactory,
    PageInteractionPolicy,
    default_page_interaction_factory,
)
from configs.language import default_language_factory
from configs.llm import (
    BASE_FULL_INSTRUCTION,
    BASE_PRESENCE_INSTRUCTION,
    IndustryAwareStrategyCache,
    LLMExtractionFactory,
    provider_strategy_from_llm_model_selector,
)
from configs.md import default_md_factory
from configs.models import (
    Company,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_TERMINAL_DONE,
)

# Extensions
from extensions.filter import md_gating
from extensions.filter.core import build_filter_chain
from extensions.filter.dataset_external import build_dataset_externals
from extensions.filter.dual_bm25 import (
    DualBM25Filter,
    DualBM25Scorer,
    build_dual_bm25_components,
)
from extensions.guard.connectivity import ConnectivityGuard
from extensions.io import output_paths
from extensions.io.load_source import (
    IndustryEnrichmentConfig,
    load_companies_from_source_with_industry,
)
from extensions.schedule import retry as retry_mod
from extensions.schedule.adaptive import (
    AdaptiveScheduler,
    AdaptiveSchedulingConfig,
    compute_retry_exit_code_from_store,
)
from extensions.schedule.crawler_pool import CrawlerPool
from extensions.schedule.terminalization import (
    decide_from_page_summary,
    safe_mark_terminal,
    stall_sanity_override_if_crawl_finished,
    terminal_sanity_gate_if_crawl_finished,
)
from extensions.utils.logging import LoggingExtension
from extensions.utils.resource_monitor import ResourceMonitor, ResourceMonitorConfig

from extensions.crawl.recrawl_policy import apply_recrawl_policy
from extensions.crawl.runner import CrawlRunnerConfig, run_company_crawl
from extensions.crawl.state import (
    CrawlState,
    crawl_runner_done_ok,
    get_crawl_state,
    log_snapshot,
    md_ready_for_llm,
    snap_last_error,
    urls_md_done,
)

from extensions.pipeline.llm_passes import (
    build_industry_context,
    has_industry_context,
    llm_requested,
    run_full_pass_for_company,
    run_presence_pass_for_company,
)

logger = logging.getLogger("deep_crawl_runner")

RETRY_EXIT_CODE = 17
_forced_exit_code: Optional[int] = None
_retry_store_instance: Optional[retry_mod.RetryStateStore] = None

_HTTP_LANG_MAP: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
}

_CACHE_MODE_MAP: Dict[str, CacheMode] = {
    "enabled": CacheMode.ENABLED,
    "disabled": CacheMode.DISABLED,
    "read_only": CacheMode.READ_ONLY,
    "write_only": CacheMode.WRITE_ONLY,
    "bypass": CacheMode.BYPASS,
}


async def run_company_pipeline(
    company: Company,
    *,
    attempt_no: int,
    total_unique: int,
    done_counter: MutableMapping[str, int],
    logging_ext: LoggingExtension,
    state: CrawlState,
    guard: ConnectivityGuard,
    crawler_pool: CrawlerPool,
    args: argparse.Namespace,
    dataset_externals: frozenset[str],
    url_scorer: Optional[DualBM25Scorer],
    bm25_filter: Optional[DualBM25Filter],
    run_id: Optional[str],
    industry_llm_cache: Optional[IndustryAwareStrategyCache],
    dfs_factory: Optional[DeepCrawlStrategyFactoryType],
    crawler_base_cfg: Any,
    page_policy: PageInteractionPolicy,
    page_interaction_factory: PageInteractionFactory,
    retry_store: retry_mod.RetryStateStore,
    scheduler: AdaptiveScheduler,
    stop_event: asyncio.Event,
    llm_sem: asyncio.Semaphore,
    repo_root: Path,
) -> bool:
    token = logging_ext.set_company_context(company.company_id)
    clog = logging_ext.get_company_logger(company.company_id)

    progress_throttle_sec = float(args.company_progress_throttle_sec)
    last_progress_mono = 0.0

    def signal_progress(kind: str = "progress") -> None:
        nonlocal last_progress_mono
        now = time.monotonic()
        if kind == "progress" and (now - last_progress_mono) < progress_throttle_sec:
            return
        last_progress_mono = now
        if kind == "heartbeat":
            scheduler.heartbeat_company(company.company_id)
        else:
            scheduler.progress_company(company.company_id)

    async def _get_md_done(*, recompute: bool) -> int:
        snapx = await state.get_company_snapshot(
            company.company_id, recompute=recompute
        )
        log_snapshot(clog, label=f"_get_md_done(recompute={recompute})", snap=snapx)
        return int(snapx.urls_markdown_done)

    async def _persist_last_error(err: str) -> None:
        await state.upsert_company(
            company.company_id,
            last_error=(err or "")[:4000],
            name=company.name,
            root_url=company.domain_url,
        )

    outcome = retry_mod.AttemptOutcome(
        ok=False,
        stage="init",
        event=None,
        md_done=None,
        terminalized=False,
        terminal_reason=None,
        terminal_last_error=None,
        should_mark_success=False,
    )

    crawl_watchdog_task: Optional[asyncio.Task] = None
    llm_watchdog_task: Optional[asyncio.Task] = None

    try:
        await scheduler.set_company_stage(
            company.company_id,
            "crawl",
            reset_timers=True,
            reason="pipeline_enter",
        )

        await state.upsert_company(
            company.company_id,
            name=company.name,
            root_url=company.domain_url,
            industry_label=company.industry_label,
            industry=company.industry,
            nace=company.nace,
            industry_source=company.industry_source,
            write_meta=True,
        )

        snap = await state.get_company_snapshot(company.company_id, recompute=False)
        log_snapshot(clog, label="pipeline_enter(recompute=False)", snap=snap)

        status = snap.status or COMPANY_STATUS_PENDING
        md_done_enter = int(snap.urls_markdown_done)
        clog.debug(
            "pipeline_enter parsed snapshot company_id=%s status=%s urls_md_done=%d",
            company.company_id,
            status,
            md_done_enter,
        )

        do_llm_requested = llm_requested(args, industry_llm_cache)
        do_crawl = status in (COMPANY_STATUS_PENDING, COMPANY_STATUS_MD_NOT_DONE)

        if (not do_crawl) and (not do_llm_requested):
            outcome.ok = True
            outcome.stage = "skip_already_done_or_llm_disabled"
            outcome.should_mark_success = True
            await retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            return True

        done_now = int(done_counter.get("done", 0))
        clog.info(
            "=== [done=%d/%d attempt=%d] company_id=%s url=%s industry_label=%s industry=%s nace=%s status=%s llm=%s llm_requested=%s has_ctx=%s ===",
            done_now,
            total_unique,
            attempt_no,
            company.company_id,
            company.domain_url,
            company.industry_label,
            company.industry,
            company.nace,
            status,
            args.llm_mode,
            do_llm_requested,
            has_industry_context(company),
        )

        filter_chain = build_filter_chain(
            company_url=company.domain_url,
            company_id=company.company_id,
            lang=args.lang,
            dataset_externals=dataset_externals,
            bm25_filter=bm25_filter,
        )
        deep_strategy = build_deep_strategy(
            strategy=args.strategy,
            filter_chain=filter_chain,
            url_scorer=url_scorer,
            dfs_factory=dfs_factory,
            max_pages=int(args.max_pages) if args.max_pages else None,
        )

        if do_crawl:
            await scheduler.set_company_stage(
                company.company_id,
                "crawl",
                reset_timers=True,
                reason="enter_crawl",
            )
            await guard.wait_until_healthy()

            resume_roots: Optional[List[str]] = None
            direct_fetch_urls = False
            if status == COMPANY_STATUS_MD_NOT_DONE:
                pending_md = await state.get_pending_urls_for_markdown(
                    company.company_id
                )
                if pending_md:
                    resume_roots = pending_md
                    direct_fetch_urls = args.resume_md_mode == "direct"

            cache_mode = _CACHE_MODE_MAP[str(args.crawl4ai_cache_mode)]

            runner_cfg = CrawlRunnerConfig(
                page_result_concurrency=int(args.page_result_concurrency),
                page_queue_maxsize=int(args.page_queue_maxsize),
                url_index_queue_maxsize=int(args.url_index_queue_maxsize),
                arun_init_timeout_sec=float(args.arun_init_timeout_sec),
                stream_no_yield_timeout_sec=float(args.stream_no_yield_timeout_sec),
                submit_timeout_sec=float(args.submit_timeout_sec),
                direct_fetch_total_timeout_sec=float(args.direct_fetch_url_timeout_sec),
                processor_finish_timeout_sec=float(args.processor_finish_timeout_sec),
                generator_close_timeout_sec=float(args.generator_close_timeout_sec),
                hard_max_pages=int(args.max_pages) if args.max_pages else None,
                page_timeout_ms=int(args.page_timeout_ms)
                if args.page_timeout_ms
                else None,
                direct_fetch_urls=bool(direct_fetch_urls),
                crawl4ai_cache_mode=cache_mode,
            )

            heartbeat_sec = float(args.company_progress_heartbeat_sec)

            async def _crawl_watchdog() -> None:
                while True:
                    await asyncio.sleep(heartbeat_sec)
                    signal_progress("heartbeat")

            crawl_watchdog_task = asyncio.create_task(
                _crawl_watchdog(), name=f"watchdog:crawl:{company.company_id}"
            )

            last_exc: Optional[BaseException] = None
            last_event: Optional[retry_mod.RetryEvent] = None
            last_page_dec: Optional[Any] = None
            last_term_stage: str = "crawl"

            for attempt_index in range(2):
                try:
                    lease = await asyncio.wait_for(
                        crawler_pool.lease(),
                        timeout=float(args.crawler_lease_timeout_sec),
                    )

                    async with lease as crawler:

                        async def _do() -> Any:
                            return await run_company_crawl(
                                company=company,
                                crawler=crawler,
                                deep_strategy=deep_strategy,
                                guard=guard,
                                gating_cfg=md_gating.build_gating_config(),
                                crawler_base_cfg=crawler_base_cfg,
                                page_policy=page_policy,
                                page_interaction_factory=page_interaction_factory,
                                root_urls=resume_roots,
                                cfg=runner_cfg,
                                on_progress=lambda: signal_progress("progress"),
                            )

                        summary = await asyncio.wait_for(
                            _do(),
                            timeout=float(args.company_crawl_timeout_sec),
                        )

                        last_page_dec = decide_from_page_summary(summary)

                        if getattr(last_page_dec, "action", None) == "mem":
                            lease.mark_fatal("page_pipeline_mem")
                            raise retry_mod.CriticalMemoryPressure(
                                str(getattr(last_page_dec, "reason", "") or "mem"),
                                severity="critical",
                            )

                        _ = await stall_sanity_override_if_crawl_finished(
                            state, company, last_page_dec, clog
                        )

                        term_dec = retry_mod.decide_terminalization(
                            page_summary_decision=last_page_dec,
                            exception=None,
                            stage="crawl",
                            urls_md_done=md_done_enter,
                            attempt_index=attempt_index,
                        )

                        if term_dec.should_terminalize and last_page_dec is not None:
                            ignore_term = await terminal_sanity_gate_if_crawl_finished(
                                state, company, term_dec, last_page_dec, clog
                            )
                            if not ignore_term:
                                outcome.ok = True
                                outcome.stage = "terminalize"
                                outcome.terminalized = True
                                outcome.terminal_reason = term_dec.reason
                                outcome.terminal_last_error = term_dec.last_error
                                outcome.should_mark_success = False
                                last_term_stage = "crawl"
                                break

                        if getattr(last_page_dec, "action", None) == "stall":
                            raise retry_mod.CrawlerTimeoutError(
                                str(getattr(last_page_dec, "reason", "") or "stall"),
                                stage="page_pipeline_timeout_dominance",
                                company_id=company.company_id,
                                url=company.domain_url,
                            )

                    guard.record_success()
                    last_exc = None
                    last_event = None
                    break

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    last_exc = e
                    stage = str(getattr(e, "stage", "crawl") or "crawl")
                    if stage == "crawl" and "goto" in str(e).lower():
                        stage = "goto"

                    last_event = retry_mod.classify_failure(e, stage=stage)

                    if last_event.cls == "net":
                        guard.record_transport_error()

                    if retry_mod.should_fail_fast_on_goto(e, stage=stage):
                        term_dec = retry_mod.decide_terminalization(
                            page_summary_decision=None,
                            exception=e,
                            stage=stage,
                            urls_md_done=md_done_enter,
                            attempt_index=attempt_index,
                        )
                        if term_dec.should_terminalize:
                            outcome.ok = True
                            outcome.stage = "terminalize_fail_fast"
                            outcome.terminalized = True
                            outcome.terminal_reason = term_dec.reason
                            outcome.terminal_last_error = term_dec.last_error
                            outcome.should_mark_success = False
                            last_term_stage = stage
                            break

                    retry_mod.log_attempt_failure(
                        clog,
                        prefix="Crawl attempt failed",
                        attempt_index=attempt_index,
                        stage=stage,
                        event=last_event,
                        exc=e,
                        traceback_enabled=retry_mod.TRACEBACK_ENABLED,
                    )

                    if attempt_index == 0:
                        await asyncio.sleep(0.5)

            if outcome.terminalized:
                wrote = await safe_mark_terminal(
                    state,
                    company,
                    reason=str(outcome.stage or "terminalize"),
                    details={
                        "reason": outcome.terminal_reason,
                        "industry_label": company.industry_label,
                        "industry": company.industry,
                        "nace": company.nace,
                        "term_stage": last_term_stage,
                    },
                    last_error=outcome.terminal_last_error
                    or outcome.terminal_reason
                    or str(outcome.stage or "terminalize"),
                    stage=last_term_stage,
                    logger=clog,
                )

                if wrote:
                    outcome.md_done = await _get_md_done(recompute=True)
                    await retry_mod.record_attempt(
                        retry_store, company.company_id, outcome, flush=True
                    )
                    return True

                clog.info(
                    "safe_terminal refused; continuing pipeline as non-terminal company=%s stage=%s",
                    company.company_id,
                    last_term_stage,
                )
                outcome.terminalized = False
                outcome.terminal_reason = None
                outcome.terminal_last_error = None
                outcome.stage = "terminalize_refused_crawl_finished_ok"
                last_exc = None
                last_event = None

            if last_exc is not None and last_event is not None:
                outcome.ok = False
                outcome.stage = last_event.stage
                outcome.event = last_event
                outcome.md_done = await _get_md_done(recompute=True)
                await retry_mod.record_attempt(
                    retry_store, company.company_id, outcome, flush=True
                )
                await _persist_last_error(str(last_exc))
                return False

            await state.recompute_company_from_index(
                company.company_id, name=company.name, root_url=company.domain_url
            )

            snap_after_crawl = await state.get_company_snapshot(
                company.company_id, recompute=True
            )
            log_snapshot(
                clog, label="after_crawl(recompute=True)", snap=snap_after_crawl
            )

            if crawl_runner_done_ok(snap_after_crawl):
                with suppress(Exception):
                    await state.upsert_company(
                        company.company_id,
                        status=COMPANY_STATUS_MD_DONE,
                        last_error=None,
                        name=company.name,
                        root_url=company.domain_url,
                    )
            else:
                msg = snap_last_error(snap_after_crawl) or "crawl_not_finished_or_error"
                await _persist_last_error(msg)
                outcome.ok = False
                outcome.stage = "crawl_not_done"
                outcome.event = retry_mod.classify_failure(
                    RuntimeError(msg), stage="crawl_not_done"
                )
                outcome.md_done = urls_md_done(snap_after_crawl)
                await retry_mod.record_attempt(
                    retry_store, company.company_id, outcome, flush=True
                )
                return False

        if do_llm_requested:
            assert industry_llm_cache is not None

            snap_md = await state.get_company_snapshot(
                company.company_id, recompute=True
            )
            log_snapshot(clog, label="pre_llm(recompute=True)", snap=snap_md)

            if not md_ready_for_llm(snap_md):
                await _persist_last_error("llm_requested_but_crawl_not_done_ok")
                outcome.ok = False
                outcome.stage = "precheck_llm_crawl_not_done_ok"
                outcome.event = retry_mod.classify_failure(
                    RuntimeError("llm_requested_but_crawl_not_done_ok"),
                    stage="precheck_llm_crawl_not_done_ok",
                )
                outcome.md_done = urls_md_done(snap_md)
                await retry_mod.record_attempt(
                    retry_store, company.company_id, outcome, flush=True
                )
                return False

            await scheduler.set_company_stage(
                company.company_id,
                "llm",
                reset_timers=True,
                reason="enter_llm",
            )

            heartbeat_sec = float(args.company_progress_heartbeat_sec)

            async def _llm_watchdog() -> None:
                while True:
                    await asyncio.sleep(heartbeat_sec)
                    signal_progress("heartbeat")

            llm_watchdog_task = asyncio.create_task(
                _llm_watchdog(), name=f"watchdog:llm:{company.company_id}"
            )

            llm_exc: Optional[BaseException] = None
            llm_event: Optional[retry_mod.RetryEvent] = None

            llm_stage = f"llm_{args.llm_mode}"
            for attempt_index in range(2):
                try:
                    ctx = build_industry_context(company)

                    async with llm_sem:
                        if args.llm_mode == "presence":
                            strat = industry_llm_cache.get_strategy(
                                mode="presence", ctx=ctx
                            )
                            await run_presence_pass_for_company(
                                company,
                                presence_strategy=strat,
                                repo_root=repo_root,
                            )
                        elif args.llm_mode == "full":
                            strat = industry_llm_cache.get_strategy(
                                mode="schema", ctx=ctx
                            )
                            await run_full_pass_for_company(
                                company,
                                full_strategy=strat,
                                repo_root=repo_root,
                            )

                    signal_progress("progress")
                    llm_exc = None
                    llm_event = None
                    break

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    llm_exc = e
                    llm_event = retry_mod.classify_failure(e, stage=llm_stage)

                    retry_mod.log_attempt_failure(
                        clog,
                        prefix="LLM attempt failed",
                        attempt_index=attempt_index,
                        stage=llm_stage,
                        event=llm_event,
                        exc=e,
                        traceback_enabled=retry_mod.TRACEBACK_ENABLED,
                    )

                    if attempt_index == 0:
                        await asyncio.sleep(0.75)

            if llm_exc is not None and llm_event is not None:
                outcome.ok = False
                outcome.stage = llm_event.stage
                outcome.event = llm_event
                outcome.md_done = await _get_md_done(recompute=True)
                await retry_mod.record_attempt(
                    retry_store, company.company_id, outcome, flush=True
                )
                await _persist_last_error(str(llm_exc))
                return False

            with suppress(Exception):
                await state.upsert_company(
                    company.company_id,
                    status=COMPANY_STATUS_LLM_DONE,
                    last_error=None,
                    name=company.name,
                    root_url=company.domain_url,
                )

            await scheduler.set_company_stage(
                company.company_id,
                "crawl",
                reset_timers=True,
                reason="exit_llm",
            )

        if stop_event.is_set():
            clog.warning(
                "Stop requested; skipping completion marking company_id=%s",
                company.company_id,
            )
            return False

        snap_end = await state.get_company_snapshot(company.company_id, recompute=True)
        log_snapshot(clog, label="pipeline_end(recompute=True)", snap=snap_end)

        st_end = snap_end.status or COMPANY_STATUS_PENDING
        last_err = snap_last_error(snap_end)

        if st_end == COMPANY_STATUS_TERMINAL_DONE:
            outcome.ok = True
            outcome.stage = "completed_terminal"
            outcome.should_mark_success = False
            outcome.md_done = urls_md_done(snap_end)
            await retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            return True

        if not crawl_runner_done_ok(snap_end):
            msg = last_err or "incomplete_crawl_runner_not_done_ok"
            await _persist_last_error(msg)
            outcome.ok = False
            outcome.stage = "incomplete_crawl"
            outcome.event = retry_mod.classify_failure(
                RuntimeError(msg), stage="incomplete_crawl"
            )
            outcome.md_done = urls_md_done(snap_end)
            await retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            return False

        if do_llm_requested and st_end != COMPANY_STATUS_LLM_DONE:
            msg = last_err or f"llm_requested_but_not_llm_done status={st_end}"
            await _persist_last_error(msg)
            outcome.ok = False
            outcome.stage = "incomplete_llm"
            outcome.event = retry_mod.classify_failure(
                RuntimeError(msg), stage="incomplete_llm"
            )
            outcome.md_done = urls_md_done(snap_end)
            await retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            return False

        if (not do_llm_requested) and st_end != COMPANY_STATUS_MD_DONE:
            with suppress(Exception):
                await state.upsert_company(
                    company.company_id,
                    status=COMPANY_STATUS_MD_DONE,
                    last_error=None,
                    name=company.name,
                    root_url=company.domain_url,
                )

        outcome.ok = True
        outcome.stage = "completed"
        outcome.should_mark_success = True
        outcome.md_done = urls_md_done(snap_end)
        await retry_mod.record_attempt(
            retry_store, company.company_id, outcome, flush=True
        )
        return True

    except asyncio.CancelledError as e:
        msg = str(e) or ""

        if "stop:user" in msg:
            err = "cancelled_by_user"
            clog.warning("Company cancelled by user company_id=%s", company.company_id)
            outcome.ok = False
            outcome.stage = "user_cancel"
            outcome.event = retry_mod.RetryEvent(
                cls="cancel",
                stage="user_cancel",
                error=err,
                nxdomain_like=False,
                status_code=None,
                stall_kind="user",
            )
            outcome.md_done = await _get_md_done(recompute=True)
            await retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            await _persist_last_error(err)
            return False

        if "stop:term" in msg:
            err = "cancelled_by_sigterm"
            clog.warning(
                "Company cancelled by sigterm company_id=%s", company.company_id
            )
            outcome.ok = False
            outcome.stage = "sigterm_cancel"
            outcome.event = retry_mod.RetryEvent(
                cls="cancel",
                stage="sigterm_cancel",
                error=err,
                nxdomain_like=False,
                status_code=None,
                stall_kind="sigterm",
            )
            outcome.md_done = await _get_md_done(recompute=True)
            await retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            await _persist_last_error(err)
            return False

        if "scheduler:" in msg:
            err = f"cancelled_by_scheduler:{msg}"
            clog.warning(
                "Company cancelled by scheduler company_id=%s msg=%s",
                company.company_id,
                msg,
            )
            outcome.ok = False
            outcome.stage = "scheduler_cancel"
            outcome.event = retry_mod.RetryEvent(
                cls="stall",
                stage="scheduler_cancel",
                error=err,
                nxdomain_like=False,
                status_code=None,
                stall_kind="scheduler",
            )
            outcome.md_done = await _get_md_done(recompute=True)
            await retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            await _persist_last_error(err)
            return False

        clog.warning(
            "Company task cancelled (propagate) company_id=%s msg=%s",
            company.company_id,
            msg,
        )
        raise

    except retry_mod.CriticalMemoryPressure as e:
        outcome.ok = False
        outcome.stage = "critical_memory_pressure"
        outcome.event = retry_mod.RetryEvent(
            cls="mem", stage="critical_memory_pressure", error=str(e)
        )
        outcome.md_done = await _get_md_done(recompute=True)
        await retry_mod.record_attempt(
            retry_store, company.company_id, outcome, flush=True
        )
        await _persist_last_error(str(e))
        return False

    except Exception as e:
        ev = retry_mod.classify_failure(e, stage="pipeline_unhandled")
        outcome.ok = False
        outcome.stage = ev.stage
        outcome.event = ev
        outcome.md_done = await _get_md_done(recompute=True)

        if retry_mod.TRACEBACK_ENABLED:
            clog.exception(
                "Pipeline unhandled exception stage=%s err=%s",
                ev.stage,
                retry_mod.short_exc(e),
            )
        else:
            clog.error(
                "Pipeline unhandled exception stage=%s cls=%s stall=%s status=%s nx=%s err=%s",
                ev.stage,
                getattr(ev, "cls", None),
                getattr(ev, "stall_kind", None),
                getattr(ev, "status_code", None),
                getattr(ev, "nxdomain_like", None),
                retry_mod.short_exc(e),
                exc_info=False,
            )

        await retry_mod.record_attempt(
            retry_store, company.company_id, outcome, flush=True
        )
        await _persist_last_error(str(e))
        return False

    finally:
        for t in (llm_watchdog_task, crawl_watchdog_task):
            if t is not None:
                t.cancel()
                with suppress(asyncio.CancelledError):
                    await t

        try:
            snap2 = await state.get_company_snapshot(
                company.company_id, recompute=False
            )
            await state.write_company_meta_snapshot(
                company.company_id,
                snap2,
                pretty=True,
                company_ctx=None,
                set_last_crawled_at=True,
            )
        except Exception:
            pass

        try:
            await state.recompute_company_from_index(
                company.company_id,
                name=company.name,
                root_url=company.domain_url,
            )
        except Exception:
            pass

        if run_id is not None:
            if outcome.ok or outcome.terminalized:
                with suppress(Exception):
                    await state.mark_company_completed(run_id, company.company_id)

        with suppress(Exception):
            await scheduler.clear_company_stage(
                company.company_id,
                reset_timers=False,
                reason="pipeline_exit",
            )

        logging_ext.reset_company_context(token)
        logging_ext.close_company(company.company_id)
        gc.collect()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep crawl corporate websites (per company pipeline)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", type=str)
    g.add_argument("--company-file", type=str)

    p.add_argument("--company-id", type=str, default=None)
    p.add_argument("--lang", type=str, default="en")
    p.add_argument(
        "--out-dir", "--output-dir", dest="out_dir", type=str, default="outputs"
    )

    p.add_argument(
        "--strategy", choices=["bestfirst", "bfs_internal", "dfs"], default="bestfirst"
    )

    p.add_argument("--llm-mode", choices=["none", "presence", "full"], default="none")
    p.add_argument("--llm-model", type=str, default=None)

    p.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root path (used for git metadata in LLM patching).",
    )

    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    p.add_argument("--dataset-file", type=str, default=None)

    p.add_argument("--company-concurrency", type=int, default=8)
    p.add_argument("--llm-concurrency", type=int, default=4)
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--page-timeout-ms", type=int, default=30000)

    p.add_argument("--crawl4ai-cache-dir", type=str, default=None)
    p.add_argument(
        "--crawl4ai-cache-mode",
        choices=["enabled", "disabled", "read_only", "write_only", "bypass"],
        default="bypass",
    )

    p.add_argument(
        "--force-recrawl",
        action="store_true",
        help="Force full recrawl of all non-terminal companies (keeps Crawl4AI cache; skips pages==max_pages check).",
    )

    p.add_argument(
        "--retry-mode", choices=["all", "skip-retry", "only-retry"], default="all"
    )
    p.add_argument(
        "--retry-exit-code",
        type=int,
        default=RETRY_EXIT_CODE,
        help="Exit code used when retry is recommended/required.",
    )
    p.add_argument("--enable-session-log", action="store_true")
    p.add_argument("--enable-resource-monitor", action="store_true")
    p.add_argument("--finalize-in-progress-md", action="store_true")

    p.add_argument(
        "--industry-enrichment",
        action="store_true",
        help="Enable industry label enrichment.",
    )
    p.add_argument(
        "--no-industry-enrichment",
        action="store_true",
        help="Disable industry label enrichment even if LLM is enabled (overrides auto-enable).",
    )
    p.add_argument(
        "--industry-nace-path", type=str, default=None, help="Override nace.ods path."
    )
    p.add_argument(
        "--industry-fallback-path",
        type=str,
        default=None,
        help="Override industry.ods path.",
    )
    p.add_argument("--source-encoding", type=str, default="utf-8")
    p.add_argument("--source-limit", type=int, default=None)
    p.add_argument("--source-no-aggregate-same-url", action="store_true")
    p.add_argument("--source-no-interleave-domains", action="store_true")

    p.add_argument("--page-result-concurrency", type=int, default=8)
    p.add_argument("--page-queue-maxsize", type=int, default=32)
    p.add_argument("--url-index-queue-maxsize", type=int, default=1024)

    p.add_argument("--crawler-pool-size", type=int, default=4)
    p.add_argument("--crawler-recycle-after", type=int, default=12)

    p.add_argument("--max-start-per-tick", type=int, default=3)
    p.add_argument("--crawler-capacity-multiplier", type=int, default=3)
    p.add_argument("--idle-recycle-interval-sec", type=float, default=25.0)
    p.add_argument("--idle-recycle-raw-frac", type=float, default=0.88)
    p.add_argument("--idle-recycle-eff-frac", type=float, default=0.83)

    p.add_argument("--crawler-lease-timeout-sec", type=float, default=240.0)
    p.add_argument("--arun-init-timeout-sec", type=float, default=180.0)
    p.add_argument("--stream-no-yield-timeout-sec", type=float, default=600.0)
    p.add_argument("--submit-timeout-sec", type=float, default=60.0)
    p.add_argument("--resume-md-mode", choices=["direct", "deep"], default="direct")
    p.add_argument("--direct-fetch-url-timeout-sec", type=float, default=180.0)
    p.add_argument("--processor-finish-timeout-sec", type=float, default=360.0)
    p.add_argument("--generator-close-timeout-sec", type=float, default=60.0)
    p.add_argument("--company-crawl-timeout-sec", type=float, default=3600.0)

    p.add_argument("--company-progress-heartbeat-sec", type=float, default=30.0)
    p.add_argument("--company-progress-throttle-sec", type=float, default=12.0)

    p.add_argument("--global-state-write-interval-sec", type=float, default=1.5)

    return p.parse_args(list(argv) if argv is not None else None)


@dataclass(slots=True)
class _StopState:
    reason: str
    sigint_count: int


async def main_async(args: argparse.Namespace) -> None:
    global _forced_exit_code, _retry_store_instance

    out_dir = output_paths.ensure_output_root(args.out_dir)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.setLevel(log_level)

    if args.finalize_in_progress_md and args.llm_mode != "none":
        logger.warning("--finalize-in-progress-md forces --llm-mode none")
        args.llm_mode = "none"

    if args.llm_mode != "none" and (not bool(args.no_industry_enrichment)):
        if not bool(args.industry_enrichment):
            logger.info(
                "Auto-enabling --industry-enrichment because --llm-mode != none (override with --no-industry-enrichment)."
            )
            args.industry_enrichment = True

    repo_root = Path(str(args.repo_root)).expanduser().resolve()

    llm_conc = max(1, int(getattr(args, "llm_concurrency", 2)))
    llm_sem = asyncio.Semaphore(llm_conc)

    logging_ext = LoggingExtension(
        global_level=log_level,
        per_company_level=log_level,
        max_open_company_logs=128,
        enable_session_log=bool(args.enable_session_log),
        session_log_path=output_paths.global_path_obj("session.log")
        if args.enable_session_log
        else None,
    )

    resource_monitor: Optional[ResourceMonitor] = None
    if args.enable_resource_monitor:
        resource_monitor = ResourceMonitor(
            output_path=output_paths.global_path_obj("resource_usage.json"),
            config=ResourceMonitorConfig(),
        )
        resource_monitor.start()

    default_language_factory.set_language(args.lang)

    cache_mode = _CACHE_MODE_MAP[str(args.crawl4ai_cache_mode)]
    cache_dir = args.crawl4ai_cache_dir

    industry_llm_cache: Optional[IndustryAwareStrategyCache] = None
    if args.llm_mode != "none":
        provider_strategy = provider_strategy_from_llm_model_selector(args.llm_model)
        llm_factory = LLMExtractionFactory(provider_strategy=provider_strategy)
        _ = (BASE_FULL_INSTRUCTION, BASE_PRESENCE_INSTRUCTION)
        industry_llm_cache = IndustryAwareStrategyCache(
            factory=llm_factory,
            schema=None,
            extraction_type="schema",
            input_format=None,
            extra_args=None,
            verbose=False,
        )

    guard = ConnectivityGuard()
    await guard.start()

    gating_cfg = md_gating.build_gating_config()
    markdown_generator = default_md_factory.create(
        min_meaningful_words=gating_cfg.min_meaningful_words,
        interstitial_max_share=gating_cfg.interstitial_max_share,
        interstitial_min_hits=gating_cfg.interstitial_min_hits,
        cookie_max_fraction=gating_cfg.cookie_max_fraction,
        require_structure=gating_cfg.require_structure,
    )

    crawler_base_cfg = default_crawler_factory.create(
        markdown_generator=markdown_generator,
        page_timeout=int(args.page_timeout_ms) if args.page_timeout_ms else None,
        cache_mode=cache_mode,
        cache_base_dir=cache_dir,
    )

    page_policy = PageInteractionPolicy(wait_timeout_ms=int(args.page_timeout_ms))
    page_interaction_factory = default_page_interaction_factory

    state = get_crawl_state()

    # ---------------------------------------------------------------------
    # Load companies (canonical: configs.models.Company)
    # ---------------------------------------------------------------------
    companies: List[Company]
    if args.company_file:
        industry_enabled = bool(args.industry_enrichment) and (
            not bool(args.no_industry_enrichment)
        )
        cfg = IndustryEnrichmentConfig(
            enabled=industry_enabled,
            nace_path=Path(args.industry_nace_path)
            if args.industry_nace_path
            else None,
            fallback_path=Path(args.industry_fallback_path)
            if args.industry_fallback_path
            else None,
        )
        companies = load_companies_from_source_with_industry(
            Path(args.company_file),
            industry_config=cfg,
            encoding=str(args.source_encoding),
            limit=args.source_limit,
            aggregate_same_url=not bool(args.source_no_aggregate_same_url),
            interleave_domains=not bool(args.source_no_interleave_domains),
        )
    else:
        url = args.url
        assert url is not None
        cid = args.company_id
        if not cid:
            parsed = urlparse(url)
            cid = (parsed.netloc or parsed.path or "company").replace(":", "_")
        companies = [
            Company.from_input(company_id=cid, root_url=url, name=None, metadata={})
        ]

    companies_by_id: Dict[str, Company] = {c.company_id: c for c in companies}
    company_ids_all: List[str] = [c.company_id for c in companies]
    company_id_set = set(company_ids_all)

    prev_run_max_pages = await state.get_latest_run_max_pages()
    current_max_pages = int(args.max_pages)

    run_id = await state.start_run(
        "deep_crawl",
        version=None,
        args_hash=f"max_pages={current_max_pages}",
        crawl4ai_cache_base_dir=str(Path(cache_dir).expanduser().resolve())
        if cache_dir
        else None,
        crawl4ai_cache_mode=str(args.crawl4ai_cache_mode),
    )

    for c in companies:
        await state.upsert_company(
            c.company_id,
            name=c.name,
            root_url=c.domain_url,
            industry_label=c.industry_label,
            industry=c.industry,
            nace=c.nace,
            industry_source=c.industry_source,
            write_meta=True,
        )

    touched = int(
        await apply_recrawl_policy(
            state=state,
            companies=companies,
            current_max_pages=current_max_pages,
            prev_run_max_pages=prev_run_max_pages,
            force_full_recrawl=bool(args.force_recrawl),
        )
    )
    if bool(args.force_recrawl):
        logger.warning(
            "Force recrawl enabled: marked %d companies markdown_not_done (cache kept).",
            touched,
        )
    else:
        if prev_run_max_pages is not None and current_max_pages > prev_run_max_pages:
            logger.warning(
                "max_pages increased %s -> %s; re-queued %d companies where urls_total==old_cap (cache kept).",
                prev_run_max_pages,
                current_max_pages,
                touched,
            )

    await state.recompute_all_in_progress(concurrency=32)

    if args.finalize_in_progress_md:
        inprog = set(await state.get_in_progress_company_ids(limit=1_000_000))
        companies = [c for c in companies if c.company_id in inprog]
        companies_by_id = {c.company_id: c for c in companies}
        company_ids_all = [c.company_id for c in companies]
        company_id_set = set(company_ids_all)

    dataset_externals = build_dataset_externals(args=args, companies=companies)

    bm25 = build_dual_bm25_components()
    url_scorer: Optional[DualBM25Scorer] = bm25["url_scorer"]
    bm25_filter: Optional[DualBM25Filter] = bm25["url_filter"]

    dfs_factory: Optional[DeepCrawlStrategyFactoryType] = None
    if args.strategy == "dfs":
        dfs_factory = DeepCrawlStrategyFactory(
            provider=DFSDeepCrawlStrategyProvider(default_max_depth=3)
        )

    http_lang = _HTTP_LANG_MAP.get(args.lang, f"{args.lang}-US")
    browser_cfg = default_browser_factory.create(lang=http_lang, headless=True)

    crawler_pool = CrawlerPool(
        browser_cfg=browser_cfg,
        size=int(args.crawler_pool_size),
        recycle_after_companies=int(args.crawler_recycle_after),
    )
    await crawler_pool.start()

    stop_event = asyncio.Event()
    stop_state = _StopState(reason="none", sigint_count=0)

    inflight_by_cid: Dict[str, asyncio.Task] = {}

    def get_active_company_ids() -> Sequence[str]:
        return [cid for cid, t in inflight_by_cid.items() if not t.done()]

    def request_cancel_companies(ids: Sequence[str]) -> None:
        for cid in ids:
            t = inflight_by_cid.get(cid)
            if t and not t.done():
                t.cancel("scheduler:cancel")

    async def request_recycle_idle(count: int, reason: str) -> int:
        logger.info("request_recycle_idle count=%d reason=%s (noop)", count, reason)
        return 0

    sched_cfg = AdaptiveSchedulingConfig(
        retry_base_dir=(out_dir / "_retry").resolve(),
        max_start_per_tick=int(args.max_start_per_tick),
        crawler_capacity_multiplier=int(args.crawler_capacity_multiplier),
        idle_recycle_interval_sec=float(args.idle_recycle_interval_sec),
        idle_recycle_raw_frac=float(args.idle_recycle_raw_frac),
        idle_recycle_eff_frac=float(args.idle_recycle_eff_frac),
    )
    scheduler = AdaptiveScheduler(
        cfg=sched_cfg,
        get_active_company_ids=get_active_company_ids,
        request_cancel_companies=request_cancel_companies,
        request_recycle_idle=request_recycle_idle,
    )
    await scheduler.start()

    retry_store = scheduler.retry_store
    _retry_store_instance = retry_store

    async def is_company_runnable(cid: str) -> bool:
        """
        Runnable gate:
          - Terminal -> not runnable
          - If llm requested: runnable iff status != LLM_DONE
          - Else: runnable iff status != MD_DONE
        NOTE: recompute=True so we trust crawl.runner meta (crawl_finished).
        """
        if cid not in company_id_set:
            return False

        company = companies_by_id.get(cid)
        if company is None:
            return False

        snap = await state.get_company_snapshot(cid, recompute=True)
        logger.debug(
            "is_company_runnable snapshot cid=%s status=%s crawl_finished=%s urls_total=%d md_done=%d llm_done=%d last_error=%r",
            cid,
            snap.status,
            bool(snap.crawl_finished),
            int(snap.urls_total),
            int(snap.urls_markdown_done),
            int(snap.urls_llm_done),
            snap.last_error,
        )
        st = snap.status or COMPANY_STATUS_PENDING

        if st == COMPANY_STATUS_TERMINAL_DONE:
            return False

        do_llm_requested = llm_requested(args, industry_llm_cache)

        if do_llm_requested:
            return st != COMPANY_STATUS_LLM_DONE

        return st != COMPANY_STATUS_MD_DONE

    pending_retry_ids = set(await retry_store.pending_ids(exclude_quarantined=True))
    for rid in pending_retry_ids:
        if rid not in company_id_set:
            await retry_store.mark_success(
                rid, stage="startup_cleanup", note="orphan_retry_id"
            )
            continue
        if not await is_company_runnable(rid):
            await retry_store.mark_success(
                rid, stage="startup_cleanup", note="already_done_or_terminal"
            )

    await scheduler.cleanup_completed_retry_ids(
        is_company_runnable=is_company_runnable,
        treat_non_runnable_as_done=True,
    )

    runnable_ids: List[str] = []
    for cid in company_ids_all:
        if await is_company_runnable(cid):
            runnable_ids.append(cid)

    await state.update_run_totals(run_id, total_companies=len(runnable_ids))

    await scheduler.set_worklist(
        runnable_ids,
        retry_mode=str(args.retry_mode),
        is_company_runnable=is_company_runnable,
    )

    last_global_write = 0.0
    write_interval = float(args.global_state_write_interval_sec)

    async def maybe_write_global_state() -> None:
        nonlocal last_global_write
        now = time.monotonic()
        if (now - last_global_write) < write_interval:
            return
        last_global_write = now
        await state.write_global_state_throttled(
            min_interval_sec=max(0.05, write_interval)
        )

    async def force_write_global_state() -> None:
        nonlocal last_global_write
        last_global_write = time.monotonic()
        await state.write_global_state_throttled(min_interval_sec=0.0)

    def _cancel_inflight(tag: str) -> None:
        for t in inflight_by_cid.values():
            if not t.done():
                t.cancel(tag)

    def on_sigint() -> None:
        global _forced_exit_code
        stop_state.sigint_count += 1
        if stop_state.sigint_count >= 2:
            _forced_exit_code = 130
            os._exit(130)
        stop_state.reason = "sigint"
        _forced_exit_code = 130
        logger.warning("SIGINT received -> graceful shutdown")
        stop_event.set()
        _cancel_inflight("stop:user")

    def on_sigterm() -> None:
        global _forced_exit_code
        if getattr(scheduler, "restart_recommended", False):
            stop_state.reason = "restart_recommended"
            _forced_exit_code = int(args.retry_exit_code)
            logger.error(
                "SIGTERM received and restart is recommended -> exiting retry-exit-code=%s",
                _forced_exit_code,
            )
            stop_event.set()
            _cancel_inflight("stop:term")
            return
        stop_state.reason = "sigterm"
        _forced_exit_code = 143
        logger.warning("SIGTERM received -> graceful shutdown")
        stop_event.set()
        _cancel_inflight("stop:term")

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, on_sigint)
    loop.add_signal_handler(signal.SIGTERM, on_sigterm)

    if not runnable_ids and not scheduler.has_pending():
        with suppress(Exception):
            await force_write_global_state()
        _forced_exit_code = 0
        with suppress(Exception):
            await crawler_pool.stop()
        with suppress(Exception):
            await guard.stop()
        if resource_monitor is not None:
            resource_monitor.stop()
        return

    done_counter: Dict[str, int] = {"done": 0}
    total_unique = len(runnable_ids)

    cap = max(1, int(args.company_concurrency))
    mult = max(1, int(args.crawler_capacity_multiplier))
    free_crawlers_for_sched = max(
        1, min(int(args.crawler_pool_size), (cap + mult - 1) // mult)
    )

    async def drain_reconcile_and_reseed() -> bool:
        await state.recompute_all_in_progress(concurrency=32)

        unfinished: List[str] = []
        for cid in company_ids_all:
            if await is_company_runnable(cid):
                unfinished.append(cid)

        if not unfinished:
            return True

        added = 0
        try:
            added = int(await scheduler.ensure_worklist(unfinished))
        except Exception as e:
            logger.warning("ensure_worklist failed err=%s", retry_mod.short_exc(e))
            added = 0

        if added <= 0:
            for cid in unfinished:
                with suppress(Exception):
                    await scheduler.requeue_company(
                        cid, force=True, reason="drain_reconcile"
                    )

        with suppress(Exception):
            await force_write_global_state()
        return False

    try:
        while not stop_event.is_set():
            finished: List[str] = [
                cid for cid, t in inflight_by_cid.items() if t.done()
            ]
            for cid in finished:
                t = inflight_by_cid.pop(cid)

                ok = False
                try:
                    ok = bool(t.result())
                except asyncio.CancelledError:
                    ok = False
                except Exception as e:
                    logger.error(
                        "Company task crashed company_id=%s err=%s",
                        cid,
                        retry_mod.short_exc(e),
                    )
                    ok = False

                scheduler.register_company_completed()
                if ok:
                    done_counter["done"] = int(done_counter["done"]) + 1

            with suppress(Exception):
                await maybe_write_global_state()

            if getattr(scheduler, "restart_recommended", False):
                stop_state.reason = "restart_recommended"
                _forced_exit_code = int(args.retry_exit_code)
                logger.error(
                    "Scheduler recommends restart -> stopping run exit_code=%s",
                    _forced_exit_code,
                )
                stop_event.set()
                _cancel_inflight("stop:term")
                break

            if len(inflight_by_cid) < cap:
                to_start = await scheduler.plan_start_batch(
                    free_crawlers=int(free_crawlers_for_sched)
                )
                for cid in to_start:
                    if stop_event.is_set():
                        break
                    if cid in inflight_by_cid and not inflight_by_cid[cid].done():
                        continue
                    c = companies_by_id.get(cid)
                    if c is None:
                        continue

                    async def _runner(company: Company) -> bool:
                        return await run_company_pipeline(
                            company,
                            attempt_no=1,
                            total_unique=total_unique,
                            done_counter=done_counter,
                            logging_ext=logging_ext,
                            state=state,
                            guard=guard,
                            crawler_pool=crawler_pool,
                            args=args,
                            dataset_externals=dataset_externals,
                            url_scorer=url_scorer,
                            bm25_filter=bm25_filter,
                            run_id=run_id,
                            industry_llm_cache=industry_llm_cache,
                            dfs_factory=dfs_factory,
                            crawler_base_cfg=crawler_base_cfg,
                            page_policy=page_policy,
                            page_interaction_factory=page_interaction_factory,
                            retry_store=retry_store,
                            scheduler=scheduler,
                            stop_event=stop_event,
                            llm_sem=llm_sem,
                            repo_root=repo_root,
                        )

                    inflight_by_cid[cid] = asyncio.create_task(
                        _runner(c), name=f"company:{cid}"
                    )

            if not inflight_by_cid and not scheduler.has_pending():
                should_break = await drain_reconcile_and_reseed()
                if should_break:
                    break
                continue

            hint = float(scheduler.sleep_hint_sec())
            await asyncio.sleep(max(0.05, min(0.5, hint)))

    finally:
        if stop_state.reason == "sigint":
            _cancel_inflight("stop:user")
        elif stop_state.reason in ("sigterm", "restart_recommended"):
            _cancel_inflight("stop:term")

        if inflight_by_cid:
            await asyncio.gather(*inflight_by_cid.values(), return_exceptions=True)

        with suppress(Exception):
            await crawler_pool.stop()
        with suppress(Exception):
            await guard.stop()

        if resource_monitor is not None:
            resource_monitor.stop()

        with suppress(Exception):
            await force_write_global_state()

        if stop_state.reason == "sigint":
            _forced_exit_code = 130
            return
        if stop_state.reason == "sigterm":
            _forced_exit_code = 143
            return
        if stop_state.reason == "restart_recommended":
            _forced_exit_code = int(args.retry_exit_code)
            return

        await scheduler.cleanup_completed_retry_ids(
            is_company_runnable=is_company_runnable,
            treat_non_runnable_as_done=True,
        )

        with suppress(Exception):
            should_break = await drain_reconcile_and_reseed()
            if should_break:
                await scheduler.cleanup_completed_retry_ids(
                    is_company_runnable=is_company_runnable,
                    treat_non_runnable_as_done=True,
                )

        if not scheduler.has_pending():
            _forced_exit_code = 0
            return

        _forced_exit_code = int(
            compute_retry_exit_code_from_store(
                scheduler.retry_store,
                retry_exit_code=int(args.retry_exit_code),
            )
        )


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    asyncio.run(main_async(args))
    raise SystemExit(int(_forced_exit_code or 0))


if __name__ == "__main__":
    main()
