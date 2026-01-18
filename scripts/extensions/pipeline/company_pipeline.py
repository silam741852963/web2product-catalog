from __future__ import annotations

import argparse
import asyncio
import gc
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from configs.deep_crawl import DeepCrawlStrategyFactory as DeepCrawlStrategyFactoryType
from configs.js_injection import PageInteractionFactory, PageInteractionPolicy
from configs.llm import IndustryAwareStrategyCache
from configs.models import (
    Company,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_PENDING,
)

from extensions.crawl.runner import CrawlRunnerConfig, run_company_crawl
from extensions.crawl.runner_constants import CACHE_MODE_MAP
from extensions.crawl.state import (
    CrawlState,
    log_snapshot,
    snap_last_error,
    urls_md_done,
)
from extensions.filter import md_gating
from extensions.filter.core import build_filter_chain
from extensions.filter.dual_bm25 import DualBM25Filter, DualBM25Scorer
from extensions.guard.connectivity import ConnectivityGuard
from extensions.pipeline.llm_passes import (
    build_industry_context,
    has_industry_context,
    llm_requested,
    run_full_pass_for_company,
    run_presence_pass_for_company,
)
from extensions.pipeline.pipeline_helpers import (
    crawl_or_terminal_done_ok,
    llm_precheck_ok,
    should_short_circuit_llm_due_to_no_markdown,
)
from extensions.schedule import retry as retry_mod
from extensions.schedule.adaptive import AdaptiveScheduler
from extensions.schedule.crawler_pool import CrawlerPool
from extensions.schedule.terminalization import (
    decide_from_page_summary,
    safe_mark_terminal,
    stall_sanity_override_if_crawl_finished,
    terminal_sanity_gate_if_crawl_finished,
)
from extensions.utils.logging import LoggingExtension


# --------------------------------------------------------------------------------------
# Small objects to make flow readable
# --------------------------------------------------------------------------------------
@dataclass(slots=True)
class PipelineCtx:
    company: Company
    attempt_no: int
    logging_ext: LoggingExtension
    state: CrawlState
    guard: ConnectivityGuard
    crawler_pool: CrawlerPool
    args: argparse.Namespace
    dataset_externals: frozenset[str]
    url_scorer: Optional[DualBM25Scorer]
    bm25_filter: Optional[DualBM25Filter]
    run_id: Optional[str]
    industry_llm_cache: Optional[IndustryAwareStrategyCache]
    dfs_factory: Optional[DeepCrawlStrategyFactoryType]
    crawler_base_cfg: Any
    page_policy: PageInteractionPolicy
    page_interaction_factory: PageInteractionFactory
    retry_store: retry_mod.RetryStateStore
    scheduler: AdaptiveScheduler
    stop_event: asyncio.Event
    repo_root: Path

    clog: Any  # company logger


@dataclass(slots=True)
class PipelinePlan:
    status: str
    md_done_enter: int
    do_llm_requested: bool
    do_crawl: bool


class Progress:
    def __init__(
        self, *, scheduler: AdaptiveScheduler, company_id: str, throttle_sec: float
    ):
        self._scheduler = scheduler
        self._company_id = company_id
        self._throttle_sec = float(throttle_sec)
        self._last_progress_mono = 0.0

    def progress(self) -> None:
        now = time.monotonic()
        if (now - self._last_progress_mono) < self._throttle_sec:
            return
        self._last_progress_mono = now
        self._scheduler.progress_company(self._company_id)

    def heartbeat(self) -> None:
        self._scheduler.heartbeat_company(self._company_id)


# --------------------------------------------------------------------------------------
# Shared tiny helpers
# --------------------------------------------------------------------------------------
async def _get_md_done(ctx: PipelineCtx, *, recompute: bool) -> int:
    snapx = await ctx.state.get_company_snapshot(
        ctx.company.company_id, recompute=recompute
    )
    log_snapshot(ctx.clog, label=f"_get_md_done(recompute={recompute})", snap=snapx)
    return int(getattr(snapx, "urls_markdown_done", 0) or 0)


async def _persist_last_error(ctx: PipelineCtx, err: str) -> None:
    await ctx.state.upsert_company(
        ctx.company.company_id,
        last_error=(err or "")[:4000],
        name=ctx.company.name,
        root_url=ctx.company.domain_url,
    )


async def _record_and_return(
    ctx: PipelineCtx,
    outcome: retry_mod.AttemptOutcome,
    *,
    ok: bool,
    md_recompute: bool,
    persist_err: Optional[str],
) -> bool:
    outcome.ok = bool(ok)
    outcome.md_done = await _get_md_done(ctx, recompute=md_recompute)
    await retry_mod.record_attempt(
        ctx.retry_store, ctx.company.company_id, outcome, flush=True
    )
    if persist_err is not None:
        await _persist_last_error(ctx, persist_err)
    return bool(ok)


# --------------------------------------------------------------------------------------
# Stage: enter / plan
# --------------------------------------------------------------------------------------
async def _pipeline_enter(ctx: PipelineCtx) -> PipelinePlan:
    await ctx.scheduler.set_company_stage(
        ctx.company.company_id,
        "crawl",
        reset_timers=True,
        reason="pipeline_enter",
    )

    await ctx.state.upsert_company(
        ctx.company.company_id,
        name=ctx.company.name,
        root_url=ctx.company.domain_url,
        industry_label=ctx.company.industry_label,
        industry=ctx.company.industry,
        nace=ctx.company.nace,
        industry_source=ctx.company.industry_source,
        write_meta=True,
    )

    snap = await ctx.state.get_company_snapshot(ctx.company.company_id, recompute=False)
    log_snapshot(ctx.clog, label="pipeline_enter(recompute=False)", snap=snap)

    status = (
        getattr(snap, "status", None) or COMPANY_STATUS_PENDING
    ) or COMPANY_STATUS_PENDING
    md_done_enter = int(getattr(snap, "urls_markdown_done", 0) or 0)

    do_llm = llm_requested(ctx.args, ctx.industry_llm_cache)
    do_crawl = status in (COMPANY_STATUS_PENDING, COMPANY_STATUS_MD_NOT_DONE)

    done_db, total_db = await ctx.state.get_db_progress_counts(llm_requested=do_llm)

    ctx.clog.info(
        "=== [done=%d/%d attempt=%d] company_id=%s url=%s industry_label=%s industry=%s nace=%s status=%s llm=%s llm_requested=%s has_ctx=%s ===",
        done_db,
        total_db,
        ctx.attempt_no,
        ctx.company.company_id,
        ctx.company.domain_url,
        ctx.company.industry_label,
        ctx.company.industry,
        ctx.company.nace,
        status,
        ctx.args.llm_mode,
        do_llm,
        has_industry_context(ctx.company),
    )

    return PipelinePlan(
        status=str(status),
        md_done_enter=int(md_done_enter),
        do_llm_requested=bool(do_llm),
        do_crawl=bool(do_crawl),
    )


# --------------------------------------------------------------------------------------
# Stage: crawl
# --------------------------------------------------------------------------------------
@dataclass(slots=True)
class CrawlResult:
    ok: bool
    terminalized: bool
    terminal_reason: Optional[str]
    terminal_last_error: Optional[str]
    fail_event: Optional[retry_mod.RetryEvent]
    fail_exc: Optional[BaseException]
    last_term_stage: str


async def _run_crawl_stage(
    ctx: PipelineCtx,
    plan: PipelinePlan,
    progress: Progress,
) -> CrawlResult:
    if not plan.do_crawl:
        return CrawlResult(
            ok=True,
            terminalized=False,
            terminal_reason=None,
            terminal_last_error=None,
            fail_event=None,
            fail_exc=None,
            last_term_stage="crawl",
        )

    await ctx.scheduler.set_company_stage(
        ctx.company.company_id, "crawl", reset_timers=True, reason="enter_crawl"
    )
    await ctx.guard.wait_until_healthy()

    resume_roots: Optional[List[str]] = None
    direct_fetch_urls = False
    if plan.status == COMPANY_STATUS_MD_NOT_DONE:
        pending_md = await ctx.state.get_pending_urls_for_markdown(
            ctx.company.company_id
        )
        if pending_md:
            resume_roots = pending_md
            direct_fetch_urls = ctx.args.resume_md_mode == "direct"

    filter_chain = build_filter_chain(
        company_url=ctx.company.domain_url,
        company_id=ctx.company.company_id,
        lang=ctx.args.lang,
        dataset_externals=ctx.dataset_externals,
        bm25_filter=ctx.bm25_filter,
    )

    # NOTE: keep local import to keep run.py thin.
    from configs.deep_crawl import build_deep_strategy

    deep_strategy = build_deep_strategy(
        strategy=ctx.args.strategy,
        filter_chain=filter_chain,
        url_scorer=ctx.url_scorer,
        dfs_factory=ctx.dfs_factory,
        max_pages=int(ctx.args.max_pages) if ctx.args.max_pages else None,
    )

    cache_mode = CACHE_MODE_MAP[str(ctx.args.crawl4ai_cache_mode)]
    runner_cfg = CrawlRunnerConfig(
        page_result_concurrency=int(ctx.args.page_result_concurrency),
        page_queue_maxsize=int(ctx.args.page_queue_maxsize),
        url_index_queue_maxsize=int(ctx.args.url_index_queue_maxsize),
        arun_init_timeout_sec=float(ctx.args.arun_init_timeout_sec),
        stream_no_yield_timeout_sec=float(ctx.args.stream_no_yield_timeout_sec),
        submit_timeout_sec=float(ctx.args.submit_timeout_sec),
        direct_fetch_total_timeout_sec=float(ctx.args.direct_fetch_url_timeout_sec),
        processor_finish_timeout_sec=float(ctx.args.processor_finish_timeout_sec),
        generator_close_timeout_sec=float(ctx.args.generator_close_timeout_sec),
        hard_max_pages=int(ctx.args.max_pages) if ctx.args.max_pages else None,
        page_timeout_ms=int(ctx.args.page_timeout_ms)
        if ctx.args.page_timeout_ms
        else None,
        direct_fetch_urls=bool(direct_fetch_urls),
        crawl4ai_cache_mode=cache_mode,
    )

    heartbeat_sec = float(ctx.args.company_progress_heartbeat_sec)

    async def _watchdog() -> None:
        while True:
            await asyncio.sleep(heartbeat_sec)
            progress.heartbeat()

    watchdog_task = asyncio.create_task(
        _watchdog(), name=f"watchdog:crawl:{ctx.company.company_id}"
    )

    last_exc: Optional[BaseException] = None
    last_event: Optional[retry_mod.RetryEvent] = None
    last_page_dec: Optional[Any] = None
    last_term_stage: str = "crawl"

    try:
        for attempt_index in range(2):
            try:
                lease = await asyncio.wait_for(
                    ctx.crawler_pool.lease(),
                    timeout=float(ctx.args.crawler_lease_timeout_sec),
                )

                async with lease as crawler:

                    async def _do() -> Any:
                        return await run_company_crawl(
                            company=ctx.company,
                            crawler=crawler,
                            deep_strategy=deep_strategy,
                            guard=ctx.guard,
                            gating_cfg=md_gating.build_gating_config(),
                            crawler_base_cfg=ctx.crawler_base_cfg,
                            page_policy=ctx.page_policy,
                            page_interaction_factory=ctx.page_interaction_factory,
                            root_urls=resume_roots,
                            cfg=runner_cfg,
                            on_progress=lambda: progress.progress(),
                        )

                    summary = await asyncio.wait_for(
                        _do(),
                        timeout=float(ctx.args.company_crawl_timeout_sec),
                    )

                    last_page_dec = decide_from_page_summary(summary)

                    if getattr(last_page_dec, "action", None) == "mem":
                        lease.mark_fatal("page_pipeline_mem")
                        raise retry_mod.CriticalMemoryPressure(
                            str(getattr(last_page_dec, "reason", "") or "mem"),
                            severity="critical",
                        )

                    _ = await stall_sanity_override_if_crawl_finished(
                        ctx.state, ctx.company, last_page_dec, ctx.clog
                    )

                    term_dec = retry_mod.decide_terminalization(
                        page_summary_decision=last_page_dec,
                        exception=None,
                        stage="crawl",
                        urls_md_done=plan.md_done_enter,
                        attempt_index=attempt_index,
                    )

                    if term_dec.should_terminalize and last_page_dec is not None:
                        ignore_term = await terminal_sanity_gate_if_crawl_finished(
                            ctx.state, ctx.company, term_dec, last_page_dec, ctx.clog
                        )
                        if not ignore_term:
                            return CrawlResult(
                                ok=True,
                                terminalized=True,
                                terminal_reason=term_dec.reason,
                                terminal_last_error=term_dec.last_error,
                                fail_event=None,
                                fail_exc=None,
                                last_term_stage="crawl",
                            )

                    if getattr(last_page_dec, "action", None) == "stall":
                        raise retry_mod.CrawlerTimeoutError(
                            str(getattr(last_page_dec, "reason", "") or "stall"),
                            stage="page_pipeline_timeout_dominance",
                            company_id=ctx.company.company_id,
                            url=ctx.company.domain_url,
                        )

                ctx.guard.record_success()
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
                    ctx.guard.record_transport_error()

                if retry_mod.should_fail_fast_on_goto(e, stage=stage):
                    term_dec = retry_mod.decide_terminalization(
                        page_summary_decision=None,
                        exception=e,
                        stage=stage,
                        urls_md_done=plan.md_done_enter,
                        attempt_index=attempt_index,
                    )
                    if term_dec.should_terminalize:
                        return CrawlResult(
                            ok=True,
                            terminalized=True,
                            terminal_reason=term_dec.reason,
                            terminal_last_error=term_dec.last_error,
                            fail_event=None,
                            fail_exc=None,
                            last_term_stage=stage,
                        )

                retry_mod.log_attempt_failure(
                    ctx.clog,
                    prefix="Crawl attempt failed",
                    attempt_index=attempt_index,
                    stage=stage,
                    event=last_event,
                    exc=e,
                    traceback_enabled=retry_mod.TRACEBACK_ENABLED,
                )

                if attempt_index == 0:
                    await asyncio.sleep(0.5)

        if last_exc is not None and last_event is not None:
            return CrawlResult(
                ok=False,
                terminalized=False,
                terminal_reason=None,
                terminal_last_error=None,
                fail_event=last_event,
                fail_exc=last_exc,
                last_term_stage="crawl",
            )

        # recompute & mark MD_DONE when crawl completed normally
        await ctx.state.recompute_company_from_index(
            ctx.company.company_id,
            name=ctx.company.name,
            root_url=ctx.company.domain_url,
        )

        snap_after = await ctx.state.get_company_snapshot(
            ctx.company.company_id, recompute=True
        )
        log_snapshot(ctx.clog, label="after_crawl(recompute=True)", snap=snap_after)

        if crawl_or_terminal_done_ok(snap_after):
            # Only force MD_DONE when the crawler indicates normal completion.
            # If we are already terminal_done, keep it as-is.
            st = (getattr(snap_after, "status", "") or "").strip()
            if st != "terminal_done":
                with suppress(Exception):
                    await ctx.state.upsert_company(
                        ctx.company.company_id,
                        status=COMPANY_STATUS_MD_DONE,
                        # clear last_error on clean crawl completion only
                        last_error=None,
                        name=ctx.company.name,
                        root_url=ctx.company.domain_url,
                    )
            return CrawlResult(
                ok=True,
                terminalized=False,
                terminal_reason=None,
                terminal_last_error=None,
                fail_event=None,
                fail_exc=None,
                last_term_stage="crawl",
            )

        msg = snap_last_error(snap_after) or "crawl_not_finished_or_error"
        return CrawlResult(
            ok=False,
            terminalized=False,
            terminal_reason=None,
            terminal_last_error=None,
            fail_event=retry_mod.classify_failure(
                RuntimeError(msg), stage="crawl_not_done"
            ),
            fail_exc=RuntimeError(msg),
            last_term_stage="crawl",
        )

    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task


async def _apply_terminalization_if_any(
    ctx: PipelineCtx,
    crawl_res: CrawlResult,
    outcome: retry_mod.AttemptOutcome,
) -> bool:
    """
    IMPORTANT POLICY CHANGE:
      - terminal_done is treated like markdown_done for the purpose of LLM.
      - We DO NOT stop the pipeline after terminalizing.
      - We DO NOT clear last_error during terminalization (it is often useful signal).
    """
    if not crawl_res.terminalized:
        return False

    wrote = await safe_mark_terminal(
        ctx.state,
        ctx.company,
        reason="terminalize",
        details={
            "reason": crawl_res.terminal_reason,
            "industry_label": ctx.company.industry_label,
            "industry": ctx.company.industry,
            "nace": ctx.company.nace,
            "term_stage": crawl_res.last_term_stage,
        },
        last_error=(
            crawl_res.terminal_last_error or crawl_res.terminal_reason or "terminalize"
        ),
        stage=crawl_res.last_term_stage,
        logger=ctx.clog,
    )

    if wrote:
        outcome.terminalized = True
        outcome.terminal_reason = crawl_res.terminal_reason
        outcome.terminal_last_error = crawl_res.terminal_last_error
        ctx.clog.info(
            "Terminalized company but continuing pipeline (LLM allowed) company_id=%s stage=%s reason=%s",
            ctx.company.company_id,
            crawl_res.last_term_stage,
            str(crawl_res.terminal_reason or ""),
        )
        return True

    ctx.clog.info(
        "safe_terminal refused; continuing pipeline as non-terminal company=%s stage=%s",
        ctx.company.company_id,
        crawl_res.last_term_stage,
    )
    return False


# --------------------------------------------------------------------------------------
# Stage: llm
# --------------------------------------------------------------------------------------
async def _run_llm_stage(
    ctx: PipelineCtx,
    plan: PipelinePlan,
    progress: Progress,
    outcome: retry_mod.AttemptOutcome,
) -> bool:
    if not plan.do_llm_requested:
        return True

    assert ctx.industry_llm_cache is not None

    snap_md = await ctx.state.get_company_snapshot(
        ctx.company.company_id, recompute=True
    )
    log_snapshot(ctx.clog, label="pre_llm(recompute=True)", snap=snap_md)

    # POLICY: terminal_done is allowed here (handled in llm_precheck_ok)
    if not llm_precheck_ok(snap_md):
        await _persist_last_error(
            ctx,
            f"llm_requested_but_status_not_ready status={getattr(snap_md, 'status', None)!r}",
        )
        outcome.ok = False
        outcome.stage = "precheck_llm_status_not_ready"
        outcome.event = retry_mod.classify_failure(
            RuntimeError("llm_requested_but_status_not_ready"),
            stage="precheck_llm_status_not_ready",
        )
        await retry_mod.record_attempt(
            ctx.retry_store, ctx.company.company_id, outcome, flush=True
        )
        return False

    if should_short_circuit_llm_due_to_no_markdown(snap_md, llm_is_requested=True):
        with suppress(Exception):
            await ctx.state.upsert_company(
                ctx.company.company_id,
                status=COMPANY_STATUS_LLM_DONE,
                # preserve last_error on purpose
                name=ctx.company.name,
                root_url=ctx.company.domain_url,
            )
        outcome.ok = True
        outcome.stage = "llm_short_circuit_no_markdown"
        outcome.should_mark_success = True
        outcome.md_done = urls_md_done(snap_md)
        await retry_mod.record_attempt(
            ctx.retry_store, ctx.company.company_id, outcome, flush=True
        )
        return True

    await ctx.scheduler.set_company_stage(
        ctx.company.company_id, "llm", reset_timers=True, reason="enter_llm"
    )

    heartbeat_sec = float(ctx.args.company_progress_heartbeat_sec)

    async def _watchdog() -> None:
        while True:
            await asyncio.sleep(heartbeat_sec)
            progress.heartbeat()

    watchdog_task = asyncio.create_task(
        _watchdog(), name=f"watchdog:llm:{ctx.company.company_id}"
    )

    llm_exc: Optional[BaseException] = None
    llm_event: Optional[retry_mod.RetryEvent] = None
    llm_stage = f"llm_{ctx.args.llm_mode}"

    try:
        for attempt_index in range(2):
            try:
                ictx = build_industry_context(ctx.company)

                if ctx.args.llm_mode == "presence":
                    strat = ctx.industry_llm_cache.get_strategy(
                        mode="presence", ctx=ictx
                    )
                    await run_presence_pass_for_company(
                        ctx.company, presence_strategy=strat, repo_root=ctx.repo_root
                    )
                elif ctx.args.llm_mode == "full":
                    strat = ctx.industry_llm_cache.get_strategy(mode="schema", ctx=ictx)
                    await run_full_pass_for_company(
                        ctx.company, full_strategy=strat, repo_root=ctx.repo_root
                    )
                else:
                    raise ValueError(f"Unsupported llm_mode={ctx.args.llm_mode!r}")

                progress.progress()
                llm_exc = None
                llm_event = None
                break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                llm_exc = e
                llm_event = retry_mod.classify_failure(e, stage=llm_stage)

                retry_mod.log_attempt_failure(
                    ctx.clog,
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
            outcome.md_done = await _get_md_done(ctx, recompute=True)
            await retry_mod.record_attempt(
                ctx.retry_store, ctx.company.company_id, outcome, flush=True
            )
            await _persist_last_error(ctx, str(llm_exc))
            return False

        with suppress(Exception):
            await ctx.state.upsert_company(
                ctx.company.company_id,
                status=COMPANY_STATUS_LLM_DONE,
                # do NOT require last_error empty to proceed; but on successful LLM we clear it
                last_error=None,
                name=ctx.company.name,
                root_url=ctx.company.domain_url,
            )

        await ctx.scheduler.set_company_stage(
            ctx.company.company_id, "crawl", reset_timers=True, reason="exit_llm"
        )
        return True

    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task


# --------------------------------------------------------------------------------------
# Stage: finalize
# --------------------------------------------------------------------------------------
async def _finalize_pipeline(
    ctx: PipelineCtx,
    plan: PipelinePlan,
    outcome: retry_mod.AttemptOutcome,
) -> bool:
    if ctx.stop_event.is_set():
        ctx.clog.warning(
            "Stop requested; skipping completion marking company_id=%s",
            ctx.company.company_id,
        )
        return False

    snap_end = await ctx.state.get_company_snapshot(
        ctx.company.company_id, recompute=True
    )
    log_snapshot(ctx.clog, label="pipeline_end(recompute=True)", snap=snap_end)

    st_end = (
        getattr(snap_end, "status", None) or COMPANY_STATUS_PENDING
    ) or COMPANY_STATUS_PENDING
    last_err = snap_last_error(snap_end)

    # POLICY: terminal_done is treated as "crawl completed"
    if not crawl_or_terminal_done_ok(snap_end):
        msg = last_err or "incomplete_crawl_not_done_ok"
        await _persist_last_error(ctx, msg)
        outcome.ok = False
        outcome.stage = "incomplete_crawl"
        outcome.event = retry_mod.classify_failure(
            RuntimeError(msg), stage="incomplete_crawl"
        )
        outcome.md_done = urls_md_done(snap_end)
        await retry_mod.record_attempt(
            ctx.retry_store, ctx.company.company_id, outcome, flush=True
        )
        return False

    if plan.do_llm_requested and st_end != COMPANY_STATUS_LLM_DONE:
        msg = last_err or f"llm_requested_but_not_llm_done status={st_end}"
        await _persist_last_error(ctx, msg)
        outcome.ok = False
        outcome.stage = "incomplete_llm"
        outcome.event = retry_mod.classify_failure(
            RuntimeError(msg), stage="incomplete_llm"
        )
        outcome.md_done = urls_md_done(snap_end)
        await retry_mod.record_attempt(
            ctx.retry_store, ctx.company.company_id, outcome, flush=True
        )
        return False

    if (not plan.do_llm_requested) and st_end != COMPANY_STATUS_MD_DONE:
        # don't clobber terminal_done; keep it if it was set
        if str(st_end).strip() != "terminal_done":
            with suppress(Exception):
                await ctx.state.upsert_company(
                    ctx.company.company_id,
                    status=COMPANY_STATUS_MD_DONE,
                    last_error=None,
                    name=ctx.company.name,
                    root_url=ctx.company.domain_url,
                )

    outcome.ok = True
    outcome.stage = "completed"
    outcome.should_mark_success = True
    outcome.md_done = urls_md_done(snap_end)
    await retry_mod.record_attempt(
        ctx.retry_store, ctx.company.company_id, outcome, flush=True
    )
    return True


# --------------------------------------------------------------------------------------
# Exception handling (pulled out of the mega function)
# --------------------------------------------------------------------------------------
async def _handle_cancelled(
    ctx: PipelineCtx,
    outcome: retry_mod.AttemptOutcome,
    e: asyncio.CancelledError,
) -> bool:
    msg = str(e) or ""

    if "stop:user" in msg:
        outcome.ok = False
        outcome.stage = "user_cancel"
        outcome.event = retry_mod.RetryEvent(
            cls="cancel",
            stage="user_cancel",
            error="cancelled_by_user",
            nxdomain_like=False,
            status_code=None,
            stall_kind="user",
        )
        return await _record_and_return(
            ctx, outcome, ok=False, md_recompute=True, persist_err="cancelled_by_user"
        )

    if "stop:term" in msg:
        outcome.ok = False
        outcome.stage = "sigterm_cancel"
        outcome.event = retry_mod.RetryEvent(
            cls="cancel",
            stage="sigterm_cancel",
            error="cancelled_by_sigterm",
            nxdomain_like=False,
            status_code=None,
            stall_kind="sigterm",
        )
        return await _record_and_return(
            ctx,
            outcome,
            ok=False,
            md_recompute=True,
            persist_err="cancelled_by_sigterm",
        )

    if "scheduler:" in msg:
        err = f"cancelled_by_scheduler:{msg}"
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
        return await _record_and_return(
            ctx, outcome, ok=False, md_recompute=True, persist_err=err
        )

    raise e


async def _handle_critical_mem(
    ctx: PipelineCtx,
    outcome: retry_mod.AttemptOutcome,
    e: retry_mod.CriticalMemoryPressure,
) -> bool:
    outcome.ok = False
    outcome.stage = "critical_memory_pressure"
    outcome.event = retry_mod.RetryEvent(
        cls="mem", stage="critical_memory_pressure", error=str(e)
    )
    return await _record_and_return(
        ctx, outcome, ok=False, md_recompute=True, persist_err=str(e)
    )


async def _handle_unhandled(
    ctx: PipelineCtx,
    outcome: retry_mod.AttemptOutcome,
    e: Exception,
) -> bool:
    ev = retry_mod.classify_failure(e, stage="pipeline_unhandled")
    outcome.ok = False
    outcome.stage = ev.stage
    outcome.event = ev
    outcome.md_done = await _get_md_done(ctx, recompute=True)

    if retry_mod.TRACEBACK_ENABLED:
        ctx.clog.exception(
            "Pipeline unhandled exception stage=%s err=%s",
            ev.stage,
            retry_mod.short_exc(e),
        )
    else:
        ctx.clog.error(
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
        ctx.retry_store, ctx.company.company_id, outcome, flush=True
    )
    await _persist_last_error(ctx, str(e))
    return False


async def _finish_finally(ctx: PipelineCtx, outcome: retry_mod.AttemptOutcome) -> None:
    try:
        snap2 = await ctx.state.get_company_snapshot(
            ctx.company.company_id, recompute=False
        )
        await ctx.state.write_company_meta_snapshot(
            ctx.company.company_id,
            snap2,
            pretty=True,
            company_ctx=None,
            set_last_crawled_at=True,
        )
    except Exception:
        pass

    try:
        await ctx.state.recompute_company_from_index(
            ctx.company.company_id,
            name=ctx.company.name,
            root_url=ctx.company.domain_url,
        )
    except Exception:
        pass

    if ctx.run_id is not None:
        if outcome.ok or outcome.terminalized:
            with suppress(Exception):
                await ctx.state.mark_company_completed(
                    ctx.run_id, ctx.company.company_id
                )

    with suppress(Exception):
        await ctx.scheduler.clear_company_stage(
            ctx.company.company_id,
            reset_timers=False,
            reason="pipeline_exit",
        )

    gc.collect()


# --------------------------------------------------------------------------------------
# Public API (unchanged signature)
# --------------------------------------------------------------------------------------
async def run_company_pipeline(
    company: Company,
    *,
    attempt_no: int,
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
    repo_root: Path,
) -> bool:
    token = logging_ext.set_company_context(company.company_id)
    clog = logging_ext.get_company_logger(company.company_id)

    ctx = PipelineCtx(
        company=company,
        attempt_no=int(attempt_no),
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
        repo_root=repo_root,
        clog=clog,
    )

    progress = Progress(
        scheduler=scheduler,
        company_id=company.company_id,
        throttle_sec=float(args.company_progress_throttle_sec),
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

    try:
        plan = await _pipeline_enter(ctx)

        # If no crawl and no llm, short-circuit
        if (not plan.do_crawl) and (not plan.do_llm_requested):
            outcome.ok = True
            outcome.stage = "skip_already_done_or_llm_disabled"
            outcome.should_mark_success = True
            await retry_mod.record_attempt(
                ctx.retry_store, ctx.company.company_id, outcome, flush=True
            )
            return True

        # -------------------- Crawl --------------------
        crawl_res = await _run_crawl_stage(ctx, plan, progress)

        # terminalization path (IMPORTANT: do NOT exit; allow LLM)
        if crawl_res.terminalized:
            _ = await _apply_terminalization_if_any(ctx, crawl_res, outcome)

        # crawl failure path
        if (
            not crawl_res.ok
            and crawl_res.fail_event is not None
            and crawl_res.fail_exc is not None
        ):
            outcome.ok = False
            outcome.stage = crawl_res.fail_event.stage
            outcome.event = crawl_res.fail_event
            return await _record_and_return(
                ctx,
                outcome,
                ok=False,
                md_recompute=True,
                persist_err=str(crawl_res.fail_exc),
            )

        # -------------------- LLM --------------------
        ok_llm = await _run_llm_stage(ctx, plan, progress, outcome)
        if not ok_llm:
            return False

        # -------------------- Finalize --------------------
        return await _finalize_pipeline(ctx, plan, outcome)

    except asyncio.CancelledError as e:
        return await _handle_cancelled(ctx, outcome, e)

    except retry_mod.CriticalMemoryPressure as e:
        return await _handle_critical_mem(ctx, outcome, e)

    except Exception as e:
        return await _handle_unhandled(ctx, outcome, e)

    finally:
        with suppress(Exception):
            await _finish_finally(ctx, outcome)

        logging_ext.reset_company_context(token)
        logging_ext.close_company(company.company_id)
