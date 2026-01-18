from __future__ import annotations

import argparse
import asyncio
import logging
import time
from contextlib import suppress
from typing import Dict, Iterable, List, Optional, Sequence

from cli.deep_crawl.args import parse_args

from configs.language import default_language_factory
from extensions.crawl.runtime_context import build_runner_context
from extensions.crawl.workload import (
    build_dataset_externals_for_companies,
    build_scoring_and_strategy,
    load_companies,
    prepare_run_and_state,
)
from extensions.pipeline.company_pipeline import run_company_pipeline
from extensions.schedule.adaptive import AdaptiveScheduler
from extensions.schedule.run_loop import run_scheduler_loop

logger = logging.getLogger("deep_crawl_runner")


def _fmt_bool(x: object) -> str:
    return "true" if bool(x) else "false"


def _summarize_llm_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "llm_mode", "none") or "none")
    if mode == "none":
        return "llm=disabled (crawl-only)"
    if mode == "presence":
        return "llm=presence (crawl -> llm presence)"
    if mode == "full":
        return "llm=full (crawl -> llm full)"
    return f"llm={mode!r} (crawl -> llm)"


def _resolve_run_name(args: argparse.Namespace) -> str:
    """
    This name is persisted into crawl_global_state.json (latest_run.pipeline).
    It must reflect the effective pipeline mode (crawl-only vs llm presence/full).
    """
    mode = str(getattr(args, "llm_mode", "none") or "none").strip() or "none"
    if mode == "none":
        return "deep_crawl"
    if mode == "presence":
        return "deep_crawl_llm_presence"
    if mode == "full":
        return "deep_crawl_llm_full"
    return f"deep_crawl_llm_{mode}"


async def main_async(args: argparse.Namespace) -> int:
    t0 = time.monotonic()

    logger.info("runner_start %s", _summarize_llm_mode(args))
    logger.info(
        "runner_args finalize_in_progress_md=%s llm_mode=%s industry_enrichment=%s no_industry_enrichment=%s "
        "strategy=%s lang=%s max_pages=%s company_concurrency=%s crawler_pool_size=%s crawl4ai_cache_mode=%s",
        _fmt_bool(getattr(args, "finalize_in_progress_md", False)),
        str(getattr(args, "llm_mode", None)),
        _fmt_bool(getattr(args, "industry_enrichment", False)),
        _fmt_bool(getattr(args, "no_industry_enrichment", False)),
        str(getattr(args, "strategy", None)),
        str(getattr(args, "lang", None)),
        str(getattr(args, "max_pages", None)),
        str(getattr(args, "company_concurrency", None)),
        str(getattr(args, "crawler_pool_size", None)),
        str(getattr(args, "crawl4ai_cache_mode", None)),
    )

    if args.finalize_in_progress_md and args.llm_mode != "none":
        logger.warning("--finalize-in-progress-md forces --llm-mode none")
        args.llm_mode = "none"

    if args.llm_mode != "none" and (not bool(args.no_industry_enrichment)):
        if not bool(args.industry_enrichment):
            logger.info(
                "Auto-enabling --industry-enrichment because --llm-mode != none "
                "(override with --no-industry-enrichment)."
            )
            args.industry_enrichment = True

    run_name = _resolve_run_name(args)

    logger.info("runner_effective %s", _summarize_llm_mode(args))
    logger.info("runner_pipeline_name %s", run_name)
    logger.info(
        "runner_policy terminal_done_treated_as_markdown_done=true llm_gating_ignores_last_error=true"
    )

    t_ctx0 = time.monotonic()
    ctx = await build_runner_context(args)

    # build_runner_context initializes language; log the additive effective langs explicitly.
    lang_effective = default_language_factory.effective_langs()
    logger.info(
        "language_effective lang_target=%s lang_effective=%s",
        str(getattr(args, "lang", None)),
        ",".join(lang_effective),
    )

    logger.info(
        "context_ready elapsed=%.2fs crawler_pool=%s guard=%s state=%s llm_cache=%s quarantine_enabled=%s last_one_standing_quarantine_enabled=%s",
        time.monotonic() - t_ctx0,
        type(ctx.crawler_pool).__name__,
        type(ctx.guard).__name__,
        type(ctx.state).__name__,
        "present" if ctx.industry_llm_cache is not None else "none",
        _fmt_bool(getattr(ctx.scheduler_cfg, "quarantine_enabled", True)),
        _fmt_bool(
            getattr(ctx.scheduler_cfg, "last_one_standing_quarantine_enabled", True)
        ),
    )

    t_load0 = time.monotonic()
    companies = load_companies(args)
    logger.info(
        "companies_loaded n=%d elapsed=%.2fs",
        len(companies),
        time.monotonic() - t_load0,
    )

    t_state0 = time.monotonic()
    (
        run_id,
        companies,
        companies_by_id,
        company_ids_all,
        company_id_set,
    ) = await prepare_run_and_state(
        state=ctx.state,
        args=args,
        companies=companies,
        cache_dir=args.crawl4ai_cache_dir,
        run_name=run_name,
    )
    logger.info(
        "run_prepared run_id=%s pipeline=%s companies_in_run=%d company_ids_all=%d company_id_set=%d elapsed=%.2fs",
        str(run_id),
        run_name,
        len(companies),
        len(company_ids_all),
        len(company_id_set),
        time.monotonic() - t_state0,
    )

    t_ext0 = time.monotonic()
    dataset_externals = build_dataset_externals_for_companies(
        args=args, companies=companies
    )
    logger.info(
        "dataset_externals_built unique=%d elapsed=%.2fs",
        len(dataset_externals),
        time.monotonic() - t_ext0,
    )

    t_score0 = time.monotonic()
    url_scorer, bm25_filter, dfs_factory = build_scoring_and_strategy(args=args)
    logger.info(
        "scoring_strategy_ready elapsed=%.2fs url_scorer=%s bm25_filter=%s dfs_factory=%s",
        time.monotonic() - t_score0,
        type(url_scorer).__name__ if url_scorer is not None else "none",
        type(bm25_filter).__name__ if bm25_filter is not None else "none",
        type(dfs_factory).__name__ if dfs_factory is not None else "none",
    )

    runnable_debug_seen = 0

    async def is_company_runnable(cid: str) -> bool:
        nonlocal runnable_debug_seen

        if cid not in company_id_set:
            if logger.isEnabledFor(logging.DEBUG) and runnable_debug_seen < 25:
                runnable_debug_seen += 1
                logger.debug(
                    "runnable_check cid=%s in_set=false => runnable=false", cid
                )
            return False

        snap = await ctx.state.get_company_snapshot(cid, recompute=True)
        st = (getattr(snap, "status", None) or "").strip() or "pending"
        md_done = int(getattr(snap, "urls_markdown_done", 0) or 0)
        last_err = (getattr(snap, "last_error", None) or "").strip()

        if args.llm_mode != "none":
            runnable = st != "llm_done"
            if logger.isEnabledFor(logging.DEBUG) and (
                runnable_debug_seen < 50 or runnable_debug_seen % 10000 == 0
            ):
                runnable_debug_seen += 1
                logger.debug(
                    "runnable_check llm_enabled=true cid=%s status=%s md_done=%d last_error_len=%d => runnable=%s "
                    "(policy: terminal_done_allowed; last_error_ignored)",
                    cid,
                    st,
                    md_done,
                    len(last_err),
                    _fmt_bool(runnable),
                )
            return runnable

        runnable = st not in ("markdown_done", "terminal_done")
        if logger.isEnabledFor(logging.DEBUG) and (
            runnable_debug_seen < 50 or runnable_debug_seen % 10000 == 0
        ):
            runnable_debug_seen += 1
            logger.debug(
                "runnable_check llm_enabled=false cid=%s status=%s md_done=%d => runnable=%s",
                cid,
                st,
                md_done,
                _fmt_bool(runnable),
            )
        return runnable

    t_filter0 = time.monotonic()
    runnable_ids: List[str] = [
        cid for cid in company_ids_all if await is_company_runnable(cid)
    ]
    logger.info(
        "runnable_filtered pipeline=%s llm_mode=%s runnable=%d of_total=%d elapsed=%.2fs",
        run_name,
        str(args.llm_mode),
        len(runnable_ids),
        len(company_ids_all),
        time.monotonic() - t_filter0,
    )
    await ctx.state.update_run_totals(run_id, total_companies=len(runnable_ids))
    logger.info(
        "run_totals_updated run_id=%s total_companies=%d",
        str(run_id),
        len(runnable_ids),
    )

    inflight_by_cid: Dict[str, asyncio.Task] = {}

    def get_active_company_ids() -> Sequence[str]:
        return [cid for cid, t in inflight_by_cid.items() if not t.done()]

    def request_cancel_companies(ids: Sequence[str]) -> None:
        if not ids:
            return
        logger.info("scheduler_cancel_request n=%d", len(ids))
        for cid in ids:
            t = inflight_by_cid.get(cid)
            if t is not None and (not t.done()):
                t.cancel("scheduler:cancel")

    async def request_recycle_idle(count: int, reason: str) -> int:
        logger.debug(
            "request_recycle_idle count=%d reason=%s (noop)", int(count), str(reason)
        )
        return 0

    scheduler = AdaptiveScheduler(
        cfg=ctx.scheduler_cfg,
        get_active_company_ids=get_active_company_ids,
        request_cancel_companies=request_cancel_companies,
        request_recycle_idle=request_recycle_idle,
    )
    await scheduler.start()
    logger.info("scheduler_started cfg=%s", type(ctx.scheduler_cfg).__name__)

    retry_store = scheduler.retry_store
    logger.info("retry_store_ready type=%s", type(retry_store).__name__)

    async def pipeline_runner(
        cid: str, attempt_no: int, stop_event: asyncio.Event
    ) -> bool:
        company = companies_by_id.get(cid)
        assert company is not None

        logger.debug(
            "pipeline_dispatch cid=%s attempt=%d pipeline=%s llm_mode=%s url=%s",
            cid,
            int(attempt_no),
            run_name,
            str(args.llm_mode),
            str(getattr(company, "domain_url", "")),
        )

        return await run_company_pipeline(
            company,
            attempt_no=int(attempt_no),
            logging_ext=ctx.logging_ext,
            state=ctx.state,
            guard=ctx.guard,
            crawler_pool=ctx.crawler_pool,
            args=args,
            dataset_externals=dataset_externals,
            url_scorer=url_scorer,
            bm25_filter=bm25_filter,
            run_id=run_id,
            industry_llm_cache=ctx.industry_llm_cache,
            dfs_factory=dfs_factory,
            crawler_base_cfg=ctx.crawler_base_cfg,
            page_policy=ctx.page_policy,
            page_interaction_factory=ctx.page_interaction_factory,
            retry_store=retry_store,
            scheduler=scheduler,
            stop_event=stop_event,
            repo_root=ctx.repo_root,
        )

    try:
        logger.info(
            "scheduler_loop_enter run_id=%s pipeline=%s runnable=%d company_concurrency=%s crawler_pool_size=%s %s",
            str(run_id),
            run_name,
            len(runnable_ids),
            str(getattr(args, "company_concurrency", None)),
            str(getattr(args, "crawler_pool_size", None)),
            _summarize_llm_mode(args),
        )

        if str(args.llm_mode) == "full":
            logger.info(
                "llm_pipeline_plan mode=full stages=[crawl, llm_full] policy_terminal_done_allows_llm=true"
            )

        exit_code = await run_scheduler_loop(
            args=args,
            state=ctx.state,
            scheduler=scheduler,
            retry_store=retry_store,
            run_id=run_id,
            runnable_ids=runnable_ids,
            company_ids_all=company_ids_all,
            company_id_set=company_id_set,
            inflight_by_cid=inflight_by_cid,
            pipeline_runner=pipeline_runner,
            is_company_runnable=is_company_runnable,
        )
        logger.info(
            "scheduler_loop_exit run_id=%s pipeline=%s exit_code=%s elapsed_total=%.2fs",
            str(run_id),
            run_name,
            str(exit_code),
            time.monotonic() - t0,
        )
        return int(exit_code)
    finally:
        logger.info("runner_shutdown begin")
        with suppress(Exception):
            await scheduler.stop()
            logger.info("scheduler_stopped")
        with suppress(Exception):
            await ctx.guard.stop()
            logger.info("guard_stopped")
        if ctx.resource_monitor is not None:
            with suppress(Exception):
                ctx.resource_monitor.stop()
                logger.info("resource_monitor_stopped")
        logger.info("runner_shutdown done elapsed_total=%.2fs", time.monotonic() - t0)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    exit_code = asyncio.run(main_async(args))
    raise SystemExit(int(exit_code))


if __name__ == "__main__":
    main()
