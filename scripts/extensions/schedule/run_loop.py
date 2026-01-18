from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List

from extensions.crawl.state import CrawlState
from extensions.schedule.adaptive import (
    AdaptiveScheduler,
    compute_retry_exit_code_from_store,
)
from extensions.schedule.retry import RetryStateStore


PipelineRunner = Callable[[str, int, asyncio.Event], Awaitable[bool]]


@dataclass(slots=True)
class _StopState:
    reason: str
    sigint_count: int


async def run_scheduler_loop(
    *,
    args: argparse.Namespace,
    state: CrawlState,
    scheduler: AdaptiveScheduler,
    retry_store: RetryStateStore,
    run_id: str,
    runnable_ids: List[str],
    company_ids_all: List[str],
    company_id_set: set[str],
    inflight_by_cid: Dict[str, asyncio.Task],
    pipeline_runner: PipelineRunner,
    is_company_runnable: Callable[[str], Awaitable[bool]],
) -> int:
    stop_event = asyncio.Event()
    stop_state = _StopState(reason="none", sigint_count=0)

    exclude_quarantined = bool(getattr(scheduler.cfg, "quarantine_enabled", True))

    def _cancel_inflight(tag: str) -> None:
        for t in inflight_by_cid.values():
            if not t.done():
                t.cancel(tag)

    def on_sigint() -> None:
        stop_state.sigint_count += 1
        if stop_state.sigint_count >= 2:
            os._exit(130)
        stop_state.reason = "sigint"
        stop_event.set()
        _cancel_inflight("stop:user")

    def on_sigterm() -> None:
        if scheduler.restart_recommended:
            stop_state.reason = "restart_recommended"
        else:
            stop_state.reason = "sigterm"
        stop_event.set()
        _cancel_inflight("stop:term")

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, on_sigint)
    loop.add_signal_handler(signal.SIGTERM, on_sigterm)

    # Startup cleanup: if quarantine is disabled, DO NOT exclude quarantined IDs here
    pending_retry_ids = set(
        await retry_store.pending_ids(exclude_quarantined=exclude_quarantined)
    )
    for rid in pending_retry_ids:
        if rid not in company_id_set:
            await retry_store.mark_success(
                rid, stage="startup_cleanup", note="orphan_retry_id"
            )
            continue
        if not await is_company_runnable(rid):
            await retry_store.mark_success(
                rid, stage="startup_cleanup", note="already_done_or_not_runnable"
            )

    await scheduler.cleanup_completed_retry_ids(
        is_company_runnable=is_company_runnable,
        treat_non_runnable_as_done=True,
    )

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

    cap = max(1, int(args.company_concurrency))

    free_crawlers_for_sched = int(args.crawler_pool_size)
    free_crawlers_for_sched = max(1, free_crawlers_for_sched)

    async def drain_reconcile_and_reseed() -> bool:
        await state.recompute_all_in_progress(concurrency=32)
        unfinished = [cid for cid in company_ids_all if await is_company_runnable(cid)]
        if not unfinished:
            return True

        added = int(
            await scheduler.ensure_worklist(unfinished, reason="drain_reconcile")
        )
        if added <= 0:
            for cid in unfinished:
                await scheduler.requeue_company(
                    cid, force=True, reason="drain_reconcile"
                )

        await force_write_global_state()
        return False

    try:
        while not stop_event.is_set():
            finished = [cid for cid, t in inflight_by_cid.items() if t.done()]
            for cid in finished:
                t = inflight_by_cid.pop(cid)
                try:
                    _ = bool(t.result())
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                scheduler.register_company_completed()

            await maybe_write_global_state()

            if scheduler.restart_recommended:
                stop_state.reason = "restart_recommended"
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

                    attempt_no = await state.mark_company_attempt_started(run_id, cid)

                    async def _runner(cid_: str, attempt_no_: int) -> bool:
                        return await pipeline_runner(cid_, attempt_no_, stop_event)

                    inflight_by_cid[cid] = asyncio.create_task(
                        _runner(cid, attempt_no), name=f"company:{cid}"
                    )

            if not inflight_by_cid and not scheduler.has_pending():
                done = await drain_reconcile_and_reseed()
                if done:
                    break
                continue

            await asyncio.sleep(max(0.05, min(0.5, float(scheduler.sleep_hint_sec()))))

    finally:
        if stop_state.reason in ("sigint", "sigterm", "restart_recommended"):
            _cancel_inflight("stop:final")

        if inflight_by_cid:
            await asyncio.gather(*inflight_by_cid.values(), return_exceptions=True)

        with contextlib.suppress(Exception):
            await force_write_global_state()

        if stop_state.reason == "sigint":
            return 130
        if stop_state.reason == "sigterm":
            return 143
        if stop_state.reason == "restart_recommended":
            return int(args.retry_exit_code)

        await scheduler.cleanup_completed_retry_ids(
            is_company_runnable=is_company_runnable,
            treat_non_runnable_as_done=True,
        )

        with contextlib.suppress(Exception):
            done = await drain_reconcile_and_reseed()
            if done:
                await scheduler.cleanup_completed_retry_ids(
                    is_company_runnable=is_company_runnable,
                    treat_non_runnable_as_done=True,
                )

        if not scheduler.has_pending():
            return 0

        return int(
            compute_retry_exit_code_from_store(
                scheduler.retry_store,
                retry_exit_code=int(args.retry_exit_code),
                exclude_quarantined=exclude_quarantined,
            )
        )
