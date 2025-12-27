from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence
from urllib.parse import urlparse

from crawl4ai.deep_crawling.filters import FilterChain

# Repo config factories
from configs.browser import default_browser_factory
from configs.crawler import default_crawler_factory
from configs.deep_crawl import (
    DFSDeepCrawlStrategyProvider,
    DeepCrawlStrategyFactory,
    build_deep_strategy,
)
from configs.js_injection import (
    PageInteractionFactory,
    PageInteractionPolicy,
    default_page_interaction_factory,
)
from configs.md import default_md_factory
from configs.language import default_language_factory
from configs.llm import (
    DEFAULT_FULL_INSTRUCTION,
    DEFAULT_PRESENCE_INSTRUCTION,
    IndustryStrategyCache,
    LLMExtractionFactory,
    provider_strategy_from_llm_model_selector,
)

# Extensions
from extensions.load_source import CompanyInput, load_companies_from_source
from extensions.logging import LoggingExtension
from extensions import md_gating
from extensions import output_paths
from extensions.output_paths import ensure_company_dirs
from extensions.resource_monitor import ResourceMonitor, ResourceMonitorConfig
from extensions.retry_state import RetryStateStore
from extensions.adaptive_scheduling import (
    AdaptiveScheduler,
    AdaptiveSchedulingConfig,
    compute_retry_exit_code_from_store,
)
from extensions.connectivity_guard import ConnectivityGuard
from extensions.crawl_state import (
    get_crawl_state,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_TERMINAL_DONE,
)

from extensions.filtering import (
    FirstTimeURLFilter,
    HTMLContentFilter,
    LanguageAwareURLFilter,
    UniversalExternalFilter,
)
from extensions.dataset_external import build_dataset_externals
from extensions.dual_bm25 import (
    build_dual_bm25_components,
    DualBM25Scorer,
    DualBM25Filter,
)
from extensions.llm_passes import (
    run_presence_pass_for_company,
    run_full_pass_for_company,
)

from extensions.crawler_pool import CrawlerPool
from extensions.crawl_runner import CrawlRunnerConfig, run_company_crawl
from extensions.retry_policy import (
    RetryEvent,
    classify_failure,
    CriticalMemoryPressure,
    CrawlerTimeoutError,
    should_fail_fast_on_goto,
)
from extensions.terminalization import decide_from_page_summary
from extensions.oom_guard import OOMGuard, OOMGuardConfig


logger = logging.getLogger("deep_crawl_runner")

RETRY_EXIT_CODE = 17
_forced_exit_code: Optional[int] = None
_retry_store_instance: Optional[RetryStateStore] = None

_HTTP_LANG_MAP: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
}

_UNCLASSIFIED_INDUSTRY_LABEL = "unclassified"


def _normalize_industry_code(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _industry_label(code: Optional[str]) -> str:
    return code or _UNCLASSIFIED_INDUSTRY_LABEL


def _set_output_root(out_dir: str | Path) -> Path:
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    output_paths.set_output_root(str(p))
    return p


def _global_path(name: str) -> Path:
    return Path(output_paths.global_path(name))


@dataclass(slots=True)
class Company:
    company_id: str
    domain_url: str
    name: Optional[str] = None
    industry_code: Optional[str] = None
    industry_label: str = _UNCLASSIFIED_INDUSTRY_LABEL
    metadata: Dict[str, Any] = field(default_factory=dict)


def _company_from_input(ci: CompanyInput) -> Company:
    md = dict(ci.metadata or {})
    raw_primary = md.get("industry_primary")
    raw_industry = md.get("industry")
    raw_code = md.get("industry_code") or md.get("industryCode")

    code = _normalize_industry_code(
        raw_primary if raw_primary is not None else raw_industry
    )
    if code is None:
        code = _normalize_industry_code(raw_code)

    return Company(
        company_id=str(ci.bvdid),
        domain_url=str(ci.url),
        name=(str(getattr(ci, "name", "")).strip() or None),
        industry_code=code,
        industry_label=_industry_label(code),
        metadata=md,
    )


def _companies_from_source(path: Path) -> List[Company]:
    return [_company_from_input(ci) for ci in load_companies_from_source(path)]


def _read_in_progress_companies(out_dir: Path) -> List[str]:
    p = out_dir / "crawl_global_state.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        str(x).strip()
        for x in (data.get("in_progress_companies") or [])
        if str(x).strip()
    ]


def _build_filter_chain(
    company: Company,
    *,
    lang: str,
    dataset_externals: frozenset[str],
    bm25_filter: Optional[DualBM25Filter],
) -> FilterChain:
    universal = UniversalExternalFilter(
        dataset_externals=sorted(dataset_externals),
        name=f"UniversalExternalFilter[{company.company_id}]",
    )
    universal.set_company_url(company.domain_url)

    filters = [
        HTMLContentFilter(),
        FirstTimeURLFilter(),
        universal,
        LanguageAwareURLFilter(lang_code=lang),
    ]
    if bm25_filter is not None:
        filters.append(bm25_filter)
    return FilterChain(filters)


def _should_skip_company(status: str, llm_mode: str) -> bool:
    # "skip" here means company is already in a terminal-enough state for this run mode.
    if llm_mode == "none":
        return status in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            COMPANY_STATUS_TERMINAL_DONE,
        )
    return status in (COMPANY_STATUS_LLM_DONE, COMPANY_STATUS_TERMINAL_DONE)


def _crawl_meta_path(company_id: str) -> Path:
    dirs = ensure_company_dirs(company_id)
    meta_dir = (
        dirs.get("metadata")
        or dirs.get("checkpoints")
        or (_global_path(company_id) / "metadata")
    )
    Path(meta_dir).mkdir(parents=True, exist_ok=True)
    return Path(meta_dir) / "crawl_meta.json"


def _write_crawl_meta(company: Company, snapshot: Any) -> None:
    path = _crawl_meta_path(company.company_id)
    payload = {
        "company_id": company.company_id,
        "name": company.name,
        "industry_code": company.industry_code,
        "industry_label": company.industry_label,
        "root_url": company.domain_url,
        "status": getattr(snapshot, "status", None),
        "urls_total": getattr(snapshot, "urls_total", None),
        "urls_markdown_done": getattr(snapshot, "urls_markdown_done", None),
        "urls_llm_done": getattr(snapshot, "urls_llm_done", None),
        "last_error": getattr(snapshot, "last_error", None),
        "last_crawled_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _scheduler_progress(scheduler: AdaptiveScheduler, cid: str, *, kind: str) -> None:
    """
    Best-effort progress signal wiring (works across small API drifts).
    """
    fn = getattr(scheduler, "progress_company", None)
    if callable(fn):
        try:
            fn(cid)
            return
        except Exception:
            pass

    fn2 = getattr(scheduler, "touch_company", None)
    if callable(fn2):
        try:
            fn2(cid, kind=kind)
            return
        except Exception:
            pass


def _scheduler_pending_total(scheduler: AdaptiveScheduler) -> int:
    fn = getattr(scheduler, "pending_total", None)
    if callable(fn):
        try:
            return int(fn() or 0)
        except Exception:
            return 0
    try:
        return 1 if bool(scheduler.has_pending()) else 0
    except Exception:
        return 0


def _retry_pending_total(retry_store: RetryStateStore) -> int:
    fn = getattr(retry_store, "pending_total", None)
    if callable(fn):
        try:
            return int(fn(exclude_quarantined=True) or 0)
        except TypeError:
            try:
                return int(fn() or 0)
            except Exception:
                return 0
        except Exception:
            return 0
    return 0


async def _db_in_progress(state: Any) -> bool:
    """
    Authoritative DB-backed check, relying on CrawlState.has_in_progress_companies().
    """
    fn = getattr(state, "has_in_progress_companies", None)
    if not callable(fn):
        return True
    try:
        v = fn(include_pending=True)
        if asyncio.iscoroutine(v):
            v = await v
        return bool(v)
    except TypeError:
        try:
            v2 = fn()
            if asyncio.iscoroutine(v2):
                v2 = await v2
            return bool(v2)
        except Exception:
            return True
    except Exception:
        return True


def _retry_has_pending_for_company_best_effort(
    retry_store: RetryStateStore, cid: str
) -> Optional[bool]:
    for name in ("has_pending", "company_has_pending", "is_pending"):
        fn = getattr(retry_store, name, None)
        if callable(fn):
            try:
                return bool(fn(cid))
            except Exception:
                pass
    for name in ("pending_ids", "get_pending_ids"):
        fn = getattr(retry_store, name, None)
        if callable(fn):
            try:
                ids = fn()
                return cid in set(ids or [])
            except Exception:
                pass
    return None


def _best_effort_next_eligible_delay_sec(
    retry_store: RetryStateStore, cid: str, *, default_delay_sec: float = 5.0
) -> float:
    """
    If RetryStateStore exposes next eligible timestamp, return max(0, ts-now),
    else default_delay_sec.
    """
    now = time.time()
    for fn_name in ("next_eligible_at", "get_next_eligible_at", "next_eligible_ts"):
        fn = getattr(retry_store, fn_name, None)
        if callable(fn):
            try:
                ts = fn(cid)
                if ts is None:
                    break
                ts_f = float(ts)
                return max(0.0, ts_f - now)
            except Exception:
                break
    return float(default_delay_sec)


async def _best_effort_requeue_company(
    scheduler: AdaptiveScheduler,
    cid: str,
    *,
    delay_sec: float,
    reason: str,
) -> None:
    """
    Uses the new scheduler requeue API if present.
    """
    fn = getattr(scheduler, "requeue_company", None)
    if callable(fn):
        try:
            v = fn(cid, delay_sec=delay_sec, reason=reason)
            if asyncio.iscoroutine(v):
                await v
            return
        except TypeError:
            # signature drift fallback
            try:
                v2 = fn(cid)
                if asyncio.iscoroutine(v2):
                    await v2
                return
            except Exception:
                return
        except Exception:
            return


async def run_company_pipeline(
    company: Company,
    *,
    attempt_no: int,
    total_unique: int,
    done_counter: MutableMapping[str, int],
    logging_ext: LoggingExtension,
    state: Any,
    guard: ConnectivityGuard,
    crawler_pool: CrawlerPool,
    args: argparse.Namespace,
    dataset_externals: frozenset[str],
    url_scorer: Optional[DualBM25Scorer],
    bm25_filter: Optional[DualBM25Filter],
    run_id: Optional[str],
    industry_llm_cache: Optional[IndustryStrategyCache],
    dfs_factory: Optional[DeepCrawlStrategyFactory],
    crawler_base_cfg: Any,
    page_policy: PageInteractionPolicy,
    page_interaction_factory: PageInteractionFactory,
    retry_store: RetryStateStore,
    scheduler: AdaptiveScheduler,
    cancel_reason_by_cid: Optional[MutableMapping[str, str]] = None,
) -> bool:
    token = logging_ext.set_company_context(company.company_id)
    clog = logging_ext.get_company_logger(company.company_id)

    # Throttled progress signal (avoid super-spam)
    progress_throttle_sec = float(getattr(args, "company_progress_throttle_sec", 12.0))
    last_progress_mono = 0.0

    def signal_progress(kind: str = "progress") -> None:
        nonlocal last_progress_mono
        now = time.monotonic()
        if (now - last_progress_mono) < progress_throttle_sec and kind == "progress":
            return
        last_progress_mono = now
        _scheduler_progress(scheduler, company.company_id, kind=kind)

    async def _get_md_done(*, recompute: bool) -> int:
        try:
            snapx = await state.get_company_snapshot(
                company.company_id, recompute=recompute
            )
            return int(getattr(snapx, "urls_markdown_done", 0) or 0)
        except Exception:
            return 0

    async def _persist_last_error(err: str) -> None:
        with contextlib.suppress(Exception):
            await state.upsert_company(
                company.company_id,
                last_error=(err or "")[:4000],
                name=company.name,
                root_url=company.domain_url,
            )

    completed_ok = False
    watchdog_task: Optional[asyncio.Task] = None

    try:
        snap = await state.get_company_snapshot(company.company_id)
        status = snap.status or COMPANY_STATUS_PENDING
        urls_md_done0 = int(getattr(snap, "urls_markdown_done", 0) or 0)

        finalize_in_progress_md = bool(getattr(args, "finalize_in_progress_md", False))
        will_llm = (
            not finalize_in_progress_md
            and args.llm_mode in ("presence", "full")
            and industry_llm_cache is not None
        )

        do_crawl = status in (COMPANY_STATUS_PENDING, COMPANY_STATUS_MD_NOT_DONE)
        if not do_crawl and not will_llm:
            completed_ok = True
            return True

        # Better counters: attempt is attempt_no; progress is done/total_unique
        done_now = int(done_counter.get("done", 0))
        clog.info(
            "=== [done=%d/%d attempt=%d] company_id=%s url=%s industry=%s status=%s llm=%s ===",
            done_now,
            total_unique,
            attempt_no,
            company.company_id,
            company.domain_url,
            company.industry_label,
            status,
            args.llm_mode,
        )

        filter_chain = _build_filter_chain(
            company,
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

        # ---------------- Crawl stage ----------------
        if do_crawl:
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

            lease_timeout = float(args.crawler_lease_timeout_sec)
            company_crawl_timeout_sec = float(args.company_crawl_timeout_sec)
            hard_max_pages = int(args.max_pages) if args.max_pages else None

            runner_cfg = CrawlRunnerConfig(
                page_result_concurrency=int(args.page_result_concurrency),
                page_queue_maxsize=int(args.page_queue_maxsize),
                url_index_flush_every=int(args.url_index_flush_every),
                url_index_flush_interval_sec=float(args.url_index_flush_interval_sec),
                url_index_queue_maxsize=int(args.url_index_queue_maxsize),
                arun_init_timeout_sec=float(args.arun_init_timeout_sec),
                stream_no_yield_timeout_sec=float(args.stream_no_yield_timeout_sec),
                submit_timeout_sec=float(args.submit_timeout_sec),
                direct_fetch_total_timeout_sec=float(args.direct_fetch_url_timeout_sec),
                processor_finish_timeout_sec=float(args.processor_finish_timeout_sec),
                generator_close_timeout_sec=float(args.generator_close_timeout_sec),
                hard_max_pages=hard_max_pages,
                page_timeout_ms=int(args.page_timeout_ms)
                if args.page_timeout_ms
                else None,
                direct_fetch_urls=bool(direct_fetch_urls),
            )

            # Watchdog: emit *heartbeat* while crawl is alive.
            # Heartbeat should NOT reset inactivity/progress timers; only real crawl events should.
            heartbeat_sec = float(getattr(args, "company_progress_heartbeat_sec", 30.0))

            async def _watchdog() -> None:
                try:
                    while True:
                        await asyncio.sleep(heartbeat_sec)
                        signal_progress("heartbeat")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return

            watchdog_task = asyncio.create_task(
                _watchdog(), name=f"watchdog:{company.company_id}"
            )

            last_exc: Optional[BaseException] = None
            last_event: Optional[RetryEvent] = None

            terminalize = False
            terminal_reason = ""
            terminal_last_error: Optional[str] = None

            for attempt in range(2):
                try:
                    lease_task = asyncio.create_task(
                        crawler_pool.lease(), name=f"lease:{company.company_id}"
                    )
                    lease = await asyncio.wait_for(lease_task, timeout=lease_timeout)

                    async with lease as crawler:

                        async def _do() -> Any:
                            try:
                                kwargs: Dict[str, Any] = {
                                    "company": company,
                                    "crawler": crawler,
                                    "deep_strategy": deep_strategy,
                                    "guard": guard,
                                    "gating_cfg": md_gating.build_gating_config(),
                                    "crawler_base_cfg": crawler_base_cfg,
                                    "page_policy": page_policy,
                                    "page_interaction_factory": page_interaction_factory,
                                    "root_urls": resume_roots,
                                    "cfg": runner_cfg,
                                }

                                # Only real crawl events should signal "progress"
                                progress_cb = lambda: signal_progress("progress")

                                try:
                                    return await run_company_crawl(  # type: ignore[misc]
                                        **kwargs,
                                        on_progress=progress_cb,  # type: ignore[arg-type]
                                    )
                                except TypeError:
                                    try:
                                        return await run_company_crawl(  # type: ignore[misc]
                                            **kwargs,
                                            progress_cb=progress_cb,  # type: ignore[arg-type]
                                        )
                                    except TypeError:
                                        try:
                                            return await run_company_crawl(  # type: ignore[misc]
                                                **kwargs,
                                                heartbeat_cb=progress_cb,  # type: ignore[arg-type]
                                            )
                                        except TypeError:
                                            return await run_company_crawl(**kwargs)
                            except asyncio.CancelledError:
                                raise

                        summary = await asyncio.wait_for(
                            _do(), timeout=company_crawl_timeout_sec
                        )

                        dec = decide_from_page_summary(summary)
                        if dec.action == "mem":
                            lease.mark_fatal("page_pipeline_mem")
                            raise CriticalMemoryPressure(
                                dec.reason, severity="critical"
                            )

                        if dec.action == "terminal" and urls_md_done0 == 0:
                            terminalize = True
                            terminal_reason = dec.reason
                            terminal_last_error = dec.reason
                            break

                        if dec.action == "stall":
                            raise CrawlerTimeoutError(
                                dec.reason,
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
                    stage = getattr(e, "stage", "crawl")
                    if stage == "crawl" and "goto" in str(e).lower():
                        stage = "goto"

                    last_exc = e
                    last_event = classify_failure(e, stage=stage)
                    if last_event.cls == "net":
                        guard.record_transport_error()

                    if should_fail_fast_on_goto(e, stage=stage):
                        if urls_md_done0 == 0:
                            terminalize = True
                            terminal_reason = f"goto_fail_fast: {e}"
                            terminal_last_error = str(e)
                        break

                    if attempt == 0:
                        await asyncio.sleep(0.5)

            if terminalize:
                with contextlib.suppress(Exception):
                    await state.mark_company_terminal(
                        company.company_id,
                        reason="terminalize",
                        details={
                            "reason": terminal_reason,
                            "industry_code": company.industry_code,
                        },
                        last_error=terminal_last_error,
                        name=company.name,
                        root_url=company.domain_url,
                    )
                with contextlib.suppress(Exception):
                    retry_store.mark_success(
                        company.company_id, stage="terminal_done", note="terminalize"
                    )
                completed_ok = True
                return True

            if last_exc is not None and last_event is not None:
                md_done = await _get_md_done(recompute=True)
                with contextlib.suppress(Exception):
                    retry_store.mark_failure(
                        company.company_id,
                        cls=last_event.cls,
                        error=last_event.error,
                        stage=last_event.stage,
                        status_code=last_event.status_code,
                        nxdomain_like=last_event.nxdomain_like,
                        md_done=md_done,
                    )
                await _persist_last_error(str(last_exc))
                return False

            await state.recompute_company_from_index(
                company.company_id, name=company.name, root_url=company.domain_url
            )

        # ---------------- LLM stage ----------------
        if will_llm:
            llm_exc: Optional[BaseException] = None
            llm_event: Optional[RetryEvent] = None

            llm_stage = f"llm_{args.llm_mode}"
            for attempt in range(2):
                try:
                    assert industry_llm_cache is not None
                    if args.llm_mode == "presence":
                        strat = industry_llm_cache.get_strategy(
                            mode="presence", industry_code=company.industry_code
                        )
                        await run_presence_pass_for_company(
                            company, presence_strategy=strat
                        )
                    elif args.llm_mode == "full":
                        strat = industry_llm_cache.get_strategy(
                            mode="schema", industry_code=company.industry_code
                        )
                        await run_full_pass_for_company(company, full_strategy=strat)
                    llm_exc = None
                    llm_event = None
                    break
                except Exception as e:
                    llm_exc = e
                    llm_event = classify_failure(e, stage=llm_stage)
                    if attempt == 0:
                        await asyncio.sleep(0.75)

            if llm_exc is not None and llm_event is not None:
                md_done = await _get_md_done(recompute=True)
                with contextlib.suppress(Exception):
                    retry_store.mark_failure(
                        company.company_id,
                        cls=llm_event.cls,
                        error=llm_event.error,
                        stage=llm_event.stage,
                        status_code=llm_event.status_code,
                        nxdomain_like=llm_event.nxdomain_like,
                        md_done=md_done,
                    )
                await _persist_last_error(str(llm_exc))
                return False

        # ---------------- Post-check: "done" means truly terminal for this run mode ----------------
        snap_end = await state.get_company_snapshot(company.company_id, recompute=True)
        st_end = getattr(snap_end, "status", None) or COMPANY_STATUS_PENDING

        if args.llm_mode == "none" or bool(
            getattr(args, "finalize_in_progress_md", False)
        ):
            ok_statuses = (COMPANY_STATUS_MD_DONE, COMPANY_STATUS_TERMINAL_DONE)
            if st_end not in ok_statuses:
                md_done = int(getattr(snap_end, "urls_markdown_done", 0) or 0)
                with contextlib.suppress(Exception):
                    retry_store.mark_failure(
                        company.company_id,
                        cls="partial",
                        error=f"incomplete_status={st_end}",
                        stage="postcheck_md",
                        md_done=md_done,
                    )
                return False
        else:
            ok_statuses = (COMPANY_STATUS_LLM_DONE, COMPANY_STATUS_TERMINAL_DONE)
            if st_end not in ok_statuses:
                md_done = int(getattr(snap_end, "urls_markdown_done", 0) or 0)
                with contextlib.suppress(Exception):
                    retry_store.mark_failure(
                        company.company_id,
                        cls="partial",
                        error=f"incomplete_status={st_end}",
                        stage="postcheck_llm",
                        md_done=md_done,
                    )
                return False

        completed_ok = True
        return True

    except asyncio.CancelledError:
        is_sched_cancel = (
            cancel_reason_by_cid is not None
            and cancel_reason_by_cid.pop(company.company_id, None) == "scheduler_cancel"
        )

        if is_sched_cancel:
            with contextlib.suppress(Exception):
                await state.recompute_company_from_index(
                    company.company_id, name=company.name, root_url=company.domain_url
                )
            md_done = await _get_md_done(recompute=False)
            with contextlib.suppress(Exception):
                retry_store.mark_transient_cancel(
                    company.company_id,
                    reason="cancelled by scheduler",
                    stage="scheduler_cancel",
                    md_done=md_done,
                )
        raise

    except CriticalMemoryPressure as e:
        md_done = await _get_md_done(recompute=True)
        with contextlib.suppress(Exception):
            retry_store.mark_failure(
                company.company_id,
                cls="mem",
                error=str(e),
                stage="critical_memory_pressure",
                md_done=md_done,
            )
        with contextlib.suppress(Exception):
            await state.upsert_company(
                company.company_id,
                last_error=(str(e) or "")[:4000],
                name=company.name,
                root_url=company.domain_url,
            )
        return False

    except Exception as e:
        ev = classify_failure(e, stage="pipeline_unhandled")
        md_done = await _get_md_done(recompute=True)
        with contextlib.suppress(Exception):
            retry_store.mark_failure(
                company.company_id,
                cls=ev.cls,
                error=ev.error,
                stage=ev.stage,
                status_code=ev.status_code,
                nxdomain_like=ev.nxdomain_like,
                md_done=md_done,
            )
        with contextlib.suppress(Exception):
            await state.upsert_company(
                company.company_id,
                last_error=(str(e) or "")[:4000],
                name=company.name,
                root_url=company.domain_url,
            )
        return False

    finally:
        if watchdog_task is not None:
            watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watchdog_task

        with contextlib.suppress(Exception):
            snap2 = await state.get_company_snapshot(company.company_id)
            _write_crawl_meta(company, snap2)

        if run_id is not None and completed_ok:
            with contextlib.suppress(Exception):
                await state.mark_company_completed(run_id, company.company_id)

        if completed_ok:
            with contextlib.suppress(Exception):
                retry_store.mark_success(
                    company.company_id, stage="completed", note="ok"
                )

        with contextlib.suppress(Exception):
            logging_ext.reset_company_context(token)
            logging_ext.close_company(company.company_id)

        with contextlib.suppress(Exception):
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
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    p.add_argument("--dataset-file", type=str, default=None)

    p.add_argument("--company-concurrency", type=int, default=8)
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--page-timeout-ms", type=int, default=30000)

    p.add_argument(
        "--retry-mode", choices=["all", "skip-retry", "only-retry"], default="all"
    )
    p.add_argument("--enable-session-log", action="store_true")
    p.add_argument("--enable-resource-monitor", action="store_true")
    p.add_argument("--finalize-in-progress-md", action="store_true")

    # page pipeline knobs
    p.add_argument("--page-result-concurrency", type=int, default=6)
    p.add_argument("--page-queue-maxsize", type=int, default=32)
    p.add_argument("--url-index-flush-every", type=int, default=18)
    p.add_argument("--url-index-flush-interval-sec", type=float, default=0.5)
    p.add_argument("--url-index-queue-maxsize", type=int, default=1024)

    # crawler pool
    p.add_argument("--crawler-pool-size", type=int, default=6)
    p.add_argument("--crawler-recycle-after", type=int, default=12)

    # scheduler
    p.add_argument("--max-start-per-tick", type=int, default=3)
    p.add_argument("--crawler-capacity-multiplier", type=int, default=3)
    p.add_argument("--idle-recycle-interval-sec", type=float, default=25.0)
    p.add_argument("--idle-recycle-raw-frac", type=float, default=0.88)
    p.add_argument("--idle-recycle-eff-frac", type=float, default=0.83)

    # timeouts / hang prevention
    p.add_argument("--crawler-lease-timeout-sec", type=float, default=240.0)
    p.add_argument("--arun-init-timeout-sec", type=float, default=180.0)
    p.add_argument("--stream-no-yield-timeout-sec", type=float, default=600.0)
    p.add_argument("--submit-timeout-sec", type=float, default=60.0)
    p.add_argument("--resume-md-mode", choices=["direct", "deep"], default="direct")
    p.add_argument("--direct-fetch-url-timeout-sec", type=float, default=180.0)
    p.add_argument("--processor-finish-timeout-sec", type=float, default=360.0)
    p.add_argument("--generator-close-timeout-sec", type=float, default=60.0)
    p.add_argument("--company-crawl-timeout-sec", type=float, default=3600.0)

    # progress/heartbeat (new)
    p.add_argument("--company-progress-heartbeat-sec", type=float, default=30.0)
    p.add_argument("--company-progress-throttle-sec", type=float, default=12.0)

    # OOM guard
    p.add_argument("--oom-soft-frac", type=float, default=0.90)
    p.add_argument("--oom-hard-frac", type=float, default=0.95)
    p.add_argument("--oom-check-interval-sec", type=float, default=2.0)
    p.add_argument("--oom-soft-pause-sec", type=float, default=20.0)

    return p.parse_args(list(argv) if argv is not None else None)


async def main_async(args: argparse.Namespace) -> None:
    global _forced_exit_code, _retry_store_instance

    out_dir = _set_output_root(args.out_dir)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    logger.setLevel(log_level)

    if args.finalize_in_progress_md and args.llm_mode != "none":
        logger.warning("--finalize-in-progress-md forces --llm-mode none")
        args.llm_mode = "none"

    logging_ext = LoggingExtension(
        global_level=log_level,
        per_company_level=log_level,
        max_open_company_logs=128,
        enable_session_log=bool(args.enable_session_log),
        session_log_path=_global_path("session.log")
        if args.enable_session_log
        else None,
    )

    resource_monitor: Optional[ResourceMonitor] = None
    if args.enable_resource_monitor:
        resource_monitor = ResourceMonitor(
            output_path=_global_path("resource_usage.json"),
            config=ResourceMonitorConfig(),
        )
        resource_monitor.start()

    default_language_factory.set_language(args.lang)

    industry_llm_cache: Optional[IndustryStrategyCache] = None
    if args.llm_mode != "none":
        provider_strategy = provider_strategy_from_llm_model_selector(args.llm_model)
        llm_factory = LLMExtractionFactory(
            provider_strategy=provider_strategy,
            default_full_instruction=DEFAULT_FULL_INSTRUCTION,
            default_presence_instruction=DEFAULT_PRESENCE_INSTRUCTION,
        )
        industry_llm_cache = IndustryStrategyCache(
            factory=llm_factory, schema=None, extraction_type="schema"
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
    )

    page_policy = PageInteractionPolicy(wait_timeout_ms=int(args.page_timeout_ms))
    page_interaction_factory = default_page_interaction_factory

    state = get_crawl_state(db_path=_global_path("crawl_state.sqlite3"))

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

    if args.finalize_in_progress_md:
        inprog = set(_read_in_progress_companies(out_dir))
        if not inprog:
            logger.info("--finalize-in-progress-md: no in_progress_companies; exiting.")
            return
        companies = [c for c in companies if c.company_id in inprog]
        if not companies:
            logger.info(
                "--finalize-in-progress-md: none of in_progress IDs exist in input; exiting."
            )
            return

    companies_by_id = {c.company_id: c for c in companies}
    company_ids_all = [c.company_id for c in companies]

    run_id = await state.start_run("deep_crawl", version=None, args_hash=None)

    for c in companies:
        await state.upsert_company(c.company_id, name=c.name, root_url=c.domain_url)

    with contextlib.suppress(Exception):
        await state.recompute_global_state()

    runnable_ids = await state.filter_runnable_company_ids(
        company_ids_all,
        llm_mode=str(args.llm_mode),
        refresh_pending=False,
        chunk_size=500,
        concurrency=32,
    )
    await state.update_run_totals(run_id, total_companies=len(runnable_ids))

    dataset_externals = build_dataset_externals(args=args, companies=companies)

    bm25 = build_dual_bm25_components()
    url_scorer: Optional[DualBM25Scorer] = bm25["url_scorer"]
    bm25_filter: Optional[DualBM25Filter] = bm25["url_filter"]

    dfs_factory: Optional[DeepCrawlStrategyFactory] = None
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

    inflight_by_cid: Dict[str, asyncio.Task] = {}
    cid_by_task: Dict[asyncio.Task, str] = {}
    cancel_reason_by_cid: Dict[str, str] = {}

    def get_active_company_ids() -> List[str]:
        return [cid for cid, t in inflight_by_cid.items() if not t.done()]

    def request_cancel_companies(ids: Sequence[str]) -> None:
        for cid in ids:
            cancel_reason_by_cid[cid] = "scheduler_cancel"
            t = inflight_by_cid.get(cid)
            if t and not t.done():
                t.cancel()

    async def request_recycle_idle(count: int, reason: str) -> int:
        return 0

    scheduler_cfg = AdaptiveSchedulingConfig(
        log_path=_global_path("adaptive_scheduling_state.jsonl"),
        heartbeat_path=_global_path("heartbeat.json"),
        initial_target=min(4, int(args.company_concurrency)),
        max_target=int(args.company_concurrency),
        retry_base_dir=out_dir / "_retry",
        max_start_per_tick=int(args.max_start_per_tick),
        idle_recycle_interval_sec=float(args.idle_recycle_interval_sec),
        idle_recycle_raw_frac=float(args.idle_recycle_raw_frac),
        idle_recycle_eff_frac=float(args.idle_recycle_eff_frac),
        crawler_capacity_multiplier=max(1, int(args.crawler_capacity_multiplier)),
    )

    scheduler = AdaptiveScheduler(
        cfg=scheduler_cfg,
        get_active_company_ids=get_active_company_ids,
        request_cancel_companies=request_cancel_companies,
        request_recycle_idle=request_recycle_idle,
    )
    await scheduler.start()

    retry_store = scheduler.retry_store
    _retry_store_instance = retry_store

    async def is_company_runnable(cid: str, *, recompute: bool = False) -> bool:
        snap = await state.get_company_snapshot(cid, recompute=recompute)
        st = getattr(snap, "status", None) or COMPANY_STATUS_PENDING
        return not _should_skip_company(st, args.llm_mode)

    with contextlib.suppress(Exception):
        await scheduler.cleanup_completed_retry_ids(
            is_company_runnable=lambda cid: is_company_runnable(cid, recompute=False),
            treat_non_runnable_as_done=True,
            stage="startup_cleanup",
        )

    await scheduler.set_worklist(
        runnable_ids,
        retry_mode=str(args.retry_mode),
        is_company_runnable=lambda cid: is_company_runnable(cid, recompute=False),
    )

    stop_event = asyncio.Event()

    def _on_stop(sig: str) -> None:
        logger.warning("[Signal] %s received; stopping.", sig)
        stop_event.set()

    loop = asyncio.get_running_loop()
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGTERM, lambda: _on_stop("SIGTERM"))
        loop.add_signal_handler(signal.SIGINT, lambda: _on_stop("SIGINT"))

    def _on_oom_hard(used_frac: float) -> None:
        global _forced_exit_code
        _forced_exit_code = RETRY_EXIT_CODE
        stop_event.set()

    oom = OOMGuard(
        cfg=OOMGuardConfig(
            soft_frac=float(args.oom_soft_frac),
            hard_frac=float(args.oom_hard_frac),
            check_interval_sec=float(args.oom_check_interval_sec),
            soft_pause_sec=float(args.oom_soft_pause_sec),
        ),
        on_hard=_on_oom_hard,
    )
    oom.start(stop_event)

    attempt_counter = 0
    total_unique = int(len(runnable_ids))
    done_counter: Dict[str, int] = {"done": 0}

    async def _get_md_done_for(cid: str, *, recompute: bool) -> int:
        try:
            snapx = await state.get_company_snapshot(cid, recompute=recompute)
            return int(getattr(snapx, "urls_markdown_done", 0) or 0)
        except Exception:
            return 0

    async def _finalize_reconcile() -> None:
        payload: Dict[str, Any] = {}
        with contextlib.suppress(Exception):
            payload = await state.recompute_global_state()

        active_set = set(get_active_company_ids())
        scheduler_pending_total = _scheduler_pending_total(scheduler)

        for cid in runnable_ids:
            if cid in active_set:
                continue

            c = companies_by_id.get(cid)
            try:
                snap = await state.get_company_snapshot(cid, recompute=True)
                st = getattr(snap, "status", None) or COMPANY_STATUS_PENDING

                if st in (
                    COMPANY_STATUS_MD_DONE,
                    COMPANY_STATUS_LLM_DONE,
                    COMPANY_STATUS_TERMINAL_DONE,
                ):
                    continue

                # Safety: if there is still markdown work, do NOT terminalize just because queues look empty.
                with contextlib.suppress(Exception):
                    pending_md = await state.get_pending_urls_for_markdown(cid)
                    if pending_md:
                        continue

                is_quarantined = False
                q_reason: Optional[str] = None
                for fn_name in ("is_quarantined", "company_is_quarantined"):
                    fn = getattr(retry_store, fn_name, None)
                    if callable(fn):
                        try:
                            is_quarantined = bool(fn(cid))
                            break
                        except Exception:
                            pass

                if is_quarantined:
                    for fn_name in ("get_quarantine_info", "quarantine_info"):
                        fn = getattr(retry_store, fn_name, None)
                        if callable(fn):
                            try:
                                info = fn(cid)
                                if info is not None:
                                    q_reason = str(info)
                                    break
                            except Exception:
                                pass

                    with contextlib.suppress(Exception):
                        await state.mark_company_terminal(
                            cid,
                            reason="quarantined",
                            details={"retry": q_reason or "quarantined"},
                            last_error=q_reason or "quarantined",
                            name=(c.name if c else None),
                            root_url=(c.domain_url if c else None),
                        )
                    with contextlib.suppress(Exception):
                        retry_store.mark_success(
                            cid, stage="terminal_done", note="quarantined_finalize"
                        )
                    continue

                retry_pending_total = _retry_pending_total(retry_store)
                company_retry_pending = _retry_has_pending_for_company_best_effort(
                    retry_store, cid
                )

                no_more_scheduler = scheduler_pending_total == 0
                no_more_retry_global = retry_pending_total == 0
                no_more_retry_for_company = company_retry_pending is False or (
                    company_retry_pending is None and no_more_retry_global
                )

                if (
                    no_more_scheduler
                    and no_more_retry_for_company
                    and no_more_retry_global
                ):
                    last_err = getattr(snap, "last_error", None)
                    with contextlib.suppress(Exception):
                        await state.mark_company_terminal(
                            cid,
                            reason="finalize_no_more_work",
                            details={
                                "status": st,
                                "note": "in-progress but scheduler/retry empty at finalize",
                            },
                            last_error=last_err or "finalize_no_more_work",
                            name=(c.name if c else None),
                            root_url=(c.domain_url if c else None),
                        )
                    with contextlib.suppress(Exception):
                        retry_store.mark_success(
                            cid, stage="terminal_done", note="finalize_no_more_work"
                        )
                    continue

            except Exception:
                continue

        with contextlib.suppress(Exception):
            _ = await state.recompute_global_state()
        _ = payload

    try:
        while not stop_event.is_set():
            active = [cid for cid, t in inflight_by_cid.items() if not t.done()]
            active_n = len(active)

            pending_total = _scheduler_pending_total(scheduler)
            retry_pending = _retry_pending_total(retry_store)
            db_in_prog = await _db_in_progress(state)

            if (
                active_n == 0
                and pending_total == 0
                and retry_pending == 0
                and not db_in_prog
            ):
                payload: Dict[str, Any] = {}
                with contextlib.suppress(Exception):
                    payload = await state.recompute_global_state()
                if payload and (payload.get("in_progress_companies") or []):
                    await asyncio.sleep(0.5)
                    continue
                break

            free_crawlers = (
                0
                if time.time() < oom.pause_until_ts
                else crawler_pool.free_slots_approx()
            )
            start_ids = await scheduler.plan_start_batch(free_crawlers=free_crawlers)

            for cid in start_ids:
                c = companies_by_id.get(cid)
                if not c:
                    continue

                attempt_counter += 1
                _scheduler_progress(scheduler, cid, kind="start")

                t = asyncio.create_task(
                    run_company_pipeline(
                        c,
                        attempt_no=attempt_counter,
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
                        cancel_reason_by_cid=cancel_reason_by_cid,
                    ),
                    name=f"company-{cid}",
                )
                inflight_by_cid[cid] = t
                cid_by_task[t] = cid

            if not inflight_by_cid:
                with contextlib.suppress(Exception):
                    payload = await state.recompute_global_state()
                    if payload and (payload.get("in_progress_companies") or []):
                        await asyncio.sleep(1.0)
                        continue

                pending_total = _scheduler_pending_total(scheduler)
                retry_pending = _retry_pending_total(retry_store)
                db_in_prog = await _db_in_progress(state)

                if pending_total > 0 or retry_pending > 0 or db_in_prog:
                    sleep_hint_fn = getattr(scheduler, "sleep_hint_sec", None)
                    sleep_hint = 1.0
                    if callable(sleep_hint_fn):
                        with contextlib.suppress(Exception):
                            sleep_hint = float(sleep_hint_fn() or 1.0)
                    await asyncio.sleep(max(0.25, sleep_hint))
                    continue

                break

            done, _ = await asyncio.wait(
                list(inflight_by_cid.values()),
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in done:
                cid = cid_by_task.pop(t, None)
                if cid:
                    inflight_by_cid.pop(cid, None)

                if t.cancelled():
                    if (
                        cid
                        and cancel_reason_by_cid.pop(cid, None) == "scheduler_cancel"
                    ):
                        with contextlib.suppress(Exception):
                            c = companies_by_id.get(cid)
                            await state.recompute_company_from_index(
                                cid,
                                name=(c.name if c else None),
                                root_url=(c.domain_url if c else None),
                            )
                        md_done = await _get_md_done_for(cid, recompute=False)
                        with contextlib.suppress(Exception):
                            retry_store.mark_transient_cancel(
                                cid,
                                reason="cancelled by scheduler",
                                stage="scheduler_cancel",
                                md_done=md_done,
                            )
                        _scheduler_progress(scheduler, cid, kind="cancel")
                    continue

                ok = False
                try:
                    ok = bool(t.result())
                except Exception as e:
                    logger.exception("Company task failed (company_id=%s)", cid)
                    if cid:
                        ev = classify_failure(e, stage="task_exception")
                        md_done = await _get_md_done_for(cid, recompute=True)
                        with contextlib.suppress(Exception):
                            retry_store.mark_failure(
                                cid,
                                cls=ev.cls,
                                error=ev.error,
                                stage=ev.stage,
                                status_code=ev.status_code,
                                nxdomain_like=ev.nxdomain_like,
                                md_done=md_done,
                            )
                        with contextlib.suppress(Exception):
                            await state.upsert_company(
                                cid, last_error=(str(e) or "")[:4000]
                            )
                        _scheduler_progress(scheduler, cid, kind="fail")

                if not cid:
                    continue

                # After each attempt, re-check runnable status and requeue if needed.
                still_runnable = False
                with contextlib.suppress(Exception):
                    still_runnable = await is_company_runnable(cid, recompute=True)

                if still_runnable and not stop_event.is_set():
                    delay_sec = _best_effort_next_eligible_delay_sec(retry_store, cid)
                    await _best_effort_requeue_company(
                        scheduler,
                        cid,
                        delay_sec=delay_sec,
                        reason="post_attempt_runnable",
                    )
                    _scheduler_progress(scheduler, cid, kind="requeue")
                    continue

                # Only count as "done" if pipeline reported ok and DB is now non-runnable/terminal for this run mode.
                if ok and not still_runnable:
                    done_counter["done"] = int(done_counter.get("done", 0)) + 1
                    with contextlib.suppress(Exception):
                        scheduler.register_company_completed()
                    _scheduler_progress(scheduler, cid, kind="done")
                else:
                    # If not ok, retry store already has the failure; scheduler may also have its own retry work.
                    _scheduler_progress(scheduler, cid, kind="fail")

        with contextlib.suppress(Exception):
            await _finalize_reconcile()

        payload: Dict[str, Any] = {}
        with contextlib.suppress(Exception):
            payload = await state.recompute_global_state()

        retry_code = compute_retry_exit_code_from_store(retry_store, RETRY_EXIT_CODE)
        pending_total = _scheduler_pending_total(scheduler)
        retry_pending = _retry_pending_total(retry_store)
        db_in_prog = await _db_in_progress(state)

        in_progress_payload = bool(
            (payload.get("in_progress_companies") or []) if payload else False
        )

        if pending_total > 0 or retry_pending > 0 or db_in_prog or in_progress_payload:
            _forced_exit_code = RETRY_EXIT_CODE
        else:
            _forced_exit_code = retry_code

    finally:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await oom.stop()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await scheduler.stop()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await guard.stop()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await crawler_pool.stop()
        with contextlib.suppress(Exception):
            logging_ext.close()
        if resource_monitor:
            with contextlib.suppress(Exception):
                resource_monitor.stop()
        with contextlib.suppress(Exception):
            gc.collect()


def main(argv: Optional[Iterable[str]] = None) -> None:
    global _forced_exit_code, _retry_store_instance

    args = parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt; shutting down.")
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
