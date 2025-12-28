from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
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
from configs.language import default_language_factory
from configs.llm import (
    DEFAULT_FULL_INSTRUCTION,
    DEFAULT_PRESENCE_INSTRUCTION,
    IndustryStrategyCache,
    LLMExtractionFactory,
    provider_strategy_from_llm_model_selector,
)
from configs.llm_industry import get_industry_profile
from configs.md import default_md_factory

# Extensions
from extensions.load_source import CompanyInput, load_companies_from_source
from extensions.logging import LoggingExtension
from extensions import md_gating
from extensions import output_paths
from extensions.output_paths import ensure_company_dirs
from extensions.resource_monitor import ResourceMonitor, ResourceMonitorConfig
from extensions.adaptive_scheduling import (
    AdaptiveScheduler,
    AdaptiveSchedulingConfig,
    compute_retry_exit_code_from_store,
)
from extensions.connectivity_guard import ConnectivityGuard
from extensions.crawl_state import (
    get_crawl_state,
    normalize_company_industry_fields,
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
from extensions.crawl_runner import (
    CrawlRunnerConfig,
    run_company_crawl,
)
from extensions.terminalization import decide_from_page_summary
from extensions.oom_guard import OOMGuard, OOMGuardConfig
from extensions import retry as retry_mod

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

_TRACEBACK_ON_ATTEMPT_FAIL = os.getenv("RUN_TRACEBACK", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
} or os.getenv("RETRY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _short_exc(e: BaseException, limit: int = 900) -> str:
    msg = f"{type(e).__name__}: {e}"
    msg = " ".join((msg or "").split())
    if len(msg) > limit:
        return msg[: limit - 1] + "…"
    return msg


def _log_attempt_failure(
    clog: logging.Logger,
    *,
    prefix: str,
    attempt_index: int,
    stage: str,
    event: Optional[retry_mod.RetryEvent],
    exc: BaseException,
) -> None:
    cls = getattr(event, "cls", None)
    sk = getattr(event, "stall_kind", None)
    sc = getattr(event, "status_code", None)
    nx = getattr(event, "nxdomain_like", None)

    if _TRACEBACK_ON_ATTEMPT_FAIL:
        clog.exception(
            "%s attempt=%d stage=%s cls=%s stall=%s status=%s nx=%s err=%s",
            prefix,
            attempt_index,
            stage,
            cls,
            sk,
            sc,
            nx,
            _short_exc(exc),
        )
        return

    clog.error(
        "%s attempt=%d stage=%s cls=%s stall=%s status=%s nx=%s err=%s",
        prefix,
        attempt_index,
        stage,
        cls,
        sk,
        sc,
        nx,
        _short_exc(exc),
        exc_info=False,
    )


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
    industry_label: Optional[str] = None
    industry_codes: Optional[str] = None
    industry_profile_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


def _company_from_input(ci: CompanyInput) -> Company:
    md = dict(ci.metadata or {})
    industry_code, industry_label, industry_codes = normalize_company_industry_fields(
        md.get("industry")
    )
    prof = get_industry_profile(industry_code)

    return Company(
        company_id=str(ci.bvdid),
        domain_url=str(ci.url),
        name=(str(getattr(ci, "name", "")).strip() or None),
        industry_code=industry_code,
        industry_label=industry_label,
        industry_codes=industry_codes,
        industry_profile_id=getattr(prof, "profile_id", None),
        metadata=md,
    )


def _companies_from_source(path: Path) -> List[Company]:
    return [_company_from_input(ci) for ci in load_companies_from_source(path)]


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
    if llm_mode == "none":
        return status in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            COMPANY_STATUS_TERMINAL_DONE,
        )
    return status in (COMPANY_STATUS_LLM_DONE, COMPANY_STATUS_TERMINAL_DONE)


def _crawl_meta_path(company_id: str) -> Path:
    dirs = ensure_company_dirs(company_id)
    meta_dir = dirs["metadata"] if "metadata" in dirs else dirs["checkpoints"]
    Path(meta_dir).mkdir(parents=True, exist_ok=True)
    return Path(meta_dir) / "crawl_meta.json"


def _write_crawl_meta(company: Company, snapshot: Any) -> None:
    path = _crawl_meta_path(company.company_id)
    payload = {
        "company_id": company.company_id,
        "name": company.name,
        "industry_code": company.industry_code,
        "industry_label": company.industry_label,
        "industry_codes": company.industry_codes,
        "root_url": company.domain_url,
        "status": snapshot.status,
        "urls_total": snapshot.urls_total,
        "urls_markdown_done": snapshot.urls_markdown_done,
        "urls_llm_done": snapshot.urls_llm_done,
        "last_error": snapshot.last_error,
        "last_crawled_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        __import__("json").dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


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
    retry_store: retry_mod.RetryStateStore,
    scheduler: AdaptiveScheduler,
    stop_event: asyncio.Event,
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
        scheduler.touch_company(company.company_id, kind=kind)

    async def _get_md_done(*, recompute: bool) -> int:
        snapx = await state.get_company_snapshot(
            company.company_id, recompute=recompute
        )
        return int(snapx.urls_markdown_done or 0)

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

    watchdog_task: Optional[asyncio.Task] = None

    try:
        # Persist industry + basics at start (DB + crawl_meta.json via CrawlState)
        if (
            company.industry_code is None
            and company.industry_label is None
            and company.industry_codes is None
        ):
            industry_code, industry_label, industry_codes = (
                normalize_company_industry_fields(
                    company.metadata.get("industry")
                    if isinstance(company.metadata, dict)
                    else None
                )
            )
            company.industry_code = industry_code
            company.industry_label = industry_label
            company.industry_codes = industry_codes

        if not company.industry_profile_id:
            prof = get_industry_profile(company.industry_code)
            company.industry_profile_id = getattr(prof, "profile_id", None)

        await state.upsert_company(
            company.company_id,
            name=company.name,
            root_url=company.domain_url,
            industry_code=company.industry_code,
            industry_label=company.industry_label,
            industry_codes=company.industry_codes,
            write_meta=True,
        )

        snap = await state.get_company_snapshot(company.company_id, recompute=False)
        status = snap.status or COMPANY_STATUS_PENDING
        urls_md_done0 = int(snap.urls_markdown_done or 0)

        finalize_in_progress_md = bool(args.finalize_in_progress_md)
        will_llm = (
            (not finalize_in_progress_md)
            and (args.llm_mode in ("presence", "full"))
            and (industry_llm_cache is not None)
        )

        do_crawl = status in (COMPANY_STATUS_PENDING, COMPANY_STATUS_MD_NOT_DONE)

        if (not do_crawl) and (not will_llm):
            outcome.ok = True
            outcome.stage = "skip_already_done"
            outcome.should_mark_success = True
            retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            return True

        done_now = int(done_counter.get("done", 0))
        clog.info(
            "=== [done=%d/%d attempt=%d] company_id=%s url=%s industry=%s (code=%s codes=%s profile=%s) status=%s llm=%s ===",
            done_now,
            total_unique,
            attempt_no,
            company.company_id,
            company.domain_url,
            company.industry_label,
            company.industry_code,
            company.industry_codes,
            company.industry_profile_id,
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
                hard_max_pages=int(args.max_pages) if args.max_pages else None,
                page_timeout_ms=int(args.page_timeout_ms)
                if args.page_timeout_ms
                else None,
                direct_fetch_urls=bool(direct_fetch_urls),
            )

            heartbeat_sec = float(args.company_progress_heartbeat_sec)

            async def _watchdog() -> None:
                while True:
                    await asyncio.sleep(heartbeat_sec)
                    signal_progress("heartbeat")

            watchdog_task = asyncio.create_task(
                _watchdog(), name=f"watchdog:{company.company_id}"
            )

            last_exc: Optional[BaseException] = None
            last_event: Optional[retry_mod.RetryEvent] = None
            last_page_dec: Optional[Any] = None

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

                        term_dec = retry_mod.decide_terminalization(
                            page_summary_decision=last_page_dec,
                            exception=None,
                            stage="crawl",
                            urls_md_done0=urls_md_done0,
                            attempt_index=attempt_index,
                        )
                        if term_dec.should_terminalize:
                            outcome.ok = True
                            outcome.stage = "terminalize"
                            outcome.terminalized = True
                            outcome.terminal_reason = term_dec.reason
                            outcome.terminal_last_error = term_dec.last_error
                            outcome.should_mark_success = False
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
                            urls_md_done0=urls_md_done0,
                            attempt_index=attempt_index,
                        )
                        if term_dec.should_terminalize:
                            outcome.ok = True
                            outcome.stage = "terminalize"
                            outcome.terminalized = True
                            outcome.terminal_reason = term_dec.reason
                            outcome.terminal_last_error = term_dec.last_error
                            outcome.should_mark_success = False
                            break

                    _log_attempt_failure(
                        clog,
                        prefix="Crawl attempt failed",
                        attempt_index=attempt_index,
                        stage=stage,
                        event=last_event,
                        exc=e,
                    )

                    if attempt_index == 0:
                        await asyncio.sleep(0.5)

            if outcome.terminalized:
                await state.mark_company_terminal(
                    company.company_id,
                    reason="terminalize",
                    details={
                        "reason": outcome.terminal_reason,
                        "industry_code": company.industry_code,
                        "industry_codes": company.industry_codes,
                        "industry_profile_id": company.industry_profile_id,
                    },
                    last_error=outcome.terminal_last_error or outcome.terminal_reason,
                    name=company.name,
                    root_url=company.domain_url,
                )
                outcome.md_done = await _get_md_done(recompute=True)
                retry_mod.record_attempt(
                    retry_store, company.company_id, outcome, flush=True
                )
                return True

            if last_exc is not None and last_event is not None:
                outcome.ok = False
                outcome.stage = last_event.stage
                outcome.event = last_event
                outcome.md_done = await _get_md_done(recompute=True)
                retry_mod.record_attempt(
                    retry_store, company.company_id, outcome, flush=True
                )
                await _persist_last_error(str(last_exc))
                return False

            await state.recompute_company_from_index(
                company.company_id, name=company.name, root_url=company.domain_url
            )

        # ---------------- LLM stage ----------------
        if will_llm:
            llm_exc: Optional[BaseException] = None
            llm_event: Optional[retry_mod.RetryEvent] = None

            llm_stage = f"llm_{args.llm_mode}"
            for attempt_index in range(2):
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
                    llm_event = retry_mod.classify_failure(e, stage=llm_stage)

                    _log_attempt_failure(
                        clog,
                        prefix="LLM attempt failed",
                        attempt_index=attempt_index,
                        stage=llm_stage,
                        event=llm_event,
                        exc=e,
                    )

                    if attempt_index == 0:
                        await asyncio.sleep(0.75)

            if llm_exc is not None and llm_event is not None:
                outcome.ok = False
                outcome.stage = llm_event.stage
                outcome.event = llm_event
                outcome.md_done = await _get_md_done(recompute=True)
                retry_mod.record_attempt(
                    retry_store, company.company_id, outcome, flush=True
                )
                await _persist_last_error(str(llm_exc))
                return False

        if stop_event.is_set():
            clog.warning(
                "Stop requested; skipping post-check and completion marking company_id=%s",
                company.company_id,
            )
            return False

        snap_end = await state.get_company_snapshot(company.company_id, recompute=True)
        st_end = snap_end.status or COMPANY_STATUS_PENDING
        last_err = str(getattr(snap_end, "last_error", "") or "").strip()

        clog.info(
            "Post-check company=%s status=%s md_done=%s llm_done=%s last_error=%s",
            company.company_id,
            st_end,
            snap_end.urls_markdown_done,
            snap_end.urls_llm_done,
            (last_err[:240] + "…") if len(last_err) > 240 else last_err,
        )

        def _postcheck_failure(stage_name: str) -> bool:
            msg = f"incomplete_status={st_end}"
            if last_err:
                msg = f"{msg}; last_error={last_err}"

            ev = retry_mod.classify_failure(
                RuntimeError(last_err or msg), stage=stage_name
            )

            outcome.ok = False
            outcome.stage = stage_name
            outcome.event = retry_mod.RetryEvent(
                cls=ev.cls,
                stage=stage_name,
                error=msg,
                nxdomain_like=getattr(ev, "nxdomain_like", False),
                status_code=getattr(ev, "status_code", None),
                stall_kind=getattr(ev, "stall_kind", None),
            )
            outcome.md_done = int(snap_end.urls_markdown_done or 0)
            retry_mod.record_attempt(
                retry_store, company.company_id, outcome, flush=True
            )
            return False

        if args.llm_mode == "none" or bool(args.finalize_in_progress_md):
            ok_statuses = (COMPANY_STATUS_MD_DONE, COMPANY_STATUS_TERMINAL_DONE)
            if st_end not in ok_statuses:
                await _persist_last_error(last_err or f"incomplete_status={st_end}")
                return _postcheck_failure("postcheck_md")
        else:
            ok_statuses = (COMPANY_STATUS_LLM_DONE, COMPANY_STATUS_TERMINAL_DONE)
            if st_end not in ok_statuses:
                await _persist_last_error(last_err or f"incomplete_status={st_end}")
                return _postcheck_failure("postcheck_llm")

        outcome.ok = True
        outcome.stage = "completed"
        outcome.should_mark_success = True
        outcome.md_done = int(snap_end.urls_markdown_done or 0)
        retry_mod.record_attempt(retry_store, company.company_id, outcome, flush=True)
        return True

    except asyncio.CancelledError:
        clog.warning(
            "Company task cancelled (no marking) company_id=%s", company.company_id
        )
        raise

    except retry_mod.CriticalMemoryPressure as e:
        outcome.ok = False
        outcome.stage = "critical_memory_pressure"
        outcome.event = retry_mod.RetryEvent(
            cls="mem", stage="critical_memory_pressure", error=str(e)
        )
        outcome.md_done = await _get_md_done(recompute=True)
        retry_mod.record_attempt(retry_store, company.company_id, outcome, flush=True)
        await _persist_last_error(str(e))
        return False

    except Exception as e:
        ev = retry_mod.classify_failure(e, stage="pipeline_unhandled")
        outcome.ok = False
        outcome.stage = ev.stage
        outcome.event = ev
        outcome.md_done = await _get_md_done(recompute=True)

        if _TRACEBACK_ON_ATTEMPT_FAIL:
            clog.exception(
                "Pipeline unhandled exception stage=%s err=%s", ev.stage, _short_exc(e)
            )
        else:
            clog.error(
                "Pipeline unhandled exception stage=%s cls=%s stall=%s status=%s nx=%s err=%s",
                ev.stage,
                getattr(ev, "cls", None),
                getattr(ev, "stall_kind", None),
                getattr(ev, "status_code", None),
                getattr(ev, "nxdomain_like", None),
                _short_exc(e),
                exc_info=False,
            )

        retry_mod.record_attempt(retry_store, company.company_id, outcome, flush=True)
        await _persist_last_error(str(e))
        return False

    finally:
        if watchdog_task is not None:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass

        if stop_event.is_set():
            logging_ext.reset_company_context(token)
            logging_ext.close_company(company.company_id)
            gc.collect()
            return

        snap2 = await state.get_company_snapshot(company.company_id, recompute=False)
        _write_crawl_meta(company, snap2)

        await state.recompute_company_from_index(
            company.company_id,
            name=company.name,
            root_url=company.domain_url,
        )

        if run_id is not None:
            if outcome.ok or outcome.terminalized:
                await state.mark_company_completed(run_id, company.company_id)

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

    p.add_argument(
        "--sync-industry-from-csv",
        dest="sync_industry_from_csv",
        action="store_true",
        help="Update industry fields in DB for companies already present (no inserts). Also writes crawl_meta.json.",
    )

    # crawl_runner knobs
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

    # progress/heartbeat
    p.add_argument("--company-progress-heartbeat-sec", type=float, default=30.0)
    p.add_argument("--company-progress-throttle-sec", type=float, default=12.0)

    # OOM guard
    p.add_argument("--oom-soft-frac", type=float, default=0.90)
    p.add_argument("--oom-hard-frac", type=float, default=0.95)
    p.add_argument("--oom-check-interval-sec", type=float, default=2.0)
    p.add_argument("--oom-soft-pause-sec", type=float, default=20.0)

    # global state writer throttle
    p.add_argument("--global-state-write-interval-sec", type=float, default=1.5)

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

    if bool(getattr(args, "sync_industry_from_csv", False)) and args.company_file:
        await state.update_industry_from_csv(
            csv_path=args.company_file,
            bvdid_column="bvdid",
            industry_column="industry",
            write_meta=True,
        )

    if args.company_file:
        companies = _companies_from_source(Path(args.company_file))
    else:
        url = args.url
        assert url is not None
        cid = args.company_id
        if not cid:
            parsed = urlparse(url)
            cid = (parsed.netloc or parsed.path or "company").replace(":", "_")

        industry_code, industry_label, industry_codes = (
            normalize_company_industry_fields(None)
        )
        prof = get_industry_profile(industry_code)
        companies = [
            Company(
                company_id=cid,
                domain_url=url,
                industry_code=industry_code,
                industry_label=industry_label,
                industry_codes=industry_codes,
                industry_profile_id=getattr(prof, "profile_id", None),
                metadata={},
            )
        ]

    companies_by_id = {c.company_id: c for c in companies}
    company_ids_all = [c.company_id for c in companies]

    run_id = await state.start_run("deep_crawl", version=None, args_hash=None)

    for c in companies:
        await state.upsert_company(
            c.company_id,
            name=c.name,
            root_url=c.domain_url,
            industry_code=c.industry_code,
            industry_label=c.industry_label,
            industry_codes=c.industry_codes,
            write_meta=True,
        )

    # DB truth first: refresh in-progress snapshots based on url_index.json
    await state.recompute_all_in_progress(concurrency=32)

    if args.finalize_in_progress_md:
        inprog = set(await state.get_in_progress_company_ids(limit=1_000_000))
        companies = [c for c in companies if c.company_id in inprog]
        companies_by_id = {c.company_id: c for c in companies}
        company_ids_all = [c.company_id for c in companies]

    # Compute runnable_ids using new CrawlState snapshot semantics (no legacy filter API)
    runnable_ids: List[str] = []
    for cid in company_ids_all:
        snap = await state.get_company_snapshot(cid, recompute=False)
        st = snap.status or COMPANY_STATUS_PENDING
        if not _should_skip_company(st, str(args.llm_mode)):
            runnable_ids.append(cid)

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

    def get_active_company_ids() -> List[str]:
        return [cid for cid, t in inflight_by_cid.items() if not t.done()]

    def request_cancel_companies(ids: Sequence[str]) -> None:
        for cid in ids:
            t = inflight_by_cid.get(cid)
            if t and not t.done():
                t.cancel()

    async def request_recycle_idle(count: int, reason: str) -> int:
        logger.info("request_recycle_idle count=%d reason=%s (noop)", count, reason)
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

    retry_store: retry_mod.RetryStateStore = scheduler.retry_store
    _retry_store_instance = retry_store

    async def is_company_runnable(cid: str, *, recompute: bool = False) -> bool:
        if retry_store.is_quarantined(cid):
            return False
        snap = await state.get_company_snapshot(cid, recompute=recompute)
        st = snap.status or COMPANY_STATUS_PENDING
        return not _should_skip_company(st, args.llm_mode)

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
    stop_reason: Optional[str] = None
    stop_sig: Optional[str] = None

    def _on_stop(sig: str) -> None:
        nonlocal stop_reason, stop_sig
        logger.warning(
            "[Signal] %s received; cancelling in-flight work and shutting down.", sig
        )
        stop_reason = "user_interrupt"
        stop_sig = sig
        stop_event.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, lambda: _on_stop("SIGTERM"))
        loop.add_signal_handler(signal.SIGINT, lambda: _on_stop("SIGINT"))
    except NotImplementedError:
        logger.warning("Signal handlers not supported on this platform")

    def _on_oom_hard(used_frac: float) -> None:
        global _forced_exit_code
        nonlocal stop_reason, stop_sig
        logger.error(
            "OOM hard triggered used_frac=%.4f forcing exit_code=%d",
            used_frac,
            RETRY_EXIT_CODE,
        )
        _forced_exit_code = RETRY_EXIT_CODE
        stop_reason = "oom_hard"
        stop_sig = "OOM_HARD"
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

    global_writer_task: Optional[asyncio.Task] = None
    global_write_interval = max(
        0.2, float(getattr(args, "global_state_write_interval_sec", 1.5))
    )

    async def _global_writer() -> None:
        while not stop_event.is_set():
            await state.write_global_state_throttled(
                min_interval_sec=global_write_interval
            )
            await asyncio.sleep(0.25)

    global_writer_task = asyncio.create_task(
        _global_writer(), name="global_state_writer"
    )

    attempt_counter = 0
    total_unique = int(len(runnable_ids))
    done_counter: Dict[str, int] = {"done": 0}

    try:
        while True:
            if stop_event.is_set():
                for t in list(inflight_by_cid.values()):
                    if not t.done():
                        t.cancel()

                if inflight_by_cid:
                    await asyncio.gather(
                        *list(inflight_by_cid.values()), return_exceptions=True
                    )

                logger.warning(
                    "Shutdown requested reason=%s sig=%s active_before_cancel=%d",
                    stop_reason,
                    stop_sig,
                    len(inflight_by_cid),
                )

                await state.write_global_state_from_db_only()
                break

            active_n = sum(1 for _cid, t in inflight_by_cid.items() if not t.done())
            pending_total = scheduler.pending_total()
            retry_pending = retry_store.pending_total(exclude_quarantined=True)
            db_in_prog = await state.has_in_progress_companies()

            if (
                active_n == 0
                and pending_total == 0
                and retry_pending == 0
                and not db_in_prog
            ):
                payload = await state.write_global_state_from_db_only()
                if payload and (payload.get("in_progress_company_ids") or []):
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
                c = companies_by_id[cid]
                attempt_counter += 1
                scheduler.touch_company(cid, kind="start")

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
                        stop_event=stop_event,
                    ),
                    name=f"company-{cid}",
                )
                inflight_by_cid[cid] = t
                cid_by_task[t] = cid

            if not inflight_by_cid:
                await asyncio.sleep(max(0.25, float(retry_store.sleep_hint_sec())))
                continue

            done, _ = await asyncio.wait(
                list(inflight_by_cid.values()),
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in done:
                cid = cid_by_task.pop(t)
                inflight_by_cid.pop(cid, None)

                if t.cancelled():
                    logger.warning(
                        "task cancelled cid=%s (NO marking in pipeline)", cid
                    )
                    scheduler.touch_company(cid, kind="cancel")
                    continue

                ok = bool(t.result())
                still_runnable = await is_company_runnable(cid, recompute=True)

                if still_runnable and not stop_event.is_set():
                    rq = retry_mod.decide_requeue(
                        store=retry_store,
                        company_id=cid,
                        is_runnable=True,
                        stop_requested=False,
                    )
                    if rq.should_requeue:
                        await scheduler.requeue_company(
                            cid,
                            delay_sec=float(rq.delay_sec),
                            reason=str(rq.reason),
                        )
                        scheduler.touch_company(cid, kind="requeue")
                        continue

                if ok and not still_runnable:
                    done_counter["done"] = int(done_counter.get("done", 0)) + 1
                    scheduler.register_company_completed()
                    scheduler.touch_company(cid, kind="done")
                else:
                    scheduler.touch_company(cid, kind="fail")

                await state.write_global_state_throttled(
                    min_interval_sec=global_write_interval
                )

        pending_total = scheduler.pending_total()
        retry_pending = retry_store.pending_total(exclude_quarantined=True)
        db_in_prog = await state.has_in_progress_companies()
        payload = await state.write_global_state_from_db_only()
        in_progress_payload = bool(
            (payload.get("in_progress_company_ids") or []) if payload else False
        )

        _forced_exit_code = retry_mod.decide_exit_code(
            forced_exit_code=_forced_exit_code,
            retry_exit_code=RETRY_EXIT_CODE,
            scheduler_pending_total=pending_total,
            retry_pending_total=retry_pending,
            db_in_progress=db_in_prog,
            in_progress_payload=in_progress_payload,
        )

    finally:
        if global_writer_task is not None:
            global_writer_task.cancel()
            try:
                await global_writer_task
            except asyncio.CancelledError:
                pass

        await oom.stop()
        await scheduler.stop()
        await guard.stop()
        await crawler_pool.stop()
        logging_ext.close()

        if resource_monitor:
            resource_monitor.stop()

        state.close()
        gc.collect()


def main(argv: Optional[Iterable[str]] = None) -> None:
    global _forced_exit_code, _retry_store_instance

    args = parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt; exiting.")
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
