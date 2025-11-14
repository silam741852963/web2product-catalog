from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# --- Windows stdio bump to reduce EMFILE (best effort)
try:
    import msvcrt  # type: ignore
    try:
        msvcrt.setmaxstdio(2048)
    except Exception:
        pass
except Exception:
    pass

from crawl4ai import AsyncWebCrawler

from configs.language_settings import load_lang
from configs.browser_settings import browser_cfg

from components.source_loader import CompanyInput, load_companies_from_source, load_companies_from_dir
from components.url_seeder import seed_urls
from components.md_generator import evaluate_markdown
from components.llm_extractor import parse_presence_result, _short_preview

from extensions.artifact_scanner import ArtifactScanner
from extensions.checkpoint import CheckpointManager
from extensions.connectivity_guard import ConnectivityGuard
from extensions.global_state import GlobalState, ResumePlan
from extensions.logging import LoggingExtension
from extensions.output_paths import save_stage_output, ensure_company_dirs

from extensions.run_utils import metrics

from extensions.run_utils import (
    build_parser,
    parse_pipeline,
    make_dispatcher,
    mk_md_config,
    mk_llm_config,
    aggregate_seed_by_root,
    classify_failure,
    prefer_local_html,
    prefer_local_md,
    load_url_index,
    upsert_url_index,
    write_url_index_seed_only,
    read_last_crawl_date,
    write_last_crawl_date,
    filter_by_last_crawl_date,
    company_log_tail_has_error,
    recommend_company_timeouts,
    is_llm_extracted_empty,
    get_default_include_patterns,
    get_default_exclude_patterns,
    build_seeder_kwargs,
)

from extensions.slot_allocator import SlotAllocator, SlotConfig, SessionSnapshot, CompanySnapshot

# Page interaction: use both first + retry configs
from extensions.page_interaction import PageInteractionPolicy, first_pass_config, retry_pass_config


# --------------------------------------------------------------------------------------
# Compatibility shim: make sure Crawl4AI config objects expose the attributes
# arun_many() expects across versions (e.g., "stream", "js_code", etc.).
# --------------------------------------------------------------------------------------
def _ensure_c4a_config_defaults(cfg: Any, *, default_stream: bool = True) -> Any:
    """
    Idempotently attach required attributes if missing. Works with both real
    Crawl4AI CrawlerRunConfig and our _Dummy fallback from run_utils.
    """
    if cfg is None:
        class _Bare:  # ultra-safe fallback
            pass
        cfg = _Bare()

    # Whether arun_many will yield an async stream (typical) or return a list.
    if not hasattr(cfg, "stream"):
        setattr(cfg, "stream", default_stream)

    # JS/scroll knobs used by page interaction recipes
    for a in ("js_code", "wait_for", "virtual_scroll_config"):
        if not hasattr(cfg, a):
            setattr(cfg, a, None)

    # HTML delay knob (float)
    if not hasattr(cfg, "delay_before_return_html"):
        setattr(cfg, "delay_before_return_html", 0.0)

    # Common flags (don’t force values; just ensure attrs exist for older versions)
    for a in ("extract_markdown", "llm_extraction", "llm_presence_only"):
        if not hasattr(cfg, a):
            setattr(cfg, a, False)

    return cfg


def _prune_redundant_seeding_fields(cp) -> None:
    """
    Delete legacy top-level seeding keys that duplicate cp.data['seeding'].
    Keeps all details inside cp.data['seeding'] only.
    """
    if not hasattr(cp, "data") or not isinstance(cp.data, dict):
        return
    for k in ("seed_counts_by_root", "seed_roots", "seed_brand_roots", "seed_brand_count"):
        cp.data.pop(k, None)


def _save_result_for_pipeline(
    bvdid: str,
    url: str,
    pipeline: List[str],
    result,
    logger: logging.Logger,
    md_min_words: int,
    *,
    save_html_ref: bool = True,
    md_gate_opts: Optional[Dict[str, Any]] = None,
    presence_only: bool = False,
    html_saved_urls: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Save artifacts based on pipeline and evaluate_markdown() gate.
    Returns stats incl. whether 'retry' was suggested and reason.
    Also updates url_index.json statuses via upsert_url_index().
    """
    stats = {
        "saved_html": False, "saved_md": False, "saved_json": False,
        "md_suppressed": False, "md_retry_suggested": False, "md_reason": "n/a",
        "presence_written": False,
    }

    html = getattr(result, "html", None) or getattr(result, "cleaned_html", None)
    md_obj = getattr(result, "markdown", None)
    fit_md = getattr(md_obj, "fit_markdown", None) if md_obj is not None else None
    extracted = getattr(result, "extracted_content", None)

    html_path = md_path = json_path = None

    if save_html_ref and html:
        from extensions.run_utils import find_existing_artifact
        # de-dupe: skip saving if already recorded this run or exists on disk
        already_written = (html_saved_urls is not None and url in html_saved_urls)
        existing = find_existing_artifact(bvdid, url, "html")
        if not already_written and not existing:
            save_stage_output(bvdid, url, html, stage="html")
            stats["saved_html"] = True
            if html_saved_urls is not None:
                html_saved_urls.add(url)
            existing = find_existing_artifact(bvdid, url, "html")
        html_path = existing

        try:
            # record path; status precedence handled in upsert_url_index
            upsert_url_index(bvdid, url, html_path=html_path)
        except Exception as e:
            logger.debug("[%s] url_index upsert (html) failed for %s: %s", bvdid, url, e)

    if ("markdown" in pipeline or "llm" in pipeline):
        chosen_md = (fit_md or "").strip()
        if chosen_md:
            action, reason, md_stats = evaluate_markdown(
                chosen_md,
                allow_retry=True,
                min_meaningful_words=max(1, int(md_min_words)),
                cookie_max_fraction=(md_gate_opts or {}).get("cookie_max_fraction", 0.15),
                require_structure=bool((md_gate_opts or {}).get("require_structure", True)),
            )
            stats["md_reason"] = reason
            if action == "save":
                save_stage_output(bvdid, url, chosen_md, stage="markdown")
                stats["saved_md"] = True
                from extensions.run_utils import find_existing_artifact
                md_path = find_existing_artifact(bvdid, url, "markdown")
                metrics.incr("markdown.saved", bvdid=bvdid)
                metrics.observe("markdown.words", float(md_stats.get("total_words", 0.0)), bvdid=bvdid)
                try:
                    upsert_url_index(
                        bvdid,
                        url,
                        markdown_path=md_path,
                        markdown_words=int(md_stats.get("total_words", 0) or 0),
                        status="markdown_saved",
                        reason=reason,
                    )
                except Exception as e:
                    logger.debug("[%s] url_index upsert (markdown_saved) failed for %s: %s", bvdid, url, e)

            elif action == "retry":
                stats["md_retry_suggested"] = True
                metrics.incr("markdown.retry_suggested", bvdid=bvdid, reason=reason)
                try:
                    upsert_url_index(bvdid, url, status="markdown_retry", reason=reason)
                except Exception as e:
                    logger.debug("[%s] url_index upsert (markdown_retry) failed for %s: %s", bvdid, url, e)
            else:
                stats["md_suppressed"] = True
                metrics.incr("markdown.suppressed", bvdid=bvdid, reason=reason)
                try:
                    upsert_url_index(bvdid, url, status="markdown_suppressed", reason=reason)
                except Exception as e:
                    logger.debug("[%s] url_index upsert (markdown_suppressed) failed for %s: %s", bvdid, url, e)
        else:
            stats["md_suppressed"] = True
            stats["md_reason"] = "empty"
            try:
                upsert_url_index(bvdid, url, status="markdown_suppressed", reason="empty")
            except Exception as e:
                logger.debug("[%s] url_index upsert (markdown_suppressed-empty) failed for %s: %s", bvdid, url, e)

    if "llm" in pipeline and extracted is not None:
        if presence_only:
            has, _conf, _rat = parse_presence_result(extracted, default=False)
            has_int = 1 if has else 0
            try:
                upsert_url_index(bvdid, url, has_offering=has_int, presence_checked=True, status="presence_checked")
            except Exception as e:
                logger.debug("[%s] url_index upsert failed while writing presence for %s: %s", bvdid, url, e)
            stats["presence_written"] = True
            stats["saved_json"] = False
        else:
            save_stage_output(bvdid, url, extracted, stage="llm")
            stats["saved_json"] = True
            from extensions.run_utils import find_existing_artifact
            json_path = find_existing_artifact(bvdid, url, "llm")
            metrics.incr("llm.saved", bvdid=bvdid)
            try:
                empty = is_llm_extracted_empty(extracted)
                if empty:
                    upsert_url_index(bvdid, url, json_path=json_path, llm_extracted_empty=True, status="llm_extracted_empty")
                else:
                    upsert_url_index(bvdid, url, json_path=json_path, llm_extracted_empty=False, status="llm_extracted")
            except Exception as e:
                logger.debug("[%s] url_index upsert (llm) failed for %s: %s", bvdid, url, e)

    return stats


async def _crawl_company(
    company: CompanyInput,
    *,
    pipeline: List[str],
    seeding_source: str,
    max_concurrency_for_company: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    respect_crawl_date: bool,
    bypass_local: bool,
    force_seeder_cache: bool,
    bm25_query: Optional[str],
    bm25_score_threshold: Optional[float],
    max_urls: int,
    company_max_pages: int,
    hits_per_sec: Optional[int],
    cp_mgr: CheckpointManager,
    log_ext: LoggingExtension,
    discover_brands: bool,
    drop_universal_externals: bool,
    lang_primary: str,
    lang_accept_en_regions: set[str],
    lang_strict_cctld: bool,
    args: argparse.Namespace,
    state: GlobalState,
    net: ConnectivityGuard,
    slot_alloc: SlotAllocator,
) -> None:
    bvdid = company.bvdid
    logger = log_ext.get_company_logger(bvdid)
    ensure_company_dirs(bvdid)

    await state.upsert_company(bvdid, company.name, company.url, stage=",".join(pipeline), status="pending")
    await state.mark_in_progress(bvdid, stage=",".join(pipeline))

    await cp_mgr.append_company(bvdid, company_name=company.name)
    cp = await cp_mgr.get(bvdid, company_name=company.name)
    await cp.mark_start(stage="|".join(pipeline), total_urls=0)
    logger.info("[%s] Start: %s (%s) pipeline=%s", bvdid, company.name, company.url, pipeline)
    metrics.incr("company.start", bvdid=bvdid)

    processed_once: set[str] = set()

    async def mark_done_once(u: str) -> None:
        if u and u not in processed_once:
            processed_once.add(u)
            await cp.mark_url_done(u)

    plan: ResumePlan = await state.recommend_resume(
        bvdid,
        requested_pipeline=pipeline,
        force_seeder_cache=force_seeder_cache,
        bypass_local=bypass_local,
    )
    logger.info("[%s] Resume plan: %s", bvdid, plan.reason)

    completion_stage = pipeline[-1]
    if ((completion_stage == "llm" and plan.skip_llm) or
        (completion_stage == "markdown" and plan.skip_markdown)):
        row = await state.get_company(bvdid)
        if row:
            tot = int(row.get("urls_total") or 0)
            cp.data["urls_total"] = tot
            cp.data["urls_done"] = tot
            cp.data["urls_failed"] = int(row.get("urls_failed") or 0)
            await cp.save()
            try:
                await state.update_counts(bvdid, urls_total=tot, urls_done=tot, urls_failed=int(row.get("urls_failed") or 0))
            except Exception:
                pass
        await cp.mark_finished()
        write_last_crawl_date(bvdid)
        await state.mark_done(
            bvdid,
            stage=",".join(pipeline),
            presence_only=bool(getattr(args, "presence_only", False))
        )
        logger.info("[%s] %s already complete per global DB. Skipping.", bvdid, completion_stage.upper())
        metrics.incr("company.skipped", bvdid=bvdid, stage=completion_stage)
        return

    # 1) URL discovery (seeder) or resume from url_index
    seeded_items: List[Dict[str, Any]] = []
    seed_stats: Dict[str, Any] = {}
    used_index_for_urls = False
    url_index: Dict[str, Any] = {}

    pipeline_set = set(pipeline)

    # --- Shortcut: resume using url_index when plan says so
    if plan.skip_seeding_entirely:
        url_index = load_url_index(bvdid)
        used_index_for_urls = True

        # Determine desired statuses based on requested pipeline
        if ("llm" in pipeline_set) and plan.skip_markdown:
            desired_statuses = {"markdown_saved"}
        elif "markdown" in pipeline_set:
            desired_statuses = {"seeded", "html_saved"}
        else:
            desired_statuses = {"seeded"}

        urls_from_index = [
            u for u, ent in (url_index or {}).items()
            if str(ent.get("status", "")).lower() in desired_statuses
        ]
        seeded_items = [{"url": u, "status": "valid"} for u in urls_from_index]
        # minimal resume summary (avoid clashing with seeding_metrics)
        seed_stats = {
            "seed_roots": [company.url],
            "seed_brand_count": 0,
            "resume_mode": f"url_index[{','.join(sorted(desired_statuses))}]",
        }
        await state.set_resume_mode(bvdid, "url_index")
        logger.info("[%s] Using url_index.json with %d URL(s) filtered by status in %s.", bvdid, len(seeded_items), sorted(desired_statuses))

    # --- Discovery path (seeder only)
    if not used_index_for_urls:
        while True:
            await net.wait_until_healthy()
            opened_before = net.snapshot_open_events()
            try:
                with metrics.time("seeding.duration_s", bvdid=bvdid):
                    seeded_items, seed_stats = await seed_urls(
                        company.url,
                        **build_seeder_kwargs(args) | dict(
                            include=include_patterns,
                            exclude=exclude_patterns,
                            max_urls=max_urls,
                            company_max_pages=company_max_pages,
                            hits_per_sec=hits_per_sec,
                        )
                    )
                net.record_success()
            except Exception as e:
                logger.error("[%s] Seeding error (network?): %s", bvdid, e)
                net.record_transport_error()
                if not net.is_healthy():
                    logger.warning("[%s] ConnectivityGuard OPEN — waiting to retry seeding...", bvdid)
                    await state.mark_paused_net(bvdid)
                    await net.wait_until_healthy()
                    await state.mark_in_progress(bvdid, stage=",".join(pipeline))
                    logger.info("[%s] Connectivity restored — retrying seeding.", bvdid)
                    continue
                seeded_items, seed_stats = [], {"discovered_total": 0, "filtered_total": 0, "seed_roots": [company.url], "seed_brand_count": 0}

            disc = int(seed_stats.get("discovered_total", len(seeded_items)))
            opened_after = net.snapshot_open_events()
            metrics.set("seeding.discovered", float(disc), bvdid=bvdid)
            if disc == 0 and (not net.is_healthy() or opened_after > opened_before):
                logger.warning("[%s] 0 URLs discovered during unstable connectivity — will retry.", bvdid)
                await state.mark_paused_net(bvdid)
                await net.wait_until_healthy()
                await state.mark_in_progress(bvdid, stage=",".join(pipeline))
                logger.info("[%s] Connectivity restored — retrying seeding.", bvdid)
                continue
            break

    # keep all seed summaries under 'seeding'
    cp.data.setdefault("seeding", {}).update(seed_stats)
    cp.data["seeding"].update(aggregate_seed_by_root(seeded_items, base_root=company.url))
    _prune_redundant_seeding_fields(cp)
    await cp.save()

    logger.info(
        "[%s] Discovery: discovered=%s, filtered=%s, roots=%d, brands=%d",
        bvdid,
        seed_stats.get("discovered_total", 0),
        seed_stats.get("filtered_total", 0),
        len(seed_stats.get("seed_roots", [])),
        seed_stats.get("seed_brand_count", 0),
    )
    metrics.set("seeding.filtered", float(seed_stats.get("filtered_total", 0)), bvdid=bvdid)

    # If pipeline is just seed, finalize now
    if pipeline == ["seed"]:
        seeded_urls_for_index = [it.get("final_url") or it.get("url") for it in seeded_items if (it.get("final_url") or it.get("url"))]
        total_index = write_url_index_seed_only(bvdid, seeded_urls_for_index)
        await state.record_url_index(bvdid, count=total_index)
        cp.data.setdefault("seeding", {}).update({"urls_total": int(seed_stats.get("filtered_total", 0)), "seed_source": "seeder"})
        cp.data["seeding"].update(aggregate_seed_by_root(seeded_items, base_root=company.url))
        _prune_redundant_seeding_fields(cp)
        await cp.save()
        await cp.mark_finished()
        write_last_crawl_date(bvdid)
        await state.mark_done(
            bvdid,
            stage=",".join(pipeline),
            presence_only=bool(getattr(args, "presence_only", False))
        )
        logger.info("[%s] SEED stage complete. url_index.json now has %d URL(s).", bvdid, total_index)
        metrics.incr("company.done", bvdid=bvdid, stage="seed")
        return

    # If discovery used network, mirror to url_index
    if not used_index_for_urls:
        seeded_urls_for_index = [it.get("final_url") or it.get("url") for it in seeded_items if (it.get("final_url") or it.get("url"))]
        total_index = write_url_index_seed_only(bvdid, seeded_urls_for_index)
        try:
            await state.record_url_index(bvdid, count=total_index)
        except Exception:
            pass

    # Build URL list
    seeded = list(seeded_items)
    if respect_crawl_date and not used_index_for_urls:
        last_dt = read_last_crawl_date(bvdid)
        if last_dt:
            before = len(seeded)
            seeded = filter_by_last_crawl_date(seeded, last_dt)
            logger.info("[%s] Respect last crawl (%s): kept %d/%d", bvdid, last_dt.isoformat(), len(seeded), before)

    url_to_item: Dict[str, Dict[str, Any]] = {}
    seeded_urls: List[str] = []
    for u in seeded:
        url = u.get("final_url") or u.get("url")
        if url:
            url_to_item[url] = u
            seeded_urls.append(url)

    # Presence gating
    if getattr(args, "require_presence", False) and used_index_for_urls and url_index:
        filtered = []
        for url in seeded_urls:
            ent = url_index.get(url, {})
            has_off = int(ent.get("has_offering", 0) or 0)
            pres = bool(ent.get("presence_checked", False))
            if has_off == 1 and pres:
                filtered.append(url)
            else:
                logger.debug("[%s] Skipping %s due to require-presence (has_offering=%s presence_checked=%s)", bvdid, url, has_off, pres)
        seeded_urls = filtered
        logger.info("[%s] After require-presence gating: %d URLs remain", bvdid, len(seeded_urls))
    elif getattr(args, "require_presence", False) and not used_index_for_urls:
        # First-time runs with require_presence set intentionally skip processing
        logger.info("[%s] --require-presence used but no url_index.json available; skipping all processing for this company", bvdid)
        seeded_urls = []

    cp.data["urls_total"] = len(seeded_urls)
    await cp.save()
    await state.set_urls_total(bvdid, len(seeded_urls))
    metrics.set("company.urls_total", float(len(seeded_urls)), bvdid=bvdid)

    if not seeded_urls:
        await cp.mark_finished()
        write_last_crawl_date(bvdid)
        try:
            await state.update_counts(bvdid, urls_total=0, urls_done=0, urls_failed=0)
        except Exception:
            pass
        await state.mark_done(
            bvdid,
            stage=",".join(pipeline),
            presence_only=bool(getattr(args, "presence_only", False))
        )
        logger.info("[%s] No URLs after discovery/resume. DONE.", bvdid)
        metrics.incr("company.done", bvdid=bvdid, stage="none")
        return

    llm_in_pipeline = ("llm" in pipeline_set)

    logger.info("[%s] Resume check: plan.skip_markdown=%s used_index_for_urls=%s",
                bvdid, bool(plan.skip_markdown), bool(used_index_for_urls))

    completion_stage = pipeline[-1]
    if (completion_stage == "llm" and plan.skip_llm) or (completion_stage == "markdown" and plan.skip_markdown):
        for url in seeded_urls:
            await mark_done_once(url)
        await cp.mark_finished()
        write_last_crawl_date(bvdid)
        try:
            await state.update_counts(
                bvdid,
                urls_total=len(seeded_urls),
                urls_done=len(seeded_urls),
                urls_failed=cp.data.get("urls_failed", 0),
            )
        except Exception:
            pass
        await state.mark_done(
            bvdid,
            stage=",".join(pipeline),
            presence_only=bool(getattr(args, "presence_only", False))
        )
        logger.info("[%s] %s already complete per global DB. Skipping work.", bvdid, completion_stage.upper())
        metrics.incr("company.done", bvdid=bvdid, stage=completion_stage)
        return

    # ---------- Slot allocator helpers (dynamic tail boosting) ----------
    async def _session_snapshot() -> SessionSnapshot:
        try:
            summary = await cp_mgr.summary()
        except Exception:
            summary = {}
        total = max(1, len(summary))  # avoid zero division
        finished = sum(1 for s in summary.values() if bool(s.get("finished")))
        running = max(1, total - finished)
        return SessionSnapshot(total_companies=total, finished_companies=finished, running_companies=running)

    async def _company_snapshot() -> CompanySnapshot:
        try:
            cpi = await cp_mgr.get(bvdid, company_name=company.name)
            urls_total = int(cpi.data.get("urls_total") or 0)
            urls_done  = int(cpi.data.get("urls_done")  or 0)
        except Exception:
            urls_total = urls_done = 0
        return CompanySnapshot(
            urls_total=urls_total if urls_total > 0 else None,
            urls_done=urls_done if urls_done > 0 else None,
            timeout_rate=None,
        )

    def _apply_max_permit(dispatcher, new_permit: int) -> None:
        try:
            if hasattr(dispatcher, "set_max_session_permit"):
                dispatcher.set_max_session_permit(int(new_permit))
            elif hasattr(dispatcher, "max_session_permit"):
                setattr(dispatcher, "max_session_permit", int(new_permit))
        except Exception:
            pass

    # ---------- Local HTML → MD -----------------------------------------
    local_html_needed: List[Tuple[str, Path]] = []
    if (("markdown" in pipeline_set) or llm_in_pipeline) and not plan.skip_markdown and not bypass_local:
        for url in seeded_urls:
            if getattr(args, "require_presence", False) and used_index_for_urls and url_index:
                ent = url_index.get(url, {})
                if int(ent.get("has_offering", 0) or 0) != 1 or not bool(ent.get("presence_checked", False)):
                    continue
            html_local = prefer_local_html(bvdid, url)
            if html_local:
                local_html_needed.append((url, html_local))

    # Remote fetch needed if plan doesn't skip
    need_remote_fetch = (not plan.skip_markdown)

    saved_html_run = 0
    saved_md_run = 0
    saved_json_run = 0
    md_suppressed_run = 0
    md_retry_suggested_run = 0

    if local_html_needed:
        async with AsyncWebCrawler(config=browser_cfg) as local_crawler:
            file_urls = [f"file://{p.resolve()}" for _, p in local_html_needed]
            md_cfg = _ensure_c4a_config_defaults(mk_md_config(args))
            results = await local_crawler.arun_many(file_urls, config=md_cfg)
            file2orig = {f"file://{p.resolve()}": u for (u, p) in local_html_needed}
            for r in (results or []):
                file_url = getattr(r, "url", None)
                url = file2orig.get(file_url)
                if not url:
                    continue
                if getattr(r, "success", False):
                    s = _save_result_for_pipeline(
                        bvdid, url, ["markdown"], r, logger, args.md_min_words, save_html_ref=False,
                        md_gate_opts={"cookie_max_fraction": args.md_cookie_max_frac, "require_structure": args.md_require_structure},
                        presence_only=bool(args.presence_only)
                    )
                    saved_md_run += int(s["saved_md"])
                    md_suppressed_run += int(s["md_suppressed"])
                    md_retry_suggested_run += int(s["md_retry_suggested"])
                    if not llm_in_pipeline:
                        if s.get("saved_md") or s.get("md_suppressed") or s.get("md_retry_suggested"):
                            await mark_done_once(url)
                else:
                    err = getattr(r, "error_message", "local-md-fail")
                    logger.warning("[%s] Local HTML→MD failed: %s err=%s", bvdid, url, err)
                    await cp.mark_url_failed(url, f"local-md-error: {err}")

    # ---------- Remote fetch for Markdown (+ JS-only retry) --------------
    if need_remote_fetch:
        local_set = {u for (u, _) in local_html_needed}
        all_remote = [(url_to_item.get(url) or {"url": url, "final_url": url}) for url in seeded_urls if url not in local_set]

        if getattr(args, "require_presence", False) and used_index_for_urls and url_index:
            filtered_remote = []
            for it in all_remote:
                url = (it.get("final_url") or it.get("url"))
                ent = url_index.get(url, {})
                if int(ent.get("has_offering", 0) or 0) == 1 and bool(ent.get("presence_checked", False)):
                    filtered_remote.append(it)
                else:
                    logger.debug("[%s] Skipping remote fetch %s due to require-presence gating", bvdid, url)
            all_remote = filtered_remote
        elif getattr(args, "require_presence", False) and not used_index_for_urls:
            all_remote = []

        pending: set[str] = {(it.get("final_url") or it.get("url")) for it in all_remote if (it.get("final_url") or it.get("url"))}

        if pending:
            logger.info("[%s] Remote fetch for Markdown (and HTML reference): %d URL(s)", bvdid, len(pending))

            md_cfg = _ensure_c4a_config_defaults(mk_md_config(args))

            # Use page-timeout-ms (not company-timeout) for per-page waits
            ipol = PageInteractionPolicy(
                enable_cookie_playbook=True,
                max_in_session_retries=0,
                wait_timeout_ms=max(60000, int(getattr(args, "page_timeout_ms", 120000))),
                delay_before_return_sec=1.5,
            )
            icfg = first_pass_config("about:blank", ipol)

            md_cfg.js_code = icfg.js_code
            md_cfg.wait_for = icfg.wait_for
            md_cfg.delay_before_return_html = max(
                getattr(md_cfg, "delay_before_return_html", 0) or 0.0,
                icfg.delay_before_return_html
            )
            md_cfg.virtual_scroll_config = icfg.virtual_scroll_config

            js_retry_cfg = _ensure_c4a_config_defaults(mk_md_config(args))
            rcfg2 = retry_pass_config("about:blank", ipol)
            js_retry_cfg.js_code = rcfg2.js_code
            js_retry_cfg.wait_for = rcfg2.wait_for
            js_retry_cfg.delay_before_return_html = max(
                getattr(js_retry_cfg, "delay_before_return_html", 0) or 0.0,
                rcfg2.delay_before_return_html
            )
            js_retry_cfg.virtual_scroll_config = rcfg2.virtual_scroll_config
            try:
                setattr(js_retry_cfg, "js_only", True)
            except Exception:
                pass

            # initial per-company slots provided by caller (already capped by SlotAllocator.initial_per_company)
            dispatcher = make_dispatcher(max_concurrency_for_company)
            html_saved_urls: Set[str] = set()

            async def _session_snapshot() -> SessionSnapshot:
                try:
                    summary = await cp_mgr.summary()
                except Exception:
                    summary = {}
                total = max(1, len(summary))  # avoid zero division
                finished = sum(1 for s in summary.values() if bool(s.get("finished")))
                running = max(1, total - finished)
                return SessionSnapshot(total_companies=total, finished_companies=finished, running_companies=running)

            async def _company_snapshot() -> CompanySnapshot:
                try:
                    cpi = await cp_mgr.get(bvdid, company_name=company.name)
                    urls_total = int(cpi.data.get("urls_total") or 0)
                    urls_done  = int(cpi.data.get("urls_done")  or 0)
                except Exception:
                    urls_total = urls_done = 0
                return CompanySnapshot(
                    urls_total=urls_total if urls_total > 0 else None,
                    urls_done=urls_done if urls_done > 0 else None,
                    timeout_rate=None,
                )

            def _apply_max_permit(dispatcher, new_permit: int) -> None:
                try:
                    if hasattr(dispatcher, "set_max_session_permit"):
                        dispatcher.set_max_session_permit(int(new_permit))
                    elif hasattr(dispatcher, "max_session_permit"):
                        setattr(dispatcher, "max_session_permit", int(new_permit))
                except Exception:
                    pass

            async def _maybe_resize_dispatcher(disp) -> None:
                try:
                    ss = await _session_snapshot()
                    # recompute using this company's checkpoint
                    cpi = await cp_mgr.get(bvdid, company_name=company.name)
                    urls_total = int(cpi.data.get("urls_total") or 0)
                    urls_done  = int(cpi.data.get("urls_done")  or 0)
                    cs = CompanySnapshot(
                        urls_total=urls_total if urls_total > 0 else None,
                        urls_done=urls_done if urls_done > 0 else None,
                        timeout_rate=None,
                    )
                    new_slots = slot_alloc.recommend_for_company(bvdid, ss, cs)
                    _apply_max_permit(disp, new_slots)
                    logger.debug("[%s] SlotAllocator recommend: running=%d finished=%d/%d → per-company=%d",
                                 bvdid, ss.running_companies, ss.finished_companies, ss.total_companies, new_slots)
                except Exception:
                    pass

            attempts_by_url: Dict[str, int] = {}
            max_trans = int(args.retry_transport)
            max_soft = int(args.retry_soft_timeout)

            # NEW: batched remote fetch
            batch_size = max(8, int(getattr(args, "remote_batch_size", 64)))
            logger.info("[%s] Remote fetch batching: batch_size=%d page_timeout_ms=%d",
                        bvdid, batch_size, int(getattr(args, "page_timeout_ms", 120000)))

            def _pop_some(p: set[str], n: int) -> List[str]:
                out = []
                for _ in range(min(n, len(p))):
                    out.append(p.pop())
                return out

            attempt = 0
            while pending:
                await net.wait_until_healthy()
                attempt += 1
                urls_this_round = _pop_some(pending, batch_size)
                logger.info("[%s] Remote fetch pass #%d: processing %d of %d pending",
                            bvdid, attempt, len(urls_this_round), len(pending) + len(urls_this_round))
                to_retry: set[str] = set()
                cookie_js_retry: set[str] = set()

                # resize permits based on current tail state before each batch
                await _maybe_resize_dispatcher(dispatcher)

                async with AsyncWebCrawler(config=browser_cfg) as crawler:
                    stream = await crawler.arun_many(urls=urls_this_round, config=md_cfg, dispatcher=dispatcher)

                    async def _handle_result(r):
                        nonlocal saved_html_run, saved_md_run, md_suppressed_run, md_retry_suggested_run
                        url = getattr(r, "url", None)
                        if not url:
                            return
                        if getattr(r, "success", False):
                            s = _save_result_for_pipeline(
                                bvdid, url, ["markdown"], r, logger, args.md_min_words, save_html_ref=True,
                                md_gate_opts={"cookie_max_fraction": args.md_cookie_max_frac, "require_structure": args.md_require_structure},
                                presence_only=bool(args.presence_only),
                                html_saved_urls=html_saved_urls,
                            )
                            saved_html_run += int(s["saved_html"])
                            saved_md_run += int(s["saved_md"])
                            md_suppressed_run += int(s["md_suppressed"])
                            md_retry_suggested_run += int(s["md_retry_suggested"])
                            net.record_success()

                            cookie_case = (s.get("md_retry_suggested") or (s.get("md_suppressed") and ("cookie" in (s.get("md_reason") or "").lower())))
                            if cookie_case:
                                cookie_js_retry.add(url)
                                metrics.incr("markdown.cookie_retry_scheduled", bvdid=bvdid, url=url)
                            else:
                                if not llm_in_pipeline and (s.get("saved_md") or s.get("md_suppressed")):
                                    await mark_done_once(url)
                            return

                        # fallthrough for failure
                        err = getattr(r, "error_message", "") or ""
                        code = getattr(r, "status_code", None)
                        kind = classify_failure(err, code, treat_timeouts_as_transport=bool(args.treat_timeouts_as_transport))

                        if kind == "download":
                            logger.info("[%s] Download navigation; skipping: %s", bvdid, url)
                            await cp.add_note(f"download-skip: {url}")
                            await mark_done_once(url)
                            net.record_success()
                            return

                        if kind == "transport":
                            attempts_by_url[url] = attempts_by_url.get(url, 0) + 1
                            if attempts_by_url[url] <= max_trans:
                                to_retry.add(url)
                                net.record_transport_error()
                                logger.warning("[%s] Transport-like failure; retry %d/%d: %s (code=%s err=%s)",
                                               bvdid, attempts_by_url[url], max_trans, url, code, err)
                            else:
                                logger.warning("[%s] Transport failure exceeded retries: %s", bvdid, url)
                                await cp.mark_url_failed(url, f"http-transport: {code} {err}")
                            return

                        if kind == "soft_timeout":
                            attempts_by_url[url] = attempts_by_url.get(url, 0) + 1
                            if attempts_by_url[url] <= max_soft:
                                to_retry.add(url)
                                logger.warning("[%s] Soft timeout; retry %d/%d; %s (err=%s)",
                                               bvdid, attempts_by_url[url], max_soft, url, err)
                            else:
                                logger.warning("[%s] Soft timeout exceeded retries; marking failed: %s", bvdid, url)
                                await cp.mark_url_failed(url, f"soft-timeout: {err}")
                            return

                        logger.warning("[%s] Failed (non-transport): %s status=%s err=%s", bvdid, url, code, err)
                        await cp.mark_url_failed(url, f"http-error: {code} {err}")

                    if hasattr(stream, "__aiter__"):
                        async for r in stream:
                            if not net.is_healthy():
                                logger.warning("[%s] ConnectivityGuard OPEN — pausing stream...", bvdid)
                                await state.mark_paused_net(bvdid)
                                await net.wait_until_healthy()
                                await state.mark_in_progress(bvdid, stage=",".join(pipeline))
                                logger.info("[%s] Connectivity restored — resuming.", bvdid)
                            await _handle_result(r)
                    else:
                        for r in (stream or []):
                            if not net.is_healthy():
                                logger.warning("[%s] ConnectivityGuard OPEN — pausing stream...", bvdid)
                                await state.mark_paused_net(bvdid)
                                await net.wait_until_healthy()
                                await state.mark_in_progress(bvdid, stage=",".join(pipeline))
                                logger.info("[%s] Connectivity restored — resuming.", bvdid)
                            await _handle_result(r)

                    # JS-only cookie/interstitial retry batch
                    if cookie_js_retry:
                        logger.info("[%s] JS-only retry for %d URL(s) due to cookie/interstitial dominance",
                                    bvdid, len(cookie_js_retry))
                        # Resize again before retry; tail may have advanced
                        await _maybe_resize_dispatcher(dispatcher)
                        retry_results = await crawler.arun_many(
                            urls=list(cookie_js_retry), config=js_retry_cfg, dispatcher=dispatcher
                        )

                        async def _handle_retry(rr):
                            nonlocal saved_html_run, saved_md_run, md_suppressed_run, md_retry_suggested_run
                            u = getattr(rr, "url", None)
                            if not u:
                                return
                            if getattr(rr, "success", False):
                                s2 = _save_result_for_pipeline(
                                    bvdid, u, ["markdown"], rr, logger, args.md_min_words, save_html_ref=True,
                                    md_gate_opts={"cookie_max_fraction": args.md_cookie_max_frac, "require_structure": args.md_require_structure},
                                    presence_only=bool(args.presence_only),
                                    html_saved_urls=html_saved_urls,
                                )
                                saved_html_run += int(s2["saved_html"])
                                saved_md_run += int(s2["saved_md"])
                                md_suppressed_run += int(s2["md_suppressed"])
                                md_retry_suggested_run += int(s2["md_retry_suggested"])
                                net.record_success()
                                if not llm_in_pipeline and (s2.get("saved_md") or s2.get("md_suppressed") or s2.get("md_retry_suggested")):
                                    await mark_done_once(u)
                                metrics.incr("markdown.retry_performed", bvdid=bvdid, url=u)
                            else:
                                err2 = getattr(rr, "error_message", "js-retry-fail")
                                logger.warning("[%s] JS-only retry failed: %s err=%s", bvdid, u, err2)
                                await cp.mark_url_failed(u, f"js-only-retry-error: {err2}")

                        if hasattr(retry_results, "__aiter__"):
                            async for rr in retry_results:
                                await _handle_retry(rr)
                        else:
                            for rr in (retry_results or []):
                                await _handle_retry(rr)
                        cookie_js_retry.clear()

                if to_retry:
                    logger.info("[%s] %d URL(s) scheduled for transport/soft-timeout retry (will be re-queued).", bvdid, len(to_retry))
                    pending |= to_retry

    # ---------- Local LLM over Markdown ---------------------------------
    if llm_in_pipeline and not plan.skip_llm:
        llm_targets: List[Tuple[str, Path]] = []

        used_index_for_urls_flag = used_index_for_urls and bool(url_index)
        if used_index_for_urls_flag:
            for url in seeded_urls:
                ent = url_index.get(url, {})
                # Only accept URLs whose current status is markdown_saved when reusing index for LLM
                if str(ent.get("status", "")).lower() != "markdown_saved":
                    continue
                md_path = ent.get("markdown_path")
                if md_path:
                    p = Path(md_path)
                    if p.exists():
                        if getattr(args, "require_presence", False):
                            if int(ent.get("has_offering", 0) or 0) != 1 or not bool(ent.get("presence_checked", False)):
                                continue
                        llm_targets.append((url, p))

        if not llm_targets:
            for url in seeded_urls:
                if getattr(args, "require_presence", False) and not used_index_for_urls:
                    continue
                md_path = prefer_local_md(bvdid, url)
                if md_path:
                    llm_targets.append((url, md_path))

        logger.info("[%s] LLM local pass over Markdown: %d URL(s) | presence_only=%s",
                    bvdid, len(llm_targets), getattr(args, "presence_only", False))

        if not llm_targets:
            logger.warning("[%s] No Markdown files found for LLM stage. Skipping.", bvdid)
        else:
            async with AsyncWebCrawler(config=browser_cfg) as local_crawler:
                raw_inputs: List[str] = []
                ordered_urls: List[str] = []
                skipped_urls: List[str] = []

                for url, md_path in llm_targets:
                    try:
                        text = md_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception as e:
                        logger.warning("[%s] Failed to read Markdown: %s err=%s", bvdid, md_path, e)
                        await cp.mark_url_failed(url, f"read-md-error: {e}")
                        continue

                    if not text.strip():
                        logger.debug("[%s] Empty Markdown, skip LLM for %s", bvdid, url)
                        skipped_urls.append(url)
                        await mark_done_once(url)
                        continue

                    action, reason, md_stats = evaluate_markdown(
                        text, allow_retry=False,
                        min_meaningful_words=max(1, int(args.md_min_words)),
                        cookie_max_fraction=args.md_cookie_max_frac,
                        require_structure=args.md_require_structure,
                    )

                    if action != "save":
                        logger.debug("[%s] Markdown gated (%s); skipping LLM for %s", bvdid, reason, url)
                        skipped_urls.append(url)
                        await mark_done_once(url)
                        md_suppressed_run += 1
                        continue

                    raw_inputs.append(f"raw:{text}")
                    ordered_urls.append(url)

                total_inputs = len(raw_inputs)
                if not raw_inputs:
                    logger.info("[%s] No valid Markdown to feed into LLM after gating. Skipped=%d",
                                bvdid, len(skipped_urls))
                else:
                    llm_cfg = _ensure_c4a_config_defaults(mk_llm_config(args))
                    logger.info("[%s] Starting LLM local run for %d input(s)...", bvdid, total_inputs)

                    try:
                        LLM_PER_RUN_TIMEOUT = int(getattr(args, "llm_per_run_timeout", None) or 900)
                    except Exception:
                        LLM_PER_RUN_TIMEOUT = 900
                    try:
                        PER_ITEM_TIMEOUT = int(getattr(args, "llm_per_item_timeout", None) or 180)
                    except Exception:
                        PER_ITEM_TIMEOUT = 180

                    try:
                        cor = local_crawler.arun_many(raw_inputs, config=llm_cfg)
                    except Exception as e:
                        logger.exception("[%s] Failed to call arun_many: %s", bvdid, e)
                        cor = None

                    results_iter = None
                    results_list = None

                    try:
                        if cor is None:
                            results_list = []
                        elif hasattr(cor, "__aiter__"):
                            results_iter = cor.__aiter__()  # type: ignore[attr-defined]
                        elif hasattr(cor, "__await__"):
                            try:
                                resolved = await asyncio.wait_for(cor, timeout=float(LLM_PER_RUN_TIMEOUT))
                                if hasattr(resolved, "__aiter__"):
                                    results_iter = resolved.__aiter__()  # type: ignore[attr-defined]
                                else:
                                    results_list = list(resolved or [])
                            except asyncio.TimeoutError:
                                logger.exception("[%s] LLM local run timed out after %ss", bvdid, LLM_PER_RUN_TIMEOUT)
                                for u in ordered_urls:
                                    with contextlib.suppress(Exception):
                                        await cp.mark_url_failed(u, f"local-llm-timeout-{LLM_PER_RUN_TIMEOUT}s")
                                results_list = []
                            except Exception as e:
                                logger.exception("[%s] LLM local crawler run failed: %s", bvdid, e)
                                results_list = []
                        else:
                            try:
                                resolved = await asyncio.wait_for(cor, timeout=float(LLM_PER_RUN_TIMEOUT))
                                if hasattr(resolved, "__aiter__"):
                                    results_iter = resolved.__aiter__()  # type: ignore[attr-defined]
                                else:
                                    results_list = list(resolved or [])
                            except Exception as e:
                                logger.exception("[%s] Could not resolve arun_many result: %s", bvdid, e)
                                results_list = []
                    except Exception as e:
                        logger.exception("[%s] Unexpected error while preparing LLM run: %s", bvdid, e)
                        results_list = []

                    async def _process_result_at_index(i: int, r):
                        nonlocal saved_json_run, success_count, fail_count
                        url = ordered_urls[i] if i < len(ordered_urls) else None
                        if not url:
                            return
                        try:
                            try:
                                resp_preview = _short_preview(getattr(r, "extracted_content", getattr(r, "raw", None)), length=800)
                                logger.debug("[%s] LLM response preview for idx=%d url=%s: %s", bvdid, i, url, resp_preview)
                            except Exception:
                                pass

                            if getattr(r, "success", False):
                                s = _save_result_for_pipeline(
                                    bvdid, url, ["llm"], r, logger, args.md_min_words,
                                    presence_only=bool(args.presence_only)
                                )
                                saved_json_run += int(s["saved_json"])
                                success_count += 1
                                await mark_done_once(url)
                            else:
                                err = getattr(r, "error_message", "local-llm-fail")
                                logger.warning("[%s] LLM failed for %s err=%s", bvdid, url, err)
                                await cp.mark_url_failed(url, f"local-llm-error: {err}")
                                fail_count += 1
                        except Exception as e:
                            logger.exception("[%s] Error while processing LLM result for %s: %s", bvdid, url, e)
                            with contextlib.suppress(Exception):
                                await cp.mark_url_failed(url, f"local-llm-result-processing-error: {e}")
                            fail_count += 1

                    success_count = fail_count = 0
                    saved_json_run = 0

                    if results_list is not None:
                        for i, r in enumerate(results_list or []):
                            await _process_result_at_index(i, r)
                    else:
                        i = 0
                        while True:
                            try:
                                r = await asyncio.wait_for(results_iter.__anext__(), timeout=float(PER_ITEM_TIMEOUT))  # type: ignore[union-attr]
                                await _process_result_at_index(i, r)
                                i += 1
                                if i >= len(ordered_urls):
                                    break
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                logger.exception("[%s] LLM streaming item #%d timed out after %ss (continuing)", bvdid, i, PER_ITEM_TIMEOUT)
                                with contextlib.suppress(Exception):
                                    if i < len(ordered_urls):
                                        await cp.mark_url_failed(ordered_urls[i], f"local-llm-stream-item-timeout-{PER_ITEM_TIMEOUT}s")
                                i += 1
                                continue
                            except asyncio.CancelledError:
                                logger.warning("[%s] LLM stream iteration cancelled; stopping early", bvdid)
                                break
                            except Exception as e:
                                logger.exception("[%s] Exception while iterating LLM stream: %s", bvdid, e)
                                with contextlib.suppress(Exception):
                                    if i < len(ordered_urls):
                                        await cp.mark_url_failed(ordered_urls[i], f"local-llm-stream-iteration-exc: {e}")
                                i += 1
                                continue

                    logger.info(
                        "[%s] LLM stage summary: total=%d success=%d fail=%d skipped=%d presence_only=%s",
                        bvdid, total_inputs, success_count, fail_count, len(skipped_urls),
                        getattr(args, "presence_only", False),
                    )

    cp.data.setdefault("saves", {})
    prev = cp.data["saves"]
    prev["saved_html_total"] = prev.get("saved_html_total", 0) + saved_html_run
    prev["saved_md_total"]   = prev.get("saved_md_total", 0) + saved_md_run
    prev["saved_json_total"] = prev.get("saved_json_total", 0) + saved_json_run
    prev["md_suppressed_total"] = prev.get("md_suppressed_total", 0) + md_suppressed_run
    prev["md_retry_suggested_total"] = prev.get("md_retry_suggested_total", 0) + md_retry_suggested_run
    await cp.save()

    with contextlib.suppress(Exception):
        await state.update_artifacts(
            bvdid,
            saved_html_total=prev["saved_html_total"],
            saved_md_total=prev["saved_md_total"],
            saved_json_total=prev["saved_json_total"],
        )
        await state.update_counts(
            bvdid,
            urls_total=int(cp.data.get("urls_total", 0)),
            urls_done=int(cp.data.get("urls_done", 0)),
            urls_failed=int(cp.data.get("urls_failed", 0)),
        )

    await cp.mark_finished()
    write_last_crawl_date(bvdid)
    await state.mark_done(
        bvdid,
        stage=",".join(pipeline),
        presence_only=bool(getattr(args, "presence_only", False))
    )
    logger.info(
        "[%s] DONE: processed=%s/%s failed=%s | run: saved_html=%d saved_md=%d saved_json=%d md_suppressed=%d md_retry=%d",
        bvdid,
        cp.data.get("urls_done"), cp.data.get("urls_total"), cp.data.get("urls_failed"),
        saved_html_run, saved_md_run, saved_json_run, md_suppressed_run, md_retry_suggested_run,
    )
    metrics.incr("company.done", bvdid=bvdid, stage=",".join(pipeline))


async def main_async() -> None:
    parser = build_parser()
    args = parser.parse_args()

    level = getattr(logging, args.log_level)
    log_ext = LoggingExtension(
        global_level=level,
        per_company_level=level,
        max_open_company_logs=args.max_open_company_logs,
        enable_session_log=args.enable_session_log,
    )
    root_logger = logging.getLogger("run_crawl")
    root_logger.setLevel(level)

    try:
        load_lang(args.lang or "en")
        root_logger.info("Loaded language settings for: %s", args.lang or "en")
    except Exception as e:
        root_logger.warning("Failed to load language settings for '%s': %s — falling back to default", args.lang, e)

    pipeline = parse_pipeline(args.pipeline)
    root_logger.info("Pipeline=%s", pipeline)
    root_logger.info("Seeder live-check: %s | DualBM25=%s α=%.2f",
                     "ON" if args.live_check else "OFF",
                     "ON" if args.use_dual_bm25 else "OFF",
                     float(args.dual_alpha))
    root_logger.info(
        "MD config: min_words=%d threshold=%.2f thresh_type=%s min_block_words=%d src=%s ignore_links=%s ignore_images=%s cookie_max_frac=%.2f require_structure=%s",
        args.md_min_words, args.md_threshold, args.md_threshold_type, args.md_min_block_words,
        args.md_content_source, bool(args.md_ignore_links), bool(args.md_ignore_images),
        float(args.md_cookie_max_frac), bool(args.md_require_structure),
    )
    root_logger.info("Fetch tuning: page_timeout_ms=%d remote_batch_size=%d",
                     int(getattr(args, "page_timeout_ms", 120000)),
                     int(getattr(args, "remote_batch_size", 64)))
    root_logger.info("LLM config: presence_only=%s require_presence=%s", args.presence_only, getattr(args, "require_presence", False))

    # --- Load input companies (multi-format) ---
    if getattr(args, "source_dir", None):
        src_dir: Path = args.source_dir
        if not src_dir.exists() or not src_dir.is_dir():
            root_logger.error("--source-dir is not a directory: %s", src_dir)
            log_ext.close()
            return

        patterns = [p.strip() for p in (args.source_pattern or "").split(",") if p.strip()]
        companies: List[CompanyInput] = load_companies_from_dir(src_dir, patterns=patterns, recursive=True)

        if not companies:
            root_logger.error("No supported data files (or no valid rows) found under: %s", src_dir)
            log_ext.close()
            return

        if args.limit is not None and args.limit > 0:
            companies = companies[: args.limit]
            root_logger.info("Applied global limit: %d", args.limit)

        root_logger.info("Loaded %d companies from %s", len(companies), src_dir)

    else:
        companies = load_companies_from_source(args.source, limit=args.limit)
        if not companies:
            root_logger.error("No valid rows in source: %s", args.source)
            log_ext.close()
            return
        root_logger.info("Loaded %d companies from %s", len(companies), args.source)

    if not companies:
        root_logger.error("No valid rows in input. Exiting.")
        log_ext.close()
        return

    # Use language-aware defaults (load_lang has been called above)
    include_patterns = get_default_include_patterns() + ([p.strip() for p in args.include.split(",") if p.strip()] if args.include else [])
    exclude_patterns = get_default_exclude_patterns() + ([p.strip() for p in args.exclude.split(",") if p.strip()] if args.exclude else [])
    lang_regions = set(x.strip().lower() for x in args.lang_accept_en_regions.split(",") if x.strip())

    n = len(companies)

    # Build SlotAllocator from CLI
    slot_cfg = SlotConfig(
        max_slots=int(args.max_slots),
        per_company_cap=int(args.slot_cap_per_company),
        per_company_min=int(args.slot_min_per_company),
        tail_start_fraction=float(args.slot_tail_frac),
        tail_boost_cap=(int(args.slot_tail_cap) if int(args.slot_tail_cap) > 0 else None),
    )
    slot_alloc = SlotAllocator(slot_cfg)

    per_company_initial = slot_alloc.initial_per_company(total_companies=n)
    root_logger.info(
        "Loaded %d companies | global slots=%d → per-company(initial)=%d [cap=%d, min=%d, tail_frac=%.2f, tail_cap=%s]",
        n, args.max_slots, per_company_initial,
        slot_cfg.per_company_cap, slot_cfg.per_company_min,
        slot_cfg.tail_start_fraction,
        (slot_cfg.tail_boost_cap if slot_cfg.tail_boost_cap is not None else "reuse-cap"),
    )
    root_logger.debug("Include patterns: %s", include_patterns)
    root_logger.debug("Exclude patterns: %s", exclude_patterns)
    metrics.set("session.company_count", float(n))
    metrics.set("session.per_company_slots", float(per_company_initial))

    cp_mgr = CheckpointManager()
    await cp_mgr.mark_global_start()
    scanner = ArtifactScanner()
    state = GlobalState()

    # Connectivity guard
    net = ConnectivityGuard(
        probe_host="8.8.8.8",
        probe_port=53,
        interval_s=5.0,
        trip_heartbeats=2,
        connect_timeout_s=2.5,
        base_cooloff_s=5.0,
        max_cooloff_s=300.0,
        backoff_factor=2.0,
        jitter_s=0.4,
        logger=logging.getLogger("run_crawl.net"),
    )
    await net.start()

    sem = asyncio.Semaphore(min(n, args.max_slots))

    async def _runner(c: CompanyInput):
        async with sem:
            token = log_ext.set_company_context(c.bvdid)

            async def _attempt(timeout_s: float, note: str) -> bool:
                try:
                    with metrics.time("company.duration_s", bvdid=c.bvdid):
                        await asyncio.wait_for(
                            _crawl_company(
                                c,
                                pipeline=pipeline,
                                seeding_source=args.seeding_source,
                                max_concurrency_for_company=per_company_initial,
                                include_patterns=include_patterns,
                                exclude_patterns=exclude_patterns,
                                respect_crawl_date=args.respect_crawl_date,
                                bypass_local=args.bypass_local,
                                force_seeder_cache=args.force_seeder_cache,
                                bm25_query=args.query,
                                bm25_score_threshold=args.score_threshold,
                                max_urls=args.max_urls,
                                company_max_pages=args.company_max_pages,
                                hits_per_sec=args.hits_per_sec,
                                cp_mgr=cp_mgr,
                                log_ext=log_ext,
                                discover_brands=args.discover_brands,
                                drop_universal_externals=args.drop_universal_externals,
                                lang_primary=args.lang_primary.lower(),
                                lang_accept_en_regions=lang_regions,
                                lang_strict_cctld=args.lang_strict_cctld,
                                args=args,
                                state=state,
                                net=net,
                                slot_alloc=slot_alloc,
                            ),
                            timeout=float(timeout_s),
                        )
                    return True
                except asyncio.TimeoutError:
                    await state.mark_failed(c.bvdid, f"timeout-after-{timeout_s}s")
                    with contextlib.suppress(Exception):
                        cp = await cp_mgr.get(c.bvdid, company_name=c.name)
                        await cp.add_note(f"{note}: Timed out after {int(timeout_s)}s")
                        await cp.save()
                    logging.getLogger("run_crawl").error("[%s] Timeout after %ss; attempt aborted", c.bvdid, timeout_s)
                    metrics.incr("company.timeout", bvdid=c.bvdid)
                    return False

            try:
                first_ok = await _attempt(float(args.company_timeout), note="attempt#1")
                if first_ok:
                    return

                urls_total = 0
                urls_done = 0
                with contextlib.suppress(Exception):
                    cp = await cp_mgr.get(c.bvdid, company_name=c.name)
                    urls_total = int(cp.data.get("urls_total") or 0)
                    urls_done  = int(cp.data.get("urls_done")  or 0)

                has_err, pat = (False, "")
                try:
                    get_log_path = getattr(log_ext, "get_company_log_path", None)
                    company_log_path = get_log_path(c.bvdid) if callable(get_log_path) else None
                    if company_log_path:
                        has_err, pat = company_log_tail_has_error(company_log_path)
                except Exception:
                    pass

                if has_err:
                    await state.mark_failed(c.bvdid, f"log-tail-error: {pat}")
                    logging.getLogger("run_crawl").error("[%s] Aborting retry due to log-tail error: %s", c.bvdid, pat)
                    return

                rec = recommend_company_timeouts(
                    urls_total,
                    base_timeout=int(args.company_timeout),
                    per_url_seconds=0.9,
                    startup_buffer=300,
                )
                retry_budget = max(int(args.company_timeout) + 900, int(rec["company_timeout"]))
                logging.getLogger("run_crawl").info(
                    "[%s] Retrying with extended budget: urls_total=%d, urls_done=%d, timeout=%ss",
                    c.bvdid, urls_total, urls_done, retry_budget
                )
                second_ok = await _attempt(float(retry_budget), note="attempt#2-extended")
                if second_ok:
                    return

            except Exception as e:
                await state.mark_failed(c.bvdid, f"runner-exception: {e}")
                logging.getLogger("run_crawl").exception("Unhandled error for %s: %s", c.bvdid, e)
                metrics.incr("company.exception", bvdid=c.bvdid)
            finally:
                log_ext.reset_company_context(token)

    tasks = [asyncio.create_task(_runner(c), name=c.bvdid) for c in companies]

    pending: set[asyncio.Task] = set(tasks)
    last_progress = time.monotonic()
    last_done_by_bvdid: dict[str, int] = {}

    try:
        while pending:
            done, pending = await asyncio.wait(pending, timeout=float(args.stall_interval))
            now = time.monotonic()
            if done:
                last_progress = now

            with contextlib.suppress(Exception):
                summary = await CheckpointManager().summary()
                for bvdid, s in summary.items():
                    cur = int(s.get("done") or 0)
                    prev = last_done_by_bvdid.get(bvdid, -1)
                    if cur > prev:
                        last_progress = now
                    last_done_by_bvdid[bvdid] = cur

            if pending and (now - last_progress) > float(args.stall_timeout):
                logging.getLogger("run_crawl").warning("Stall detected: %d task(s) still pending; cancelling stragglers...", len(pending))
                for t in pending:
                    t.cancel()
                with contextlib.suppress(Exception):
                    await asyncio.gather(*pending, return_exceptions=True)
                break
    finally:
        summary = await CheckpointManager().summary()
        logging.getLogger("run_crawl").info("Session summary:")
        for hid, s in summary.items():
            logging.getLogger("run_crawl").info(
                "  %s: %s/%s done, %s failed, ratio=%.2f%%, finished=%s",
                hid, s["done"], s["total"], s["failed"], 100.0 * s["ratio"], s["finished"]
            )
        with contextlib.suppress(Exception):
            await net.stop()
        for h in logging.getLogger().handlers:
            with contextlib.suppress(Exception):
                h.flush()
        metrics.set("session.finished_companies", float(len(summary)))
        log_ext.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()