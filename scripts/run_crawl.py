from __future__ import annotations

import os
import argparse
import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from config.language_settings import load_lang

from components.csv_loader import CompanyInput, load_companies_from_csv
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
    per_company_slots,
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
)

# Page interaction: use both first + retry configs
from extensions.page_interaction import PageInteractionPolicy, first_pass_config, retry_pass_config


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
        save_stage_output(bvdid, url, html, stage="html")
        stats["saved_html"] = True
        from extensions.run_utils import find_existing_artifact
        html_path = find_existing_artifact(bvdid, url, "html")
        # persist status
        try:
            upsert_url_index(bvdid, url, html_path=html_path, status="html_saved")
        except Exception as e:
            logger.debug("[%s] url_index upsert (html_saved) failed for %s: %s", bvdid, url, e)

    if ("markdown" in pipeline or "llm" in pipeline):
        chosen_md = (fit_md or "").strip()
        if chosen_md:
            action, reason, md_stats = evaluate_markdown(
                chosen_md,
                url=url,
                allow_retry=True,
                min_meaningful_words=max(1, int(md_min_words)),
                interstitial_max_share=0.60,
                interstitial_min_hits=2,
                cookie_max_fraction=(md_gate_opts or {}).get("cookie_max_fraction", 0.15),
                require_structure=bool((md_gate_opts or {}).get("require_structure", True)),
                generator_ignores_links=True,
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
                    # pass markdown_words into url_index for later resume/insight
                    upsert_url_index(
                        bvdid,
                        url,
                        markdown_path=md_path,
                        status="markdown_saved",
                        markdown_words=int(md_stats.get("total_words", 0) or 0),
                    )
                except Exception as e:
                    logger.debug("[%s] url_index upsert (markdown_saved) failed for %s: %s", bvdid, url, e)

            elif action == "retry":
                stats["md_retry_suggested"] = True
                metrics.incr("markdown.retry_suggested", bvdid=bvdid, reason=reason)
                try:
                    # persist retry status so operators can inspect and resume behavior
                    upsert_url_index(bvdid, url, status="markdown_retry")
                except Exception as e:
                    logger.debug("[%s] url_index upsert (markdown_retry) failed for %s: %s", bvdid, url, e)
            else:
                stats["md_suppressed"] = True
                metrics.incr("markdown.suppressed", bvdid=bvdid, reason=reason)
                try:
                    upsert_url_index(bvdid, url, status="markdown_suppressed")
                except Exception as e:
                    logger.debug("[%s] url_index upsert (markdown_suppressed) failed for %s: %s", bvdid, url, e)
        else:
            stats["md_suppressed"] = True
            stats["md_reason"] = "empty"
            try:
                upsert_url_index(bvdid, url, status="markdown_suppressed")
            except Exception as e:
                logger.debug("[%s] url_index upsert (markdown_suppressed-empty) failed for %s: %s", bvdid, url, e)

    # LLM stage: presence-only handling
    if "llm" in pipeline and extracted is not None:
        if presence_only:
            # parse presence result (0 or 1)
            has, _conf, _rat = parse_presence_result(extracted, default=False)
            has_int = 1 if has else 0
            try:
                # write presence flag into url_index.json (checkpoints) and mark presence_checked
                upsert_url_index(bvdid, url, has_offering=has_int, presence_checked=True, status="presence_checked")
            except Exception as e:
                logger.debug("[%s] url_index upsert failed while writing presence for %s: %s", bvdid, url, e)
            stats["presence_written"] = True
            # Do not write separate llm artifact file in presence-only mode
            stats["saved_json"] = False
        else:
            # original behavior: save extracted content as llm artifact
            # save LLM artifacts under the 'json' stage (extensions.output_paths expects html|markdown|json)
            save_stage_output(bvdid, url, extracted, stage="json")
            stats["saved_json"] = True
            from extensions.run_utils import find_existing_artifact
            json_path = find_existing_artifact(bvdid, url, "llm")
            metrics.incr("llm.saved", bvdid=bvdid)
            # detect empty-extraction semantics (offerings empty)
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
    scanner: ArtifactScanner,
    state: GlobalState,
    net: ConnectivityGuard,
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

    # 1) URL set (index vs network)
    seeded_items: List[Dict[str, Any]] = []
    seed_stats: Dict[str, Any] = {}
    used_index_for_urls = False
    url_index: Dict[str, Any] = {}

    if plan.skip_seeding_entirely:
        url_index = load_url_index(bvdid)
        used_index_for_urls = True
        urls_from_index = list(url_index.keys())
        seeded_items = [{"url": u, "status": "valid"} for u in urls_from_index]
        seed_stats = {
            "discovered_total": len(seeded_items),
            "filtered_total": len(seeded_items),
            "seed_roots": [company.url],
            "seed_brand_count": 0,
            "resume_mode": "url_index",
        }
        await state.set_resume_mode(bvdid, "url_index")
        logger.info("[%s] Using url_index.json with %d URL(s).", bvdid, len(seeded_items))

    if not used_index_for_urls:
        while True:
            await net.wait_until_healthy()
            opened_before = net.snapshot_open_events()
            try:
                with metrics.time("seeding.duration_s", bvdid=bvdid):
                    seeded_items, seed_stats = await seed_urls(
                        company.url,
                        source=seeding_source,
                        include=include_patterns,
                        exclude=exclude_patterns,
                        query=bm25_query,
                        score_threshold=bm25_score_threshold,
                        live_check=args.live_check,
                        force=force_seeder_cache,
                        max_urls=max_urls,
                        company_max_pages=company_max_pages,
                        hits_per_sec=hits_per_sec,
                        drop_universal_externals=drop_universal_externals,
                        lang_primary=lang_primary,
                        lang_accept_en_regions=lang_accept_en_regions,
                        lang_strict_cctld=lang_strict_cctld,
                        discover_brands=discover_brands,
                        use_dual_bm25=bool(args.use_dual_bm25),
                        dual_alpha=float(args.dual_alpha),
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

    cp.data.setdefault("seeding", {}).update(seed_stats)
    cp.data.update(aggregate_seed_by_root(seeded_items, base_root=company.url))
    await cp.save()

    logger.info(
        "[%s] Seeding: discovered=%s, filtered=%s, roots=%d, brands=%d",
        bvdid,
        seed_stats.get("discovered_total", 0),
        seed_stats.get("filtered_total", 0),
        len(seed_stats.get("seed_roots", [])),
        seed_stats.get("seed_brand_count", 0),
    )
    metrics.set("seeding.filtered", float(seed_stats.get("filtered_total", 0)), bvdid=bvdid)

    if pipeline == ["seed"]:
        seeded_urls_for_index = [it.get("final_url") or it.get("url") for it in seeded_items if (it.get("final_url") or it.get("url"))]
        total_index = write_url_index_seed_only(bvdid, seeded_urls_for_index)
        await state.record_url_index(bvdid, count=total_index)
        cp.data.update({"urls_total": int(seed_stats.get("filtered_total", 0)), "seed_source": seeding_source})
        cp.data.update(aggregate_seed_by_root(seeded_items, base_root=company.url))
        await cp.save()
        await cp.mark_finished()
        write_last_crawl_date(bvdid)
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

    if not used_index_for_urls:
        seeded_urls_for_index = [it.get("final_url") or it.get("url") for it in seeded_items if (it.get("final_url") or it.get("url"))]
        total_index = write_url_index_seed_only(bvdid, seeded_urls_for_index)
        try:
            await state.record_url_index(bvdid, count=total_index)
        except Exception:
            pass

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

    # If require_presence flag is set and we used the index, filter URLs that do not
    # have both presence_checked==True and has_offering==1.
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
        # conservative approach: if we don't have the index we cannot satisfy presence gating
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
        logger.info("[%s] No URLs after seeding/resume. DONE.", bvdid)
        metrics.incr("company.done", bvdid=bvdid, stage="none")
        return

    pipeline_set = set(pipeline)
    llm_in_pipeline = ("llm" in pipeline_set)

    logger.info("[%s] Resume check: plan.skip_markdown=%s used_index_for_urls=%s",
                bvdid, bool(plan.skip_markdown), bool(used_index_for_urls))

    completion_stage = pipeline[-1]
    if (completion_stage == "llm" and plan.skip_llm) or (completion_stage == "markdown" and plan.skip_markdown):
        for url in seeded_urls:
            await cp.mark_url_done(url)
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

    # Build list of local HTML files to run markdown generation over (existing local HTML)
    local_html_needed: List[Tuple[str, Path]] = []
    if (("markdown" in pipeline_set) or llm_in_pipeline) and not plan.skip_markdown and not bypass_local:
        for url in seeded_urls:
            # If require_presence true, ensure url_index entry satisfies it
            if getattr(args, "require_presence", False) and used_index_for_urls and url_index:
                ent = url_index.get(url, {})
                if int(ent.get("has_offering", 0) or 0) != 1 or not bool(ent.get("presence_checked", False)):
                    continue
            html_local = prefer_local_html(bvdid, url)
            if html_local:
                local_html_needed.append((url, html_local))

    need_remote_fetch = not plan.skip_markdown

    if llm_in_pipeline and ("markdown" not in pipeline_set) and (not need_remote_fetch):
        logger.info("[%s] Skipping remote fetch: plan.skip_markdown=%s", bvdid, plan.skip_markdown)

    saved_html_run = 0
    saved_md_run = 0
    saved_json_run = 0
    md_suppressed_run = 0
    md_retry_suggested_run = 0

    # --- Local HTML → MD ----------------------------------------------------
    if local_html_needed:
        async with AsyncWebCrawler() as local_crawler:
            file_urls = [f"file://{p.resolve()}" for _, p in local_html_needed]

            md_cfg = mk_md_config(args)
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

                    # Mark as processed for checkpointing if this run is not doing LLM.
                    # Treat saved / suppressed / retry as "processed" so future runs won't re-seed/re-fetch.
                    if not llm_in_pipeline:
                        if s.get("saved_md") or s.get("md_suppressed") or s.get("md_retry_suggested"):
                            await cp.mark_url_done(url)

                else:
                    err = getattr(r, "error_message", "local-md-fail")
                    logger.warning("[%s] Local HTML→MD failed: %s err=%s", bvdid, url, err)
                    await cp.mark_url_failed(url, f"local-md-error: {err}")

    # --- Remote fetch for Markdown (+ AUTO JS-only retry on retry/suppress) --
    if need_remote_fetch:
        local_set = {u for (u, _) in local_html_needed}
        # Build all_remote only for seeded_urls not present locally
        all_remote = [(url_to_item.get(url) or {"url": url, "final_url": url}) for url in seeded_urls if url not in local_set]

        # If require_presence and using index, filter remote targets to only presence==1 and presence_checked
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

            md_cfg = mk_md_config(args)

            ipol = PageInteractionPolicy(
                enable_cookie_playbook=True,
                max_in_session_retries=0,  # we'll manage second pass explicitly below
                wait_timeout_ms=max(60000, int(args.company_timeout * 1000)),
                delay_before_return_sec=1.5,
            )
            icfg = first_pass_config("about:blank", ipol)

            # Merge interaction into md_cfg so the normal remote pass waits correctly
            md_cfg.js_code = icfg.js_code
            md_cfg.wait_for = icfg.wait_for
            md_cfg.delay_before_return_html = max(
                getattr(md_cfg, "delay_before_return_html", 0) or 0.0,
                icfg.delay_before_return_html
            )
            md_cfg.virtual_scroll_config = icfg.virtual_scroll_config
            # Keep md_cfg.cache_mode as provided by mk_md_config; override only if you want enabled:
            # md_cfg.cache_mode = CacheMode.ENABLED

            # Prepare explicit JS-only retry config (for batch retries) — base it on mk_md_config too
            js_retry_cfg = mk_md_config(args)
            rcfg2 = retry_pass_config("about:blank", ipol)
            # merge retry policy bits into js_retry_cfg
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

            dispatcher = make_dispatcher(max_concurrency_for_company)

            attempts_by_url: Dict[str, int] = {}
            max_trans = int(args.retry_transport)
            max_soft = int(args.retry_soft_timeout)

            attempt = 0
            while pending:
                await net.wait_until_healthy()
                attempt += 1
                logger.info("[%s] Remote fetch pass #%d: %d URL(s) pending", bvdid, attempt, len(pending))
                urls_this_round = list(pending)
                pending.clear()
                to_retry: set[str] = set()
                cookie_js_retry: set[str] = set()

                async with AsyncWebCrawler() as crawler:
                    # Use md_cfg (markdown-enabled) for the normal remote pass
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
                                presence_only=bool(args.presence_only)
                            )
                            saved_html_run += int(s["saved_html"])
                            saved_md_run += int(s["saved_md"])
                            md_suppressed_run += int(s["md_suppressed"])
                            md_retry_suggested_run += int(s["md_retry_suggested"])
                            net.record_success()

                            # Mark as processed for checkpointing if this run is not doing LLM.
                            # Count pages that are saved, suppressed, or marked retry as "done" so future runs will skip them.
                            if not llm_in_pipeline:
                                if s.get("saved_md") or s.get("md_suppressed") or s.get("md_retry_suggested"):
                                    await cp.mark_url_done(url)

                            # If gate suggested retry/suppression due to cookie-dominant, schedule JS-only retry in batch
                            if (s.get("md_retry_suggested") or (s.get("md_suppressed") and ("cookie" in (s.get("md_reason") or "").lower()))):
                                cookie_js_retry.add(url)
                                metrics.incr("markdown.cookie_retry_scheduled", bvdid=bvdid, url=url)
                            return


                        err = getattr(r, "error_message", "") or ""
                        code = getattr(r, "status_code", None)
                        kind = classify_failure(err, code, treat_timeouts_as_transport=bool(args.treat_timeouts_as_transport))

                        if kind == "download":
                            logger.info("[%s] Download navigation; skipping: %s", bvdid, url)
                            await cp.add_note(f"download-skip: {url}")
                            await cp.mark_url_done(url)
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

                    # iterate the stream (async generator or list)
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

                    # ---- AUTO JS-only second pass for cookie/interstitial cases (batch)
                    if cookie_js_retry:
                        logger.info("[%s] JS-only retry for %d URL(s) due to cookie/interstitial dominance",
                                    bvdid, len(cookie_js_retry))

                        # Use js_retry_cfg for the batch retry (it includes markdown_generator)
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
                                    presence_only=bool(args.presence_only)
                                )
                                saved_html_run += int(s2["saved_html"])
                                saved_md_run += int(s2["saved_md"])
                                md_suppressed_run += int(s2["md_suppressed"])
                                md_retry_suggested_run += int(s2["md_retry_suggested"])
                                net.record_success()

                                # Count this URL as done for checkpointing if not running LLM in this pass.
                                if not llm_in_pipeline:
                                    if s2.get("saved_md") or s2.get("md_suppressed") or s2.get("md_retry_suggested"):
                                        await cp.mark_url_done(u)
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
                    logger.info("[%s] %d URL(s) scheduled for transport/soft-timeout retry.", bvdid, len(to_retry))
                    pending |= to_retry

    # --- Local LLM over Markdown -------------------------------------------
    if llm_in_pipeline and not plan.skip_llm:
        llm_targets: List[Tuple[str, Path]] = []

        used_index_for_urls_flag = used_index_for_urls and bool(url_index)
        if used_index_for_urls_flag:
            for url in seeded_urls:
                ent = url_index.get(url, {})
                md_path = ent.get("markdown_path")
                if md_path:
                    p = Path(md_path)
                    if p.exists():
                        # If require_presence is set, ensure ent satisfies gating
                        if getattr(args, "require_presence", False):
                            if int(ent.get("has_offering", 0) or 0) != 1 or not bool(ent.get("presence_checked", False)):
                                continue
                        llm_targets.append((url, p))

        if not llm_targets:
            for url in seeded_urls:
                # If require_presence is set and we don't have index use conservative skip
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
            async with AsyncWebCrawler() as local_crawler:
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
                        await cp.mark_url_done(url)
                        continue

                    action, reason, md_stats = evaluate_markdown(
                        text, url=url, allow_retry=False,
                        min_meaningful_words=max(1, int(args.md_min_words)),
                        interstitial_max_share=0.60, interstitial_min_hits=2,
                        cookie_max_fraction=args.md_cookie_max_frac,
                        require_structure=args.md_require_structure,
                        generator_ignores_links=True,
                    )

                    if action != "save":
                        logger.debug("[%s] Markdown gated (%s); skipping LLM for %s", bvdid, reason, url)
                        skipped_urls.append(url)
                        await cp.mark_url_done(url)
                        md_suppressed_run += 1
                        continue

                    logger.debug(
                        "[%s] Queuing for LLM (len=%d words≈%d): %s",
                        bvdid, len(text), md_stats.get("total_words", 0), url
                    )
                    # prefix "raw:" works with the crawler/processor to mark inline input
                    raw_inputs.append(f"raw:{text}")
                    ordered_urls.append(url)

                total_inputs = len(raw_inputs)
                if not raw_inputs:
                    logger.info("[%s] No valid Markdown to feed into LLM after gating. Skipped=%d",
                                bvdid, len(skipped_urls))
                else:
                    llm_cfg = mk_llm_config(args)
                    logger.info("[%s] Starting LLM local run for %d input(s)...", bvdid, total_inputs)
                    logger.debug("[%s] LLM config preview: provider=%s base_url=%s stream=%s",
                                 bvdid,
                                 getattr(getattr(llm_cfg, "extraction_strategy", None), "llm_config", None) and getattr(llm_cfg.extraction_strategy.llm_config, "provider", None),
                                 getattr(getattr(llm_cfg, "extraction_strategy", None), "llm_config", None) and getattr(llm_cfg.extraction_strategy.llm_config, "base_url", None),
                                 getattr(llm_cfg, "stream", None))

                    # Timeout per-LUN run so we do not stall the whole company run if model hangs.
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
                            results_iter = cor.__aiter__()
                            logger.debug("[%s] arun_many returned async generator (direct).", bvdid)
                        elif hasattr(cor, "__await__"):
                            try:
                                resolved = await asyncio.wait_for(cor, timeout=float(LLM_PER_RUN_TIMEOUT))
                                if hasattr(resolved, "__aiter__"):
                                    results_iter = resolved.__aiter__()
                                    logger.debug("[%s] arun_many resolved to async generator.", bvdid)
                                else:
                                    results_list = list(resolved or [])
                                    logger.debug("[%s] arun_many resolved to list (items=%d).", bvdid, len(results_list))
                            except asyncio.TimeoutError:
                                logger.exception("[%s] LLM local run timed out after %ss", bvdid, LLM_PER_RUN_TIMEOUT)
                                for u in ordered_urls:
                                    try:
                                        await cp.mark_url_failed(u, f"local-llm-timeout-{LLM_PER_RUN_TIMEOUT}s")
                                    except Exception:
                                        pass
                                results_list = []
                            except Exception as e:
                                logger.exception("[%s] LLM local crawler run failed: %s", bvdid, e)
                                results_list = []
                        else:
                            try:
                                resolved = await asyncio.wait_for(cor, timeout=float(LLM_PER_RUN_TIMEOUT))
                                if hasattr(resolved, "__aiter__"):
                                    results_iter = resolved.__aiter__()
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
                                extracted = getattr(r, "extracted_content", "")
                                snippet = str(extracted)[:200].replace("\n", " ") if extracted else ""
                                logger.debug("[%s] LLM OK: %s | extracted_preview=\"%s\"", bvdid, url, snippet[:200])
                                # mark as done for checkpointing
                                await cp.mark_url_done(url)
                            else:
                                err = getattr(r, "error_message", "local-llm-fail")
                                logger.warning("[%s] LLM failed for %s err=%s", bvdid, url, err)
                                await cp.mark_url_failed(url, f"local-llm-error: {err}")
                                fail_count += 1
                        except Exception as e:
                            logger.exception("[%s] Error while processing LLM result for %s: %s", bvdid, url, e)
                            try:
                                await cp.mark_url_failed(url, f"local-llm-result-processing-error: {e}")
                            except Exception:
                                pass
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
                                r = await asyncio.wait_for(results_iter.__anext__(), timeout=float(PER_ITEM_TIMEOUT))
                                await _process_result_at_index(i, r)
                                i += 1
                                if i >= len(ordered_urls):
                                    break
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                logger.exception("[%s] LLM streaming item #%d timed out after %ss (marking this URL failed, continuing)", bvdid, i, PER_ITEM_TIMEOUT)
                                try:
                                    raw_in_preview = _short_preview(raw_inputs[i] if i < len(raw_inputs) else "<no-input>", length=800)
                                    logger.debug("[%s] LLM timed-out input preview idx=%d: %s", bvdid, i, raw_in_preview)
                                except Exception:
                                    pass
                                if i < len(ordered_urls):
                                    try:
                                        await cp.mark_url_failed(ordered_urls[i], f"local-llm-stream-item-timeout-{PER_ITEM_TIMEOUT}s")
                                    except Exception:
                                        pass
                                i += 1
                                continue
                            except asyncio.CancelledError:
                                logger.warning("[%s] LLM stream iteration cancelled; stopping early", bvdid)
                                break
                            except Exception as e:
                                logger.exception("[%s] Exception while iterating LLM stream: %s", bvdid, e)
                                if i < len(ordered_urls):
                                    try:
                                        await cp.mark_url_failed(ordered_urls[i], f"local-llm-stream-iteration-exc: {e}")
                                    except Exception:
                                        pass
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

    try:
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
    except Exception:
        pass

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
    root_logger.info("LLM config: presence_only=%s require_presence=%s", args.presence_only, getattr(args, "require_presence", False))

    if getattr(args, "csv_dir", None):
        csv_dir: Path = args.csv_dir
        if not csv_dir.exists() or not csv_dir.is_dir():
            root_logger.error("--csv-dir is not a directory: %s", csv_dir)
            log_ext.close()
            return
        files = sorted(p for p in csv_dir.rglob("*.csv") if p.is_file())
        if not files:
            root_logger.error("No CSV files found under: %s", csv_dir)
            log_ext.close()
            return
        root_logger.info("Discovered %d CSV file(s) in %s", len(files), csv_dir)
        companies: List[CompanyInput] = []
        for f in files:
            companies.extend(load_companies_from_csv(f, limit=None))
        if args.limit is not None and args.limit > 0:
            companies = companies[:args.limit]
            root_logger.info("Applied global limit: %d", args.limit)
    else:
        companies = load_companies_from_csv(args.csv, limit=args.limit)

    if not companies:
        root_logger.error("No valid rows in CSV input. Exiting.")
        log_ext.close()
        return

    # Use language-aware defaults (load_lang has been called above)
    include_patterns = get_default_include_patterns() + ([p.strip() for p in args.include.split(",") if p.strip()] if args.include else [])
    exclude_patterns = get_default_exclude_patterns() + ([p.strip() for p in args.exclude.split(",") if p.strip()] if args.exclude else [])
    lang_regions = set(x.strip().lower() for x in args.lang_accept_en_regions.split(",") if x.strip())

    n = len(companies)
    per_company = per_company_slots(n, args.max_slots)
    root_logger.info("Loaded %d companies | global slots=%d → per-company=%d", n, args.max_slots, per_company)
    root_logger.debug("Include patterns: %s", include_patterns)
    root_logger.debug("Exclude patterns: %s", exclude_patterns)
    metrics.set("session.company_count", float(n))
    metrics.set("session.per_company_slots", float(per_company))

    cp_mgr = CheckpointManager()
    await cp_mgr.mark_global_start()
    scanner = ArtifactScanner()
    state = GlobalState()

    net = ConnectivityGuard(
        interval_s=5.0,
        trip_heartbeats=2,
        error_ratio_threshold=0.6,
        error_window=20,
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
                                max_concurrency_for_company=per_company,
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
                                scanner=scanner,
                                state=state,
                                net=net,
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