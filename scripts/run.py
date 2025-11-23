from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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
    default_bestfirst_factory,
    default_bfs_internal_factory,
    DFSDeepCrawlStrategyProvider,
    DeepCrawlStrategyFactory,
)
from configs.browser import default_browser_factory
from configs.llm import (
    LLMExtractionFactory,
    RemoteAPIProviderStrategy,
    default_ollama_provider_strategy,
    parse_extracted_payload,
    parse_presence_result,
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

# Resource monitoring
from extensions.resource_monitor import ResourceMonitor, ResourceMonitorConfig

# Stall management
from extensions.stall_guard import StallGuard, StallGuardConfig

logger = logging.getLogger("deep_crawl_runner")

_HTTP_LANG_MAP: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Company:
    company_id: str
    domain_url: str


# ---------------------------------------------------------------------------
# Company loading
# ---------------------------------------------------------------------------


def _companies_from_source(path: Path) -> List[Company]:
    inputs: List[CompanyInput] = load_companies_from_source(path)
    companies: List[Company] = [
        Company(company_id=ci.bvdid, domain_url=ci.url) for ci in inputs
    ]

    logger.info("Loaded %d companies from source %s", len(companies), path)
    return companies


# ---------------------------------------------------------------------------
# URL/meta helpers (crawl_meta.json only; url_index via crawl_state)
# ---------------------------------------------------------------------------


def _company_metadata_dir(company_id: str) -> Path:
    dirs = ensure_company_dirs(company_id)
    meta_dir = dirs.get("metadata") or dirs.get("checkpoints")
    if meta_dir is None:
        meta_dir = Path(output_paths.OUTPUT_ROOT) / company_id / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def _crawl_meta_path(company_id: str) -> Path:
    return _company_metadata_dir(company_id) / "crawl_meta.json"


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
        # no pretty-printing → smaller, faster to write
        json.dump(payload, f, ensure_ascii=False)
        try:
            f.flush()
        except Exception:
            pass
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------


async def process_page_result(
    page_result: Any,
    *,
    company: Company,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    stall_guard: Optional[StallGuard] = None,
) -> None:
    """
    Process a single page_result coming from deep crawl.

    JS injection is configured at the crawler config level.
    """
    if guard is not None:
        await guard.wait_until_healthy()

    # Cache getattr locally to reduce attribute lookup overhead in tight loops
    _getattr = getattr

    requested_url = _getattr(page_result, "url", None)
    final_url = _getattr(page_result, "final_url", None) or requested_url

    if not final_url:
        logger.warning("Page result missing URL; skipping entry")
        return

    url = final_url

    markdown = _getattr(page_result, "markdown", None)
    status_code = _getattr(page_result, "status_code", None)
    error = _getattr(page_result, "error", None) or _getattr(
        page_result, "error_message", None
    )

    html = _getattr(page_result, "html", None)
    if html is None:
        html = _getattr(page_result, "final_html", None)

    # --- Gating decision (save / suppress only) ----------------------------
    action, reason, stats = md_gating.evaluate_markdown(
        markdown or "",
        min_meaningful_words=gating_cfg.min_meaningful_words,
        cookie_max_fraction=gating_cfg.cookie_max_fraction,
        require_structure=gating_cfg.require_structure,
    )

    # --- Save HTML ---------------------------------------------------------
    html_path: Optional[str] = None
    if html:
        try:
            p_html = save_stage_output(
                bvdid=company.company_id,
                url=url,
                data=html,
                stage="html",
            )
            if p_html is not None:
                html_path = str(p_html)
        except Exception as e:
            logger.error(
                "Failed to write HTML for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )

    # --- Connectivity guard accounting ------------------------------------
    if guard is not None:
        try:
            code_int = int(status_code) if status_code is not None else None
        except Exception:
            code_int = None

        if error or (code_int is not None and code_int >= 500):
            guard.record_transport_error()
        else:
            guard.record_success()

    # --- Final markdown save / suppression --------------------------------
    gating_accept = action == "save"
    md_path: Optional[str] = None
    md_status: Optional[str] = None

    if gating_accept and markdown:
        try:
            p = save_stage_output(
                bvdid=company.company_id,
                url=url,
                data=markdown,
                stage="markdown",
            )
            if p is not None:
                md_path = str(p)
            md_status = "markdown_saved"
        except Exception as e:
            logger.error(
                "Failed to write markdown for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
    else:
        md_status = "markdown_suppressed"

    entry: Dict[str, Any] = {
        "url": url,
        "requested_url": requested_url,
        "status_code": status_code,
        "error": error,
        "depth": _getattr(page_result, "depth", None),
        "presence": 0,
        "extracted": 0,
        "gating_accept": gating_accept,
        "gating_action": action,
        "gating_reason": reason,
        "md_total_words": stats.get("total_words"),
    }

    if md_status is not None:
        entry["status"] = md_status
    if md_path is not None:
        entry["markdown_path"] = md_path
    if html_path is not None:
        entry["html_path"] = html_path

    upsert_url_index_entry(company.company_id, url, entry)

    # Stall guard progress heartbeat per successfully processed page
    if stall_guard is not None:
        stall_guard.record_page(company.company_id, url)


# ---------------------------------------------------------------------------
# LLM presence second pass
# ---------------------------------------------------------------------------


async def run_presence_pass_for_company(
    company: Company,
    *,
    presence_strategy: Any,
    stall_guard: Optional[StallGuard] = None,
) -> None:
    state = get_crawl_state()
    pending_urls = await state.get_pending_urls_for_llm(company.company_id)
    if not pending_urls:
        logger.info(
            "LLM presence: no pending URLs for company_id=%s", company.company_id
        )
        await state.recompute_company_from_index(
            company.company_id,
            name=None,
            root_url=company.domain_url,
        )
        return

    logger.info(
        "LLM presence: %d pending URLs for company_id=%s",
        len(pending_urls),
        company.company_id,
    )

    index = load_url_index(company.company_id)
    if not isinstance(index, dict):
        index = {}

    updated = 0

    for url in pending_urls:
        ent = index.get(url) or {}
        md_path = ent.get("markdown_path")

        if not md_path:
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_extracted_empty",
                    "llm_presence_reason": "no_markdown",
                },
            )
            updated += 1
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_presence")
            continue

        try:
            text = Path(md_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.error(
                "LLM presence: failed reading markdown for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_extracted_empty",
                    "llm_presence_reason": "markdown_read_error",
                },
            )
            updated += 1
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_presence")
            continue

        if not text.strip():
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_extracted_empty",
                    "llm_presence_reason": "empty_markdown",
                },
            )
            updated += 1
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_presence")
            continue

        try:
            raw_result = await asyncio.to_thread(
                presence_strategy.extract,
                url,
                text,
            )
        except Exception as e:
            logger.exception(
                "LLM presence: error for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            await state.mark_url_failed(
                company.company_id,
                url,
                f"presence_error:{type(e).__name__}",
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_extracted_empty",
                    "llm_presence_reason": "presence_exception",
                },
            )
            updated += 1
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_presence")
            continue

        has_offering, confidence, preview = parse_presence_result(
            raw_result, default=False
        )

        patch: Dict[str, Any] = {
            "presence": 1 if has_offering else 0,
            "presence_checked": True,
            "status": "llm_extracted" if has_offering else "llm_extracted_empty",
        }
        if confidence is not None:
            patch["llm_presence_confidence"] = confidence
        if preview is not None:
            patch["llm_presence_preview"] = preview

        upsert_url_index_entry(company.company_id, url, patch)
        updated += 1

        if stall_guard is not None:
            stall_guard.record_heartbeat("llm_presence")

    logger.info(
        "LLM presence: updated %d URLs for company_id=%s",
        updated,
        company.company_id,
    )

    await state.recompute_company_from_index(
        company.company_id,
        name=None,
        root_url=company.domain_url,
    )


# ---------------------------------------------------------------------------
# LLM full extraction second pass
# ---------------------------------------------------------------------------


async def run_full_pass_for_company(
    company: Company,
    *,
    full_strategy: Any,
    stall_guard: Optional[StallGuard] = None,
) -> None:
    index = load_url_index(company.company_id)
    if not isinstance(index, dict) or not index:
        logger.info(
            "LLM full: no url_index entries for company_id=%s", company.company_id
        )
        return

    updated = 0

    for url, ent in index.items():
        if not isinstance(ent, dict):
            continue

        md_path = ent.get("markdown_path")
        if not md_path or ent.get("extracted"):
            # Already processed or no markdown; skip without heartbeat
            continue

        try:
            text = Path(md_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.error(
                "LLM full: failed reading markdown for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "extracted": 0,
                    "status": "llm_full_markdown_read_error",
                },
            )
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_full")
            continue

        if not text.strip():
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "extracted": 0,
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_full_empty_markdown",
                },
            )
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_full")
            continue

        try:
            raw_result = await asyncio.to_thread(
                full_strategy.extract,
                url,
                text,
            )
        except Exception as e:
            logger.exception(
                "LLM full: error for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "extracted": 0,
                    "status": f"llm_full_error:{type(e).__name__}",
                },
            )
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_full")
            continue

        payload = parse_extracted_payload(raw_result)
        payload_dict = payload.model_dump()

        try:
            product_path = save_stage_output(
                bvdid=company.company_id,
                url=url,
                data=json.dumps(payload_dict, ensure_ascii=False),
                stage="product",
            )
        except Exception as e:
            logger.error(
                "LLM full: failed writing product JSON for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "extracted": 0,
                    "status": "llm_full_write_error",
                },
            )
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_full")
            continue

        presence_flag = 1 if payload.offerings else 0

        patch: Dict[str, Any] = {
            "extracted": 1,
            "presence": presence_flag,
            "presence_checked": True,
            "status": "llm_full_extracted",
        }
        if product_path is not None:
            patch["product_path"] = str(product_path)

        upsert_url_index_entry(company.company_id, url, patch)
        updated += 1

        if stall_guard is not None:
            stall_guard.record_heartbeat("llm_full")

    logger.info(
        "LLM full: wrote product JSON for %d URLs (company_id=%s)",
        updated,
        company.company_id,
    )

    state = get_crawl_state()
    await state.recompute_company_from_index(
        company.company_id,
        name=None,
        root_url=company.domain_url,
    )


# ---------------------------------------------------------------------------
# Company-level crawl
# ---------------------------------------------------------------------------


async def crawl_company(
    company: Company,
    *,
    crawler: AsyncWebCrawler,
    deep_strategy: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    stall_guard: Optional[StallGuard] = None,
    root_urls: Optional[List[str]] = None,
    crawler_base_cfg: Any = None,
    page_policy: Optional[PageInteractionPolicy] = None,
    page_interaction_factory: Optional[PageInteractionFactory] = None,
    max_pages: Optional[int] = None,
) -> None:
    logger.info(
        "Starting crawl for company_id=%s url=%s",
        company.company_id,
        company.domain_url,
    )

    _company_metadata_dir(company.company_id)

    start_urls: List[str] = list(root_urls) if root_urls else [company.domain_url]

    # Per-company page counter
    pages_processed = 0

    for start_url in start_urls:
        # If we already hit the limit earlier (e.g. on another root URL), stop.
        if max_pages is not None and pages_processed >= max_pages:
            logger.info(
                "Per-company max_pages limit (%d) already reached for company_id=%s, "
                "skipping remaining roots",
                max_pages,
                company.company_id,
            )
            break

        logger.info(
            "Deep crawl: company_id=%s root=%s pending_roots=%d",
            company.company_id,
            start_url,
            len(start_urls),
        )

        if crawler_base_cfg is not None:
            # Use repo crawler config, + JS interaction layer when available.
            clone_kwargs: Dict[str, Any] = {
                "cache_mode": CacheMode.BYPASS,
                "remove_overlay_elements": True,
                "deep_crawl_strategy": deep_strategy,
            }

            if page_policy is not None and page_interaction_factory is not None:
                interaction_cfg = page_interaction_factory.base_config(
                    url=start_url,
                    policy=page_policy,
                    js_only=False,
                )

                # Combine JS snippets into a single code string.
                js_code = (
                    "\n\n".join(interaction_cfg.js_code)
                    if interaction_cfg.js_code
                    else ""
                )

                clone_kwargs.update(
                    js_code=js_code,
                    js_only=interaction_cfg.js_only,
                    wait_for=interaction_cfg.wait_for,
                    # "delay_before_return_html": interaction_cfg.delay_before_return_sec,
                )

            config = crawler_base_cfg.clone(**clone_kwargs)
        else:
            # Fallback (should not normally happen)
            from crawl4ai import CrawlerRunConfig

            config = CrawlerRunConfig(
                deep_crawl_strategy=deep_strategy,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                stream=True,
                verbose=False,
            )

        results_or_gen = await crawler.arun(
            start_url,
            config=config,
        )

        # ------------------------------------------------------------------ #
        # Streaming mode: manually drive async generator to honour max_pages
        # without triggering unhandled async-generator close exceptions.
        # ------------------------------------------------------------------ #
        if not isinstance(results_or_gen, list):
            agen = results_or_gen

            # Normalise to the underlying async iterator
            if hasattr(agen, "__aiter__"):
                agen = agen.__aiter__()

            while True:
                # Check limit *before* pulling the next page
                if max_pages is not None and pages_processed >= max_pages:
                    logger.info(
                        "Per-company max_pages limit (%d) reached for company_id=%s, "
                        "stopping crawl for this company",
                        max_pages,
                        company.company_id,
                    )
                    # Explicitly close the async generator to avoid the
                    # "Task exception was never retrieved" caused by the
                    # ContextVar reset in Crawl4AI's result_wrapper.
                    aclose = getattr(agen, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception as e:
                            # Swallow errors from aclose to keep logs clean;
                            # the crawl is already logically complete.
                            logger.debug(
                                "Error while closing deep crawl generator for company=%s: %s",
                                company.company_id,
                                e,
                            )
                    break

                try:
                    page_result = await agen.__anext__()
                except StopAsyncIteration:
                    # Normal end of stream
                    break
                except Exception as e:
                    logger.exception(
                        "Error fetching next deep crawl result (company=%s): %s",
                        company.company_id,
                        e,
                    )
                    # Try to close the generator and then bail out
                    aclose = getattr(agen, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception:
                            pass
                    break

                # Process page
                try:
                    await process_page_result(
                        page_result=page_result,
                        company=company,
                        guard=guard,
                        gating_cfg=gating_cfg,
                        stall_guard=stall_guard,
                    )
                    pages_processed += 1
                except Exception as e:
                    url = getattr(page_result, "url", None)
                    logger.exception(
                        "Error processing page %s (company=%s): %s",
                        url,
                        company.company_id,
                        e,
                    )

        # ------------------------------------------------------------------ #
        # Non-stream mode: same as before, just respect max_pages.
        # ------------------------------------------------------------------ #
        else:
            for page_result in results_or_gen:
                try:
                    await process_page_result(
                        page_result=page_result,
                        company=company,
                        guard=guard,
                        gating_cfg=gating_cfg,
                        stall_guard=stall_guard,
                    )
                    pages_processed += 1
                except Exception as e:
                    url = getattr(page_result, "url", None)
                    logger.exception(
                        "Error processing page %s (company=%s): %s",
                        url,
                        company.company_id,
                        e,
                    )

                if max_pages is not None and pages_processed >= max_pages:
                    logger.info(
                        "Per-company max_pages limit (%d) reached for company_id=%s, "
                        "stopping crawl for this company",
                        max_pages,
                        company.company_id,
                    )
                    break


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------


def build_dual_bm25_components() -> Dict[str, Any]:
    """
    Use language-specific BM25 tokens from configs.language.
    """
    product_tokens: List[str] = (
        default_language_factory.get("PRODUCT_TOKENS", []) or []
    )
    exclude_tokens: List[str] = (
        default_language_factory.get("EXCLUDE_TOKENS", []) or []
    )

    positive_terms = set(product_tokens)
    negative_terms = set(exclude_tokens)

    positive_query = " ".join(sorted(positive_terms))
    negative_query = " ".join(sorted(negative_terms))

    scorer_cfg = DualBM25Config(
        threshold=None,
        alpha=0.7,
        k1=1.2,
        b=0.75,
        avgdl=1000,
    )

    filter_cfg = DualBM25Config(
        threshold=0.5,
        alpha=0.5,
        k1=1.2,
        b=0.75,
        avgdl=1000,
    )

    url_scorer = DualBM25Scorer(
        positive_query=positive_query,
        negative_query=negative_query,
        cfg=scorer_cfg,
        doc_index=None,
        weight=1.0,
    )

    url_filter = DualBM25Filter(
        positive_query=positive_query,
        negative_query=negative_query,
        cfg=filter_cfg,
        name="DualBM25Filter",
    )

    return {
        "positive_query": positive_query,
        "negative_query": negative_query,
        "scorer_cfg": scorer_cfg,
        "filter_cfg": filter_cfg,
        "url_scorer": url_scorer,
        "url_filter": url_filter,
    }


# ---------------------------------------------------------------------------
# Dataset externals helper
# ---------------------------------------------------------------------------


def _build_dataset_externals(
    companies: List[Company],
    dataset_file: Optional[str],
) -> List[str]:
    dataset_hosts: set[str] = set()

    def _add_host(raw_url: str) -> None:
        try:
            host = urlparse(raw_url).hostname or ""
        except Exception:
            host = ""
        if not host:
            return
        dataset_hosts.add(host)
        if host.startswith("www."):
            dataset_hosts.add(host[4:])
        else:
            dataset_hosts.add(f"www.{host}")

    for c in companies:
        _add_host(c.domain_url)

    if dataset_file:
        try:
            ds_inputs: List[CompanyInput] = load_companies_from_source(Path(dataset_file))
            for ci in ds_inputs:
                _add_host(ci.url)
        except Exception as e:
            logger.exception("Failed to load dataset-file %s: %s", dataset_file, e)

    return sorted(dataset_hosts)


# ---------------------------------------------------------------------------
# Company pipeline runner (per-company, concurrent)
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
    crawler: AsyncWebCrawler,
    args: argparse.Namespace,
    dataset_externals: List[str],
    url_scorer: Optional[DualBM25Scorer],
    bm25_filter: Optional[DualBM25Filter],
    sem: asyncio.Semaphore,
    run_id: Optional[str],
    presence_llm: Any,
    full_llm: Any,
    stall_guard: Optional[StallGuard],
    crawler_base_cfg: Any = None,
    page_policy: Optional[PageInteractionPolicy] = None,
    page_interaction_factory: Optional[PageInteractionFactory] = None,
) -> None:
    async with sem:
        token = logging_ext.set_company_context(company.company_id)
        company_logger = logging_ext.get_company_logger(company.company_id)
        company_logger.info(
            "=== [%d/%d] Company company_id=%s url=%s ===",
            idx,
            total,
            company.company_id,
            company.domain_url,
        )

        if stall_guard is not None:
            stall_guard.record_heartbeat("company_start")

        # Only mark as completed if the pipeline runs to the end of the try-block.
        completed_ok = False

        snap = await state.get_company_snapshot(company.company_id)
        status = snap.status or COMPANY_STATUS_PENDING

        do_crawl = False
        resume_roots: Optional[List[str]] = None

        if status == COMPANY_STATUS_PENDING:
            company_logger.info(
                "State=PENDING → fresh crawl for %s", company.company_id
            )
            do_crawl = True

        elif status == COMPANY_STATUS_MD_NOT_DONE:
            pending_md = await state.get_pending_urls_for_markdown(
                company.company_id
            )
            if pending_md:
                company_logger.info(
                    "State=MARKDOWN_NOT_DONE → resuming crawl for %s from %d pending URLs",
                    company.company_id,
                    len(pending_md),
                )
                do_crawl = True
                resume_roots = pending_md
            else:
                company_logger.info(
                    "State=MARKDOWN_NOT_DONE but no pending URLs for %s; treating as no-crawl",
                    company.company_id,
                )
                do_crawl = False

        elif status in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            COMPANY_STATUS_LLM_NOT_DONE,
        ):
            company_logger.info(
                "State=%s → skipping crawl for %s",
                status,
                company.company_id,
            )
            do_crawl = False

        else:
            company_logger.info(
                "State=%s (unknown) → treating as PENDING for %s",
                status,
                company.company_id,
            )
            do_crawl = True

        if not do_crawl and args.llm_mode == "none":
            company_logger.info(
                "No crawl and llm_mode=none → skipping company %s",
                company.company_id,
            )
            logging_ext.reset_company_context(token)
            logging_ext.close_company(company.company_id)
            return

        # Build per-company filter & deep strategy
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

        filter_chain = FilterChain(filters)

        max_pages_per_company: Optional[int] = None
        if getattr(args, "max_pages", None) and args.max_pages > 0:
            max_pages_per_company = args.max_pages

        # Use deep crawl strategies from configs.deep_crawl
        if args.strategy == "bestfirst":
            deep_strategy = default_bestfirst_factory.create(
                filter_chain=filter_chain,
                url_scorer=url_scorer,
            )
        elif args.strategy == "bfs_internal":
            deep_strategy = default_bfs_internal_factory.create(
                filter_chain=filter_chain,
                url_scorer=None,
            )
        else:
            dfs_factory = DeepCrawlStrategyFactory(
                provider=DFSDeepCrawlStrategyProvider(
                    default_max_depth=3,
                    default_include_external=False,
                    default_score_threshold=None,
                )
            )
            deep_strategy = dfs_factory.create(
                filter_chain=filter_chain,
                url_scorer=None,
            )

        try:
            if do_crawl:
                await guard.wait_until_healthy()
                await crawl_company(
                    company=company,
                    crawler=crawler,
                    deep_strategy=deep_strategy,
                    guard=guard,
                    gating_cfg=gating_cfg,
                    stall_guard=stall_guard,
                    root_urls=resume_roots,
                    crawler_base_cfg=crawler_base_cfg,
                    page_policy=page_policy,
                    page_interaction_factory=page_interaction_factory,
                    max_pages=max_pages_per_company,
                )

                await state.recompute_company_from_index(
                    company.company_id,
                    name=None,
                    root_url=company.domain_url,
                )

            # LLM phases
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

            # If we reached here without cancellation, treat this company as completed
            completed_ok = True

        except asyncio.CancelledError:
            company_logger.warning(
                "Company pipeline cancelled for company_id=%s",
                company.company_id,
            )
            # Do NOT mark as completed; propagate cancellation so the run can stop.
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
                # Only increment run-level completed count when the pipeline truly finished.
                if run_id is not None and completed_ok:
                    await state.mark_company_completed(run_id, company.company_id)
                    await state.recompute_global_state()
            except Exception:
                logger.exception(
                    "Failed to update run/global state for company_id=%s",
                    company.company_id,
                )

            if stall_guard is not None and completed_ok:
                # Only treat as "completed" for stall detection when the pipeline finished
                stall_guard.record_company_completed(company.company_id)

            logging_ext.reset_company_context(token)
            logging_ext.close_company(company.company_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep crawl corporate websites (per-company pipeline)."
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
            "Path to input file OR directory with company data. "
            "Supports CSV/TSV/Excel/JSON/Parquet/etc via extensions.load_source."
        ),
    )

    parser.add_argument(
        "--company-id",
        type=str,
        default=None,
        help="Optional ID for single-company mode; defaults to hostname.",
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
            "'presence' = presence-only classification (0/1). "
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
        help="Provider string when --llm-provider=api, e.g. 'openai/gpt-4o-mini'.",
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
        default=1,
        help=(
            "Maximum number of companies to process concurrently. "
            "Default: 1 (sequential)."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help=(
            "Maximum number of pages to crawl per company. "
            "Enforced in run.py as a per-company limit."
        ),
    )
    parser.add_argument(
        "--enable-resource-monitor",
        action="store_true",
        help=(
            "Enable lightweight resource monitoring (CPU/RAM/network/IO) for the "
            "entire run and write outputs/resource_usage.json."
        ),
    )
    parser.add_argument(
        "--resource-monitor-interval",
        type=float,
        default=2.0,
        help="Sampling interval in seconds for the resource monitor (default: 2.0).",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths.OUTPUT_ROOT = out_dir  # type: ignore[attr-defined]

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    logging_ext = LoggingExtension(
        global_level=log_level,
        per_company_level=log_level,
        max_open_company_logs=128,
        enable_session_log=True,
        session_log_path=out_dir / "session.log",
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.setLevel(log_level)

    # Resource monitor (optional, can be disabled to avoid overhead)
    resource_monitor: Optional[ResourceMonitor] = None
    if getattr(args, "enable_resource_monitor", False):
        rm_config = ResourceMonitorConfig(
            interval_sec=float(getattr(args, "resource_monitor_interval", 2.0))
        )
        resource_monitor = ResourceMonitor(
            output_path=out_dir / "resource_usage.json",
            config=rm_config,
        )
        resource_monitor.start()

    # Language-specific config for BM25 vocab
    default_language_factory.set_language(args.lang)

    # LLM factory (optional)
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

    # Markdown gating config (extensions/md_gating)
    gating_cfg = md_gating.build_gating_config(
        min_meaningful_words=30,
        cookie_max_fraction=0.02,
        require_structure=True,
        interstitial_max_share=0.70,
        interstitial_min_hits=2,
    )

    # Crawler base config (JS-related options will be injected via clone())
    markdown_generator = default_md_factory.create(
        min_meaningful_words=gating_cfg.min_meaningful_words,
        interstitial_max_share=gating_cfg.interstitial_max_share,
        interstitial_min_hits=gating_cfg.interstitial_min_hits,
        cookie_max_fraction=gating_cfg.cookie_max_fraction,
        require_structure=gating_cfg.require_structure,
    )

    crawler_base_cfg = default_crawler_factory.create(
        markdown_generator=markdown_generator,
    )

    # Page interaction / JS injection policy (safe defaults)
    page_policy = PageInteractionPolicy(
        enable_cookie_playbook=True,
        cookie_retry_attempts=4,
        cookie_retry_interval_ms=400,
        enable_anti_interstitial=False,
        wait_timeout_ms=60000,
        delay_before_return_sec=1.2,
        min_content_chars=800,
        max_cookie_hits=3,
        virtual_scroll=False,
    )
    page_interaction_factory = default_page_interaction_factory

    # Stall guard: detect global stalls relative to per-page timeout
    stall_cfg = StallGuardConfig(
        page_timeout_sec=page_policy.wait_timeout_ms / 1000.0,
        soft_timeout_factor=2.0,   # soft stall ≥ 2 × page timeout
        hard_timeout_factor=4.0,   # hard stall ≥ 4 × page timeout
        check_interval_sec=15.0,
        min_pages_before_detection=20,
        min_companies_before_detection=0,
        min_consecutive_stall_checks=2,
        # Internal auto-kill is disabled; we will decide in a callback
        # based on ConnectivityGuard state.
        auto_kill_process=False,
        auto_kill_exit_code=3,
        dump_state_path=out_dir / "stall_state.json",
    )
    stall_guard = StallGuard(config=stall_cfg)

    # If StallGuard supports an on_stall hook, gate killing on connectivity state.
    async def _handle_stall(snapshot: Any) -> None:
        # When connectivity is healthy, treat stall as a real stall and exit.
        if guard.is_healthy():
            logger.error(
                "StallGuard detected a hard stall with healthy connectivity; "
                "terminating process with exit code %d",
                stall_cfg.auto_kill_exit_code,
            )
            # Best-effort flush of root logger handlers so logs aren't lost.
            root_logger = logging.getLogger()
            for h in list(root_logger.handlers):
                try:
                    h.flush()
                except Exception:
                    pass
            os._exit(stall_cfg.auto_kill_exit_code)
        else:
            # ConnectivityGuard says we're in outage / not healthy → do NOT kill.
            logger.warning(
                "StallGuard detected a stall but ConnectivityGuard is %s; "
                "suppressing auto-kill.",
                guard.state(),
            )

    # Only attach if StallGuard exposes this attribute (for safety with older versions).
    if hasattr(stall_guard, "on_stall"):
        # type: ignore[attr-defined] to satisfy type checkers if you use them
        stall_guard.on_stall = _handle_stall  # type: ignore[attr-defined]
    else:
        logger.warning(
            "StallGuard has no 'on_stall' hook; auto_kill_process is disabled "
            "to avoid killing the run during connectivity outages."
        )

    await stall_guard.start()


    state = get_crawl_state()

    try:
        # Build company list
        if args.company_file:
            companies: List[Company] = _companies_from_source(Path(args.company_file))
        else:
            url = args.url
            assert url is not None
            company_id = args.company_id
            if not company_id:
                parsed = urlparse(url)
                company_id = (parsed.netloc or parsed.path or "company").replace(
                    ":", "_"
                )
            companies = [Company(company_id=company_id, domain_url=url)]

        total = len(companies)

        # Initialize run row; DO NOT reset per-company status here.
        # This allows resume/skip logic to work across multiple runs:
        # - Companies that reached max_pages and were marked markdown_done
        #   will stay done and be skipped on the next run.
        run_id = await state.start_run(
            pipeline="deep_crawl",
            version=None,
            args_hash=None,
        )
        await state.update_run_totals(
            run_id,
            total_companies=total,
        )

        for c in companies:
            # Ensure the company exists in the DB and its root_url is set.
            # We intentionally do NOT touch the status, so previous completion
            # state derived from url_index.json is preserved.
            await state.upsert_company(
                c.company_id,
                name=None,
                root_url=c.domain_url,
            )

        try:
            await state.recompute_global_state()
        except Exception:
            logger.exception("Failed to recompute global crawl state at startup")

        # Build dataset_externals host whitelist
        dataset_externals: List[str] = _build_dataset_externals(
            companies=companies,
            dataset_file=args.dataset_file,
        )

        # BM25 scorer + filter
        bm25_components = build_dual_bm25_components()
        url_scorer: Optional[DualBM25Scorer] = bm25_components["url_scorer"]
        bm25_filter: Optional[DualBM25Filter] = bm25_components["url_filter"]

        max_companies = max(1, int(args.company_concurrency))

        # Browser config (no dispatcher)
        http_lang = _HTTP_LANG_MAP.get(args.lang, f"{args.lang}-US")

        browser_cfg = default_browser_factory.create(
            lang=http_lang,
            add_common_cookies_for=[],
            headless=True,
        )

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            sem = asyncio.Semaphore(max_companies)

            batch_size = max_companies * 4
            if batch_size < max_companies:
                batch_size = max_companies

            total = len(companies)

            for batch_start in range(0, total, batch_size):
                batch = companies[batch_start : batch_start + batch_size]

                tasks: List[asyncio.Task] = []

                for offset, company in enumerate(batch, start=batch_start + 1):
                    task = asyncio.create_task(
                        run_company_pipeline(
                            company=company,
                            idx=offset,
                            total=total,
                            logging_ext=logging_ext,
                            state=state,
                            guard=guard,
                            gating_cfg=gating_cfg,
                            crawler=crawler,
                            args=args,
                            dataset_externals=dataset_externals,
                            url_scorer=url_scorer,
                            bm25_filter=bm25_filter,
                            sem=sem,
                            run_id=run_id,
                            presence_llm=presence_llm,
                            full_llm=full_llm,
                            stall_guard=stall_guard,
                            crawler_base_cfg=crawler_base_cfg,
                            page_policy=page_policy,
                            page_interaction_factory=page_interaction_factory,
                        )
                    )
                    tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

    finally:
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


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()