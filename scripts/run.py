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
from extensions.dual_bm25 import DualBM25Config, DualBM25Filter, DualBM25Scorer
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
    wait_for_global_hard_stall,
)
from extensions.memory_guard import MemoryGuard, CriticalMemoryPressure
from extensions.retry_tracker import RetryTracker, RetryTrackerConfig
from extensions.adaptive_concurrency import (
    AdaptiveConcurrencyConfig,
    AdaptiveConcurrencyController,
)

logger = logging.getLogger("deep_crawl_runner")

_HTTP_LANG_MAP: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
}

# Special exit code so outer wrapper knows there are companies to retry.
RETRY_EXIT_CODE = int(os.environ.get("DEEP_CRAWL_RETRY_EXIT_CODE", "17"))

_retry_tracker: Optional[RetryTracker] = None


def set_retry_tracker(tracker: RetryTracker) -> None:
    global _retry_tracker
    _retry_tracker = tracker


def mark_company_timeout(company_id: str) -> None:
    if _retry_tracker is not None:
        _retry_tracker.mark_timeout(company_id)


def mark_company_stalled(company_id: str) -> None:
    if _retry_tracker is not None:
        _retry_tracker.mark_stalled(company_id)


def mark_company_memory_pressure(company_id: str) -> None:
    if _retry_tracker is not None:
        _retry_tracker.mark_memory(company_id)


def record_company_attempt(company_id: str) -> None:
    if _retry_tracker is not None:
        _retry_tracker.record_attempt(company_id)


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
# Per-page processing
# ---------------------------------------------------------------------------


async def process_page_result(
    page_result: Any,
    *,
    company: Company,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    timeout_error_marker: str,
    stall_guard: Optional[StallGuard] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> None:
    if guard is not None:
        await guard.wait_until_healthy()

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
        page_result,
        "error_message",
        None,
    )

    if memory_guard is not None:
        memory_guard.check_page_error(
            error=error,
            company_id=company.company_id,
            url=url,
            mark_company_memory=mark_company_memory_pressure,
        )

    timeout_exceeded = isinstance(error, str) and timeout_error_marker in error
    if timeout_exceeded:
        mark_company_timeout(company.company_id)

    html = _getattr(page_result, "html", None) or _getattr(
        page_result, "final_html", None
    )

    action, reason, stats = md_gating.evaluate_markdown(
        markdown or "",
        min_meaningful_words=gating_cfg.min_meaningful_words,
        cookie_max_fraction=gating_cfg.cookie_max_fraction,
        require_structure=gating_cfg.require_structure,
    )

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

    if guard is not None:
        try:
            code_int = int(status_code) if status_code is not None else None
        except Exception:
            code_int = None

        if error or (code_int is not None and code_int >= 500):
            guard.record_transport_error()
        else:
            guard.record_success()

    gating_accept = action == "save"
    md_path: Optional[str] = None
    md_status: str

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
            md_status = "markdown_error"
    else:
        md_status = "markdown_suppressed"

    if timeout_exceeded:
        md_status = "timeout_page_exceeded"

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
        "status": md_status,
    }

    if timeout_exceeded:
        entry["timeout_page_exceeded"] = True
        entry["scheduled_retry"] = True

    if md_path is not None:
        entry["markdown_path"] = md_path
    if html_path is not None:
        entry["html_path"] = html_path

    upsert_url_index_entry(company.company_id, url, entry)

    if stall_guard is not None:
        stall_guard.record_heartbeat("page", company_id=company.company_id)


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

    index = load_url_index(company.company_id) or {}
    updated = 0

    def _update(url: str, patch: Dict[str, Any], reason: str) -> None:
        nonlocal updated
        patch_full = {
            "presence": 0,
            "presence_checked": True,
            "status": "llm_extracted_empty",
            "llm_presence_reason": reason,
        }
        patch_full.update(patch)
        upsert_url_index_entry(company.company_id, url, patch_full)
        updated += 1
        if stall_guard is not None:
            stall_guard.record_heartbeat("llm_presence", company_id=company.company_id)

    for url in pending_urls:
        ent = index.get(url) or {}
        md_path = ent.get("markdown_path")

        if not md_path:
            _update(url, {}, "no_markdown")
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
            _update(url, {}, "markdown_read_error")
            continue

        if not text.strip():
            _update(url, {}, "empty_markdown")
            continue

        try:
            raw_result = await asyncio.to_thread(presence_strategy.extract, url, text)
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
            _update(url, {}, "presence_exception")
            continue

        has_offering, confidence, preview = parse_presence_result(
            raw_result,
            default=False,
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
            stall_guard.record_heartbeat("llm_presence", company_id=company.company_id)

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
                {"extracted": 0, "status": "llm_full_markdown_read_error"},
            )
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_full", company_id=company.company_id)
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
                stall_guard.record_heartbeat("llm_full", company_id=company.company_id)
            continue

        try:
            raw_result = await asyncio.to_thread(full_strategy.extract, url, text)
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
                {"extracted": 0, "status": f"llm_full_error:{type(e).__name__}"},
            )
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_full", company_id=company.company_id)
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
                {"extracted": 0, "status": "llm_full_write_error"},
            )
            if stall_guard is not None:
                stall_guard.record_heartbeat("llm_full", company_id=company.company_id)
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
            stall_guard.record_heartbeat("llm_full", company_id=company.company_id)

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
    timeout_error_marker: str,
    stall_guard: Optional[StallGuard] = None,
    root_urls: Optional[List[str]] = None,
    crawler_base_cfg: Any = None,
    page_policy: Optional[PageInteractionPolicy] = None,
    page_interaction_factory: Optional[PageInteractionFactory] = None,
    max_pages: Optional[int] = None,
    page_timeout_ms: Optional[int] = None,
    memory_guard: Optional[MemoryGuard] = None,
) -> None:
    logger.info(
        "Starting crawl for company_id=%s url=%s",
        company.company_id,
        company.domain_url,
    )

    ensure_company_dirs(company.company_id)

    start_urls: List[str] = list(root_urls) if root_urls else [company.domain_url]
    pages_processed = 0

    for start_url in start_urls:
        if max_pages is not None and pages_processed >= max_pages:
            logger.info(
                "Per-company max_pages limit (%d) already reached for company_id=%s, skipping remaining roots",
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

                js_code = (
                    "\n\n".join(interaction_cfg.js_code)
                    if interaction_cfg.js_code
                    else ""
                )

                clone_kwargs.update(
                    js_code=js_code,
                    js_only=interaction_cfg.js_only,
                    wait_for=interaction_cfg.wait_for,
                )

            config = crawler_base_cfg.clone(**clone_kwargs)
        else:
            from crawl4ai import CrawlerRunConfig

            config_kwargs: Dict[str, Any] = dict(
                deep_crawl_strategy=deep_strategy,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                stream=True,
                verbose=False,
            )
            if page_timeout_ms is not None:
                config_kwargs["page_timeout"] = page_timeout_ms

            config = CrawlerRunConfig(**config_kwargs)

        results_or_gen = await crawler.arun(start_url, config=config)

        if not isinstance(results_or_gen, list):
            agen = results_or_gen
            if hasattr(agen, "__aiter__"):
                agen = agen.__aiter__()

            while True:
                if max_pages is not None and pages_processed >= max_pages:
                    logger.info(
                        "Per-company max_pages limit (%d) reached for company_id=%s, stopping crawl for this company",
                        max_pages,
                        company.company_id,
                    )
                    aclose = getattr(agen, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception as e:
                            logger.debug(
                                "Error while closing deep crawl generator for company=%s: %s",
                                company.company_id,
                                e,
                            )
                    break

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
                    )
                    pages_processed += 1
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

        else:
            for page_result in results_or_gen:
                try:
                    await process_page_result(
                        page_result=page_result,
                        company=company,
                        guard=guard,
                        gating_cfg=gating_cfg,
                        timeout_error_marker=timeout_error_marker,
                        stall_guard=stall_guard,
                        memory_guard=memory_guard,
                    )
                    pages_processed += 1
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

                if max_pages is not None and pages_processed >= max_pages:
                    logger.info(
                        "Per-company max_pages limit (%d) reached for company_id=%s, stopping crawl for this company",
                        max_pages,
                        company.company_id,
                    )
                    break


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------


def build_dual_bm25_components() -> Dict[str, Any]:
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
# Adaptive concurrency helpers
# ---------------------------------------------------------------------------


class ActiveCounter:
    def __init__(self) -> None:
        self._value: int = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self, limit: int) -> bool:
        async with self._lock:
            if self._value < limit:
                self._value += 1
                return True
            return False

    async def release(self) -> None:
        async with self._lock:
            if self._value > 0:
                self._value -= 1

    async def current(self) -> int:
        async with self._lock:
            return self._value


async def wait_for_adaptive_slot(
    company_id: str,
    ac_controller: AdaptiveConcurrencyController,
    active_counter: ActiveCounter,
    poll_interval: float = 0.5,
) -> None:
    while True:
        limit = await ac_controller.get_limit()
        if limit <= 0:
            await asyncio.sleep(poll_interval)
            continue

        acquired = await active_counter.try_acquire(limit)
        if acquired:
            logger.debug(
                "[AdaptiveConcurrency] company_id=%s acquired slot (limit=%d)",
                company_id,
                limit,
            )
            return

        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Small helpers
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


def _build_deep_strategy(
    args: argparse.Namespace,
    filter_chain: FilterChain,
    url_scorer: Optional[DualBM25Scorer],
    dfs_factory: Optional[DeepCrawlStrategyFactory],
) -> Any:
    if args.strategy == "bestfirst":
        return default_bestfirst_factory.create(
            filter_chain=filter_chain,
            url_scorer=url_scorer,
        )
    if args.strategy == "bfs_internal":
        return default_bfs_internal_factory.create(
            filter_chain=filter_chain,
            url_scorer=None,
        )
    assert dfs_factory is not None
    return dfs_factory.create(
        filter_chain=filter_chain,
        url_scorer=None,
    )


def _should_skip_company(status: str, llm_mode: str) -> bool:
    if llm_mode == "none":
        return status in (COMPANY_STATUS_MD_DONE, COMPANY_STATUS_LLM_DONE)
    return status == COMPANY_STATUS_LLM_DONE


def _session_ranges(n: int, per_session: int) -> List[tuple[int, int]]:
    if per_session <= 0:
        return [(0, n)]
    return [
        (start, min(start + per_session, n))
        for start in range(0, n, per_session)
    ]


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
    timeout_error_marker: str,
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
    memory_guard: Optional[MemoryGuard],
    ac_controller: AdaptiveConcurrencyController,
    active_counter: ActiveCounter,
    dfs_factory: Optional[DeepCrawlStrategyFactory],
    crawler_base_cfg: Any = None,
    page_policy: Optional[PageInteractionPolicy] = None,
    page_interaction_factory: Optional[PageInteractionFactory] = None,
    page_timeout_ms: Optional[int] = None,
) -> None:
    acquired_adaptive_slot = False
    completed_ok = False
    token: Optional[Any] = None

    try:
        async with sem:
            try:
                snap = await state.get_company_snapshot(company.company_id)
                status = snap.status or COMPANY_STATUS_PENDING
            except Exception as e:
                logger.exception(
                    "Pre-check: failed to get snapshot for company_id=%s: %s",
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
                pending_md = await state.get_pending_urls_for_markdown(
                    company.company_id
                )
                if pending_md:
                    do_crawl = True
                    resume_roots = pending_md
                    state_log_tpl = (
                        "State=MARKDOWN_NOT_DONE -> resuming crawl for %s from %d pending URLs"
                    )
                    state_log_args = (company.company_id, len(pending_md))
                else:
                    state_log_tpl = (
                        "State=MARKDOWN_NOT_DONE but no pending URLs for %s; treating as no-crawl"
                    )
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

            will_run_llm_presence = (
                args.llm_mode == "presence" and presence_llm is not None
            )
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

            await wait_for_adaptive_slot(
                company_id=company.company_id,
                ac_controller=ac_controller,
                active_counter=active_counter,
            )
            acquired_adaptive_slot = True
            await ac_controller.notify_work_started()

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

            max_pages_per_company: Optional[int] = None
            if getattr(args, "max_pages", None) and args.max_pages > 0:
                max_pages_per_company = args.max_pages

            deep_strategy = _build_deep_strategy(
                args=args,
                filter_chain=filter_chain,
                url_scorer=url_scorer,
                dfs_factory=dfs_factory,
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
                    max_pages=max_pages_per_company,
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
        if acquired_adaptive_slot:
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

            if stall_guard is not None and completed_ok:
                stall_guard.record_company_completed(company.company_id)

            if token is not None:
                logging_ext.reset_company_context(token)
                logging_ext.close_company(company.company_id)

            await active_counter.release()


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
            "Supports CSV or TSV or Excel or JSON or Parquet via extensions.load_source."
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
            "'presence' = presence-only classification (0 or 1). "
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
        default=2048,
        help=(
            "Maximum number of companies to process concurrently. "
            "Acts as a hard upper bound for adaptive concurrency."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help=(
            "Maximum number of pages to crawl per company. "
            "Enforced here as a per-company limit."
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
            "Per-page timeout in milliseconds for Playwright and Crawl4AI. "
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
            "Enable hard-stop behavior on critical host memory usage. "
            "When enabled, MemoryGuard will cancel running company tasks and "
            "cause the run to exit with the retry exit code when host memory "
            "crosses the configured hard limit. Default: disabled."
        ),
    )
    parser.add_argument(
        "--adaptive-disable-cpu",
        action="store_true",
        help=(
            "Disable CPU-based adaptive concurrency decisions. "
            "When set, only memory (if enabled) will influence scaling."
        ),
    )
    parser.add_argument(
        "--adaptive-disable-mem",
        action="store_true",
        help=(
            "Disable memory-based adaptive concurrency decisions. "
            "When set, only CPU (if enabled) will influence scaling. "
            "If both CPU and memory are disabled, concurrency will slowly "
            "rise to --company-concurrency and stay there (fixed mode)."
        ),
    )
    parser.add_argument(
        "--companies-per-session",
        type=int,
        default=0,
        help=(
            "Optional limit on how many companies to process per browser session. "
            "When > 0, the crawler will close and recreate the Playwright/Chromium "
            "stack after each block of this many companies to reduce long-term "
            "memory accumulation. Default: 0 (use a single browser session for "
            "the whole run)."
        ),
    )

    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths.OUTPUT_ROOT = out_dir  # type: ignore[attr-defined]

    retry_cfg = RetryTrackerConfig(
        out_dir=out_dir,
        flush_threshold=int(os.environ.get("RETRY_FLUSH_THRESHOLD", "1")),
    )
    retry_tracker = RetryTracker(retry_cfg)
    set_retry_tracker(retry_tracker)

    retry_company_mode: str = getattr(args, "retry_mode", "all")

    prev_retry_data: Dict[str, Any] = retry_tracker.load_existing()
    retry_list = prev_retry_data.get("retry_companies") or []
    prev_retry_ids: set[str] = set(str(x) for x in retry_list) if isinstance(
        retry_list, list
    ) else set()

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
        soft_timeout_factor=1.5,
        hard_timeout_factor=3.0,
        check_interval_sec=30.0,
    )
    stall_guard = StallGuard(config=stall_cfg)

    # Optional hard memory guard.
    memory_guard: Optional[MemoryGuard] = None
    if getattr(args, "enable_hard_memory_guard", False):
        memory_guard = MemoryGuard()
        logger.info("Hard memory guard enabled (process will abort on critical host memory).")
    else:
        logger.info(
            "Hard memory guard disabled (process will not auto-stop on critical host memory; "
            "rely on OS or container limits)."
        )

    company_tasks: Dict[str, asyncio.Task] = {}

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
            logger.warning(
                "StallGuard: stall detected for company_id=%s but no active task found",
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

    ac_controller: Optional[AdaptiveConcurrencyController] = None
    active_counter: Optional[ActiveCounter] = None
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

        # Pre-check statuses and skip already-completed companies.
        companies_to_run: List[Company] = []
        skipped_companies: List[Company] = []

        for c in companies:
            try:
                snap = await state.get_company_snapshot(c.company_id)
                status = getattr(snap, "status", None) or COMPANY_STATUS_PENDING
            except Exception as e:
                logger.exception(
                    "Pre-check: failed to get snapshot for company_id=%s: %s",
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
                "Pre-check: %d/%d companies already completed for llm_mode=%s; %d companies remain to process",
                len(skipped_companies),
                len(companies),
                args.llm_mode,
                len(companies_to_run),
            )
        else:
            logger.info(
                "Pre-check: all %d companies require processing for llm_mode=%s",
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

        dataset_externals: List[str] = _build_dataset_externals(
            companies=companies,
            dataset_file=args.dataset_file,
        )

        bm25_components = build_dual_bm25_components()
        url_scorer: Optional[DualBM25Scorer] = bm25_components["url_scorer"]
        bm25_filter: Optional[DualBM25Filter] = bm25_components["url_filter"]

        max_companies = max(1, int(args.company_concurrency))

        # Adaptive concurrency config
        ac_cfg = AdaptiveConcurrencyConfig(
            max_concurrency=max_companies,
            min_concurrency=1,
            target_mem_low=float(os.environ.get("AC_TARGET_MEM_LOW", "0.60")),
            target_mem_high=float(os.environ.get("AC_TARGET_MEM_HIGH", "0.70")),
            target_cpu_low=float(os.environ.get("AC_TARGET_CPU_LOW", "0.80")),
            target_cpu_high=float(os.environ.get("AC_TARGET_CPU_HIGH", "0.90")),
            sample_interval_sec=float(os.environ.get("AC_SAMPLE_INTERVAL", "1.0")),
            smoothing_window_sec=float(
                os.environ.get("AC_SMOOTHING_WINDOW_SEC", "10.0")
            ),
            log_path=(
                Path(os.environ["AC_LOG_PATH"]).resolve()
                if os.environ.get("AC_LOG_PATH")
                else None
            ),
            use_cpu=not getattr(args, "adaptive_disable_cpu", False),
            use_mem=not getattr(args, "adaptive_disable_mem", False),
        )
        ac_controller = AdaptiveConcurrencyController(cfg=ac_cfg)
        await ac_controller.start()
        active_counter = ActiveCounter()

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

        # Configure hard memory guard only if enabled.
        if memory_guard is not None:
            memory_guard.config.host_soft_limit = float(
                os.environ.get("HOST_MEM_SOFT_LIMIT", "0.98")
            )
            memory_guard.config.host_hard_limit = float(
                os.environ.get("HOST_MEM_HARD_LIMIT", "0.99")
            )
            memory_guard.config.check_interval_sec = float(
                os.environ.get("HOST_MEM_CHECK_INTERVAL", "2.0")
            )

            def _on_critical_memory(company_id: str) -> None:
                nonlocal abort_run
                abort_run = True
                logger.error(
                    "Host memory usage exceeded hard limit; cancelling all running company tasks (triggered by company_id=%s).",
                    company_id,
                )

                if ac_controller is not None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None:
                        loop.create_task(ac_controller.on_memory_pressure())

                for cid, task in list(company_tasks.items()):
                    if not task.done():
                        logger.error(
                            "MemoryGuard: cancelling company task company_id=%s",
                            cid,
                        )
                        mark_company_memory_pressure(cid)
                        task.cancel()

            memory_guard.config.on_critical = _on_critical_memory

        async def _global_stall_watchdog() -> None:
            nonlocal abort_run
            try:
                idle = await wait_for_global_hard_stall(
                    stall_guard,
                    page_timeout_sec=page_timeout_ms / 1000.0,
                    factor=float(os.environ.get("GLOBAL_STALL_FACTOR", "4.5")),
                    check_interval_sec=stall_cfg.check_interval_sec,
                )
            except asyncio.CancelledError:
                return

            abort_run = True

            logger.error(
                "Global StallGuard: no company-level progress for %.1fs, treating this run as globally stalled. Marking companies for retry.",
                idle,
            )

            for c in companies:
                try:
                    snap = await state.get_company_snapshot(c.company_id)
                    status = getattr(snap, "status", None)
                except Exception as e:
                    logger.warning(
                        "Global StallGuard: failed to get snapshot for %s while marking stall: %s",
                        c.company_id,
                        e,
                    )
                    mark_company_stalled(c.company_id)
                    continue

                if status != COMPANY_STATUS_LLM_DONE:
                    mark_company_stalled(c.company_id)

            for company_id, task in list(company_tasks.items()):
                if not task.done():
                    logger.error(
                        "Global StallGuard: cancelling company task company_id=%s",
                        company_id,
                    )
                    task.cancel()

        # Session / browser lifecycle
        total = len(companies)
        companies_per_session = int(getattr(args, "companies_per_session", 0) or 0)
        ranges = _session_ranges(total, companies_per_session)
        session_count = len(ranges)

        for session_idx, (session_start, session_end) in enumerate(ranges, start=1):
            if abort_run:
                logger.error(
                    "Abort flag set (memory pressure or global stall). "
                    "Not starting browser session %d/%d.",
                    session_idx,
                    session_count,
                )
                break

            logger.info(
                "Starting browser session %d/%d for companies[%d:%d]",
                session_idx,
                session_count,
                session_start,
                session_end,
            )

            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                global_stall_task = asyncio.create_task(
                    _global_stall_watchdog(),
                    name=f"global-stall-watchdog-{session_idx}",
                )

                sem = asyncio.Semaphore(max_companies)
                batch_size = max(max_companies * 2, max_companies)

                for batch_start in range(session_start, session_end, batch_size):
                    if abort_run:
                        logger.error(
                            "Abort flag set (memory pressure or global stall). "
                            "Not scheduling further batches in session %d.",
                            session_idx,
                        )
                        break

                    batch = companies[
                        batch_start : min(batch_start + batch_size, session_end)
                    ]

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
                                timeout_error_marker=timeout_error_marker,
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
                                memory_guard=memory_guard,
                                ac_controller=ac_controller,
                                active_counter=active_counter,
                                dfs_factory=dfs_factory,
                                crawler_base_cfg=crawler_base_cfg,
                                page_policy=page_policy,
                                page_interaction_factory=page_interaction_factory,
                                page_timeout_ms=page_timeout_ms,
                            ),
                            name=f"company-{company.company_id}",
                        )
                        company_tasks[company.company_id] = task
                        tasks.append(task)

                    if not tasks:
                        continue

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for company in batch:
                        company_tasks.pop(company.company_id, None)

                    for r in results:
                        # CriticalMemoryPressure triggers global abort only when hard guard enabled.
                        if (
                            isinstance(r, CriticalMemoryPressure)
                            and getattr(args, "enable_hard_memory_guard", False)
                        ):
                            logger.error(
                                "Critical memory pressure reported by at least one "
                                "company; aborting remaining batches so the wrapper "
                                "can restart the run.",
                            )
                            abort_run = True
                            break

                if global_stall_task is not None:
                    global_stall_task.cancel()
                    try:
                        await global_stall_task
                    except asyncio.CancelledError:
                        pass
                    global_stall_task = None

            # Exiting async-with closes AsyncWebCrawler and Chromium
            if abort_run:
                logger.error(
                    "Abort flag set; stopping after browser session %d/%d.",
                    session_idx,
                    session_count,
                )
                break

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

        if ac_controller is not None:
            try:
                await ac_controller.stop()
            except Exception:
                logger.exception("Error while stopping AdaptiveConcurrencyController")

    if _retry_tracker is not None:
        exit_code = _retry_tracker.finalize_and_exit_code(RETRY_EXIT_CODE)
        if exit_code != 0:
            raise SystemExit(exit_code)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()