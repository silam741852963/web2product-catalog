from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.deep_crawling.filters import FilterChain

from . import md_gating
from .connectivity_guard import ConnectivityGuard
from .output_paths import ensure_company_dirs
from .page_pipeline import (
    ConcurrentPageResultProcessor,
    UrlIndexFlushConfig,
    PagePipelineSummary,
    MEMORY_PRESSURE_MARKER,
)
from .retry_policy import (
    CrawlerTimeoutError,
    CrawlerFatalError,
    is_goto_timeout_error,
    is_playwright_driver_disconnect,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class CompanyLike(Protocol):
    company_id: str
    domain_url: str


@dataclass(frozen=True, slots=True)
class CrawlRunnerConfig:
    # processor / pipeline knobs
    page_result_concurrency: int
    page_queue_maxsize: int
    url_index_flush_every: int
    url_index_flush_interval_sec: float
    url_index_queue_maxsize: int

    # timeouts
    arun_init_timeout_sec: float
    stream_no_yield_timeout_sec: float
    submit_timeout_sec: float
    direct_fetch_total_timeout_sec: float
    processor_finish_timeout_sec: float
    generator_close_timeout_sec: float

    # limits
    hard_max_pages: Optional[int] = None

    # behavior
    page_timeout_ms: Optional[int] = None
    direct_fetch_urls: bool = False


async def run_company_crawl(
    company: CompanyLike,
    *,
    crawler: AsyncWebCrawler,
    deep_strategy: Any,
    guard: ConnectivityGuard,
    gating_cfg: md_gating.MarkdownGatingConfig,
    crawler_base_cfg: Any,
    page_policy: Any,
    page_interaction_factory: Any,
    root_urls: Optional[List[str]],
    cfg: CrawlRunnerConfig,
) -> PagePipelineSummary:
    ensure_company_dirs(company.company_id)

    start_urls = list(root_urls) if root_urls else [company.domain_url]
    if cfg.hard_max_pages and len(start_urls) > cfg.hard_max_pages:
        start_urls = start_urls[: cfg.hard_max_pages]

    flush_cfg = UrlIndexFlushConfig(
        flush_every=max(1, int(cfg.url_index_flush_every)),
        flush_interval_sec=max(0.2, float(cfg.url_index_flush_interval_sec)),
        queue_maxsize=max(256, int(cfg.url_index_queue_maxsize)),
    )

    processor = ConcurrentPageResultProcessor(
        company=company,
        guard=guard,
        gating_cfg=gating_cfg,
        timeout_error_marker="",
        mark_company_timeout_cb=None,
        mark_company_memory_cb=None,
        concurrency=max(1, int(cfg.page_result_concurrency)),
        page_queue_maxsize=max(1, int(cfg.page_queue_maxsize)),
        url_index_flush_cfg=flush_cfg,
    )

    pages_seen = 0
    await processor.start()

    async def submit_one(page_result: Any) -> None:
        nonlocal pages_seen
        if cfg.hard_max_pages and pages_seen >= cfg.hard_max_pages:
            raise StopAsyncIteration

        try:
            await asyncio.wait_for(
                processor.submit(page_result), timeout=float(cfg.submit_timeout_sec)
            )
        except asyncio.TimeoutError as e:
            raise CrawlerTimeoutError(
                f"processor.submit timeout ({cfg.submit_timeout_sec:.1f}s)",
                stage="submit",
                company_id=company.company_id,
                url=getattr(page_result, "url", None) or company.domain_url,
            ) from e

        pages_seen += 1

    async def process_result(result_or_gen: Any, *, origin_url: str) -> None:
        # list
        if isinstance(result_or_gen, list):
            for item in result_or_gen:
                await submit_one(item)
            return

        # async generator
        if hasattr(result_or_gen, "__anext__"):
            agen = result_or_gen
            try:
                while True:
                    try:
                        page_result = await asyncio.wait_for(
                            agen.__anext__(),
                            timeout=float(cfg.stream_no_yield_timeout_sec),
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as e:
                        raise CrawlerTimeoutError(
                            f"stream no-yield timeout ({cfg.stream_no_yield_timeout_sec:.1f}s)",
                            stage="stream_no_yield",
                            company_id=company.company_id,
                            url=origin_url,
                        ) from e

                    await submit_one(page_result)

            except Exception as e:
                if is_goto_timeout_error(e):
                    raise CrawlerTimeoutError(
                        f"[GOTO Timeout] {type(e).__name__}: {e}",
                        stage="goto",
                        company_id=company.company_id,
                        url=origin_url,
                    ) from e

                if is_playwright_driver_disconnect(e):
                    raise CrawlerFatalError("Playwright driver disconnected") from e

                raise
            finally:
                aclose = getattr(agen, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(
                            aclose(), timeout=float(cfg.generator_close_timeout_sec)
                        )
            return

        # single item
        await submit_one(result_or_gen)

    def make_config(url: str, *, deep: bool) -> Any:
        interaction = page_interaction_factory.base_config(
            url=url, policy=page_policy, js_only=False
        )
        js_code = "\n\n".join(interaction.js_code) if interaction.js_code else ""
        c = crawler_base_cfg.clone(
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            deep_crawl_strategy=(deep_strategy if deep else None),
            stream=deep,
            js_code=js_code,
            js_only=interaction.js_only,
            wait_for=interaction.wait_for,
        )
        if cfg.page_timeout_ms is not None:
            c.page_timeout = int(cfg.page_timeout_ms)  # type: ignore[attr-defined]
        return c

    try:
        if cfg.direct_fetch_urls:
            for u in start_urls:
                c = make_config(u, deep=False)
                try:
                    res = await asyncio.wait_for(
                        crawler.arun(u, config=c),
                        timeout=float(cfg.direct_fetch_total_timeout_sec),
                    )
                except asyncio.TimeoutError as e:
                    raise CrawlerTimeoutError(
                        f"direct fetch timeout ({cfg.direct_fetch_total_timeout_sec:.1f}s)",
                        stage="direct_fetch",
                        company_id=company.company_id,
                        url=u,
                    ) from e
                await process_result(res, origin_url=u)
                if cfg.hard_max_pages and pages_seen >= cfg.hard_max_pages:
                    break
        else:
            for u in start_urls:
                c = make_config(u, deep=True)
                try:
                    res_or_gen = await asyncio.wait_for(
                        crawler.arun(u, config=c),
                        timeout=float(cfg.arun_init_timeout_sec),
                    )
                except asyncio.TimeoutError as e:
                    raise CrawlerTimeoutError(
                        f"crawler.arun init timeout ({cfg.arun_init_timeout_sec:.1f}s)",
                        stage="arun_init",
                        company_id=company.company_id,
                        url=u,
                    ) from e
                await process_result(res_or_gen, origin_url=u)
                if cfg.hard_max_pages and pages_seen >= cfg.hard_max_pages:
                    break

    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                processor.finish(), timeout=float(cfg.processor_finish_timeout_sec)
            )
        with contextlib.suppress(Exception):
            gc.collect()

    with contextlib.suppress(Exception):
        return await processor.get_summary()
    return PagePipelineSummary()


def memory_pressure_exception(summary_reason: str) -> RuntimeError:
    return RuntimeError(f"{MEMORY_PRESSURE_MARKER} | {summary_reason}")
