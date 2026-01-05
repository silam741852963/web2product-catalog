from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from crawl4ai import AsyncWebCrawler, CacheMode

from configs.models import (
    Company,
    URL_INDEX_META_KEY,
    UrlIndexEntry,
    UrlIndexEntryStatus,
    UrlIndexMeta,
)

from extensions.crawl.state import patch_url_index_meta, upsert_url_index_entry
from extensions.filter import md_gating
from extensions.guard.connectivity import ConnectivityGuard
from extensions.io.output_paths import ensure_company_dirs, save_stage_output
from extensions.schedule.retry import (
    CrawlerFatalError,
    CrawlerTimeoutError,
    is_goto_timeout_error,
    is_playwright_driver_disconnect,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------------------------
# Failure markers + detection (single source of truth)
# --------------------------------------------------------------------------------------

MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"

_TIMEOUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btimeout\s+\d+\s*ms\s+exceeded\b", re.IGNORECASE),
    re.compile(r"\bnavigation\s+timeout\s+of\s+\d+\s*ms\s+exceeded\b", re.IGNORECASE),
    re.compile(r"\btimed\s+out\b", re.IGNORECASE),
    re.compile(r"\btimeouterror\b", re.IGNORECASE),
)

_MEMORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(re.escape(MEMORY_PRESSURE_MARKER), re.IGNORECASE),
    re.compile(r"\bcritical\s+memory\s+pressure\b", re.IGNORECASE),
    re.compile(r"\bmemory\s+pressure\b", re.IGNORECASE),
    re.compile(r"\bout\s+of\s+memory\b", re.IGNORECASE),
    re.compile(r"\boomed\b|\boom\b", re.IGNORECASE),
)

_TRANSPORT_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    # DNS / resolution
    re.compile(r"\bnxdomain\b", re.IGNORECASE),
    re.compile(r"\bname or service not known\b", re.IGNORECASE),
    re.compile(r"\btemporary failure in name resolution\b", re.IGNORECASE),
    re.compile(r"\bno address associated with hostname\b", re.IGNORECASE),
    re.compile(r"\bgetaddrinfo\b", re.IGNORECASE),
    re.compile(r"\bname resolution\b", re.IGNORECASE),
    # TCP / routing
    re.compile(r"\bconnection refused\b", re.IGNORECASE),
    re.compile(r"\bconnection reset\b", re.IGNORECASE),
    re.compile(r"\bconnection aborted\b", re.IGNORECASE),
    re.compile(r"\bnetwork is unreachable\b", re.IGNORECASE),
    re.compile(r"\bno route to host\b", re.IGNORECASE),
    re.compile(r"\bhost is down\b", re.IGNORECASE),
    re.compile(r"\bbroken pipe\b", re.IGNORECASE),
    re.compile(r"\bsocket hang up\b", re.IGNORECASE),
    re.compile(r"\bserver disconnected\b", re.IGNORECASE),
    re.compile(r"\bssl\b.*\bhandshake\b", re.IGNORECASE),
    re.compile(r"\btls\b.*\bhandshake\b", re.IGNORECASE),
)

_GOTO_TIMEOUT_RE = re.compile(r"(page\.goto: timeout|timeout \d+ms exceeded)", re.I)


def is_timeout_error_message(error: str, marker: Optional[str] = None) -> bool:
    if not error:
        return False
    if marker and marker in error:
        return True
    for rx in _TIMEOUT_PATTERNS:
        if rx.search(error):
            return True
    return False


def is_memory_pressure_error_message(error: str) -> bool:
    if not error:
        return False
    for rx in _MEMORY_PATTERNS:
        if rx.search(error):
            return True
    return False


def _looks_transport_error_for_guard(error: str) -> bool:
    if not error:
        return False
    # Do NOT treat generic timeouts as transport errors (too noisy / site-specific).
    if is_timeout_error_message(error):
        return False
    for rx in _TRANSPORT_ERROR_PATTERNS:
        if rx.search(error):
            return True
    return False


def _update_connectivity_guard_from_page(
    guard: ConnectivityGuard,
    *,
    error_str: str,
    status_code: Any,
) -> None:
    code_int: Optional[int]
    try:
        code_int = int(status_code) if status_code is not None else None
    except Exception:
        code_int = None

    if _looks_transport_error_for_guard(error_str):
        guard.record_transport_error()
        return

    # Any HTTP response => connectivity exists (even 404/500).
    if code_int is not None:
        guard.record_success()
        return

    # No status code and no error => success. Otherwise neutral.
    if not (error_str or "").strip():
        guard.record_success()
        return


# --------------------------------------------------------------------------------------
# Summary + per-page outcome
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class PagePipelineSummary:
    total_pages: int = 0
    markdown_saved: int = 0
    markdown_suppressed: int = 0
    timeout_pages: int = 0
    memory_pressure_pages: int = 0

    last_error: str = ""
    last_error_url: str = ""
    last_status_code: Optional[int] = None
    last_error_is_timeout: bool = False
    last_error_is_memory: bool = False

    def as_dict(self) -> Dict[str, Any]:
        tp = max(0, int(self.total_pages))
        return {
            "total_pages": tp,
            "markdown_saved": int(self.markdown_saved),
            "markdown_suppressed": int(self.markdown_suppressed),
            "timeout_pages": int(self.timeout_pages),
            "memory_pressure_pages": int(self.memory_pressure_pages),
            "timeout_ratio": float(self.timeout_pages) / float(tp or 1),
            "mem_ratio": float(self.memory_pressure_pages) / float(tp or 1),
            "last_error": self.last_error,
            "last_error_url": self.last_error_url,
            "last_status_code": self.last_status_code,
            "last_error_is_timeout": bool(self.last_error_is_timeout),
            "last_error_is_memory": bool(self.last_error_is_memory),
        }


@dataclass(slots=True)
class PageProcessOutcome:
    url: str
    md_saved: bool
    md_suppressed: bool
    timeout_exceeded: bool
    memory_pressure: bool
    error_str: str = ""
    status_code: Optional[int] = None


# --------------------------------------------------------------------------------------
# url_index single-writer (uses crawl.state.py helpers)
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class UrlIndexWriteConfig:
    queue_maxsize: int = 8192


class UrlIndexUpdateWriter:
    """
    Single writer that serializes url_index mutations via crawl.state.py APIs:
      - upsert_url_index_entry(bvdid, url, patch)
      - patch_url_index_meta(bvdid, patch)
    """

    META_KEY = URL_INDEX_META_KEY

    def __init__(self, bvdid: str, *, cfg: UrlIndexWriteConfig) -> None:
        self.bvdid = bvdid
        self.cfg = cfg
        self._q: asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]] = asyncio.Queue(
            maxsize=max(1, int(cfg.queue_maxsize))
        )
        self._task: Optional[asyncio.Task[None]] = None
        self._fatal: Optional[BaseException] = None

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("UrlIndexUpdateWriter.start called twice")
        self._task = asyncio.create_task(
            self._run(), name=f"url-index-writer-{self.bvdid}"
        )

    async def enqueue(self, url: str, patch: Dict[str, Any]) -> None:
        if self._fatal is not None:
            raise self._fatal
        await self._q.put((url, patch))

    async def patch_meta(self, patch: Dict[str, Any]) -> None:
        if self._fatal is not None:
            raise self._fatal
        await self._q.put((self.META_KEY, patch))

    async def close(self) -> None:
        if self._task is None:
            raise RuntimeError("UrlIndexUpdateWriter.close called before start")
        await self._q.put(None)
        await self._task
        self._task = None
        if self._fatal is not None:
            raise self._fatal

    async def _run(self) -> None:
        try:
            while True:
                item = await self._q.get()
                if item is None:
                    return
                url, patch = item
                if url == self.META_KEY:
                    await asyncio.to_thread(patch_url_index_meta, self.bvdid, patch)
                else:
                    await asyncio.to_thread(
                        upsert_url_index_entry, self.bvdid, url, patch
                    )
        except BaseException as e:
            self._fatal = e
            return


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------


async def _save_stage_threaded(
    *, bvdid: str, url: str, data: str, stage: str
) -> Optional[str]:
    def _write() -> Optional[str]:
        p = save_stage_output(bvdid=bvdid, url=url, data=data, stage=stage)
        return str(p) if p is not None else None

    try:
        return await asyncio.to_thread(_write)
    except Exception:
        logger.exception(
            "Failed to write stage=%s (company=%s url=%s)", stage, bvdid, url
        )
        return None


_warned_company_callbacks = False


async def process_page_result(
    page_result: Any,
    *,
    company: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    url_index_sink: Callable[[str, Dict[str, Any]], Awaitable[None]],
    timeout_error_marker: str = "",
    mark_company_timeout_cb: Optional[Callable[[str], None]] = None,
    mark_company_memory_cb: Optional[Callable[[str], None]] = None,
) -> PageProcessOutcome:
    global _warned_company_callbacks

    if (
        mark_company_timeout_cb is not None or mark_company_memory_cb is not None
    ) and not _warned_company_callbacks:
        logger.warning(
            "process_page_result received company-level retry callbacks, but they are intentionally ignored "
            "(per-page signals remain URL-level only)."
        )
        _warned_company_callbacks = True

    company_id = getattr(company, "company_id", None)
    if not company_id:
        logger.warning(
            "process_page_result called without company_id on company=%r", company
        )
        return PageProcessOutcome(
            url="",
            md_saved=False,
            md_suppressed=False,
            timeout_exceeded=False,
            memory_pressure=False,
        )

    _getattr = getattr
    requested_url = _getattr(page_result, "url", None)
    final_url = _getattr(page_result, "final_url", None) or requested_url
    if not final_url:
        logger.warning(
            "Page result missing URL; skipping entry (company_id=%s)", company_id
        )
        return PageProcessOutcome(
            url="",
            md_saved=False,
            md_suppressed=False,
            timeout_exceeded=False,
            memory_pressure=False,
        )
    url = final_url

    markdown = _getattr(page_result, "markdown", None)
    status_code = _getattr(page_result, "status_code", None)
    error = _getattr(page_result, "error", None) or _getattr(
        page_result, "error_message", None
    )

    error_str = error if isinstance(error, str) else ""
    timeout_exceeded = is_timeout_error_message(
        error_str, marker=(timeout_error_marker or None)
    )
    memory_pressure = is_memory_pressure_error_message(error_str)

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
        html_path = await _save_stage_threaded(
            bvdid=company_id, url=url, data=html, stage="html"
        )

    if guard is not None:
        with contextlib.suppress(Exception):
            _update_connectivity_guard_from_page(
                guard, error_str=error_str, status_code=status_code
            )

    gating_accept = action == "save"
    md_path: Optional[str] = None
    md_status: UrlIndexEntryStatus

    if gating_accept and markdown:
        md_path = await _save_stage_threaded(
            bvdid=company_id, url=url, data=markdown, stage="markdown"
        )
        md_status = "markdown_saved" if md_path is not None else "markdown_error"
    else:
        md_status = "markdown_suppressed"

    # Override status to make the signal visible in url_index
    if timeout_exceeded:
        md_status = "timeout_page_exceeded"
    if memory_pressure:
        md_status = "memory_pressure"

    entry = UrlIndexEntry(
        company_id=str(company_id),
        url=str(url),
        requested_url=str(requested_url) if requested_url else None,
        status_code=status_code if status_code is not None else None,
        error=error,
        depth=_getattr(page_result, "depth", None),
        presence=0,
        extracted=0,
        gating_accept=bool(gating_accept),
        gating_action=action,
        gating_reason=reason,
        md_total_words=stats.get("total_words"),
        status=str(md_status),
        updated_at=_now_iso(),
        markdown_path=md_path,
        html_path=html_path,
        scheduled_retry=True if (timeout_exceeded or memory_pressure) else None,
        timeout_page_exceeded=True if timeout_exceeded else None,
        memory_pressure=True if memory_pressure else None,
    )

    await url_index_sink(url, entry.to_dict())

    sc_int: Optional[int] = None
    try:
        sc_int = int(status_code) if status_code is not None else None
    except Exception:
        sc_int = None

    return PageProcessOutcome(
        url=url,
        md_saved=(md_path is not None and md_status == "markdown_saved"),
        md_suppressed=(md_status == "markdown_suppressed"),
        timeout_exceeded=timeout_exceeded,
        memory_pressure=memory_pressure,
        error_str=(error_str or "").strip(),
        status_code=sc_int,
    )


class ConcurrentPageResultProcessor:
    def __init__(
        self,
        *,
        company: Any,
        guard: Optional[ConnectivityGuard],
        gating_cfg: md_gating.MarkdownGatingConfig,
        concurrency: int = 8,
        page_queue_maxsize: int = 0,
        url_index_write_cfg: Optional[UrlIndexWriteConfig] = None,
        timeout_error_marker: str = "",
    ) -> None:
        self.company = company
        self.company_id = getattr(company, "company_id", None)
        if not self.company_id:
            raise ValueError(
                f"company missing required attribute company_id: {company!r}"
            )

        self.guard = guard
        self.gating_cfg = gating_cfg
        self.timeout_error_marker = timeout_error_marker

        self.concurrency = max(1, int(concurrency))
        maxsize = (
            int(page_queue_maxsize)
            if page_queue_maxsize and page_queue_maxsize > 0
            else max(16, self.concurrency * 2)
        )
        self._q: asyncio.Queue[Optional[Any]] = asyncio.Queue(maxsize=maxsize)

        self._workers: List[asyncio.Task[None]] = []
        self._fatal: Optional[BaseException] = None

        self._writer = UrlIndexUpdateWriter(
            str(self.company_id), cfg=(url_index_write_cfg or UrlIndexWriteConfig())
        )

        self._summary = PagePipelineSummary()
        self._summary_lock = asyncio.Lock()

        self._crawl_finished_meta_patch: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        ensure_company_dirs(str(self.company_id))
        await self._writer.start()

        # Write initial meta patch early so partial runs still have stable identity fields.
        now = _now_iso()
        meta0 = UrlIndexMeta(
            company_id=str(self.company_id),
            created_at=now,
            updated_at=now,
        )
        await self._writer.patch_meta(meta0.to_dict())

        for i in range(self.concurrency):
            t = asyncio.create_task(
                self._worker_loop(i), name=f"page-worker-{self.company_id}-{i}"
            )
            self._workers.append(t)

    async def submit(self, page_result: Any) -> None:
        if self._fatal is not None:
            raise self._fatal
        await self._q.put(page_result)
        if self._fatal is not None:
            raise self._fatal

    def mark_crawl_finished(
        self, *, reason: str = "ok", meta_patch: Optional[Dict[str, Any]] = None
    ) -> None:
        patch: Dict[str, Any] = {
            "company_id": str(self.company_id),
            "updated_at": _now_iso(),
            "crawl_finished": True,
            "crawl_finished_at": _now_iso(),
            "crawl_reason": str(reason or "ok"),
        }
        if meta_patch:
            patch.update(meta_patch)
        self._crawl_finished_meta_patch = patch

    async def finish(self) -> None:
        for _ in self._workers:
            await self._q.put(None)
        await asyncio.gather(*self._workers, return_exceptions=False)
        self._workers.clear()

        if self._crawl_finished_meta_patch is not None:
            async with self._summary_lock:
                self._crawl_finished_meta_patch.update(
                    {
                        "company_id": str(self.company_id),
                        "updated_at": _now_iso(),
                        "total_pages": int(self._summary.total_pages),
                        "markdown_saved": int(self._summary.markdown_saved),
                        "markdown_suppressed": int(self._summary.markdown_suppressed),
                        "timeout_pages": int(self._summary.timeout_pages),
                        "memory_pressure_pages": int(
                            self._summary.memory_pressure_pages
                        ),
                    }
                )
            await self._writer.patch_meta(self._crawl_finished_meta_patch)

        await self._writer.close()

        if self._fatal is not None:
            raise self._fatal

    async def get_summary(self) -> PagePipelineSummary:
        async with self._summary_lock:
            return PagePipelineSummary(
                total_pages=self._summary.total_pages,
                markdown_saved=self._summary.markdown_saved,
                markdown_suppressed=self._summary.markdown_suppressed,
                timeout_pages=self._summary.timeout_pages,
                memory_pressure_pages=self._summary.memory_pressure_pages,
                last_error=self._summary.last_error,
                last_error_url=self._summary.last_error_url,
                last_status_code=self._summary.last_status_code,
                last_error_is_timeout=self._summary.last_error_is_timeout,
                last_error_is_memory=self._summary.last_error_is_memory,
            )

    async def _accumulate(self, outcome: PageProcessOutcome) -> None:
        async with self._summary_lock:
            self._summary.total_pages += 1
            if outcome.md_saved:
                self._summary.markdown_saved += 1
            if outcome.md_suppressed:
                self._summary.markdown_suppressed += 1
            if outcome.timeout_exceeded:
                self._summary.timeout_pages += 1
            if outcome.memory_pressure:
                self._summary.memory_pressure_pages += 1

            err = (outcome.error_str or "").strip()
            if err:
                prefer = outcome.timeout_exceeded or outcome.memory_pressure
                cur = (self._summary.last_error or "").strip()
                cur_is_strong = bool(
                    self._summary.last_error_is_timeout
                    or self._summary.last_error_is_memory
                )
                if prefer or not cur or not cur_is_strong:
                    self._summary.last_error = err[:4000]
                    self._summary.last_error_url = outcome.url
                    self._summary.last_status_code = outcome.status_code
                    self._summary.last_error_is_timeout = bool(outcome.timeout_exceeded)
                    self._summary.last_error_is_memory = bool(outcome.memory_pressure)

    async def _worker_loop(self, worker_idx: int) -> None:
        while True:
            item = await self._q.get()
            try:
                if item is None:
                    return
                outcome = await process_page_result(
                    page_result=item,
                    company=self.company,
                    guard=self.guard,
                    gating_cfg=self.gating_cfg,
                    url_index_sink=self._writer.enqueue,
                    timeout_error_marker=self.timeout_error_marker,
                )
                await self._accumulate(outcome)
            except BaseException as e:
                self._fatal = e
                return
            finally:
                self._q.task_done()


# =====================================================================
# Runner
# =====================================================================

ProgressCB = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class CrawlRunnerConfig:
    page_result_concurrency: int
    page_queue_maxsize: int

    arun_init_timeout_sec: float
    stream_no_yield_timeout_sec: float
    submit_timeout_sec: float
    direct_fetch_total_timeout_sec: float
    processor_finish_timeout_sec: float
    generator_close_timeout_sec: float

    hard_max_pages: Optional[int] = None
    page_timeout_ms: Optional[int] = None
    direct_fetch_urls: bool = False

    url_index_queue_maxsize: int = 8192
    crawl4ai_cache_mode: CacheMode = CacheMode.BYPASS


def _string_looks_like_goto_timeout(s: str) -> bool:
    if not s:
        return False
    if _GOTO_TIMEOUT_RE.search(s):
        return True
    return is_timeout_error_message(s)


def _raise_if_fatal_transport(e: BaseException) -> None:
    if is_playwright_driver_disconnect(e):
        raise CrawlerFatalError("Playwright driver disconnected") from e


def _raise_if_goto_timeout(e: BaseException, *, company_id: str, url: str) -> None:
    if is_goto_timeout_error(e) or _string_looks_like_goto_timeout(str(e)):
        raise CrawlerTimeoutError(
            f"[GOTO Timeout] {type(e).__name__}: {e}",
            stage="goto",
            company_id=company_id,
            url=url,
        ) from e


async def run_company_crawl(
    company: Company,
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
    on_progress: Optional[ProgressCB] = None,
) -> PagePipelineSummary:
    ensure_company_dirs(company.company_id)

    start_urls = list(root_urls) if root_urls else [company.domain_url]
    if cfg.hard_max_pages is not None and len(start_urls) > int(cfg.hard_max_pages):
        start_urls = start_urls[: int(cfg.hard_max_pages)]

    processor = ConcurrentPageResultProcessor(
        company=company,
        guard=guard,
        gating_cfg=gating_cfg,
        concurrency=max(1, int(cfg.page_result_concurrency)),
        page_queue_maxsize=max(1, int(cfg.page_queue_maxsize)),
        url_index_write_cfg=UrlIndexWriteConfig(
            queue_maxsize=max(1, int(cfg.url_index_queue_maxsize))
        ),
        timeout_error_marker="",
    )

    pages_seen = 0
    hard_max_pages_hit = False

    await processor.start()

    def _tick_progress() -> None:
        if on_progress is not None:
            with contextlib.suppress(Exception):
                on_progress()

    def _hit_hard_cap_now() -> bool:
        return cfg.hard_max_pages is not None and pages_seen >= int(cfg.hard_max_pages)

    async def submit_one(page_result: Any) -> bool:
        nonlocal pages_seen, hard_max_pages_hit
        if _hit_hard_cap_now():
            hard_max_pages_hit = True
            return False
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
        _tick_progress()
        if _hit_hard_cap_now():
            hard_max_pages_hit = True
        return True

    async def _aclose_strict(agen: Any) -> None:
        aclose = getattr(agen, "aclose", None)
        if callable(aclose):
            await asyncio.wait_for(
                aclose(), timeout=float(cfg.generator_close_timeout_sec)
            )

    async def process_result(result_or_gen: Any, *, origin_url: str) -> None:
        nonlocal hard_max_pages_hit
        if isinstance(result_or_gen, list):
            for item in result_or_gen:
                ok = await submit_one(item)
                if not ok:
                    break
            return

        if hasattr(result_or_gen, "__anext__"):
            agen = result_or_gen
            try:
                while True:
                    if _hit_hard_cap_now():
                        hard_max_pages_hit = True
                        break
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

                    ok = await submit_one(page_result)
                    if not ok:
                        break
            except BaseException as e:
                _raise_if_fatal_transport(e)
                _raise_if_goto_timeout(e, company_id=company.company_id, url=origin_url)
                raise
            finally:
                await _aclose_strict(agen)
            return

        await submit_one(result_or_gen)

    def make_config(url: str, *, deep: bool) -> Any:
        interaction = page_interaction_factory.base_config(
            url=url, policy=page_policy, js_only=False
        )
        js_code = "\n\n".join(interaction.js_code) if interaction.js_code else ""
        c = crawler_base_cfg.clone(
            cache_mode=cfg.crawl4ai_cache_mode,
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

    success = False
    try:
        if cfg.direct_fetch_urls:
            for u in start_urls:
                if _hit_hard_cap_now():
                    hard_max_pages_hit = True
                    break
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
                except BaseException as e:
                    _raise_if_fatal_transport(e)
                    _raise_if_goto_timeout(e, company_id=company.company_id, url=u)
                    raise

                await process_result(res, origin_url=u)
        else:
            for u in start_urls:
                if _hit_hard_cap_now():
                    hard_max_pages_hit = True
                    break
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
                except BaseException as e:
                    _raise_if_fatal_transport(e)
                    _raise_if_goto_timeout(e, company_id=company.company_id, url=u)
                    raise

                await process_result(res_or_gen, origin_url=u)

        success = True
    finally:
        if success:
            reason = "max_pages_hit" if hard_max_pages_hit else "ok"
            processor.mark_crawl_finished(
                reason=reason,
                meta_patch={
                    "company_id": company.company_id,
                    "updated_at": _now_iso(),
                    "pages_seen": int(pages_seen),
                    "hard_max_pages": (
                        int(cfg.hard_max_pages)
                        if cfg.hard_max_pages is not None
                        else None
                    ),
                    "hard_max_pages_hit": bool(hard_max_pages_hit),
                },
            )

        try:
            await asyncio.wait_for(
                processor.finish(), timeout=float(cfg.processor_finish_timeout_sec)
            )
        except asyncio.TimeoutError as e:
            raise CrawlerTimeoutError(
                f"processor.finish timeout ({cfg.processor_finish_timeout_sec:.1f}s)",
                stage="processor_finish",
                company_id=company.company_id,
                url=(start_urls[0] if start_urls else company.domain_url),
            ) from e

        gc.collect()

    summary = await processor.get_summary()

    last_err = (summary.last_error or "").strip()
    if last_err and _string_looks_like_goto_timeout(last_err):
        raise CrawlerTimeoutError(
            f"[GOTO Timeout] {last_err}",
            stage="goto",
            company_id=company.company_id,
            url=(
                summary.last_error_url
                or (start_urls[0] if start_urls else company.domain_url)
            ),
        )

    return summary


def memory_pressure_exception(summary_reason: str) -> RuntimeError:
    return RuntimeError(f"{MEMORY_PRESSURE_MARKER} | {summary_reason}")
