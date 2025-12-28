from __future__ import annotations

import asyncio
import contextlib
import errno
import gc
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

from crawl4ai import AsyncWebCrawler, CacheMode

from . import md_gating
from .connectivity_guard import ConnectivityGuard
from .output_paths import ensure_company_dirs, sanitize_bvdid, save_stage_output
from .retry import (
    CrawlerFatalError,
    CrawlerTimeoutError,
    is_goto_timeout_error,
    is_playwright_driver_disconnect,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------------------
# Unified failure markers + smarter detection (single source of truth)
# --------------------------------------------------------------------------------------

# This marker is emitted by your crawler layer today; keep as a strong signal.
MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"

# Timeout messages vary across Playwright / Crawl4AI / underlying stacks.
_TIMEOUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btimeout\s+\d+\s*ms\s+exceeded\b", re.IGNORECASE),
    re.compile(r"\bnavigation\s+timeout\s+of\s+\d+\s*ms\s+exceeded\b", re.IGNORECASE),
    re.compile(r"\btimed\s+out\b", re.IGNORECASE),
    re.compile(r"\btimeouterror\b", re.IGNORECASE),
)

# Memory pressure messages vary too; keep conservative.
_MEMORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(re.escape(MEMORY_PRESSURE_MARKER), re.IGNORECASE),
    re.compile(r"\bcritical\s+memory\s+pressure\b", re.IGNORECASE),
    re.compile(r"\bmemory\s+pressure\b", re.IGNORECASE),
    re.compile(r"\bout\s+of\s+memory\b", re.IGNORECASE),
    re.compile(r"\boomed\b|\boom\b", re.IGNORECASE),
)

# Conservative "transport-ish" patterns used only for ConnectivityGuard EWMA
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

_TRANSPORT_LIKE_HTTP = {502, 503, 504, 520, 521, 522, 523, 524}


def is_timeout_error_message(error: str, marker: Optional[str] = None) -> bool:
    if not error:
        return False
    if marker:
        try:
            if marker in error:
                return True
        except Exception:
            pass
    for rx in _TIMEOUT_PATTERNS:
        try:
            if rx.search(error):
                return True
        except Exception:
            continue
    return False


def is_memory_pressure_error_message(error: str) -> bool:
    if not error:
        return False
    for rx in _MEMORY_PATTERNS:
        try:
            if rx.search(error):
                return True
        except Exception:
            continue
    return False


def _looks_transport_error_for_guard(error: str) -> bool:
    if not error:
        return False
    # Do NOT treat generic timeouts as transport errors (too noisy / site-specific).
    if is_timeout_error_message(error):
        return False
    for rx in _TRANSPORT_ERROR_PATTERNS:
        try:
            if rx.search(error):
                return True
        except Exception:
            continue
    return False


def _update_connectivity_guard_from_page(
    guard: ConnectivityGuard,
    *,
    error_str: str,
    status_code: Any,
) -> None:
    try:
        code_int = int(status_code) if status_code is not None else None
    except Exception:
        code_int = None

    transport_like = _looks_transport_error_for_guard(error_str)
    if transport_like:
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
# Small, fast atomic JSON write
# --------------------------------------------------------------------------------------


def _retry_emfile(fn, attempts: int = 6, base_delay: float = 0.15):
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENFILE) or "Too many open files" in str(
                e
            ):
                time.sleep(base_delay * (2**i))
                continue
            raise


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    tmp = Path(f"{str(path)}.tmp.{stamp}")

    def _write() -> None:
        try:
            with open(tmp, "w", encoding=encoding, newline="") as f:
                f.write(data)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists() and tmp != path:
                    tmp.unlink()
            except Exception:
                pass

    _retry_emfile(_write)


def _atomic_write_json_compact(path: Path, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    _atomic_write_text(path, payload, "utf-8")


def _json_load_nocache(path: Path) -> Any:
    def _read() -> Any:
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return {}

    return _retry_emfile(_read)


def _url_index_path_for(company_id: str) -> Path:
    dirs = ensure_company_dirs(company_id)
    meta_dir = dirs.get("metadata") or dirs.get("checkpoints")
    if meta_dir is None:
        safe = sanitize_bvdid(company_id)
        meta_dir = Path(os.environ.get("OUTPUT_ROOT", "outputs")) / safe / "metadata"
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir / "url_index.json"


# --------------------------------------------------------------------------------------
# Batched url_index writer
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class UrlIndexFlushConfig:
    flush_every: int = 64
    flush_interval_sec: float = 1.0
    queue_maxsize: int = 8192


class UrlIndexBatchWriter:
    """
    Single-writer, queued updates to url_index.json.

    IMPORTANT: all mutations to the underlying index dict happen in the writer task
    (via the queue), including __meta__ patches. This avoids races with the worker
    task that also flushes on intervals.
    """

    META_KEY = "__meta__"

    def __init__(self, company_id: str, *, cfg: UrlIndexFlushConfig) -> None:
        self.company_id = company_id
        self.cfg = cfg
        self.path = _url_index_path_for(company_id)

        self._q: asyncio.Queue[Optional[Tuple[str, Dict[str, Any]]]] = asyncio.Queue(
            maxsize=max(1, int(cfg.queue_maxsize))
        )
        self._task: Optional[asyncio.Task] = None
        self._index: Dict[str, Any] = {}
        self._dirty = 0
        self._last_flush = 0.0

    async def start(self) -> None:
        def _load() -> Dict[str, Any]:
            if self.path.exists():
                obj = _json_load_nocache(self.path)
                return obj if isinstance(obj, dict) else {}
            return {}

        self._index = await asyncio.to_thread(_load)
        self._last_flush = time.time()
        self._task = asyncio.create_task(
            self._run(), name=f"url-index-writer-{self.company_id}"
        )

    async def enqueue(self, url: str, patch: Dict[str, Any]) -> None:
        await self._q.put((url, patch))

    async def patch_meta(self, patch: Dict[str, Any]) -> None:
        """
        Patch url_index['__meta__'] in a race-free way (queued to writer task).
        """
        await self._q.put((self.META_KEY, patch))

    async def mark_crawl_finished(self, *, reason: str = "ok") -> None:
        """
        Mark crawl as finished for downstream recompute gate:
          url_index['__meta__']['crawl_finished'] = True
        Only call this on successful company crawl completion.
        """
        meta_patch = {
            "crawl_finished": True,
            "crawl_finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "crawl_reason": reason,
        }
        await self.patch_meta(meta_patch)

    async def close(self) -> None:
        if self._task is None:
            return
        await self._q.put(None)
        await self._task
        self._task = None

    async def _flush(self) -> None:
        if self._dirty <= 0:
            return
        index_ref = self._index

        def _write() -> None:
            _atomic_write_json_compact(self.path, index_ref)

        await asyncio.to_thread(_write)
        self._dirty = 0
        self._last_flush = time.time()

    async def _run(self) -> None:
        cfg = self.cfg
        flush_every = max(1, int(cfg.flush_every))
        flush_interval = max(0.2, float(cfg.flush_interval_sec))

        while True:
            timeout = max(0.05, flush_interval - (time.time() - self._last_flush))
            try:
                item = await asyncio.wait_for(self._q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if self._dirty > 0:
                    try:
                        await self._flush()
                    except Exception:
                        logger.exception(
                            "url_index flush failed (company=%s)", self.company_id
                        )
                continue

            if item is None:
                try:
                    await self._flush()
                except Exception:
                    logger.exception(
                        "url_index final flush failed (company=%s)", self.company_id
                    )
                return

            url, patch = item

            if url == self.META_KEY:
                cur = self._index.get(self.META_KEY)
                cur_dict = dict(cur) if isinstance(cur, dict) else {}
                cur_dict.update(patch)
                self._index[self.META_KEY] = cur_dict
                self._dirty += 1
            else:
                ent = self._index.get(url)
                ent_dict = dict(ent) if isinstance(ent, dict) else {}
                ent_dict.update(patch)
                self._index[url] = ent_dict
                self._dirty += 1

            if self._dirty >= flush_every:
                try:
                    await self._flush()
                except Exception:
                    logger.exception(
                        "url_index flush failed (company=%s)", self.company_id
                    )


# --------------------------------------------------------------------------------------
# Summary + per-page outcome (UPDATED: carries last_error)
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class PagePipelineSummary:
    total_pages: int = 0
    markdown_saved: int = 0
    markdown_suppressed: int = 0
    timeout_pages: int = 0
    memory_pressure_pages: int = 0

    # NEW: propagate the most recent (or most severe) error upward
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


_warned_company_callbacks = False


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


async def process_page_result(
    page_result: Any,
    *,
    company: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    timeout_error_marker: str = "",
    mark_company_timeout_cb: Optional[Callable[[str], None]] = None,
    mark_company_memory_cb: Optional[Callable[[str], None]] = None,
    url_index_sink: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
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

    # ConnectivityGuard update
    if guard is not None:
        with contextlib.suppress(Exception):
            _update_connectivity_guard_from_page(
                guard, error_str=error_str, status_code=status_code
            )

    gating_accept = action == "save"
    md_path: Optional[str] = None
    md_status: str

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
    if memory_pressure:
        entry["memory_pressure"] = True
        entry["scheduled_retry"] = True

    if md_path is not None:
        entry["markdown_path"] = md_path
    if html_path is not None:
        entry["html_path"] = html_path

    if url_index_sink is not None:
        await url_index_sink(url, entry)
    else:
        # Fallback: write via crawl_state if present
        from extensions.crawl_state import upsert_url_index_entry  # type: ignore

        await asyncio.to_thread(upsert_url_index_entry, company_id, url, entry)

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
        error_str=(error_str or ""),
        status_code=sc_int,
    )


class ConcurrentPageResultProcessor:
    def __init__(
        self,
        *,
        company: Any,
        guard: Optional[ConnectivityGuard],
        gating_cfg: md_gating.MarkdownGatingConfig,
        timeout_error_marker: str = "",
        mark_company_timeout_cb: Optional[Callable[[str], None]] = None,
        mark_company_memory_cb: Optional[Callable[[str], None]] = None,
        concurrency: int = 8,
        page_queue_maxsize: int = 0,
        url_index_flush_cfg: Optional[UrlIndexFlushConfig] = None,
    ) -> None:
        self.company = company
        self.company_id = getattr(company, "company_id", "unknown")
        self.guard = guard
        self.gating_cfg = gating_cfg
        self.timeout_error_marker = timeout_error_marker

        # kept for signature compatibility
        self.mark_company_timeout_cb = mark_company_timeout_cb
        self.mark_company_memory_cb = mark_company_memory_cb

        self.concurrency = max(1, int(concurrency))
        maxsize = (
            int(page_queue_maxsize)
            if page_queue_maxsize and page_queue_maxsize > 0
            else max(16, self.concurrency * 2)
        )
        self._q: asyncio.Queue[Optional[Any]] = asyncio.Queue(maxsize=maxsize)

        self._workers: list[asyncio.Task] = []
        self._fatal: Optional[BaseException] = None

        flush_cfg = url_index_flush_cfg or UrlIndexFlushConfig()
        self._writer = UrlIndexBatchWriter(self.company_id, cfg=flush_cfg)

        self._summary = PagePipelineSummary()
        self._summary_lock = asyncio.Lock()

    async def start(self) -> None:
        ensure_company_dirs(self.company_id)
        await self._writer.start()
        for i in range(self.concurrency):
            t = asyncio.create_task(
                self._worker_loop(i), name=f"page-worker-{self.company_id}-{i}"
            )
            self._workers.append(t)

    async def submit(self, page_result: Any) -> None:
        if self._fatal is not None:
            raise self._fatal
        while True:
            if self._fatal is not None:
                raise self._fatal
            try:
                self._q.put_nowait(page_result)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0.05)

    async def mark_crawl_finished(self, *, reason: str = "ok") -> None:
        """
        Marks url_index.__meta__.crawl_finished=true (queued to writer).
        Call only when the company crawl completed successfully.
        """
        await self._writer.mark_crawl_finished(reason=reason)

    async def finish(self) -> None:
        for _ in self._workers:
            await self._q.put(None)
        try:
            await asyncio.gather(*self._workers, return_exceptions=False)
        finally:
            self._workers.clear()
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

            # NEW: bubble the latest meaningful error upward.
            # Prefer timeout/memory errors (more actionable) over generic.
            err = (outcome.error_str or "").strip()
            if err:
                prefer = outcome.timeout_exceeded or outcome.memory_pressure
                cur = (self._summary.last_error or "").strip()
                cur_is_strong = bool(
                    self._summary.last_error_is_timeout
                    or self._summary.last_error_is_memory
                )
                if prefer or not cur or not cur_is_strong:
                    # truncate to avoid exploding db/logs
                    self._summary.last_error = err[:4000]
                    self._summary.last_error_url = outcome.url
                    self._summary.last_status_code = outcome.status_code
                    self._summary.last_error_is_timeout = bool(outcome.timeout_exceeded)
                    self._summary.last_error_is_memory = bool(outcome.memory_pressure)

    async def _worker_loop(self, worker_idx: int) -> None:
        try:
            while True:
                item = await self._q.get()
                if item is None:
                    return

                try:
                    outcome = await process_page_result(
                        page_result=item,
                        company=self.company,
                        guard=self.guard,
                        gating_cfg=self.gating_cfg,
                        timeout_error_marker=self.timeout_error_marker,
                        mark_company_timeout_cb=self.mark_company_timeout_cb,
                        mark_company_memory_cb=self.mark_company_memory_cb,
                        url_index_sink=self._writer.enqueue,
                    )
                    with contextlib.suppress(Exception):
                        await self._accumulate(outcome)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # Treat CriticalMemoryPressure (defined elsewhere) as fatal by name.
                    if e.__class__.__name__ == "CriticalMemoryPressure":
                        self._fatal = e
                        return
                    logger.exception(
                        "Error processing page_result (company=%s worker=%d): %s",
                        self.company_id,
                        worker_idx,
                        e,
                    )
                finally:
                    self._q.task_done()
        finally:
            return


# =====================================================================
# crawl_runner.py (MERGED + FIXED)
# =====================================================================


class CompanyLike(Protocol):
    company_id: str
    domain_url: str


ProgressCB = Callable[[], Any]  # sync callback is enough (we call it inline)

_GOTO_TIMEOUT_RE = re.compile(r"(page\.goto: timeout|timeout \d+ms exceeded)", re.I)


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


def _string_looks_like_goto_timeout(s: str) -> bool:
    if not s:
        return False
    if _GOTO_TIMEOUT_RE.search(s):
        return True
    # also treat the shared timeout detector as a secondary signal
    return is_timeout_error_message(s)


def _raise_if_fatal_transport(e: BaseException) -> None:
    if is_playwright_driver_disconnect(e):
        raise CrawlerFatalError("Playwright driver disconnected") from e


def _raise_if_goto_timeout(
    e: BaseException, *, company_id: str, url: str, stage: str
) -> None:
    if is_goto_timeout_error(e) or _string_looks_like_goto_timeout(str(e)):
        raise CrawlerTimeoutError(
            f"[GOTO Timeout] {type(e).__name__}: {e}",
            stage="goto",
            company_id=company_id,
            url=url,
        ) from e


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
    on_progress: Optional[ProgressCB] = None,
) -> PagePipelineSummary:
    """
    Runs crawl4ai for a company and streams results into ConcurrentPageResultProcessor.

    Key guarantees:
      - If navigation hits Page.goto timeout (even when wrapped), we RAISE CrawlerTimeoutError(stage='goto').
      - Async generators are always closed (aclose) on early stop / errors.
      - PagePipelineSummary now carries last_error/last_error_url so post-check can see it.
      - IMPORTANT FIX: url_index.json.__meta__.crawl_finished is only set on successful completion.
    """
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

    def _tick_progress() -> None:
        if on_progress is None:
            return
        with contextlib.suppress(Exception):
            on_progress()

    def _hit_hard_cap() -> bool:
        return bool(
            cfg.hard_max_pages is not None and pages_seen >= int(cfg.hard_max_pages)
        )

    async def submit_one(page_result: Any) -> bool:
        nonlocal pages_seen
        if _hit_hard_cap():
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
        return True

    async def _aclose_safely(agen: Any) -> None:
        aclose = getattr(agen, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    aclose(), timeout=float(cfg.generator_close_timeout_sec)
                )

    async def process_result(result_or_gen: Any, *, origin_url: str) -> None:
        # list
        if isinstance(result_or_gen, list):
            for item in result_or_gen:
                ok = await submit_one(item)
                if not ok:
                    break
            return

        # async generator
        if hasattr(result_or_gen, "__anext__"):
            agen = result_or_gen
            try:
                while True:
                    if _hit_hard_cap():
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

            except Exception as e:
                _raise_if_fatal_transport(e)
                _raise_if_goto_timeout(
                    e, company_id=company.company_id, url=origin_url, stage="stream"
                )
                raise
            finally:
                await _aclose_safely(agen)
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

    submitted_any = False
    success = False  # IMPORTANT: only set True once the crawl loop completes normally

    try:
        if cfg.direct_fetch_urls:
            for u in start_urls:
                if _hit_hard_cap():
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
                except Exception as e:
                    _raise_if_fatal_transport(e)
                    _raise_if_goto_timeout(
                        e, company_id=company.company_id, url=u, stage="direct_fetch"
                    )
                    raise

                before = pages_seen
                await process_result(res, origin_url=u)
                submitted_any = submitted_any or (pages_seen > before)

        else:
            for u in start_urls:
                if _hit_hard_cap():
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
                except Exception as e:
                    _raise_if_fatal_transport(e)
                    _raise_if_goto_timeout(
                        e, company_id=company.company_id, url=u, stage="arun_init"
                    )
                    raise

                before = pages_seen
                await process_result(res_or_gen, origin_url=u)
                submitted_any = submitted_any or (pages_seen > before)

        # If we reach here, the crawl loop finished normally.
        success = True

    finally:
        # IMPORTANT FIX:
        # Mark crawl_finished ONLY on success, and BEFORE processor.finish() (so it is flushed).
        if success:
            with contextlib.suppress(Exception):
                await processor.mark_crawl_finished(reason="ok")

        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                processor.finish(), timeout=float(cfg.processor_finish_timeout_sec)
            )
        with contextlib.suppress(Exception):
            gc.collect()

    # IMPORTANT: Now summary includes last_error from page processing.
    try:
        summary = await processor.get_summary()
    except Exception:
        return PagePipelineSummary()

    last_err = (summary.last_error or "").strip()

    # If we got no usable pages, but page_pipeline saw a goto-like timeout error => raise.
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

    # If crawl4ai produced nothing at all, but still no summary error, keep summary as-is.
    # (This case usually means crawl4ai swallowed the exception without yielding a page_result.)
    return summary


def memory_pressure_exception(summary_reason: str) -> RuntimeError:
    return RuntimeError(f"{MEMORY_PRESSURE_MARKER} | {summary_reason}")
