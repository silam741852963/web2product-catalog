from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from extensions import md_gating
from extensions import output_paths
from extensions.connectivity_guard import ConnectivityGuard
from extensions.output_paths import (
    ensure_company_dirs,
    sanitize_bvdid,
    save_stage_output,
)

logger = logging.getLogger("page_pipeline")

# --------------------------------------------------------------------------------------
# Unified failure markers + smarter detection (single source of truth)
# --------------------------------------------------------------------------------------

# This marker is emitted by your crawler layer today; keep as a strong signal.
MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"

# Timeout messages vary across Playwright / Crawl4AI / underlying stacks.
# We detect *families* of timeout texts rather than a single exact string.
_TIMEOUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Crawl4AI style: "Timeout 60000ms exceeded."
    re.compile(r"\btimeout\s+\d+\s*ms\s+exceeded\b", re.IGNORECASE),
    re.compile(r"\btimeout\s+\d+\s*ms\s+exceeded\.\b", re.IGNORECASE),
    # Playwright style: "Navigation timeout of 30000 ms exceeded"
    re.compile(r"\bnavigation\s+timeout\s+of\s+\d+\s*ms\s+exceeded\b", re.IGNORECASE),
    # Generic: "... timed out ..."
    re.compile(r"\btimed\s+out\b", re.IGNORECASE),
    # TimeoutError wrappers
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

# IMPORTANT:
# These patterns are used ONLY to feed ConnectivityGuard's EWMA counters.
# We must be conservative to prevent "false outage" when:
#   - a site is slow / rate-limiting / timing out,
#   - a site returns 5xx,
#   - the event loop is busy and some awaits are delayed.
# Therefore, we only count errors that look like *network/transport* failures.
_TRANSPORT_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    # DNS / resolution
    re.compile(r"\bnxdomain\b", re.IGNORECASE),
    re.compile(r"\bname or service not known\b", re.IGNORECASE),
    re.compile(r"\btemporary failure in name resolution\b", re.IGNORECASE),
    re.compile(r"\bno address associated with hostname\b", re.IGNORECASE),
    re.compile(r"\bgetaddrinfo\b", re.IGNORECASE),
    re.compile(r"\bdns\b", re.IGNORECASE),
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

# Certain HTTP status codes are often "transport-ish" (gateway/proxy issues),
# but still can be site-specific. We'll count them as transport ONLY when paired
# with a transport-like error string OR when there's no usable content at all.
_TRANSPORT_LIKE_HTTP = {502, 503, 504, 520, 521, 522, 523, 524}


def is_timeout_error_message(error: str, marker: Optional[str] = None) -> bool:
    """
    Smarter timeout detection:
      - optional legacy substring marker (if someone still passes one)
      - pattern-based matching for common timeout families
    """
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
    """
    Conservative "transport" detector for ConnectivityGuard EWMA signal.

    We intentionally do NOT treat generic timeouts as transport errors, because:
      - slow/blocked sites are common,
      - event-loop starvation can delay awaits,
      - and ConnectivityGuard already performs independent TCP probes.

    So: timeouts => NOT counted here (let probes decide).
    """
    if not error:
        return False

    # Exclude timeout family from guard's transport signal to reduce false outage.
    if is_timeout_error_message(error):
        return False

    e = error
    for rx in _TRANSPORT_ERROR_PATTERNS:
        try:
            if rx.search(e):
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
    """
    Feed ConnectivityGuard with *conservative* signals to avoid false internet outage.

    Policy:
      - Count "transport errors" only when they look like DNS/TCP/routing failures.
      - Do NOT count generic timeouts as transport errors (too noisy / site-specific / event-loop delays).
      - Do NOT count ordinary HTTP 5xx as transport errors (often site-specific).
      - Count "success" when we have a concrete HTTP response (any status code), OR when no error.
      - For 5xx with no transport-like error string, count as success (still means connectivity exists).
    """
    try:
        code_int = int(status_code) if status_code is not None else None
    except Exception:
        code_int = None

    transport_like = _looks_transport_error_for_guard(error_str)

    if transport_like:
        guard.record_transport_error()
        return

    # If we got an HTTP response at all (even 404/500), connectivity exists.
    if code_int is not None:
        guard.record_success()
        return

    # No status code: if there's no error text, count as success; otherwise be neutral
    # (do not push error ratio up unless it's transport-like).
    if not error_str.strip():
        guard.record_success()
        return

    # Neutral: do nothing


# --------------------------------------------------------------------------------------
# Small, fast atomic JSON write (local, because we want batched url_index writing here)
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
        meta_dir = Path(output_paths.OUTPUT_ROOT) / safe / "metadata"
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir / "url_index.json"


# --------------------------------------------------------------------------------------
# Batched url_index writer (single writer task per company)
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class UrlIndexFlushConfig:
    flush_every: int = 64
    flush_interval_sec: float = 1.0
    queue_maxsize: int = 8192


class UrlIndexBatchWriter:
    """
    Single-writer batching for url_index.json:
      - load once
      - merge per-url patches in memory
      - flush periodically or every N updates
    """

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
# Per-company summary (this is what run.py can use to decide retry class)
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class PagePipelineSummary:
    total_pages: int = 0
    markdown_saved: int = 0
    markdown_suppressed: int = 0
    timeout_pages: int = 0
    memory_pressure_pages: int = 0

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
        }


@dataclass(slots=True)
class PageProcessOutcome:
    url: str
    md_saved: bool
    md_suppressed: bool
    timeout_exceeded: bool
    memory_pressure: bool


# --------------------------------------------------------------------------------------
# Page result processing
# --------------------------------------------------------------------------------------

# NOTE: We intentionally do NOT call company-level retry callbacks here.
# Per-page timeouts/memory pressure are URL-level noise and should stay in url_index only.
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
    """
    Per page processing.

    IMPORTANT POLICY:
      - We do NOT promote per-page timeout/memory pressure into company-level retry callbacks.
        These are recorded in url_index only; company retry classification happens at pipeline boundaries.
    """
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

    # ConnectivityGuard update (FIX: avoid false "internet outage")
    if guard is not None:
        try:
            _update_connectivity_guard_from_page(
                guard,
                error_str=error_str,
                status_code=status_code,
            )
        except Exception:
            # Guard should never break page processing.
            pass

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
        from extensions.crawl_state import upsert_url_index_entry

        await asyncio.to_thread(upsert_url_index_entry, company_id, url, entry)

    return PageProcessOutcome(
        url=url,
        md_saved=(md_path is not None and md_status == "markdown_saved"),
        md_suppressed=(md_status == "markdown_suppressed"),
        timeout_exceeded=timeout_exceeded,
        memory_pressure=memory_pressure,
    )


# --------------------------------------------------------------------------------------
# Concurrent page pipeline
# --------------------------------------------------------------------------------------


class ConcurrentPageResultProcessor:
    """
    Feed page_result objects via submit(); workers process concurrently and enqueue url_index patches
    into a single UrlIndexBatchWriter.

    Backpressure:
      - page queue is bounded (prevents holding too many huge page_result objects in memory)
    """

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

        # kept for signature compatibility, but intentionally unused by process_page_result
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

        # summary state (read by run.py via crawl_company return)
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
                    # update summary for run.py
                    try:
                        await self._accumulate(outcome)
                    except Exception:
                        pass

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
