from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Awaitable, Tuple

from extensions.connectivity_guard import ConnectivityGuard
from extensions import md_gating
from extensions import output_paths
from extensions.output_paths import (
    save_stage_output,
    ensure_company_dirs,
    sanitize_bvdid,
)

logger = logging.getLogger("page_pipeline")

MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"

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
    # compact JSON => much faster & smaller for big url_index.json
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
        self._stop = asyncio.Event()

    async def start(self) -> None:
        # Load once (thread offload for big files)
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
        # Patch objects are small; allow backpressure if queue is full
        await self._q.put((url, patch))

    async def close(self) -> None:
        if self._task is None:
            return
        # sentinel
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
                # periodic flush when dirty
                if self._dirty > 0:
                    try:
                        await self._flush()
                    except Exception:
                        logger.exception(
                            "url_index flush failed (company=%s)", self.company_id
                        )
                continue

            if item is None:
                # final flush
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
# Page result processing (now async-friendly: offloads disk I/O; supports batched writer)
# --------------------------------------------------------------------------------------


async def _maybe_call(cb: Optional[Callable[[str], None]], company_id: str) -> None:
    if cb is None:
        return
    try:
        cb(company_id)
    except Exception:
        logger.exception("company callback failed (company=%s)", company_id)


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
    timeout_error_marker: str,
    mark_company_timeout_cb: Optional[Callable[[str], None]] = None,
    mark_company_memory_cb: Optional[Callable[[str], None]] = None,
    # NEW: if provided, we do NOT touch url_index.json directly; we enqueue patches.
    url_index_sink: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
) -> None:
    """
    Per page processing.

    PERF changes:
      - No longer calls guard.wait_until_healthy() (page processing does not perform requests).
      - Disk writes are offloaded via asyncio.to_thread.
      - url_index updates can be batched via url_index_sink.
    """
    company_id = getattr(company, "company_id", None)
    if not company_id:
        logger.warning(
            "process_page_result called without company_id on company=%r", company
        )
        return

    _getattr = getattr
    requested_url = _getattr(page_result, "url", None)
    final_url = _getattr(page_result, "final_url", None) or requested_url
    if not final_url:
        logger.warning(
            "Page result missing URL; skipping entry (company_id=%s)", company_id
        )
        return
    url = final_url

    markdown = _getattr(page_result, "markdown", None)
    status_code = _getattr(page_result, "status_code", None)
    error = _getattr(page_result, "error", None) or _getattr(
        page_result, "error_message", None
    )

    error_str = error if isinstance(error, str) else ""
    timeout_exceeded = bool(error_str and timeout_error_marker in error_str)
    memory_pressure = bool(error_str and MEMORY_PRESSURE_MARKER in error_str)

    if timeout_exceeded:
        await _maybe_call(mark_company_timeout_cb, company_id)
    if memory_pressure:
        await _maybe_call(mark_company_memory_cb, company_id)

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
        md_path = await _save_stage_threaded(
            bvdid=company_id, url=url, data=markdown, stage="markdown"
        )
        md_status = "markdown_saved" if md_path is not None else "markdown_error"
    else:
        md_status = "markdown_suppressed"

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
        # Fallback (slower): immediate write. Keep it thread-offloaded to avoid blocking loop.
        from extensions.crawl_state import upsert_url_index_entry

        await asyncio.to_thread(upsert_url_index_entry, company_id, url, entry)


# --------------------------------------------------------------------------------------
# Concurrent page pipeline (bounded worker pool + backpressure)
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
        timeout_error_marker: str,
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
        self.mark_company_timeout_cb = mark_company_timeout_cb
        self.mark_company_memory_cb = mark_company_memory_cb

        self.concurrency = max(1, int(concurrency))
        maxsize = (
            int(page_queue_maxsize)
            if page_queue_maxsize and page_queue_maxsize > 0
            else max(16, self.concurrency * 2)
        )
        self._q: asyncio.Queue[Optional[Any]] = asyncio.Queue(maxsize=maxsize)

        self._workers: List[asyncio.Task] = []
        self._fatal: Optional[BaseException] = None

        flush_cfg = url_index_flush_cfg or UrlIndexFlushConfig()
        self._writer = UrlIndexBatchWriter(self.company_id, cfg=flush_cfg)

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

        # Put with periodic checks to avoid deadlock if a fatal error happens while queue is full.
        while True:
            if self._fatal is not None:
                raise self._fatal
            try:
                self._q.put_nowait(page_result)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0.05)

    async def finish(self) -> None:
        # Signal workers
        for _ in self._workers:
            await self._q.put(None)

        # Wait workers
        try:
            await asyncio.gather(*self._workers, return_exceptions=False)
        finally:
            self._workers.clear()
            await self._writer.close()

        if self._fatal is not None:
            raise self._fatal

    async def _worker_loop(self, worker_idx: int) -> None:
        try:
            while True:
                item = await self._q.get()
                if item is None:
                    return

                try:
                    await process_page_result(
                        page_result=item,
                        company=self.company,
                        guard=self.guard,
                        gating_cfg=self.gating_cfg,
                        timeout_error_marker=self.timeout_error_marker,
                        mark_company_timeout_cb=self.mark_company_timeout_cb,
                        mark_company_memory_cb=self.mark_company_memory_cb,
                        url_index_sink=self._writer.enqueue,
                    )
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
            # Ensure we don't leave q.task_done unbalanced (best-effort)
            return
