from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, List

from crawl4ai import AsyncWebCrawler

from extensions.schedule.retry import (
    CriticalMemoryPressure,
    CrawlerFatalError,
    is_playwright_driver_disconnect,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass(slots=True)
class _CrawlerSlot:
    idx: int
    crawler: AsyncWebCrawler
    processed_companies: int = 0


class CrawlerPool:
    """
    A simple async pool of AsyncWebCrawler instances.
    - Recycles a crawler after N companies (recycle_after_companies)
    - Recycles immediately on fatal signals (driver disconnect / memory pressure)
    """

    def __init__(
        self, *, browser_cfg: Any, size: int, recycle_after_companies: int
    ) -> None:
        self._browser_cfg = browser_cfg
        self._size = max(1, int(size))
        self._recycle_after = max(0, int(recycle_after_companies))
        self._q: asyncio.Queue[_CrawlerSlot] = asyncio.Queue()
        self._all: List[_CrawlerSlot] = []
        self._closing = False

    @property
    def size(self) -> int:
        return self._size

    def free_slots_approx(self) -> int:
        return self._q.qsize()

    async def start(self) -> None:
        created: List[_CrawlerSlot] = []
        for i in range(self._size):
            c = AsyncWebCrawler(config=self._browser_cfg)
            await c.__aenter__()
            created.append(_CrawlerSlot(idx=i, crawler=c))
        self._all = created
        for s in created:
            self._q.put_nowait(s)
        logger.info(
            "[CrawlerPool] started size=%d recycle_after=%d",
            self._size,
            self._recycle_after,
        )

    async def stop(self) -> None:
        self._closing = True
        for s in self._all:
            with contextlib.suppress(Exception):
                await s.crawler.__aexit__(None, None, None)
        self._all.clear()
        logger.info("[CrawlerPool] stopped")

    async def _restart(self, slot: _CrawlerSlot, *, reason: str) -> _CrawlerSlot:
        with contextlib.suppress(Exception):
            await slot.crawler.__aexit__(None, None, None)
        c = AsyncWebCrawler(config=self._browser_cfg)
        await c.__aenter__()
        slot.crawler = c
        slot.processed_companies = 0
        logger.warning("[CrawlerPool] restarted slot=%d reason=%s", slot.idx, reason)
        return slot

    async def lease(self) -> "_CrawlerLease":
        slot = await self._q.get()
        return _CrawlerLease(pool=self, slot=slot)

    async def _release(self, slot: _CrawlerSlot, *, fatal: bool, reason: str) -> None:
        if self._closing:
            with contextlib.suppress(Exception):
                await slot.crawler.__aexit__(None, None, None)
            return

        if fatal:
            slot = await self._restart(slot, reason=reason)
        else:
            slot.processed_companies += 1
            if self._recycle_after and slot.processed_companies >= self._recycle_after:
                slot = await self._restart(slot, reason="periodic_recycle")

        with contextlib.suppress(Exception):
            self._q.put_nowait(slot)


class _CrawlerLease:
    def __init__(self, *, pool: CrawlerPool, slot: _CrawlerSlot) -> None:
        self._pool = pool
        self._slot = slot
        self._fatal = False
        self._reason = "ok"

    def mark_fatal(self, reason: str) -> None:
        self._fatal = True
        self._reason = reason or "fatal"

    async def __aenter__(self) -> AsyncWebCrawler:
        return self._slot.crawler

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # cancellations should recycle (safer: avoid stuck state)
        if exc_type is asyncio.CancelledError:
            self._fatal = True
            self._reason = "cancelled"

        if exc is not None:
            if isinstance(exc, CriticalMemoryPressure):
                self._fatal = True
                self._reason = "memory_pressure"
            if isinstance(exc, CrawlerFatalError) or is_playwright_driver_disconnect(
                exc
            ):
                self._fatal = True
                self._reason = "driver_disconnect"

        await asyncio.shield(
            self._pool._release(self._slot, fatal=self._fatal, reason=self._reason)
        )
