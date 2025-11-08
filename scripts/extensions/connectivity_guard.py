from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import socket
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Tuple
from urllib.parse import urlparse


@dataclass
class _Cfg:
    interval_s: float = 5.0
    trip_heartbeats: int = 3
    error_ratio_threshold: float = 0.8
    error_window: int = 50
    connect_timeout_s: float = 2.5
    base_cooloff_s: float = 5.0
    max_cooloff_s: float = 300.0
    backoff_factor: float = 2.0
    jitter_s: float = 0.25


def _to_host_port(endpoint: str) -> Tuple[str, int]:
    if "://" in endpoint:
        u = urlparse(endpoint)
        host = u.hostname or "example.com"
        port = u.port or (443 if u.scheme == "https" else 80)
        return host, port
    if ":" in endpoint:
        host, port = endpoint.rsplit(":", 1)
        try:
            return host, int(port)
        except Exception:
            return endpoint, 443
    return endpoint, 443


class ConnectivityGuard:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        probe_endpoints: Iterable[str] = (
            "https://www.gstatic.com/generate_204",
            "http://example.com",
            "1.1.1.1:443",
        ),
        *,
        interval_s: float = 5.0,
        trip_heartbeats: int = 3,
        error_ratio_threshold: float = 0.8,
        error_window: int = 50,
        connect_timeout_s: float = 2.5,
        base_cooloff_s: float = 5.0,
        max_cooloff_s: float = 300.0,
        backoff_factor: float = 2.0,
        jitter_s: float = 0.25,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = _Cfg(
            interval_s=interval_s,
            trip_heartbeats=trip_heartbeats,
            error_ratio_threshold=error_ratio_threshold,
            error_window=error_window,
            connect_timeout_s=connect_timeout_s,
            base_cooloff_s=base_cooloff_s,
            max_cooloff_s=max_cooloff_s,
            backoff_factor=backoff_factor,
            jitter_s=jitter_s,
        )
        self._endpoints = [_to_host_port(e) for e in probe_endpoints]
        self._state = self.CLOSED
        self._hb_fail_streak = 0
        self._recent: deque[bool] = deque(maxlen=self.cfg.error_window)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._healthy_event = asyncio.Event()
        self._healthy_event.set()
        self._cooloff_until: float = 0.0

        # Backoff counter (resets on CLOSED) + cumulative OPEN counter (never resets)
        self._open_count: int = 0
        self._open_seq_total: int = 0

        self._lock = asyncio.Lock()
        self._logger = logger or logging.getLogger("run_crawl.net")

    # ---------- public API ----------

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run())
            self._logger.debug("ConnectivityGuard started.")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            with contextlib.suppress(Exception):
                await self._task
        self._logger.debug("ConnectivityGuard stopped.")

    def is_healthy(self) -> bool:
        return self._state == self.CLOSED

    async def wait_until_healthy(self) -> None:
        while not self._healthy_event.is_set():
            await asyncio.sleep(0.1)

    def record_transport_error(self) -> None:
        self._recent.append(False)
        self._maybe_trip_from_ratio()

    def record_success(self) -> None:
        self._recent.append(True)

    # NEW: snapshot counters/state for “did it open during my section?”
    def snapshot_open_events(self) -> int:
        return self._open_seq_total

    def state(self) -> str:
        return self._state

    # ---------- internals ----------

    def _maybe_trip_from_ratio(self) -> None:
        if not self._recent:
            return
        fail = self._recent.count(False)
        ratio = fail / max(1, len(self._recent))
        if ratio >= self.cfg.error_ratio_threshold:
            self._logger.warning(
                "ConnectivityGuard: error ratio %.2f >= %.2f (window=%d) → OPEN",
                ratio, self.cfg.error_ratio_threshold, len(self._recent),
            )
            self._transition_to(self.OPEN)

    def _compute_cooloff_s(self) -> float:
        base = self.cfg.base_cooloff_s * (self.cfg.backoff_factor ** self._open_count)
        base = min(base, self.cfg.max_cooloff_s)
        return base + random.uniform(0.0, self.cfg.jitter_s)

    def _transition_to(self, new_state: str) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state

        if new_state == self.CLOSED:
            self._hb_fail_streak = 0
            self._open_count = 0
            self._healthy_event.set()
            self._logger.info("ConnectivityGuard: %s → CLOSED (network OK).", old.upper())

        elif new_state == self.OPEN:
            self._healthy_event.clear()
            self._open_count += 1
            self._open_seq_total += 1  # cumulative
            cool = self._compute_cooloff_s()
            self._cooloff_until = time.time() + cool
            self._logger.warning(
                "ConnectivityGuard: %s → OPEN (pausing %.1fs before re-probe; backoff #%d).",
                old.upper(), cool, self._open_count,
            )

        elif new_state == self.HALF_OPEN:
            self._healthy_event.clear()
            self._logger.info("ConnectivityGuard: %s → HALF_OPEN (probing).", old.upper())

    async def _probe_once(self) -> bool:
        results = await asyncio.gather(
            *[self._tcp_connect_ok(h, p, self.cfg.connect_timeout_s) for (h, p) in self._endpoints],
            return_exceptions=True,
        )
        return any(isinstance(r, bool) and r for r in results)

    async def _tcp_connect_ok(self, host: str, port: int, timeout: float) -> bool:
        try:
            fut = asyncio.open_connection(host=host, port=port, family=socket.AF_UNSPEC)
            r, w = await asyncio.wait_for(fut, timeout=timeout)
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            return True
        except Exception:
            return False

    async def _run(self) -> None:
        while not self._stop.is_set():
            if self._state == self.OPEN:
                remaining = self._cooloff_until - time.time()
                if remaining > 0:
                    await asyncio.sleep(min(remaining, self.cfg.interval_s))
                    continue
                self._transition_to(self.HALF_OPEN)

            ok = await self._probe_once()
            if ok:
                self._hb_fail_streak = 0
                if self._state in (self.OPEN, self.HALF_OPEN):
                    self._transition_to(self.CLOSED)
            else:
                self._hb_fail_streak += 1
                if self._state == self.CLOSED and self._hb_fail_streak >= self.cfg.trip_heartbeats:
                    self._transition_to(self.OPEN)
                elif self._state == self.HALF_OPEN:
                    self._transition_to(self.OPEN)

            await asyncio.sleep(self.cfg.interval_s)