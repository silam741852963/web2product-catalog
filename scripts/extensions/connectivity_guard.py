from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

# Small helper to parse host:port strings if desired when passing probe_endpoints
from urllib.parse import urlparse


@dataclass
class _Cfg:
    # probe target (host) and port (we use TCP connect to determine reachability)
    probe_host: str = "8.8.8.8"
    probe_port: int = 53

    # how often (seconds) to run a probe cycle
    interval_s: float = 5.0

    # number of consecutive failing probe cycles before tripping to OPEN
    trip_heartbeats: int = 3

    # when OPEN, base cooloff before HALF_OPEN probe (exponential backoff applied)
    base_cooloff_s: float = 5.0
    max_cooloff_s: float = 300.0
    backoff_factor: float = 2.0

    # small random jitter added to cooloff to prevent thundering herd
    jitter_s: float = 0.25

    # timeout for the TCP connect probe
    connect_timeout_s: float = 2.5


class ConnectivityGuard:
    """
    Simplified connectivity guard.

    Behavior:
      - Repeatedly attempts a TCP connection to (probe_host, probe_port).
      - If `trip_heartbeats` consecutive probe failures occur, transitions to OPEN.
      - While OPEN, waits an exponentially backed-off cooloff before a HALF_OPEN probe.
      - A successful probe immediately transitions to CLOSED (network OK).
      - Only probe results (connectivity to probe target) decide the state.

    Public API kept intentionally compatible with previous guard:
      - start(), stop()
      - is_healthy(), wait_until_healthy()
      - record_transport_error(), record_success() (these are kept as no-ops for compatibility)
      - snapshot_open_events(), state()
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        probe_endpoints: Iterable[str] | None = None,
        *,
        probe_host: str | None = None,
        probe_port: int | None = None,
        interval_s: float = 5.0,
        trip_heartbeats: int = 3,
        connect_timeout_s: float = 2.5,
        base_cooloff_s: float = 5.0,
        max_cooloff_s: float = 300.0,
        backoff_factor: float = 2.0,
        jitter_s: float = 0.25,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        probe_endpoints: optional legacy list of endpoints (like "8.8.8.8:53" or "https://host")
                         If provided, the first parsed host:port will be used.
                         probe_host/port override the parsed values.
        probe_host/probe_port: explicit host/port override.
        """

        # choose defaults and allow override
        ph = probe_host or "8.8.8.8"
        pp = int(probe_port or 53)

        # if a probe_endpoints iterable is provided, try to parse the first usable host:port
        if probe_endpoints:
            for e in probe_endpoints:
                try:
                    s = str(e).strip()
                    if not s:
                        continue
                    # try simple "host:port"
                    if ":" in s and not s.startswith("http"):
                        host_part, port_part = s.rsplit(":", 1)
                        ph = host_part.strip() or ph
                        try:
                            pp = int(port_part)
                        except Exception:
                            pass
                        break
                    # try URL parse (e.g., https://host)
                    if s.startswith("http://") or s.startswith("https://"):
                        up = urlparse(s)
                        if up.hostname:
                            ph = up.hostname
                            if up.port:
                                pp = up.port
                            else:
                                pp = 443 if up.scheme == "https" else 80
                            break
                    # fallback: treat as host only
                    ph = s
                    break
                except Exception:
                    continue

        self.cfg = _Cfg(
            probe_host=ph,
            probe_port=pp,
            interval_s=interval_s,
            trip_heartbeats=trip_heartbeats,
            connect_timeout_s=connect_timeout_s,
            base_cooloff_s=base_cooloff_s,
            max_cooloff_s=max_cooloff_s,
            backoff_factor=backoff_factor,
            jitter_s=jitter_s,
        )

        self._state = self.CLOSED
        self._hb_fail_streak = 0  # consecutive failing probe cycles
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._healthy_event = asyncio.Event()
        self._healthy_event.set()  # initially healthy
        self._cooloff_until: float = 0.0

        # counts for backoff and diagnostics
        self._open_count: int = 0
        self._open_seq_total: int = 0

        # small lock used for safe transitions
        self._lock = asyncio.Lock()
        self._logger = logger or logging.getLogger("connectivity_guard")

        # Manual legacy counters (kept for compatibility but not used for open/close decision)
        self._manual_recent_successes = 0
        self._manual_recent_failures = 0

    # ---------- public API ----------

    async def start(self) -> None:
        """
        Start the background probe loop.
        """
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run())
            self._logger.debug("ConnectivityGuard started (probe=%s:%d).", self.cfg.probe_host, self.cfg.probe_port)

    async def stop(self) -> None:
        """
        Stop the background probe loop and wait for it to finish.
        """
        self._stop.set()
        if self._task:
            with contextlib.suppress(Exception):
                await self._task
        self._logger.debug("ConnectivityGuard stopped.")

    def is_healthy(self) -> bool:
        """
        True when state == CLOSED (network OK).
        """
        return self._state == self.CLOSED

    async def wait_until_healthy(self) -> None:
        """
        Await until the guard considers the network healthy (CLOSED).
        """
        while not self._healthy_event.is_set():
            await asyncio.sleep(0.1)

    def record_transport_error(self) -> None:
        """
        Legacy compatibility: record that caller observed a transport error.
        This simplified guard ignores these manual reports for decision-making;
        they are only recorded for diagnostics (kept very lightweight).
        """
        self._manual_recent_failures += 1
        # keep the number bounded
        if self._manual_recent_failures > 1_000_000:
            self._manual_recent_failures = 1_000_000
        # Debug log
        self._logger.debug("ConnectivityGuard.record_transport_error() called (ignored for probe decision).")

    def record_success(self) -> None:
        """
        Legacy compatibility: record that caller observed success.
        No impact on probe-based decision; recorded for diagnostics only.
        """
        self._manual_recent_successes += 1
        if self._manual_recent_successes > 1_000_000:
            self._manual_recent_successes = 1_000_000
        self._logger.debug("ConnectivityGuard.record_success() called (ignored for probe decision).")

    def snapshot_open_events(self) -> int:
        """
        Cumulative count of times the guard transitioned to OPEN.
        """
        return self._open_seq_total

    def state(self) -> str:
        return self._state

    # ---------- internals ----------

    def _compute_cooloff_s(self) -> float:
        """
        Exponential backoff with jitter for OPEN state cooloff period.
        """
        base = self.cfg.base_cooloff_s * (self.cfg.backoff_factor ** self._open_count)
        base = min(base, self.cfg.max_cooloff_s)
        return base + random.uniform(0.0, self.cfg.jitter_s)

    def _transition_to(self, new_state: str) -> None:
        """
        Internal state transition with side-effects (events, logging).
        """
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state

        if new_state == self.CLOSED:
            # reset fail streak / open counter
            self._hb_fail_streak = 0
            self._open_count = 0
            self._healthy_event.set()
            self._logger.info("ConnectivityGuard: %s -> CLOSED (network OK).", old.upper())

        elif new_state == self.OPEN:
            self._healthy_event.clear()
            self._open_count += 1
            self._open_seq_total += 1
            cool = self._compute_cooloff_s()
            self._cooloff_until = time.time() + cool
            self._logger.warning(
                "ConnectivityGuard: %s -> OPEN (pausing %.1fs before re-probe; backoff #%d).",
                old.upper(), cool, self._open_count
            )

        elif new_state == self.HALF_OPEN:
            self._healthy_event.clear()
            self._logger.info("ConnectivityGuard: %s -> HALF_OPEN (probing).", old.upper())

    async def _tcp_probe(self, host: str, port: int, timeout: float) -> bool:
        """
        Attempt an asyncio TCP connect to host:port within timeout.
        Returns True if connect succeeded, False otherwise.
        """
        try:
            fut = asyncio.open_connection(host=host, port=port, family=0)
            r, w = await asyncio.wait_for(fut, timeout=timeout)
            # close writer gracefully
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            return True
        except Exception as e:
            self._logger.debug("ConnectivityGuard._tcp_probe failed: %s:%d err=%s", host, port, repr(e))
            return False

    async def _probe_once(self) -> bool:
        """
        Probe the configured probe_host:probe_port once.
        Return True if reachable (probe succeeded), False otherwise.
        """
        return await self._tcp_probe(self.cfg.probe_host, self.cfg.probe_port, timeout=self.cfg.connect_timeout_s)

    async def _run(self) -> None:
        """
        Background loop that periodically probes the target and sets OPEN/CLOSED.
        Only probe results control the state machine.
        """
        while not self._stop.is_set():
            try:
                # If currently OPEN and cooloff not expired, wait a bit (but loop again to allow stop)
                if self._state == self.OPEN:
                    remaining = self._cooloff_until - time.time()
                    if remaining > 0:
                        # sleep for min(remaining, interval) so we wake periodically and respect stop event
                        await asyncio.sleep(min(remaining, max(0.1, self.cfg.interval_s)))
                        continue
                    # transition to HALF_OPEN and perform a probe immediately
                    self._transition_to(self.HALF_OPEN)

                # perform a probe
                ok = await self._probe_once()

                if ok:
                    # successful probe -> immediate CLOSED
                    if self._state != self.CLOSED:
                        self._logger.debug("ConnectivityGuard probe succeeded; transitioning to CLOSED.")
                    self._transition_to(self.CLOSED)
                else:
                    # probe failed -> increment consecutive fail streak
                    self._hb_fail_streak += 1
                    self._logger.debug("ConnectivityGuard probe failed (streak=%d/%d).",
                                       self._hb_fail_streak, self.cfg.trip_heartbeats)
                    # If fail streak reached threshold, go OPEN
                    if self._state == self.CLOSED and self._hb_fail_streak >= max(1, int(self.cfg.trip_heartbeats)):
                        self._transition_to(self.OPEN)
                    elif self._state == self.HALF_OPEN:
                        # when half-open and probe failed, go back to OPEN and compute cooloff
                        self._transition_to(self.OPEN)

                # pause until next probe cycle (if not in OPEN waiting period)
                # If OPEN, the loop will compute remaining and sleep at top of loop.
                await asyncio.sleep(self.cfg.interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Unexpected exceptions in the probe loop should not kill the loop.
                self._logger.exception("ConnectivityGuard probe loop exception: %s", e)
                # Conservative behavior: treat this as a single failed probe
                self._hb_fail_streak += 1
                if self._hb_fail_streak >= max(1, int(self.cfg.trip_heartbeats)):
                    self._transition_to(self.OPEN)
                await asyncio.sleep(max(0.1, self.cfg.interval_s))

        # cleanup on exit
        self._logger.debug("ConnectivityGuard._run() exiting.")