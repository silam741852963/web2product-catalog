from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple
from urllib.parse import urlparse


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ConnectivityGuardConfig:
    """
    Configuration for ConnectivityGuard.

    Multiple probe targets; success if ANY succeed, fail if ALL fail.
    """

    probe_targets: List[Tuple[str, int]] = field(
        default_factory=lambda: [
            ("1.1.1.1", 443),
            ("8.8.8.8", 443),
            ("9.9.9.9", 443),
        ]
    )

    # Probing cadence
    interval_s: float = 15.0
    trip_heartbeats: int = 4  # consecutive ALL-fail rounds to consider outage

    # IMPORTANT: this is the *socket connect* timeout; keep >= 4s on busy droplets.
    connect_timeout_s: float = 4.0

    # Hysteresis / backoff
    min_open_s: float = 8.0  # hold OPEN at least this long
    min_closed_s: float = 3.0  # after closing, ignore brief fail bursts
    base_cooloff_s: float = 8.0
    max_cooloff_s: float = 180.0
    backoff_factor: float = 2.0
    jitter_s: float = 0.35

    # Hybrid decision with crawler signals (decayed window)
    ewma_half_life_s: float = 30.0  # half-life for manual counters
    err_ratio_trip: float = 0.85  # require high transport error ratio to OPEN
    err_ratio_floor_samples: float = 10  # min effective samples before ratio considered

    # Logging control for low-level probe failures
    log_probe_failures: bool = False


def _parse_probe_endpoints(
    probe_endpoints: Iterable[str] | None,
    probe_host: str | None,
    probe_port: int | None,
) -> List[Tuple[str, int]]:
    """
    Build the final list of (host, port) probe targets from various input forms.
    """
    targets: List[Tuple[str, int]] = []

    if probe_endpoints:
        for e in probe_endpoints:
            s = str(e).strip()
            if not s:
                continue

            # URL-style input
            if s.startswith("http://") or s.startswith("https://"):
                up = urlparse(s)
                if up.hostname:
                    targets.append(
                        (up.hostname, up.port or (443 if up.scheme == "https" else 80))
                    )
                continue

            # host:port form
            if ":" in s and not s.lower().startswith(("tcp://", "udp://")):
                host, port = s.rsplit(":", 1)
                try:
                    targets.append((host.strip(), int(port)))
                    continue
                except Exception:
                    pass

            # fallback: bare host, assume 443
            targets.append((s, 443))

    # single host/port override
    if probe_host:
        targets = [(probe_host, int(probe_port or 443))]

    if not targets:
        targets = list(ConnectivityGuardConfig().probe_targets)

    return targets


# --------------------------------------------------------------------------- #
# ConnectivityGuard
# --------------------------------------------------------------------------- #


class ConnectivityGuard:
    """
    Robust, low-false-positive connectivity guard.

    Decision to OPEN (pause) requires:
      1) ALL configured probe targets fail for `trip_heartbeats` consecutive rounds, AND
      2) recent crawler transport error ratio is high (>= err_ratio_trip),
    OR
      3) probes keep failing long enough (>= 2 * trip_heartbeats), regardless of crawler signals.

    HALF_OPEN is a transient probing state and **does not block callers**.
    Only OPEN causes `wait_until_healthy()` to block.
    """

    __slots__ = (
        "cfg",
        "targets",
        "_state",
        "_open_seq_total",
        "_hb_allfail_streak",
        "_healthy_event",
        "_stop",
        "_task",
        "_logger",
        "_last_open_ts",
        "_last_closed_ts",
        "_ew_ok",
        "_ew_err",
        "_last_decay_ts",
        "_open_backoff_stage",
        "_cooloff_until",
    )

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        probe_endpoints: Iterable[str] | None = None,
        *,
        probe_host: str | None = None,
        probe_port: int | None = None,
        interval_s: float = 15.0,
        trip_heartbeats: int = 4,
        connect_timeout_s: float = 4.0,
        base_cooloff_s: float = 8.0,
        max_cooloff_s: float = 180.0,
        backoff_factor: float = 2.0,
        jitter_s: float = 0.35,
        min_open_s: float = 8.0,
        min_closed_s: float = 3.0,
        ewma_half_life_s: float = 30.0,
        err_ratio_trip: float = 0.85,
        err_ratio_floor_samples: float = 10.0,
        log_probe_failures: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = ConnectivityGuardConfig(
            interval_s=float(interval_s),
            trip_heartbeats=int(trip_heartbeats),
            connect_timeout_s=float(connect_timeout_s),
            base_cooloff_s=float(base_cooloff_s),
            max_cooloff_s=float(max_cooloff_s),
            backoff_factor=float(backoff_factor),
            jitter_s=float(jitter_s),
            min_open_s=float(min_open_s),
            min_closed_s=float(min_closed_s),
            ewma_half_life_s=float(ewma_half_life_s),
            err_ratio_trip=float(err_ratio_trip),
            err_ratio_floor_samples=int(err_ratio_floor_samples),
            log_probe_failures=bool(log_probe_failures)
            if log_probe_failures is not None
            else False,
        )

        targets = _parse_probe_endpoints(probe_endpoints, probe_host, probe_port)
        self.cfg.probe_targets = targets
        self.targets: Tuple[Tuple[str, int], ...] = tuple(targets)

        self._state: str = self.CLOSED
        self._open_seq_total: int = 0
        self._hb_allfail_streak: int = 0

        self._healthy_event = asyncio.Event()
        self._healthy_event.set()

        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._logger = logger or logging.getLogger("connectivity_guard")

        now = time.time()
        self._last_open_ts: float = 0.0
        self._last_closed_ts: float = now

        # EWMA manual counters (optional: caller may feed in transport signal)
        self._ew_ok: float = 1.0
        self._ew_err: float = 0.0
        self._last_decay_ts: float = now

        self._open_backoff_stage: int = 0
        self._cooloff_until: float = 0.0

    # ---------- public API ----------

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="connectivity-guard")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            with contextlib.suppress(Exception):
                await self._task
        self._logger.debug("ConnectivityGuard stopped.")

    def is_healthy(self) -> bool:
        return self._state == self.CLOSED

    async def wait_until_healthy(self) -> None:
        """
        Block only when state is OPEN.
        HALF_OPEN is probing and should not block workers.
        """
        while self._state == self.OPEN and not self._healthy_event.is_set():
            await asyncio.sleep(0.1)

    def record_transport_error(self) -> None:
        self._decay_ewma()
        self._ew_err += 1.0
        self._cap_ewma()

    def record_success(self) -> None:
        self._decay_ewma()
        self._ew_ok += 1.0
        self._cap_ewma()

    def snapshot_open_events(self) -> int:
        return self._open_seq_total

    def state(self) -> str:
        return self._state

    # ---------- internals: EWMA ----------

    def _cap_ewma(self) -> None:
        if self._ew_ok > 1e6 or self._ew_err > 1e6:
            scale = 1e-6
            self._ew_ok *= scale
            self._ew_err *= scale

    def _decay_ewma(self) -> None:
        now = time.time()
        dt = max(0.0, now - self._last_decay_ts)
        if dt <= 0.05:
            return
        hl = max(1e-3, self.cfg.ewma_half_life_s)
        k = 0.5 ** (dt / hl)
        self._ew_ok *= k
        self._ew_err *= k
        self._last_decay_ts = now

    def _err_ratio(self) -> Tuple[float, float]:
        self._decay_ewma()
        total = self._ew_ok + self._ew_err
        ratio = (self._ew_err / total) if total > 0 else 0.0
        return ratio, total

    # ---------- internals: state machine ----------

    def _compute_cooloff_s(self) -> float:
        stage = self._open_backoff_stage
        base = self.cfg.base_cooloff_s * (self.cfg.backoff_factor**stage)
        base = min(base, self.cfg.max_cooloff_s)
        return base + random.uniform(0.0, self.cfg.jitter_s)

    def _transition(self, new_state: str) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state

        if new_state == self.CLOSED:
            self._hb_allfail_streak = 0
            self._open_backoff_stage = 0
            self._healthy_event.set()
            self._last_closed_ts = time.time()
            self._logger.info("ConnectivityGuard: %s -> CLOSED", old.upper())

        elif new_state == self.OPEN:
            self._healthy_event.clear()
            self._open_seq_total += 1
            self._open_backoff_stage += 1
            cool = max(self.cfg.min_open_s, self._compute_cooloff_s())
            now = time.time()
            self._cooloff_until = now + cool
            self._last_open_ts = now
            self._logger.warning(
                "ConnectivityGuard: %s -> OPEN (cooloff %.1fs, stage %d)",
                old.upper(),
                cool,
                self._open_backoff_stage,
            )

        elif new_state == self.HALF_OPEN:
            self._healthy_event.clear()
            self._logger.info("ConnectivityGuard: %s -> HALF_OPEN", old.upper())

    # ---------- internals: probes ----------

    @staticmethod
    def _blocking_tcp_connect(host: str, port: int, timeout: float) -> bool:
        """
        Blocking connect to avoid event-loop starvation false positives.
        """
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    async def _tcp_probe(self, host: str, port: int, timeout: float) -> bool:
        """
        Probe via a blocking socket connect in a thread, then await with some slack.

        Even with to_thread, the *await* still needs scheduling; slack reduces false timeouts.
        """
        slack = max(0.75, timeout * 0.25)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._blocking_tcp_connect, host, port, timeout),
                timeout=timeout + slack,
            )
        except Exception as e:
            if self.cfg.log_probe_failures:
                self._logger.debug(
                    "ConnectivityGuard probe failed: %s:%d err=%r", host, port, e
                )
            return False

    async def _probe_round(self) -> Tuple[int, int]:
        """
        Probe all targets in parallel; success if any succeed.
        Early-exit on first success to reduce load.
        """
        if not self.cfg.probe_targets:
            return 0, 0

        targets = self.cfg.probe_targets
        n_targets = len(targets)
        tasks = [
            asyncio.create_task(
                self._tcp_probe(h, p, self.cfg.connect_timeout_s),
                name=f"cg-probe-{h}:{p}",
            )
            for (h, p) in targets
        ]

        ok = 0
        pending = set(tasks)

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    try:
                        res = bool(t.result())
                    except Exception:
                        res = False
                    if res:
                        ok += 1

                if ok > 0:
                    for t in pending:
                        t.cancel()
                    with contextlib.suppress(Exception):
                        await asyncio.gather(*pending, return_exceptions=True)
                    pending.clear()
                    break
        finally:
            for t in pending:
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)

        return ok, n_targets

    # ---------- internals: main loop ----------

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._state == self.OPEN:
                    remaining = self._cooloff_until - time.time()
                    if remaining > 0:
                        await asyncio.sleep(min(remaining, self.cfg.interval_s))
                        continue
                    self._transition(self.HALF_OPEN)

                ok_count, n_targets = await self._probe_round()
                all_failed = n_targets > 0 and ok_count == 0

                # ignore very brief fail bursts right after closing
                if (
                    self._state == self.CLOSED
                    and (time.time() - self._last_closed_ts) < self.cfg.min_closed_s
                ):
                    all_failed = False

                if not all_failed:
                    if self._state != self.CLOSED:
                        self._transition(self.CLOSED)
                    else:
                        self._hb_allfail_streak = 0
                else:
                    self._hb_allfail_streak += 1
                    ratio, total = self._err_ratio()
                    self._logger.debug(
                        "ConnectivityGuard all-fail streak=%d/%d | ewma err-ratio=%.2f (n≈%.1f)",
                        self._hb_allfail_streak,
                        self.cfg.trip_heartbeats,
                        ratio,
                        total,
                    )

                    open_due_to_hybrid = self._hb_allfail_streak >= max(
                        1, int(self.cfg.trip_heartbeats)
                    ) and (
                        total >= float(self.cfg.err_ratio_floor_samples)
                        and ratio >= float(self.cfg.err_ratio_trip)
                    )
                    open_due_to_persistence = self._hb_allfail_streak >= max(
                        2, int(self.cfg.trip_heartbeats) * 2
                    )

                    if self._state == self.CLOSED and (
                        open_due_to_hybrid or open_due_to_persistence
                    ):
                        self._transition(self.OPEN)
                    elif self._state == self.HALF_OPEN:
                        self._transition(self.OPEN)

                await asyncio.sleep(self.cfg.interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.exception("ConnectivityGuard loop exception: %r", e)
                self._hb_allfail_streak += 1
                if self._state in (
                    self.CLOSED,
                    self.HALF_OPEN,
                ) and self._hb_allfail_streak >= max(
                    2, int(self.cfg.trip_heartbeats) * 2
                ):
                    self._transition(self.OPEN)
                await asyncio.sleep(self.cfg.interval_s)

        self._logger.debug("ConnectivityGuard._run() exiting.")
