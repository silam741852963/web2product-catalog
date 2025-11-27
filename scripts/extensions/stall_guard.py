from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Callable, Awaitable, Dict

logger = logging.getLogger("stall_guard")


# ---------------------------------------------------------------------------
# Config / models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StallGuardConfig:
    """
    Configuration for StallGuard.

    Purely time-based stall detection at the company level.

    A company is considered stalled when no progress has been recorded for
    longer than hard_timeout_sec = page_timeout_sec * hard_timeout_factor.
    """

    page_timeout_sec: float = 60.0
    soft_timeout_factor: float = 1.5
    hard_timeout_factor: float = 3.0
    check_interval_sec: float = 30.0

    @property
    def soft_timeout_sec(self) -> float:
        return self.page_timeout_sec * self.soft_timeout_factor

    @property
    def hard_timeout_sec(self) -> float:
        return self.page_timeout_sec * self.hard_timeout_factor


@dataclass(slots=True)
class StallSnapshot:
    """
    Immutable snapshot of a company-level stall.
    """

    detected_at: str
    company_id: str
    idle_seconds: float
    last_progress_at: Optional[str]
    last_event: Optional[str]
    reason: str


class StallDetectedError(RuntimeError):
    """Raised when StallGuard detects a stall (optional use)."""


@dataclass(slots=True)
class _CompanyState:
    company_id: str
    last_progress_mono: Optional[float] = None
    last_progress_wall: Optional[datetime] = None
    last_event: Optional[str] = None
    active: bool = True
    stalled: bool = False


# ---------------------------------------------------------------------------
# StallGuard: async company-level monitor
# ---------------------------------------------------------------------------


class StallGuard:
    """
    Async stall monitor working at company level.

    - Each company gets its own independent idle timer.
    - Progress is recorded via:
        * record_company_start(company_id)
        * record_company_completed(company_id)
        * record_heartbeat(source, company_id=...)
    - A stall is declared for a company when the time since its last
      progress exceeds hard_timeout_sec.

    StallGuard itself does not stop the run or write to disk.
    It only keeps snapshots in memory and optionally calls on_stall.
    """

    def __init__(
        self,
        config: Optional[StallGuardConfig] = None,
        *,
        logger_name: str = "stall_guard",
    ) -> None:
        self.config = config or StallGuardConfig()
        self._log = logging.getLogger(logger_name)

        # Company states keyed by company_id
        self._companies: Dict[str, _CompanyState] = {}

        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # First stall snapshot (for wait_for_stall()) and event
        self._stall_event: asyncio.Event = asyncio.Event()
        self._first_snapshot: Optional[StallSnapshot] = None

        # Optional user callback invoked on each company stall
        # Signature: async def on_stall(snapshot: StallSnapshot) -> None
        self.on_stall: Optional[Callable[[StallSnapshot], Awaitable[None]]] = None

        # Snapshots for all companies stalled during this process
        self._stalled_snapshots: Dict[str, StallSnapshot] = {}

    # ------------------------------------------------------------------ #
    # Public API: lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Start the background monitor task.

        Safe to call multiple times; subsequent calls are ignored.
        If no running event loop is available, the guard logs and does nothing.
        """
        if self._running:
            self._log.debug("StallGuard.start() called but already running")
            return

        self._stall_event.clear()
        self._running = True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called from a context without an event loop (bug in caller).
            self._running = False
            self._log.error(
                "StallGuard.start() called without a running event loop; guard not started"
            )
            return

        self._monitor_task = loop.create_task(
            self._monitor_loop(),
            name="stall-guard-monitor",
        )

        self._log.info(
            "StallGuard started (company-level): page_timeout=%.1fs soft>=%.1fs hard>=%.1fs interval=%.1fs",
            self.config.page_timeout_sec,
            self.config.soft_timeout_sec,
            self.config.hard_timeout_sec,
            self.config.check_interval_sec,
        )

    async def stop(self) -> None:
        """
        Stop the background monitor task.

        This is best-effort and will not raise if the task is already done
        or if cancellation fails.
        """
        if not self._running and self._monitor_task is None:
            return

        self._running = False
        task = self._monitor_task
        self._monitor_task = None

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._log.debug("Error while stopping StallGuard monitor: %s", e)

    async def wait_for_stall(self) -> StallSnapshot:
        """
        Wait until the first stall is detected and return its snapshot.
        """
        await self._stall_event.wait()
        assert self._first_snapshot is not None
        return self._first_snapshot

    def is_stalled(self) -> bool:
        """
        True if any company has been marked as stalled.
        """
        return self._first_snapshot is not None

    def last_progress_age(self) -> Optional[float]:
        """
        Return the minimum age (seconds) since last progress across all
        active companies. None if no progress has been recorded yet.
        """
        now = time.monotonic()
        ages = []
        for st in self._companies.values():
            if not st.active or st.last_progress_mono is None:
                continue
            ages.append(now - st.last_progress_mono)
        if not ages:
            return None
        return min(ages)

    def stalled_companies(self) -> Dict[str, StallSnapshot]:
        """
        Return a mapping company_id -> StallSnapshot for all companies that
        were marked as stalled during this StallGuard's lifetime.
        """
        return dict(self._stalled_snapshots)

    # ------------------------------------------------------------------ #
    # Public API: progress hooks
    # ------------------------------------------------------------------ #

    def record_company_start(self, company_id: str) -> None:
        """
        Mark the beginning of a company pipeline.

        This ensures that timeouts can trigger even if we never manage to
        fetch a single page.
        """
        st = self._companies.get(company_id)
        if st is None:
            st = _CompanyState(company_id=company_id)
            self._companies[company_id] = st
        st.active = True
        st.stalled = False
        self._touch_progress(st, reason="company_start")

    def record_company_completed(self, company_id: str) -> None:
        """
        Record that a company's pipeline has completed successfully.

        The company will no longer be monitored for stalls.
        """
        st = self._companies.get(company_id)
        if st is None:
            st = _CompanyState(company_id=company_id)
            self._companies[company_id] = st
        self._touch_progress(st, reason="company_completed")
        st.active = False

    def record_heartbeat(
        self,
        source: str = "generic",
        company_id: Optional[str] = None,
    ) -> None:
        """
        Record a generic heartbeat.

        - If company_id is provided, it is treated as company-level progress.
        - If company_id is None, it is treated as a global heartbeat and does
          not affect stall detection.

        This lets you use StallGuard in scripts that only want its logging
        without it killing anything. Just omit the company_id argument.
        """
        if company_id is None:
            self._log.debug(
                "StallGuard global heartbeat (%s) - ignored for stall detection",
                source,
            )
            return

        st = self._companies.get(company_id)
        if st is None:
            st = _CompanyState(company_id=company_id)
            self._companies[company_id] = st
        self._touch_progress(st, reason=f"heartbeat:{source}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _touch_progress(self, st: _CompanyState, *, reason: str) -> None:
        now_mono = time.monotonic()
        now_wall = datetime.now(timezone.utc)

        st.last_progress_mono = now_mono
        st.last_progress_wall = now_wall
        st.last_event = reason

        self._log.debug(
            "StallGuard progress heartbeat (%s): company=%s",
            reason,
            st.company_id,
        )

    async def _monitor_loop(self) -> None:
        try:
            while self._running:
                try:
                    await asyncio.sleep(self.config.check_interval_sec)
                except asyncio.CancelledError:
                    # Normal shutdown path.
                    break

                try:
                    self._check_for_stalls()
                except Exception as e:
                    self._log.exception("StallGuard monitor iteration failed: %s", e)
        except asyncio.CancelledError:
            # Defensive double catch in case cancellation lands here directly.
            return

    def _check_for_stalls(self) -> None:
        if not self._companies:
            self._log.debug("StallGuard: no companies registered; skipping stall check")
            return

        now = time.monotonic()
        soft = self.config.soft_timeout_sec
        hard = self.config.hard_timeout_sec

        for company_id, st in list(self._companies.items()):
            if not st.active or st.stalled:
                continue

            if st.last_progress_mono is None:
                # Company started but no progress yet; wait for first heartbeat.
                self._log.debug(
                    "StallGuard check: company=%s has no progress timestamps yet; "
                    "waiting for first heartbeat",
                    company_id,
                )
                continue

            idle = now - st.last_progress_mono

            if idle < 0:
                # Monotonic clock should not go backwards, but be defensive.
                self._log.debug(
                    "StallGuard check: company=%s had negative idle %.3fs, skipping",
                    company_id,
                    idle,
                )
                continue

            self._log.debug(
                "StallGuard check: company=%s idle=%.1fs soft>=%.1fs hard>=%.1fs",
                company_id,
                idle,
                soft,
                hard,
            )

            # Pure time-based stall: only condition is idle >= hard_timeout_sec
            if idle >= hard:
                self._declare_company_stall(st, idle_seconds=idle)

    def _declare_company_stall(self, st: _CompanyState, *, idle_seconds: float) -> None:
        if st.stalled:
            return

        st.stalled = True
        st.active = False

        detected_at = datetime.now(timezone.utc)
        snapshot = StallSnapshot(
            detected_at=detected_at.isoformat(),
            company_id=st.company_id,
            idle_seconds=float(idle_seconds),
            last_progress_at=(
                st.last_progress_wall.isoformat()
                if st.last_progress_wall is not None
                else None
            ),
            last_event=st.last_event,
            reason=(
                "company_idle_for_%.1fs_gt_hard_threshold_%.1fs"
                % (idle_seconds, self.config.hard_timeout_sec)
            ),
        )

        # Store the first snapshot for wait_for_stall()
        if self._first_snapshot is None:
            self._first_snapshot = snapshot
            self._stall_event.set()

        # Keep the snapshot available for later inspection.
        self._stalled_snapshots[st.company_id] = snapshot

        self._log.error(
            "STALL DETECTED for company_id=%s: idle=%.1fs (see snapshot)",
            st.company_id,
            idle_seconds,
        )

        # Fire async callback, if any
        if self.on_stall is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop; nothing we can do.
                self._log.warning(
                    "StallGuard: stall detected for %s but no running loop; "
                    "on_stall callback not scheduled",
                    st.company_id,
                )
            else:
                loop.create_task(
                    self._run_on_stall(snapshot),
                    name=f"stall-guard-on-stall-{st.company_id}",
                )

    async def _run_on_stall(self, snapshot: StallSnapshot) -> None:
        if self.on_stall is None:
            return
        try:
            await self.on_stall(snapshot)
        except Exception as e:
            self._log.exception("StallGuard on_stall callback failed: %s", e)


# ---------------------------------------------------------------------------
# Global hard stall helper
# ---------------------------------------------------------------------------


async def wait_for_global_hard_stall(
    stall_guard: StallGuard,
    *,
    page_timeout_sec: float,
    factor: float = 4.5,
    check_interval_sec: float = 30.0,
) -> float:
    """
    Wait until the whole run appears globally stalled.

    Definition here:
      - StallGuard.last_progress_age() returns the minimum idle time across
        all active companies.
      - If that age is greater than or equal to page_timeout_sec * factor,
        we consider the system globally stalled.

    Returns the observed idle seconds at the moment of detection.
    """

    if factor <= 0:
        factor = 4.5
    if check_interval_sec <= 0:
        check_interval_sec = page_timeout_sec

    hard = page_timeout_sec * factor

    logger.info(
        "Global StallGuard watchdog started: page_timeout=%.1fs factor=%.1f hard>=%.1fs check_interval=%.1fs",
        page_timeout_sec,
        factor,
        hard,
        check_interval_sec,
    )

    while True:
        await asyncio.sleep(check_interval_sec)
        age = stall_guard.last_progress_age()

        if age is None:
            # No active companies with progress timestamps yet.
            # This normally happens only at startup or when all companies
            # have been completed or already stalled at the company level.
            logger.debug(
                "Global StallGuard: last_progress_age() is None, skipping iteration"
            )
            continue

        logger.debug(
            "Global StallGuard: min idle across companies = %.1fs (hard>=%.1fs)",
            age,
            hard,
        )

        if age >= hard:
            logger.error(
                "Global StallGuard: hard stall detected, min idle=%.1fs >= %.1fs",
                age,
                hard,
            )
            return age


__all__ = [
    "StallGuardConfig",
    "StallSnapshot",
    "StallGuard",
    "StallDetectedError",
    "wait_for_global_hard_stall",
]