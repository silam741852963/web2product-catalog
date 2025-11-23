from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Callable, Awaitable


logger = logging.getLogger("stall_guard")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StallGuardConfig:
    """
    Configuration for StallGuard.

    Parameters
    ----------
    page_timeout_sec:
        The nominal per-page timeout used by the crawler / JS policy.
        Stall detection thresholds will be derived from this.

    soft_timeout_factor:
        Soft stall threshold = page_timeout_sec * soft_timeout_factor.
        When idle time exceeds this, StallGuard starts counting "stall checks".

    hard_timeout_factor:
        Hard stall threshold = page_timeout_sec * hard_timeout_factor.
        When idle time exceeds this, StallGuard declares a stall.

    check_interval_sec:
        How often (in seconds) StallGuard wakes up to inspect idleness.

    min_pages_before_detection:
        Do not declare stalls until at least this many pages have been
        successfully processed. Helps avoid false positives at startup.

    min_companies_before_detection:
        Do not declare stalls until at least this many companies have finished.

    min_consecutive_stall_checks:
        Require this many consecutive "idle above soft threshold" checks before
        declaring a stall. This smooths out short pauses.

    auto_kill_process:
        If True, StallGuard will terminate the current process with
        os._exit(auto_kill_exit_code) when a stall is detected. This is
        aggressive but guarantees any leaked Chromium processes are cleaned up.
        Default is False – you can instead react via wait_for_stall() or an
        on_stall callback.

    auto_kill_exit_code:
        Exit code used when auto_kill_process is True.

    dump_state_path:
        Optional path to a JSON file where StallGuard will dump a snapshot of
        stall state when detected (for debugging).
    """

    page_timeout_sec: float = 60.0
    soft_timeout_factor: float = 2.0
    hard_timeout_factor: float = 5.0
    check_interval_sec: float = 15.0

    min_pages_before_detection: int = 10
    min_companies_before_detection: int = 0
    min_consecutive_stall_checks: int = 2

    auto_kill_process: bool = False
    auto_kill_exit_code: int = 3

    dump_state_path: Optional[Path] = None

    @property
    def soft_timeout_sec(self) -> float:
        return self.page_timeout_sec * self.soft_timeout_factor

    @property
    def hard_timeout_sec(self) -> float:
        return self.page_timeout_sec * self.hard_timeout_factor


@dataclass(slots=True)
class StallSnapshot:
    """
    Immutable snapshot of the world when a stall is detected.
    """

    detected_at: str
    idle_seconds: float
    last_progress_at: Optional[str]
    pages_processed: int
    companies_completed: int
    consecutive_stall_checks: int
    reason: str


class StallDetectedError(RuntimeError):
    """Raised when StallGuard detects a stall (optional use)."""


# ---------------------------------------------------------------------------
# StallGuard: async stall monitor
# ---------------------------------------------------------------------------


class StallGuard:
    """
    Lightweight async "stall manager" for long-running crawls.

    Usage pattern (high level)
    --------------------------
    1. Create + start:

        stall_guard = StallGuard(config=StallGuardConfig(
            page_timeout_sec=60.0,
            dump_state_path=out_dir / "stall_state.json",
        ))
        await stall_guard.start()

    2. Wire progress callbacks:

        - In per-page processing (e.g. process_page_result):
              stall_guard.record_page(company_id, url)

        - In per-company completion:
              stall_guard.record_company_completed(company_id)

        - In LLM passes:
              stall_guard.record_heartbeat("llm_presence")

    3. Option A: let StallGuard kill the process when stalled
           (config.auto_kill_process = True)

       Option B: observe stall from main():
           snapshot = await stall_guard.wait_for_stall()
           # log, clean up resources, exit gracefully, etc.

       Option C: register an on_stall callback:
           stall_guard.on_stall = my_async_handler

    Stall criteria
    --------------
    A stall is declared when ALL of the following are true:

      * last_progress > soft_timeout_sec (derived from page_timeout_sec)
      * pages_processed >= min_pages_before_detection
      * companies_completed >= min_companies_before_detection
      * For at least min_consecutive_stall_checks monitor cycles
      * AND last_progress > hard_timeout_sec

    Once a stall is declared:
      * A StallSnapshot is created & optionally written to dump_state_path
      * An internal asyncio.Event is set (wait_for_stall() unblocks)
      * on_stall callback is scheduled (if provided)
      * If auto_kill_process=True, os._exit(auto_kill_exit_code) is called
        after the callback finishes (or immediately if none).
    """

    def __init__(
        self,
        config: Optional[StallGuardConfig] = None,
        *,
        logger_name: str = "stall_guard",
    ) -> None:
        self.config = config or StallGuardConfig()
        self._log = logging.getLogger(logger_name)

        # Progress counters
        self.pages_processed: int = 0
        self.companies_completed: int = 0

        # Timestamps (monotonic + wall-clock)
        self._last_progress_mono: Optional[float] = None
        self._last_progress_wall: Optional[datetime] = None

        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Stall bookkeeping
        self._stall_event: asyncio.Event = asyncio.Event()
        self._stall_snapshot: Optional[StallSnapshot] = None
        self._stall_detected_at_mono: Optional[float] = None
        self._consecutive_stall_checks: int = 0

        # Optional user callback
        self.on_stall: Optional[Callable[[StallSnapshot], Awaitable[None]]] = None

    # ------------------------------------------------------------------ #
    # Public API: lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Start the background monitoring task.
        """
        if self._running:
            return
        self._running = True
        self._stall_event.clear()
        self._monitor_task = asyncio.create_task(self._monitor_loop(), name="stall-guard-monitor")

        self._log.info(
            "StallGuard started: page_timeout=%.1fs soft>=%.1fs hard>=%.1fs interval=%.1fs",
            self.config.page_timeout_sec,
            self.config.soft_timeout_sec,
            self.config.hard_timeout_sec,
            self.config.check_interval_sec,
        )

    async def stop(self) -> None:
        """
        Stop the background monitoring task.
        """
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
        Wait until a stall is detected and return the StallSnapshot.
        """
        await self._stall_event.wait()
        # _stall_snapshot is set before the event is triggered
        assert self._stall_snapshot is not None
        return self._stall_snapshot

    def is_stalled(self) -> bool:
        """
        True if StallGuard has already declared a stall.
        """
        return self._stall_snapshot is not None

    def last_progress_age(self) -> Optional[float]:
        """
        How many seconds since the last recorded progress (page/company/heartbeat).
        Returns None if no progress has ever been recorded.
        """
        if self._last_progress_mono is None:
            return None
        return time.monotonic() - self._last_progress_mono

    # ------------------------------------------------------------------ #
    # Public API: progress hooks
    # ------------------------------------------------------------------ #

    def record_page(self, company_id: str, url: str) -> None:
        """
        Record that a page has been successfully processed.

        This should be called from per-page processing (e.g. process_page_result)
        *after* all saving and url_index updates succeed.
        """
        self.pages_processed += 1
        self._touch_progress(reason=f"page:{company_id}")

    def record_company_completed(self, company_id: str) -> None:
        """
        Record that a company pipeline has completed (crawl + LLM, etc.).
        """
        self.companies_completed += 1
        self._touch_progress(reason=f"company:{company_id}")

    def record_heartbeat(self, source: str = "generic") -> None:
        """
        Record a generic heartbeat for progress-like events that are not
        strictly page or company completion (e.g. LLM passes).
        """
        self._touch_progress(reason=f"heartbeat:{source}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _touch_progress(self, *, reason: str) -> None:
        """
        Update last progress timestamps and reset stall counters.
        """
        now_mono = time.monotonic()
        now_wall = datetime.now(timezone.utc)

        self._last_progress_mono = now_mono
        self._last_progress_wall = now_wall
        self._consecutive_stall_checks = 0
        self._stall_detected_at_mono = None

        self._log.debug(
            "StallGuard progress heartbeat (%s): pages=%d companies=%d",
            reason,
            self.pages_processed,
            self.companies_completed,
        )

    async def _monitor_loop(self) -> None:
        """
        Background coroutine that periodically checks for stalls.
        """
        try:
            while self._running:
                await asyncio.sleep(self.config.check_interval_sec)
                try:
                    self._check_for_stall()
                except Exception as e:
                    self._log.exception("StallGuard monitor iteration failed: %s", e)
        except asyncio.CancelledError:
            # Normal shutdown
            return

    def _check_for_stall(self) -> None:
        """
        Evaluate whether the current state should be considered a stall.
        """
        # Once stalled, we don't re-evaluate further.
        if self._stall_snapshot is not None:
            return

        # No progress has ever been recorded: nothing to measure yet
        if self._last_progress_mono is None:
            self._log.debug("StallGuard: no progress yet; skipping stall check")
            return

        idle = time.monotonic() - self._last_progress_mono
        soft = self.config.soft_timeout_sec
        hard = self.config.hard_timeout_sec

        self._log.debug(
            "StallGuard check: idle=%.1fs soft>=%.1fs hard>=%.1fs pages=%d companies=%d",
            idle,
            soft,
            hard,
            self.pages_processed,
            self.companies_completed,
        )

        # Not idle enough yet
        if idle < soft:
            self._consecutive_stall_checks = 0
            return

        # Idle beyond soft threshold
        self._consecutive_stall_checks += 1

        # Do not consider stall too early in the run
        if self.pages_processed < self.config.min_pages_before_detection:
            self._log.debug(
                "StallGuard: idle above soft threshold but pages_processed=%d < min=%d; "
                "not declaring stall yet",
                self.pages_processed,
                self.config.min_pages_before_detection,
            )
            return

        if self.companies_completed < self.config.min_companies_before_detection:
            self._log.debug(
                "StallGuard: idle above soft threshold but companies_completed=%d < min=%d; "
                "not declaring stall yet",
                self.companies_completed,
                self.config.min_companies_before_detection,
            )
            return

        if self._consecutive_stall_checks < self.config.min_consecutive_stall_checks:
            self._log.debug(
                "StallGuard: idle above soft threshold but consecutive_checks=%d < min=%d; "
                "waiting for another cycle",
                self._consecutive_stall_checks,
                self.config.min_consecutive_stall_checks,
            )
            return

        # Hard stall condition
        if idle < hard:
            # Idle but not yet "hard" stall
            self._log.warning(
                "StallGuard: long idle (%.1fs) but below hard threshold (%.1fs); "
                "waiting one more cycle",
                idle,
                hard,
            )
            return

        # At this point we consider it a real stall.
        self._declare_stall(idle_seconds=idle)

    def _declare_stall(self, *, idle_seconds: float) -> None:
        """
        Create a snapshot, dump it if configured, signal waiters, and possibly
        kill the process.
        """
        if self._stall_snapshot is not None:
            return  # already declared

        detected_at = datetime.now(timezone.utc)

        snapshot = StallSnapshot(
            detected_at=detected_at.isoformat(),
            idle_seconds=float(idle_seconds),
            last_progress_at=(
                self._last_progress_wall.isoformat()
                if self._last_progress_wall is not None
                else None
            ),
            pages_processed=self.pages_processed,
            companies_completed=self.companies_completed,
            consecutive_stall_checks=self._consecutive_stall_checks,
            reason=(
                "no_progress_for_%.1fs_gt_hard_threshold_%.1fs"
                % (idle_seconds, self.config.hard_timeout_sec)
            ),
        )

        self._stall_snapshot = snapshot
        self._stall_event.set()

        self._log.error(
            "STALL DETECTED: idle=%.1fs pages=%d companies=%d (see snapshot)",
            idle_seconds,
            self.pages_processed,
            self.companies_completed,
        )

        # Optional dump to JSON for debugging
        if self.config.dump_state_path is not None:
            try:
                payload = asdict(snapshot)
                path = Path(self.config.dump_state_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._log.info("StallGuard wrote snapshot to %s", path)
            except Exception as e:
                self._log.exception(
                    "StallGuard failed to write stall snapshot to %s: %s",
                    self.config.dump_state_path,
                    e,
                )

        # Schedule user callback, if any
        if self.on_stall is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                loop.create_task(self._run_on_stall(snapshot))
            else:
                # No running loop; best-effort synchronous call
                try:
                    coro = self.on_stall(snapshot)
                    # If user gave us an async callable but no loop is running,
                    # we can't really run it; just log this situation.
                    self._log.error(
                        "StallGuard.on_stall provided but no running loop available; "
                        "cannot execute callback."
                    )
                except Exception as e:
                    self._log.exception(
                        "StallGuard failed to schedule on_stall callback: %s", e
                    )

        # Optionally kill the process (very aggressive)
        if self.config.auto_kill_process:
            self._log.error(
                "StallGuard auto_kill_process=True → terminating process with exit code %d",
                self.config.auto_kill_exit_code,
            )
            try:
                # Try to flush logs before hard exit
                for h in logging.getLogger().handlers:
                    try:
                        h.flush()
                    except Exception:
                        pass
            finally:
                os._exit(self.config.auto_kill_exit_code)

    async def _run_on_stall(self, snapshot: StallSnapshot) -> None:
        """
        Internal helper to run the on_stall callback and handle its errors.
        """
        if self.on_stall is None:
            return
        try:
            await self.on_stall(snapshot)
        except Exception as e:
            self._log.exception("StallGuard on_stall callback failed: %s", e)


__all__ = [
    "StallGuardConfig",
    "StallSnapshot",
    "StallGuard",
    "StallDetectedError",
]