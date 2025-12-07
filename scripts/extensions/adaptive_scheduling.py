from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable, Awaitable

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# psutil is optional; degrade gracefully if missing
try:
    import psutil  # type: ignore

    PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Memory pressure types (merged from memory_guard)
# ---------------------------------------------------------------------------

MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"


class CriticalMemoryPressure(RuntimeError):
    """
    Raised when memory pressure is high enough that we should abort
    the current company task.

    severity:
        "soft"      - company level problem (e.g. page marker)
        "hard"      - host reached critical level (but run may continue)
        "emergency" - host reached emergency limit, run should exit
    """

    def __init__(self, message: str, severity: str = "emergency") -> None:
        super().__init__(message)
        self.severity = severity


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSchedulingConfig:
    """
    Configuration for adaptive scheduling based on host CPU and memory,
    augmented with host-level memory protection, cancellation logic,
    and coarse-grained stall detection.

    Semantics:

      - run.py owns:
          * The list of companies to run.
          * The creation and tracking of asyncio.Task per company.
          * The count of active company tasks (active_tasks).
          * The pending / completed sets and progress persistence.

      - AdaptiveScheduler:
          * Periodically samples CPU and memory and maintains smoothed values.
          * Classifies the current load into three bands:
                - low band:  both CPU and mem at or below target_*_low
                - cautious: between low and high thresholds
                - high:     CPU or mem at or above target_*_high
          * Decides whether it is safe to admit a single new company.
          * Enforces cooldowns:
                - between successful admissions
                - extra cooldown when in the cautious band
                - a recheck cooldown for the high band
          * Can enter a "rush" mode when CPU and memory are underused
            and stable, which speeds up admissions for a short period.
          * Enforces a hard cap via max_concurrency, using active_tasks.

      - Host-level memory safety:
          * Monitors raw memory usage.
          * When memory crosses a critical threshold (mem_critical_limit),
            the scheduler can select a company to cancel (preferably the
            newest one) to avoid OOM.
          * Cancel decisions are:
                - based only on memory, not CPU.
                - subject to a cooldown: cancel -> wait -> new sample -> maybe cancel again.
          * After a cancel, new admissions are blocked for
            mem_post_relief_block_sec seconds to let memory settle.

      - Priority requeue:
          * When a company is cancelled due to memory, run.py should:
                - safely persist its progress
                - mark it as not-done (e.g. markdown not finished)
                - put it back into a "priority waiting queue" so that
                  it can resume immediately once admissions are allowed.
          * AdaptiveScheduler stores a memory-priority queue of company IDs
            and exposes helper methods to manage it.

      - Host-level stall detection (coarse):
          * Tracks repeated denials with active_tasks == 0
            (for example, "high_band_cooldown" with active=0).
          * Tracks long CPU/memory plateaus where both fluctuate in a
            narrow band (±host_stall_band_width) over host_stall_window_sec,
            regardless of absolute usage (0..100 percent).
          * When either condition persists longer than
            host_stall_min_duration_sec, the scheduler marks
            host_stall_suspected=True in its state snapshot. The caller
            (run.py) is responsible for:
                - saving progress of current tasks, and
                - exiting the process with code 17 so an outer wrapper
                  can restart the run.

      - The caller uses:
            ok, reason = await scheduler.can_start_new_company(active_tasks)

        If ok is True:
            - run.py should start exactly one new company task
            - and call scheduler.notify_admitted(company_id, reason)

        If ok is False:
            - run.py should not start a new company
            - and may log or inspect the returned reason.
    """

    max_concurrency: int

    target_mem_low: float = 0.60
    target_mem_high: float = 0.90
    target_cpu_low: float = 0.70
    target_cpu_high: float = 0.85

    sample_interval_sec: float = 1.0
    smoothing_window_sec: float = 10.0

    log_path: Optional[Path] = None

    use_cpu: bool = True
    use_mem: bool = True

    # Cooldowns for admissions
    admission_cooldown_sec: float = 5.0
    cautious_admission_cooldown_sec: float = 15.0
    block_cooldown_sec: float = 30.0

    # Legacy memory-pressure block, still used by on_memory_pressure()
    mem_pressure_block_sec: float = 15.0

    # Rush / plateau settings
    rush_enabled: bool = True
    rush_cpu_threshold: float = 0.50  # 50 percent
    rush_mem_threshold: float = 0.80  # 80 percent
    rush_band_width: float = 0.05  # ±5 percent band
    rush_min_samples: int = 4
    rush_duration_sec: float = 30.0

    # Host-level memory cancellation
    mem_critical_limit: float = 0.94  # 94 percent
    mem_cancel_cooldown_sec: float = 20.0  # cooldown between cancels
    mem_post_relief_block_sec: float = 180.0  # block new admissions ~3 minutes
    mem_cancel_enabled: bool = True

    # Host-level stall detection
    host_stall_enabled: bool = True
    # Time window over which to look for CPU/memory plateaus (seconds).
    host_stall_window_sec: float = 300.0
    # Narrow band for plateau detection (fraction 0..1, default 1 percent).
    host_stall_band_width: float = 0.01
    # Minimum duration (seconds) for either stall condition (plateau or
    # repeated denials with active=0) before marking host_stall_suspected.
    host_stall_min_duration_sec: float = 300.0

    # Page-level marker
    mem_marker: str = MEMORY_PRESSURE_MARKER


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class AdaptiveScheduler:
    """
    Adaptive scheduler that gates admission of new company tasks based
    on smoothed CPU and memory usage, three-band classification, rush
    mode, and cooldowns; and also provides host-level memory protection,
    a priority queue for memory-cancelled companies, and coarse
    host-level stall detection.

    It does not manage asyncio.Task objects directly. Instead:

      - Caller:
          * owns the Task objects
          * decides when to call cancellation on them
          * persists progress and crawl_meta
          * marks company as NOT_DONE before requeue

      - AdaptiveScheduler:
          * tells caller:
                - when it's safe to admit more work (can_start_new_company)
                - which company to cancel when host memory is critical
                  (maybe_select_company_to_cancel)
                - which company should be re-prioritized after cancel
                  (memory priority queue APIs)
                - whether a host-level stall is suspected, via get_state().
    """

    def __init__(self, cfg: AdaptiveSchedulingConfig) -> None:
        self.cfg = cfg

        self._lock = asyncio.Lock()
        # Sliding window samples for admission band classification and rush.
        self._samples: List[
            Tuple[float, float, float]
        ] = []  # (ts_mono, cpu_frac, mem_frac)
        # Longer window for stall plateau detection.
        self._stall_samples: List[
            Tuple[float, float, float]
        ] = []  # (ts_mono, cpu_frac, mem_frac)
        self._task: Optional[asyncio.Task] = None

        self._last_avg_cpu: float = 0.0
        self._last_avg_mem: float = 0.0
        self._last_raw_cpu: float = 0.0
        self._last_raw_mem: float = 0.0
        self._last_update_ts: float = 0.0  # wall clock
        self._last_sample_mono: float = 0.0

        # Cooldown timestamps (monotonic)
        self._last_admission_ts: float = 0.0
        self._last_cautious_admission_ts: float = 0.0
        self._last_high_block_ts: float = 0.0
        self._blocked_until_ts: float = 0.0

        # Memory pressure tracking
        self._memory_pressure_hits: int = 0

        # Rush mode tracking (monotonic times)
        self._rush_active: bool = False
        self._rush_start_ts: float = 0.0
        self._rush_until_ts: float = 0.0
        self._rush_start_cpu: float = 0.0
        self._rush_start_mem: float = 0.0

        # Memory-based cancellation tracking
        self._next_mem_cancel_ts: float = 0.0
        self._last_cancel_sample_mono: float = 0.0

        # Admission order (for picking "newest" company to cancel)
        # Stores company_ids in order of admission (oldest first).
        self._admission_order: List[str] = []

        # Memory-priority queue: companies cancelled due to memory
        # that should be resumed as soon as the scheduler allows.
        # New entries are appended; consumers should treat as FIFO.
        self._memory_priority_queue: List[str] = []

        # Host-level stall detection state
        self._host_stall_suspected: bool = False
        self._host_stall_since_mono: float = 0.0
        self._host_stall_reason: str = ""
        self._host_stall_last_denial_reason: str = ""
        self._host_stall_last_denial_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Start background sampling of CPU and memory.
        """
        if self._task is not None:
            return
        interval = max(float(self.cfg.sample_interval_sec), 0.5)
        self._task = asyncio.create_task(
            self._loop(interval),
            name="adaptive-scheduling-sampler",
        )
        logger.info(
            "[AdaptiveScheduling] started "
            "(max_concurrency=%d, interval=%.2fs, psutil=%s)",
            self.cfg.max_concurrency,
            interval,
            PSUTIL_AVAILABLE,
        )

    async def stop(self) -> None:
        """
        Stop background sampling.
        """
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info(
            "[AdaptiveScheduling] stopped (memory_pressure_hits=%d)",
            self._memory_pressure_hits,
        )

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                await self._sample_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[AdaptiveScheduling] error in sampling loop")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ #
    # Public API for caller: admissions
    # ------------------------------------------------------------------ #

    @property
    def sample_interval_sec(self) -> float:
        """
        Convenience accessor for callers that want to align their waits
        with the sampling cadence.
        """
        return float(self.cfg.sample_interval_sec)

    async def can_start_new_company(self, active_tasks: int) -> Tuple[bool, str]:
        """
        Decide whether it is safe to admit a single new company task.

        Returns (ok, reason):

          - ok == True  -> caller should start exactly one new company.
          - ok == False -> caller should not start a new company now.

        Reasons include (non exhaustive):
          - "max_concurrency"
          - "mem_pressure_block"
          - "admission_cooldown"
          - "cautious_cooldown"
          - "high_band"
          - "high_band_cooldown"
          - "low_band"
          - "cautious_band"
          - "rush_mode"
          - "no_metrics"

        Host-level stall detection is passive: it never raises here.
        Instead, repeated denials with active_tasks == 0 contribute to
        host_stall_suspected in get_state().
        """
        now = time.monotonic()
        cfg = self.cfg

        # 1) Hard cap by active_tasks vs max_concurrency
        if active_tasks >= max(1, int(cfg.max_concurrency)):
            return False, "max_concurrency"

        async with self._lock:
            # 2) Global block due to memory or external pressure
            if now < self._blocked_until_ts:
                reason = "mem_pressure_block"
                self._update_decision_stall_locked(
                    active_tasks=active_tasks,
                    admitted=False,
                    reason=reason,
                    now=now,
                )
                return False, reason

            avg_cpu = self._last_avg_cpu
            avg_mem = self._last_avg_mem

            # If we have no samples yet and psutil is missing, admit conservatively.
            if not self._samples and not PSUTIL_AVAILABLE:
                self._last_admission_ts = now
                logger.info(
                    "[AdaptiveScheduling] psutil missing and no prior samples; "
                    "admitting new company by default.",
                )
                # No stall update: we are admitting work.
                return True, "no_metrics"

            # If we have no samples yet but psutil exists, force a sample now.
            if not self._samples and PSUTIL_AVAILABLE:
                cpu_frac, mem_frac = self._read_usage()
                self._update_samples_locked(
                    now_mono=time.monotonic(),
                    cpu=cpu_frac,
                    mem=mem_frac,
                )
                avg_cpu = self._last_avg_cpu
                avg_mem = self._last_avg_mem

            # Classify the current band.
            band = self._classify_band_locked()

            # Rush mode: fast lane when underutilized plateau was detected.
            if self._rush_active and now < self._rush_until_ts:
                if band == "high":
                    # Even in rush, high band is not safe. Do not admit.
                    self._last_high_block_ts = now
                    reason = "high_band"
                    logger.info(
                        "[AdaptiveScheduling] rush blocked by high band "
                        "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                        active_tasks,
                        avg_cpu,
                        avg_mem,
                    )
                    self._update_decision_stall_locked(
                        active_tasks=active_tasks,
                        admitted=False,
                        reason=reason,
                        now=now,
                    )
                    return False, reason
                # In low or cautious band during rush, skip cooldowns and admit.
                self._last_admission_ts = now
                if band == "cautious":
                    self._last_cautious_admission_ts = now
                logger.info(
                    "[AdaptiveScheduling] rush admission granted "
                    "(active=%d, band=%s, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    band,
                    avg_cpu,
                    avg_mem,
                )
                # Admission resets denial-based stall tracking.
                self._update_decision_stall_locked(
                    active_tasks=active_tasks,
                    admitted=True,
                    reason="rush_mode",
                    now=now,
                )
                return True, "rush_mode"

            # 3) Band specific logic when not in rush

            # High band: do not admit, respect a separate high band cooldown.
            if band == "high":
                if (
                    self._last_high_block_ts > 0.0
                    and now - self._last_high_block_ts < cfg.block_cooldown_sec
                ):
                    reason = "high_band_cooldown"
                    self._update_decision_stall_locked(
                        active_tasks=active_tasks,
                        admitted=False,
                        reason=reason,
                        now=now,
                    )
                    return False, reason

                self._last_high_block_ts = now
                reason = "high_band"
                logger.info(
                    "[AdaptiveScheduling] high band - admission denied "
                    "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    avg_cpu,
                    avg_mem,
                )
                self._update_decision_stall_locked(
                    active_tasks=active_tasks,
                    admitted=False,
                    reason=reason,
                    now=now,
                )
                return False, reason

            # From here, band is either "low" or "cautious".
            # First, respect the global admission cooldown.
            if (
                self._last_admission_ts > 0.0
                and now - self._last_admission_ts < cfg.admission_cooldown_sec
            ):
                reason = "admission_cooldown"
                self._update_decision_stall_locked(
                    active_tasks=active_tasks,
                    admitted=False,
                    reason=reason,
                    now=now,
                )
                return False, reason

            if band == "low":
                # Low band: only global admission cooldown applied.
                self._last_admission_ts = now
                reason = "low_band"
                logger.info(
                    "[AdaptiveScheduling] low band admission granted "
                    "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    avg_cpu,
                    avg_mem,
                )
                self._update_decision_stall_locked(
                    active_tasks=active_tasks,
                    admitted=True,
                    reason=reason,
                    now=now,
                )
                return True, reason

            # Cautious band.
            # Apply an extra cooldown window inside this band.
            if (
                self._last_cautious_admission_ts > 0.0
                and now - self._last_cautious_admission_ts
                < cfg.cautious_admission_cooldown_sec
            ):
                reason = "cautious_cooldown"
                logger.info(
                    "[AdaptiveScheduling] cautious band cooldown - "
                    "admission denied (active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    avg_cpu,
                    avg_mem,
                )
                self._update_decision_stall_locked(
                    active_tasks=active_tasks,
                    admitted=False,
                    reason=reason,
                    now=now,
                )
                return False, reason

            self._last_admission_ts = now
            self._last_cautious_admission_ts = now
            reason = "cautious_band"
            logger.info(
                "[AdaptiveScheduling] cautious band admission granted "
                "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                active_tasks,
                avg_cpu,
                avg_mem,
            )
            self._update_decision_stall_locked(
                active_tasks=active_tasks,
                admitted=True,
                reason=reason,
                now=now,
            )
            return True, reason

    def notify_admitted(
        self,
        company_id: Optional[str] = None,
        reason: str = "ok",
    ) -> None:
        """
        Optional helper for callers that want to log an explicit admission
        event tied to a specific company, and let the scheduler track
        "newest first" ordering for memory-based cancellations.
        """
        if company_id is None:
            return

        logger.info(
            "[AdaptiveScheduling] company admitted company_id=%s reason=%s",
            company_id,
            reason,
        )

        # Track admission order for "cancel newest first" semantics.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If not in an event loop, update synchronously without lock.
            if company_id not in self._admission_order:
                self._admission_order.append(company_id)
            return

        async def _update() -> None:
            async with self._lock:
                if company_id not in self._admission_order:
                    self._admission_order.append(company_id)

        loop.create_task(_update())

    async def notify_completed(self, company_id: str) -> None:
        """
        Inform the scheduler that a company finished (successfully or
        permanently failed). Removes it from internal tracking structures.
        """
        async with self._lock:
            if company_id in self._admission_order:
                self._admission_order = [
                    cid for cid in self._admission_order if cid != company_id
                ]
            if company_id in self._memory_priority_queue:
                self._memory_priority_queue = [
                    cid for cid in self._memory_priority_queue if cid != company_id
                ]

    # ------------------------------------------------------------------ #
    # Public API for caller: host-level memory cancellation
    # ------------------------------------------------------------------ #

    async def maybe_select_company_to_cancel(
        self,
        active_company_ids: Iterable[str],
    ) -> Optional[str]:
        """
        Based purely on host memory (not CPU), decide whether a running
        company should be cancelled to avoid OOM, and if so which one.

        Behavior:
          - Only acts if mem_cancel_enabled is True and psutil is available.
          - Only acts if last_raw_mem (or avg_mem) >= mem_critical_limit.
          - Enforced cooldown:
                cancel -> wait mem_cancel_cooldown_sec
                -> require a new sample
                -> maybe cancel another.
          - Picks the newest active company according to admission order.
          - After choosing a victim:
                * increments memory_pressure_hits
                * sets _next_mem_cancel_ts
                * sets _blocked_until_ts for mem_post_relief_block_sec seconds.

        Returns:
          - company_id to cancel, or None if no cancellation is needed.
        """
        if not self.cfg.mem_cancel_enabled or not PSUTIL_AVAILABLE:
            return None

        now = time.monotonic()
        active_set = set(active_company_ids)
        if not active_set:
            return None

        async with self._lock:
            # Require a new sample after the last cancellation.
            if self._last_sample_mono <= self._last_cancel_sample_mono:
                return None

            # Enforce cooldown between cancellations.
            if now < self._next_mem_cancel_ts:
                return None

            mem_frac = self._last_raw_mem or self._last_avg_mem
            if mem_frac < self.cfg.mem_critical_limit:
                return None

            # Choose "newest" active company: last admitted that is still active.
            victim: Optional[str] = None
            for cid in reversed(self._admission_order):
                if cid in active_set:
                    victim = cid
                    break

            # Fallback: pick any active company (e.g. last one).
            if victim is None:
                victim = next(iter(active_set))

            self._memory_pressure_hits += 1
            self._last_cancel_sample_mono = self._last_sample_mono
            self._next_mem_cancel_ts = now + float(self.cfg.mem_cancel_cooldown_sec)

            # Block new admissions for a relief window (for example 3 minutes).
            relief_until = now + float(self.cfg.mem_post_relief_block_sec)
            self._blocked_until_ts = max(self._blocked_until_ts, relief_until)

            logger.error(
                "[AdaptiveScheduling] host memory critical (mem=%.3f >= %.3f); "
                "selected company_id=%s for cancellation; "
                "cooldown_until=%.1fs, blocked_until=%.1fs (monotonic zero).",
                mem_frac,
                self.cfg.mem_critical_limit,
                victim,
                self._next_mem_cancel_ts,
                self._blocked_until_ts,
            )

            return victim

    # ------------------------------------------------------------------ #
    # Public API for caller: memory priority queue
    # ------------------------------------------------------------------ #

    async def register_memory_requeued(self, company_id: str) -> None:
        """
        Register that a company was cancelled due to memory and requeued.
        The scheduler maintains a priority queue so that such companies
        can resume as soon as admissions allow.

        Caller should invoke this after:
          - cancelling the company's Task
          - persisting its progress
          - marking it as NOT_DONE (for example markdown not done)
        """
        async with self._lock:
            if company_id not in self._memory_priority_queue:
                self._memory_priority_queue.append(company_id)
                logger.info(
                    "[AdaptiveScheduling] memory-priority requeue company_id=%s",
                    company_id,
                )

    async def pop_priority_company(
        self,
        pending_company_ids: Iterable[str],
    ) -> Optional[str]:
        """
        Pop the next memory-priority company that is still pending.

        pending_company_ids:
            Set/list of companies that are eligible to run (not completed).

        Returns:
            company_id to run next with priority, or None if there is no
            matching entry.
        """
        pending = set(pending_company_ids)
        async with self._lock:
            if not self._memory_priority_queue:
                return None

            # Find first in queue that is still pending.
            for cid in list(self._memory_priority_queue):
                if cid in pending:
                    self._memory_priority_queue.remove(cid)
                    logger.info(
                        "[AdaptiveScheduling] memory-priority pop company_id=%s",
                        cid,
                    )
                    return cid

            # Nothing in queue matches current pending set.
            return None

    # ------------------------------------------------------------------ #
    # Public API for caller / legacy: external memory pressure
    # ------------------------------------------------------------------ #

    async def on_memory_pressure(self, severity: str = "hard") -> None:
        """
        Called when an external component detects a critical memory event.

        This does not touch any "limit". Instead, it:

          - increments a counter for diagnostics
          - blocks all new admissions for mem_pressure_block_sec seconds
            (for soft / hard) or mem_post_relief_block_sec for emergency.
        """
        async with self._lock:
            self._memory_pressure_hits += 1
            now = time.monotonic()

            if severity == "emergency":
                # More conservative block when emergency triggered from outside.
                extra = max(
                    float(self.cfg.mem_post_relief_block_sec),
                    float(self.cfg.mem_pressure_block_sec),
                )
            else:
                extra = float(self.cfg.mem_pressure_block_sec)

            self._blocked_until_ts = max(self._blocked_until_ts, now + extra)
            logger.warning(
                "[AdaptiveScheduling] external memory pressure event "
                "(severity=%s, hits=%d); blocking new admissions until %.1fs "
                "from monotonic zero.",
                severity,
                self._memory_pressure_hits,
                self._blocked_until_ts,
            )

    async def get_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of internal state for debugging / monitoring.

        The snapshot includes:
          - CPU/memory averages and last raw values
          - current band
          - rush mode flags
          - memory pressure and cancellation state
          - admission order and memory-priority queue
          - host-level stall suspicion and supporting fields
        """
        async with self._lock:
            band = self._classify_band_locked() if self._samples else "unknown"
            return {
                "max_concurrency": self.cfg.max_concurrency,
                "last_avg_cpu": self._last_avg_cpu,
                "last_avg_mem": self._last_avg_mem,
                "last_raw_cpu": self._last_raw_cpu,
                "last_raw_mem": self._last_raw_mem,
                "samples_count": len(self._samples),
                "last_update_ts": self._last_update_ts,
                "last_sample_mono": self._last_sample_mono,
                "last_admission_ts": self._last_admission_ts,
                "last_cautious_admission_ts": self._last_cautious_admission_ts,
                "last_high_block_ts": self._last_high_block_ts,
                "blocked_until_ts": self._blocked_until_ts,
                "memory_pressure_hits": self._memory_pressure_hits,
                "band": band,
                "rush_active": self._rush_active,
                "rush_start_ts": self._rush_start_ts,
                "rush_until_ts": self._rush_until_ts,
                "rush_start_cpu": self._rush_start_cpu,
                "rush_start_mem": self._rush_start_mem,
                "next_mem_cancel_ts": self._next_mem_cancel_ts,
                "last_cancel_sample_mono": self._last_cancel_sample_mono,
                "admission_order": list(self._admission_order),
                "memory_priority_queue": list(self._memory_priority_queue),
                "host_stall_enabled": self.cfg.host_stall_enabled,
                "host_stall_suspected": self._host_stall_suspected,
                "host_stall_since_mono": self._host_stall_since_mono,
                "host_stall_reason": self._host_stall_reason,
                "host_stall_last_denial_reason": self._host_stall_last_denial_reason,
                "host_stall_last_denial_ts": self._host_stall_last_denial_ts,
            }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_usage(self) -> Tuple[float, float]:
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0

        cpu_frac = 0.0
        mem_frac = 0.0

        try:
            cpu_pct = psutil.cpu_percent(interval=None)  # type: ignore[union-attr]
            cpu_frac = max(0.0, min(1.0, float(cpu_pct) / 100.0))
        except Exception:
            pass

        try:
            vm = psutil.virtual_memory()  # type: ignore[union-attr]
            mem_frac = max(0.0, min(1.0, float(vm.percent) / 100.0))
        except Exception:
            pass

        return cpu_frac, mem_frac

    async def _sample_once(self) -> None:
        """
        Sample CPU and memory once and update smoothed averages.
        Also updates rush mode based on plateau detection and keeps
        a separate, longer window for host-level stall plateau detection.
        """
        if not PSUTIL_AVAILABLE:
            # When psutil is missing, keep existing averages and simply
            # log once that we do not have usage metrics.
            async with self._lock:
                if self._last_update_ts == 0.0:
                    self._last_update_ts = time.time()
                    logger.warning(
                        "[AdaptiveScheduling] psutil not available; "
                        "admissions will not be based on real CPU/memory.",
                    )
            return

        raw_cpu, raw_mem = self._read_usage()
        now_mono = time.monotonic()
        wall_ts = time.time()
        cfg = self.cfg

        window = (
            float(cfg.smoothing_window_sec) if cfg.smoothing_window_sec > 0.0 else 0.0
        )

        async with self._lock:
            # Short window for admission decisions / rush.
            self._update_samples_locked(now_mono=now_mono, cpu=raw_cpu, mem=raw_mem)
            avg_cpu = self._last_avg_cpu
            avg_mem = self._last_avg_mem
            self._last_raw_cpu = raw_cpu
            self._last_raw_mem = raw_mem
            self._last_update_ts = wall_ts
            self._last_sample_mono = now_mono

            # Rush mode plateau detection.
            self._update_rush_mode_locked(now_mono=now_mono)

            # Longer window for host-level stall plateau detection.
            self._update_stall_plateau_locked(
                now_mono=now_mono, cpu=raw_cpu, mem=raw_mem
            )

            state_snapshot = {
                "ts": wall_ts,
                "monotonic": now_mono,
                "avg_cpu": avg_cpu,
                "avg_mem": avg_mem,
                "raw_cpu": raw_cpu,
                "raw_mem": raw_mem,
                "samples_window_sec": window,
                "samples_count": len(self._samples),
                "memory_pressure_hits": self._memory_pressure_hits,
                "rush_active": self._rush_active,
                "host_stall_suspected": self._host_stall_suspected,
                "host_stall_reason": self._host_stall_reason,
            }

        self._maybe_log_state(state_snapshot)

    def _update_samples_locked(self, now_mono: float, cpu: float, mem: float) -> None:
        """
        Update sliding window and averages for admission / rush logic.
        Caller must hold _lock.
        """
        window = float(self.cfg.smoothing_window_sec)
        self._samples.append((now_mono, cpu, mem))

        if window > 0.0:
            cutoff = now_mono - window
            self._samples = [(t, c, m) for (t, c, m) in self._samples if t >= cutoff]
        else:
            # Only keep the latest sample when smoothing is disabled.
            self._samples = self._samples[-1:]

        if not self._samples:
            self._last_avg_cpu = cpu
            self._last_avg_mem = mem
            return

        sum_cpu = 0.0
        sum_mem = 0.0
        for _, c, m in self._samples:
            sum_cpu += c
            sum_mem += m
        count = len(self._samples)
        self._last_avg_cpu = sum_cpu / count
        self._last_avg_mem = sum_mem / count

    def _classify_band_locked(self) -> str:
        """
        Classify current load into 'low', 'cautious', or 'high'.
        Caller must hold _lock.
        """
        if not self._samples:
            return "low"

        cfg = self.cfg
        use_cpu = cfg.use_cpu
        use_mem = cfg.use_mem
        cpu = self._last_avg_cpu
        mem = self._last_avg_mem

        # High band: any metric at or above its high threshold.
        high = False
        if use_cpu and cpu >= cfg.target_cpu_high:
            high = True
        if use_mem and mem >= cfg.target_mem_high:
            high = True
        if high:
            return "high"

        # Low band: both metrics at or below low thresholds (if enabled).
        low_cpu_ok = (not use_cpu) or cpu <= cfg.target_cpu_low
        low_mem_ok = (not use_mem) or mem <= cfg.target_mem_low
        if low_cpu_ok and low_mem_ok:
            return "low"

        # Otherwise we are in the cautious band.
        return "cautious"

    def _update_rush_mode_locked(self, now_mono: float) -> None:
        """
        Update rush mode state based on plateau detection and duration.

        Rush mode is entered when CPU and memory are below configured
        thresholds and fluctuate inside a ±rush_band_width band for
        at least rush_min_samples samples.

        When rush mode ends, log how CPU and memory changed over that
        window.
        """
        cfg = self.cfg
        if not cfg.rush_enabled:
            return

        if not self._samples:
            return

        # First, handle rush window completion.
        if self._rush_active and now_mono >= self._rush_until_ts:
            avg_cpu = self._last_avg_cpu
            avg_mem = self._last_avg_mem
            cpu_delta = avg_cpu - self._rush_start_cpu
            mem_delta = avg_mem - self._rush_start_mem

            logger.info(
                "[AdaptiveScheduling] rush window complete: "
                "cpu %.3f -> %.3f (delta=%.3f), "
                "mem %.3f -> %.3f (delta=%.3f)",
                self._rush_start_cpu,
                avg_cpu,
                cpu_delta,
                self._rush_start_mem,
                avg_mem,
                mem_delta,
            )

            self._rush_active = False
            self._rush_start_ts = 0.0
            self._rush_until_ts = 0.0
            return

        # If already in rush and not finished, do not try to restart.
        if self._rush_active:
            return

        # Not in rush: check for underutilized plateau.
        if len(self._samples) < max(1, cfg.rush_min_samples):
            return

        cpus = [c for (_, c, _) in self._samples]
        mems = [m for (_, _, m) in self._samples]

        min_cpu = min(cpus)
        max_cpu = max(cpus)
        min_mem = min(mems)
        max_mem = max(mems)

        avg_cpu = self._last_avg_cpu
        avg_mem = self._last_avg_mem

        cpu_range = max_cpu - min_cpu
        mem_range = max_mem - min_mem

        plateau_cpu = (
            avg_cpu < cfg.rush_cpu_threshold and cpu_range <= cfg.rush_band_width
        )
        plateau_mem = (
            avg_mem < cfg.rush_mem_threshold and mem_range <= cfg.rush_band_width
        )

        if plateau_cpu and plateau_mem:
            # Enter rush mode.
            self._rush_active = True
            self._rush_start_ts = now_mono
            self._rush_until_ts = now_mono + float(cfg.rush_duration_sec)
            self._rush_start_cpu = avg_cpu
            self._rush_start_mem = avg_mem

            logger.info(
                "[AdaptiveScheduling] entering rush mode "
                "(avg_cpu=%.3f range=%.3f, avg_mem=%.3f range=%.3f, "
                "duration=%.1fs)",
                avg_cpu,
                cpu_range,
                avg_mem,
                mem_range,
                cfg.rush_duration_sec,
            )

    def _update_stall_plateau_locked(
        self, now_mono: float, cpu: float, mem: float
    ) -> None:
        """
        Maintain a longer sliding window for host-level stall plateau detection.

        Definition:
          - We look at the last host_stall_window_sec seconds of raw
            CPU and memory samples (0..1 fractions).
          - If both CPU and memory stay within a narrow band
            (<= host_stall_band_width) AND there have been no admissions
            for at least host_stall_min_duration_sec, we mark a stall
            due to plateau, regardless of absolute usage (0..100 percent).

        This does not raise; it only updates internal flags that appear
        in get_state().
        """
        cfg = self.cfg
        if not cfg.host_stall_enabled:
            return

        window = float(cfg.host_stall_window_sec)
        if window <= 0.0:
            return

        self._stall_samples.append((now_mono, cpu, mem))
        cutoff = now_mono - window
        self._stall_samples = [
            (t, c, m) for (t, c, m) in self._stall_samples if t >= cutoff
        ]

        if len(self._stall_samples) < 2:
            return

        cpus = [c for (_, c, _) in self._stall_samples]
        mems = [m for (_, _, m) in self._stall_samples]

        min_cpu = min(cpus)
        max_cpu = max(cpus)
        min_mem = min(mems)
        max_mem = max(mems)

        cpu_range = max_cpu - min_cpu
        mem_range = max_mem - min_mem

        band = float(cfg.host_stall_band_width)
        if band <= 0.0:
            return

        # Total span of the stall window we currently hold.
        span = self._stall_samples[-1][0] - self._stall_samples[0][0]
        min_duration = float(cfg.host_stall_min_duration_sec)

        if span >= min_duration and cpu_range <= band and mem_range <= band:
            # Additional guard: do not mark plateau stall if we have
            # been admitting new companies recently; use last_admission_ts
            # as a crude proxy for "progress".
            no_recent_admissions = (
                self._last_admission_ts > 0.0
                and (now_mono - self._last_admission_ts) >= min_duration
            )
            if no_recent_admissions and not self._host_stall_suspected:
                self._host_stall_suspected = True
                self._host_stall_since_mono = self._stall_samples[0][0]
                if not self._host_stall_reason:
                    self._host_stall_reason = "cpu_mem_plateau"
                logger.error(
                    "[AdaptiveScheduling] host stall suspected due to CPU/memory plateau: "
                    "cpu_range=%.3f mem_range=%.3f span=%.1fs (band<=%.3f, min_duration=%.1fs)",
                    cpu_range,
                    mem_range,
                    span,
                    band,
                    min_duration,
                )

    def _update_decision_stall_locked(
        self,
        *,
        active_tasks: int,
        admitted: bool,
        reason: str,
        now: float,
    ) -> None:
        """
        Update host-level stall detection based on admission decisions.

        Condition:
          - host_stall_enabled is True
          - active_tasks == 0
          - we repeatedly deny admissions with the same reason
          - this persists for at least host_stall_min_duration_sec

        When triggered, sets host_stall_suspected=True and fills
        host_stall_reason="no_active_tasks_with_denials:<reason>".
        """
        if not self.cfg.host_stall_enabled:
            return

        if admitted:
            # Any successful admission resets the denial-based stall timer.
            self._host_stall_last_denial_ts = 0.0
            self._host_stall_last_denial_reason = ""
            return

        # Only consider the "active=0, repeated denial" pattern described
        # in the logs:
        #
        #   INFO: [Scheduler] not starting new company
        #         (active=0, reason=high_band_cooldown, remaining=...)
        #
        if active_tasks != 0:
            return

        min_duration = float(self.cfg.host_stall_min_duration_sec)
        if min_duration <= 0.0:
            return

        if self._host_stall_last_denial_ts == 0.0:
            # First denial in a possible streak.
            self._host_stall_last_denial_ts = now
            self._host_stall_last_denial_reason = reason
            return

        # If the reason changed, start a new streak.
        if reason != self._host_stall_last_denial_reason:
            self._host_stall_last_denial_ts = now
            self._host_stall_last_denial_reason = reason
            return

        elapsed = now - self._host_stall_last_denial_ts
        if elapsed < 0.0:
            return

        if elapsed >= min_duration and not self._host_stall_suspected:
            self._host_stall_suspected = True
            self._host_stall_since_mono = self._host_stall_last_denial_ts
            if not self._host_stall_reason:
                self._host_stall_reason = f"no_active_tasks_with_denials:{reason}"
            logger.error(
                "[AdaptiveScheduling] host stall suspected: "
                "active_tasks=0, repeated denial reason=%s for %.1fs "
                "(min_duration=%.1fs)",
                reason,
                elapsed,
                min_duration,
            )

    def _maybe_log_state(self, state: Dict[str, Any]) -> None:
        """
        Optional structured logging of sampling state.
        """
        if self.cfg.log_path is None:
            return

        try:
            self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.log_path.open("a", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            logger.debug(
                "[AdaptiveScheduling] failed writing log to %s",
                self.cfg.log_path,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Lightweight page-level MemoryGuard (marker only)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryGuardConfig:
    """
    Lightweight configuration for MemoryGuard.

    This variant only deals with page-level memory markers from Crawl4AI.

    - marker:
        String that Crawl4AI puts into page errors on critical pressure.

    - severity_for_marker:
        Severity to use when the marker is detected. Default: "soft".
    """

    marker: str = MEMORY_PRESSURE_MARKER
    severity_for_marker: str = "soft"


class MemoryGuard:
    """
    Lightweight memory guard used at the page level.

    Responsibilities:
      - Inspect page level `error` objects.
      - If the configured marker is present, mark the company for memory
        retry and raise CriticalMemoryPressure(severity="soft") to abort
        the current company gracefully, leaving the run alive.

    Host-level memory protection (when to cancel companies based on
    overall RAM usage) is handled by AdaptiveScheduler and not by this
    class anymore.
    """

    def __init__(self, config: Optional[MemoryGuardConfig] = None) -> None:
        self.config = config or MemoryGuardConfig()

    def check_page_error(
        self,
        *,
        error: Any,
        company_id: str,
        url: Optional[str],
        mark_company_memory: Callable[[str], None],
    ) -> None:
        """
        Inspect a page level `error` and, if the marker is present:

          - mark the company via mark_company_memory(company_id)
          - raise CriticalMemoryPressure(severity="soft")

        Host-wide memory checks are intentionally not performed here;
        they are handled by AdaptiveScheduler via psutil.
        """
        if not isinstance(error, str):
            return

        if self.config.marker not in error:
            return

        logger.error(
            "Critical memory pressure marker detected for company_id=%s url=%s; "
            "marking for retry and signalling abort of this company.",
            company_id,
            url,
        )

        mark_company_memory(company_id)

        raise CriticalMemoryPressure(
            f"Critical memory pressure for company_id={company_id} url={url!r}",
            severity=self.config.severity_for_marker,
        )


# ---------------------------------------------------------------------------
# StallGuard: async company level monitor (merged from stall_guard.py)
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
    Immutable snapshot of a company level stall.
    """

    detected_at: str
    company_id: str
    idle_seconds: float
    last_progress_at: Optional[str]
    last_event: Optional[str]
    reason: str


class StallDetectedError(RuntimeError):
    """Raised when StallGuard detects a stall (optional use)."""


class GlobalStallDetectedError(RuntimeError):
    """Raised when wait_for_global_hard_stall detects a global stall."""


@dataclass(slots=True)
class _CompanyState:
    company_id: str
    last_progress_mono: Optional[float] = None
    last_progress_wall: Optional[datetime] = None
    last_event: Optional[str] = None
    active: bool = True
    stalled: bool = False


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

        # Global progress trackers (across all companies, active or not)
        self._last_any_progress_mono: Optional[float] = None
        self._last_any_progress_wall: Optional[datetime] = None

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
            # Called from a context without an event loop.
            self._running = False
            self._log.error(
                "StallGuard.start() called without a running event loop; guard not started",
            )
            return

        self._monitor_task = loop.create_task(
            self._monitor_loop(),
            name="stall-guard-monitor",
        )

        self._log.info(
            "StallGuard started (company level): page_timeout=%.1fs soft>=%.1fs hard>=%.1fs interval=%.1fs",
            self.config.page_timeout_sec,
            self.config.soft_timeout_sec,
            self.config.hard_timeout_sec,
            self.config.check_interval_sec,
        )

    async def stop(self) -> None:
        """
        Stop the background monitor task.

        This is best effort and will not raise if the task is already done
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

    def last_any_progress_age(self) -> Optional[float]:
        """
        Age in seconds since the last progress event on any company,
        active or inactive. None if no progress has ever been recorded.
        """
        if self._last_any_progress_mono is None:
            return None
        age = time.monotonic() - self._last_any_progress_mono
        if age < 0:
            # Monotonic clock should not go backwards, but be defensive.
            return 0.0
        return age

    def active_company_count(self) -> int:
        """
        Current number of companies marked as active.
        """
        return sum(1 for st in self._companies.values() if st.active)

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

        - If company_id is provided, it is treated as company level progress.
        - If company_id is None, it is treated as a global heartbeat and does
        not affect stall detection.
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

        # Global progress trackers
        self._last_any_progress_mono = now_mono
        self._last_any_progress_wall = now_wall

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
# Global hard stall helper (merged from stall_guard.py)
# ---------------------------------------------------------------------------


async def wait_for_global_hard_stall(
    stall_guard: StallGuard,
    *,
    page_timeout_sec: float,
    factor: float = 4.5,
    check_interval_sec: float = 30.0,
    raise_on_stall: bool = False,
) -> float:
    """
    Wait until the whole run appears globally stalled.

    Definition here:
      - StallGuard.last_progress_age() returns the minimum idle time across
        all active companies.
      - If that age is greater than or equal to page_timeout_sec * factor,
        we consider the system globally stalled.

    In addition, this helper also detects a "zombie" condition where:

      * there are active companies but no per company timestamps at all, or
      * there are zero active companies but no progress anywhere for a long
        time.

    In both cases it falls back to the age since the last progress on any
    company and returns once that exceeds the same hard threshold.

    If raise_on_stall is True, the helper raises GlobalStallDetectedError
    instead of just returning the age. This is useful if you want to abort
    the whole run on a global stall and let an outer retry wrapper restart
    the process.
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

    # Track how long we have been waiting when there is no progress at all.
    start_mono = time.monotonic()

    while True:
        await asyncio.sleep(check_interval_sec)

        age_active = stall_guard.last_progress_age()
        age_any = stall_guard.last_any_progress_age()
        active_count = stall_guard.active_company_count()

        # New: handle the "no progress at all yet" zombie case
        if age_active is None and age_any is None:
            elapsed = time.monotonic() - start_mono

            if active_count > 0:
                logger.warning(
                    "Global StallGuard: %d active companies but no progress "
                    "recorded yet; elapsed=%.1fs (hard>=%.1fs)",
                    active_count,
                    elapsed,
                    hard,
                )
            else:
                logger.debug(
                    "Global StallGuard: no progress recorded yet; "
                    "elapsed=%.1fs (hard>=%.1fs)",
                    elapsed,
                    hard,
                )

            if elapsed >= hard:
                msg = (
                    "Global StallGuard: zombie stall detected before first "
                    "heartbeat; no progress for %.1fs >= %.1fs" % (elapsed, hard)
                )
                logger.error(msg)
                if raise_on_stall:
                    raise GlobalStallDetectedError(msg)
                return elapsed

            continue

        # Normal global hard stall: at least one active company with a known
        # progress timestamp and that idle time exceeds the threshold.
        if age_active is not None:
            logger.debug(
                "Global StallGuard: min idle across ACTIVE companies = %.1fs (hard>=%.1fs)",
                age_active,
                hard,
            )
            if age_active >= hard:
                msg = (
                    "Global StallGuard: hard stall detected, "
                    "min idle (active) = %.1fs >= %.1fs" % (age_active, hard)
                )
                logger.error(msg)
                if raise_on_stall:
                    raise GlobalStallDetectedError(msg)
                return age_active

        # Zombie branch 1: active companies but last_progress_age() could not
        # compute anything, fall back to global last progress age.
        if active_count > 0 and age_active is None and age_any is not None:
            logger.warning(
                "Global StallGuard: %d active companies but no per company "
                "progress timestamps; using last_any_progress_age=%.1fs "
                "(hard>=%.1fs) for stall detection",
                active_count,
                age_any,
                hard,
            )
            if age_any >= hard:
                msg = (
                    "Global StallGuard: zombie stall detected "
                    "(active companies, no progress for %.1fs >= %.1fs)"
                    % (age_any, hard)
                )
                logger.error(msg)
                if raise_on_stall:
                    raise GlobalStallDetectedError(msg)
                return age_any

        # Zombie branch 2: no active companies, but there has been no progress
        # anywhere for a long time. This can happen if the outer logic is stuck.
        if active_count == 0 and age_any is not None:
            logger.debug(
                "Global StallGuard: zero active companies, "
                "last_any_progress_age=%.1fs (hard>=%.1fs)",
                age_any,
                hard,
            )
            if age_any >= hard:
                msg = (
                    "Global StallGuard: zombie stall detected with zero active "
                    "companies; no progress for %.1fs >= %.1fs" % (age_any, hard)
                )
                logger.error(msg)
                if raise_on_stall:
                    raise GlobalStallDetectedError(msg)
                return age_any

        # Otherwise we are either making progress or not yet past the
        # threshold; keep waiting.


async def global_stall_watchdog(
    *,
    stall_guard: StallGuard,
    stall_cfg: StallGuardConfig,
    company_tasks: Dict[str, asyncio.Task[Any]],
    mark_company_stalled: Callable[[str], None],
    on_global_stall: Optional[Callable] = None,
) -> None:
    """
    Background task that waits for a global hard stall and then:

      - optionally notifies the caller via `on_global_stall(idle_seconds)`
      - marks all active companies as stalled via `mark_company_stalled`
      - cancels their running asyncio tasks

    This is the lifted version of the old `_global_stall_watchdog()` from run.py.

    Parameters
    ----------
    stall_guard:
        The StallGuard instance monitoring per company progress.
    stall_cfg:
        Its StallGuardConfig, used to pick page_timeout and check interval.
    company_tasks:
        Mapping company_id -> asyncio.Task for active company pipelines.
    mark_company_stalled:
        Callback that records stalled companies in the retry tracker or logs.
    on_global_stall:
        Optional callback that is invoked once with the detected idle duration
        (in seconds) before cancelling tasks. You can use this to set an
        `abort_run` flag in the caller.
    """
    try:
        idle = await wait_for_global_hard_stall(
            stall_guard,
            page_timeout_sec=stall_cfg.page_timeout_sec,
            check_interval_sec=stall_cfg.check_interval_sec,
        )
    except asyncio.CancelledError:
        return
    except GlobalStallDetectedError:
        # Already handled inside wait_for_global_hard_stall when raise_on_stall=True.
        return
    except Exception as e:
        logger.exception(
            "Global StallGuard watchdog encountered an unexpected error: %s",
            e,
        )
        return

    # Let the caller flip any outer flags before we touch tasks.
    if on_global_stall is not None:
        try:
            on_global_stall(idle)
        except Exception as e:
            logger.exception("Global stall callback failed: %s", e)

    logger.error(
        "Global StallGuard: no company level progress for %.1fs, treating this run "
        "as globally stalled and marking active companies for retry.",
        idle,
    )

    active_company_ids: List[str] = []
    for cid, task in list(company_tasks.items()):
        if not task.done():
            active_company_ids.append(cid)

    if not active_company_ids:
        logger.error(
            "Global StallGuard: no active company tasks found at stall detection time.",
        )

    for cid in active_company_ids:
        mark_company_stalled(cid)

    for cid in active_company_ids:
        task = company_tasks.get(cid)
        if task is not None and not task.done():
            logger.error(
                "Global StallGuard: cancelling company task company_id=%s",
                cid,
            )
            task.cancel()


__all__ = [
    "AdaptiveSchedulingConfig",
    "AdaptiveScheduler",
    "MemoryGuardConfig",
    "MemoryGuard",
    "CriticalMemoryPressure",
    "MEMORY_PRESSURE_MARKER",
    "StallGuardConfig",
    "StallSnapshot",
    "StallGuard",
    "StallDetectedError",
    "GlobalStallDetectedError",
    "wait_for_global_hard_stall",
    "global_stall_watchdog",
]
