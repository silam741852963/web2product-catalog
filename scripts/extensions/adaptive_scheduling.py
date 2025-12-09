from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import asyncio
import gc
import json
import logging
import time

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
# Optional marker and exception kept for compatibility with callers
# ---------------------------------------------------------------------------

MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"


class CriticalMemoryPressure(RuntimeError):
    """
    Raised by external components when memory pressure is high enough that
    the current company task should be aborted.

    This module no longer uses it internally but keeps the type so that
    other parts of the codebase can continue to import it.
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
    Memory-based adaptive scheduling configuration.

    All thresholds are expressed as fractions of the effective memory limit
    (host or cgroup).
    """

    mem_cap_frac: float = 0.85
    mem_high_frac: float = 0.90
    mem_crit_high_frac: float = 0.95
    mem_crit_low_frac: float = 0.90

    min_active_keep: int = 0

    peak_history_size: int = 100
    per_company_safety_factor: float = 1.3
    per_company_min_reservation_mb: float = 256.0

    emergency_check_interval_sec: float = 1.0

    log_path: Optional[Path] = None
    use_psutil: bool = True

    # (1) Additional signals: cgroup + swap
    prefer_cgroup_limits: bool = True
    swap_block_frac: float = 0.60
    swap_emergency_frac: float = 0.80

    # (2) Trend based throttling
    mem_trend_window_sec: float = 10.0
    mem_trend_slope_high_mb_per_s: float = 100.0
    mem_trend_margin_frac: float = 0.03

    # Extra trend threshold for emergency preemption
    mem_trend_emergency_slope_mb_per_s: float = 200.0

    # (4) AIMD concurrency controller
    ai_step: int = 2
    md_factor: float = 0.6
    min_target: int = 1
    max_target: int = 512

    # Faster ramp-up at the start of the run
    initial_target: int = 4
    warmup_updates: int = 10
    warmup_ai_step: int = 8

    # (5) Safety margin auto-tuning
    near_oom_used_frac: float = 0.97
    near_oom_swap_frac: float = 0.50
    near_oom_mem_cap_step: float = 0.01
    mem_cap_min_frac: float = 0.70
    mem_high_min_frac: float = 0.80
    mem_crit_min_frac: float = 0.90

    # (3) Per-company profiles
    company_profile_max_size: int = 2000

    # Unstall heuristics
    # If slots_by_mem is too low but used_frac is clearly below this, we relax estimates.
    unstall_low_frac: float = 0.70
    # Cooldown between automatic resets in seconds
    unstall_cooldown_sec: float = 60.0
    # Treat stalls also when slots_by_mem is small but not zero
    unstall_slots_threshold: int = 1
    # Margin from mem_cap_frac to still consider as "safe" for unstall
    unstall_margin_frac: float = 0.05

    # Emergency cancellation behavior
    # Minimum time between two emergency cancellations
    emergency_cancel_cooldown_sec: float = 3.0
    # Time to wait after canceling companies to allow memory to drop
    emergency_post_cancel_delay_sec: float = 1.5
    # Max number of companies to cancel in one super critical step
    max_emergency_cancel_per_step: int = 3
    # Number of emergency rounds after which we recommend restart
    emergency_persistent_rounds_threshold: int = 3

    # Peak history: time horizon for considering samples (seconds)
    peak_history_horizon_sec: float = 1800.0  # 30 minutes

    # Plateau rescue heuristics
    plateau_detect_sec: float = 1800.0  # 30 minutes of "stuck" behavior
    plateau_min_num_waiting: int = 100
    plateau_slots_threshold: int = 1
    plateau_used_margin_frac: float = 0.05
    plateau_shrink_factor: float = 0.7

    # Full estimator reset when we have not been in a healthy regime for a long time
    estimator_reset_interval_sec: float = 3600.0  # 1 hour
    estimator_reset_parallel_threshold: int = 16
    estimator_reset_used_frac_threshold: float = 0.80
    estimator_reset_factor: float = 1.5

    # Global hard cap for per-company estimate (in MB)
    per_company_max_reservation_mb: float = 1024.0


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class AdaptiveScheduler:
    """
    Memory based adaptive scheduler with:

      - Host/cgroup + swap awareness
      - Memory trend estimation
      - Optional per-company heaviness profiles
      - AIMD-style concurrency target
      - Safety margin auto-tuning
      - Stall / plateau rescue to avoid poisoned estimates
    """

    def __init__(
        self,
        cfg: AdaptiveSchedulingConfig,
        get_active_company_ids: Callable[[], Sequence[str]],
        request_cancel_companies: Callable[[Sequence[str]], None],
    ) -> None:
        self.cfg = cfg

        self._psutil_available: bool = bool(PSUTIL_AVAILABLE and cfg.use_psutil)

        # Callbacks into the caller
        self._get_active_company_ids = get_active_company_ids
        self._request_cancel_companies = request_cancel_companies

        # Effective total memory limit (host or cgroup) in bytes
        self._total_mem_bytes: Optional[int] = None

        # History of approximate per company usage in MB: list of (timestamp, value)
        self._peak_history_mb: List[Tuple[float, float]] = []

        # Estimated per company peak memory in MB (global)
        self._per_company_est_mb: float = cfg.per_company_min_reservation_mb

        # Per-company heaviness (company_id -> peak_mb), simple LRU
        self._company_peak_mb: Dict[str, float] = {}
        self._company_order: List[str] = []

        # Async emergency watchdog task
        self._emergency_task: Optional[asyncio.Task] = None

        # Lock protecting internal state and estimation history
        self._lock = asyncio.Lock()

        # Flag that a controlled restart is recommended
        self._restart_recommended: bool = False

        # Last timestamp we wrote a JSON state line to disk
        self._last_log_ts: float = 0.0

        # Cache last memory snapshot for get_state_snapshot
        self._last_used_bytes: int = 0
        self._last_used_frac: float = 0.0
        self._last_swap_used_frac: float = 0.0

        # Trend estimation
        self._mem_samples: List[Tuple[float, int]] = []
        self._last_trend_slope_mb_s: float = 0.0

        # AIMD concurrency target (companies)
        self._target_parallel: int = min(
            max(self.cfg.min_target, self.cfg.initial_target), self.cfg.max_target
        )

        # Count how many times we've updated target_parallel (for warmup)
        self._update_calls: int = 0

        # Near-OOM bookkeeping for safety margin auto-tuning
        self._near_oom_events: int = 0
        self._last_near_oom_flag: bool = False

        # Unstall bookkeeping
        self._last_unstall_ts: float = 0.0

        # Emergency cancellation bookkeeping
        self._last_emergency_cancel_ts: float = 0.0
        self._emergency_rounds: int = 0

        # Base RSS estimate (memory used when no companies are active)
        self._base_rss_bytes: float = 0.0
        self._base_rss_samples: int = 0

        # Plateau detection window: list of (ts, num_active, slots_by_mem, used_frac, near_oom_events)
        self._plateau_window: List[Tuple[float, int, int, float, int]] = []

        # Last time we observed a "healthy" period (for full estimator reset)
        self._last_healthy_ts: float = time.time()

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def psutil_available(self) -> bool:
        return self._psutil_available

    @property
    def restart_recommended(self) -> bool:
        return self._restart_recommended

    # ------------------------------------------------------------------ #
    # Public API for per-company profiles
    # ------------------------------------------------------------------ #

    def record_company_peak(self, company_id: str, peak_mb: float) -> None:
        """
        Optional hint from the caller: record a per-company peak memory estimate
        (in MB). This is used to tighten the global estimate for future runs.
        """
        if not company_id or peak_mb <= 0:
            return

        prev = self._company_peak_mb.get(company_id)
        if prev is None or peak_mb > prev:
            self._company_peak_mb[company_id] = peak_mb

        if company_id not in self._company_order:
            self._company_order.append(company_id)

        max_size = max(1, int(self.cfg.company_profile_max_size))
        if len(self._company_order) > max_size:
            oldest = self._company_order.pop(0)
            self._company_peak_mb.pop(oldest, None)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Initialize memory information and start the emergency watchdog loop.
        """
        if self._psutil_available:
            total, used = self._read_memory_usage_bytes()
            self._total_mem_bytes = total or None
            self._last_used_bytes = used
            self._last_used_frac = (float(used) / total) if total > 0 else 0.0

            logger.info(
                "[AdaptiveScheduling] started (total_mem_mb=%.1f, psutil=True, "
                "initial_target_parallel=%d)",
                (float(total) / 1e6) if total > 0 else -1.0,
                self._target_parallel,
            )

            interval = max(0.5, float(self.cfg.emergency_check_interval_sec))
            self._emergency_task = asyncio.create_task(
                self._emergency_loop(interval),
                name="adaptive-scheduling-emergency-watchdog",
            )
        else:
            logger.warning(
                "[AdaptiveScheduling] psutil not available or disabled; "
                "running in conservative mode with no emergency watchdog.",
            )

    async def stop(self) -> None:
        """
        Stop the emergency watchdog loop, if any.
        """
        task = self._emergency_task
        self._emergency_task = None

        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug(
                "[AdaptiveScheduling] error while stopping emergency watchdog",
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # Admission decision
    # ------------------------------------------------------------------ #

    async def admissible_slots(self, num_waiting: int) -> int:
        """
        Decide how many new companies can be started right now.
        """
        if num_waiting <= 0:
            return 0

        if not self._psutil_available:
            # Conservative fallback: do not ramp without metrics.
            # Caller still caps by --company-concurrency.
            return 1

        async with self._lock:
            total, used = self._read_memory_usage_bytes()
            if total <= 0:
                return 0

            used_frac = float(used) / float(total)
            self._last_used_bytes = used
            self._last_used_frac = used_frac

            now = time.time()
            self._record_memory_sample_locked(ts=now, used_bytes=used)

            swap_frac = self._read_swap_used_frac()
            self._last_swap_used_frac = swap_frac

            # Safety margin auto-tuning (near-OOM events)
            self._register_near_oom_locked(used_frac, swap_frac)

            cfg = self.cfg

            # Precritical detection based on trend
            precritical = (
                used_frac >= max(0.0, cfg.mem_cap_frac - cfg.mem_trend_margin_frac)
                and self._last_trend_slope_mb_s >= cfg.mem_trend_slope_high_mb_per_s
            )

            # Swap alone should not block admissions when RAM is low.
            high_swap_block = (
                swap_frac >= cfg.swap_block_frac and used_frac >= cfg.mem_high_frac
            )
            mem_trouble = used_frac >= cfg.mem_high_frac or high_swap_block

            # Update base RSS estimate before per-company estimate
            active_ids = list(self._get_active_company_ids())
            num_active = len(active_ids)
            self._update_base_rss_locked(used_bytes=used, num_active=num_active)

            # AIMD target update
            self._update_target_parallel_locked(
                used_frac=used_frac,
                precritical=precritical,
                high_swap=high_swap_block,
                mem_trouble=mem_trouble,
            )

            # Above high watermark or critical swap with high RAM: do not admit.
            if mem_trouble:
                reason = (
                    "swap_block"
                    if high_swap_block and used_frac < cfg.mem_high_frac
                    else "mem_high_block"
                )
                self._maybe_log_state_locked(
                    reason=reason,
                    extra={
                        "total_mem_mb": float(total) / 1e6,
                        "used_mem_mb": float(used) / 1e6,
                        "used_frac": used_frac,
                        "swap_used_frac": swap_frac,
                    },
                )
                # Still record plateau patterns even when blocked
                self._record_plateau_sample_locked(
                    ts=now,
                    num_active=num_active,
                    slots_by_mem=0,
                    used_frac=used_frac,
                )
                return 0

            # Headroom with respect to the safe cap.
            mem_cap_bytes = int(cfg.mem_cap_frac * float(total))
            headroom_bytes = mem_cap_bytes - used
            if headroom_bytes <= 0:
                self._maybe_log_state_locked(
                    reason="mem_cap_block",
                    extra={
                        "total_mem_mb": float(total) / 1e6,
                        "used_mem_mb": float(used) / 1e6,
                        "used_frac": used_frac,
                        "swap_used_frac": swap_frac,
                    },
                )
                self._record_plateau_sample_locked(
                    ts=now,
                    num_active=num_active,
                    slots_by_mem=0,
                    used_frac=used_frac,
                )
                return 0

            # Update per-company estimate
            self._update_per_company_estimate_locked(
                used_bytes=used, num_active=num_active, used_frac=used_frac
            )

            # Effective per-company estimate (global + per-company profiles)
            effective_est_mb = self._per_company_est_mb
            company_p95 = self._get_company_p95_est_mb_locked()
            if company_p95 is not None and company_p95 > effective_est_mb:
                effective_est_mb = company_p95

            # Bound effective estimate by safe cap and global hard cap.
            total_mb = float(total) / 1e6
            max_safe_mb = cfg.mem_cap_frac * total_mb
            if max_safe_mb > 0.0 and effective_est_mb > max_safe_mb:
                effective_est_mb = max_safe_mb

            if effective_est_mb > cfg.per_company_max_reservation_mb:
                effective_est_mb = cfg.per_company_max_reservation_mb

            per_company_bytes = max(
                int(effective_est_mb * 1e6),
                int(self.cfg.per_company_min_reservation_mb * 1e6),
            )
            if per_company_bytes <= 0:
                self._record_plateau_sample_locked(
                    ts=now,
                    num_active=num_active,
                    slots_by_mem=0,
                    used_frac=used_frac,
                )
                return 0

            slots_by_mem = int(headroom_bytes // per_company_bytes)

            # Key anti-stall 1: if nothing is active, always allow at least one
            # company to start (below mem_high_frac), even if the model says 0.
            if num_active == 0 and slots_by_mem <= 0 and used_frac < cfg.mem_high_frac:
                slots_by_mem = 1

            # Extended unstall detection: if slots_by_mem is too low
            if (
                slots_by_mem <= cfg.unstall_slots_threshold
                and not mem_trouble
                and headroom_bytes > 0
                and num_waiting >= cfg.plateau_min_num_waiting
                and used_frac < max(0.0, cfg.mem_cap_frac - cfg.unstall_margin_frac)
            ):
                self._maybe_unstall_locked(
                    total=total,
                    used=used,
                    headroom_bytes=headroom_bytes,
                    used_frac=used_frac,
                )
                # Recompute effective_est_mb and slots_by_mem after unstall
                effective_est_mb = self._per_company_est_mb
                company_p95 = self._get_company_p95_est_mb_locked()
                if company_p95 is not None and company_p95 > effective_est_mb:
                    effective_est_mb = company_p95

                total_mb = float(total) / 1e6
                max_safe_mb = cfg.mem_cap_frac * total_mb
                if max_safe_mb > 0.0 and effective_est_mb > max_safe_mb:
                    effective_est_mb = max_safe_mb

                if effective_est_mb > cfg.per_company_max_reservation_mb:
                    effective_est_mb = cfg.per_company_max_reservation_mb

                per_company_bytes = max(
                    int(effective_est_mb * 1e6),
                    int(self.cfg.per_company_min_reservation_mb * 1e6),
                )
                if per_company_bytes > 0:
                    slots_by_mem = int(headroom_bytes // per_company_bytes)

            # Plateau rescue (pattern-based stall detection)
            self._record_plateau_sample_locked(
                ts=now,
                num_active=num_active,
                slots_by_mem=slots_by_mem,
                used_frac=used_frac,
            )
            self._maybe_plateau_rescue_locked(
                num_waiting=num_waiting,
                used_frac=used_frac,
            )

            if slots_by_mem <= 0:
                # Still no safe slot.
                return 0

            # Apply AIMD target cap
            max_by_target = max(0, self._target_parallel - num_active)
            slots = min(slots_by_mem, num_waiting, max_by_target)

            # If trend looks precritical, be conservative and clamp to 1 slot
            if precritical and slots > 1:
                slots = 1

            # Mark healthy periods for possible future full reset avoidance
            if (
                not mem_trouble
                and num_active >= cfg.estimator_reset_parallel_threshold
                and used_frac < cfg.estimator_reset_used_frac_threshold
            ):
                self._last_healthy_ts = now

            # Full estimator reset if we have not seen a healthy period for a long time
            self._maybe_full_estimator_reset_locked(
                now=now,
                used_frac=used_frac,
                mem_trouble=mem_trouble,
            )

            self._maybe_log_state_locked(
                reason="admission",
                extra={
                    "slots": slots,
                    "num_waiting": num_waiting,
                    "num_active": num_active,
                    "per_company_est_mb": self._per_company_est_mb,
                    "effective_per_company_est_mb": effective_est_mb,
                    "total_mem_mb": total_mb,
                    "used_mem_mb": float(used) / 1e6,
                    "used_frac": used_frac,
                    "swap_used_frac": swap_frac,
                    "trend_slope_mb_s": self._last_trend_slope_mb_s,
                    "target_parallel": self._target_parallel,
                    "slots_by_mem": slots_by_mem,
                    "base_rss_mb": self._base_rss_bytes / 1e6,
                },
            )

            return slots

    def _maybe_unstall_locked(
        self,
        *,
        total: int,
        used: int,
        headroom_bytes: int,
        used_frac: float,
    ) -> None:
        """
        Detect and fix stalls caused by an over-conservative per-company
        estimate.

        Strategy:
          - Only trigger when memory usage is clearly below mem_cap_frac
            and below a "low" unstall threshold.
          - Clear history and per-company profiles.
          - Reset the global per-company estimate to something compatible
            with current headroom (but not below per_company_min_reservation_mb).
        """
        now = time.time()
        cfg = self.cfg

        if used_frac >= cfg.unstall_low_frac:
            return

        if now - self._last_unstall_ts < cfg.unstall_cooldown_sec:
            return

        headroom_mb = float(headroom_bytes) / 1e6
        if headroom_mb <= cfg.per_company_min_reservation_mb:
            # Not enough room even for one min-sized company; nothing to do.
            return

        old_est = self._per_company_est_mb

        # Reset global estimate to allow at least one company in headroom,
        # but keep at or above the configured minimum.
        new_est = min(old_est, headroom_mb)
        new_est = max(cfg.per_company_min_reservation_mb, new_est)

        # Apply global hard cap
        if new_est > cfg.per_company_max_reservation_mb:
            new_est = cfg.per_company_max_reservation_mb

        # Clear history and per-company profiles so we do not immediately
        # re-apply old heavy tails.
        self._peak_history_mb.clear()
        self._company_peak_mb.clear()
        self._company_order.clear()

        self._per_company_est_mb = new_est
        self._last_unstall_ts = now

        logger.warning(
            "[AdaptiveScheduling] unstall reset: per_company_est_mb %.1fMB -> %.1fMB "
            "(used_frac=%.3f, headroom_mb=%.1f)",
            old_est,
            new_est,
            used_frac,
            headroom_mb,
        )

    def _record_plateau_sample_locked(
        self,
        *,
        ts: float,
        num_active: int,
        slots_by_mem: int,
        used_frac: float,
    ) -> None:
        """
        Record a coarse-grained sample for plateau detection.
        """
        self._plateau_window.append(
            (ts, num_active, slots_by_mem, used_frac, self._near_oom_events)
        )
        horizon = float(self.cfg.plateau_detect_sec)
        cutoff = ts - horizon
        self._plateau_window = [s for s in self._plateau_window if s[0] >= cutoff]

    def _maybe_plateau_rescue_locked(
        self,
        *,
        num_waiting: int,
        used_frac: float,
    ) -> None:
        """
        Treat a long-lived plateau with low slots_by_mem and low parallelism
        as a signal to shrink the per-company estimate.
        """
        cfg = self.cfg
        if num_waiting < cfg.plateau_min_num_waiting:
            return

        if not self._plateau_window:
            return

        now = time.time()
        earliest_ts = self._plateau_window[0][0]
        duration = now - earliest_ts
        if duration < cfg.plateau_detect_sec:
            return

        # If we have near-oom events in the window, do not treat as benign plateau
        window_near_oom_events = {s[4] for s in self._plateau_window}
        if len(window_near_oom_events) > 1:
            # near_oom_events changed during this window
            return

        # Check that we have been mostly in the "stuck" regime
        stuck_samples = 0
        for _, num_active, slots_by_mem, u_frac, _ in self._plateau_window:
            if (
                slots_by_mem <= cfg.plateau_slots_threshold
                and num_active <= cfg.estimator_reset_parallel_threshold
                and u_frac < max(0.0, cfg.mem_cap_frac - cfg.plateau_used_margin_frac)
            ):
                stuck_samples += 1

        if not self._plateau_window:
            return

        fraction_stuck = float(stuck_samples) / float(len(self._plateau_window))
        if fraction_stuck < 0.8:
            # Not clearly a plateau
            return

        old_est = self._per_company_est_mb
        new_est = max(
            cfg.per_company_min_reservation_mb,
            old_est * cfg.plateau_shrink_factor,
        )
        if new_est > cfg.per_company_max_reservation_mb:
            new_est = cfg.per_company_max_reservation_mb

        # Clear detailed history to avoid re-inflation from stale peaks
        self._peak_history_mb.clear()
        self._company_peak_mb.clear()
        self._company_order.clear()

        self._per_company_est_mb = new_est
        self._last_unstall_ts = now  # share cooldown with unstall

        logger.warning(
            "[AdaptiveScheduling] plateau rescue: per_company_est_mb %.1fMB -> %.1fMB "
            "(duration=%.1fs, used_frac=%.3f, num_waiting=%d)",
            old_est,
            new_est,
            duration,
            used_frac,
            num_waiting,
        )

    def _maybe_full_estimator_reset_locked(
        self,
        *,
        now: float,
        used_frac: float,
        mem_trouble: bool,
    ) -> None:
        """
        Nuclear option: if we have not seen a "healthy" period (many actives,
        comfortable memory) for a long time, reset the estimator towards
        a small multiple of the minimum.
        """
        cfg = self.cfg
        if mem_trouble:
            return

        if now - self._last_healthy_ts < cfg.estimator_reset_interval_sec:
            return

        # Do not reset if we are near OOM; that is handled elsewhere
        if used_frac >= cfg.mem_high_frac:
            return

        old_est = self._per_company_est_mb
        target = cfg.per_company_min_reservation_mb * cfg.estimator_reset_factor
        new_est = max(cfg.per_company_min_reservation_mb, target)
        if new_est > cfg.per_company_max_reservation_mb:
            new_est = cfg.per_company_max_reservation_mb

        self._peak_history_mb.clear()
        self._company_peak_mb.clear()
        self._company_order.clear()

        self._per_company_est_mb = new_est
        self._last_healthy_ts = now

        logger.warning(
            "[AdaptiveScheduling] full estimator reset after long unhealthy period: "
            "per_company_est_mb %.1fMB -> %.1fMB (used_frac=%.3f)",
            old_est,
            new_est,
            used_frac,
        )

    # ------------------------------------------------------------------ #
    # Emergency watchdog
    # ------------------------------------------------------------------ #

    async def _emergency_loop(self, interval: float) -> None:
        cfg = self.cfg

        while True:
            try:
                await asyncio.sleep(interval)

                if not self._psutil_available:
                    continue

                cancel_ids: List[str] = []

                async with self._lock:
                    total, used = self._read_memory_usage_bytes()
                    if total <= 0:
                        continue

                    used_frac = float(used) / float(total)
                    self._last_used_bytes = used
                    self._last_used_frac = used_frac

                    now = time.time()
                    self._record_memory_sample_locked(ts=now, used_bytes=used)

                    swap_frac = self._read_swap_used_frac()
                    self._last_swap_used_frac = swap_frac

                    # Safety margin auto-tuning (near-OOM events)
                    self._register_near_oom_locked(used_frac, swap_frac)

                    # Update base RSS estimate
                    active_ids = list(self._get_active_company_ids())
                    num_active = len(active_ids)
                    self._update_base_rss_locked(used_bytes=used, num_active=num_active)

                    # Update per company estimate based on current usage and active count
                    self._update_per_company_estimate_locked(
                        used_bytes=used, num_active=num_active, used_frac=used_frac
                    )

                    # Effective emergency thresholds
                    eff_mem_crit_high = self._effective_mem_crit_high_frac(
                        total_bytes=total
                    )

                    # Emergency condition: high RAM or very high swap while RAM is already high
                    in_emergency = (
                        used_frac >= eff_mem_crit_high
                        or (
                            swap_frac >= cfg.swap_emergency_frac
                            and used_frac >= cfg.mem_crit_low_frac
                        )
                        or (
                            used_frac >= cfg.mem_cap_frac
                            and self._last_trend_slope_mb_s
                            >= cfg.mem_trend_emergency_slope_mb_per_s
                        )
                    )
                    if not in_emergency:
                        continue

                    # Respect emergency cancellation cooldown to avoid rapid-fire cancels
                    if (
                        now - self._last_emergency_cancel_ts
                        < cfg.emergency_cancel_cooldown_sec
                    ):
                        self._maybe_log_state_locked(
                            reason="emergency_cooldown",
                            extra={
                                "total_mem_mb": float(total) / 1e6,
                                "used_mem_mb": float(used) / 1e6,
                                "used_frac": used_frac,
                                "swap_used_frac": swap_frac,
                                "num_active": num_active,
                            },
                        )
                        continue

                    # AIMD: strong multiplicative decrease on emergency
                    high_swap = (
                        swap_frac >= cfg.swap_block_frac
                        and used_frac >= cfg.mem_high_frac
                    )
                    self._update_target_parallel_locked(
                        used_frac=used_frac,
                        precritical=False,
                        high_swap=high_swap,
                        mem_trouble=True,
                    )

                    if num_active <= 0:
                        if not self._restart_recommended:
                            self._restart_recommended = True
                            logger.error(
                                "[AdaptiveScheduling] memory critical (used_frac=%.3f swap_frac=%.3f) "
                                "with zero active companies; restart recommended.",
                                used_frac,
                                swap_frac,
                            )
                        continue

                    # Respect min_active_keep.
                    max_cancelable = max(0, num_active - int(cfg.min_active_keep))
                    if max_cancelable <= 0:
                        # Cannot cancel anything but memory is critical: recommend restart.
                        if not self._restart_recommended:
                            self._restart_recommended = True
                            logger.error(
                                "[AdaptiveScheduling] memory critical (used_frac=%.3f swap_frac=%.3f) "
                                "but num_active=%d <= min_active_keep=%d; restart recommended.",
                                used_frac,
                                swap_frac,
                                num_active,
                                cfg.min_active_keep,
                            )
                        continue

                    # Determine how many companies we ideally need to cancel to reach mem_crit_low_frac
                    total_mb = float(total) / 1e6
                    mem_crit_low_bytes = int(cfg.mem_crit_low_frac * float(total))
                    excess_bytes = max(0, used - mem_crit_low_bytes)

                    # Effective per-company estimate (global + per-company profiles)
                    effective_est_mb = self._per_company_est_mb
                    company_p95 = self._get_company_p95_est_mb_locked()
                    if company_p95 is not None and company_p95 > effective_est_mb:
                        effective_est_mb = company_p95

                    max_safe_mb = cfg.mem_cap_frac * total_mb
                    if max_safe_mb > 0.0 and effective_est_mb > max_safe_mb:
                        effective_est_mb = max_safe_mb

                    if effective_est_mb > cfg.per_company_max_reservation_mb:
                        effective_est_mb = cfg.per_company_max_reservation_mb

                    per_company_bytes = max(
                        int(effective_est_mb * 1e6),
                        int(self.cfg.per_company_min_reservation_mb * 1e6),
                    )
                    if per_company_bytes <= 0:
                        needed_cancel = 1
                    else:
                        needed_cancel = int(
                            (excess_bytes + per_company_bytes - 1) // per_company_bytes
                        )
                        needed_cancel = max(1, needed_cancel)

                    # Decide how many to cancel in this round
                    super_critical = used_frac >= 0.98 or (
                        used_frac >= cfg.mem_cap_frac
                        and self._last_trend_slope_mb_s
                        >= cfg.mem_trend_emergency_slope_mb_per_s
                    )
                    if super_critical:
                        to_cancel_count = min(
                            needed_cancel,
                            cfg.max_emergency_cancel_per_step,
                            max_cancelable,
                        )
                    else:
                        to_cancel_count = min(1, max_cancelable)

                    if to_cancel_count <= 0:
                        continue

                    # Choose companies to cancel, preferring the heaviest known ones.
                    to_cancel = self._select_heaviest_companies_locked(
                        active_ids, count=to_cancel_count
                    )

                    if not to_cancel:
                        # Fallback: cancel last N active companies
                        to_cancel = list(active_ids)[-to_cancel_count:]

                    cancel_ids = list(to_cancel)
                    self._last_emergency_cancel_ts = now
                    self._emergency_rounds += 1

                    # Log emergency cancellation
                    logger.error(
                        "[AdaptiveScheduling] emergency memory pressure: "
                        "used_frac=%.3f, swap_used_frac=%.3f, "
                        "total_mem_mb=%.1f, used_mem_mb=%.1f, "
                        "num_active=%d, per_company_est_mb=%.1f, "
                        "effective_per_company_est_mb=%.1f, "
                        "excess_bytes=%.1fMB, canceling=%d companies: %s",
                        used_frac,
                        swap_frac,
                        total_mb,
                        float(used) / 1e6,
                        num_active,
                        self._per_company_est_mb,
                        effective_est_mb,
                        float(excess_bytes) / 1e6,
                        len(cancel_ids),
                        cancel_ids,
                    )

                    # If emergency persists for many rounds, recommend restart
                    if (
                        self._emergency_rounds
                        >= cfg.emergency_persistent_rounds_threshold
                        and used_frac >= cfg.mem_crit_low_frac
                        and not self._restart_recommended
                    ):
                        self._restart_recommended = True
                        logger.error(
                            "[AdaptiveScheduling] repeated emergency rounds (%d) "
                            "with used_frac=%.3f; restart recommended.",
                            self._emergency_rounds,
                            used_frac,
                        )

                if cancel_ids:
                    try:
                        self._request_cancel_companies(cancel_ids)
                    except Exception:
                        logger.exception(
                            "[AdaptiveScheduling] request_cancel_companies failed"
                        )

                    # Allow some time for tasks, browser processes and GC to release memory
                    try:
                        await asyncio.sleep(cfg.emergency_post_cancel_delay_sec)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # Any error here is non-fatal to the watchdog
                        logger.debug(
                            "[AdaptiveScheduling] error during post-cancel sleep",
                            exc_info=True,
                        )

                    # Trigger GC to help RSS drop a bit faster
                    try:
                        gc.collect()
                    except Exception:
                        logger.debug(
                            "[AdaptiveScheduling] gc.collect() failed",
                            exc_info=True,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "[AdaptiveScheduling] error in emergency watchdog loop"
                )

    # ------------------------------------------------------------------ #
    # State inspection
    # ------------------------------------------------------------------ #

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Return a lightweight snapshot of the scheduler state for logging
        or debugging.
        """
        total = self._total_mem_bytes or 0
        used = self._last_used_bytes
        used_frac = self._last_used_frac

        return {
            "total_mem_mb": float(total) / 1e6 if total > 0 else 0.0,
            "used_mem_mb": float(used) / 1e6 if used > 0 else 0.0,
            "used_frac": used_frac,
            "swap_used_frac": self._last_swap_used_frac,
            "per_company_est_mb": self._per_company_est_mb,
            "peak_history_len": len(self._peak_history_mb),
            "trend_slope_mb_s": self._last_trend_slope_mb_s,
            "target_parallel": self._target_parallel,
            "near_oom_events": self._near_oom_events,
            "psutil_available": self._psutil_available,
            "restart_recommended": self._restart_recommended,
            "mem_cap_frac": self.cfg.mem_cap_frac,
            "mem_high_frac": self.cfg.mem_high_frac,
            "mem_crit_high_frac": self.cfg.mem_crit_high_frac,
            "base_rss_mb": self._base_rss_bytes / 1e6,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_cgroup_memory_bytes(self) -> Tuple[int, int]:
        """
        Try to read cgroup v2 memory limits (memory.max, memory.current).

        Returns (limit, used) or (0, 0) if not available.
        """
        try:
            base = Path("/sys/fs/cgroup")
            max_path = base / "memory.max"
            cur_path = base / "memory.current"
            if not max_path.exists() or not cur_path.exists():
                return 0, 0

            max_raw = max_path.read_text(encoding="utf-8").strip()
            cur_raw = cur_path.read_text(encoding="utf-8").strip()
            if not cur_raw:
                return 0, 0

            used = int(cur_raw)
            if not max_raw or max_raw == "max":
                return 0, used

            limit = int(max_raw)
            return limit, used
        except Exception:
            return 0, 0

    def _read_memory_usage_bytes(self) -> Tuple[int, int]:
        """
        Read total and used memory in bytes.
        """
        if not self._psutil_available or psutil is None:
            return 0, 0

        if self.cfg.prefer_cgroup_limits:
            limit, used = self._read_cgroup_memory_bytes()
            if limit > 0:
                if self._total_mem_bytes is None:
                    self._total_mem_bytes = limit
                return limit, used

        try:
            vm = psutil.virtual_memory()  # type: ignore[union-attr]
            total = int(vm.total)
            used = int(vm.total - vm.available)
            if self._total_mem_bytes is None and total > 0:
                self._total_mem_bytes = total
            return total, used
        except Exception:
            logger.debug(
                "[AdaptiveScheduling] failed to read memory usage via psutil",
                exc_info=True,
            )
            return 0, 0

    def _read_swap_used_frac(self) -> float:
        """
        Return swap_used / swap_total or 0.0 if not available.
        """
        if not self._psutil_available or psutil is None:
            return 0.0
        try:
            sm = psutil.swap_memory()  # type: ignore[union-attr]
            if sm.total <= 0:
                return 0.0
            return float(sm.used) / float(sm.total)
        except Exception:
            return 0.0

    def _record_memory_sample_locked(self, *, ts: float, used_bytes: int) -> None:
        """
        Update rolling window of memory samples and recompute slope (MB/s).
        """
        window = max(1.0, float(self.cfg.mem_trend_window_sec))
        self._mem_samples.append((ts, used_bytes))
        cutoff = ts - window
        self._mem_samples = [(t, u) for (t, u) in self._mem_samples if t >= cutoff]

        if len(self._mem_samples) >= 2:
            t0, u0 = self._mem_samples[0]
            t1, u1 = self._mem_samples[-1]
            dt = t1 - t0
            if dt > 0:
                self._last_trend_slope_mb_s = (float(u1) - float(u0)) / dt / 1e6
            else:
                self._last_trend_slope_mb_s = 0.0
        else:
            self._last_trend_slope_mb_s = 0.0

    def _get_company_p95_est_mb_locked(self) -> Optional[float]:
        """
        Compute 95th percentile of per-company peak estimates, if any.
        """
        if not self._company_peak_mb:
            return None
        vals = sorted(self._company_peak_mb.values())
        idx = max(0, int(0.95 * len(vals)) - 1)
        return vals[idx]

    def _update_base_rss_locked(self, *, used_bytes: int, num_active: int) -> None:
        """
        Update our best guess of base RSS (memory used with zero active companies).
        """
        if used_bytes <= 0:
            return
        if num_active != 0:
            return

        # Simple exponential moving average with mild smoothing
        alpha = 0.1
        if self._base_rss_samples == 0:
            self._base_rss_bytes = float(used_bytes)
            self._base_rss_samples = 1
        else:
            self._base_rss_bytes = (1.0 - alpha) * self._base_rss_bytes + alpha * float(
                used_bytes
            )
            self._base_rss_samples += 1

    def _update_per_company_estimate_locked(
        self,
        *,
        used_bytes: int,
        num_active: int,
        used_frac: float,
    ) -> None:
        """
        Update the global per-company memory estimate.

        We want:
          - Robust upward learning when several companies are active.
          - Slow, controlled downward adjustment when memory is clearly safe,
            even if only 1 company is active, to avoid stalls at the tail.
          - Estimates that exclude baseline RSS where possible.
        """
        if used_bytes <= 0 or self._total_mem_bytes is None:
            return

        cfg = self.cfg
        total = float(self._total_mem_bytes)
        used = float(used_bytes)

        if num_active <= 0:
            return

        # Separate base RSS from per-company usage
        effective_used = used - self._base_rss_bytes
        if effective_used < 0:
            effective_used = 0.0

        if num_active > 0:
            per_company_now_mb = (effective_used / float(num_active)) / 1e6
        else:
            per_company_now_mb = 0.0

        # Only add to peak history when we have at least 2 actives.
        now = time.time()
        if num_active >= 2 and per_company_now_mb > 0.0:
            self._peak_history_mb.append((now, per_company_now_mb))
            # Trim by size
            if len(self._peak_history_mb) > cfg.peak_history_size:
                self._peak_history_mb = self._peak_history_mb[-cfg.peak_history_size :]
            # And by time horizon
            horizon = float(cfg.peak_history_horizon_sec)
            cutoff = now - horizon
            self._peak_history_mb = [
                (t, v) for (t, v) in self._peak_history_mb if t >= cutoff
            ]

        # Upward estimate from history (q95 * safety_factor)
        if self._peak_history_mb:
            values = [v for (_, v) in self._peak_history_mb]
            sorted_hist = sorted(values)
            idx = max(0, int(0.95 * len(sorted_hist)) - 1)
            q95 = sorted_hist[idx]
            base = max(q95, cfg.per_company_min_reservation_mb)
            upward_est = cfg.per_company_safety_factor * base
        else:
            upward_est = self._per_company_est_mb

        # Bound upward estimate by safe cap and hard cap.
        max_safe_mb = (cfg.mem_cap_frac * total) / 1e6
        if max_safe_mb > 0.0 and upward_est > max_safe_mb:
            upward_est = max_safe_mb

        if upward_est > cfg.per_company_max_reservation_mb:
            upward_est = cfg.per_company_max_reservation_mb

        old_est = self._per_company_est_mb
        new_est = max(cfg.per_company_min_reservation_mb, upward_est)

        # Controlled downward adjustment:
        # If memory usage is clearly in a safe region, and the naive
        # per_company_now_mb is significantly smaller than our current
        # estimate, gently move the estimate downwards.
        if used_frac < cfg.unstall_low_frac and per_company_now_mb * 1.5 < old_est:
            # Move 10 percent of the way towards the current observation (smoothed)
            target = max(cfg.per_company_min_reservation_mb, per_company_now_mb)
            lowered = old_est * 0.9 + target * 0.1
            if lowered < new_est:
                new_est = lowered

        if new_est > cfg.per_company_max_reservation_mb:
            new_est = cfg.per_company_max_reservation_mb

        self._per_company_est_mb = new_est

        if self._per_company_est_mb > old_est * 1.1:
            logger.info(
                "[AdaptiveScheduling] per company estimate updated: "
                "%.1fMB -> %.1fMB (active=%d, used_frac=%.3f)",
                old_est,
                self._per_company_est_mb,
                num_active,
                used_frac,
            )

    def _register_near_oom_locked(
        self,
        used_frac: float,
        swap_frac: float,
    ) -> None:
        """
        Detect near-OOM situations and auto-tune mem_cap_frac / mem_high_frac /
        mem_crit_high_frac downward for the remainder of the process lifetime.
        """
        cfg = self.cfg
        is_near = used_frac >= cfg.near_oom_used_frac or (
            swap_frac >= cfg.near_oom_swap_frac and used_frac >= cfg.mem_crit_low_frac
        )

        if is_near and not self._last_near_oom_flag:
            self._near_oom_events += 1
            old_cap = cfg.mem_cap_frac
            old_high = cfg.mem_high_frac
            old_crit = cfg.mem_crit_high_frac
            step = cfg.near_oom_mem_cap_step

            if step > 0.0:
                cfg.mem_cap_frac = max(cfg.mem_cap_min_frac, cfg.mem_cap_frac - step)
                cfg.mem_high_frac = max(cfg.mem_high_min_frac, cfg.mem_high_frac - step)
                cfg.mem_crit_high_frac = max(
                    cfg.mem_crit_min_frac, cfg.mem_crit_high_frac - step
                )

            logger.warning(
                "[AdaptiveScheduling] near-oom event #%d: "
                "used_frac=%.3f swap_frac=%.3f "
                "mem_cap_frac: %.3f -> %.3f, mem_high_frac: %.3f -> %.3f, "
                "mem_crit_high_frac: %.3f -> %.3f",
                self._near_oom_events,
                used_frac,
                swap_frac,
                old_cap,
                cfg.mem_cap_frac,
                old_high,
                cfg.mem_high_frac,
                old_crit,
                cfg.mem_crit_high_frac,
            )

        self._last_near_oom_flag = is_near

    def _effective_mem_crit_high_frac(self, *, total_bytes: int) -> float:
        """
        Adjust mem_crit_high_frac based on total memory size to react earlier
        on small hosts.
        """
        cfg = self.cfg
        eff = cfg.mem_crit_high_frac
        if total_bytes <= 0:
            return eff

        total_gb = float(total_bytes) / (1024.0**3)
        if total_gb <= 16.0:
            # For small hosts be a bit more conservative
            eff = min(eff, 0.92)
        return max(cfg.mem_crit_low_frac, eff)

    def _update_target_parallel_locked(
        self,
        *,
        used_frac: float,
        precritical: bool,
        high_swap: bool,
        mem_trouble: bool,
    ) -> None:
        """
        AIMD style concurrency controller for target_parallel.
        """
        cfg = self.cfg
        old_target = self._target_parallel

        # Count updates so we know if we are in warm-up phase
        self._update_calls += 1

        if mem_trouble or high_swap or precritical:
            # Multiplicative decrease on trouble or strong warning signs
            new_target = int(self._target_parallel * cfg.md_factor)
            if new_target < cfg.min_target:
                new_target = cfg.min_target
            if new_target < 1:
                new_target = 1
        else:
            # Additive increase when comfortably below cap and trend is safe
            margin = cfg.mem_trend_margin_frac
            if (
                used_frac < max(0.0, cfg.mem_cap_frac - margin)
                and self._last_trend_slope_mb_s <= cfg.mem_trend_slope_high_mb_per_s
            ):
                # During warm-up, use a larger step to ramp up faster
                if self._update_calls <= cfg.warmup_updates:
                    step = max(cfg.ai_step, cfg.warmup_ai_step)
                else:
                    step = cfg.ai_step
                new_target = min(cfg.max_target, self._target_parallel + step)
            else:
                new_target = self._target_parallel

        self._target_parallel = new_target

        if new_target != old_target and logger.isEnabledFor(logging.INFO):
            logger.info(
                "[AdaptiveScheduling] target_parallel updated: %d -> %d "
                "(used_frac=%.3f, swap_frac=%.3f, slope=%.1fMB/s, "
                "precritical=%s, high_swap=%s, mem_trouble=%s, "
                "update_calls=%d)",
                old_target,
                new_target,
                used_frac,
                self._last_swap_used_frac,
                self._last_trend_slope_mb_s,
                precritical,
                high_swap,
                mem_trouble,
                self._update_calls,
            )

    def _maybe_log_state_locked(
        self,
        *,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Optionally write a JSON state line to log_path for offline analysis.
        """
        if self.cfg.log_path is None:
            return

        now = time.time()
        if self._last_log_ts and now - self._last_log_ts < 5.0:
            return
        self._last_log_ts = now

        state: Dict[str, Any] = {
            "ts": now,
            "reason": reason,
            "total_mem_bytes": self._total_mem_bytes,
            "last_used_bytes": self._last_used_bytes,
            "last_used_frac": self._last_used_frac,
            "last_swap_used_frac": self._last_swap_used_frac,
            "per_company_est_mb": self._per_company_est_mb,
            "peak_history_len": len(self._peak_history_mb),
            "trend_slope_mb_s": self._last_trend_slope_mb_s,
            "target_parallel": self._target_parallel,
            "near_oom_events": self._near_oom_events,
            "restart_recommended": self._restart_recommended,
            "mem_cap_frac": self.cfg.mem_cap_frac,
            "mem_high_frac": self.cfg.mem_high_frac,
            "mem_crit_high_frac": self.cfg.mem_crit_high_frac,
            "base_rss_mb": self._base_rss_bytes / 1e6,
        }
        if extra:
            state.update(extra)

        try:
            path = self.cfg.log_path
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            logger.debug(
                "[AdaptiveScheduling] failed writing state log to %s",
                self.cfg.log_path,
                exc_info=True,
            )

    def _select_heaviest_companies_locked(
        self,
        active_ids: Sequence[str],
        count: int,
    ) -> List[str]:
        """
        Select up to `count` "heaviest" active companies based on recorded peaks.

        If no peak info is available for any active company, returns [] and the
        caller should fall back to a simple policy.
        """
        if not active_ids or count <= 0:
            return []

        # Gather (id, peak_mb) for those with known peaks
        weighted: List[Tuple[str, float]] = []
        for cid in active_ids:
            mb = self._company_peak_mb.get(cid)
            if mb is not None:
                weighted.append((cid, mb))

        if not weighted:
            return []

        # Sort by peak_mb descending and pick top count
        weighted.sort(key=lambda x: x[1], reverse=True)
        selected = [cid for (cid, _) in weighted[:count]]
        return selected


__all__ = [
    "AdaptiveSchedulingConfig",
    "AdaptiveScheduler",
    "CriticalMemoryPressure",
    "MEMORY_PRESSURE_MARKER",
]
