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
# Config
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSchedulingConfig:
    """
    Memory-based adaptive scheduling configuration.

    All thresholds are expressed as fractions of the effective memory limit
    (host or cgroup), except where otherwise noted.
    """

    # Conservative defaults tuned for headless-chrome workloads (not too aggressive)
    mem_cap_frac: float = 0.78
    mem_high_frac: float = 0.83
    mem_crit_high_frac: float = 0.90
    mem_crit_low_frac: float = 0.88

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
    mem_trend_emergency_slope_mb_per_s: float = 40.0

    # (4) AIMD concurrency controller
    ai_step: int = 1
    md_factor: float = 0.5
    min_target: int = 1
    max_target: int = 512

    # Faster ramp-up at the start of the run
    initial_target: int = 4
    warmup_updates: int = 10
    warmup_ai_step: int = 3

    # (5) Safety margin auto-tuning
    near_oom_used_frac: float = 0.93
    near_oom_swap_frac: float = 0.50
    near_oom_mem_cap_step: float = 0.02
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
    emergency_post_cancel_delay_sec: float = 3.0
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

    # ------------------------------------------------------------------
    # Progress-based stall detection (CPU / network / disk / completions)
    # ------------------------------------------------------------------

    # Rolling window for progress samples (seconds)
    progress_window_sec: float = 900.0  # 15 minutes
    # Minimum duration within the window to consider a stall (seconds)
    stall_min_duration_sec: float = 900.0  # 15 minutes
    # Cooldown between stall rescue actions (seconds)
    stall_cooldown_sec: float = 600.0  # 10 minutes

    # CPU considered "flat" when max-min <= stall_cpu_band and mean below stall_cpu_max_mean
    # Values are fractions in [0, 1], not percentages.
    stall_cpu_band: float = 0.03  # 3% band
    stall_cpu_max_mean: float = 0.80  # 80% mean

    # Network and disk thresholds in KB/s (mean over window)
    stall_net_threshold_kb_s: float = 200.0
    stall_disk_threshold_kb_s: float = 300.0

    # Minimum completed companies per minute to consider "making progress"
    stall_min_completed_per_min: float = 1.0

    # Stall rescue actions
    stall_per_company_shrink_factor: float = 0.6
    stall_target_parallel_boost: int = 4
    stall_max_history_shrink_frac: float = 0.3
    # Step to reduce mem_cap_frac / mem_high_frac on stall (fraction)
    stall_mem_cap_step: float = 0.02

    # ------------------------------------------------------------------
    # Per-task RSS monitoring (early detection of runaway browser children)
    # ------------------------------------------------------------------

    # If a single company's process tree exceeds this RSS (MB) for
    # per_task_rss_duration_sec seconds, immediately request cancel.
    per_task_rss_limit_mb: float = 500.0
    per_task_rss_duration_sec: float = 2.0
    # Do per-task checks inside emergency loop (same interval as emergency_check_interval_sec).
    per_task_monitor_enabled: bool = True

    # Slope-based early pre-oom cancellation
    preoom_used_frac: float = 0.83
    preoom_slope_mb_s: float = 20.0
    preoom_cancel_count: int = 1  # 1-2 (conservative by default)


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
      - Progress-based stall rescue (CPU / network / disk / completions)
      - Per-task RSS monitoring to catch runaway browser children early
    """

    def __init__(
        self,
        cfg: AdaptiveSchedulingConfig,
        get_active_company_ids: Callable[[], Sequence[str]],
        request_cancel_companies: Callable[[Sequence[str]], None],
        # Optional callback: given a company_id -> Sequence[int] of PIDs (root processes)
        get_company_pids: Optional[Callable[[str], Sequence[int]]] = None,
    ) -> None:
        self.cfg = cfg

        self._psutil_available: bool = bool(PSUTIL_AVAILABLE and cfg.use_psutil)

        # Callbacks into the caller
        self._get_active_company_ids = get_active_company_ids
        self._request_cancel_companies = request_cancel_companies
        self._get_company_pids = get_company_pids

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

        # ------------------------------------------------------------------
        # Progress-based stall detection state
        # ------------------------------------------------------------------
        # Progress samples: (ts, cpu_frac, net_bytes_per_s, disk_bytes_per_s, completed_counter)
        self._progress_samples: List[Tuple[float, float, float, float, int]] = []
        self._last_progress_raw_ts: float = 0.0
        self._last_net_bytes: Optional[int] = None
        self._last_disk_write_bytes: Optional[int] = None
        self._last_progress_stall_ts: float = 0.0

        # Run-level completed companies counter (to be bumped by the caller)
        self._completed_counter: int = 0

        # ------------------------------------------------------------------
        # Per-task RSS monitoring state
        # ------------------------------------------------------------------
        # Map company_id -> first timestamp when per-task RSS exceeded threshold
        self._per_task_over_thresh_ts: Dict[str, float] = {}
        # Map company_id -> last observed rss (bytes)
        self._per_task_last_rss_bytes: Dict[str, int] = {}

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
    # Public API for per-company profiles / progress
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

    def register_company_completed(self) -> None:
        """
        Optional: caller can invoke this whenever a company finishes successfully.
        Used only for low-progress stall detection; safe to ignore if unused.
        """
        self._completed_counter += 1

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

            # Progress-based stall detection and rescue (does not kill tasks)
            # Runs before the hard memory-based blocks so that it can relax
            # estimates and boost target_parallel in low-progress regimes.
            self._record_progress_sample_locked(ts=now)
            self._maybe_progress_stall_rescue_locked(
                num_waiting=num_waiting,
                num_active=num_active,
                used_frac=used_frac,
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
                        "num_active": num_active,
                        "slots_by_mem": 0,
                        "trend_slope_mb_s": self._last_trend_slope_mb_s,
                        "target_parallel": self._target_parallel,
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
                        "num_active": num_active,
                        "slots_by_mem": 0,
                        "trend_slope_mb_s": self._last_trend_slope_mb_s,
                        "target_parallel": self._target_parallel,
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

                    # -------------------------
                    # Per-task RSS monitoring
                    # -------------------------
                    # Conservative, non-aggressive early-cancel for runaway tasks.
                    if (
                        cfg.per_task_monitor_enabled
                        and self._get_company_pids is not None
                        and active_ids
                    ):
                        per_task_limit_bytes = int(cfg.per_task_rss_limit_mb * 1e6)
                        for cid in active_ids:
                            try:
                                pids = list(self._get_company_pids(cid) or [])
                            except Exception:
                                # If callback fails, skip per-task check for this company.
                                pids = []

                            if not pids:
                                # nothing to check
                                self._per_task_over_thresh_ts.pop(cid, None)
                                self._per_task_last_rss_bytes.pop(cid, None)
                                continue

                            # compute process-tree RSS for these root pids
                            rss_total = 0
                            for pid in pids:
                                try:
                                    proc = psutil.Process(pid)  # type: ignore
                                    # include rss of proc and its children
                                    try:
                                        rss_total += proc.memory_info().rss
                                    except Exception:
                                        # if memory_info fails, skip
                                        pass
                                    # children
                                    try:
                                        for ch in proc.children(recursive=True):
                                            try:
                                                rss_total += ch.memory_info().rss
                                            except Exception:
                                                pass
                                    except Exception:
                                        # some processes may restrict child inspection
                                        pass
                                except Exception:
                                    # process may have died
                                    pass

                            # store last rss (helpful for logs / diagnostics)
                            self._per_task_last_rss_bytes[cid] = int(rss_total)

                            if rss_total >= per_task_limit_bytes:
                                first_ts = self._per_task_over_thresh_ts.get(cid)
                                if first_ts is None:
                                    self._per_task_over_thresh_ts[cid] = now
                                else:
                                    if now - first_ts >= float(
                                        cfg.per_task_rss_duration_sec
                                    ):
                                        # sustained exceed -> request cancel
                                        logger.warning(
                                            "[AdaptiveScheduling] per-task RSS exceeded for company %s: "
                                            "rss=%.1fMB limit=%.1fMB sustained=%.1fs; canceling",
                                            cid,
                                            float(rss_total) / 1e6,
                                            float(cfg.per_task_rss_limit_mb),
                                            now - first_ts,
                                        )
                                        cancel_ids.append(cid)
                                        # reset timer for this company
                                        self._per_task_over_thresh_ts.pop(cid, None)
                                        self._per_task_last_rss_bytes.pop(cid, None)
                            else:
                                # below threshold -> clear any pending timestamp
                                self._per_task_over_thresh_ts.pop(cid, None)

                    # Effective emergency thresholds
                    eff_mem_crit_high = self._effective_mem_crit_high_frac(
                        total_bytes=total
                    )

                    slope = self._last_trend_slope_mb_s

                    # Pre-OOM slope-based early cancel (conservative)
                    if (
                        used_frac >= cfg.preoom_used_frac
                        and slope >= cfg.preoom_slope_mb_s
                    ):
                        # cancel 1..preoom_cancel_count heaviest companies (conservative)
                        to_cancel_count = min(
                            max(1, int(cfg.preoom_cancel_count)),
                            max(0, num_active - int(cfg.min_active_keep)),
                        )
                        if to_cancel_count > 0:
                            selected = self._select_heaviest_companies_locked(
                                active_ids, count=to_cancel_count
                            )
                            if not selected:
                                selected = list(active_ids)[-to_cancel_count:]
                            logger.warning(
                                "[AdaptiveScheduling] pre-oom slope trigger: used_frac=%.3f slope=%.1fMB/s canceling=%d companies: %s",
                                used_frac,
                                slope,
                                len(selected),
                                selected,
                            )
                            cancel_ids.extend(selected)
                            self._last_emergency_cancel_ts = now
                            self._emergency_rounds += 1

                    # Emergency condition: high RAM or very high swap while RAM is already high
                    in_emergency = (
                        used_frac >= eff_mem_crit_high
                        or (
                            swap_frac >= cfg.swap_emergency_frac
                            and used_frac >= cfg.mem_crit_low_frac
                        )
                        or (
                            used_frac >= cfg.mem_cap_frac
                            and slope >= cfg.mem_trend_emergency_slope_mb_per_s
                        )
                        or (
                            used_frac >= cfg.mem_high_frac
                            and slope >= cfg.mem_trend_emergency_slope_mb_per_s
                        )
                    )

                    if not in_emergency and not cancel_ids:
                        # Nothing more to do this round
                        continue

                    # Respect emergency cancellation cooldown to avoid rapid-fire cancels
                    if (
                        now - self._last_emergency_cancel_ts
                        < cfg.emergency_cancel_cooldown_sec
                    ) and not cancel_ids:
                        self._maybe_log_state_locked(
                            reason="emergency_cooldown",
                            extra={
                                "total_mem_mb": float(total) / 1e6,
                                "used_mem_mb": float(used) / 1e6,
                                "used_frac": used_frac,
                                "swap_used_frac": swap_frac,
                                "num_active": num_active,
                                "trend_slope_mb_s": slope,
                            },
                        )
                        continue

                    # If we already decided to cancel due to per-task RSS or pre-oom slope,
                    # we still run the AIMD decrease and other bookkeeping below.
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

                    # If cancel_ids already collected (from per-task or pre-oom), respect max_cancelable
                    if cancel_ids:
                        cancel_ids = cancel_ids[:max_cancelable]
                    else:
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
                                (excess_bytes + per_company_bytes - 1)
                                // per_company_bytes
                            )
                            needed_cancel = max(1, needed_cancel)

                        # Decide how many to cancel in this round
                        super_critical = used_frac >= 0.98 or (
                            used_frac >= cfg.mem_cap_frac
                            and slope >= cfg.mem_trend_emergency_slope_mb_per_s
                        )

                        mid_emergency = (
                            used_frac >= cfg.mem_high_frac
                            and slope >= cfg.mem_trend_slope_high_mb_per_s
                        ) or (
                            used_frac >= 0.92
                            and slope >= cfg.mem_trend_emergency_slope_mb_per_s
                        )

                        if super_critical:
                            to_cancel_count = min(
                                needed_cancel,
                                cfg.max_emergency_cancel_per_step,
                                max_cancelable,
                            )
                        elif mid_emergency:
                            to_cancel_count = min(2, needed_cancel, max_cancelable)
                        else:
                            to_cancel_count = min(1, max_cancelable)

                        if to_cancel_count <= 0:
                            # nothing to cancel
                            continue

                        to_cancel = self._select_heaviest_companies_locked(
                            active_ids, count=to_cancel_count
                        )

                        if not to_cancel:
                            # Fallback: cancel last N active companies
                            to_cancel = list(active_ids)[-to_cancel_count:]

                        cancel_ids = list(to_cancel)

                    # Update emergency bookkeeping
                    self._last_emergency_cancel_ts = now
                    self._emergency_rounds += 1

                    # Log emergency cancellation
                    logger.error(
                        "[AdaptiveScheduling] emergency memory pressure: "
                        "used_frac=%.3f, swap_used_frac=%.3f, "
                        "total_mem_mb=%.1f, used_mem_mb=%.1f, "
                        "num_active=%d, per_company_est_mb=%.1f, "
                        "effective_per_company_est_mb=%.1f, "
                        "canceling=%d companies: %s, "
                        "trend_slope_mb_s=%.1f",
                        used_frac,
                        swap_frac,
                        float(total) / 1e6,
                        float(used) / 1e6,
                        num_active,
                        self._per_company_est_mb,
                        effective_est_mb
                        if "effective_est_mb" in locals()
                        else self._per_company_est_mb,
                        len(cancel_ids),
                        cancel_ids,
                        slope,
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

                # release lock before calling external cancellation (but keep consistent state)
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

        # per-task rss samples (convert to MB) - include few entries
        per_task_rss_mb = {
            cid: (bytes_ / 1e6) for cid, bytes_ in self._per_task_last_rss_bytes.items()
        }

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
            "completed_counter": self._completed_counter,
            "progress_samples_len": len(self._progress_samples),
            "per_task_rss_mb_sample": per_task_rss_mb,
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

    def _record_progress_sample_locked(self, *, ts: float) -> None:
        """
        Record CPU / network / disk progress sample for stall detection.

        Stores:
          (ts, cpu_frac, net_bytes_per_s, disk_bytes_per_s, completed_counter)
        """
        if not self._psutil_available or psutil is None:
            return

        try:
            # CPU percent since last call, normalized to [0, 1].
            cpu_percent = psutil.cpu_percent(interval=None)  # type: ignore[union-attr]
            cpu_frac = float(cpu_percent) / 100.0

            net = psutil.net_io_counters()  # type: ignore[union-attr]
            disk = psutil.disk_io_counters()  # type: ignore[union-attr]
        except Exception:
            return

        net_bytes = int(net.bytes_sent + net.bytes_recv)
        disk_bytes = int(disk.write_bytes)  # type: ignore

        if (
            self._last_net_bytes is None
            or self._last_disk_write_bytes is None
            or self._last_progress_raw_ts <= 0.0
        ):
            net_rate = 0.0
            disk_rate = 0.0
        else:
            dt = ts - self._last_progress_raw_ts
            if dt <= 0:
                net_rate = 0.0
                disk_rate = 0.0
            else:
                net_rate = float(net_bytes - self._last_net_bytes) / dt
                disk_rate = float(disk_bytes - self._last_disk_write_bytes) / dt

        self._last_progress_raw_ts = ts
        self._last_net_bytes = net_bytes
        self._last_disk_write_bytes = disk_bytes

        self._progress_samples.append(
            (ts, cpu_frac, net_rate, disk_rate, self._completed_counter)
        )

        window = max(1.0, float(self.cfg.progress_window_sec))
        cutoff = ts - window
        self._progress_samples = [s for s in self._progress_samples if s[0] >= cutoff]

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

    def _maybe_progress_stall_rescue_locked(
        self,
        *,
        num_waiting: int,
        num_active: int,
        used_frac: float,
    ) -> None:
        """
        Detect a long-lived low-progress plateau using CPU / network / disk / completions
        and gently relax the estimator and target_parallel WITHOUT canceling tasks.

        Intended to wake up droplets where CPU and RAM are flat, but network, disk
        and completions are near zero.
        """
        cfg = self.cfg

        if num_waiting <= 0 or num_active <= 0:
            return

        if not self._progress_samples:
            return

        now = time.time()
        if now - self._last_progress_stall_ts < float(cfg.stall_cooldown_sec):
            return

        # Require a minimum duration inside the progress window.
        ts0 = self._progress_samples[0][0]
        ts1 = self._progress_samples[-1][0]
        duration = ts1 - ts0
        if duration <= 0 or duration < float(cfg.stall_min_duration_sec):
            return

        cpu_vals = [s[1] for s in self._progress_samples]
        net_rates = [s[2] for s in self._progress_samples]
        disk_rates = [s[3] for s in self._progress_samples]
        completed0 = self._progress_samples[0][4]
        completed1 = self._progress_samples[-1][4]

        if not cpu_vals:
            return

        cpu_min = min(cpu_vals)
        cpu_max = max(cpu_vals)
        cpu_mean = sum(cpu_vals) / float(len(cpu_vals))
        cpu_band = cpu_max - cpu_min

        mean_net = sum(net_rates) / float(len(net_rates)) if net_rates else 0.0
        mean_disk = sum(disk_rates) / float(len(disk_rates)) if disk_rates else 0.0

        total_completed = max(0, completed1 - completed0)
        completed_per_min = (
            float(total_completed) / (duration / 60.0) if duration > 0 else float("inf")
        )

        stalled_cpu = (
            cpu_band <= cfg.stall_cpu_band and cpu_mean <= cfg.stall_cpu_max_mean
        )
        stalled_net = mean_net <= cfg.stall_net_threshold_kb_s * 1024.0
        stalled_disk = mean_disk <= cfg.stall_disk_threshold_kb_s * 1024.0
        stalled_completion = completed_per_min <= cfg.stall_min_completed_per_min

        if not (stalled_cpu and stalled_net and stalled_disk and stalled_completion):
            return

        # Do not trigger this when already near the memory cap; that is handled elsewhere.
        if used_frac >= cfg.mem_cap_frac:
            return

        self._last_progress_stall_ts = now

        # Shrink per-company estimate
        old_est = self._per_company_est_mb
        new_est = max(
            cfg.per_company_min_reservation_mb,
            old_est * cfg.stall_per_company_shrink_factor,
        )
        if new_est > cfg.per_company_max_reservation_mb:
            new_est = cfg.per_company_max_reservation_mb

        self._per_company_est_mb = new_est

        # Boost target_parallel to force more companies to start
        old_target = self._target_parallel
        self._target_parallel = min(
            cfg.max_target, self._target_parallel + cfg.stall_target_parallel_boost
        )

        # Shrink peak history to forget a chunk of heavy tails
        hist_len = len(self._peak_history_mb)
        if hist_len > 0 and cfg.stall_max_history_shrink_frac > 0.0:
            drop = int(hist_len * cfg.stall_max_history_shrink_frac)
            if drop > 0:
                self._peak_history_mb = self._peak_history_mb[drop:]

        # Gently reduce mem_cap_frac and mem_high_frac to keep some headroom
        old_cap = cfg.mem_cap_frac
        old_high = cfg.mem_high_frac
        step = max(0.0, float(cfg.stall_mem_cap_step))
        if step > 0.0:
            cfg.mem_cap_frac = max(cfg.mem_cap_min_frac, cfg.mem_cap_frac - step)
            cfg.mem_high_frac = max(cfg.mem_high_min_frac, cfg.mem_high_frac - step)

        logger.warning(
            "[AdaptiveScheduling] progress stall detected: cpu_mean=%.3f cpu_band=%.3f, "
            "net_kb_s=%.1f, disk_kb_s=%.1f, completed_per_min=%.2f, "
            "num_active=%d, num_waiting=%d; "
            "per_company_est_mb %.1f -> %.1f, target_parallel %d -> %d, "
            "mem_cap_frac %.3f -> %.3f, mem_high_frac %.3f -> %.3f",
            cpu_mean,
            cpu_band,
            mean_net / 1024.0,
            mean_disk / 1024.0,
            completed_per_min,
            num_active,
            num_waiting,
            old_est,
            self._per_company_est_mb,
            old_target,
            self._target_parallel,
            old_cap,
            cfg.mem_cap_frac,
            old_high,
            cfg.mem_high_frac,
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

        # Soft near-OOM when already high and ramping fast
        soft_high_and_fast = (
            used_frac >= cfg.mem_high_frac
            and self._last_trend_slope_mb_s >= cfg.mem_trend_slope_high_mb_per_s
        )

        is_near = (
            used_frac >= cfg.near_oom_used_frac
            or (
                swap_frac >= cfg.near_oom_swap_frac
                and used_frac >= cfg.mem_crit_low_frac
            )
            or soft_high_and_fast
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
                "used_frac=%.3f swap_frac=%.3f slope=%.1fMB/s "
                "mem_cap_frac: %.3f -> %.3f, mem_high_frac: %.3f -> %.3f, "
                "mem_crit_high_frac: %.3f -> %.3f",
                self._near_oom_events,
                used_frac,
                swap_frac,
                self._last_trend_slope_mb_s,
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
            eff = min(eff, 0.91)
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
            "completed_counter": self._completed_counter,
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
]
