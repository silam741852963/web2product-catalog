from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import asyncio
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

    min_active_keep: int = 1

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

    # (3) Per-company profiles
    company_profile_max_size: int = 2000

    # Unstall heuristics
    # If slots_by_mem == 0 but used_frac is clearly below this, we relax estimates.
    unstall_low_frac: float = 0.70
    # Cooldown between automatic resets in seconds
    unstall_cooldown_sec: float = 60.0


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

        # History of approximate per company usage in MB
        self._peak_history_mb: List[float] = []

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
                return 0

            # Active companies for estimation and target cap
            active_ids = list(self._get_active_company_ids())
            num_active = len(active_ids)
            self._update_per_company_estimate_locked(
                used_bytes=used, num_active=num_active, used_frac=used_frac
            )

            # Effective per-company estimate (global + per-company profiles)
            effective_est_mb = self._per_company_est_mb
            company_p95 = self._get_company_p95_est_mb_locked()
            if company_p95 is not None and company_p95 > effective_est_mb:
                effective_est_mb = company_p95

            # Bound effective estimate so one bad episode can't permanently
            # exceed the safe cap.
            max_safe_mb = (cfg.mem_cap_frac * float(total)) / 1e6
            if max_safe_mb > 0.0 and effective_est_mb > max_safe_mb:
                effective_est_mb = max_safe_mb

            per_company_bytes = max(
                int(effective_est_mb * 1e6),
                int(self.cfg.per_company_min_reservation_mb * 1e6),
            )
            if per_company_bytes <= 0:
                return 0

            slots_by_mem = int(headroom_bytes // per_company_bytes)

            # Key anti-stall 1: if nothing is active, always allow at least one
            # company to start (below mem_high_frac), even if the model says 0.
            if num_active == 0 and slots_by_mem <= 0 and used_frac < cfg.mem_high_frac:
                slots_by_mem = 1

            # Key anti-stall 2: estimator has become over-conservative.
            # Condition: we are below mem_cap_frac, not in mem_trouble,
            # there is headroom, num_active > 0, num_waiting > 0,
            # but slots_by_mem == 0.
            if (
                slots_by_mem <= 0
                and not mem_trouble
                and headroom_bytes > 0
                and num_active > 0
                and num_waiting > 0
                and used_frac < cfg.mem_cap_frac
            ):
                self._maybe_unstall_locked(
                    total=total,
                    used=used,
                    headroom_bytes=headroom_bytes,
                    used_frac=used_frac,
                )
                # recompute effective_est_mb and slots_by_mem after reset
                effective_est_mb = self._per_company_est_mb
                company_p95 = self._get_company_p95_est_mb_locked()
                if company_p95 is not None and company_p95 > effective_est_mb:
                    effective_est_mb = company_p95

                max_safe_mb = (cfg.mem_cap_frac * float(total)) / 1e6
                if max_safe_mb > 0.0 and effective_est_mb > max_safe_mb:
                    effective_est_mb = max_safe_mb

                per_company_bytes = max(
                    int(effective_est_mb * 1e6),
                    int(self.cfg.per_company_min_reservation_mb * 1e6),
                )
                if per_company_bytes > 0:
                    slots_by_mem = int(headroom_bytes // per_company_bytes)

            if slots_by_mem <= 0:
                # Still no safe slot.
                return 0

            # Apply AIMD target cap
            max_by_target = max(0, self._target_parallel - num_active)
            slots = min(slots_by_mem, num_waiting, max_by_target)

            # If trend looks precritical, be conservative and clamp to 1 slot
            if precritical and slots > 1:
                slots = 1

            self._maybe_log_state_locked(
                reason="admission",
                extra={
                    "slots": slots,
                    "num_waiting": num_waiting,
                    "num_active": num_active,
                    "per_company_est_mb": self._per_company_est_mb,
                    "effective_per_company_est_mb": effective_est_mb,
                    "total_mem_mb": float(total) / 1e6,
                    "used_mem_mb": float(used) / 1e6,
                    "used_frac": used_frac,
                    "swap_used_frac": swap_frac,
                    "trend_slope_mb_s": self._last_trend_slope_mb_s,
                    "target_parallel": self._target_parallel,
                    "slots_by_mem": slots_by_mem,
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

        # Clear history and per-company profiles so we don't immediately
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

                    # Update per company estimate based on current usage and active count
                    active_ids = list(self._get_active_company_ids())
                    num_active = len(active_ids)
                    self._update_per_company_estimate_locked(
                        used_bytes=used, num_active=num_active, used_frac=used_frac
                    )

                    # Emergency condition: high RAM or very high swap while RAM is already high
                    in_emergency = used_frac >= cfg.mem_crit_high_frac or (
                        swap_frac >= cfg.swap_emergency_frac
                        and used_frac >= cfg.mem_crit_low_frac
                    )
                    if not in_emergency:
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

                    # Determine how many companies we need to cancel to get
                    # down to mem_crit_low_frac.
                    target_used_bytes = int(cfg.mem_crit_low_frac * float(total))
                    excess_bytes = max(0, used - target_used_bytes)

                    # Effective per-company estimate (global + per-company profiles)
                    effective_est_mb = self._per_company_est_mb
                    company_p95 = self._get_company_p95_est_mb_locked()
                    if company_p95 is not None and company_p95 > effective_est_mb:
                        effective_est_mb = company_p95

                    max_safe_mb = (cfg.mem_cap_frac * float(total)) / 1e6
                    if max_safe_mb > 0.0 and effective_est_mb > max_safe_mb:
                        effective_est_mb = max_safe_mb

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

                    # Respect min_active_keep.
                    max_cancelable = max(0, num_active - int(cfg.min_active_keep))
                    if max_cancelable <= 0:
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

                    to_cancel_count = min(needed_cancel, max_cancelable)
                    to_cancel = list(active_ids)[-to_cancel_count:]

                    logger.error(
                        "[AdaptiveScheduling] emergency memory pressure: "
                        "used_frac=%.3f, swap_used_frac=%.3f, "
                        "total_mem_mb=%.1f, used_mem_mb=%.1f, "
                        "num_active=%d, per_company_est_mb=%.1f, "
                        "effective_per_company_est_mb=%.1f, "
                        "excess_bytes=%.1fMB, canceling=%d companies.",
                        used_frac,
                        swap_frac,
                        float(total) / 1e6,
                        float(used) / 1e6,
                        num_active,
                        self._per_company_est_mb,
                        effective_est_mb,
                        float(excess_bytes) / 1e6,
                        len(to_cancel),
                    )

                    cancel_ids = list(to_cancel)

                if cancel_ids:
                    try:
                        self._request_cancel_companies(cancel_ids)
                    except Exception:
                        logger.exception(
                            "[AdaptiveScheduling] request_cancel_companies failed"
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
        """
        if used_bytes <= 0 or self._total_mem_bytes is None:
            return

        cfg = self.cfg
        total = float(self._total_mem_bytes)
        used = float(used_bytes)

        if num_active <= 0:
            return

        per_company_now_mb = (used / float(num_active)) / 1e6

        # Only add to peak history when we have at least 2 actives.
        if num_active >= 2:
            self._peak_history_mb.append(per_company_now_mb)
            if len(self._peak_history_mb) > cfg.peak_history_size:
                self._peak_history_mb.pop(0)

        # Upward estimate from history (q95 * safety_factor)
        sorted_hist = sorted(self._peak_history_mb)
        if sorted_hist:
            idx = max(0, int(0.95 * len(sorted_hist)) - 1)
            q95 = sorted_hist[idx]
            base = max(q95, cfg.per_company_min_reservation_mb)
            upward_est = cfg.per_company_safety_factor * base
        else:
            upward_est = self._per_company_est_mb

        # Bound upward estimate by safe cap.
        max_safe_mb = (cfg.mem_cap_frac * total) / 1e6
        if max_safe_mb > 0.0 and upward_est > max_safe_mb:
            upward_est = max_safe_mb

        old_est = self._per_company_est_mb
        new_est = max(cfg.per_company_min_reservation_mb, upward_est)

        # Controlled downward adjustment:
        # If memory usage is clearly in a safe region, and the naive
        # per-company_now_mb is significantly smaller than our current
        # estimate, gently move the estimate downwards.
        if used_frac < cfg.unstall_low_frac and per_company_now_mb * 1.5 < old_est:
            # Move 10% of the way towards the current observation (smoothed)
            target = max(cfg.per_company_min_reservation_mb, per_company_now_mb)
            lowered = old_est * 0.9 + target * 0.1
            if lowered < new_est:
                new_est = lowered

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
        Detect near-OOM situations and auto-tune mem_cap_frac / mem_high_frac
        downward for the remainder of the process lifetime.
        """
        cfg = self.cfg
        is_near = used_frac >= cfg.near_oom_used_frac or (
            swap_frac >= cfg.near_oom_swap_frac and used_frac >= cfg.mem_crit_low_frac
        )

        if is_near and not self._last_near_oom_flag:
            self._near_oom_events += 1
            old_cap = cfg.mem_cap_frac
            old_high = cfg.mem_high_frac
            step = cfg.near_oom_mem_cap_step

            if step > 0.0:
                cfg.mem_cap_frac = max(cfg.mem_cap_min_frac, cfg.mem_cap_frac - step)
                cfg.mem_high_frac = max(cfg.mem_high_min_frac, cfg.mem_high_frac - step)

            logger.warning(
                "[AdaptiveScheduling] near-oom event #%d: "
                "used_frac=%.3f swap_frac=%.3f "
                "mem_cap_frac: %.3f -> %.3f, mem_high_frac: %.3f -> %.3f",
                self._near_oom_events,
                used_frac,
                swap_frac,
                old_cap,
                cfg.mem_cap_frac,
                old_high,
                cfg.mem_high_frac,
            )

        self._last_near_oom_flag = is_near

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


__all__ = [
    "AdaptiveSchedulingConfig",
    "AdaptiveScheduler",
    "CriticalMemoryPressure",
    "MEMORY_PRESSURE_MARKER",
]
