# adaptive_scheduling.py
from __future__ import annotations

import asyncio
import collections
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

_MB = 1_000_000


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

    # -----------------------------
    # Core watermarks
    # -----------------------------
    mem_cap_frac: float = 0.78
    mem_high_frac: float = 0.83
    mem_crit_high_frac: float = 0.90
    mem_crit_low_frac: float = 0.88

    # -----------------------------
    # Estimation (per-company)
    # -----------------------------
    peak_history_size: int = 100
    peak_history_horizon_sec: float = 1800.0  # 30 minutes

    per_company_safety_factor: float = 1.3
    per_company_min_reservation_mb: float = 256.0
    per_company_max_reservation_mb: float = 1024.0

    # -----------------------------
    # Sampling / overhead controls
    # -----------------------------
    emergency_check_interval_sec: float = 1.0
    min_admission_sample_interval_sec: float = 0.35  # avoid over-sampling on fast loops
    swap_sample_interval_sec: float = 2.0

    log_path: Optional[Path] = None
    use_psutil: bool = True
    prefer_cgroup_limits: bool = True

    # Swap signals
    swap_block_frac: float = 0.60
    swap_emergency_frac: float = 0.80

    # Memory trend
    mem_trend_window_sec: float = 10.0
    mem_trend_slope_high_mb_per_s: float = 100.0
    mem_trend_margin_frac: float = 0.03
    mem_trend_emergency_slope_mb_s: float = 40.0

    # -----------------------------
    # AIMD concurrency controller
    # -----------------------------
    ai_step: int = 1
    md_factor: float = 0.5
    min_target: int = 1
    max_target: int = 512
    initial_target: int = 4
    warmup_updates: int = 10
    warmup_ai_step: int = 3

    # -----------------------------
    # Auto-tune on near-OOM
    # -----------------------------
    near_oom_used_frac: float = 0.93
    near_oom_swap_frac: float = 0.50
    near_oom_mem_cap_step: float = 0.02
    mem_cap_min_frac: float = 0.70
    mem_high_min_frac: float = 0.80
    mem_crit_min_frac: float = 0.90

    # -----------------------------
    # Per-company profile cache
    # -----------------------------
    company_profile_max_size: int = 2000
    company_p95_cache_interval_sec: float = 10.0

    # -----------------------------
    # Unstall / poisoned estimate rescue
    # -----------------------------
    unstall_low_frac: float = 0.70
    unstall_cooldown_sec: float = 60.0
    unstall_slots_threshold: int = 1
    unstall_margin_frac: float = 0.05

    low_mem_stall_relax_used_frac: float = 0.65
    low_mem_stall_relax_cooldown_sec: float = 20.0
    low_mem_stall_relax_shrink: float = 0.75

    # -----------------------------
    # SAFE BURST
    # -----------------------------
    safe_burst_used_frac: float = 0.72
    safe_burst_slope_mb_s: float = 60.0
    safe_burst_swap_frac: float = 0.30
    safe_target_align: bool = True

    # -----------------------------
    # Emergency cancellation
    # -----------------------------
    min_active_keep: int = 0
    emergency_cancel_cooldown_sec: float = 3.0
    emergency_post_cancel_delay_sec: float = 3.0
    max_emergency_cancel_per_step: int = 3
    emergency_persistent_rounds_threshold: int = 3

    # -----------------------------
    # Per-task RSS monitoring (optional; needs PID callback)
    # -----------------------------
    per_task_rss_limit_mb: float = 500.0
    per_task_rss_duration_sec: float = 2.0
    per_task_monitor_enabled: bool = True
    per_task_monitor_interval_sec: float = 2.0
    per_task_monitor_max_companies: int = 32

    # Slope-based early pre-oom cancellation
    preoom_used_frac: float = 0.83
    preoom_slope_mb_s: float = 20.0
    preoom_cancel_count: int = 1

    # -----------------------------
    # SYSTEM STALL detector (CPU plateau + low disk/net)
    # -----------------------------
    # IMPORTANT CHANGE (fix for your #3/#5 stalls):
    #   Stall detection now ALSO requires "no progress" for some time,
    #   and it works even when num_active == 1 (single stuck company).
    #
    # Stall criteria (still based on CPU plateau + low IO), gated by:
    #   - no company completion for >= stall_no_progress_sec
    #   - samples cover >= stall_window_sec (or at least stall_min_window_sec)
    stall_window_sec: float = 180.0
    stall_min_window_sec: float = 45.0
    stall_no_progress_sec: float = 300.0  # gate to avoid canceling slow-but-alive runs

    stall_min_active: int = 1  # <--- changed from 2 to handle single stuck company
    stall_cpu_plateau_band_pct: float = 2.0
    stall_cpu_max_pct: float = 98.0
    stall_disk_rate_bytes_per_s: float = 32_000.0  # 32 KB/s
    stall_net_rate_bytes_per_s: float = 16_000.0  # 16 KB/s

    # Extra guard: allow SOME net activity but still "stalled" if it's tiny and flat
    stall_net_soft_cap_bytes_per_s: float = 64_000.0  # 64KB/s
    stall_net_soft_cap_required_plateau: bool = True

    # Stall actions
    stall_action_cooldown_sec: float = 30.0
    stall_cancel_count: int = 1
    stall_escalate_every_rounds: int = 2
    stall_cancel_max: int = 3
    stall_max_rounds_before_restart: int = 4
    stall_restart_after_sec: float = 900.0

    # When stall is detected, also reduce target_parallel (helps “unstick” loops)
    stall_reduce_target: bool = True
    stall_reduce_target_factor: float = 0.7

    # -----------------------------
    # NEW: "No-progress deadlock" breaker (for leaked resources after cancel)
    # -----------------------------
    # This targets the case you described: tasks get cancelled but CPU/RAM stays pinned,
    # so admission never recovers and the run stops making progress.
    deadlock_no_progress_sec: float = 600.0
    deadlock_check_interval_sec: float = 5.0
    deadlock_used_frac_floor: float = (
        0.50  # only consider if process is meaningfully loaded
    )
    deadlock_slope_abs_mb_s: float = 2.0
    deadlock_cancel_max: int = 2
    deadlock_max_active: int = 3
    deadlock_action_cooldown_sec: float = 60.0
    deadlock_rounds_before_restart: int = 3


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class AdaptiveScheduler:
    """
    Memory-based adaptive scheduler with:
      - Memory headroom admission control
      - Emergency near-OOM canceller
      - System-stall breaker (CPU plateau + low disk/net) gated by "no progress"
      - Deadlock breaker for "cancelled-but-resources-still-pinned" situations
    """

    def __init__(
        self,
        cfg: AdaptiveSchedulingConfig,
        get_active_company_ids: Callable[[], Sequence[str]],
        request_cancel_companies: Callable[[Sequence[str]], None],
        get_company_pids: Optional[Callable[[str], Sequence[int]]] = None,
    ) -> None:
        self.cfg = cfg
        self._psutil_available: bool = bool(PSUTIL_AVAILABLE and cfg.use_psutil)

        self._get_active_company_ids = get_active_company_ids
        self._request_cancel_companies = request_cancel_companies
        self._get_company_pids = get_company_pids

        self._total_mem_bytes: Optional[int] = None

        # Histories
        self._peak_history_mb: collections.deque[Tuple[float, float]] = (
            collections.deque()
        )
        self._mem_samples: collections.deque[Tuple[float, int]] = collections.deque()

        # Estimator state
        self._per_company_est_mb: float = cfg.per_company_min_reservation_mb

        # Per-company LRU peak cache
        self._company_peak_mb: "collections.OrderedDict[str, float]" = (
            collections.OrderedDict()
        )
        self._company_p95_cache_mb: Optional[float] = None
        self._company_p95_cache_ts: float = 0.0

        # Watchdog / sync
        self._emergency_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        self._restart_recommended: bool = False
        self._last_log_ts: float = 0.0

        # Cached samples (hot path uses these)
        self._last_used_bytes: int = 0
        self._last_used_frac: float = 0.0
        self._last_swap_used_frac: float = 0.0
        self._last_sample_ts: float = 0.0
        self._last_swap_sample_ts: float = 0.0

        # Trend estimation
        self._last_trend_slope_mb_s: float = 0.0

        # AIMD target
        self._target_parallel: int = min(
            max(self.cfg.min_target, self.cfg.initial_target), self.cfg.max_target
        )
        self._update_calls: int = 0

        # Near-OOM tuning
        self._near_oom_events: int = 0
        self._last_near_oom_flag: bool = False

        # Unstall bookkeeping
        self._last_low_mem_relax_ts: float = 0.0

        # Emergency bookkeeping
        self._last_emergency_cancel_ts: float = 0.0
        self._emergency_rounds: int = 0

        # Base RSS estimate
        self._base_rss_bytes: float = 0.0
        self._base_rss_samples: int = 0

        # Per-task RSS monitoring
        self._per_task_over_thresh_ts: Dict[str, float] = {}
        self._per_task_last_rss_bytes: Dict[str, int] = {}
        self._last_per_task_check_ts: float = 0.0

        # Progress counters
        self._completed_counter: int = 0
        self._last_num_waiting: int = 0
        self._last_progress_ts: float = time.time()
        self._last_completed_counter_seen: int = 0

        # Stall tracking (CPU plateau + low I/O)
        # samples: (ts, cpu_pct, disk_bytes_total, net_bytes_total)
        self._stall_samples: collections.deque[Tuple[float, float, int, int]] = (
            collections.deque()
        )
        self._stall_rounds: int = 0
        self._stall_first_detect_ts: Optional[float] = None
        self._last_stall_action_ts: float = 0.0

        self._last_cpu_pct: float = 0.0
        self._last_disk_rate_bps: float = 0.0
        self._last_net_rate_bps: float = 0.0

        # Deadlock tracking
        self._deadlock_rounds: int = 0
        self._last_deadlock_action_ts: float = 0.0
        self._last_deadlock_check_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def psutil_available(self) -> bool:
        return self._psutil_available

    @property
    def restart_recommended(self) -> bool:
        return self._restart_recommended

    def register_company_completed(self) -> None:
        self._completed_counter += 1
        self._last_progress_ts = time.time()

    def record_company_peak(self, company_id: str, peak_mb: float) -> None:
        if not company_id or peak_mb <= 0:
            return
        prev = self._company_peak_mb.get(company_id)
        if prev is None or peak_mb > prev:
            self._company_peak_mb[company_id] = peak_mb
        self._company_peak_mb.move_to_end(company_id, last=True)

        max_size = max(1, int(self.cfg.company_profile_max_size))
        while len(self._company_peak_mb) > max_size:
            self._company_peak_mb.popitem(last=False)

        self._company_p95_cache_mb = None

    async def start(self) -> None:
        if not self._psutil_available:
            logger.warning(
                "[AdaptiveScheduling] psutil not available/disabled; conservative mode (no watchdog)."
            )
            return

        total, used = self._read_memory_usage_bytes()
        self._total_mem_bytes = total or None
        self._last_used_bytes = used
        self._last_used_frac = (float(used) / float(total)) if total > 0 else 0.0

        # sample swap once at boot
        self._last_swap_used_frac = self._read_swap_used_frac()
        now = time.time()
        self._last_swap_sample_ts = now
        self._last_sample_ts = now

        # prime cpu_percent internal state (first call is often 0.0)
        try:
            if psutil is not None:
                psutil.cpu_percent(interval=None)  # type: ignore[union-attr]
        except Exception:
            pass

        # prime stall samples with disk/net baselines
        try:
            cpu_pct, disk_b, net_b = self._read_cpu_disk_net_sample()
            self._stall_samples.clear()
            self._stall_samples.append((now, cpu_pct, disk_b, net_b))
            self._last_cpu_pct = cpu_pct
        except Exception:
            pass

        logger.info(
            "[AdaptiveScheduling] started (total_mem_mb=%.1f, psutil=True, initial_target_parallel=%d)",
            (float(total) / _MB) if total > 0 else -1.0,
            self._target_parallel,
        )

        interval = max(0.5, float(self.cfg.emergency_check_interval_sec))
        self._emergency_task = asyncio.create_task(
            self._emergency_loop(interval),
            name="adaptive-scheduling-emergency-watchdog",
        )

    async def stop(self) -> None:
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
                "[AdaptiveScheduling] error while stopping watchdog", exc_info=True
            )

    async def admissible_slots(self, num_waiting: int) -> int:
        """
        Decide how many new companies can be started right now.

        Hot-path goals:
          - minimize psutil calls (cached sampling + swap throttling)
          - keep lock region small
        """
        if num_waiting <= 0:
            return 0
        self._last_num_waiting = num_waiting

        if not self._psutil_available:
            return 1  # conservative fallback

        now = time.time()
        cfg = self.cfg

        async with self._lock:
            # 1) Sample memory only if needed
            if (now - self._last_sample_ts) >= cfg.min_admission_sample_interval_sec:
                total, used = self._read_memory_usage_bytes()
                if total <= 0:
                    return 0

                self._total_mem_bytes = total
                self._last_used_bytes = used
                self._last_used_frac = float(used) / float(total)
                self._last_sample_ts = now

                self._record_memory_sample_locked(ts=now, used_bytes=used)

                # swap sampling throttled
                if (now - self._last_swap_sample_ts) >= cfg.swap_sample_interval_sec:
                    self._last_swap_used_frac = self._read_swap_used_frac()
                    self._last_swap_sample_ts = now

                self._register_near_oom_locked(
                    self._last_used_frac, self._last_swap_used_frac
                )

            total = self._total_mem_bytes or 0
            if total <= 0:
                return 0

            used = self._last_used_bytes
            used_frac = self._last_used_frac
            swap_frac = self._last_swap_used_frac
            slope = self._last_trend_slope_mb_s

            active_ids = self._get_active_company_ids()
            num_active = len(active_ids)

            # base RSS estimate only in idle
            self._update_base_rss_locked(used_bytes=used, num_active=num_active)

            # 2) Update AIMD target (cheap)
            precritical = (
                used_frac >= max(0.0, cfg.mem_cap_frac - cfg.mem_trend_margin_frac)
                and slope >= cfg.mem_trend_slope_high_mb_per_s
            )
            high_swap_block = (
                swap_frac >= cfg.swap_block_frac and used_frac >= cfg.mem_high_frac
            )
            mem_trouble = used_frac >= cfg.mem_high_frac or high_swap_block

            self._update_target_parallel_locked(
                used_frac=used_frac,
                precritical=precritical,
                high_swap=high_swap_block,
                mem_trouble=mem_trouble,
            )

            if mem_trouble:
                self._maybe_log_state_locked(
                    reason="mem_high_block",
                    extra={
                        "used_frac": used_frac,
                        "swap_used_frac": swap_frac,
                        "num_active": num_active,
                        "trend_slope_mb_s": slope,
                        "target_parallel": self._target_parallel,
                    },
                )
                return 0

            # 3) Compute headroom and mem-limited slots
            mem_cap_bytes = int(cfg.mem_cap_frac * float(total))
            headroom_bytes = mem_cap_bytes - used
            if headroom_bytes <= 0:
                self._maybe_log_state_locked(
                    reason="mem_cap_block",
                    extra={
                        "used_frac": used_frac,
                        "swap_used_frac": swap_frac,
                        "num_active": num_active,
                        "trend_slope_mb_s": slope,
                        "target_parallel": self._target_parallel,
                    },
                )
                return 0

            # 4) Update per-company estimator
            self._update_per_company_estimate_locked(
                used_bytes=used, num_active=num_active, used_frac=used_frac
            )
            effective_est_mb = self._effective_per_company_est_mb_locked()

            per_company_bytes = max(
                int(effective_est_mb * _MB),
                int(cfg.per_company_min_reservation_mb * _MB),
            )
            if per_company_bytes <= 0:
                return 0

            slots_by_mem = int(headroom_bytes // per_company_bytes)

            # Poisoned estimator rescue (low-mem stall relax)
            if slots_by_mem <= 0:
                self._maybe_low_mem_stall_relax_locked(
                    total=total,
                    used=used,
                    headroom_bytes=headroom_bytes,
                    used_frac=used_frac,
                )
                effective_est_mb = self._effective_per_company_est_mb_locked()
                per_company_bytes = max(
                    int(effective_est_mb * _MB),
                    int(cfg.per_company_min_reservation_mb * _MB),
                )
                if per_company_bytes > 0:
                    slots_by_mem = int(headroom_bytes // per_company_bytes)

            if num_active == 0 and slots_by_mem <= 0 and used_frac < cfg.mem_high_frac:
                slots_by_mem = 1

            if slots_by_mem <= 0:
                return 0

            # 5) Soft target cap, allow safe burst
            max_by_target = max(0, self._target_parallel - num_active)

            safe_burst = (
                used_frac <= cfg.safe_burst_used_frac
                and slope <= cfg.safe_burst_slope_mb_s
                and swap_frac <= cfg.safe_burst_swap_frac
            )

            if cfg.safe_target_align and safe_burst:
                desired = min(
                    cfg.max_target, max(cfg.min_target, num_active + slots_by_mem)
                )
                if desired > self._target_parallel:
                    self._target_parallel = desired

            if safe_burst:
                hard_cap = max(0, cfg.max_target - num_active)
                slots = min(slots_by_mem, num_waiting, hard_cap)
            else:
                slots = min(slots_by_mem, num_waiting, max_by_target)

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
                    "total_mem_mb": float(total) / _MB,
                    "used_mem_mb": float(used) / _MB,
                    "used_frac": used_frac,
                    "swap_used_frac": swap_frac,
                    "trend_slope_mb_s": slope,
                    "target_parallel": self._target_parallel,
                    "slots_by_mem": slots_by_mem,
                    "base_rss_mb": self._base_rss_bytes / _MB,
                    "safe_burst": safe_burst,
                },
            )

            return slots

    def get_state_snapshot(self) -> Dict[str, Any]:
        total = self._total_mem_bytes or 0
        used = self._last_used_bytes

        per_task_items = list(self._per_task_last_rss_bytes.items())
        per_task_items.sort(key=lambda kv: kv[1], reverse=True)
        per_task_items = per_task_items[:10]
        per_task_rss_mb = {cid: (b / _MB) for cid, b in per_task_items}

        stall_age = 0.0
        if self._stall_first_detect_ts is not None:
            stall_age = max(0.0, time.time() - self._stall_first_detect_ts)

        no_progress_age = max(0.0, time.time() - self._last_progress_ts)

        return {
            "total_mem_mb": float(total) / _MB if total > 0 else 0.0,
            "used_mem_mb": float(used) / _MB if used > 0 else 0.0,
            "used_frac": self._last_used_frac,
            "swap_used_frac": self._last_swap_used_frac,
            "per_company_est_mb": self._per_company_est_mb,
            "trend_slope_mb_s": self._last_trend_slope_mb_s,
            "target_parallel": self._target_parallel,
            "near_oom_events": self._near_oom_events,
            "psutil_available": self._psutil_available,
            "restart_recommended": self._restart_recommended,
            "mem_cap_frac": self.cfg.mem_cap_frac,
            "mem_high_frac": self.cfg.mem_high_frac,
            "mem_crit_high_frac": self.cfg.mem_crit_high_frac,
            "base_rss_mb": self._base_rss_bytes / _MB,
            "completed_counter": self._completed_counter,
            "no_progress_age_sec": no_progress_age,
            "stall_rounds": self._stall_rounds,
            "stall_age_sec": stall_age,
            "stall_last_cpu_pct": self._last_cpu_pct,
            "stall_last_disk_rate_bps": self._last_disk_rate_bps,
            "stall_last_net_rate_bps": self._last_net_rate_bps,
            "deadlock_rounds": self._deadlock_rounds,
            "per_task_rss_mb_sample": per_task_rss_mb,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_cgroup_memory_bytes(self) -> Tuple[int, int]:
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
        if not self._psutil_available or psutil is None:
            return 0, 0

        if self.cfg.prefer_cgroup_limits:
            limit, used = self._read_cgroup_memory_bytes()
            if limit > 0:
                return limit, used

        try:
            vm = psutil.virtual_memory()  # type: ignore[union-attr]
            total = int(vm.total)
            used = int(vm.total - vm.available)
            return total, used
        except Exception:
            logger.debug(
                "[AdaptiveScheduling] failed to read memory via psutil", exc_info=True
            )
            return 0, 0

    def _read_swap_used_frac(self) -> float:
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
        window = max(1.0, float(self.cfg.mem_trend_window_sec))
        self._mem_samples.append((ts, used_bytes))
        cutoff = ts - window
        while self._mem_samples and self._mem_samples[0][0] < cutoff:
            self._mem_samples.popleft()

        if len(self._mem_samples) >= 2:
            t0, u0 = self._mem_samples[0]
            t1, u1 = self._mem_samples[-1]
            dt = t1 - t0
            self._last_trend_slope_mb_s = (
                (((float(u1) - float(u0)) / dt) / _MB) if dt > 0 else 0.0
            )
        else:
            self._last_trend_slope_mb_s = 0.0

    def _update_base_rss_locked(self, *, used_bytes: int, num_active: int) -> None:
        if used_bytes <= 0 or num_active != 0:
            return
        alpha = 0.1
        if self._base_rss_samples == 0:
            self._base_rss_bytes = float(used_bytes)
            self._base_rss_samples = 1
        else:
            self._base_rss_bytes = (1.0 - alpha) * self._base_rss_bytes + alpha * float(
                used_bytes
            )
            self._base_rss_samples += 1

    def _effective_per_company_est_mb_locked(self) -> float:
        cfg = self.cfg
        effective_est_mb = self._per_company_est_mb

        p95 = self._get_company_p95_est_mb_locked()
        if p95 is not None and p95 > effective_est_mb:
            effective_est_mb = p95

        total = self._total_mem_bytes
        if total:
            total_mb = float(total) / _MB
            max_safe_mb = cfg.mem_cap_frac * total_mb
            if max_safe_mb > 0.0 and effective_est_mb > max_safe_mb:
                effective_est_mb = max_safe_mb

        if effective_est_mb > cfg.per_company_max_reservation_mb:
            effective_est_mb = cfg.per_company_max_reservation_mb
        if effective_est_mb < cfg.per_company_min_reservation_mb:
            effective_est_mb = cfg.per_company_min_reservation_mb

        return effective_est_mb

    def _get_company_p95_est_mb_locked(self) -> Optional[float]:
        if not self._company_peak_mb:
            return None

        now = time.time()
        if (
            self._company_p95_cache_mb is not None
            and (now - self._company_p95_cache_ts)
            < self.cfg.company_p95_cache_interval_sec
        ):
            return self._company_p95_cache_mb

        vals = sorted(self._company_peak_mb.values())
        idx = max(0, int(0.95 * len(vals)) - 1)
        p95 = vals[idx] if vals else None

        self._company_p95_cache_mb = p95
        self._company_p95_cache_ts = now
        return p95

    def _update_per_company_estimate_locked(
        self,
        *,
        used_bytes: int,
        num_active: int,
        used_frac: float,
    ) -> None:
        if used_bytes <= 0 or self._total_mem_bytes is None or num_active <= 0:
            return

        cfg = self.cfg
        total = float(self._total_mem_bytes)
        used = float(used_bytes)

        effective_used = used - self._base_rss_bytes
        if effective_used < 0:
            effective_used = 0.0
        per_company_now_mb = (effective_used / float(num_active)) / _MB

        now = time.time()
        if num_active >= 1 and per_company_now_mb > 0.0:
            self._peak_history_mb.append((now, per_company_now_mb))
            while len(self._peak_history_mb) > cfg.peak_history_size:
                self._peak_history_mb.popleft()
            cutoff = now - float(cfg.peak_history_horizon_sec)
            while self._peak_history_mb and self._peak_history_mb[0][0] < cutoff:
                self._peak_history_mb.popleft()

        if self._peak_history_mb:
            values = [v for (_, v) in self._peak_history_mb]
            values.sort()
            idx = max(0, int(0.95 * len(values)) - 1)
            q95 = values[idx]
            base = max(q95, cfg.per_company_min_reservation_mb)
            upward_est = cfg.per_company_safety_factor * base
        else:
            upward_est = self._per_company_est_mb

        max_safe_mb = (cfg.mem_cap_frac * total) / _MB
        if max_safe_mb > 0.0 and upward_est > max_safe_mb:
            upward_est = max_safe_mb
        if upward_est > cfg.per_company_max_reservation_mb:
            upward_est = cfg.per_company_max_reservation_mb

        old_est = self._per_company_est_mb
        new_est = max(cfg.per_company_min_reservation_mb, upward_est)

        # gentle downward nudge in safe regime
        if used_frac < cfg.unstall_low_frac and (per_company_now_mb * 1.5) < old_est:
            target = max(cfg.per_company_min_reservation_mb, per_company_now_mb)
            lowered = old_est * 0.9 + target * 0.1
            if lowered < new_est:
                new_est = lowered

        if new_est > cfg.per_company_max_reservation_mb:
            new_est = cfg.per_company_max_reservation_mb

        self._per_company_est_mb = new_est

        if self._per_company_est_mb > old_est * 1.1:
            logger.info(
                "[AdaptiveScheduling] per-company estimate updated: %.1fMB -> %.1fMB (active=%d, used_frac=%.3f)",
                old_est,
                self._per_company_est_mb,
                num_active,
                used_frac,
            )

    def _maybe_low_mem_stall_relax_locked(
        self,
        *,
        total: int,
        used: int,
        headroom_bytes: int,
        used_frac: float,
    ) -> None:
        cfg = self.cfg
        now = time.time()

        if used_frac >= cfg.low_mem_stall_relax_used_frac:
            return
        if now - self._last_low_mem_relax_ts < cfg.low_mem_stall_relax_cooldown_sec:
            return
        if headroom_bytes <= int(cfg.per_company_min_reservation_mb * _MB):
            return

        old_est = self._per_company_est_mb
        new_est = max(
            cfg.per_company_min_reservation_mb, old_est * cfg.low_mem_stall_relax_shrink
        )
        if new_est > cfg.per_company_max_reservation_mb:
            new_est = cfg.per_company_max_reservation_mb

        self._peak_history_mb.clear()

        self._per_company_est_mb = new_est
        self._last_low_mem_relax_ts = now

        logger.warning(
            "[AdaptiveScheduling] low-mem stall relax: per_company_est_mb %.1fMB -> %.1fMB (used_frac=%.3f, headroom_mb=%.1f)",
            old_est,
            new_est,
            used_frac,
            float(headroom_bytes) / _MB,
        )

    def _register_near_oom_locked(self, used_frac: float, swap_frac: float) -> None:
        cfg = self.cfg
        soft_high_and_fast = (
            used_frac >= cfg.mem_high_frac
            and self._last_trend_slope_mb_s >= cfg.mem_trend_slope_high_mb_s
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
                "[AdaptiveScheduling] near-oom event #%d: used_frac=%.3f swap_frac=%.3f slope=%.1fMB/s "
                "mem_cap_frac %.3f->%.3f mem_high_frac %.3f->%.3f mem_crit_high_frac %.3f->%.3f",
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

    def _update_target_parallel_locked(
        self,
        *,
        used_frac: float,
        precritical: bool,
        high_swap: bool,
        mem_trouble: bool,
    ) -> None:
        cfg = self.cfg
        old_target = self._target_parallel
        self._update_calls += 1

        if mem_trouble or high_swap or precritical:
            new_target = int(self._target_parallel * cfg.md_factor)
            new_target = max(cfg.min_target, max(1, new_target))
        else:
            margin = cfg.mem_trend_margin_frac
            if (
                used_frac < max(0.0, cfg.mem_cap_frac - margin)
                and self._last_trend_slope_mb_s <= cfg.mem_trend_slope_high_mb_s
            ):
                step = (
                    cfg.warmup_ai_step
                    if self._update_calls <= cfg.warmup_updates
                    else cfg.ai_step
                )
                new_target = min(cfg.max_target, self._target_parallel + step)
            else:
                new_target = self._target_parallel

        self._target_parallel = new_target

        if new_target != old_target and logger.isEnabledFor(logging.INFO):
            logger.info(
                "[AdaptiveScheduling] target_parallel %d -> %d (used_frac=%.3f swap=%.3f slope=%.1fMB/s precritical=%s mem_trouble=%s)",
                old_target,
                new_target,
                used_frac,
                self._last_swap_used_frac,
                self._last_trend_slope_mb_s,
                precritical,
                mem_trouble,
            )

    def _maybe_log_state_locked(
        self, *, reason: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.cfg.log_path is None:
            return
        now = time.time()
        if self._last_log_ts and now - self._last_log_ts < 5.0:
            return
        self._last_log_ts = now

        stall_age = 0.0
        if self._stall_first_detect_ts is not None:
            stall_age = max(0.0, now - self._stall_first_detect_ts)

        no_progress_age = max(0.0, now - self._last_progress_ts)

        state: Dict[str, Any] = {
            "ts": now,
            "reason": reason,
            "total_mem_bytes": self._total_mem_bytes,
            "last_used_bytes": self._last_used_bytes,
            "last_used_frac": self._last_used_frac,
            "last_swap_used_frac": self._last_swap_used_frac,
            "per_company_est_mb": self._per_company_est_mb,
            "trend_slope_mb_s": self._last_trend_slope_mb_s,
            "target_parallel": self._target_parallel,
            "near_oom_events": self._near_oom_events,
            "restart_recommended": self._restart_recommended,
            "mem_cap_frac": self.cfg.mem_cap_frac,
            "mem_high_frac": self.cfg.mem_high_frac,
            "mem_crit_high_frac": self.cfg.mem_crit_high_frac,
            "base_rss_mb": self._base_rss_bytes / _MB,
            "completed_counter": self._completed_counter,
            "no_progress_age_sec": no_progress_age,
            "stall_rounds": self._stall_rounds,
            "stall_age_sec": stall_age,
            "stall_last_cpu_pct": self._last_cpu_pct,
            "stall_last_disk_rate_bps": self._last_disk_rate_bps,
            "stall_last_net_rate_bps": self._last_net_rate_bps,
            "deadlock_rounds": self._deadlock_rounds,
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
            logger.debug("[AdaptiveScheduling] failed writing state log", exc_info=True)

    def _select_heaviest_companies_locked(
        self, active_ids: Sequence[str], count: int
    ) -> List[str]:
        if not active_ids or count <= 0:
            return []
        weighted: List[Tuple[str, float]] = []
        for cid in active_ids:
            mb = self._company_peak_mb.get(cid)
            if mb is not None:
                weighted.append((cid, mb))
        if not weighted:
            return []
        weighted.sort(key=lambda x: x[1], reverse=True)
        return [cid for (cid, _) in weighted[:count]]

    # ------------------------------------------------------------------ #
    # Stall sampling
    # ------------------------------------------------------------------ #

    def _read_cpu_disk_net_sample(self) -> Tuple[float, int, int]:
        """
        Returns:
          cpu_pct: float (0..100)
          disk_bytes_total: read_bytes + write_bytes (cumulative)
          net_bytes_total: bytes_sent + bytes_recv (cumulative)
        """
        if not self._psutil_available or psutil is None:
            return 0.0, 0, 0

        try:
            cpu_pct = float(psutil.cpu_percent(interval=None))  # type: ignore[union-attr]
        except Exception:
            cpu_pct = 0.0

        disk_b = 0
        try:
            dio = psutil.disk_io_counters()  # type: ignore[union-attr]
            if dio is not None:
                disk_b = int(getattr(dio, "read_bytes", 0)) + int(
                    getattr(dio, "write_bytes", 0)
                )
        except Exception:
            disk_b = 0

        net_b = 0
        try:
            nio = psutil.net_io_counters()  # type: ignore[union-attr]
            if nio is not None:
                net_b = int(getattr(nio, "bytes_sent", 0)) + int(
                    getattr(nio, "bytes_recv", 0)
                )
        except Exception:
            net_b = 0

        return cpu_pct, disk_b, net_b

    def _record_stall_sample_locked(self, *, now: float) -> None:
        cfg = self.cfg
        cpu_pct, disk_b, net_b = self._read_cpu_disk_net_sample()
        self._stall_samples.append((now, cpu_pct, disk_b, net_b))

        # update "last" telemetry (for logs/snapshots) using last delta
        if len(self._stall_samples) >= 2:
            t0, _, d0, n0 = self._stall_samples[-2]
            t1, c1, d1, n1 = self._stall_samples[-1]
            dt = max(1e-6, t1 - t0)
            self._last_cpu_pct = float(c1)
            self._last_disk_rate_bps = float(max(0, d1 - d0)) / dt
            self._last_net_rate_bps = float(max(0, n1 - n0)) / dt
        else:
            self._last_cpu_pct = float(cpu_pct)
            self._last_disk_rate_bps = 0.0
            self._last_net_rate_bps = 0.0

        cutoff = now - float(cfg.stall_window_sec)
        while self._stall_samples and self._stall_samples[0][0] < cutoff:
            self._stall_samples.popleft()

    def _compute_stall_metrics_locked(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Stall criteria (CPU plateau + low IO), gated by:
          - samples cover enough time
          - no progress for cfg.stall_no_progress_sec
        """
        cfg = self.cfg

        if len(self._stall_samples) < 2:
            return False, {}

        t0, _, d0, n0 = self._stall_samples[0]
        t1, cpu1, d1, n1 = self._stall_samples[-1]
        dt = t1 - t0
        if dt <= 0.0:
            return False, {}

        # require minimum window coverage
        if dt < float(cfg.stall_min_window_sec):
            return False, {"stall_dt_sec": dt, "window_ok": False}

        # "no progress" gate
        no_progress_age = max(0.0, t1 - self._last_progress_ts)
        if no_progress_age < float(cfg.stall_no_progress_sec):
            return False, {
                "stall_dt_sec": dt,
                "window_ok": True,
                "no_progress_age_sec": no_progress_age,
                "no_progress_ok": False,
            }

        cpus = [c for (_, c, _, _) in self._stall_samples]
        cpu_min = min(cpus) if cpus else 0.0
        cpu_max = max(cpus) if cpus else 0.0
        cpu_band = cpu_max - cpu_min
        cpu_level = float(cpu1)

        disk_rate = float(max(0, d1 - d0)) / dt
        net_rate = float(max(0, n1 - n0)) / dt

        plateau_ok = cpu_band <= float(cfg.stall_cpu_plateau_band_pct)
        cpu_level_ok = cpu_level <= float(cfg.stall_cpu_max_pct)
        disk_ok = disk_rate <= float(cfg.stall_disk_rate_bytes_per_s)
        net_ok = net_rate <= float(cfg.stall_net_rate_bytes_per_s)

        # soft net allowance (for tiny steady traffic)
        net_soft_ok = True
        if cfg.stall_net_soft_cap_required_plateau:
            net_soft_ok = net_rate <= float(cfg.stall_net_soft_cap_bytes_per_s)

        stalled = bool(
            plateau_ok and cpu_level_ok and disk_ok and (net_ok or net_soft_ok)
        )

        meta = {
            "stall_dt_sec": dt,
            "no_progress_age_sec": no_progress_age,
            "cpu_level": cpu_level,
            "cpu_min": cpu_min,
            "cpu_max": cpu_max,
            "cpu_band": cpu_band,
            "disk_rate_bps": disk_rate,
            "net_rate_bps": net_rate,
            "plateau_ok": plateau_ok,
            "cpu_level_ok": cpu_level_ok,
            "disk_ok": disk_ok,
            "net_ok": net_ok,
            "net_soft_ok": net_soft_ok,
        }
        return stalled, meta

    def _maybe_stall_actions_locked(
        self, *, now: float, active_ids: Sequence[str]
    ) -> List[str]:
        cfg = self.cfg
        num_active = len(active_ids)
        if num_active < int(cfg.stall_min_active):
            self._stall_first_detect_ts = None
            self._stall_rounds = 0
            return []

        stalled, meta = self._compute_stall_metrics_locked()
        if not stalled:
            self._stall_first_detect_ts = None
            self._stall_rounds = 0
            return []

        if self._stall_first_detect_ts is None:
            self._stall_first_detect_ts = now

        # cooldown
        if (now - self._last_stall_action_ts) < float(cfg.stall_action_cooldown_sec):
            self._maybe_log_state_locked(
                reason="stall_detected_cooldown",
                extra={"stall": meta, "num_active": num_active},
            )
            return []

        max_cancelable = max(0, num_active - int(cfg.min_active_keep))
        if max_cancelable <= 0:
            stall_age = now - (self._stall_first_detect_ts or now)
            if stall_age >= float(cfg.stall_restart_after_sec):
                self._restart_recommended = True
                logger.error(
                    "[AdaptiveScheduling] stall persists %.1fs but cannot cancel (active=%d min_keep=%d); restart recommended.",
                    stall_age,
                    num_active,
                    cfg.min_active_keep,
                )
            return []

        base = max(1, int(cfg.stall_cancel_count))
        extra = 0
        if int(cfg.stall_escalate_every_rounds) > 0:
            extra = self._stall_rounds // int(cfg.stall_escalate_every_rounds)
        to_cancel = min(
            max_cancelable,
            min(int(cfg.stall_cancel_max), base + extra),
        )

        cancel_ids = self._select_heaviest_companies_locked(active_ids, count=to_cancel)
        if not cancel_ids:
            cancel_ids = list(active_ids)[-to_cancel:]

        self._stall_rounds += 1
        self._last_stall_action_ts = now

        if cfg.stall_reduce_target:
            old_tp = self._target_parallel
            new_tp = int(old_tp * float(cfg.stall_reduce_target_factor))
            new_tp = max(int(cfg.min_target), max(1, new_tp))
            if new_tp < old_tp:
                self._target_parallel = new_tp

        stall_age = now - (self._stall_first_detect_ts or now)

        logger.error(
            "[AdaptiveScheduling] SYSTEM STALL detected (round=%d age=%.1fs active=%d) "
            "cpu=%.1f%% band=%.2f%% disk=%.1fB/s net=%.1fB/s no_progress=%.1fs -> cancel=%s target_parallel=%d",
            self._stall_rounds,
            stall_age,
            num_active,
            float(meta.get("cpu_level", 0.0)),
            float(meta.get("cpu_band", 0.0)),
            float(meta.get("disk_rate_bps", 0.0)),
            float(meta.get("net_rate_bps", 0.0)),
            float(meta.get("no_progress_age_sec", 0.0)),
            cancel_ids,
            self._target_parallel,
        )

        self._maybe_log_state_locked(
            reason="stall_cancel",
            extra={
                "stall": meta,
                "stall_rounds": self._stall_rounds,
                "stall_age_sec": stall_age,
                "num_active": num_active,
                "cancel_ids": cancel_ids,
                "target_parallel": self._target_parallel,
            },
        )

        if self._stall_rounds >= int(
            cfg.stall_max_rounds_before_restart
        ) or stall_age >= float(cfg.stall_restart_after_sec):
            self._restart_recommended = True
            logger.error(
                "[AdaptiveScheduling] stall persists (rounds=%d age=%.1fs); restart recommended.",
                self._stall_rounds,
                stall_age,
            )
            self._maybe_log_state_locked(
                reason="stall_restart",
                extra={
                    "stall": meta,
                    "stall_rounds": self._stall_rounds,
                    "stall_age_sec": stall_age,
                    "num_active": num_active,
                },
            )

        return cancel_ids

    def _maybe_deadlock_actions_locked(
        self,
        *,
        now: float,
        total: int,
        used: int,
        used_frac: float,
        active_ids: Sequence[str],
    ) -> List[str]:
        """
        Detect "no progress + stable memory + few actives" deadlock.
        This is specifically aimed at: cancellations occurred, but leaked child processes / contexts
        keep RSS pinned and progress never resumes.
        """
        cfg = self.cfg

        if (now - self._last_deadlock_check_ts) < float(
            cfg.deadlock_check_interval_sec
        ):
            return []
        self._last_deadlock_check_ts = now

        num_active = len(active_ids)
        if num_active <= 0:
            self._deadlock_rounds = 0
            return []

        no_progress_age = max(0.0, now - self._last_progress_ts)
        if no_progress_age < float(cfg.deadlock_no_progress_sec):
            self._deadlock_rounds = 0
            return []

        if num_active > int(cfg.deadlock_max_active):
            self._deadlock_rounds = 0
            return []

        if used_frac < float(cfg.deadlock_used_frac_floor):
            self._deadlock_rounds = 0
            return []

        # must be stable-ish (not rapidly changing)
        if abs(float(self._last_trend_slope_mb_s)) > float(cfg.deadlock_slope_abs_mb_s):
            self._deadlock_rounds = 0
            return []

        # cooldown between actions
        if (now - self._last_deadlock_action_ts) < float(
            cfg.deadlock_action_cooldown_sec
        ):
            return []

        max_cancelable = max(0, num_active - int(cfg.min_active_keep))
        if max_cancelable <= 0:
            self._deadlock_rounds += 1
            if self._deadlock_rounds >= int(cfg.deadlock_rounds_before_restart):
                self._restart_recommended = True
                logger.error(
                    "[AdaptiveScheduling] deadlock persists but cannot cancel (active=%d min_keep=%d); restart recommended.",
                    num_active,
                    cfg.min_active_keep,
                )
            return []

        to_cancel = min(int(cfg.deadlock_cancel_max), max_cancelable)
        cancel_ids = self._select_heaviest_companies_locked(active_ids, count=to_cancel)
        if not cancel_ids:
            cancel_ids = list(active_ids)[-to_cancel:]

        self._deadlock_rounds += 1
        self._last_deadlock_action_ts = now

        logger.error(
            "[AdaptiveScheduling] DEADLOCK detected (round=%d no_progress=%.1fs active=%d used=%.1f%% slope=%.2fMB/s) -> cancel=%s",
            self._deadlock_rounds,
            no_progress_age,
            num_active,
            used_frac * 100.0,
            float(self._last_trend_slope_mb_s),
            cancel_ids,
        )
        self._maybe_log_state_locked(
            reason="deadlock_cancel",
            extra={
                "deadlock_rounds": self._deadlock_rounds,
                "no_progress_age_sec": no_progress_age,
                "num_active": num_active,
                "used_frac": used_frac,
                "trend_slope_mb_s": self._last_trend_slope_mb_s,
                "cancel_ids": cancel_ids,
            },
        )

        if self._deadlock_rounds >= int(cfg.deadlock_rounds_before_restart):
            self._restart_recommended = True
            logger.error(
                "[AdaptiveScheduling] deadlock persists (rounds=%d); restart recommended.",
                self._deadlock_rounds,
            )
            self._maybe_log_state_locked(
                reason="deadlock_restart",
                extra={
                    "deadlock_rounds": self._deadlock_rounds,
                    "no_progress_age_sec": no_progress_age,
                    "num_active": num_active,
                    "used_frac": used_frac,
                },
            )

        return cancel_ids

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

                    now = time.time()
                    used_frac = float(used) / float(total)

                    self._total_mem_bytes = total
                    self._last_used_bytes = used
                    self._last_used_frac = used_frac
                    self._record_memory_sample_locked(ts=now, used_bytes=used)

                    # swap sampling throttled even in watchdog
                    if (
                        now - self._last_swap_sample_ts
                    ) >= cfg.swap_sample_interval_sec:
                        self._last_swap_used_frac = self._read_swap_used_frac()
                        self._last_swap_sample_ts = now
                    swap_frac = self._last_swap_used_frac

                    self._register_near_oom_locked(used_frac, swap_frac)

                    active_ids = self._get_active_company_ids()
                    num_active = len(active_ids)
                    self._update_base_rss_locked(used_bytes=used, num_active=num_active)

                    self._update_per_company_estimate_locked(
                        used_bytes=used, num_active=num_active, used_frac=used_frac
                    )

                    slope = self._last_trend_slope_mb_s

                    # Progress bookkeeping safety (if someone increments completed_counter externally)
                    if self._completed_counter != self._last_completed_counter_seen:
                        self._last_completed_counter_seen = self._completed_counter
                        self._last_progress_ts = now

                    # Record stall sample every watchdog tick; update last_cpu/disk/net telemetry.
                    try:
                        self._record_stall_sample_locked(now=now)
                    except Exception:
                        logger.debug(
                            "[AdaptiveScheduling] stall sampling failed", exc_info=True
                        )

                    # 1) Stall breaker (CPU plateau + low IO) gated by no-progress
                    try:
                        stall_cancels = self._maybe_stall_actions_locked(
                            now=now, active_ids=active_ids
                        )
                        cancel_ids.extend(stall_cancels)
                    except Exception:
                        logger.debug(
                            "[AdaptiveScheduling] stall decision failed", exc_info=True
                        )

                    # 2) Deadlock breaker (no-progress + stable memory + few actives)
                    try:
                        deadlock_cancels = self._maybe_deadlock_actions_locked(
                            now=now,
                            total=total,
                            used=used,
                            used_frac=used_frac,
                            active_ids=active_ids,
                        )
                        cancel_ids.extend(deadlock_cancels)
                    except Exception:
                        logger.debug(
                            "[AdaptiveScheduling] deadlock decision failed",
                            exc_info=True,
                        )

                    # -----------------------------
                    # Per-task RSS monitoring (optional)
                    # -----------------------------
                    if (
                        cfg.per_task_monitor_enabled
                        and self._get_company_pids is not None
                        and active_ids
                        and psutil is not None
                    ):
                        if (
                            now - self._last_per_task_check_ts
                        ) >= cfg.per_task_monitor_interval_sec:
                            self._last_per_task_check_ts = now
                            per_task_limit_bytes = int(cfg.per_task_rss_limit_mb * _MB)

                            sample_ids = list(active_ids)[
                                : cfg.per_task_monitor_max_companies
                            ]
                            for cid in sample_ids:
                                try:
                                    pids = list(self._get_company_pids(cid) or [])
                                except Exception:
                                    pids = []

                                if not pids:
                                    self._per_task_over_thresh_ts.pop(cid, None)
                                    self._per_task_last_rss_bytes.pop(cid, None)
                                    continue

                                rss_total = 0
                                for pid in pids:
                                    try:
                                        proc = psutil.Process(pid)  # type: ignore
                                    except Exception:
                                        continue

                                    try:
                                        rss_total += proc.memory_info().rss
                                    except Exception:
                                        pass
                                    try:
                                        for ch in proc.children(recursive=True):
                                            try:
                                                rss_total += ch.memory_info().rss
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                self._per_task_last_rss_bytes[cid] = int(rss_total)

                                if rss_total >= per_task_limit_bytes:
                                    first_ts = self._per_task_over_thresh_ts.get(cid)
                                    if first_ts is None:
                                        self._per_task_over_thresh_ts[cid] = now
                                    else:
                                        if now - first_ts >= float(
                                            cfg.per_task_rss_duration_sec
                                        ):
                                            logger.warning(
                                                "[AdaptiveScheduling] per-task RSS exceeded for %s: rss=%.1fMB limit=%.1fMB; canceling",
                                                cid,
                                                float(rss_total) / _MB,
                                                float(cfg.per_task_rss_limit_mb),
                                            )
                                            cancel_ids.append(cid)
                                            self._per_task_over_thresh_ts.pop(cid, None)
                                            self._per_task_last_rss_bytes.pop(cid, None)
                                else:
                                    self._per_task_over_thresh_ts.pop(cid, None)

                    # -----------------------------
                    # Pre-oom slope trigger
                    # -----------------------------
                    if (
                        used_frac >= cfg.preoom_used_frac
                        and slope >= cfg.preoom_slope_mb_s
                    ):
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
                                "[AdaptiveScheduling] pre-oom slope trigger: used_frac=%.3f slope=%.1fMB/s canceling=%d: %s",
                                used_frac,
                                slope,
                                len(selected),
                                selected,
                            )
                            cancel_ids.extend(selected)

                    # -----------------------------
                    # Emergency / near-oom cancellation
                    # -----------------------------
                    in_emergency = (
                        used_frac >= cfg.mem_crit_high_frac
                        or (
                            swap_frac >= cfg.swap_emergency_frac
                            and used_frac >= cfg.mem_crit_low_frac
                        )
                        or (
                            used_frac >= cfg.mem_cap_frac
                            and slope >= cfg.mem_trend_emergency_slope_mb_s
                        )
                        or (
                            used_frac >= cfg.mem_high_frac
                            and slope >= cfg.mem_trend_emergency_slope_mb_s
                        )
                    )

                    if not in_emergency and not cancel_ids:
                        continue

                    if (
                        now - self._last_emergency_cancel_ts
                    ) < cfg.emergency_cancel_cooldown_sec and not cancel_ids:
                        continue

                    if num_active <= 0:
                        self._restart_recommended = True
                        logger.error(
                            "[AdaptiveScheduling] emergency with zero active companies; restart recommended (used_frac=%.3f swap=%.3f).",
                            used_frac,
                            swap_frac,
                        )
                        continue

                    max_cancelable = max(0, num_active - int(cfg.min_active_keep))
                    if max_cancelable <= 0:
                        self._restart_recommended = True
                        logger.error(
                            "[AdaptiveScheduling] emergency but cannot cancel (num_active=%d min_keep=%d); restart recommended.",
                            num_active,
                            cfg.min_active_keep,
                        )
                        continue

                    # Dedup + cap cancel list
                    if cancel_ids:
                        seen = set()
                        uniq: List[str] = []
                        for cid in cancel_ids:
                            if cid in seen:
                                continue
                            seen.add(cid)
                            uniq.append(cid)
                        cancel_ids = uniq[:max_cancelable]
                    else:
                        effective_est_mb = self._effective_per_company_est_mb_locked()
                        per_company_bytes = max(
                            int(effective_est_mb * _MB),
                            int(cfg.per_company_min_reservation_mb * _MB),
                        )
                        mem_crit_low_bytes = int(cfg.mem_crit_low_frac * float(total))
                        excess_bytes = max(0, used - mem_crit_low_bytes)
                        needed = (
                            1
                            if per_company_bytes <= 0
                            else int(
                                (excess_bytes + per_company_bytes - 1)
                                // per_company_bytes
                            )
                        )
                        needed = max(1, needed)

                        super_critical = used_frac >= 0.98 or (
                            used_frac >= cfg.mem_cap_frac
                            and slope >= cfg.mem_trend_emergency_slope_mb_s
                        )
                        if super_critical:
                            to_cancel_count = min(
                                needed,
                                cfg.max_emergency_cancel_per_step,
                                max_cancelable,
                            )
                        else:
                            to_cancel_count = min(1, max_cancelable)

                        cancel_ids = self._select_heaviest_companies_locked(
                            active_ids, count=to_cancel_count
                        )
                        if not cancel_ids:
                            cancel_ids = list(active_ids)[-to_cancel_count:]

                    self._last_emergency_cancel_ts = now
                    self._emergency_rounds += 1

                    logger.error(
                        "[AdaptiveScheduling] emergency: used_frac=%.3f swap=%.3f total_mb=%.1f used_mb=%.1f active=%d est_mb=%.1f cancel=%s slope=%.1fMB/s",
                        used_frac,
                        swap_frac,
                        float(total) / _MB,
                        float(used) / _MB,
                        num_active,
                        self._per_company_est_mb,
                        cancel_ids,
                        slope,
                    )

                    if (
                        self._emergency_rounds
                        >= cfg.emergency_persistent_rounds_threshold
                        and used_frac >= cfg.mem_crit_low_frac
                        and not self._restart_recommended
                    ):
                        self._restart_recommended = True
                        logger.error(
                            "[AdaptiveScheduling] repeated emergency rounds (%d) with used_frac=%.3f; restart recommended.",
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

                    try:
                        await asyncio.sleep(cfg.emergency_post_cancel_delay_sec)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.debug(
                            "[AdaptiveScheduling] error during post-cancel sleep",
                            exc_info=True,
                        )

                    try:
                        gc.collect()
                    except Exception:
                        logger.debug(
                            "[AdaptiveScheduling] gc.collect failed", exc_info=True
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[AdaptiveScheduling] error in watchdog loop")


__all__ = ["AdaptiveSchedulingConfig", "AdaptiveScheduler"]
