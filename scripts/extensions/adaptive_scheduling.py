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

try:
    import psutil  # type: ignore

    PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

_MB = 1_000_000


@dataclass
class AdaptiveSchedulingConfig:
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
    peak_history_horizon_sec: float = 1800.0

    per_company_safety_factor: float = 1.3
    per_company_min_reservation_mb: float = 256.0
    per_company_max_reservation_mb: float = 1024.0

    # -----------------------------
    # Sampling / overhead controls
    # -----------------------------
    emergency_check_interval_sec: float = 1.0
    min_admission_sample_interval_sec: float = 0.35
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
    mem_trend_emergency_slope_mb_per_s: float = 40.0

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
    # Per-task RSS monitoring (optional)
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
    # NEW: Progress-stall detector
    # -----------------------------
    # If we have active companies but **no completions** for a while and memory
    # keeps rising (linear leak / spin), proactively cancel a few tasks to break
    # the stall. If it persists, recommend restart.
    progress_stall_window_sec: float = 180.0
    progress_stall_used_frac: float = 0.70
    progress_stall_slope_mb_s: float = 15.0
    progress_stall_min_active: int = 2
    progress_stall_cancel_count: int = 1
    progress_stall_cancel_cooldown_sec: float = 30.0
    progress_stall_max_rounds_before_restart: int = 3
    progress_stall_restart_multiplier: float = (
        2.0  # restart if idle_for >= window * multiplier
    )


class AdaptiveScheduler:
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

        self._peak_history_mb: collections.deque[Tuple[float, float]] = (
            collections.deque()
        )
        self._mem_samples: collections.deque[Tuple[float, int]] = collections.deque()

        self._per_company_est_mb: float = cfg.per_company_min_reservation_mb

        self._company_peak_mb: "collections.OrderedDict[str, float]" = (
            collections.OrderedDict()
        )
        self._company_p95_cache_mb: Optional[float] = None
        self._company_p95_cache_ts: float = 0.0

        self._emergency_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        self._restart_recommended: bool = False
        self._last_log_ts: float = 0.0

        self._last_used_bytes: int = 0
        self._last_used_frac: float = 0.0
        self._last_swap_used_frac: float = 0.0
        self._last_sample_ts: float = 0.0
        self._last_swap_sample_ts: float = 0.0

        self._last_trend_slope_mb_s: float = 0.0

        self._target_parallel: int = min(
            max(self.cfg.min_target, self.cfg.initial_target), self.cfg.max_target
        )
        self._update_calls: int = 0

        self._near_oom_events: int = 0
        self._last_near_oom_flag: bool = False

        self._last_low_mem_relax_ts: float = 0.0

        self._last_emergency_cancel_ts: float = 0.0
        self._emergency_rounds: int = 0

        self._base_rss_bytes: float = 0.0
        self._base_rss_samples: int = 0

        self._per_task_over_thresh_ts: Dict[str, float] = {}
        self._per_task_last_rss_bytes: Dict[str, int] = {}
        self._last_per_task_check_ts: float = 0.0

        self._completed_counter: int = 0
        self._last_num_waiting: int = 0

        # NEW: stall tracking
        self._last_progress_ts: float = time.time()
        self._stall_rounds: int = 0
        self._last_stall_cancel_ts: float = 0.0

    @property
    def psutil_available(self) -> bool:
        return self._psutil_available

    @property
    def restart_recommended(self) -> bool:
        return self._restart_recommended

    def register_company_completed(self) -> None:
        self._completed_counter += 1
        self._last_progress_ts = time.time()
        self._stall_rounds = 0  # reset stall counter on progress

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

        self._last_swap_used_frac = self._read_swap_used_frac()
        now = time.time()
        self._last_swap_sample_ts = now
        self._last_sample_ts = now

        self._last_progress_ts = now

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
        if num_waiting <= 0:
            return 0
        self._last_num_waiting = num_waiting

        if not self._psutil_available:
            return 1

        now = time.time()
        cfg = self.cfg

        async with self._lock:
            if (now - self._last_sample_ts) >= cfg.min_admission_sample_interval_sec:
                total, used = self._read_memory_usage_bytes()
                if total <= 0:
                    return 0

                self._total_mem_bytes = total
                self._last_used_bytes = used
                self._last_used_frac = float(used) / float(total)
                self._last_sample_ts = now

                self._record_memory_sample_locked(ts=now, used_bytes=used)

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

            self._update_base_rss_locked(used_bytes=used, num_active=num_active)

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

            self._update_per_company_estimate_locked(
                used_bytes=used,
                num_active=num_active,
                used_frac=used_frac,
            )
            effective_est_mb = self._effective_per_company_est_mb_locked()

            per_company_bytes = max(
                int(effective_est_mb * _MB),
                int(cfg.per_company_min_reservation_mb * _MB),
            )
            if per_company_bytes <= 0:
                return 0

            slots_by_mem = int(headroom_bytes // per_company_bytes)

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
            "last_progress_age_sec": max(0.0, time.time() - self._last_progress_ts),
            "stall_rounds": self._stall_rounds,
            "per_task_rss_mb_sample": per_task_rss_mb,
        }

    # ---------------- internal ----------------

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
        self, *, used_bytes: int, num_active: int, used_frac: float
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
        if num_active >= 2 and per_company_now_mb > 0.0:
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
        self, *, total: int, used: int, headroom_bytes: int, used_frac: float
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
        self, *, used_frac: float, precritical: bool, high_swap: bool, mem_trouble: bool
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
                and self._last_trend_slope_mb_s <= cfg.mem_trend_slope_high_mb_per_s
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
            "last_progress_age_sec": max(0.0, time.time() - self._last_progress_ts),
            "stall_rounds": self._stall_rounds,
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

    def _maybe_progress_stall_actions_locked(
        self,
        *,
        now: float,
        used_frac: float,
        swap_frac: float,
        slope: float,
        active_ids: Sequence[str],
    ) -> List[str]:
        """
        Detect the stall pattern you described:
          - RAM rising steadily (positive slope)
          - CPU high but no useful progress (we approximate by: no completions)
          - active companies exist
        Action:
          - cancel a small number of companies to break the stall
          - if repeated, recommend restart
        """
        cfg = self.cfg
        num_active = len(active_ids)
        if num_active < int(cfg.progress_stall_min_active):
            return []

        idle_for = now - self._last_progress_ts
        if idle_for < float(cfg.progress_stall_window_sec):
            return []

        if used_frac < float(cfg.progress_stall_used_frac):
            return []

        if slope < float(cfg.progress_stall_slope_mb_s):
            return []

        if (now - self._last_stall_cancel_ts) < float(
            cfg.progress_stall_cancel_cooldown_sec
        ):
            return []

        max_cancelable = max(0, num_active - int(cfg.min_active_keep))
        if max_cancelable <= 0:
            # cannot cancel: recommend restart if the stall is strong
            self._restart_recommended = True
            self._maybe_log_state_locked(
                reason="progress_stall_restart_nocancel",
                extra={
                    "idle_for_sec": idle_for,
                    "used_frac": used_frac,
                    "swap_used_frac": swap_frac,
                    "trend_slope_mb_s": slope,
                    "num_active": num_active,
                },
            )
            return []

        to_cancel = min(max_cancelable, max(1, int(cfg.progress_stall_cancel_count)))
        cancel_ids = self._select_heaviest_companies_locked(active_ids, count=to_cancel)
        if not cancel_ids:
            cancel_ids = list(active_ids)[-to_cancel:]

        self._last_stall_cancel_ts = now
        self._stall_rounds += 1

        logger.error(
            "[AdaptiveScheduling] progress-stall detected: idle_for=%.1fs used_frac=%.3f slope=%.1fMB/s active=%d -> cancel=%s (stall_rounds=%d)",
            idle_for,
            used_frac,
            slope,
            num_active,
            cancel_ids,
            self._stall_rounds,
        )

        self._maybe_log_state_locked(
            reason="progress_stall_cancel",
            extra={
                "idle_for_sec": idle_for,
                "used_frac": used_frac,
                "swap_used_frac": swap_frac,
                "trend_slope_mb_s": slope,
                "num_active": num_active,
                "cancel_ids": cancel_ids,
                "stall_rounds": self._stall_rounds,
            },
        )

        # Escalate to restart if it persists
        if self._stall_rounds >= int(cfg.progress_stall_max_rounds_before_restart):
            if idle_for >= float(cfg.progress_stall_window_sec) * float(
                cfg.progress_stall_restart_multiplier
            ):
                self._restart_recommended = True
                logger.error(
                    "[AdaptiveScheduling] progress-stall persists (%d rounds, idle_for=%.1fs); restart recommended.",
                    self._stall_rounds,
                    idle_for,
                )
                self._maybe_log_state_locked(
                    reason="progress_stall_restart",
                    extra={
                        "idle_for_sec": idle_for,
                        "used_frac": used_frac,
                        "swap_used_frac": swap_frac,
                        "trend_slope_mb_s": slope,
                        "num_active": num_active,
                        "stall_rounds": self._stall_rounds,
                    },
                )

        return cancel_ids

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

                    # --- NEW: progress-stall actions (runs even if not "near-oom") ---
                    slope = self._last_trend_slope_mb_s
                    stall_cancels = self._maybe_progress_stall_actions_locked(
                        now=now,
                        used_frac=used_frac,
                        swap_frac=swap_frac,
                        slope=slope,
                        active_ids=active_ids,
                    )
                    cancel_ids.extend(stall_cancels)

                    # Per-task RSS monitoring
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

                    # Pre-oom slope trigger
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

                    in_emergency = (
                        used_frac >= cfg.mem_crit_high_frac
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
                        # preserve order, dedup
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
                            and slope >= cfg.mem_trend_emergency_slope_mb_per_s
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
