from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# psutil optional
try:
    import psutil  # type: ignore

    PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

_MB = 1_000_000


@dataclass
class AdaptiveSchedulingConfig:
    """
    Simple, effective scheduling:
    - Uses *effective* cgroup memory when available (cgroup v2: current - inactive_file).
      This prevents false "RAM stays high forever" stalls caused by page cache accounting.

    - ALSO tracks *raw* cgroup usage (memory.current / usage_in_bytes), because cgroup OOM
      triggers based on raw usage (page cache included). Raw is what you must avoid hitting.

    - Adds "memory ramp" protection: if memory fraction increases too fast (% / sec),
      throttle admission, and preemptively cancel tasks before hitting OOM.

    - Adds per-company "no activity" cancel: if an active company produces no activity
      (no page results processed / reported) for <= 5 minutes, cancel it.
    """

    # Effective memory watermarks (fractions of effective limit)
    mem_cap_frac: float = 0.78
    mem_high_frac: float = 0.83
    mem_crit_frac: float = 0.90

    # Raw (non-subtracted) cgroup usage fractions (what OOM cares about)
    mem_high_raw_frac: float = 0.88
    mem_crit_raw_frac: float = 0.92
    mem_kill_raw_frac: float = (
        0.96  # if exceeded -> recommend restart (+ optional SIGTERM)
    )

    # Memory ramp detection (fractions per second)
    mem_ramp_window_sec: float = 3.0
    mem_ramp_high_frac_per_sec: float = 0.020
    mem_ramp_crit_frac_per_sec: float = 0.030
    mem_ramp_kill_frac_per_sec: float = 0.050

    # Estimation (based on process-tree RSS by default)
    per_company_min_mb: float = 256.0
    per_company_max_mb: float = 1024.0
    per_company_safety_factor: float = 1.30

    # Sampling
    sample_interval_sec: float = 0.50

    # AIMD target concurrency
    min_target: int = 1
    max_target: int = 512
    initial_target: int = 4
    ai_step: int = 1
    md_factor: float = 0.50

    # Emergency cancellation
    min_active_keep: int = 0
    emergency_cancel_max: int = 4
    emergency_cancel_cooldown_sec: float = 3.0

    # Restart gating
    restart_gate_min_uptime_sec: float = 180.0
    restart_gate_min_completed: int = 3

    # “No progress” policy (GLOBAL). MUST NOT be > 5 minutes if enabled.
    no_progress_timeout_sec: float = 300.0
    kill_on_no_progress: bool = True

    # Per-company no activity (cancel stalled companies). MUST NOT be > 5 minutes if enabled.
    company_no_activity_timeout_sec: float = 300.0
    company_no_activity_cancel_max: int = 2
    company_no_activity_cooldown_sec: float = 30.0
    company_no_activity_min_active_keep: int = 0

    # Hard watchdog thread
    hard_watchdog_enabled: bool = True
    hard_watchdog_interval_sec: float = 5.0
    hard_watchdog_startup_grace_sec: float = 180.0
    hard_watchdog_no_heartbeat_timeout_sec: float = 120.0
    hard_watchdog_kill_signal: int = signal.SIGTERM

    # Optional heartbeat file
    heartbeat_path: Optional[Path] = None
    heartbeat_write_interval_sec: float = 1.0  # rate-limit file writes

    # Logging (optional JSONL state snapshots)
    log_path: Optional[Path] = None

    # Prefer cgroup limits if available
    prefer_cgroup_limits: bool = True
    use_psutil: bool = True

    # Important: subtract reclaimable cache from cgroup usage (prevents stalls)
    cgroup_subtract_inactive_file: bool = True

    # Use process-tree RSS for estimator baseline/EMA (better than system used)
    use_process_tree_rss_for_estimate: bool = True


class AdaptiveScheduler:
    """
    Public API preserved:
      - start/stop
      - admissible_slots(num_waiting)
      - register_company_completed()
      - record_company_peak(company_id, peak_mb)
      - restart_recommended
      - get_state_snapshot()

    Added:
      - record_company_activity(company_id)
    """

    def __init__(
        self,
        cfg: AdaptiveSchedulingConfig,
        get_active_company_ids: Callable[[], Sequence[str]],
        request_cancel_companies: Callable[[Sequence[str]], None],
    ) -> None:
        self.cfg = cfg

        # Enforce "not longer than 5 minutes" when enabled.
        if (
            self.cfg.no_progress_timeout_sec
            and self.cfg.no_progress_timeout_sec > 300.0
        ):
            self.cfg.no_progress_timeout_sec = 300.0
        if (
            self.cfg.company_no_activity_timeout_sec
            and self.cfg.company_no_activity_timeout_sec > 300.0
        ):
            self.cfg.company_no_activity_timeout_sec = 300.0

        self._psutil_available = bool(PSUTIL_AVAILABLE and cfg.use_psutil)

        self._get_active_company_ids = get_active_company_ids
        self._request_cancel_companies = request_cancel_companies

        # Memory sampling cache (system/cgroup)
        self._total_mem_bytes: int = 0
        self._used_bytes: int = 0  # effective
        self._used_frac: float = 0.0
        self._used_bytes_raw: int = 0
        self._used_frac_raw: float = 0.0
        self._last_sample_ts: float = 0.0

        # Memory ramp (%/sec)
        self._mem_frac_hist: Deque[Tuple[float, float, float]] = deque(
            maxlen=256
        )  # (mono, eff_frac, raw_frac)
        self._ramp_eff_frac_per_sec: float = 0.0
        self._ramp_raw_frac_per_sec: float = 0.0

        # Process tree RSS (python + chromium children)
        self._proc_rss_bytes: int = 0

        # Estimator (MB)
        self._per_company_est_mb: float = cfg.per_company_min_mb
        self._base_rss_bytes: float = 0.0
        self._base_rss_samples: int = 0

        # Optional per-company peak tracking
        self._company_peak_mb: Dict[str, float] = {}

        # Per-company activity tracking (monotonic timestamps)
        self._company_last_activity_mono: Dict[str, float] = {}
        self._last_company_stall_cancel_mono: float = 0.0

        # AIMD target
        self._target_parallel: int = max(
            cfg.min_target, min(cfg.initial_target, cfg.max_target)
        )

        # Progress / gating
        self._completed_counter: int = 0
        self._last_num_waiting: int = 0
        self._ever_admitted: bool = False
        self._ever_had_active: bool = False
        self._started_ts: float = time.time()
        self._started_mono: float = time.monotonic()

        # Heartbeats (monotonic)
        self._last_heartbeat_mono: float = time.monotonic()
        self._last_progress_mono: float = time.monotonic()

        # Heartbeat file write rate-limit
        self._last_heartbeat_write_ts: float = 0.0

        # “Observed” to infer progress
        self._last_obs_active_n: int = 0
        self._last_obs_waiting_n: int = 0
        self._last_obs_completed: int = 0

        # Emergency bookkeeping
        self._restart_recommended: bool = False
        self._last_emergency_cancel_ts: float = 0.0

        # Async + thread watchdog
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self._hard_stop = threading.Event()
        self._hard_thread: Optional[threading.Thread] = None

        self._last_log_ts: float = 0.0

    @property
    def psutil_available(self) -> bool:
        return self._psutil_available

    @property
    def restart_recommended(self) -> bool:
        return self._restart_recommended

    async def start(self) -> None:
        self._started_ts = time.time()
        self._started_mono = time.monotonic()
        self._mark_heartbeat()
        self._mark_progress()

        if self.cfg.hard_watchdog_enabled:
            self._start_hard_watchdog_thread()

        if not self._psutil_available:
            logger.warning(
                "[AdaptiveScheduling] psutil disabled/unavailable -> conservative admission."
            )
            return

        total, used_eff, used_raw = self._read_memory_usage_bytes()
        self._total_mem_bytes = total
        self._used_bytes = used_eff
        self._used_frac = (float(used_eff) / float(total)) if total > 0 else 0.0
        self._used_bytes_raw = used_raw
        self._used_frac_raw = (float(used_raw) / float(total)) if total > 0 else 0.0

        self._proc_rss_bytes = self._read_process_tree_rss_bytes()
        self._last_sample_ts = time.time()

        now_mono = time.monotonic()
        self._update_ramp_locked(now_mono, self._used_frac, self._used_frac_raw)

        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name="adaptive-scheduling-watchdog"
        )

        logger.info(
            "[AdaptiveScheduling] started total_mem_mb=%.1f initial_target_parallel=%d",
            float(self._total_mem_bytes) / _MB if self._total_mem_bytes else -1.0,
            self._target_parallel,
        )

    async def stop(self) -> None:
        t = self._watchdog_task
        self._watchdog_task = None
        if t is not None:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("[AdaptiveScheduling] watchdog stop error", exc_info=True)

        self._stop_hard_watchdog_thread()

    def register_company_completed(self) -> None:
        self._completed_counter += 1
        self._mark_progress()

    def record_company_peak(self, company_id: str, peak_mb: float) -> None:
        if not company_id or peak_mb <= 0:
            return
        prev = self._company_peak_mb.get(company_id)
        if prev is None or peak_mb > prev:
            self._company_peak_mb[company_id] = peak_mb

    def record_company_activity(self, company_id: str) -> None:
        """
        Called by run.py on any meaningful per-company activity (page processed, etc.).
        This prevents long single-company crawls from looking like "no progress".
        """
        if not company_id:
            self._mark_progress()
            return
        now_mono = time.monotonic()
        self._company_last_activity_mono[company_id] = now_mono
        self._mark_progress()

    async def admissible_slots(self, num_waiting: int) -> int:
        self._last_num_waiting = max(0, int(num_waiting))
        self._mark_heartbeat()

        if num_waiting <= 0:
            return 0

        if not self._psutil_available:
            self._ever_admitted = True
            self._mark_progress()
            return 1

        async with self._lock:
            now = time.time()
            if (now - self._last_sample_ts) >= float(self.cfg.sample_interval_sec):
                self._sample_memory_locked(now)

            active_ids = self._get_active_company_ids()
            active_n = len(active_ids)
            if active_n > 0:
                self._ever_had_active = True

            # Ensure active companies have an activity baseline timestamp
            now_mono = time.monotonic()
            for cid in active_ids:
                self._company_last_activity_mono.setdefault(cid, now_mono)

            # baseline + estimator
            self._update_base_rss_locked(active_n)
            self._update_per_company_est_locked(active_n)

            # AIMD target update
            self._update_target_parallel_locked(active_n)

            if self._used_frac_raw >= float(self.cfg.mem_high_raw_frac):
                self._maybe_log_state_locked(
                    "block_mem_high_raw", extra={"active": active_n}
                )
                return 0

            if active_n > 0 and self._ramp_raw_frac_per_sec >= float(
                self.cfg.mem_ramp_high_frac_per_sec
            ):
                self._maybe_log_state_locked(
                    "block_mem_ramp_high",
                    extra={
                        "active": active_n,
                        "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
                        "used_frac_raw": self._used_frac_raw,
                    },
                )
                return 0

            if self._used_frac >= float(self.cfg.mem_high_frac):
                self._maybe_log_state_locked(
                    "block_mem_high", extra={"active": active_n}
                )
                return 0

            total = self._total_mem_bytes
            used_eff = self._used_bytes
            if total <= 0:
                return 0

            cap_bytes = int(self.cfg.mem_cap_frac * float(total))
            headroom = cap_bytes - used_eff

            if headroom <= 0:
                if active_n == 0 and self._used_frac < float(self.cfg.mem_high_frac):
                    slots = 1
                else:
                    self._maybe_log_state_locked(
                        "block_mem_cap",
                        extra={"active": active_n, "headroom_mb": headroom / _MB},
                    )
                    return 0
            else:
                per_company_bytes = self._per_company_reservation_bytes_locked()
                slots_by_mem = (
                    int(headroom // per_company_bytes) if per_company_bytes > 0 else 0
                )
                slots_by_target = max(0, self._target_parallel - active_n)
                slots = min(int(num_waiting), slots_by_mem, slots_by_target)

                if (
                    slots <= 0
                    and active_n == 0
                    and self._used_frac < float(self.cfg.mem_high_frac)
                ):
                    slots = 1

            if slots > 0:
                self._ever_admitted = True
                self._mark_progress()
                self._maybe_log_state_locked(
                    "admission",
                    extra={
                        "slots": slots,
                        "waiting": int(num_waiting),
                        "active": active_n,
                        "target_parallel": self._target_parallel,
                        "used_frac": self._used_frac,
                        "used_frac_raw": self._used_frac_raw,
                        "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
                        "per_company_est_mb": self._per_company_est_mb,
                        "proc_rss_mb": float(self._proc_rss_bytes) / _MB
                        if self._proc_rss_bytes
                        else 0.0,
                    },
                )

            return max(0, int(slots))

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "total_mem_mb": float(self._total_mem_bytes) / _MB
            if self._total_mem_bytes
            else 0.0,
            "used_mem_mb": float(self._used_bytes) / _MB if self._used_bytes else 0.0,
            "used_frac": self._used_frac,
            "used_raw_mb": float(self._used_bytes_raw) / _MB
            if self._used_bytes_raw
            else 0.0,
            "used_frac_raw": self._used_frac_raw,
            "ramp_eff_frac_per_sec": self._ramp_eff_frac_per_sec,
            "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
            "proc_rss_mb": float(self._proc_rss_bytes) / _MB
            if self._proc_rss_bytes
            else 0.0,
            "per_company_est_mb": self._per_company_est_mb,
            "target_parallel": self._target_parallel,
            "completed_counter": self._completed_counter,
            "waiting": self._last_num_waiting,
            "restart_recommended": self._restart_recommended,
            "psutil_available": self._psutil_available,
            "ever_admitted": self._ever_admitted,
            "ever_had_active": self._ever_had_active,
            "last_heartbeat_age_sec": max(
                0.0, time.monotonic() - self._last_heartbeat_mono
            ),
            "last_progress_age_sec": max(
                0.0, time.monotonic() - self._last_progress_mono
            ),
        }

    # ------------------------------------------------------------------
    # Heartbeat + progress
    # ------------------------------------------------------------------
    def _mark_heartbeat(self) -> None:
        self._last_heartbeat_mono = time.monotonic()
        self._maybe_write_heartbeat_file()

    def _mark_progress(self) -> None:
        self._last_progress_mono = time.monotonic()
        self._maybe_write_heartbeat_file()

    def _maybe_write_heartbeat_file(self) -> None:
        path = self.cfg.heartbeat_path
        if path is None:
            return

        now = time.time()
        interval = max(0.1, float(self.cfg.heartbeat_write_interval_sec))
        if (
            self._last_heartbeat_write_ts
            and (now - self._last_heartbeat_write_ts) < interval
        ):
            return

        try:
            self._last_heartbeat_write_ts = now
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": now,
                "mono": time.monotonic(),
                "completed": self._completed_counter,
                "waiting": self._last_num_waiting,
                "ever_admitted": self._ever_admitted,
                "ever_had_active": self._ever_had_active,
                "restart_recommended": self._restart_recommended,
                "used_frac": self._used_frac,
                "used_frac_raw": self._used_frac_raw,
                "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
                "target_parallel": self._target_parallel,
                "proc_rss_mb": float(self._proc_rss_bytes) / _MB
                if self._proc_rss_bytes
                else 0.0,
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            return

    # ------------------------------------------------------------------
    # Memory reading helpers
    # ------------------------------------------------------------------
    def _parse_kv_lines(self, text: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            k, v = parts
            try:
                out[k] = int(v)
            except Exception:
                continue
        return out

    def _read_cgroup_memory_bytes(self) -> Tuple[int, int, int]:
        """
        Returns (limit_bytes, used_effective_bytes, used_raw_bytes)
          - used_raw is memory.current/usage_in_bytes (what OOM cares about)
          - used_effective subtracts inactive_file (better for admission)
        """
        try:
            base = Path("/sys/fs/cgroup")

            # cgroup v2
            max_path = base / "memory.max"
            cur_path = base / "memory.current"
            stat_path = base / "memory.stat"
            if max_path.exists() and cur_path.exists():
                max_raw = max_path.read_text(encoding="utf-8").strip()
                cur_raw = cur_path.read_text(encoding="utf-8").strip()
                current = int(cur_raw) if cur_raw else 0

                inactive_file = 0
                if self.cfg.cgroup_subtract_inactive_file and stat_path.exists():
                    st = self._parse_kv_lines(stat_path.read_text(encoding="utf-8"))
                    inactive_file = int(st.get("inactive_file", 0))

                used_effective = max(0, current - inactive_file)
                used_raw_bytes = current

                if not max_raw or max_raw == "max":
                    return 0, used_effective, used_raw_bytes
                limit = int(max_raw)
                return limit, used_effective, used_raw_bytes

            # cgroup v1
            max_path = base / "memory" / "memory.limit_in_bytes"
            cur_path = base / "memory" / "memory.usage_in_bytes"
            stat_path = base / "memory" / "memory.stat"
            if max_path.exists() and cur_path.exists():
                limit = int(max_path.read_text(encoding="utf-8").strip() or "0")
                usage = int(cur_path.read_text(encoding="utf-8").strip() or "0")

                inactive_file = 0
                if self.cfg.cgroup_subtract_inactive_file and stat_path.exists():
                    st = self._parse_kv_lines(stat_path.read_text(encoding="utf-8"))
                    inactive_file = int(
                        st.get("total_inactive_file", st.get("inactive_file", 0))
                    )

                used_effective = max(0, usage - inactive_file)
                used_raw_bytes = usage

                if limit > 0 and limit < (1 << 60):
                    return limit, used_effective, used_raw_bytes
                return 0, used_effective, used_raw_bytes

        except Exception:
            pass

        return 0, 0, 0

    def _read_process_tree_rss_bytes(self) -> int:
        if not self._psutil_available or psutil is None:
            return 0
        try:
            p = psutil.Process(os.getpid())  # type: ignore[union-attr]
            rss = int(p.memory_info().rss)
            for ch in p.children(recursive=True):
                try:
                    rss += int(ch.memory_info().rss)
                except Exception:
                    continue
            return rss
        except Exception:
            return 0

    def _read_memory_usage_bytes(self) -> Tuple[int, int, int]:
        """
        Returns (total_bytes, used_effective_bytes, used_raw_bytes).
        If cgroup is available and limited, raw=memory.current, effective=current-inactive_file.
        If not, raw==effective==psutil_used.
        """
        if not self._psutil_available or psutil is None:
            return 0, 0, 0

        if self.cfg.prefer_cgroup_limits:
            limit, used_eff, used_raw = self._read_cgroup_memory_bytes()
            if limit > 0:
                return limit, used_eff, used_raw

        try:
            vm = psutil.virtual_memory()  # type: ignore[union-attr]
            total = int(vm.total)
            used = int(vm.total - vm.available)
            return total, used, used
        except Exception:
            return 0, 0, 0

    def _update_ramp_locked(
        self, now_mono: float, used_frac_eff: float, used_frac_raw: float
    ) -> None:
        cfg = self.cfg
        window = max(0.5, float(cfg.mem_ramp_window_sec))

        self._mem_frac_hist.append((now_mono, used_frac_eff, used_frac_raw))

        while self._mem_frac_hist and (now_mono - self._mem_frac_hist[0][0]) > window:
            self._mem_frac_hist.popleft()

        if len(self._mem_frac_hist) < 2:
            self._ramp_eff_frac_per_sec = 0.0
            self._ramp_raw_frac_per_sec = 0.0
            return

        t0, eff0, raw0 = self._mem_frac_hist[0]
        t1, eff1, raw1 = self._mem_frac_hist[-1]
        dt = float(t1 - t0)
        if dt <= 1e-3:
            self._ramp_eff_frac_per_sec = 0.0
            self._ramp_raw_frac_per_sec = 0.0
            return

        self._ramp_eff_frac_per_sec = float(eff1 - eff0) / dt
        self._ramp_raw_frac_per_sec = float(raw1 - raw0) / dt

    def _sample_memory_locked(self, now: float) -> None:
        total, used_eff, used_raw = self._read_memory_usage_bytes()
        if total > 0:
            self._total_mem_bytes = total
            self._used_bytes = used_eff
            self._used_frac = float(used_eff) / float(total)

            self._used_bytes_raw = used_raw
            self._used_frac_raw = float(used_raw) / float(total)

            self._last_sample_ts = now

            now_mono = time.monotonic()
            self._update_ramp_locked(now_mono, self._used_frac, self._used_frac_raw)

        self._proc_rss_bytes = self._read_process_tree_rss_bytes()

    # ------------------------------------------------------------------
    # Estimators + AIMD
    # ------------------------------------------------------------------
    def _update_base_rss_locked(self, active_n: int) -> None:
        if active_n != 0:
            return

        base_now = (
            float(self._proc_rss_bytes)
            if (self.cfg.use_process_tree_rss_for_estimate and self._proc_rss_bytes > 0)
            else float(self._used_bytes)
        )
        if base_now <= 0:
            return

        alpha = 0.1
        if self._base_rss_samples == 0:
            self._base_rss_bytes = base_now
            self._base_rss_samples = 1
        else:
            self._base_rss_bytes = (
                1.0 - alpha
            ) * self._base_rss_bytes + alpha * base_now
            self._base_rss_samples += 1

    def _update_per_company_est_locked(self, active_n: int) -> None:
        if active_n <= 0:
            return

        used_for_est = (
            float(self._proc_rss_bytes)
            if (self.cfg.use_process_tree_rss_for_estimate and self._proc_rss_bytes > 0)
            else float(self._used_bytes)
        )
        if used_for_est <= 0:
            return

        effective_used = used_for_est - float(self._base_rss_bytes)
        if effective_used < 0:
            effective_used = 0.0

        per_company_now_mb = (effective_used / float(active_n)) / _MB
        if per_company_now_mb <= 0:
            return

        old = self._per_company_est_mb
        if per_company_now_mb > old:
            new = 0.7 * old + 0.3 * per_company_now_mb
        else:
            new = 0.9 * old + 0.1 * per_company_now_mb

        new = max(
            self.cfg.per_company_min_mb, min(float(new), self.cfg.per_company_max_mb)
        )
        self._per_company_est_mb = float(new)

    def _per_company_reservation_bytes_locked(self) -> int:
        mb = float(self._per_company_est_mb) * float(self.cfg.per_company_safety_factor)
        mb = max(self.cfg.per_company_min_mb, min(mb, self.cfg.per_company_max_mb))
        return int(mb * _MB)

    def _update_target_parallel_locked(self, active_n: int) -> None:
        cfg = self.cfg
        old = self._target_parallel

        if self._used_frac_raw >= float(
            cfg.mem_high_raw_frac
        ) or self._used_frac >= float(cfg.mem_high_frac):
            new = int(
                max(cfg.min_target, max(1, int(float(old) * float(cfg.md_factor))))
            )
        elif self._used_frac <= (cfg.mem_cap_frac - 0.03):
            new = min(cfg.max_target, old + int(cfg.ai_step))
        else:
            new = old

        if active_n > 0:
            new = max(new, active_n)

        self._target_parallel = int(max(cfg.min_target, min(new, cfg.max_target)))

        if self._target_parallel != old and logger.isEnabledFor(logging.INFO):
            logger.info(
                "[AdaptiveScheduling] target_parallel %d -> %d (used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s active=%d proc_rss_mb=%.1f)",
                old,
                self._target_parallel,
                self._used_frac_raw,
                self._used_frac,
                self._ramp_raw_frac_per_sec,
                active_n,
                float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0,
            )

    # ------------------------------------------------------------------
    # Cancellation selection
    # ------------------------------------------------------------------
    def _select_cancel_ids(self, active_ids: Sequence[str], count: int) -> List[str]:
        if count <= 0 or not active_ids:
            return []
        weighted: List[Tuple[str, float]] = []
        for cid in active_ids:
            mb = self._company_peak_mb.get(cid)
            if mb is not None:
                weighted.append((cid, mb))
        if weighted:
            weighted.sort(key=lambda kv: kv[1], reverse=True)
            return [cid for (cid, _) in weighted[:count]]
        return list(active_ids)[-count:]

    def _select_stalled_ids(self, active_ids: Sequence[str], count: int) -> List[str]:
        if count <= 0 or not active_ids:
            return []
        now_mono = time.monotonic()
        scored: List[Tuple[str, float]] = []
        for cid in active_ids:
            last = self._company_last_activity_mono.get(cid, now_mono)
            age = max(0.0, now_mono - last)
            scored.append((cid, age))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [cid for (cid, _) in scored[:count]]

    # ------------------------------------------------------------------
    # Restart gating
    # ------------------------------------------------------------------
    def _can_recommend_restart(self) -> bool:
        uptime = max(0.0, time.time() - self._started_ts)
        if uptime < float(self.cfg.restart_gate_min_uptime_sec):
            return False
        if self._completed_counter < int(self.cfg.restart_gate_min_completed):
            return False
        if not (
            self._ever_admitted or self._ever_had_active or self._completed_counter > 0
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # Async watchdog
    # ------------------------------------------------------------------
    async def _watchdog_loop(self) -> None:
        interval = max(0.5, float(self.cfg.sample_interval_sec))
        while True:
            try:
                await asyncio.sleep(interval)
                self._mark_heartbeat()

                if not self._psutil_available:
                    continue

                cancel_ids: List[str] = []
                cancel_reason: Optional[str] = None

                async with self._lock:
                    now = time.time()
                    self._sample_memory_locked(now)

                    active_ids = list(self._get_active_company_ids())
                    active_n = len(active_ids)
                    if active_n > 0:
                        self._ever_had_active = True

                    now_mono = time.monotonic()
                    for cid in active_ids:
                        self._company_last_activity_mono.setdefault(cid, now_mono)

                    # Drop activity records for inactive companies (keep dict bounded)
                    if self._company_last_activity_mono and active_n == 0:
                        self._company_last_activity_mono.clear()
                    elif self._company_last_activity_mono and active_ids:
                        active_set = set(active_ids)
                        for cid in list(self._company_last_activity_mono.keys()):
                            if cid not in active_set:
                                self._company_last_activity_mono.pop(cid, None)

                    self._update_base_rss_locked(active_n)
                    self._update_per_company_est_locked(active_n)

                    # Progress inference (coarse)
                    if (
                        (active_n != self._last_obs_active_n)
                        or (self._last_num_waiting != self._last_obs_waiting_n)
                        or (self._completed_counter != self._last_obs_completed)
                    ):
                        self._mark_progress()
                        self._last_obs_active_n = active_n
                        self._last_obs_waiting_n = self._last_num_waiting
                        self._last_obs_completed = self._completed_counter

                    # GLOBAL no-progress -> restart recommended
                    if (
                        self.cfg.no_progress_timeout_sec
                        and self.cfg.no_progress_timeout_sec > 0
                    ):
                        prog_age = time.monotonic() - self._last_progress_mono
                        if (
                            prog_age >= float(self.cfg.no_progress_timeout_sec)
                            and self._last_num_waiting > 0
                        ):
                            if self._can_recommend_restart():
                                self._restart_recommended = True
                                logger.error(
                                    "[AdaptiveScheduling] no progress for %.1fs with waiting=%d -> restart recommended",
                                    prog_age,
                                    self._last_num_waiting,
                                )
                                if self.cfg.kill_on_no_progress:
                                    os.kill(
                                        os.getpid(),
                                        int(self.cfg.hard_watchdog_kill_signal),
                                    )

                    # Per-company stall cancel (no activity)
                    if (
                        self.cfg.company_no_activity_timeout_sec
                        and self.cfg.company_no_activity_timeout_sec > 0
                        and active_n > 0
                    ):
                        stall_timeout = float(self.cfg.company_no_activity_timeout_sec)
                        ages = [
                            (
                                cid,
                                max(
                                    0.0,
                                    now_mono
                                    - self._company_last_activity_mono.get(
                                        cid, now_mono
                                    ),
                                ),
                            )
                            for cid in active_ids
                        ]
                        stalled = [cid for (cid, age) in ages if age >= stall_timeout]
                        if stalled:
                            cooldown_ok = (
                                now_mono - self._last_company_stall_cancel_mono
                            ) >= float(self.cfg.company_no_activity_cooldown_sec)
                            if cooldown_ok:
                                max_cancelable = max(
                                    0,
                                    active_n
                                    - int(self.cfg.company_no_activity_min_active_keep),
                                )
                                to_cancel = min(
                                    int(self.cfg.company_no_activity_cancel_max),
                                    len(stalled),
                                    max_cancelable,
                                )
                                if to_cancel > 0:
                                    cancel_ids = self._select_stalled_ids(
                                        stalled, to_cancel
                                    )
                                    cancel_reason = "company_no_activity"
                                    self._last_company_stall_cancel_mono = now_mono
                                    logger.error(
                                        "[AdaptiveScheduling] STALL cancel=%s (no_activity>=%.1fs) active=%d waiting=%d",
                                        cancel_ids,
                                        stall_timeout,
                                        active_n,
                                        self._last_num_waiting,
                                    )
                                    self._maybe_log_state_locked(
                                        "stall_cancel",
                                        extra={
                                            "active": active_n,
                                            "waiting": self._last_num_waiting,
                                            "cancel_ids": cancel_ids,
                                            "stall_timeout_sec": stall_timeout,
                                        },
                                    )

                    # Hard danger: raw usage is what OOM cares about
                    if self._used_frac_raw >= float(self.cfg.mem_kill_raw_frac):
                        if self._can_recommend_restart():
                            self._restart_recommended = True
                            logger.critical(
                                "[AdaptiveScheduling] raw mem %.3f >= kill %.3f -> restart recommended",
                                self._used_frac_raw,
                                float(self.cfg.mem_kill_raw_frac),
                            )
                            if self.cfg.kill_on_no_progress:
                                os.kill(
                                    os.getpid(), int(self.cfg.hard_watchdog_kill_signal)
                                )

                    # Ramp danger: if ramp is "kill-fast" while already mid/high, recommend restart early
                    if self._ramp_raw_frac_per_sec >= float(
                        self.cfg.mem_ramp_kill_frac_per_sec
                    ) and self._used_frac_raw >= float(self.cfg.mem_high_raw_frac):
                        if self._can_recommend_restart():
                            self._restart_recommended = True
                            logger.critical(
                                "[AdaptiveScheduling] mem ramp raw=%.3f/s with used_raw=%.3f (>=high %.3f) -> restart recommended",
                                self._ramp_raw_frac_per_sec,
                                self._used_frac_raw,
                                float(self.cfg.mem_high_raw_frac),
                            )
                            if self.cfg.kill_on_no_progress:
                                os.kill(
                                    os.getpid(), int(self.cfg.hard_watchdog_kill_signal)
                                )

                    # Emergency cancellation (memory)
                    ramp_trigger = self._ramp_raw_frac_per_sec >= float(
                        self.cfg.mem_ramp_crit_frac_per_sec
                    ) and self._used_frac_raw >= float(self.cfg.mem_cap_frac)
                    crit_trigger = self._used_frac_raw >= float(
                        self.cfg.mem_crit_raw_frac
                    ) or self._used_frac >= float(self.cfg.mem_crit_frac)

                    if (crit_trigger or ramp_trigger) and active_n > 0:
                        if (now - self._last_emergency_cancel_ts) >= float(
                            self.cfg.emergency_cancel_cooldown_sec
                        ):
                            max_cancelable = max(
                                0, active_n - int(self.cfg.min_active_keep)
                            )
                            desired = int(self.cfg.emergency_cancel_max)
                            to_cancel = min(desired, max_cancelable)
                            if to_cancel > 0:
                                cancel_ids = self._select_cancel_ids(
                                    active_ids, to_cancel
                                )
                                cancel_reason = "memory_emergency"
                                self._last_emergency_cancel_ts = now
                                logger.error(
                                    "[AdaptiveScheduling] EMERGENCY used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s total_mb=%.1f used_raw_mb=%.1f used_eff_mb=%.1f proc_rss_mb=%.1f active=%d cancel=%s",
                                    self._used_frac_raw,
                                    self._used_frac,
                                    self._ramp_raw_frac_per_sec,
                                    float(self._total_mem_bytes) / _MB,
                                    float(self._used_bytes_raw) / _MB,
                                    float(self._used_bytes) / _MB,
                                    float(self._proc_rss_bytes) / _MB
                                    if self._proc_rss_bytes
                                    else 0.0,
                                    active_n,
                                    cancel_ids,
                                )
                                self._maybe_log_state_locked(
                                    "emergency_cancel",
                                    extra={
                                        "active": active_n,
                                        "cancel_ids": cancel_ids,
                                        "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
                                        "used_frac_raw": self._used_frac_raw,
                                        "used_frac": self._used_frac,
                                    },
                                )

                if cancel_ids:
                    try:
                        self._request_cancel_companies(cancel_ids)
                        self._mark_progress()
                    except Exception:
                        logger.exception(
                            "[AdaptiveScheduling] request_cancel_companies failed"
                        )

                    try:
                        gc.collect()
                    except Exception:
                        pass

                    if cancel_reason:
                        logger.warning(
                            "[AdaptiveScheduling] cancel_reason=%s ids=%s",
                            cancel_reason,
                            cancel_ids,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[AdaptiveScheduling] watchdog loop error")

    # ------------------------------------------------------------------
    # Hard watchdog
    # ------------------------------------------------------------------
    def _start_hard_watchdog_thread(self) -> None:
        if self._hard_thread is not None and self._hard_thread.is_alive():
            return
        self._hard_stop.clear()
        t = threading.Thread(
            target=self._hard_watchdog_loop, name="adaptive-hard-watchdog", daemon=True
        )
        self._hard_thread = t
        t.start()

    def _stop_hard_watchdog_thread(self) -> None:
        self._hard_stop.set()
        t = self._hard_thread
        self._hard_thread = None
        if t is None:
            return
        try:
            t.join(timeout=1.0)
        except Exception:
            pass

    def _hard_watchdog_loop(self) -> None:
        cfg = self.cfg
        interval = max(1.0, float(cfg.hard_watchdog_interval_sec))
        grace = max(
            float(cfg.hard_watchdog_startup_grace_sec),
            float(cfg.restart_gate_min_uptime_sec),
        )

        while not self._hard_stop.is_set():
            try:
                time.sleep(interval)
                now_mono = time.monotonic()
                uptime = now_mono - self._started_mono
                if uptime < grace:
                    continue

                run_started = bool(
                    self._ever_admitted
                    or self._ever_had_active
                    or self._completed_counter > 0
                )
                if not run_started:
                    continue

                hb_age = now_mono - self._last_heartbeat_mono
                if hb_age >= float(cfg.hard_watchdog_no_heartbeat_timeout_sec):
                    logger.critical(
                        "[AdaptiveScheduling] HARD WATCHDOG: no heartbeat for %.1fs (timeout=%.1fs). Killing self.",
                        hb_age,
                        float(cfg.hard_watchdog_no_heartbeat_timeout_sec),
                    )
                    try:
                        os.kill(os.getpid(), int(cfg.hard_watchdog_kill_signal))
                    except Exception:
                        os._exit(2)  # noqa: S606
                    return
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _maybe_log_state_locked(
        self, reason: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.cfg.log_path is None:
            return
        now = time.time()
        if self._last_log_ts and (now - self._last_log_ts) < 5.0:
            return
        self._last_log_ts = now

        state: Dict[str, Any] = {
            "ts": now,
            "reason": reason,
            "total_mem_bytes": self._total_mem_bytes,
            "used_bytes_eff": self._used_bytes,
            "used_frac_eff": self._used_frac,
            "used_bytes_raw": self._used_bytes_raw,
            "used_frac_raw": self._used_frac_raw,
            "ramp_eff_frac_per_sec": self._ramp_eff_frac_per_sec,
            "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
            "proc_rss_bytes": self._proc_rss_bytes,
            "per_company_est_mb": self._per_company_est_mb,
            "target_parallel": self._target_parallel,
            "completed_counter": self._completed_counter,
            "waiting": self._last_num_waiting,
            "restart_recommended": self._restart_recommended,
            "ever_admitted": self._ever_admitted,
            "ever_had_active": self._ever_had_active,
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
            logger.debug("[AdaptiveScheduling] failed to write log", exc_info=True)


__all__ = ["AdaptiveSchedulingConfig", "AdaptiveScheduler"]
