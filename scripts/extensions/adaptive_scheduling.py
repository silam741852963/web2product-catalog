from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# Config (SIMPLE)
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveSchedulingConfig:
    """
    Simple, effective scheduling:

    - mem_cap_frac: admission headroom cap (aim to stay <= this)
    - mem_high_frac: above this, stop admitting and reduce target
    - mem_crit_frac: above this, cancel some running tasks

    - per_company_min_mb/max_mb: clamps for memory estimate
    - per_company_safety_factor: admission uses est * factor

    - AIMD:
        - ai_step: increase target by this when safe
        - md_factor: multiply target by this when pressured

    - Watchdogs:
        - hard_watchdog_*: kills the process if heartbeat stops updating
        - no_progress_timeout_sec: sets restart_recommended if nothing changes
    """

    # Memory watermarks (fractions of effective limit)
    mem_cap_frac: float = 0.78
    mem_high_frac: float = 0.83
    mem_crit_frac: float = 0.90

    # Estimation
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
    emergency_cancel_max: int = 2
    emergency_cancel_cooldown_sec: float = 3.0

    # Restart gating (prevents “restart too early”)
    restart_gate_min_uptime_sec: float = 180.0
    restart_gate_min_completed: int = 3

    # “No progress” policy
    no_progress_timeout_sec: float = 1800.0  # 30 min
    kill_on_no_progress: bool = False

    # Hard watchdog thread (survives async starvation)
    hard_watchdog_enabled: bool = True
    hard_watchdog_interval_sec: float = 5.0
    hard_watchdog_startup_grace_sec: float = 180.0
    hard_watchdog_no_heartbeat_timeout_sec: float = 120.0
    hard_watchdog_kill_signal: int = signal.SIGTERM

    # Optional heartbeat file for external monitoring
    heartbeat_path: Optional[Path] = None

    # Logging (optional JSONL state snapshots)
    log_path: Optional[Path] = None

    # Prefer cgroup limits if available
    prefer_cgroup_limits: bool = True
    use_psutil: bool = True


# ---------------------------------------------------------------------------
# Scheduler (SIMPLE)
# ---------------------------------------------------------------------------
class AdaptiveScheduler:
    """
    Simple memory-based scheduler + hard heartbeat watchdog.

    Public API preserved:
      - start/stop
      - admissible_slots(num_waiting)
      - register_company_completed()
      - record_company_peak(company_id, peak_mb)
      - restart_recommended
      - get_state_snapshot()
    """

    def __init__(
        self,
        cfg: AdaptiveSchedulingConfig,
        get_active_company_ids: Callable[[], Sequence[str]],
        request_cancel_companies: Callable[[Sequence[str]], None],
    ) -> None:
        self.cfg = cfg
        self._psutil_available = bool(PSUTIL_AVAILABLE and cfg.use_psutil)

        self._get_active_company_ids = get_active_company_ids
        self._request_cancel_companies = request_cancel_companies

        # Memory sampling cache
        self._total_mem_bytes: int = 0
        self._used_bytes: int = 0
        self._used_frac: float = 0.0
        self._last_sample_ts: float = 0.0

        # Estimator
        self._per_company_est_mb: float = cfg.per_company_min_mb
        self._base_rss_bytes: float = 0.0
        self._base_rss_samples: int = 0

        # Optional per-company peak tracking (for smarter cancels)
        self._company_peak_mb: Dict[str, float] = {}

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

    # -----------------------
    # Public properties
    # -----------------------
    @property
    def psutil_available(self) -> bool:
        return self._psutil_available

    @property
    def restart_recommended(self) -> bool:
        return self._restart_recommended

    # -----------------------
    # Public methods
    # -----------------------
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

        # Prime memory cache
        total, used = self._read_memory_usage_bytes()
        self._total_mem_bytes = total
        self._used_bytes = used
        self._used_frac = (float(used) / float(total)) if total > 0 else 0.0
        self._last_sample_ts = time.time()

        # Start async watchdog (emergency + no-progress)
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(),
            name="adaptive-scheduling-watchdog",
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
        self._mark_run_started()
        self._mark_progress()

    def record_company_peak(self, company_id: str, peak_mb: float) -> None:
        if not company_id or peak_mb <= 0:
            return
        prev = self._company_peak_mb.get(company_id)
        if prev is None or peak_mb > prev:
            self._company_peak_mb[company_id] = peak_mb

    async def admissible_slots(self, num_waiting: int) -> int:
        """
        Decide how many new companies may start *now*.

        Simple policy:
          slots = min(
              waiting,
              max(0, target_parallel - active),
              floor((mem_cap - used) / (per_company_est*safety))
          )
        """
        self._last_num_waiting = max(0, int(num_waiting))
        self._mark_heartbeat()

        if num_waiting <= 0:
            return 0

        # No psutil -> conservative but functional
        if not self._psutil_available:
            self._mark_run_started()
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
                self._mark_run_started()

            # Update base RSS if idle, and estimator if active
            self._update_base_rss_locked()
            self._update_per_company_est_locked(active_n)

            # AIMD target update (memory pressure only)
            self._update_target_parallel_locked(active_n)

            # If already high memory, do not admit
            if self._used_frac >= self.cfg.mem_high_frac:
                self._maybe_log_state_locked(
                    "block_mem_high", extra={"active": active_n}
                )
                return 0

            total = self._total_mem_bytes
            used = self._used_bytes
            if total <= 0:
                return 0

            cap_bytes = int(self.cfg.mem_cap_frac * float(total))
            headroom = cap_bytes - used

            # If headroom is negative but idle, still allow 1 to avoid deadlocks caused by stale est.
            if headroom <= 0:
                if active_n == 0 and self._used_frac < self.cfg.mem_high_frac:
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
                    and self._used_frac < self.cfg.mem_high_frac
                ):
                    slots = 1

            if slots > 0:
                self._ever_admitted = True
                self._mark_run_started()
                self._mark_progress()
                self._maybe_log_state_locked(
                    "admission",
                    extra={
                        "slots": slots,
                        "waiting": int(num_waiting),
                        "active": active_n,
                        "target_parallel": self._target_parallel,
                        "used_frac": self._used_frac,
                        "per_company_est_mb": self._per_company_est_mb,
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
    # Internals: heartbeat + progress
    # ------------------------------------------------------------------
    def _mark_run_started(self) -> None:
        # Run considered started once we’ve tried to admit, seen active, or completed.
        # (Kept explicit to avoid future regressions.)
        # This is used for restart gating.
        return

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
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": time.time(),
                "mono": time.monotonic(),
                "completed": self._completed_counter,
                "waiting": self._last_num_waiting,
                "ever_admitted": self._ever_admitted,
                "ever_had_active": self._ever_had_active,
                "restart_recommended": self._restart_recommended,
                "used_frac": self._used_frac,
                "target_parallel": self._target_parallel,
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            return

    # ------------------------------------------------------------------
    # Internals: memory reading
    # ------------------------------------------------------------------
    def _read_cgroup_memory_bytes(self) -> Tuple[int, int]:
        """
        Prefer cgroup v2 if present:
          /sys/fs/cgroup/memory.max
          /sys/fs/cgroup/memory.current
        Fall back to common cgroup v1 files if available.
        """
        try:
            base = Path("/sys/fs/cgroup")

            # cgroup v2
            max_path = base / "memory.max"
            cur_path = base / "memory.current"
            if max_path.exists() and cur_path.exists():
                max_raw = max_path.read_text(encoding="utf-8").strip()
                cur_raw = cur_path.read_text(encoding="utf-8").strip()
                used = int(cur_raw) if cur_raw else 0
                if not max_raw or max_raw == "max":
                    return 0, used
                limit = int(max_raw)
                return limit, used

            # cgroup v1 (common)
            max_path = base / "memory" / "memory.limit_in_bytes"
            cur_path = base / "memory" / "memory.usage_in_bytes"
            if max_path.exists() and cur_path.exists():
                limit = int(max_path.read_text(encoding="utf-8").strip() or "0")
                used = int(cur_path.read_text(encoding="utf-8").strip() or "0")
                # Some systems report absurdly large limit; treat as “no limit”
                if limit > 0 and limit < (1 << 60):
                    return limit, used
                return 0, used

        except Exception:
            pass
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
            return 0, 0

    def _sample_memory_locked(self, now: float) -> None:
        total, used = self._read_memory_usage_bytes()
        if total > 0:
            self._total_mem_bytes = total
            self._used_bytes = used
            self._used_frac = float(used) / float(total)
            self._last_sample_ts = now

    # ------------------------------------------------------------------
    # Internals: estimators + AIMD
    # ------------------------------------------------------------------
    def _update_base_rss_locked(self) -> None:
        # Track baseline “idle” RSS when no work is active
        active_n = len(self._get_active_company_ids())
        if active_n != 0 or self._used_bytes <= 0:
            return
        alpha = 0.1
        if self._base_rss_samples == 0:
            self._base_rss_bytes = float(self._used_bytes)
            self._base_rss_samples = 1
        else:
            self._base_rss_bytes = (1.0 - alpha) * self._base_rss_bytes + alpha * float(
                self._used_bytes
            )
            self._base_rss_samples += 1

    def _update_per_company_est_locked(self, active_n: int) -> None:
        if active_n <= 0 or self._used_bytes <= 0:
            return

        effective_used = float(self._used_bytes) - float(self._base_rss_bytes)
        if effective_used < 0:
            effective_used = 0.0
        per_company_now_mb = (effective_used / float(active_n)) / _MB
        if per_company_now_mb <= 0:
            return

        # EMA that reacts faster upward than downward (prevents underestimation)
        old = self._per_company_est_mb
        if per_company_now_mb > old:
            new = 0.7 * old + 0.3 * per_company_now_mb
        else:
            new = 0.9 * old + 0.1 * per_company_now_mb

        # Clamp
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

        if self._used_frac >= cfg.mem_high_frac:
            # multiplicative decrease
            new = int(
                max(cfg.min_target, max(1, int(float(old) * float(cfg.md_factor))))
            )
        elif self._used_frac <= (cfg.mem_cap_frac - 0.03):
            # additive increase
            new = min(cfg.max_target, old + int(cfg.ai_step))
        else:
            new = old

        # Don’t go below current active (helps avoid weird oscillation)
        if active_n > 0:
            new = max(new, active_n)

        self._target_parallel = int(max(cfg.min_target, min(new, cfg.max_target)))

        if self._target_parallel != old and logger.isEnabledFor(logging.INFO):
            logger.info(
                "[AdaptiveScheduling] target_parallel %d -> %d (used_frac=%.3f active=%d)",
                old,
                self._target_parallel,
                self._used_frac,
                active_n,
            )

    # ------------------------------------------------------------------
    # Internals: cancellation selection
    # ------------------------------------------------------------------
    def _select_cancel_ids(self, active_ids: Sequence[str], count: int) -> List[str]:
        if count <= 0 or not active_ids:
            return []
        # Prefer heaviest by observed peak
        weighted: List[Tuple[str, float]] = []
        for cid in active_ids:
            mb = self._company_peak_mb.get(cid)
            if mb is not None:
                weighted.append((cid, mb))
        if weighted:
            weighted.sort(key=lambda kv: kv[1], reverse=True)
            return [cid for (cid, _) in weighted[:count]]
        # Fallback: cancel last ones
        return list(active_ids)[-count:]

    # ------------------------------------------------------------------
    # Internals: restart gating
    # ------------------------------------------------------------------
    def _can_recommend_restart(self) -> bool:
        uptime = max(0.0, time.time() - self._started_ts)
        if uptime < float(self.cfg.restart_gate_min_uptime_sec):
            return False
        if self._completed_counter < int(self.cfg.restart_gate_min_completed):
            return False
        # Must have actually started doing work
        if not (
            self._ever_admitted or self._ever_had_active or self._completed_counter > 0
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # Async watchdog: emergency cancel + no-progress
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
                async with self._lock:
                    now = time.time()
                    self._sample_memory_locked(now)

                    active_ids = self._get_active_company_ids()
                    active_n = len(active_ids)
                    if active_n > 0:
                        self._ever_had_active = True

                    self._update_base_rss_locked()
                    self._update_per_company_est_locked(active_n)

                    # Progress inference
                    if (
                        (active_n != self._last_obs_active_n)
                        or (self._last_num_waiting != self._last_obs_waiting_n)
                        or (self._completed_counter != self._last_obs_completed)
                    ):
                        self._mark_progress()
                        self._last_obs_active_n = active_n
                        self._last_obs_waiting_n = self._last_num_waiting
                        self._last_obs_completed = self._completed_counter

                    # No-progress -> recommend restart (do not kill by default)
                    if self.cfg.no_progress_timeout_sec > 0:
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

                    # Emergency cancellation (critical memory)
                    if (
                        self._used_frac >= float(self.cfg.mem_crit_frac)
                        and active_n > 0
                    ):
                        if (now - self._last_emergency_cancel_ts) >= float(
                            self.cfg.emergency_cancel_cooldown_sec
                        ):
                            max_cancelable = max(
                                0, active_n - int(self.cfg.min_active_keep)
                            )
                            to_cancel = min(
                                int(self.cfg.emergency_cancel_max), max_cancelable
                            )
                            if to_cancel > 0:
                                cancel_ids = self._select_cancel_ids(
                                    active_ids, to_cancel
                                )
                                self._last_emergency_cancel_ts = now

                                logger.error(
                                    "[AdaptiveScheduling] EMERGENCY used_frac=%.3f total_mb=%.1f used_mb=%.1f active=%d cancel=%s",
                                    self._used_frac,
                                    float(self._total_mem_bytes) / _MB,
                                    float(self._used_bytes) / _MB,
                                    active_n,
                                    cancel_ids,
                                )
                                self._maybe_log_state_locked(
                                    "emergency_cancel",
                                    extra={
                                        "active": active_n,
                                        "cancel_ids": cancel_ids,
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

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[AdaptiveScheduling] watchdog loop error")

    # ------------------------------------------------------------------
    # Hard watchdog: kill on no heartbeat
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
            "used_bytes": self._used_bytes,
            "used_frac": self._used_frac,
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
