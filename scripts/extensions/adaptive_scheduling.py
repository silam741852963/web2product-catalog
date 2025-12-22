from __future__ import annotations

import asyncio
import gc
import heapq
import inspect
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .retry_state import RetryStateStore

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


def compute_retry_exit_code_from_store(
    store: RetryStateStore, retry_exit_code: int
) -> int:
    """
    Return retry_exit_code only if there exists at least one *eligible now* retry-pending company.
    Prevents tight restart loops when next_eligible_at is in the future.
    """
    try:
        now = time.time()
        ids = store.pending_ids(exclude_quarantined=True)
        if not ids:
            return 0
        mask = store.eligible_mask(ids, now=now)
        eligible_any = any(mask.values())
        return int(retry_exit_code) if eligible_any else 0
    except Exception:
        return 0


def _mark_failure_compat(
    store: RetryStateStore, company_id: str, **kwargs: Any
) -> None:
    """
    Call RetryStateStore.mark_failure using only parameters supported by the
    installed retry_state.py signature.
    """
    try:
        sig = inspect.signature(store.mark_failure)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        store.mark_failure(company_id, **filtered)  # type: ignore[arg-type]
    except Exception:
        return


@dataclass
class AdaptiveSchedulingConfig:
    """
    Scheduler owns:
      - admission control (memory + AIMD)
      - cancellation policy (critical memory OR ramp OR sustained high)
      - restart recommendation
      - retry/quarantine/backoff filtering
      - worklist queueing (ready + deferred)
    """

    # --- Memory thresholds (effective / raw) ---
    mem_cap_frac: float = 0.78
    mem_high_frac: float = 0.84
    mem_crit_frac: float = 0.90

    mem_high_raw_frac: float = 0.88
    mem_crit_raw_frac: float = 0.92
    mem_kill_raw_frac: float = 0.965  # restart recommended (+ optional kill)

    # --- Stronger OOM preemption ---
    # If raw mem is above this, start cancelling faster even if ramp isn't huge.
    oom_preempt_raw_frac: float = 0.90
    oom_preempt_cancel_max: int = 2
    oom_sustain_window_sec: float = 8.0  # if above crit for this long -> escalate
    oom_sustain_cancel_max: int = 4

    # --- Memory ramp detection ---
    mem_ramp_window_sec: float = 2.5
    mem_ramp_high_frac_per_sec: float = 0.018
    mem_ramp_crit_frac_per_sec: float = 0.028
    mem_ramp_kill_frac_per_sec: float = 0.045

    # Sampling
    sample_interval_sec: float = 0.35

    # CPU-aware escalation (optional but helps when mem+cpu spikes stall cancellation)
    cpu_high_frac: float = 0.90
    cpu_sample_interval_sec: float = 0.80

    # AIMD target concurrency
    min_target: int = 1
    max_target: int = 512
    initial_target: int = 4
    ai_step: int = 1
    md_factor: float = 0.55

    # Admission smoothing
    max_admit_per_call: int = 2
    max_admit_per_call_when_ramping: int = 1

    # Estimation
    per_company_min_mb: float = 256.0
    per_company_max_mb: float = 1280.0
    per_company_safety_factor: float = 1.35
    use_process_tree_rss_for_estimate: bool = True

    # Emergency cancel
    min_active_keep: int = 1
    emergency_cancel_max: int = 1
    emergency_cancel_cooldown_sec: float = 6.0

    # Restart gating
    restart_gate_min_uptime_sec: float = 150.0
    restart_gate_min_completed: int = 2

    # Stall detection (gated by suspend/resume)
    company_inactivity_timeout_sec: float = 240.0
    company_inactivity_cancel_max: int = 1

    admission_starvation_timeout_sec: float = 240.0
    block_log_interval_sec: float = 25.0

    no_progress_timeout_sec: float = 240.0
    kill_on_no_progress: bool = True

    # Optional heartbeat/logging
    heartbeat_path: Optional[Path] = None
    log_path: Optional[Path] = None

    # Prefer cgroup limits if available
    prefer_cgroup_limits: bool = True
    use_psutil: bool = True
    cgroup_subtract_inactive_file: bool = True

    # Optional hard watchdog thread
    hard_watchdog_enabled: bool = False
    hard_watchdog_interval_sec: float = 5.0
    hard_watchdog_startup_grace_sec: float = 160.0
    hard_watchdog_no_heartbeat_timeout_sec: float = 180.0
    hard_watchdog_kill_signal: int = 15  # SIGTERM

    # Work scheduling knobs
    max_start_per_tick: int = 3
    crawler_capacity_multiplier: int = 3

    idle_recycle_interval_sec: float = 18.0
    idle_recycle_raw_frac: float = 0.86
    idle_recycle_eff_frac: float = 0.81

    # Retry state location
    retry_base_dir: Optional[Path] = None

    # Backoff sleep smoothing
    min_idle_sleep_sec: float = 0.20
    max_idle_sleep_sec: float = 5.0

    cancel_requeue_min_delay_sec: float = 2.0

    company_cancel_repeat_cooldown_sec: float = 12.0
    company_cancel_inflight_timeout_sec: float = 600.0


class AdaptiveScheduler:
    """
    Public API:
      - start/stop
      - set_worklist(...)
      - plan_start_batch(...)
      - has_pending / pending_total / pending_ready
      - cleanup_completed_retry_ids(...)
      - register_company_completed()
      - touch_company(company_id)
      - suspend_stall_detection(...) / resume_stall_detection(...)
      - restart_recommended
      - get_state_snapshot()
    """

    def __init__(
        self,
        cfg: AdaptiveSchedulingConfig,
        get_active_company_ids: Callable[[], Sequence[str]],
        request_cancel_companies: Callable[[Sequence[str]], None],
        request_recycle_idle: Optional[Callable[[int, str], Awaitable[int]]] = None,
    ) -> None:
        self.cfg = cfg
        self._psutil_available = bool(PSUTIL_AVAILABLE and cfg.use_psutil)

        self._get_active_company_ids = get_active_company_ids
        self._request_cancel_companies = request_cancel_companies
        self._request_recycle_idle = request_recycle_idle

        if cfg.retry_base_dir is None:
            raise ValueError("AdaptiveSchedulingConfig.retry_base_dir must be set")
        self.retry_store = RetryStateStore(base_dir=Path(cfg.retry_base_dir))

        # Worklist
        self._work_ready: Deque[str] = deque()
        self._work_deferred: List[Tuple[float, str]] = []
        self._deferred_at: Dict[str, float] = {}
        self._queued: set[str] = set()
        self._work_seen: set[str] = set()
        self._work_total_hint: int = 0
        self._is_company_runnable: Optional[Callable[[str], Awaitable[bool]]] = None

        # Memory sampling cache
        self._total_mem_bytes: int = 0
        self._used_bytes_eff: int = 0
        self._used_frac_eff: float = 0.0
        self._used_bytes_raw: int = 0
        self._used_frac_raw: float = 0.0
        self._last_sample_ts: float = 0.0

        # CPU cache
        self._cpu_last_ts: float = 0.0
        self._cpu_frac: float = 0.0

        # Ramp (%/sec)
        self._mem_frac_hist: Deque[Tuple[float, float, float]] = deque(maxlen=256)
        self._ramp_eff_frac_per_sec: float = 0.0
        self._ramp_raw_frac_per_sec: float = 0.0

        # Process-tree memory
        self._proc_rss_bytes: int = 0

        # Estimator (MB)
        self._per_company_est_mb: float = cfg.per_company_min_mb
        self._base_rss_bytes: float = 0.0
        self._base_rss_samples: int = 0

        # Optional per-company peak tracking
        self._company_peak_mb: Dict[str, float] = {}

        # Company activity tracking
        self._company_last_touch_mono: Dict[str, float] = {}
        self._last_admission_mono: float = time.monotonic()
        self._last_block_log_mono: float = 0.0

        # AIMD target
        self._target_parallel: int = max(
            cfg.min_target, min(cfg.initial_target, cfg.max_target)
        )

        # Progress
        self._completed_counter: int = 0
        self._last_num_waiting: int = 0
        self._ever_admitted: bool = False
        self._ever_had_active: bool = False
        self._started_ts: float = time.time()
        self._started_mono: float = time.monotonic()

        # Heartbeats/progress
        self._last_heartbeat_mono: float = time.monotonic()
        self._last_progress_mono: float = time.monotonic()

        # Observations to infer progress
        self._last_obs_active_n: int = 0
        self._last_obs_waiting_n: int = 0
        self._last_obs_completed: int = 0

        # OOM sustain tracking
        self._oom_crit_enter_mono: Optional[float] = None

        # Emergency bookkeeping
        self._restart_recommended: bool = False
        self._last_emergency_cancel_ts: float = 0.0

        # Stall detection suspension
        self._stall_suspensions: set[str] = set()

        # Cancel-in-flight guard
        self._cancel_inflight_mono: Dict[str, float] = {}

        # Async + optional thread watchdog
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self._hard_stop = threading.Event()
        self._hard_thread: Optional[threading.Thread] = None

        self._last_log_ts: float = 0.0
        self._last_idle_recycle_mono: float = 0.0

    # ----------------------------
    # Simple worklist APIs
    # ----------------------------
    def pending_ready(self) -> int:
        return len(self._work_ready)

    def pending_total(self) -> int:
        return len(self._work_ready) + len(self._deferred_at)

    def has_pending(self) -> bool:
        return self.pending_total() > 0

    def initial_total_hint(self) -> int:
        return max(1, int(self._work_total_hint))

    def sleep_hint_sec(self) -> float:
        if self._work_ready:
            return float(self.cfg.min_idle_sleep_sec)
        if not self._work_deferred:
            return float(self.cfg.min_idle_sleep_sec)
        now = time.time()
        while self._work_deferred:
            eligible_at, cid = self._work_deferred[0]
            cur = self._deferred_at.get(cid)
            if cur is None or abs(cur - eligible_at) > 1e-6:
                heapq.heappop(self._work_deferred)
                continue
            dt = max(0.0, float(eligible_at - now))
            return float(
                max(self.cfg.min_idle_sleep_sec, min(dt, self.cfg.max_idle_sleep_sec))
            )
        return float(self.cfg.min_idle_sleep_sec)

    async def set_worklist(
        self,
        company_ids: Sequence[str],
        *,
        retry_mode: str = "all",
        is_company_runnable: Optional[Callable[[str], Awaitable[bool]]] = None,
    ) -> None:
        self._is_company_runnable = is_company_runnable
        ids = [str(x) for x in company_ids if str(x).strip()]
        self._work_total_hint = len(ids)

        pending_retry = set(self.retry_store.pending_ids(exclude_quarantined=True))
        if retry_mode == "skip-retry":
            ids = [cid for cid in ids if cid not in pending_retry]
        elif retry_mode == "only-retry":
            ids = [cid for cid in ids if cid in pending_retry]
        else:
            retry_mode = "all"

        now = time.time()
        # batch eligibility check for speed
        elig_mask: Dict[str, bool] = {}
        try:
            elig_mask = self.retry_store.eligible_mask(ids, now=now)
        except Exception:
            elig_mask = {}

        async with self._lock:
            for cid in ids:
                if cid in self._work_seen:
                    continue
                self._work_seen.add(cid)

                try:
                    if self.retry_store.is_quarantined(cid):
                        continue
                except Exception:
                    pass

                if cid in self._queued:
                    continue

                is_elig = elig_mask.get(cid, True)
                if not is_elig:
                    ts = float(self.retry_store.next_eligible_at(cid))
                    self._enqueue_deferred_locked(cid, ts)
                    continue

                self._enqueue_ready_locked(cid)

        logger.info(
            "[AdaptiveScheduling] seeded worklist ids=%d retry_mode=%s ready=%d deferred=%d",
            len(ids),
            retry_mode,
            self.pending_ready(),
            len(self._deferred_at),
        )

    async def cleanup_completed_retry_ids(
        self,
        *,
        is_company_runnable: Callable[[str], Awaitable[bool]],
        treat_non_runnable_as_done: bool,
        stage: str = "startup_cleanup",
    ) -> int:
        ids = sorted(self.retry_store.pending_ids(exclude_quarantined=True))
        if not ids:
            return 0

        cleared: List[str] = []
        for cid in ids:
            try:
                runnable = await is_company_runnable(cid)
                if treat_non_runnable_as_done and (not runnable):
                    cleared.append(cid)
            except Exception:
                continue

        if cleared:
            try:
                self.retry_store.mark_success_many(
                    cleared, stage=stage, note="already_done", flush=True
                )
            except Exception:
                for cid in cleared:
                    try:
                        self.retry_store.mark_success(
                            cid, stage=stage, note="already_done"
                        )
                    except Exception:
                        pass
        return len(cleared)

    # ----------------------------
    # Queue helpers (LOCKED)
    # ----------------------------
    def _enqueue_ready_locked(self, cid: str) -> None:
        if not cid or cid in self._queued:
            return
        self._queued.add(cid)
        self._work_ready.append(cid)

    def _enqueue_deferred_locked(self, cid: str, eligible_at: float) -> None:
        if not cid:
            return
        ts = float(eligible_at)
        if cid not in self._queued:
            self._queued.add(cid)
        prev = self._deferred_at.get(cid)
        if prev is None:
            self._deferred_at[cid] = ts
            heapq.heappush(self._work_deferred, (ts, cid))
            return
        new_ts = max(float(prev), ts)
        if abs(new_ts - float(prev)) > 1e-6:
            self._deferred_at[cid] = new_ts
            heapq.heappush(self._work_deferred, (new_ts, cid))

    def _pop_ready_locked(self) -> Optional[str]:
        while self._work_ready:
            cid = self._work_ready.popleft()
            if not cid:
                continue
            self._queued.discard(cid)
            return cid
        return None

    def _move_due_deferred_locked(self, now: float) -> None:
        moved = 0
        while self._work_deferred:
            ts, cid = self._work_deferred[0]
            cur = self._deferred_at.get(cid)
            if cur is None or abs(float(cur) - float(ts)) > 1e-6:
                heapq.heappop(self._work_deferred)
                continue
            if float(ts) > float(now):
                break

            heapq.heappop(self._work_deferred)
            self._deferred_at.pop(cid, None)

            try:
                if self.retry_store.is_quarantined(cid):
                    self._queued.discard(cid)
                    continue
            except Exception:
                pass

            try:
                if not self.retry_store.is_eligible(cid, now=now):
                    ts2 = float(self.retry_store.next_eligible_at(cid))
                    self._enqueue_deferred_locked(cid, ts2)
                    continue
            except Exception:
                pass

            self._work_ready.append(cid)
            moved += 1

        if moved and logger.isEnabledFor(logging.INFO):
            logger.info(
                "[AdaptiveScheduling] moved %d deferred -> ready (ready=%d deferred=%d)",
                moved,
                len(self._work_ready),
                len(self._deferred_at),
            )

    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def psutil_available(self) -> bool:
        return self._psutil_available

    @property
    def restart_recommended(self) -> bool:
        return self._restart_recommended

    @property
    def stall_detection_enabled(self) -> bool:
        return len(self._stall_suspensions) == 0

    # ----------------------------
    # Stall detection control
    # ----------------------------
    async def suspend_stall_detection(
        self,
        *,
        key: str,
        company_id: Optional[str] = None,
        reset_timers: bool = True,
        reason: str = "",
    ) -> None:
        if not key:
            return
        async with self._lock:
            was_enabled = self.stall_detection_enabled
            self._stall_suspensions.add(key)
            if reset_timers:
                now_mono = time.monotonic()
                self._last_progress_mono = now_mono
                self._last_heartbeat_mono = now_mono
                self._last_admission_mono = now_mono
                if company_id:
                    self._company_last_touch_mono[company_id] = now_mono
            if was_enabled and not self.stall_detection_enabled:
                logger.info(
                    "[AdaptiveScheduling] stall detection suspended key=%s reason=%s",
                    key,
                    reason or "unspecified",
                )
        if reset_timers:
            self._mark_heartbeat()
            self._mark_progress()

    async def resume_stall_detection(
        self,
        *,
        key: str,
        company_id: Optional[str] = None,
        reset_timers: bool = True,
        reason: str = "",
    ) -> None:
        if not key:
            return
        async with self._lock:
            was_enabled = self.stall_detection_enabled
            self._stall_suspensions.discard(key)
            if reset_timers:
                now_mono = time.monotonic()
                self._last_progress_mono = now_mono
                self._last_heartbeat_mono = now_mono
                self._last_admission_mono = now_mono
                if company_id:
                    self._company_last_touch_mono[company_id] = now_mono
            if (not was_enabled) and self.stall_detection_enabled:
                logger.info(
                    "[AdaptiveScheduling] stall detection resumed key=%s reason=%s",
                    key,
                    reason or "unspecified",
                )
        if reset_timers:
            self._mark_heartbeat()
            self._mark_progress()

    # ----------------------------
    # Company progress hooks
    # ----------------------------
    def touch_company(self, company_id: str) -> None:
        if not company_id:
            return
        self._company_last_touch_mono[company_id] = time.monotonic()
        self._mark_progress()

    def register_company_completed(self) -> None:
        self._completed_counter += 1
        self._mark_progress()

    def record_company_peak(self, company_id: str, peak_mb: float) -> None:
        if not company_id or peak_mb <= 0:
            return
        prev = self._company_peak_mb.get(company_id)
        if prev is None or peak_mb > prev:
            self._company_peak_mb[company_id] = peak_mb

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def start(self) -> None:
        self._started_ts = time.time()
        self._started_mono = time.monotonic()
        self._last_admission_mono = time.monotonic()
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
        self._used_bytes_eff = used_eff
        self._used_frac_eff = (float(used_eff) / float(total)) if total > 0 else 0.0
        self._used_bytes_raw = used_raw
        self._used_frac_raw = (float(used_raw) / float(total)) if total > 0 else 0.0

        self._proc_rss_bytes = self._read_process_tree_rss_bytes()
        self._last_sample_ts = time.time()
        self._cpu_last_ts = 0.0
        self._cpu_frac = 0.0

        now_mono = time.monotonic()
        self._update_ramp_locked(now_mono, self._used_frac_eff, self._used_frac_raw)

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

    # ----------------------------
    # State snapshot
    # ----------------------------
    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "total_mem_mb": float(self._total_mem_bytes) / _MB
            if self._total_mem_bytes
            else 0.0,
            "used_mem_mb": float(self._used_bytes_eff) / _MB
            if self._used_bytes_eff
            else 0.0,
            "used_frac": self._used_frac_eff,
            "used_raw_mb": float(self._used_bytes_raw) / _MB
            if self._used_bytes_raw
            else 0.0,
            "used_frac_raw": self._used_frac_raw,
            "ramp_eff_frac_per_sec": self._ramp_eff_frac_per_sec,
            "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
            "cpu_frac": self._cpu_frac,
            "proc_mem_mb": float(self._proc_rss_bytes) / _MB
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
            "stall_detection_enabled": self.stall_detection_enabled,
            "stall_suspensions": len(self._stall_suspensions),
            "cancel_inflight": len(self._cancel_inflight_mono),
            "last_heartbeat_age_sec": max(
                0.0, time.monotonic() - self._last_heartbeat_mono
            ),
            "last_progress_age_sec": max(
                0.0, time.monotonic() - self._last_progress_mono
            ),
            "last_admission_age_sec": max(
                0.0, time.monotonic() - self._last_admission_mono
            ),
            "ready": len(self._work_ready),
            "deferred": len(self._deferred_at),
        }

    # ----------------------------
    # Core: planning start batch
    # ----------------------------
    async def plan_start_batch(self, *, free_crawlers: int) -> List[str]:
        active_ids = list(self._get_active_company_ids())
        active_set = set(active_ids)

        async with self._lock:
            self._move_due_deferred_locked(time.time())

        if not self._work_ready:
            return []

        slots = await self.admissible_slots(num_waiting=len(self._work_ready))
        if slots <= 0:
            return []

        active_n = len(active_ids)
        hard_capacity = max(0, int(self.cfg.max_target) - active_n)
        if hard_capacity <= 0:
            return []

        free = int(free_crawlers)
        if free <= 0:
            return []
        mult = int(self.cfg.crawler_capacity_multiplier)
        if mult <= 0:
            return []
        crawler_cap = max(0, free * mult)
        if crawler_cap <= 0:
            return []

        want = min(
            int(slots),
            int(hard_capacity),
            int(crawler_cap),
            int(self.cfg.max_start_per_tick),
        )
        want = max(0, int(want))
        if want <= 0:
            return []

        picked: List[str] = []
        scanned = 0
        max_scan = max(want * 4, want + 8)
        now = time.time()

        while len(picked) < want and scanned < max_scan:
            scanned += 1
            async with self._lock:
                cid = self._pop_ready_locked()

            if cid is None:
                break

            if cid in active_set:
                async with self._lock:
                    self._enqueue_deferred_locked(
                        cid, now + float(self.cfg.cancel_requeue_min_delay_sec)
                    )
                continue

            if self._is_company_runnable is not None:
                try:
                    runnable = await self._is_company_runnable(cid)
                    if not runnable:
                        try:
                            self.retry_store.mark_success(
                                cid,
                                stage="scheduler_skip",
                                note="already_done_or_terminal",
                            )
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass

            picked.append(cid)

        if picked:
            self._ever_admitted = True
            self._last_admission_mono = time.monotonic()
            self._mark_progress()

        return picked

    # ----------------------------
    # Admission control
    # ----------------------------
    def _maybe_log_block(self, reason: str, active_n: int, waiting: int) -> None:
        now_mono = time.monotonic()
        if (now_mono - self._last_block_log_mono) < float(
            self.cfg.block_log_interval_sec
        ):
            return
        self._last_block_log_mono = now_mono
        logger.warning(
            "[AdaptiveScheduling] admission blocked reason=%s used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s cpu=%.2f active=%d waiting=%d proc_mem_mb=%.1f target=%d",
            reason,
            self._used_frac_raw,
            self._used_frac_eff,
            self._ramp_raw_frac_per_sec,
            self._cpu_frac,
            active_n,
            waiting,
            float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0,
            self._target_parallel,
        )

    async def admissible_slots(self, num_waiting: int) -> int:
        self._last_num_waiting = max(0, int(num_waiting))
        self._mark_heartbeat()

        if num_waiting <= 0:
            return 0

        if not self._psutil_available:
            self._ever_admitted = True
            self._mark_progress()
            self._last_admission_mono = time.monotonic()
            return 1

        async with self._lock:
            now = time.time()
            if (now - self._last_sample_ts) >= float(self.cfg.sample_interval_sec):
                self._sample_memory_locked(now)

            self._sample_cpu_locked(now)

            active_ids = list(self._get_active_company_ids())
            active_n = len(active_ids)
            if active_n > 0:
                self._ever_had_active = True

            self._update_base_rss_locked(active_n)
            self._update_per_company_est_locked(active_n)
            self._update_target_parallel_locked(active_n)

            # Block admissions earlier to prevent ramps from running away
            if self._used_frac_raw >= float(self.cfg.mem_high_raw_frac):
                self._maybe_log_block("block_mem_high_raw", active_n, num_waiting)
                return 0

            if active_n > 0 and self._ramp_raw_frac_per_sec >= float(
                self.cfg.mem_ramp_high_frac_per_sec
            ):
                self._maybe_log_block("block_mem_ramp_high", active_n, num_waiting)
                return 0

            if self._used_frac_eff >= float(self.cfg.mem_high_frac):
                self._maybe_log_block("block_mem_high_eff", active_n, num_waiting)
                return 0

            total = self._total_mem_bytes
            if total <= 0:
                self._maybe_log_block("block_total_unknown", active_n, num_waiting)
                return 0

            cap_bytes = int(float(self.cfg.mem_cap_frac) * float(total))
            headroom = cap_bytes - self._used_bytes_eff

            if headroom <= 0:
                slots = (
                    1
                    if (
                        active_n == 0
                        and self._used_frac_eff < float(self.cfg.mem_high_frac)
                    )
                    else 0
                )
                if slots == 0:
                    self._maybe_log_block("block_mem_cap", active_n, num_waiting)
            else:
                per_company_bytes = self._per_company_reservation_bytes_locked()
                slots_by_mem = (
                    int(headroom // per_company_bytes) if per_company_bytes > 0 else 0
                )
                slots_by_target = max(0, int(self._target_parallel) - active_n)
                slots = min(int(num_waiting), int(slots_by_mem), int(slots_by_target))
                if (
                    slots <= 0
                    and active_n == 0
                    and self._used_frac_eff < float(self.cfg.mem_high_frac)
                ):
                    slots = 1

            slots = max(0, int(slots))

            # Anti-burst
            slots = min(slots, max(1, int(self.cfg.max_admit_per_call)))
            if (
                active_n > 0
                and self._ramp_raw_frac_per_sec
                >= float(self.cfg.mem_ramp_high_frac_per_sec) * 0.7
            ):
                slots = min(
                    slots, max(1, int(self.cfg.max_admit_per_call_when_ramping))
                )

            if slots > 0:
                self._ever_admitted = True
                self._mark_progress()
                self._last_admission_mono = time.monotonic()
                self._maybe_log_state_locked(
                    "admission",
                    extra={
                        "slots": slots,
                        "waiting": int(num_waiting),
                        "active": active_n,
                        "target_parallel": self._target_parallel,
                        "used_frac": self._used_frac_eff,
                        "used_frac_raw": self._used_frac_raw,
                        "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
                        "cpu_frac": self._cpu_frac,
                        "per_company_est_mb": self._per_company_est_mb,
                        "proc_mem_mb": float(self._proc_rss_bytes) / _MB
                        if self._proc_rss_bytes
                        else 0.0,
                        "stall_detection_enabled": self.stall_detection_enabled,
                        "stall_suspensions": len(self._stall_suspensions),
                        "cancel_inflight": len(self._cancel_inflight_mono),
                        "ready": len(self._work_ready),
                        "deferred": len(self._deferred_at),
                    },
                )

            return slots

    # ----------------------------
    # Cancel recording + re-queueing (uses retry_state.py schedule)
    # ----------------------------
    async def _record_cancels_and_schedule_retries(
        self,
        cancel_ids: Sequence[str],
        *,
        cls_hint: str,
        error: str,
        stage: str,
        status_code: Optional[int] = None,
        permanent_reason: str = "",
    ) -> None:
        if not cancel_ids:
            return

        # Batch record failures if supported
        events = [
            {
                "company_id": str(cid),
                "cls": str(cls_hint),
                "error": str(error),
                "stage": str(stage),
                "status_code": status_code,
                "permanent_reason": str(permanent_reason or ""),
            }
            for cid in cancel_ids
        ]

        used_batch = False
        try:
            fn = getattr(self.retry_store, "mark_failure_many", None)
            if callable(fn):
                fn(events, flush=True)
                used_batch = True
        except Exception:
            used_batch = False

        if not used_batch:
            for cid in cancel_ids:
                _mark_failure_compat(
                    self.retry_store,
                    str(cid),
                    cls=str(cls_hint),
                    error=str(error),
                    stage=str(stage),
                    status_code=status_code,
                    permanent_reason=str(permanent_reason or ""),
                )
            try:
                self.retry_store.flush(force=False)
            except Exception:
                pass

        now = time.time()
        active_set = set(self._get_active_company_ids())

        # batch next_eligible_at if available
        next_map: Dict[str, float] = {}
        try:
            fn2 = getattr(self.retry_store, "next_eligible_at_many", None)
            if callable(fn2):
                next_map = fn2([str(cid) for cid in cancel_ids])  # type: ignore[misc]
        except Exception:
            next_map = {}

        items: List[Tuple[str, float]] = []
        grace = float(self.cfg.cancel_requeue_min_delay_sec)
        for cid in cancel_ids:
            scid = str(cid)
            try:
                if self.retry_store.is_quarantined(scid):
                    continue
            except Exception:
                pass

            ts = float(next_map.get(scid, 0.0))
            if ts <= 0:
                try:
                    ts = float(self.retry_store.next_eligible_at(scid))
                except Exception:
                    ts = 0.0

            min_ts = now + grace
            if scid in active_set:
                min_ts = now + grace
            ts = max(float(ts or 0.0), float(min_ts))
            items.append((scid, ts))

        if not items:
            return

        async with self._lock:
            for cid, ts in items:
                if cid in self._queued:
                    if cid in self._deferred_at:
                        self._enqueue_deferred_locked(cid, ts)
                    continue
                if ts <= now:
                    self._enqueue_ready_locked(cid)
                else:
                    self._enqueue_deferred_locked(cid, ts)

    # ----------------------------
    # Watchdog loop
    # ----------------------------
    async def _watchdog_loop(self) -> None:
        interval = max(0.25, float(self.cfg.sample_interval_sec))
        while True:
            try:
                await asyncio.sleep(interval)
                self._mark_heartbeat()

                if not self._psutil_available:
                    continue

                cancel_ids: List[str] = []
                cancel_reason: str = ""
                should_recycle_idle = False

                async with self._lock:
                    now = time.time()
                    self._sample_memory_locked(now)
                    self._sample_cpu_locked(now)

                    active_ids = list(self._get_active_company_ids())
                    active_n = len(active_ids)
                    if active_n > 0:
                        self._ever_had_active = True

                    self._update_base_rss_locked(active_n)
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

                    now_mono = time.monotonic()

                    # touch map tidy
                    active_set = set(active_ids)
                    for cid in list(self._company_last_touch_mono.keys()):
                        if cid not in active_set:
                            self._company_last_touch_mono.pop(cid, None)
                    for cid in active_ids:
                        self._company_last_touch_mono.setdefault(cid, now_mono)

                    # cancel inflight tidy
                    inflight_timeout = float(
                        self.cfg.company_cancel_inflight_timeout_sec
                    )
                    for cid, ts in list(self._cancel_inflight_mono.items()):
                        if cid not in active_set:
                            self._cancel_inflight_mono.pop(cid, None)
                            continue
                        if inflight_timeout > 0 and (now_mono - ts) >= inflight_timeout:
                            self._cancel_inflight_mono.pop(cid, None)

                    repeat_cooldown = float(self.cfg.company_cancel_repeat_cooldown_sec)

                    def _recently_cancelled(cid: str) -> bool:
                        ts = self._cancel_inflight_mono.get(cid)
                        return (
                            ts is not None
                            and repeat_cooldown > 0
                            and ((now_mono - ts) < repeat_cooldown)
                        )

                    stall_enabled = self.stall_detection_enabled

                    # 1) Per-company inactivity cancel (gated)
                    if (
                        stall_enabled
                        and float(self.cfg.company_inactivity_timeout_sec) > 0
                        and active_n > 0
                    ):
                        stuck: List[str] = []
                        for cid in active_ids:
                            if _recently_cancelled(cid):
                                continue
                            last = self._company_last_touch_mono.get(cid, now_mono)
                            if (now_mono - last) >= float(
                                self.cfg.company_inactivity_timeout_sec
                            ):
                                stuck.append(cid)
                        if stuck:
                            stuck = stuck[: int(self.cfg.company_inactivity_cancel_max)]
                            logger.error(
                                "[AdaptiveScheduling] company inactivity %.1fs -> cancelling %s",
                                float(self.cfg.company_inactivity_timeout_sec),
                                stuck,
                            )
                            cancel_ids = list(stuck)
                            cancel_reason = "stall"

                    # 2) No-progress -> restart recommended (gated)
                    if stall_enabled and float(self.cfg.no_progress_timeout_sec) > 0:
                        prog_age = now_mono - self._last_progress_mono
                        if prog_age >= float(self.cfg.no_progress_timeout_sec):
                            if self._last_num_waiting > 0 or self.has_pending():
                                if self._can_recommend_restart():
                                    if not self._restart_recommended:
                                        logger.error(
                                            "[AdaptiveScheduling] no progress for %.1fs with waiting=%d -> restart recommended",
                                            prog_age,
                                            self._last_num_waiting,
                                        )
                                    self._restart_recommended = True
                                    if self.cfg.kill_on_no_progress:
                                        try:
                                            os.kill(
                                                os.getpid(),
                                                int(self.cfg.hard_watchdog_kill_signal),
                                            )
                                        except Exception:
                                            pass

                    # 3) Admission starvation -> restart (gated)
                    if (
                        stall_enabled
                        and float(self.cfg.admission_starvation_timeout_sec) > 0
                    ):
                        if active_n == 0 and (
                            self._last_num_waiting > 0 or self.has_pending()
                        ):
                            starve_age = now_mono - self._last_admission_mono
                            if starve_age >= float(
                                self.cfg.admission_starvation_timeout_sec
                            ):
                                if (
                                    self._used_frac_raw
                                    >= float(self.cfg.mem_high_raw_frac)
                                ) or (
                                    self._used_frac_eff >= float(self.cfg.mem_high_frac)
                                ):
                                    if self._can_recommend_restart():
                                        if not self._restart_recommended:
                                            logger.critical(
                                                "[AdaptiveScheduling] admission starvation %.1fs with mem high (used_raw=%.3f used_eff=%.3f) -> restart recommended",
                                                starve_age,
                                                self._used_frac_raw,
                                                self._used_frac_eff,
                                            )
                                        self._restart_recommended = True
                                        if self.cfg.kill_on_no_progress:
                                            try:
                                                os.kill(
                                                    os.getpid(),
                                                    int(
                                                        self.cfg.hard_watchdog_kill_signal
                                                    ),
                                                )
                                            except Exception:
                                                pass

                    # 4) Raw mem kill -> restart (NOT gated)
                    if self._used_frac_raw >= float(self.cfg.mem_kill_raw_frac):
                        if self._can_recommend_restart():
                            if not self._restart_recommended:
                                logger.critical(
                                    "[AdaptiveScheduling] raw mem %.3f >= kill %.3f -> restart recommended",
                                    self._used_frac_raw,
                                    float(self.cfg.mem_kill_raw_frac),
                                )
                            self._restart_recommended = True
                            if self.cfg.kill_on_no_progress:
                                try:
                                    os.kill(
                                        os.getpid(),
                                        int(self.cfg.hard_watchdog_kill_signal),
                                    )
                                except Exception:
                                    pass

                    # 5) Ramp kill -> restart (NOT gated)
                    if self._ramp_raw_frac_per_sec >= float(
                        self.cfg.mem_ramp_kill_frac_per_sec
                    ) and self._used_frac_raw >= float(self.cfg.mem_high_raw_frac):
                        if self._can_recommend_restart():
                            if not self._restart_recommended:
                                logger.critical(
                                    "[AdaptiveScheduling] mem ramp raw=%.3f/s with used_raw=%.3f (>=high %.3f) -> restart recommended",
                                    self._ramp_raw_frac_per_sec,
                                    self._used_frac_raw,
                                    float(self.cfg.mem_high_raw_frac),
                                )
                            self._restart_recommended = True
                            if self.cfg.kill_on_no_progress:
                                try:
                                    os.kill(
                                        os.getpid(),
                                        int(self.cfg.hard_watchdog_kill_signal),
                                    )
                                except Exception:
                                    pass

                    # 6) OOM sustain tracking (escalate cancels if raw stays critical)
                    crit_now = (
                        self._used_frac_raw >= float(self.cfg.mem_crit_raw_frac)
                    ) or (self._used_frac_eff >= float(self.cfg.mem_crit_frac))
                    if crit_now:
                        if self._oom_crit_enter_mono is None:
                            self._oom_crit_enter_mono = now_mono
                    else:
                        self._oom_crit_enter_mono = None

                    # 7) Emergency cancels (NOT gated): preempt + ramp + crit + sustain
                    if active_n > 0 and not cancel_ids:
                        time_ok = (now - self._last_emergency_cancel_ts) >= float(
                            self.cfg.emergency_cancel_cooldown_sec
                        )
                        if time_ok:
                            cancelable = [
                                cid
                                for cid in active_ids
                                if not _recently_cancelled(cid)
                            ]
                            inflight_active = sum(
                                1
                                for cid in active_ids
                                if cid in self._cancel_inflight_mono
                            )
                            max_cancelable = max(
                                0,
                                (active_n - inflight_active)
                                - int(self.cfg.min_active_keep),
                            )
                            if max_cancelable > 0 and cancelable:
                                # determine severity => number to cancel
                                ramp_crit = self._ramp_raw_frac_per_sec >= float(
                                    self.cfg.mem_ramp_crit_frac_per_sec
                                ) and self._used_frac_raw >= float(
                                    self.cfg.mem_high_raw_frac
                                )
                                preempt = self._used_frac_raw >= float(
                                    self.cfg.oom_preempt_raw_frac
                                )
                                sustain = False
                                if self._oom_crit_enter_mono is not None:
                                    sustain = (
                                        now_mono - self._oom_crit_enter_mono
                                    ) >= float(self.cfg.oom_sustain_window_sec)

                                cpu_hot = self._cpu_frac >= float(
                                    self.cfg.cpu_high_frac
                                )

                                to_cancel = int(self.cfg.emergency_cancel_max)
                                reason = "mem"

                                # escalate quickly:
                                if ramp_crit:
                                    to_cancel = max(to_cancel, 2)
                                    reason = "mem_heavy"
                                if preempt and cpu_hot:
                                    to_cancel = max(
                                        to_cancel, int(self.cfg.oom_preempt_cancel_max)
                                    )
                                    reason = "mem_heavy"
                                if crit_now:
                                    to_cancel = max(to_cancel, 2)
                                    reason = "mem_heavy"
                                if sustain:
                                    to_cancel = max(
                                        to_cancel, int(self.cfg.oom_sustain_cancel_max)
                                    )
                                    reason = "mem_heavy"

                                to_cancel = min(to_cancel, max_cancelable)
                                if to_cancel > 0:
                                    cancel_ids = self._select_cancel_ids(
                                        cancelable, to_cancel
                                    )
                                    self._last_emergency_cancel_ts = now
                                    cancel_reason = reason

                                    logger.error(
                                        "[AdaptiveScheduling] EMERGENCY used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s cpu=%.2f total_mb=%.1f used_raw_mb=%.1f used_eff_mb=%.1f proc_mem_mb=%.1f active=%d cancel=%s sustain=%s",
                                        self._used_frac_raw,
                                        self._used_frac_eff,
                                        self._ramp_raw_frac_per_sec,
                                        self._cpu_frac,
                                        float(self._total_mem_bytes) / _MB,
                                        float(self._used_bytes_raw) / _MB,
                                        float(self._used_bytes_eff) / _MB,
                                        float(self._proc_rss_bytes) / _MB
                                        if self._proc_rss_bytes
                                        else 0.0,
                                        active_n,
                                        cancel_ids,
                                        "yes" if sustain else "no",
                                    )
                                    self._maybe_log_state_locked(
                                        "emergency_cancel",
                                        extra={
                                            "active": active_n,
                                            "cancel_ids": cancel_ids,
                                            "used_frac_raw": self._used_frac_raw,
                                            "used_frac": self._used_frac_eff,
                                            "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
                                            "cpu_frac": self._cpu_frac,
                                            "stall_detection_enabled": stall_enabled,
                                            "stall_suspensions": len(
                                                self._stall_suspensions
                                            ),
                                            "cancel_reason": cancel_reason,
                                            "sustain": sustain,
                                        },
                                    )

                    # 8) Idle recycle request (memory pressure, throttled)
                    if self._request_recycle_idle is not None:
                        if (now_mono - self._last_idle_recycle_mono) >= float(
                            self.cfg.idle_recycle_interval_sec
                        ):
                            if (
                                self._used_frac_raw
                                >= float(self.cfg.idle_recycle_raw_frac)
                            ) or (
                                self._used_frac_eff
                                >= float(self.cfg.idle_recycle_eff_frac)
                            ):
                                should_recycle_idle = True
                                self._last_idle_recycle_mono = now_mono

                # Outside lock: apply cancels
                if cancel_ids:
                    inflight_mark_ts = time.monotonic()
                    async with self._lock:
                        for cid in cancel_ids:
                            self._cancel_inflight_mono[str(cid)] = inflight_mark_ts

                    try:
                        self._request_cancel_companies(cancel_ids)
                        self._mark_progress()
                    except Exception:
                        logger.exception(
                            "[AdaptiveScheduling] request_cancel_companies failed"
                        )
                        async with self._lock:
                            for cid in cancel_ids:
                                self._cancel_inflight_mono.pop(str(cid), None)
                    else:
                        if cancel_reason in ("mem", "mem_heavy"):
                            cls_hint = (
                                "mem_heavy" if cancel_reason == "mem_heavy" else "mem"
                            )
                            err = (
                                f"cancelled_by_scheduler:{cls_hint} "
                                f"used_raw={self._used_frac_raw:.3f} used_eff={self._used_frac_eff:.3f} "
                                f"ramp_raw={self._ramp_raw_frac_per_sec:.3f}/s cpu={self._cpu_frac:.2f} "
                                f"proc_mb={(float(self._proc_rss_bytes) / _MB) if self._proc_rss_bytes else 0.0:.1f}"
                            )
                            await self._record_cancels_and_schedule_retries(
                                cancel_ids,
                                cls_hint=cls_hint,
                                error=err,
                                stage="scheduler_mem_cancel",
                                status_code=None,
                                permanent_reason="",
                            )
                        elif cancel_reason == "stall":
                            err = (
                                "cancelled_by_scheduler:stall "
                                f"inactivity_timeout={float(self.cfg.company_inactivity_timeout_sec):.1f}s"
                            )
                            await self._record_cancels_and_schedule_retries(
                                cancel_ids,
                                cls_hint="stall",
                                error=err,
                                stage="scheduler_inactivity_cancel",
                                status_code=None,
                                permanent_reason="",
                            )

                    try:
                        gc.collect()
                    except Exception:
                        pass

                if should_recycle_idle and self._request_recycle_idle is not None:
                    try:
                        await self._request_recycle_idle(1, "mem_pressure_idle_recycle")
                    except Exception:
                        logger.exception(
                            "[AdaptiveScheduling] request_recycle_idle failed"
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[AdaptiveScheduling] watchdog loop error")

    # ----------------------------
    # Restart gating
    # ----------------------------
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

    # ----------------------------
    # Cancel selection
    # ----------------------------
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

    # ----------------------------
    # Heartbeat/progress
    # ----------------------------
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
                "used_frac": self._used_frac_eff,
                "used_frac_raw": self._used_frac_raw,
                "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
                "cpu_frac": self._cpu_frac,
                "target_parallel": self._target_parallel,
                "proc_mem_mb": float(self._proc_rss_bytes) / _MB
                if self._proc_rss_bytes
                else 0.0,
                "stall_detection_enabled": self.stall_detection_enabled,
                "stall_suspensions": len(self._stall_suspensions),
                "cancel_inflight": len(self._cancel_inflight_mono),
                "ready": len(self._work_ready),
                "deferred": len(self._deferred_at),
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            return

    # ----------------------------
    # Memory reading helpers
    # ----------------------------
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
        Returns (limit_bytes, used_effective_bytes, used_raw_bytes).
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

            def mem_bytes(proc: Any) -> int:
                try:
                    full = proc.memory_full_info()
                    uss = getattr(full, "uss", None)
                    if uss is not None:
                        return int(uss)
                except Exception:
                    pass
                try:
                    return int(proc.memory_info().rss)
                except Exception:
                    return 0

            total = mem_bytes(p)
            for ch in p.children(recursive=True):
                total += mem_bytes(ch)
            return int(total)
        except Exception:
            return 0

    def _read_memory_usage_bytes(self) -> Tuple[int, int, int]:
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

    def _sample_cpu_locked(self, now: float) -> None:
        if not self._psutil_available or psutil is None:
            return
        if self._cpu_last_ts and (now - self._cpu_last_ts) < float(
            self.cfg.cpu_sample_interval_sec
        ):
            return
        self._cpu_last_ts = now
        try:
            # system-wide cpu percent since last call; ok for heuristic escalation
            v = psutil.cpu_percent(interval=None)  # type: ignore[union-attr]
            self._cpu_frac = max(0.0, min(1.0, float(v) / 100.0))
        except Exception:
            self._cpu_frac = 0.0

    def _update_ramp_locked(
        self, now_mono: float, used_frac_eff: float, used_frac_raw: float
    ) -> None:
        window = max(0.5, float(self.cfg.mem_ramp_window_sec))
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
            self._used_bytes_eff = used_eff
            self._used_frac_eff = float(used_eff) / float(total)

            self._used_bytes_raw = used_raw
            self._used_frac_raw = float(used_raw) / float(total)

            self._last_sample_ts = now

            now_mono = time.monotonic()
            self._update_ramp_locked(now_mono, self._used_frac_eff, self._used_frac_raw)

        self._proc_rss_bytes = self._read_process_tree_rss_bytes()

    # ----------------------------
    # Estimators + AIMD
    # ----------------------------
    def _update_base_rss_locked(self, active_n: int) -> None:
        if active_n != 0:
            return

        base_now = (
            float(self._proc_rss_bytes)
            if (self.cfg.use_process_tree_rss_for_estimate and self._proc_rss_bytes > 0)
            else float(self._used_bytes_eff)
        )
        if base_now <= 0:
            return

        alpha = 0.12
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
            else float(self._used_bytes_eff)
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
            new = 0.65 * old + 0.35 * per_company_now_mb
        else:
            new = 0.88 * old + 0.12 * per_company_now_mb

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

        # More aggressive multiplicative decrease when high memory or fast ramp
        if (self._used_frac_raw >= float(cfg.mem_high_raw_frac)) or (
            self._used_frac_eff >= float(cfg.mem_high_frac)
        ):
            new = int(
                max(cfg.min_target, max(1, int(float(old) * float(cfg.md_factor))))
            )
        elif self._used_frac_eff <= (float(cfg.mem_cap_frac) - 0.04):
            new = min(cfg.max_target, old + int(cfg.ai_step))
        else:
            new = old

        if active_n > 0:
            new = max(new, active_n)

        self._target_parallel = int(max(cfg.min_target, min(new, cfg.max_target)))

        if self._target_parallel != old and logger.isEnabledFor(logging.INFO):
            logger.info(
                "[AdaptiveScheduling] target_parallel %d -> %d (used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s cpu=%.2f active=%d proc_mem_mb=%.1f)",
                old,
                self._target_parallel,
                self._used_frac_raw,
                self._used_frac_eff,
                self._ramp_raw_frac_per_sec,
                self._cpu_frac,
                active_n,
                float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0,
            )

    # ----------------------------
    # Optional hard watchdog
    # ----------------------------
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

    # ----------------------------
    # Logging
    # ----------------------------
    def _maybe_log_state_locked(
        self, reason: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.cfg.log_path is None:
            return
        now = time.time()
        if self._last_log_ts and (now - self._last_log_ts) < 4.0:
            return
        self._last_log_ts = now

        state: Dict[str, Any] = {
            "ts": now,
            "reason": reason,
            "total_mem_bytes": self._total_mem_bytes,
            "used_bytes_eff": self._used_bytes_eff,
            "used_frac_eff": self._used_frac_eff,
            "used_bytes_raw": self._used_bytes_raw,
            "used_frac_raw": self._used_frac_raw,
            "ramp_eff_frac_per_sec": self._ramp_eff_frac_per_sec,
            "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
            "cpu_frac": self._cpu_frac,
            "proc_mem_bytes": self._proc_rss_bytes,
            "per_company_est_mb": self._per_company_est_mb,
            "target_parallel": self._target_parallel,
            "completed_counter": self._completed_counter,
            "waiting": self._last_num_waiting,
            "restart_recommended": self._restart_recommended,
            "ever_admitted": self._ever_admitted,
            "ever_had_active": self._ever_had_active,
            "stall_detection_enabled": self.stall_detection_enabled,
            "stall_suspensions": len(self._stall_suspensions),
            "cancel_inflight": len(self._cancel_inflight_mono),
            "ready": len(self._work_ready),
            "deferred": len(self._deferred_at),
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


__all__ = [
    "AdaptiveSchedulingConfig",
    "AdaptiveScheduler",
    "compute_retry_exit_code_from_store",
]
