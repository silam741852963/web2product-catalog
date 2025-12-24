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


def _load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _retry_pending_ids(store: RetryStateStore) -> set[str]:
    """
    IDs currently in retry_state.json excluding quarantined.
    """
    st = _load_json_dict(store.state_path)
    q = _load_json_dict(store.quarantine_path)
    state_ids = set(st.keys())
    quarantined = set(q.keys())
    return state_ids - quarantined


def _eligible_retry_pending_ids(
    store: RetryStateStore, *, now: Optional[float] = None
) -> set[str]:
    if now is None:
        now = time.time()
    ids = _retry_pending_ids(store)
    out: set[str] = set()
    for cid in ids:
        try:
            if store.is_eligible(cid, now=now):
                out.add(cid)
        except Exception:
            continue
    return out


def compute_retry_exit_code_from_store(
    store: RetryStateStore, retry_exit_code: int
) -> int:
    """
    Return retry_exit_code only if there exists at least one *eligible now* retry-pending company.
    Prevents tight restart loops when next_eligible_at is in the future.
    """
    try:
        eligible = _eligible_retry_pending_ids(store, now=time.time())
        return int(retry_exit_code) if eligible else 0
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
        # must never raise from scheduler watchdog
        return


@dataclass
class AdaptiveSchedulingConfig:
    """
    Scheduler owns:
      - admission control (memory + AIMD)
      - cancellation policy (critical memory OR inactivity)
      - restart recommendation (raw memory kill threshold + deadlock watchdog)
      - retry/quarantine/backoff filtering
      - worklist queueing (ready + deferred)
    """

    # --- Memory thresholds (effective / raw) ---
    mem_cap_frac: float = 0.80
    mem_high_frac: float = 0.86
    mem_crit_frac: float = 0.92

    mem_high_raw_frac: float = 0.90
    mem_crit_raw_frac: float = 0.94
    mem_kill_raw_frac: float = 0.97  # restart recommended (+ optional kill)

    # --- Memory ramp detection ---
    mem_ramp_window_sec: float = 3.0
    mem_ramp_high_frac_per_sec: float = 0.020
    mem_ramp_crit_frac_per_sec: float = 0.030
    mem_ramp_kill_frac_per_sec: float = 0.050

    # Admission smoothing: guard admissions when ramping
    ramp_admit_guard_frac_per_sec: float = 0.012

    # Sampling
    sample_interval_sec: float = 0.50

    # AIMD target concurrency
    min_target: int = 1
    max_target: int = 512
    initial_target: int = 4
    ai_step: int = 1
    md_factor: float = 0.60

    # Admission smoothing
    max_admit_per_call: int = 3
    max_admit_per_call_when_ramping: int = 1

    # Estimation
    per_company_min_mb: float = 256.0
    per_company_max_mb: float = 1024.0
    per_company_safety_factor: float = 1.30
    use_process_tree_rss_for_estimate: bool = True

    # Emergency cancel (memory danger)
    min_active_keep: int = 1
    emergency_cancel_max: int = 1
    emergency_cancel_cooldown_sec: float = 12.0

    # Restart gating (normal restarts)
    restart_gate_min_uptime_sec: float = 180.0
    restart_gate_min_completed: int = 3

    # --- NEW: Deadlock restart (active==0 + waiting>0 + mem_high for long) ---
    deadlock_restart_enabled: bool = True
    deadlock_restart_timeout_sec: float = (
        300.0  # how long we tolerate "waiting but cannot admit"
    )
    deadlock_restart_require_mem_high: bool = (
        True  # require mem_high to be true for deadlock restart
    )

    # --- Stall detection / aggressive watchdog logic (gated by suspend/resume) ---
    company_inactivity_timeout_sec: float = 240.0
    company_inactivity_cancel_max: int = 1

    admission_starvation_timeout_sec: float = 240.0
    block_log_interval_sec: float = 30.0

    no_progress_timeout_sec: float = 240.0
    kill_on_no_progress: bool = True

    # Optional heartbeat/logging
    heartbeat_path: Optional[Path] = None
    log_path: Optional[Path] = None

    # Prefer cgroup limits if available
    prefer_cgroup_limits: bool = True
    use_psutil: bool = True
    cgroup_subtract_inactive_file: bool = True

    # Optional hard watchdog thread (disabled by default)
    hard_watchdog_enabled: bool = False
    hard_watchdog_interval_sec: float = 5.0
    hard_watchdog_startup_grace_sec: float = 180.0
    hard_watchdog_no_heartbeat_timeout_sec: float = 180.0
    hard_watchdog_kill_signal: int = 15  # SIGTERM

    # --- NEW: restart escalation (when restart recommended) ---
    restart_sigterm_repeat_interval_sec: float = 30.0
    restart_escalate_sigkill_after_sec: float = (
        120.0  # after SIGTERM, last resort SIGKILL
    )

    # --- Work scheduling knobs (scheduler owns queueing) ---
    max_start_per_tick: int = 3
    crawler_capacity_multiplier: int = 3  # allow N companies per free crawler slot

    idle_recycle_interval_sec: float = 25.0
    idle_recycle_raw_frac: float = 0.88
    idle_recycle_eff_frac: float = 0.83
    idle_recycle_max_per_interval: int = 2  # NEW: recycle more when truly stuck

    # Retry state location (scheduler owns store)
    retry_base_dir: Optional[Path] = None

    # Backoff sleep smoothing
    min_idle_sleep_sec: float = 0.25
    max_idle_sleep_sec: float = 5.0

    # When the scheduler cancels a company (stall/mem), do not re-admit it immediately.
    cancel_requeue_min_delay_sec: float = 2.0

    # Cancel spam guard
    company_cancel_repeat_cooldown_sec: float = 15.0
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
        self._work_deferred: List[Tuple[float, str]] = []  # heap (eligible_at_ts, cid)
        self._deferred_at: Dict[str, float] = {}  # cid -> current eligible_at
        self._queued: set[str] = set()  # cid present in ready or deferred
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

        # Ramp (%/sec)
        self._mem_frac_hist: Deque[Tuple[float, float, float]] = deque(maxlen=512)
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

        # Emergency bookkeeping
        self._restart_recommended: bool = False
        self._restart_reason: str = ""
        self._restart_recommended_mono: Optional[float] = None
        self._restart_sigterm_sent_mono: Optional[float] = None
        self._last_emergency_cancel_ts: float = 0.0

        # Deadlock detection bookkeeping
        self._deadlock_since_mono: Optional[float] = None

        # Stall detection suspension
        self._stall_suspensions: set[str] = set()

        # Cancel-in-flight guard
        self._cancel_inflight_mono: Dict[str, float] = {}

        # Async + optional thread watchdog
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self._hard_stop = threading.Event()
        self._hard_thread: Optional[threading.Thread] = None

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
        """
        When nothing can start now (due to backoff / memory), suggest a sleep duration.
        """
        if self._work_ready:
            return float(self.cfg.min_idle_sleep_sec)
        if not self._work_deferred:
            return float(self.cfg.min_idle_sleep_sec)
        now = time.time()
        # Find the next valid deferred entry (skip stale heap entries)
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
        """
        Seed scheduler-owned queue.

        retry_mode:
          - all: include all IDs
          - skip-retry: exclude IDs currently pending in retry_state.json
          - only-retry: include only IDs pending in retry_state.json
        """
        self._is_company_runnable = is_company_runnable
        ids = [str(x) for x in company_ids if str(x).strip()]
        self._work_total_hint = len(ids)

        pending_retry = _retry_pending_ids(self.retry_store)
        if retry_mode == "skip-retry":
            ids = [cid for cid in ids if cid not in pending_retry]
        elif retry_mode == "only-retry":
            ids = [cid for cid in ids if cid in pending_retry]
        else:
            retry_mode = "all"

        now = time.time()
        async with self._lock:
            for cid in ids:
                if cid in self._work_seen:
                    continue
                self._work_seen.add(cid)

                # quarantine wins
                try:
                    if self.retry_store.is_quarantined(cid):
                        continue
                except Exception:
                    pass

                # already queued?
                if cid in self._queued:
                    continue

                # backoff
                try:
                    if not self.retry_store.is_eligible(cid, now=now):
                        ts = float(self.retry_store.next_eligible_at(cid))
                        self._enqueue_deferred_locked(cid, ts)
                        continue
                except Exception:
                    # if uncertain, allow it (better than deadlock)
                    pass

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
        """
        Remove retry entries that are already DONE (or not runnable).
        Uses a callback supplied by run.py (usually checks crawl_state snapshot).
        """
        ids = sorted(_retry_pending_ids(self.retry_store))
        if not ids:
            return 0

        cleared = 0
        for cid in ids:
            try:
                runnable = await is_company_runnable(cid)
                if treat_non_runnable_as_done and (not runnable):
                    self.retry_store.mark_success(cid, stage=stage, note="already_done")
                    cleared += 1
            except Exception:
                continue
        return cleared

    # ----------------------------
    # Queue helpers (LOCKED)
    # ----------------------------
    def _enqueue_ready_locked(self, cid: str) -> None:
        if not cid:
            return
        if cid in self._queued:
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

        # Respect the latest schedule (typically later) from retry_state.py
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
            # discard stale heap entries
            if cur is None or abs(float(cur) - float(ts)) > 1e-6:
                heapq.heappop(self._work_deferred)
                continue
            if float(ts) > float(now):
                break

            heapq.heappop(self._work_deferred)
            self._deferred_at.pop(cid, None)

            # quarantine/backoff can change; re-check
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
            "proc_mem_mb": float(self._proc_rss_bytes) / _MB
            if self._proc_rss_bytes
            else 0.0,
            "per_company_est_mb": self._per_company_est_mb,
            "target_parallel": self._target_parallel,
            "completed_counter": self._completed_counter,
            "waiting": self._last_num_waiting,
            "restart_recommended": self._restart_recommended,
            "restart_reason": self._restart_reason,
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
        """
        Decide which company IDs should start now.

        IMPORTANT:
          If free_crawlers <= 0, we start NOTHING. This prevents a churn loop where
          tasks get admitted but cannot lease a crawler slot, then hit inactivity cancels.
        """
        active_ids = list(self._get_active_company_ids())
        active_set = set(active_ids)

        # 1) move deferred due
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
            "[AdaptiveScheduling] admission blocked reason=%s used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s active=%d waiting=%d proc_mem_mb=%.1f target=%d",
            reason,
            self._used_frac_raw,
            self._used_frac_eff,
            self._ramp_raw_frac_per_sec,
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

            active_ids = list(self._get_active_company_ids())
            active_n = len(active_ids)
            if active_n > 0:
                self._ever_had_active = True

            self._update_base_rss_locked(active_n)
            self._update_per_company_est_locked(active_n)
            self._update_target_parallel_locked(active_n)

            # Block conditions
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
                    if active_n == 0
                    and self._used_frac_eff < float(self.cfg.mem_high_frac)
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
            if active_n > 0 and self._ramp_raw_frac_per_sec >= float(
                self.cfg.ramp_admit_guard_frac_per_sec
            ):
                slots = min(
                    slots, max(1, int(self.cfg.max_admit_per_call_when_ramping))
                )

            if slots > 0:
                self._ever_admitted = True
                self._mark_progress()
                self._last_admission_mono = time.monotonic()

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

        now = time.time()
        active_set = set(self._get_active_company_ids())
        items: List[Tuple[str, float]] = []
        for cid in cancel_ids:
            try:
                if self.retry_store.is_quarantined(str(cid)):
                    continue
            except Exception:
                pass

            try:
                ts = float(self.retry_store.next_eligible_at(str(cid)))
            except Exception:
                ts = 0.0

            grace = float(self.cfg.cancel_requeue_min_delay_sec)
            min_ts = now + grace
            if str(cid) in active_set:
                min_ts = now + grace
            ts = max(float(ts or 0.0), float(min_ts))
            items.append((str(cid), ts))

        if not items:
            return

        async with self._lock:
            for cid, ts in items:
                if cid in self._queued:
                    if cid in self._deferred_at:
                        self._enqueue_deferred_locked(cid, ts)
                    continue

                try:
                    if self.retry_store.is_eligible(cid, now=now) and ts <= now:
                        self._enqueue_ready_locked(cid)
                    else:
                        self._enqueue_deferred_locked(cid, ts)
                except Exception:
                    self._enqueue_deferred_locked(cid, ts)

    # ----------------------------
    # Watchdog loop
    # ----------------------------
    async def _watchdog_loop(self) -> None:
        interval = max(0.5, float(self.cfg.sample_interval_sec))
        while True:
            try:
                await asyncio.sleep(interval)
                self._mark_heartbeat()

                if not self._psutil_available:
                    continue

                cancel_ids: List[str] = []
                cancel_reason: str = ""
                recycle_count: int = 0

                async with self._lock:
                    now = time.time()
                    self._sample_memory_locked(now)

                    active_ids = list(self._get_active_company_ids())
                    active_n = len(active_ids)
                    if active_n > 0:
                        self._ever_had_active = True

                    self._update_base_rss_locked(active_n)
                    self._update_per_company_est_locked(active_n)
                    self._update_target_parallel_locked(active_n)

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

                    # Keep touch map tidy; seed active companies
                    active_set = set(active_ids)
                    for cid in list(self._company_last_touch_mono.keys()):
                        if cid not in active_set:
                            self._company_last_touch_mono.pop(cid, None)
                    for cid in active_ids:
                        self._company_last_touch_mono.setdefault(cid, now_mono)

                    # Maintain cancel-inflight map
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

                    # 2) Admission starvation -> restart recommended (gated by stall_enabled)
                    if (
                        stall_enabled
                        and float(self.cfg.admission_starvation_timeout_sec) > 0
                        and (not self._restart_recommended)
                    ):
                        if active_n == 0 and (
                            self._last_num_waiting > 0 or self.has_pending()
                        ):
                            starve_age = now_mono - self._last_admission_mono
                            if starve_age >= float(
                                self.cfg.admission_starvation_timeout_sec
                            ):
                                mem_high = (
                                    self._used_frac_raw
                                    >= float(self.cfg.mem_high_raw_frac)
                                ) or (
                                    self._used_frac_eff >= float(self.cfg.mem_high_frac)
                                )
                                if mem_high and self._can_recommend_restart_normal():
                                    self._recommend_restart_locked(
                                        reason=f"admission_starvation_mem_high starve_age={starve_age:.1f}s used_raw={self._used_frac_raw:.3f} used_eff={self._used_frac_eff:.3f}"
                                    )

                    # 3) NEW: deadlock restart (active==0, waiting>0, mem high) even if completed<min
                    if (
                        self.cfg.deadlock_restart_enabled
                        and (not self._restart_recommended)
                        and active_n == 0
                        and (self._last_num_waiting > 0 or self.has_pending())
                    ):
                        mem_high = (
                            self._used_frac_raw >= float(self.cfg.mem_high_raw_frac)
                        ) or (self._used_frac_eff >= float(self.cfg.mem_high_frac))
                        if (not self.cfg.deadlock_restart_require_mem_high) or mem_high:
                            if self._deadlock_since_mono is None:
                                self._deadlock_since_mono = now_mono
                            dead_age = now_mono - self._deadlock_since_mono
                            if dead_age >= float(self.cfg.deadlock_restart_timeout_sec):
                                # allow early restart: we are doing nothing but stuck
                                self._recommend_restart_locked(
                                    reason=f"deadlock_restart dead_age={dead_age:.1f}s mem_high={mem_high} used_raw={self._used_frac_raw:.3f} used_eff={self._used_frac_eff:.3f} proc_mb={(float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0):.1f}"
                                )
                        else:
                            self._deadlock_since_mono = None
                    else:
                        # reset if not in deadlock state
                        if not (
                            active_n == 0
                            and (self._last_num_waiting > 0 or self.has_pending())
                        ):
                            self._deadlock_since_mono = None

                    # 4) Raw mem kill -> restart (not gated)
                    if (not self._restart_recommended) and self._used_frac_raw >= float(
                        self.cfg.mem_kill_raw_frac
                    ):
                        self._recommend_restart_locked(
                            reason=f"raw_mem_kill used_raw={self._used_frac_raw:.3f} kill={float(self.cfg.mem_kill_raw_frac):.3f}"
                        )

                    # 5) Ramp kill -> restart (not gated)
                    if (
                        (not self._restart_recommended)
                        and self._ramp_raw_frac_per_sec
                        >= float(self.cfg.mem_ramp_kill_frac_per_sec)
                        and self._used_frac_raw >= float(self.cfg.mem_high_raw_frac)
                    ):
                        self._recommend_restart_locked(
                            reason=f"ramp_kill ramp_raw={self._ramp_raw_frac_per_sec:.3f}/s used_raw={self._used_frac_raw:.3f}"
                        )

                    # 6) Emergency cancel (critical memory OR ramp critical) - NOT gated
                    crit_trigger = (
                        self._used_frac_raw >= float(self.cfg.mem_crit_raw_frac)
                    ) or (self._used_frac_eff >= float(self.cfg.mem_crit_frac))

                    ramp_trigger = self._ramp_raw_frac_per_sec >= float(
                        self.cfg.mem_ramp_crit_frac_per_sec
                    ) and self._used_frac_raw >= float(self.cfg.mem_crit_raw_frac)

                    if (
                        (crit_trigger or ramp_trigger)
                        and active_n > 0
                        and not cancel_ids
                    ):
                        if (now - self._last_emergency_cancel_ts) >= float(
                            self.cfg.emergency_cancel_cooldown_sec
                        ):
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
                            to_cancel = min(
                                int(self.cfg.emergency_cancel_max), max_cancelable
                            )
                            if to_cancel > 0 and cancelable:
                                cancel_ids = self._select_cancel_ids(
                                    cancelable, to_cancel
                                )
                                self._last_emergency_cancel_ts = now
                                cancel_reason = "mem_heavy" if ramp_trigger else "mem"

                                logger.error(
                                    "[AdaptiveScheduling] EMERGENCY used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s total_mb=%.1f used_raw_mb=%.1f used_eff_mb=%.1f proc_mem_mb=%.1f active=%d cancel=%s",
                                    self._used_frac_raw,
                                    self._used_frac_eff,
                                    self._ramp_raw_frac_per_sec,
                                    float(self._total_mem_bytes) / _MB,
                                    float(self._used_bytes_raw) / _MB,
                                    float(self._used_bytes_eff) / _MB,
                                    float(self._proc_rss_bytes) / _MB
                                    if self._proc_rss_bytes
                                    else 0.0,
                                    active_n,
                                    cancel_ids,
                                )

                    # 7) Idle recycle request (memory pressure, throttled)
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
                                # If we are truly stuck (active==0 and waiting huge), recycle more.
                                max_n = max(
                                    1, int(self.cfg.idle_recycle_max_per_interval)
                                )
                                if active_n == 0 and (
                                    self._last_num_waiting > 0 or self.has_pending()
                                ):
                                    recycle_count = max_n
                                else:
                                    recycle_count = 1
                                self._last_idle_recycle_mono = now_mono

                    # 8) Restart signal management (send/repeat/escalate) - keep inside lock (cheap ops)
                    if self._restart_recommended:
                        self._maybe_send_restart_signals_locked(now_mono)

                # Outside lock: apply cancels and schedule retries
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
                                f"ramp_raw={self._ramp_raw_frac_per_sec:.3f}/s "
                                f"proc_mb={(float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0):.1f}"
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

                if recycle_count > 0 and self._request_recycle_idle is not None:
                    try:
                        await self._request_recycle_idle(
                            int(recycle_count), "mem_pressure_idle_recycle"
                        )
                    except Exception:
                        logger.exception(
                            "[AdaptiveScheduling] request_recycle_idle failed"
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[AdaptiveScheduling] watchdog loop error")

    # ----------------------------
    # Restart gating / restart action
    # ----------------------------
    def _can_recommend_restart_normal(self) -> bool:
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

    def _recommend_restart_locked(self, *, reason: str) -> None:
        if self._restart_recommended:
            return
        self._restart_recommended = True
        self._restart_reason = reason or "unspecified"
        self._restart_recommended_mono = time.monotonic()
        logger.critical(
            "[AdaptiveScheduling] restart recommended reason=%s", self._restart_reason
        )
        # If configured, send SIGTERM promptly; run.py should exit with the wrapper-friendly code.
        if self.cfg.kill_on_no_progress:
            try:
                os.kill(os.getpid(), int(self.cfg.hard_watchdog_kill_signal))
            except Exception:
                pass
        self._restart_sigterm_sent_mono = time.monotonic()

    def _maybe_send_restart_signals_locked(self, now_mono: float) -> None:
        # Re-send SIGTERM occasionally (some environments swallow / handlers defer shutdown)
        repeat = float(self.cfg.restart_sigterm_repeat_interval_sec)
        if self._restart_sigterm_sent_mono is None:
            self._restart_sigterm_sent_mono = now_mono
            try:
                os.kill(os.getpid(), int(self.cfg.hard_watchdog_kill_signal))
            except Exception:
                pass
            return

        if repeat > 0 and (now_mono - self._restart_sigterm_sent_mono) >= repeat:
            self._restart_sigterm_sent_mono = now_mono
            try:
                os.kill(os.getpid(), int(self.cfg.hard_watchdog_kill_signal))
            except Exception:
                pass

        # Escalate to SIGKILL as last resort if we're still alive too long after restart recommendation.
        esc = float(self.cfg.restart_escalate_sigkill_after_sec)
        if esc > 0 and self._restart_recommended_mono is not None:
            if (now_mono - self._restart_recommended_mono) >= esc:
                try:
                    os.kill(os.getpid(), 9)  # SIGKILL
                except Exception:
                    pass

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
                "restart_reason": self._restart_reason,
                "used_frac": self._used_frac_eff,
                "used_frac_raw": self._used_frac_raw,
                "ramp_raw_frac_per_sec": self._ramp_raw_frac_per_sec,
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
                return limit, used_effective, used_raw_bytes

        except Exception:
            pass
        return 0, 0, 0

    def _read_psutil_memory_bytes(self) -> Tuple[int, int, int]:
        if not self._psutil_available or psutil is None:
            return 0, 0, 0
        try:
            vm = psutil.virtual_memory()
            total = int(getattr(vm, "total", 0) or 0)
            available = int(getattr(vm, "available", 0) or 0)
            used = max(0, total - available)
            # psutil does not expose inactive_file cleanly; treat eff==raw
            return total, used, used
        except Exception:
            return 0, 0, 0

    def _read_memory_usage_bytes(self) -> Tuple[int, int, int]:
        if self.cfg.prefer_cgroup_limits:
            limit, used_eff, used_raw = self._read_cgroup_memory_bytes()
            # If cgroup limit is unknown, fallback to psutil for total.
            if limit <= 0:
                total, _, _ = self._read_psutil_memory_bytes()
                return total, used_eff, used_raw
            return limit, used_eff, used_raw
        return self._read_psutil_memory_bytes()

    def _read_process_tree_rss_bytes(self) -> int:
        if not self._psutil_available or psutil is None:
            return 0
        try:
            p = psutil.Process(os.getpid())
            rss = int(p.memory_info().rss)
            if not self.cfg.use_process_tree_rss_for_estimate:
                return rss
            for ch in p.children(recursive=True):
                try:
                    rss += int(ch.memory_info().rss)
                except Exception:
                    continue
            return max(0, int(rss))
        except Exception:
            return 0

    def _sample_memory_locked(self, now_ts: float) -> None:
        total, used_eff, used_raw = self._read_memory_usage_bytes()
        if total > 0:
            self._total_mem_bytes = int(total)
        self._used_bytes_eff = int(used_eff)
        self._used_bytes_raw = int(used_raw)

        total2 = float(self._total_mem_bytes) if self._total_mem_bytes > 0 else 0.0
        self._used_frac_eff = (
            (float(self._used_bytes_eff) / total2) if total2 > 0 else 0.0
        )
        self._used_frac_raw = (
            (float(self._used_bytes_raw) / total2) if total2 > 0 else 0.0
        )

        self._proc_rss_bytes = self._read_process_tree_rss_bytes()
        self._last_sample_ts = float(now_ts)

        self._update_ramp_locked(
            time.monotonic(), self._used_frac_eff, self._used_frac_raw
        )

    def _update_ramp_locked(
        self, now_mono: float, frac_eff: float, frac_raw: float
    ) -> None:
        self._mem_frac_hist.append((now_mono, float(frac_eff), float(frac_raw)))
        window = float(self.cfg.mem_ramp_window_sec)
        if window <= 0:
            self._ramp_eff_frac_per_sec = 0.0
            self._ramp_raw_frac_per_sec = 0.0
            return

        # Find oldest sample within window
        t0 = None
        fe0 = None
        fr0 = None
        for t, fe, fr in reversed(self._mem_frac_hist):
            if (now_mono - t) >= window:
                t0, fe0, fr0 = t, fe, fr
                break

        if t0 is None or fe0 is None or fr0 is None:
            self._ramp_eff_frac_per_sec = 0.0
            self._ramp_raw_frac_per_sec = 0.0
            return

        dt = max(1e-6, float(now_mono - t0))
        self._ramp_eff_frac_per_sec = float(frac_eff - fe0) / dt
        self._ramp_raw_frac_per_sec = float(frac_raw - fr0) / dt

    def _update_base_rss_locked(self, active_n: int) -> None:
        # Baseline RSS observed when no companies are active.
        if active_n != 0:
            return
        rss = float(self._proc_rss_bytes)
        if rss <= 0:
            return
        # Smooth baseline (EMA-ish): bias toward lower values, but adapt slowly upward
        if self._base_rss_samples <= 0 or self._base_rss_bytes <= 0:
            self._base_rss_bytes = rss
            self._base_rss_samples = 1
            return
        alpha = 0.10  # slow smoothing
        # If rss is lower, move faster down; if higher, move slower up
        if rss < self._base_rss_bytes:
            alpha = 0.25
        self._base_rss_bytes = (1.0 - alpha) * self._base_rss_bytes + alpha * rss
        self._base_rss_samples += 1

    def _update_per_company_est_locked(self, active_n: int) -> None:
        min_mb = float(self.cfg.per_company_min_mb)
        max_mb = float(self.cfg.per_company_max_mb)

        if active_n <= 0:
            # decay toward min when idle
            self._per_company_est_mb = max(
                min_mb, min(self._per_company_est_mb, max_mb)
            )
            return

        rss = float(self._proc_rss_bytes)
        base = float(self._base_rss_bytes) if self._base_rss_bytes > 0 else 0.0
        extra = max(0.0, rss - base)
        per = (extra / float(active_n)) / _MB if active_n > 0 else min_mb
        per = max(min_mb, min(per, max_mb))
        self._per_company_est_mb = per

    def _per_company_reservation_bytes_locked(self) -> int:
        sf = max(1.0, float(self.cfg.per_company_safety_factor))
        mb = max(1.0, float(self._per_company_est_mb))
        return int(mb * sf * _MB)

    def _update_target_parallel_locked(self, active_n: int) -> None:
        old = int(self._target_parallel)
        tgt = old

        used_raw = float(self._used_frac_raw)
        used_eff = float(self._used_frac_eff)
        ramp_raw = float(self._ramp_raw_frac_per_sec)

        # Multiplicative decrease on high memory or critical ramp
        should_decrease = (
            used_raw >= float(self.cfg.mem_high_raw_frac)
            or used_eff >= float(self.cfg.mem_high_frac)
            or (active_n > 0 and ramp_raw >= float(self.cfg.mem_ramp_high_frac_per_sec))
        )
        if should_decrease and tgt > int(self.cfg.min_target):
            tgt = max(
                int(self.cfg.min_target),
                int(max(1, int(tgt * float(self.cfg.md_factor)))),
            )
        else:
            # Additive increase when comfortable
            comfortable = (
                used_eff < float(self.cfg.mem_cap_frac)
                and used_raw < float(self.cfg.mem_high_raw_frac) * 0.98
                and ramp_raw < float(self.cfg.ramp_admit_guard_frac_per_sec)
            )
            if comfortable and tgt < int(self.cfg.max_target):
                tgt = min(int(self.cfg.max_target), tgt + int(self.cfg.ai_step))

        self._target_parallel = max(
            int(self.cfg.min_target), min(int(tgt), int(self.cfg.max_target))
        )

        if self._target_parallel != old:
            logger.info(
                "[AdaptiveScheduling] target_parallel %d -> %d (used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s active=%d proc_mem_mb=%.1f)",
                old,
                self._target_parallel,
                used_raw,
                used_eff,
                ramp_raw,
                int(active_n),
                float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0,
            )

    # ----------------------------
    # Hard watchdog thread (optional)
    # ----------------------------
    def _start_hard_watchdog_thread(self) -> None:
        if self._hard_thread is not None:
            return
        self._hard_stop.clear()

        def _thread_main() -> None:
            start = time.monotonic()
            pid = os.getpid()
            hb_path = self.cfg.heartbeat_path
            interval = max(1.0, float(self.cfg.hard_watchdog_interval_sec))
            grace = max(0.0, float(self.cfg.hard_watchdog_startup_grace_sec))
            timeout = max(5.0, float(self.cfg.hard_watchdog_no_heartbeat_timeout_sec))
            sigterm = int(self.cfg.hard_watchdog_kill_signal)

            while not self._hard_stop.is_set():
                time.sleep(interval)
                try:
                    if (time.monotonic() - start) < grace:
                        continue
                    if hb_path is None or (not Path(hb_path).exists()):
                        continue
                    try:
                        raw = Path(hb_path).read_text(encoding="utf-8")
                        data = json.loads(raw) if raw else {}
                        ts = float(data.get("mono", 0.0) or 0.0)
                    except Exception:
                        ts = 0.0
                    if ts <= 0:
                        continue
                    age = time.monotonic() - ts
                    if age >= timeout:
                        try:
                            os.kill(pid, sigterm)
                        except Exception:
                            pass
                        # If still alive much later, SIGKILL.
                        try:
                            time.sleep(min(10.0, interval))
                            os.kill(pid, 9)
                        except Exception:
                            pass
                except Exception:
                    continue

        self._hard_thread = threading.Thread(
            target=_thread_main, name="adaptive-scheduling-hard-watchdog", daemon=True
        )
        self._hard_thread.start()

    def _stop_hard_watchdog_thread(self) -> None:
        self._hard_stop.set()
        t = self._hard_thread
        self._hard_thread = None
        if t is not None:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
