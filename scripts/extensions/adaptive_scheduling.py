from __future__ import annotations

import asyncio
import contextlib
import gc
import heapq
import json
import logging
import os
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

import psutil

from .retry import RetryStateStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MB = 1_000_000


# --------------------------------------------------------------------------------------
# Retry-store helpers (NO DISK READS HERE)
# --------------------------------------------------------------------------------------
def compute_retry_exit_code_from_store(
    store: RetryStateStore, retry_exit_code: int
) -> int:
    """
    Return retry_exit_code only if there exists at least one *eligible now* retry-pending company.
    Prevents tight restart loops when next_eligible_at is in the future.

    IMPORTANT: uses store APIs only (no JSON reads).
    """
    now = time.time()
    eligible = store.pending_eligible_total(now=now)
    return int(retry_exit_code) if eligible > 0 else 0


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
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

    # --- Deadlock restart (active==0 + waiting>0 + mem_high for long) ---
    deadlock_restart_enabled: bool = True
    deadlock_restart_timeout_sec: float = 300.0
    deadlock_restart_require_mem_high: bool = True

    # --- Stall detection (progress-based; heartbeat doesn't count) ---
    company_inactivity_timeout_sec: float = 240.0
    company_inactivity_min_runtime_sec: float = 90.0
    company_inactivity_cancel_max: int = 1

    admission_starvation_timeout_sec: float = 480.0
    block_log_interval_sec: float = 60.0

    # --- Stall storm throttling ---
    stall_storm_window_sec: float = 300.0
    stall_storm_cancel_threshold: int = 10
    stall_storm_pause_start_sec: float = 60.0
    stall_storm_start_limit: int = 1
    stall_rising_window_sec: float = 60.0
    stall_rising_cancel_threshold: int = 3
    stall_rising_start_limit: int = 2

    stall_storm_restart_enabled: bool = True
    stall_storm_restart_threshold: int = 25
    stall_storm_restart_min_uptime_sec: float = 120.0

    # Restart if stall-storm AND no global progress for long
    stall_storm_restart_no_progress_sec: float = 600.0  # 10 minutes
    stall_storm_restart_require_waiting: bool = True  # only if still work remains

    # Optional heartbeat/logging
    heartbeat_path: Optional[Path] = None
    log_path: Optional[Path] = None

    # Prefer cgroup limits if available
    prefer_cgroup_limits: bool = True
    cgroup_subtract_inactive_file: bool = True

    # --- Restart escalation (when restart recommended) ---
    restart_sigterm_repeat_interval_sec: float = 30.0
    restart_escalate_sigkill_after_sec: float = 120.0

    # --- Work scheduling knobs (scheduler owns queueing) ---
    max_start_per_tick: int = 3
    crawler_capacity_multiplier: int = 3  # allow N companies per free crawler slot

    idle_recycle_interval_sec: float = 25.0
    idle_recycle_raw_frac: float = 0.88
    idle_recycle_eff_frac: float = 0.83
    idle_recycle_max_per_interval: int = 2

    # Retry state location (scheduler owns store)
    retry_base_dir: Optional[Path] = None

    # Backoff sleep smoothing
    min_idle_sleep_sec: float = 0.25
    # IMPORTANT: increase to reduce “loop tick active_n=0…” churn when only deferred remain
    max_idle_sleep_sec: float = 30.0

    # When the scheduler cancels a company (stall/mem), do not re-admit it immediately.
    cancel_requeue_min_delay_sec: float = 2.0

    # Cancel spam guard
    company_cancel_repeat_cooldown_sec: float = 15.0
    company_cancel_inflight_timeout_sec: float = 600.0

    # Signal used when recommending restart
    restart_signal: int = 15  # SIGTERM

    # --- Doomed repeat / GOTO-timeout convergence controls ---
    # After N identical failures (tracked by retry store) of kind "goto", clamp admission with a long delay.
    doomed_goto_same_error_streak_threshold: int = 2
    doomed_goto_min_cooldown_sec: float = 1800.0  # 30 minutes
    # If only 1 company remains and it is doomed, optionally quarantine it immediately (best-effort).
    last_one_standing_quarantine_enabled: bool = True
    # Throttle deferred->ready log spam (especially when 1 item cycles)
    deferred_move_log_interval_sec: float = 30.0


class AdaptiveScheduler:
    """
    Stable API (used by run.py):
      - await start()/stop()
      - retry_store property
      - await set_worklist(...)
      - await plan_start_batch(free_crawlers=...) -> list[str]
      - has_pending(), pending_total(), pending_ready()
      - sleep_hint_sec(), initial_total_hint()
      - register_company_completed()
      - await cleanup_completed_retry_ids(...)
      - suspend_stall_detection()/resume_stall_detection()
      - get_state_snapshot()
      - touch_company()/heartbeat_company()/progress_company()
      - await requeue_company(company_id, delay_sec=..., reason=...)
    """

    def __init__(
        self,
        cfg: AdaptiveSchedulingConfig,
        get_active_company_ids: Callable[[], Sequence[str]],
        request_cancel_companies: Callable[[Sequence[str]], None],
        request_recycle_idle: Optional[Callable[[int, str], Awaitable[int]]] = None,
    ) -> None:
        self.cfg = cfg

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

        # log throttles
        self._last_deferred_move_log_mono: float = 0.0

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

        # Company activity tracking (PROGRESS-BASED)
        self._company_last_heartbeat_mono: Dict[str, float] = {}
        self._company_last_progress_mono: Dict[str, float] = {}
        self._company_started_mono: Dict[str, float] = {}

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

        # Heartbeats/progress (GLOBAL)
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

        # Stall storm tracking + dynamic throttles
        self._stall_cancel_events_mono: Deque[float] = deque(maxlen=4096)
        self._pause_starts_until_mono: float = 0.0

        # Async watchdog
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None

        self._last_idle_recycle_mono: float = 0.0

    # ----------------------------
    # Retry state introspection (best-effort, backward compatible)
    # ----------------------------
    def _get_retry_state(self, cid: str) -> Optional[Any]:
        """
        Best-effort getter for a per-company retry state.
        We avoid disk reads; this assumes RetryStateStore keeps an in-memory index/cache.
        If the store doesn't expose a getter, return None and skip doomed gating.
        """
        if not cid:
            return None

        # Common patterns across versions: get(), state(), get_state(), read()
        for name in ("get", "get_state", "state", "read_state"):
            fn = getattr(self.retry_store, name, None)
            if callable(fn):
                try:
                    return fn(cid)  # type: ignore[misc]
                except TypeError:
                    try:
                        return fn(company_id=cid)  # type: ignore[misc]
                    except Exception:
                        return None
                except Exception:
                    return None
        return None

    def _is_doomed_repeat_goto(self, cid: str) -> Tuple[bool, str]:
        """
        Detect "doomed" companies (repeat identical goto timeouts) via retry store.
        Prefer a canonical store helper if present (retry.py plan A3), else fall back to
        best-effort field probing for backward compatibility.

        Returns (is_doomed, details).
        """
        thr = max(1, int(self.cfg.doomed_goto_same_error_streak_threshold))

        # Prefer canonical helper if available
        helper = getattr(self.retry_store, "is_doomed_repeat_goto", None)
        if callable(helper):
            try:
                ok = bool(helper(company_id=cid, threshold=thr))  # type: ignore[misc]
                return (ok, f"store_helper thr={thr}") if ok else (False, "")
            except TypeError:
                # maybe positional
                try:
                    ok = bool(helper(cid, thr))  # type: ignore[misc]
                    return (ok, f"store_helper thr={thr}") if ok else (False, "")
                except Exception:
                    pass
            except Exception:
                pass

        st = self._get_retry_state(cid)
        if st is None:
            return False, ""

        same_streak = int(getattr(st, "same_error_streak", 0) or 0)
        last_kind = (
            getattr(st, "last_stall_kind", None)
            or getattr(st, "stall_kind", None)
            or getattr(st, "stall_kind_hint", None)
        )

        # If retry.py A2 is implemented, this will be stable:
        last_sig = (
            getattr(st, "last_error_sig", None)
            or getattr(st, "error_sig", None)
            or getattr(st, "last_sig", None)
        )

        try:
            last_kind_s = (
                (str(last_kind) if last_kind is not None else "").strip().lower()
            )
        except Exception:
            last_kind_s = ""

        try:
            last_sig_s = (str(last_sig) if last_sig is not None else "").strip().lower()
        except Exception:
            last_sig_s = ""

        if same_streak >= thr and last_kind_s == "goto":
            # Prefer signature match if available; otherwise allow older stores that don’t have sig
            if (not last_sig_s) or (last_sig_s == "goto_timeout"):
                return (
                    True,
                    f"repeat_goto_timeout same_error_streak={same_streak} thr={thr} sig={last_sig_s or 'unknown'}",
                )
        return False, ""

    def _try_quarantine_company(self, cid: str, *, reason: str) -> bool:
        """
        Best-effort quarantine call.

        IMPORTANT FIX (B1):
          RetryStateStore.quarantine_company() requires a richer signature in newer versions:
            quarantine_company(company_id, reason, stage, error, ...)
          Adaptive previously called it with the WRONG signature and silently failed.
        """
        fn = getattr(self.retry_store, "quarantine_company", None)
        if not callable(fn):
            return False

        # best-effort; satisfy required args
        try:
            fn(
                company_id=cid,
                reason=str(reason),
                stage="scheduler_doomed",
                error=str(reason),
                cls="permanent",
                also_mark_terminal_done=True,
                flush=True,
            )  # type: ignore[misc]
            return True
        except TypeError:
            # Try a minimal keyword set (some versions don’t accept extras)
            try:
                fn(
                    company_id=cid,
                    reason=str(reason),
                    stage="scheduler_doomed",
                    error=str(reason),
                    flush=True,
                )  # type: ignore[misc]
                return True
            except TypeError:
                # Try positional legacy variants
                try:
                    fn(cid, str(reason), "scheduler_doomed", str(reason))  # type: ignore[misc]
                    return True
                except Exception:
                    return False
            except Exception:
                return False
        except Exception:
            return False

    def _only_one_left_now(self) -> bool:
        """
        Helper for "last one standing": no active, and (ready+deferred) <= 1.
        """
        try:
            active_n = len(self._get_active_company_ids())
        except Exception:
            active_n = 0
        return (active_n == 0) and (self.pending_total() <= 1)

    # ----------------------------
    # Worklist APIs
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
        """
        Seed scheduler-owned queue.

        retry_mode:
          - all: include all IDs
          - skip-retry: exclude IDs currently pending in retry store
          - only-retry: include only IDs pending in retry store
        """
        self._is_company_runnable = is_company_runnable
        ids = [str(x) for x in company_ids if str(x).strip()]
        self._work_total_hint = len(ids)

        pending_retry = set(self.retry_store.pending_ids(exclude_quarantined=True))

        rm = (retry_mode or "all").strip().lower()
        if rm == "skip-retry":
            ids = [cid for cid in ids if cid not in pending_retry]
        elif rm == "only-retry":
            ids = [cid for cid in ids if cid in pending_retry]
        else:
            rm = "all"

        now = time.time()
        async with self._lock:
            for cid in ids:
                if cid in self._work_seen:
                    continue
                self._work_seen.add(cid)

                if self.retry_store.is_quarantined(cid):
                    continue
                if cid in self._queued:
                    continue

                if not self.retry_store.is_eligible(cid, now=now):
                    ts = float(self.retry_store.next_eligible_at(cid))
                    self._enqueue_deferred_locked(cid, ts)
                    continue

                self._enqueue_ready_locked(cid)

        logger.info(
            "[AdaptiveScheduling] seeded worklist ids=%d retry_mode=%s ready=%d deferred=%d",
            len(ids),
            rm,
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

        cleared = 0
        for cid in ids:
            runnable = await is_company_runnable(cid)
            if treat_non_runnable_as_done and (not runnable):
                self.retry_store.mark_success(cid, stage=stage, note="already_done")
                cleared += 1
        return cleared

    async def requeue_company(
        self,
        company_id: str,
        *,
        delay_sec: float = 0.0,
        reason: str = "",
        force: bool = False,
    ) -> bool:
        """
        IMPORTANT (B2):
          Gate *every* re-admission path against doomed repeat-goto.
          - If last-one-standing -> quarantine immediately (best-effort).
          - Else -> push into deferred with big cooldown and NEVER into ready on this call.
        """
        cid = (company_id or "").strip()
        if not cid:
            return False
        if self.retry_store.is_quarantined(cid):
            return False

        now = time.time()
        ts = now + max(0.0, float(delay_sec))

        # Doomed gating (early)
        doomed, details = self._is_doomed_repeat_goto(cid)
        if doomed:
            if (
                bool(self.cfg.last_one_standing_quarantine_enabled)
                and self._only_one_left_now()
            ):
                quarantined = self._try_quarantine_company(
                    cid,
                    reason=f"scheduler_last_one_standing_{details or 'repeat_goto_timeout'}",
                )
                if quarantined:
                    logger.warning(
                        "[AdaptiveScheduling] quarantined last-one-standing on requeue cid=%s (%s)",
                        cid,
                        details or "repeat_goto_timeout",
                    )
                    return True
            cooldown = max(0.0, float(self.cfg.doomed_goto_min_cooldown_sec))
            ts = max(
                ts,
                now + cooldown,
                float(self.retry_store.next_eligible_at(cid) or 0.0),
            )
            async with self._lock:
                self._enqueue_deferred_locked(cid, ts)
            if reason:
                logger.warning(
                    "[AdaptiveScheduling] requeue deferred doomed cid=%s delay=%.0fs reason=%s (%s)",
                    cid,
                    max(0.0, ts - now),
                    reason,
                    details or "repeat_goto_timeout",
                )
            else:
                logger.warning(
                    "[AdaptiveScheduling] requeue deferred doomed cid=%s delay=%.0fs (%s)",
                    cid,
                    max(0.0, ts - now),
                    details or "repeat_goto_timeout",
                )
            self._mark_progress()
            return True

        if not force and not self.retry_store.is_eligible(cid, now=now):
            ts = max(ts, float(self.retry_store.next_eligible_at(cid)))

        async with self._lock:
            if cid in self._queued:
                if cid in self._deferred_at:
                    self._enqueue_deferred_locked(cid, ts)
                return True

            if cid in set(self._get_active_company_ids()):
                ts = max(ts, now + float(self.cfg.cancel_requeue_min_delay_sec))
                self._enqueue_deferred_locked(cid, ts)
                return True

            if force or (self.retry_store.is_eligible(cid, now=now) and ts <= now):
                self._enqueue_ready_locked(cid)
            else:
                self._enqueue_deferred_locked(cid, ts)

        self._mark_progress()
        if reason:
            logger.info(
                "[AdaptiveScheduling] requeued cid=%s delay=%.2fs reason=%s",
                cid,
                max(0.0, ts - now),
                reason,
            )
        return True

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
        """
        IMPORTANT:
          Also apply doomed repeat-goto gating here so that "eligible_at reached"
          doesn't immediately move a doomed company into ready again.
        """
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

            if self.retry_store.is_quarantined(cid):
                self._queued.discard(cid)
                continue

            # Still not eligible? push back.
            if not self.retry_store.is_eligible(cid, now=now):
                ts2 = float(self.retry_store.next_eligible_at(cid))
                self._enqueue_deferred_locked(cid, ts2)
                continue

            # Doomed gating on deferred->ready move
            doomed, details = self._is_doomed_repeat_goto(cid)
            if doomed:
                if (
                    bool(self.cfg.last_one_standing_quarantine_enabled)
                    and self._only_one_left_now()
                ):
                    quarantined = self._try_quarantine_company(
                        cid,
                        reason=f"scheduler_last_one_standing_{details or 'repeat_goto_timeout'}",
                    )
                    if quarantined:
                        logger.warning(
                            "[AdaptiveScheduling] quarantined last-one-standing on deferred->ready cid=%s (%s)",
                            cid,
                            details or "repeat_goto_timeout",
                        )
                        self._queued.discard(cid)
                        continue
                cooldown = max(0.0, float(self.cfg.doomed_goto_min_cooldown_sec))
                ts3 = max(
                    float(now) + cooldown,
                    float(self.retry_store.next_eligible_at(cid) or 0.0),
                )
                self._enqueue_deferred_locked(cid, ts3)
                logger.warning(
                    "[AdaptiveScheduling] deferred doomed on move cid=%s delay=%.0fs (%s)",
                    cid,
                    max(0.0, ts3 - float(now)),
                    details or "repeat_goto_timeout",
                )
                continue

            self._work_ready.append(cid)
            moved += 1

        if moved:
            now_mono = time.monotonic()
            interval = max(0.0, float(self.cfg.deferred_move_log_interval_sec))
            if (
                interval <= 0
                or (now_mono - self._last_deferred_move_log_mono) >= interval
            ):
                self._last_deferred_move_log_mono = now_mono
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
                    self._company_last_progress_mono[company_id] = now_mono
                    self._company_last_heartbeat_mono[company_id] = now_mono
                    self._company_started_mono.setdefault(company_id, now_mono)
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
                    self._company_last_progress_mono[company_id] = now_mono
                    self._company_last_heartbeat_mono[company_id] = now_mono
                    self._company_started_mono.setdefault(company_id, now_mono)
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
    # Company progress hooks (PROGRESS-BASED)
    # ----------------------------
    def touch_company(self, company_id: str, *, kind: str = "progress") -> None:
        """
        kind="heartbeat" MUST NOT reset progress timers.
        """
        if not company_id:
            return
        k = (kind or "progress").strip().lower()
        now_mono = time.monotonic()

        self._company_last_heartbeat_mono[company_id] = now_mono
        self._company_started_mono.setdefault(company_id, now_mono)

        if k in ("heartbeat", "hb"):
            self._mark_heartbeat()
            return

        self._company_last_progress_mono[company_id] = now_mono
        self._mark_progress()

    def heartbeat_company(self, company_id: str) -> None:
        self.touch_company(company_id, kind="heartbeat")

    def progress_company(self, company_id: str) -> None:
        self.touch_company(company_id, kind="progress")

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
        """
        IMPORTANT (B4):
          Stop must not raise CancelledError.
        """
        t = self._watchdog_task
        self._watchdog_task = None
        if t is not None:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    # ----------------------------
    # State snapshot
    # ----------------------------
    def get_state_snapshot(self) -> Dict[str, Any]:
        now_mono = time.monotonic()
        stall_window = float(self.cfg.stall_storm_window_sec)
        stall_cnt = self._count_stall_cancels_since(now_mono - stall_window)
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
            "stall_cancels_window": stall_cnt,
            "pause_starts_for_sec": max(0.0, self._pause_starts_until_mono - now_mono),
        }

    # ----------------------------
    # Core: planning start batch
    # ----------------------------
    async def plan_start_batch(self, *, free_crawlers: int) -> List[str]:
        active_ids = list(self._get_active_company_ids())
        active_set = set(active_ids)
        active_n = len(active_ids)

        async with self._lock:
            self._move_due_deferred_locked(time.time())

        if not self._work_ready:
            return []

        async with self._lock:
            now_mono = time.monotonic()
            if now_mono < float(self._pause_starts_until_mono):
                return []
            dynamic_start_limit = int(self._dynamic_max_start_per_tick_locked(now_mono))
            target_parallel = int(self._target_parallel)

        free = int(free_crawlers)
        if free <= 0:
            return []

        mult = int(self.cfg.crawler_capacity_multiplier)
        if mult <= 0:
            return []

        slots = await self.admissible_slots(num_waiting=len(self._work_ready))
        if slots <= 0:
            return []

        allowed_by_target = max(0, int(target_parallel) - int(active_n))
        if allowed_by_target <= 0:
            return []

        hard_capacity = max(0, int(self.cfg.max_target) - int(active_n))
        allowed_by_target = min(int(allowed_by_target), int(hard_capacity))
        if allowed_by_target <= 0:
            return []

        allowed_by_crawlers = max(0, int(free) * int(mult))
        if allowed_by_crawlers <= 0:
            return []

        allowed = min(
            int(slots),
            int(allowed_by_target),
            int(allowed_by_crawlers),
            int(self.cfg.max_start_per_tick),
            int(self.cfg.max_admit_per_call),
            int(dynamic_start_limit),
        )
        allowed = max(0, int(allowed))
        if allowed <= 0:
            return []

        picked: List[str] = []
        scanned = 0
        max_scan = max(allowed * 4, allowed + 8)
        now = time.time()

        while len(picked) < allowed and scanned < max_scan:
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
                runnable = await self._is_company_runnable(cid)
                if not runnable:
                    self.retry_store.mark_success(
                        cid, stage="scheduler_skip", note="already_done_or_terminal"
                    )
                    continue

            # --- Doomed repeat-goto gating (prevents re-admitting the same doomed company immediately) ---
            doomed, details = self._is_doomed_repeat_goto(cid)
            if doomed:
                only_one_left = (active_n == 0) and (self.pending_total() <= 1)
                if (
                    bool(self.cfg.last_one_standing_quarantine_enabled)
                    and only_one_left
                ):
                    quarantined = self._try_quarantine_company(
                        cid,
                        reason=f"scheduler_last_one_standing_{details or 'repeat_goto_timeout'}",
                    )
                    if quarantined:
                        logger.warning(
                            "[AdaptiveScheduling] quarantined last-one-standing cid=%s (%s)",
                            cid,
                            details or "repeat_goto_timeout",
                        )
                        continue  # do not pick

                cooldown = max(0.0, float(self.cfg.doomed_goto_min_cooldown_sec))
                ts = max(
                    now + cooldown, float(self.retry_store.next_eligible_at(cid) or 0.0)
                )
                async with self._lock:
                    self._enqueue_deferred_locked(cid, ts)
                logger.warning(
                    "[AdaptiveScheduling] deferred doomed cid=%s delay=%.0fs (%s)",
                    cid,
                    max(0.0, ts - now),
                    details or "repeat_goto_timeout",
                )
                continue

            picked.append(cid)

        if picked:
            self._ever_admitted = True
            self._last_admission_mono = time.monotonic()
            self._mark_progress()

            now_mono2 = time.monotonic()
            async with self._lock:
                for cid in picked:
                    self._company_started_mono.setdefault(cid, now_mono2)
                    self._company_last_heartbeat_mono.setdefault(cid, now_mono2)
                    # DO NOT set last_progress here.

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
            "[AdaptiveScheduling] admission blocked reason=%s used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s "
            "active=%d waiting=%d proc_mem_mb=%.1f target=%d pause_starts=%.1fs",
            reason,
            self._used_frac_raw,
            self._used_frac_eff,
            self._ramp_raw_frac_per_sec,
            active_n,
            waiting,
            float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0,
            self._target_parallel,
            max(0.0, self._pause_starts_until_mono - now_mono),
        )

    async def admissible_slots(self, num_waiting: int) -> int:
        self._last_num_waiting = max(0, int(num_waiting))
        self._mark_heartbeat()

        if num_waiting <= 0:
            return 0

        async with self._lock:
            now_mono = time.monotonic()
            if now_mono < float(self._pause_starts_until_mono):
                self._maybe_log_block(
                    "block_stall_pause",
                    len(self._get_active_company_ids()),
                    num_waiting,
                )
                return 0

            now = time.time()
            if (now - self._last_sample_ts) >= float(self.cfg.sample_interval_sec):
                self._sample_memory_locked(now)

            active_ids = list(self._get_active_company_ids())
            active_n = len(active_ids)
            if active_n > 0:
                self._ever_had_active = True

            self._update_base_rss_locked(active_n)
            self._update_per_company_est_locked(active_n)
            self._update_target_parallel_locked(active_n, now_mono=now_mono)

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
                slots = 0
                self._maybe_log_block("block_mem_cap", active_n, num_waiting)
            else:
                per_company_bytes = self._per_company_reservation_bytes_locked()
                slots_by_mem = (
                    int(headroom // per_company_bytes) if per_company_bytes > 0 else 0
                )
                slots_by_target = max(0, int(self._target_parallel) - active_n)
                slots = min(int(num_waiting), int(slots_by_mem), int(slots_by_target))

            slots = max(0, int(slots))
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
    # Retry scheduling
    # ----------------------------
    async def _schedule_retries(
        self,
        company_ids: Sequence[str],
        *,
        record_failure: bool,
        cls_hint: str,
        error: str,
        stage: str,
        status_code: Optional[int] = None,
        permanent_reason: str = "",
    ) -> None:
        if not company_ids:
            return

        if record_failure:
            for cid in company_ids:
                self.retry_store.mark_failure(
                    company_id=str(cid),
                    cls=str(cls_hint),
                    error=str(error),
                    stage=str(stage),
                    status_code=status_code,
                    nxdomain_like=False,
                    stall_kind_hint=None,
                    md_done=None,
                    override_allow=False,
                    flush=True,
                )

        now = time.time()
        now_mono = time.monotonic()
        active_set = set(self._get_active_company_ids())
        items: List[Tuple[str, float]] = []

        grace = float(self.cfg.cancel_requeue_min_delay_sec)
        repeat_cooldown = float(self.cfg.company_cancel_repeat_cooldown_sec)

        for cid in company_ids:
            cid = str(cid)
            if self.retry_store.is_quarantined(cid):
                continue

            ts = float(self.retry_store.next_eligible_at(cid))
            min_ts = now + max(0.0, grace)

            if cid in active_set:
                min_ts = max(min_ts, now + max(0.0, grace))

            last_cancel_mono = self._cancel_inflight_mono.get(cid)
            if last_cancel_mono is not None and repeat_cooldown > 0:
                age = max(0.0, now_mono - float(last_cancel_mono))
                remaining = max(0.0, repeat_cooldown - age)
                min_ts = max(min_ts, now + remaining)

            ts = max(float(ts or 0.0), float(min_ts))

            # clamp for doomed repeat-goto (if state exists)
            doomed, _details = self._is_doomed_repeat_goto(cid)
            if doomed:
                ts = max(
                    ts, now + max(0.0, float(self.cfg.doomed_goto_min_cooldown_sec))
                )

            items.append((cid, ts))

        if not items:
            return

        async with self._lock:
            for cid, ts in items:
                if cid in self._queued:
                    if cid in self._deferred_at:
                        self._enqueue_deferred_locked(cid, ts)
                    continue

                if self.retry_store.is_eligible(cid, now=now) and ts <= now:
                    self._enqueue_ready_locked(cid)
                else:
                    self._enqueue_deferred_locked(cid, ts)

    # ----------------------------
    # Stall storm helpers (LOCKED)
    # ----------------------------
    def _prune_stall_events_locked(self, now_mono: float) -> None:
        window = max(1.0, float(self.cfg.stall_storm_window_sec))
        cutoff = now_mono - window
        while (
            self._stall_cancel_events_mono
            and self._stall_cancel_events_mono[0] < cutoff
        ):
            self._stall_cancel_events_mono.popleft()

    def _count_stall_cancels_since(self, cutoff_mono: float) -> int:
        return sum(1 for t in self._stall_cancel_events_mono if t >= cutoff_mono)

    def _note_stall_cancel_locked(self, now_mono: float) -> None:
        self._stall_cancel_events_mono.append(now_mono)
        self._prune_stall_events_locked(now_mono)

        window = max(1.0, float(self.cfg.stall_storm_window_sec))
        storm_n = self._count_stall_cancels_since(now_mono - window)
        if storm_n >= int(self.cfg.stall_storm_cancel_threshold):
            pause = max(0.0, float(self.cfg.stall_storm_pause_start_sec))
            self._pause_starts_until_mono = max(
                self._pause_starts_until_mono, now_mono + pause
            )

    def _dynamic_max_start_per_tick_locked(self, now_mono: float) -> int:
        if now_mono < float(self._pause_starts_until_mono):
            return 0

        storm_window = max(1.0, float(self.cfg.stall_storm_window_sec))
        storm_n = self._count_stall_cancels_since(now_mono - storm_window)
        if storm_n >= int(self.cfg.stall_storm_cancel_threshold):
            return max(0, int(self.cfg.stall_storm_start_limit))

        rising_window = max(1.0, float(self.cfg.stall_rising_window_sec))
        rising_n = self._count_stall_cancels_since(now_mono - rising_window)
        if rising_n >= int(self.cfg.stall_rising_cancel_threshold):
            return max(
                1,
                min(
                    int(self.cfg.max_start_per_tick),
                    int(self.cfg.stall_rising_start_limit),
                ),
            )

        return int(self.cfg.max_start_per_tick)

    # ----------------------------
    # Watchdog loop
    # ----------------------------
    async def _watchdog_loop(self) -> None:
        interval = max(0.5, float(self.cfg.sample_interval_sec))
        while True:
            await asyncio.sleep(interval)
            self._mark_heartbeat()

            cancel_ids: List[str] = []
            cancel_reason: str = ""
            recycle_count: int = 0
            recycle_reason: str = "mem_pressure_idle_recycle"

            async with self._lock:
                now = time.time()
                self._sample_memory_locked(now)

                active_ids = list(self._get_active_company_ids())
                active_n = len(active_ids)
                if active_n > 0:
                    self._ever_had_active = True

                now_mono = time.monotonic()
                self._prune_stall_events_locked(now_mono)

                self._update_base_rss_locked(active_n)
                self._update_per_company_est_locked(active_n)
                self._update_target_parallel_locked(active_n, now_mono=now_mono)

                # Progress inference (global)
                if (
                    (active_n != self._last_obs_active_n)
                    or (self._last_num_waiting != self._last_obs_waiting_n)
                    or (self._completed_counter != self._last_obs_completed)
                ):
                    self._mark_progress()
                    self._last_obs_active_n = active_n
                    self._last_obs_waiting_n = self._last_num_waiting
                    self._last_obs_completed = self._completed_counter

                active_set = set(active_ids)

                # tidy maps
                for cid in list(self._company_last_heartbeat_mono.keys()):
                    if cid not in active_set:
                        self._company_last_heartbeat_mono.pop(cid, None)
                for cid in list(self._company_last_progress_mono.keys()):
                    if cid not in active_set:
                        self._company_last_progress_mono.pop(cid, None)
                for cid in list(self._company_started_mono.keys()):
                    if cid not in active_set:
                        self._company_started_mono.pop(cid, None)

                for cid in active_ids:
                    self._company_started_mono.setdefault(cid, now_mono)
                    self._company_last_heartbeat_mono.setdefault(cid, now_mono)

                inflight_timeout = float(self.cfg.company_cancel_inflight_timeout_sec)
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

                # 1) Progress-based inactivity cancel
                if (
                    stall_enabled
                    and float(self.cfg.company_inactivity_timeout_sec) > 0
                    and active_n > 0
                ):
                    stuck: List[str] = []
                    timeout = float(self.cfg.company_inactivity_timeout_sec)
                    min_run = max(
                        0.0, float(self.cfg.company_inactivity_min_runtime_sec)
                    )
                    for cid in active_ids:
                        if _recently_cancelled(cid):
                            continue
                        started = self._company_started_mono.get(cid, now_mono)
                        if min_run > 0 and (now_mono - started) < min_run:
                            continue

                        last_prog = self._company_last_progress_mono.get(cid, started)
                        if (now_mono - last_prog) >= timeout:
                            stuck.append(cid)

                    if stuck:
                        stuck = stuck[: int(self.cfg.company_inactivity_cancel_max)]
                        cancel_ids = list(stuck)
                        cancel_reason = "stall_inactivity"

                        for cid in cancel_ids:
                            last_prog = self._company_last_progress_mono.get(
                                cid, self._company_started_mono.get(cid, now_mono)
                            )
                            age = max(0.0, now_mono - float(last_prog))
                            err = f"no progress for {age:.1f}s (timeout={timeout:.1f}s)"

                            # transient audit + real failure (stall) for backoff escalation
                            self.retry_store.record_scheduler_cancel(
                                company_id=str(cid),
                                reason="scheduler_inactivity",
                                stage="scheduler_inactivity",
                                md_done=None,
                                flush=True,
                            )
                            self.retry_store.mark_failure(
                                company_id=str(cid),
                                cls="stall",
                                error=str(err),
                                stage="scheduler_inactivity",
                                status_code=None,
                                nxdomain_like=False,
                                stall_kind_hint="no_yield",
                                md_done=None,
                                override_allow=False,
                                flush=True,
                            )
                            self._note_stall_cancel_locked(now_mono)

                        logger.error(
                            "[AdaptiveScheduling] progress inactivity timeout=%.1fs -> cancelling %s",
                            timeout,
                            cancel_ids,
                        )

                        if self._request_recycle_idle is not None:
                            recycle_count = max(recycle_count, 1)
                            recycle_reason = "stall_inactivity_recycle"

                # 1b) Stall-storm restart (optional) when no progress for long
                if self.cfg.stall_storm_restart_enabled and (
                    not self._restart_recommended
                ):
                    uptime = max(0.0, now_mono - self._started_mono)
                    window = max(1.0, float(self.cfg.stall_storm_window_sec))
                    storm_n = self._count_stall_cancels_since(now_mono - window)

                    need_waiting = bool(self.cfg.stall_storm_restart_require_waiting)
                    has_work = (self._last_num_waiting > 0) or self.has_pending()

                    no_prog_sec = max(
                        0.0, float(self.cfg.stall_storm_restart_no_progress_sec)
                    )
                    prog_age = max(0.0, now_mono - self._last_progress_mono)

                    if (
                        storm_n >= int(self.cfg.stall_storm_restart_threshold)
                        and uptime >= float(self.cfg.stall_storm_restart_min_uptime_sec)
                        and prog_age >= no_prog_sec
                        and ((not need_waiting) or has_work)
                    ):
                        self._recommend_restart_locked(
                            reason=(
                                f"stall_storm_restart storm_n={storm_n} window={window:.0f}s "
                                f"no_progress_for={prog_age:.0f}s waiting={self._last_num_waiting} "
                                f"used_raw={self._used_frac_raw:.3f} used_eff={self._used_frac_eff:.3f} "
                                f"proc_mb={(float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0):.1f}"
                            )
                        )

                # 2) Deadlock restart
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
                            self._recommend_restart_locked(
                                reason=(
                                    f"deadlock_restart dead_age={dead_age:.1f}s mem_high={mem_high} "
                                    f"used_raw={self._used_frac_raw:.3f} used_eff={self._used_frac_eff:.3f} "
                                    f"proc_mb={(float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0):.1f}"
                                )
                            )
                    else:
                        self._deadlock_since_mono = None
                else:
                    if not (
                        active_n == 0
                        and (self._last_num_waiting > 0 or self.has_pending())
                    ):
                        self._deadlock_since_mono = None

                # 3) Raw mem kill -> restart
                if (not self._restart_recommended) and self._used_frac_raw >= float(
                    self.cfg.mem_kill_raw_frac
                ):
                    self._recommend_restart_locked(
                        reason=f"raw_mem_kill used_raw={self._used_frac_raw:.3f} kill={float(self.cfg.mem_kill_raw_frac):.3f}"
                    )

                # 4) Emergency cancel (critical memory) - not gated
                crit_trigger = (
                    self._used_frac_raw >= float(self.cfg.mem_crit_raw_frac)
                ) or (self._used_frac_eff >= float(self.cfg.mem_crit_frac))
                ramp_trigger = self._ramp_raw_frac_per_sec >= float(
                    self.cfg.mem_ramp_crit_frac_per_sec
                ) and (self._used_frac_raw >= float(self.cfg.mem_crit_raw_frac))

                if (crit_trigger or ramp_trigger) and active_n > 0 and not cancel_ids:
                    if (now - self._last_emergency_cancel_ts) >= float(
                        self.cfg.emergency_cancel_cooldown_sec
                    ):
                        cancelable = [
                            cid for cid in active_ids if not _recently_cancelled(cid)
                        ]
                        inflight_active = sum(
                            1 for cid in active_ids if cid in self._cancel_inflight_mono
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
                            cancel_ids = self._select_cancel_ids(cancelable, to_cancel)
                            self._last_emergency_cancel_ts = now
                            cancel_reason = "mem_heavy" if ramp_trigger else "mem"

                            logger.error(
                                "[AdaptiveScheduling] EMERGENCY used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s "
                                "total_mb=%.1f used_raw_mb=%.1f used_eff_mb=%.1f proc_mem_mb=%.1f active=%d cancel=%s",
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

                # 5) Idle recycle request
                if self._request_recycle_idle is not None:
                    recycle_interval = float(self.cfg.idle_recycle_interval_sec) or 1.0
                    half = recycle_interval * 0.5
                    storm_active = now_mono < float(self._pause_starts_until_mono)
                    interval_used = half if storm_active else recycle_interval

                    if (now_mono - self._last_idle_recycle_mono) >= interval_used:
                        mem_pressure = (
                            self._used_frac_raw >= float(self.cfg.idle_recycle_raw_frac)
                        ) or (
                            self._used_frac_eff >= float(self.cfg.idle_recycle_eff_frac)
                        )
                        if mem_pressure or storm_active:
                            max_n = max(1, int(self.cfg.idle_recycle_max_per_interval))
                            if storm_active:
                                recycle_count = max(recycle_count, 1)
                                recycle_reason = "stall_storm_idle_recycle"
                            else:
                                if active_n == 0 and (
                                    self._last_num_waiting > 0 or self.has_pending()
                                ):
                                    recycle_count = max(recycle_count, max_n)
                                else:
                                    recycle_count = max(recycle_count, 1)
                                recycle_reason = "mem_pressure_idle_recycle"
                            self._last_idle_recycle_mono = now_mono

                # 6) Restart signal management
                if self._restart_recommended:
                    self._maybe_send_restart_signals_locked(now_mono)

            # Outside lock: apply cancels and schedule retries
            if cancel_ids:
                inflight_mark_ts = time.monotonic()
                async with self._lock:
                    for cid in cancel_ids:
                        self._cancel_inflight_mono[str(cid)] = inflight_mark_ts

                self._request_cancel_companies(cancel_ids)
                self._mark_progress()

                if cancel_reason in ("mem", "mem_heavy"):
                    cls_hint = "mem_heavy" if cancel_reason == "mem_heavy" else "mem"
                    err = (
                        f"cancelled_by_scheduler:{cls_hint} "
                        f"used_raw={self._used_frac_raw:.3f} used_eff={self._used_frac_eff:.3f} "
                        f"ramp_raw={self._ramp_raw_frac_per_sec:.3f}/s "
                        f"proc_mb={(float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0):.1f}"
                    )
                    await self._schedule_retries(
                        cancel_ids,
                        record_failure=True,
                        cls_hint=cls_hint,
                        error=err,
                        stage="scheduler_mem_cancel",
                        status_code=None,
                        permanent_reason="",
                    )
                elif cancel_reason == "stall_inactivity":
                    await self._schedule_retries(
                        cancel_ids,
                        record_failure=False,
                        cls_hint="stall",
                        error="",
                        stage="scheduler_inactivity",
                        status_code=None,
                        permanent_reason="",
                    )

                gc.collect()

                if (
                    cancel_reason == "stall_inactivity"
                    and self._request_recycle_idle is not None
                ):
                    await self._request_recycle_idle(1, "stall_inactivity_recycle")

            if recycle_count > 0 and self._request_recycle_idle is not None:
                await self._request_recycle_idle(
                    int(recycle_count), str(recycle_reason)
                )

    # ----------------------------
    # Restart gating / restart action
    # ----------------------------
    def _recommend_restart_locked(self, *, reason: str) -> None:
        if self._restart_recommended:
            return
        self._restart_recommended = True
        self._restart_reason = reason or "unspecified"
        self._restart_recommended_mono = time.monotonic()
        logger.critical(
            "[AdaptiveScheduling] restart recommended reason=%s", self._restart_reason
        )

        os.kill(os.getpid(), int(self.cfg.restart_signal))
        self._restart_sigterm_sent_mono = time.monotonic()

    def _maybe_send_restart_signals_locked(self, now_mono: float) -> None:
        repeat = float(self.cfg.restart_sigterm_repeat_interval_sec)
        if self._restart_sigterm_sent_mono is None:
            self._restart_sigterm_sent_mono = now_mono
            os.kill(os.getpid(), int(self.cfg.restart_signal))
            return

        if repeat > 0 and (now_mono - self._restart_sigterm_sent_mono) >= repeat:
            self._restart_sigterm_sent_mono = now_mono
            os.kill(os.getpid(), int(self.cfg.restart_signal))

        esc = float(self.cfg.restart_escalate_sigkill_after_sec)
        if esc > 0 and self._restart_recommended_mono is not None:
            if (now_mono - self._restart_recommended_mono) >= esc:
                os.kill(os.getpid(), 9)  # SIGKILL

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
    # Heartbeat/progress (GLOBAL)
    # ----------------------------
    def _mark_heartbeat(self) -> None:
        self._last_heartbeat_mono = time.monotonic()
        self._write_heartbeat_file_if_enabled()

    def _mark_progress(self) -> None:
        self._last_progress_mono = time.monotonic()
        self._write_heartbeat_file_if_enabled()

    def _write_heartbeat_file_if_enabled(self) -> None:
        path = self.cfg.heartbeat_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        now_mono = time.monotonic()
        payload = {
            "ts": time.time(),
            "mono": now_mono,
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
            "pause_starts_for_sec": max(0.0, self._pause_starts_until_mono - now_mono),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

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
            out[k] = int(v)
        return out

    def _read_cgroup_memory_bytes(self) -> Tuple[int, int, int]:
        """
        Returns (limit_bytes, used_effective_bytes, used_raw_bytes).
        Reads cgroup v2 first, then v1 if present. If not present, returns (0,0,0).
        """
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

        return 0, 0, 0

    def _read_psutil_memory_bytes(self) -> Tuple[int, int, int]:
        vm = psutil.virtual_memory()
        total = int(getattr(vm, "total", 0) or 0)
        available = int(getattr(vm, "available", 0) or 0)
        used = max(0, total - available)
        return total, used, used

    def _read_memory_usage_bytes(self) -> Tuple[int, int, int]:
        if self.cfg.prefer_cgroup_limits:
            limit, used_eff, used_raw = self._read_cgroup_memory_bytes()
            if limit <= 0:
                total, _, _ = self._read_psutil_memory_bytes()
                return total, used_eff, used_raw
            return limit, used_eff, used_raw
        return self._read_psutil_memory_bytes()

    def _read_process_tree_rss_bytes(self) -> int:
        """
        IMPORTANT (B5):
          psutil.NoSuchProcess is normal while walking a live process tree.
          Return best-effort sum.
        """
        try:
            p = psutil.Process(os.getpid())
        except (psutil.NoSuchProcess, ProcessLookupError):
            return 0
        except Exception:
            return 0

        rss = 0
        try:
            rss += int(p.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            pass
        except Exception:
            pass

        if not self.cfg.use_process_tree_rss_for_estimate:
            return max(0, int(rss))

        try:
            children = p.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            children = []
        except Exception:
            children = []

        for ch in children:
            try:
                rss += int(ch.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue
            except Exception:
                continue

        return max(0, int(rss))

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

        # Best-effort; never raise out of sampling.
        try:
            self._proc_rss_bytes = self._read_process_tree_rss_bytes()
        except Exception:
            # keep previous self._proc_rss_bytes
            pass

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

        t0: Optional[float] = None
        fe0: Optional[float] = None
        fr0: Optional[float] = None
        for t, fe, fr in reversed(self._mem_frac_hist):
            if (now_mono - t) >= window:
                t0, fe0, fr0 = t, fe, fr
                break

        if t0 is None or fe0 is None or fr0 is None:
            self._ramp_eff_frac_per_sec = 0.0
            self._ramp_raw_frac_per_sec = 0.0
            return

        dt = max(1e-6, float(now_mono - t0))
        self._ramp_eff_frac_per_sec = float(frac_eff - float(fe0)) / dt
        self._ramp_raw_frac_per_sec = float(frac_raw - float(fr0)) / dt

    def _update_base_rss_locked(self, active_n: int) -> None:
        if active_n != 0:
            return
        rss = float(self._proc_rss_bytes)
        if rss <= 0:
            return
        if self._base_rss_samples <= 0 or self._base_rss_bytes <= 0:
            self._base_rss_bytes = rss
            self._base_rss_samples = 1
            return
        alpha = 0.10 if rss >= self._base_rss_bytes else 0.25
        self._base_rss_bytes = (1.0 - alpha) * self._base_rss_bytes + alpha * rss
        self._base_rss_samples += 1

    def _update_per_company_est_locked(self, active_n: int) -> None:
        min_mb = float(self.cfg.per_company_min_mb)
        max_mb = float(self.cfg.per_company_max_mb)

        if active_n <= 0:
            self._per_company_est_mb = max(
                min_mb, min(self._per_company_est_mb, max_mb)
            )
            return

        rss = float(self._proc_rss_bytes)
        base = float(self._base_rss_bytes) if self._base_rss_bytes > 0 else 0.0
        extra = max(0.0, rss - base)
        per = (extra / float(active_n)) / _MB
        per = max(min_mb, min(per, max_mb))
        self._per_company_est_mb = per

    def _per_company_reservation_bytes_locked(self) -> int:
        sf = max(1.0, float(self.cfg.per_company_safety_factor))
        mb = max(1.0, float(self._per_company_est_mb))
        return int(mb * sf * _MB)

    def _update_target_parallel_locked(self, active_n: int, *, now_mono: float) -> None:
        old = int(self._target_parallel)
        tgt = old

        used_raw = float(self._used_frac_raw)
        used_eff = float(self._used_frac_eff)
        ramp_raw = float(self._ramp_raw_frac_per_sec)

        storm_window = max(1.0, float(self.cfg.stall_storm_window_sec))
        rising_window = max(1.0, float(self.cfg.stall_rising_window_sec))
        storm_n = self._count_stall_cancels_since(now_mono - storm_window)
        rising_n = self._count_stall_cancels_since(now_mono - rising_window)
        storm_active = (storm_n >= int(self.cfg.stall_storm_cancel_threshold)) or (
            now_mono < float(self._pause_starts_until_mono)
        )
        rising_active = rising_n >= int(self.cfg.stall_rising_cancel_threshold)

        should_decrease = (
            used_raw >= float(self.cfg.mem_high_raw_frac)
            or used_eff >= float(self.cfg.mem_high_frac)
            or (active_n > 0 and ramp_raw >= float(self.cfg.mem_ramp_high_frac_per_sec))
            or storm_active
            or rising_active
        )

        if should_decrease and tgt > int(self.cfg.min_target):
            md = float(self.cfg.md_factor)
            if storm_active:
                md = min(md, 0.45)
            elif rising_active:
                md = min(md, 0.55)
            tgt = max(int(self.cfg.min_target), int(max(1, int(tgt * md))))
        else:
            comfortable = (
                used_eff < float(self.cfg.mem_cap_frac)
                and used_raw < float(self.cfg.mem_high_raw_frac) * 0.98
                and ramp_raw < float(self.cfg.ramp_admit_guard_frac_per_sec)
                and (not storm_active)
            )
            if comfortable and tgt < int(self.cfg.max_target):
                tgt = min(int(self.cfg.max_target), tgt + int(self.cfg.ai_step))

        self._target_parallel = max(
            int(self.cfg.min_target), min(int(tgt), int(self.cfg.max_target))
        )

        if self._target_parallel != old:
            logger.info(
                "[AdaptiveScheduling] target_parallel %d -> %d (used_raw=%.3f used_eff=%.3f ramp_raw=%.3f/s active=%d "
                "stall_rising=%s stall_storm=%s proc_mem_mb=%.1f)",
                old,
                self._target_parallel,
                used_raw,
                used_eff,
                ramp_raw,
                int(active_n),
                bool(rising_active),
                bool(storm_active),
                float(self._proc_rss_bytes) / _MB if self._proc_rss_bytes else 0.0,
            )
