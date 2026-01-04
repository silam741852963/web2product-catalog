from __future__ import annotations

import asyncio
import contextlib
import heapq
import logging
import signal
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
    Union,
)

import psutil

from .retry import RetryStateStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MB = 1_000_000


# --------------------------------------------------------------------------------------
# Retry-store helpers
# --------------------------------------------------------------------------------------
def compute_retry_exit_code_from_store(
    store: RetryStateStore, retry_exit_code: int
) -> int:
    """
    Return retry_exit_code only if there exists at least one eligible-now retry-pending company.

    IMPORTANT:
      - used by run.py without await, so this must remain sync.
    """
    now = time.time()
    try:
        eligible_n = int(store.pending_eligible_total_sync(now=now))
    except Exception:
        eligible_n = 0
    return int(retry_exit_code) if eligible_n > 0 else 0


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass(slots=True)
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
    mem_kill_raw_frac: float = 0.97  # restart recommended (NO self-signal)

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
    company_inactivity_min_runtime_sec: float = 90.0
    company_inactivity_cancel_max: int = 1

    # stage-aware inactivity timeouts
    crawl_inactivity_timeout_sec: float = 240.0

    # Default disable for LLM (queueing != stuck for local ollama)
    llm_inactivity_timeout_sec: float = 0.0

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
    stall_storm_restart_no_progress_sec: float = 600.0
    stall_storm_restart_require_waiting: bool = True

    # Optional heartbeat/logging
    heartbeat_path: Optional[Path] = None
    log_path: Optional[Path] = None

    # Prefer cgroup limits if available
    prefer_cgroup_limits: bool = True
    cgroup_subtract_inactive_file: bool = True

    # --- Restart escalation (DISABLED: scheduler must not signal itself) ---
    restart_sigterm_repeat_interval_sec: float = 30.0
    restart_escalate_sigkill_after_sec: float = 120.0
    restart_signal: int = int(signal.SIGTERM)

    # --- Work scheduling knobs ---
    max_start_per_tick: int = 3
    crawler_capacity_multiplier: int = 3

    idle_recycle_interval_sec: float = 25.0
    idle_recycle_raw_frac: float = 0.88
    idle_recycle_eff_frac: float = 0.83
    idle_recycle_max_per_interval: int = 2

    # Retry state location (scheduler owns store)
    retry_base_dir: Optional[Union[Path, str]] = None

    # Backoff sleep smoothing
    min_idle_sleep_sec: float = 0.25
    max_idle_sleep_sec: float = 30.0

    # When scheduler cancels a company, do not re-admit it immediately.
    cancel_requeue_min_delay_sec: float = 2.0

    # Cancel spam guard
    company_cancel_repeat_cooldown_sec: float = 15.0
    company_cancel_inflight_timeout_sec: float = 600.0

    # --- Doomed repeat / GOTO-timeout convergence controls ---
    doomed_goto_same_error_streak_threshold: int = 2
    doomed_goto_min_cooldown_sec: float = 1800.0
    last_one_standing_quarantine_enabled: bool = True
    deferred_move_log_interval_sec: float = 30.0


class AdaptiveScheduler:
    """
    Stable API (used by run.py):
      - await start()/stop()
      - retry_store property
      - await set_worklist(...)
      - await ensure_worklist(...)
      - await plan_start_batch(free_crawlers=...) -> list[str]
      - has_pending(), pending_total(), pending_ready()
      - sleep_hint_sec(), initial_total_hint()
      - register_company_completed()
      - await cleanup_completed_retry_ids(...)
      - suspend_stall_detection()/resume_stall_detection()
      - get_state_snapshot()
      - touch_company()/heartbeat_company()/progress_company()
      - await requeue_company(...)
    """

    STAGE_CRAWL = "crawl"
    STAGE_LLM = "llm"
    STAGE_IDLE = "idle"

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

        retry_base = Path(cfg.retry_base_dir).expanduser().resolve()
        self.retry_store = RetryStateStore(base_dir=retry_base)

        # Worklist
        self._work_ready: Deque[str] = deque()
        self._work_deferred: List[Tuple[float, str]] = []  # heap (eligible_at_ts, cid)
        self._deferred_at: Dict[str, float] = {}  # cid -> eligible_at
        self._queued: set[str] = set()  # present in ready or deferred
        self._work_seen: set[str] = set()
        self._work_total_hint: int = 0
        self._is_company_runnable: Optional[Callable[[str], Awaitable[bool]]] = None

        # log throttles
        self._last_deferred_move_log_mono: float = 0.0
        self._last_block_log_mono: float = 0.0

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
        self._company_stage: Dict[str, str] = {}

        # AIMD target
        self._target_parallel: int = max(
            cfg.min_target, min(cfg.initial_target, cfg.max_target)
        )

        # Progress/global counters
        self._completed_counter: int = 0
        self._last_num_waiting: int = 0
        self._ever_admitted: bool = False
        self._ever_had_active: bool = False
        self._started_ts: float = time.time()
        self._started_mono: float = time.monotonic()

        # Global timestamps
        self._last_heartbeat_mono: float = time.monotonic()
        self._last_progress_mono: float = time.monotonic()
        self._last_admission_mono: float = time.monotonic()

        # Observations to infer progress
        self._last_obs_active_n: int = 0
        self._last_obs_waiting_n: int = 0
        self._last_obs_completed: int = 0

        # Emergency bookkeeping
        self._restart_recommended: bool = False
        self._restart_reason: str = ""
        self._restart_recommended_mono: Optional[float] = None
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
    # Stage API (STRICT)
    # ----------------------------
    def _normalize_stage(self, stage: str) -> str:
        s = (stage or "").strip().lower()
        if not s:
            return self.STAGE_CRAWL
        if s in ("crawl", "crawling"):
            return self.STAGE_CRAWL
        if s in ("llm", "llm_stage", "model", "inference"):
            return self.STAGE_LLM
        if s in ("idle", "done", "completed"):
            return self.STAGE_IDLE
        return s

    def get_company_stage(self, company_id: str) -> Optional[str]:
        cid = (company_id or "").strip()
        if not cid:
            return None
        return self._company_stage.get(cid)

    async def set_company_stage(
        self,
        company_id: str,
        stage: str,
        *,
        reset_timers: bool = False,
        reason: str = "",
    ) -> None:
        cid = (company_id or "").strip()
        if not cid:
            return
        st = self._normalize_stage(stage)
        async with self._lock:
            prev = self._company_stage.get(cid)
            self._company_stage[cid] = st
            if reset_timers:
                now_mono = time.monotonic()
                self._company_last_heartbeat_mono[cid] = now_mono
                self._company_last_progress_mono[cid] = now_mono
                self._company_started_mono.setdefault(cid, now_mono)
                self._last_heartbeat_mono = now_mono
                self._last_progress_mono = now_mono
                self._last_admission_mono = now_mono

        if prev != st:
            logger.info(
                "[AdaptiveScheduling] company stage set cid=%s stage=%s prev=%s reason=%s",
                cid,
                st,
                prev or "none",
                reason or "unspecified",
            )

    async def clear_company_stage(
        self,
        company_id: str,
        *,
        reset_timers: bool = False,
        reason: str = "",
    ) -> None:
        cid = (company_id or "").strip()
        if not cid:
            return
        async with self._lock:
            prev = self._company_stage.pop(cid, None)
            if reset_timers:
                now_mono = time.monotonic()
                self._company_last_heartbeat_mono[cid] = now_mono
                self._company_last_progress_mono[cid] = now_mono
                self._company_started_mono.setdefault(cid, now_mono)
                self._last_heartbeat_mono = now_mono
                self._last_progress_mono = now_mono
                self._last_admission_mono = now_mono

        if prev is not None:
            logger.info(
                "[AdaptiveScheduling] company stage cleared cid=%s prev=%s reason=%s",
                cid,
                prev,
                reason or "unspecified",
            )

    def _stage_inactivity_timeout_sec_locked(self, cid: str) -> float:
        st = self._normalize_stage(self._company_stage.get(cid) or self.STAGE_CRAWL)
        if st == self.STAGE_LLM:
            t = float(self.cfg.llm_inactivity_timeout_sec)
            return 0.0 if t <= 0 else float(t)
        return max(0.0, float(self.cfg.crawl_inactivity_timeout_sec))

    # ----------------------------
    # Doomed repeat-goto (STRICT)
    # ----------------------------
    async def _is_doomed_repeat_goto(self, cid: str) -> bool:
        thr = max(1, int(self.cfg.doomed_goto_same_error_streak_threshold))
        return bool(await self.retry_store.is_doomed_repeat_goto(cid, threshold=thr))

    async def _quarantine_last_one_standing(self, cid: str, *, reason: str) -> None:
        await self.retry_store.quarantine_company(
            cid,
            reason=reason,
            stage="scheduler_last_one_standing",
            error=reason,
            cls="permanent",
            status_code=None,
            nxdomain_like=False,
            md_done=None,
            flush=True,
            also_mark_terminal_done=True,
        )

    def _only_one_left_now(self) -> bool:
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
        self._is_company_runnable = is_company_runnable
        ids = [str(x) for x in company_ids if str(x).strip()]
        self._work_total_hint = len(ids)

        pending_retry = set(
            await self.retry_store.pending_ids(exclude_quarantined=True)
        )
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

                if await self.retry_store.is_quarantined(cid):
                    continue
                if cid in self._queued:
                    continue

                if not await self.retry_store.is_eligible(cid, now=now):
                    ts = float(await self.retry_store.next_eligible_at(cid))
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

    async def ensure_worklist(
        self, company_ids: Sequence[str], *, reason: str = ""
    ) -> int:
        """
        Ensure the given IDs are present in scheduler queues (ready or deferred).

        This exists specifically for run.py drain_reconcile_and_reseed(), which awaits it.
        """
        ids = [str(x) for x in company_ids if str(x).strip()]
        if not ids:
            return 0

        now = time.time()
        added = 0
        async with self._lock:
            for cid in ids:
                if cid in self._queued:
                    continue
                if await self.retry_store.is_quarantined(cid):
                    continue

                if not await self.retry_store.is_eligible(cid, now=now):
                    ts = float(await self.retry_store.next_eligible_at(cid))
                    self._enqueue_deferred_locked(cid, ts)
                    added += 1
                    continue

                self._enqueue_ready_locked(cid)
                added += 1

        if added:
            logger.info(
                "[AdaptiveScheduling] ensure_worklist added=%d reason=%s ready=%d deferred=%d",
                added,
                reason or "unspecified",
                len(self._work_ready),
                len(self._deferred_at),
            )
        return int(added)

    async def cleanup_completed_retry_ids(
        self,
        *,
        is_company_runnable: Callable[[str], Awaitable[bool]],
        treat_non_runnable_as_done: bool,
        stage: str = "startup_cleanup",
    ) -> int:
        ids = sorted(await self.retry_store.pending_ids(exclude_quarantined=True))
        if not ids:
            return 0

        cleared = 0
        for cid in ids:
            runnable = await is_company_runnable(cid)
            if treat_non_runnable_as_done and (not runnable):
                await self.retry_store.mark_success(
                    cid, stage=stage, note="already_done"
                )
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
        cid = (company_id or "").strip()
        if not cid:
            return False
        if await self.retry_store.is_quarantined(cid):
            return False

        now = time.time()
        ts = now + max(0.0, float(delay_sec))

        if await self._is_doomed_repeat_goto(cid):
            if (
                self.cfg.last_one_standing_quarantine_enabled
                and self._only_one_left_now()
            ):
                await self._quarantine_last_one_standing(
                    cid, reason="scheduler_last_one_standing_repeat_goto_timeout"
                )
                logger.warning(
                    "[AdaptiveScheduling] quarantined last-one-standing on requeue cid=%s",
                    cid,
                )
                return True

            cooldown = max(0.0, float(self.cfg.doomed_goto_min_cooldown_sec))
            ts = max(
                ts,
                now + cooldown,
                float(await self.retry_store.next_eligible_at(cid) or 0.0),
            )
            async with self._lock:
                self._enqueue_deferred_locked(cid, ts)
            logger.warning(
                "[AdaptiveScheduling] requeue deferred doomed cid=%s delay=%.0fs reason=%s",
                cid,
                max(0.0, ts - now),
                reason or "doomed_repeat_goto",
            )
            self._mark_progress()
            return True

        if not force and not await self.retry_store.is_eligible(cid, now=now):
            ts = max(ts, float(await self.retry_store.next_eligible_at(cid)))

        async with self._lock:
            if cid in self._queued:
                if cid in self._deferred_at:
                    self._enqueue_deferred_locked(cid, ts)
                return True

            if cid in set(self._get_active_company_ids()):
                ts = max(ts, now + float(self.cfg.cancel_requeue_min_delay_sec))
                self._enqueue_deferred_locked(cid, ts)
                return True

            if force or (
                await self.retry_store.is_eligible(cid, now=now) and ts <= now
            ):
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

    async def _move_due_deferred(self, now: float) -> None:
        moved = 0
        async with self._lock:
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

                if await self.retry_store.is_quarantined(cid):
                    self._queued.discard(cid)
                    continue

                if not await self.retry_store.is_eligible(cid, now=now):
                    ts2 = float(await self.retry_store.next_eligible_at(cid))
                    self._enqueue_deferred_locked(cid, ts2)
                    continue

                if await self._is_doomed_repeat_goto(cid):
                    if (
                        self.cfg.last_one_standing_quarantine_enabled
                        and self._only_one_left_now()
                    ):
                        await self._quarantine_last_one_standing(
                            cid,
                            reason="scheduler_last_one_standing_repeat_goto_timeout",
                        )
                        logger.warning(
                            "[AdaptiveScheduling] quarantined last-one-standing on deferred->ready cid=%s",
                            cid,
                        )
                        self._queued.discard(cid)
                        continue

                    cooldown = max(0.0, float(self.cfg.doomed_goto_min_cooldown_sec))
                    ts3 = max(
                        float(now) + cooldown,
                        float(await self.retry_store.next_eligible_at(cid) or 0.0),
                    )
                    self._enqueue_deferred_locked(cid, ts3)
                    logger.warning(
                        "[AdaptiveScheduling] deferred doomed on move cid=%s delay=%.0fs",
                        cid,
                        max(0.0, ts3 - float(now)),
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

    async def _prune_queued_quarantined(self, *, limit: int = 2000) -> int:
        removed = 0
        if limit <= 0:
            return 0

        async with self._lock:
            if self._deferred_at:
                for cid in list(self._deferred_at.keys())[:limit]:
                    if removed >= limit:
                        break
                    if await self.retry_store.is_quarantined(cid):
                        self._deferred_at.pop(cid, None)
                        self._queued.discard(cid)
                        removed += 1

            if removed < limit and self._work_ready:
                new_ready: Deque[str] = deque()
                scan = 0
                while self._work_ready and removed < limit:
                    cid = self._work_ready.popleft()
                    scan += 1
                    if cid and await self.retry_store.is_quarantined(cid):
                        self._queued.discard(cid)
                        removed += 1
                        continue
                    new_ready.append(cid)
                    if scan >= limit:
                        new_ready.extend(self._work_ready)
                        self._work_ready.clear()
                        break
                self._work_ready = new_ready

        return removed

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
    # Company progress hooks
    # ----------------------------
    def touch_company(self, company_id: str, *, kind: str = "progress") -> None:
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
            "[AdaptiveScheduling] started total_mem_mb=%.1f initial_target_parallel=%d llm_inactivity_timeout_sec=%.1f",
            float(self._total_mem_bytes) / _MB if self._total_mem_bytes else -1.0,
            self._target_parallel,
            float(self.cfg.llm_inactivity_timeout_sec),
        )

    async def stop(self) -> None:
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

        stage_counts: Dict[str, int] = {}
        for _, st in self._company_stage.items():
            st2 = self._normalize_stage(st)
            stage_counts[st2] = stage_counts.get(st2, 0) + 1

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
            "company_stage_counts": stage_counts,
            "llm_inactivity_timeout_sec": float(self.cfg.llm_inactivity_timeout_sec),
            "crawl_inactivity_timeout_sec": float(
                self.cfg.crawl_inactivity_timeout_sec
            ),
        }

    # ----------------------------
    # Core: planning start batch
    # ----------------------------
    async def plan_start_batch(self, *, free_crawlers: int) -> List[str]:
        active_ids = list(self._get_active_company_ids())
        active_set = set(active_ids)
        active_n = len(active_ids)

        await self._move_due_deferred(time.time())

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
                    await self.retry_store.mark_success(
                        cid, stage="scheduler_skip", note="already_done_or_terminal"
                    )
                    continue

            if await self._is_doomed_repeat_goto(cid):
                only_one_left = (active_n == 0) and (self.pending_total() <= 1)
                if self.cfg.last_one_standing_quarantine_enabled and only_one_left:
                    await self._quarantine_last_one_standing(
                        cid, reason="scheduler_last_one_standing_repeat_goto_timeout"
                    )
                    logger.warning(
                        "[AdaptiveScheduling] quarantined last-one-standing cid=%s", cid
                    )
                    continue

                cooldown = max(0.0, float(self.cfg.doomed_goto_min_cooldown_sec))
                ts = max(
                    now + cooldown,
                    float(await self.retry_store.next_eligible_at(cid) or 0.0),
                )
                async with self._lock:
                    self._enqueue_deferred_locked(cid, ts)
                logger.warning(
                    "[AdaptiveScheduling] deferred doomed cid=%s delay=%.0fs",
                    cid,
                    max(0.0, ts - now),
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
                    self._company_stage.setdefault(cid, self.STAGE_CRAWL)

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
            self._update_target_parallel_locked()

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
                self._maybe_log_block("block_mem_cap", active_n, num_waiting)
                slots = 0
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

            await self._prune_queued_quarantined(limit=2000)

            async with self._lock:
                active_ids = list(self._get_active_company_ids())
                active_n = len(active_ids)
                now_mono = time.monotonic()
                active_set = set(active_ids)

                self._update_base_rss_locked(active_n)
                self._update_per_company_est_locked(active_n)
                self._update_target_parallel_locked()

                if (
                    (active_n != self._last_obs_active_n)
                    or (self._last_num_waiting != self._last_obs_waiting_n)
                    or (self._completed_counter != self._last_obs_completed)
                ):
                    self._mark_progress()
                    self._last_obs_active_n = active_n
                    self._last_obs_waiting_n = self._last_num_waiting
                    self._last_obs_completed = self._completed_counter

                for cid in list(self._company_last_heartbeat_mono.keys()):
                    if cid not in active_set:
                        self._company_last_heartbeat_mono.pop(cid, None)
                for cid in list(self._company_last_progress_mono.keys()):
                    if cid not in active_set:
                        self._company_last_progress_mono.pop(cid, None)
                for cid in list(self._company_started_mono.keys()):
                    if cid not in active_set:
                        self._company_started_mono.pop(cid, None)
                for cid in list(self._company_stage.keys()):
                    if cid not in active_set:
                        self._company_stage.pop(cid, None)

                for cid in active_ids:
                    self._company_started_mono.setdefault(cid, now_mono)
                    self._company_last_heartbeat_mono.setdefault(cid, now_mono)
                    self._company_stage.setdefault(cid, self.STAGE_CRAWL)

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

            if stall_enabled and active_n > 0:
                min_run = max(0.0, float(self.cfg.company_inactivity_min_runtime_sec))
                stuck: List[Tuple[str, float]] = []
                now_mono = time.monotonic()

                async with self._lock:
                    for cid in active_ids:
                        if _recently_cancelled(cid):
                            continue

                        started = self._company_started_mono.get(cid, now_mono)
                        if min_run > 0 and (now_mono - started) < min_run:
                            continue

                        timeout = float(self._stage_inactivity_timeout_sec_locked(cid))
                        if timeout <= 0:
                            continue

                        last_prog = self._company_last_progress_mono.get(cid, started)
                        age = float(now_mono - last_prog)
                        if age >= timeout:
                            stuck.append((cid, age))

                if stuck:
                    stuck.sort(key=lambda x: x[1], reverse=True)
                    cancel_ids = [cid for (cid, _) in stuck][
                        : int(self.cfg.company_inactivity_cancel_max)
                    ]

                    for cid, age in stuck[
                        : int(self.cfg.company_inactivity_cancel_max)
                    ]:
                        timeout = float(self.cfg.crawl_inactivity_timeout_sec)
                        stage = self.get_company_stage(cid) or self.STAGE_CRAWL
                        err = f"no progress for {age:.1f}s (timeout={timeout:.1f}s stage={stage})"

                        await self.retry_store.record_scheduler_cancel(
                            str(cid),
                            reason="scheduler_inactivity",
                            stage="scheduler_inactivity",
                            md_done=None,
                            flush=True,
                        )
                        await self.retry_store.mark_failure(
                            str(cid),
                            cls="stall",
                            error=err,
                            stage="scheduler_inactivity",
                            status_code=None,
                            nxdomain_like=False,
                            stall_kind_hint="no_yield",
                            md_done=None,
                            override_allow=False,
                            flush=True,
                        )
                        async with self._lock:
                            self._note_stall_cancel_locked(now_mono)

                    logger.error(
                        "[AdaptiveScheduling] inactivity -> cancelling %s", cancel_ids
                    )
                    if self._request_recycle_idle is not None:
                        recycle_count = max(recycle_count, 1)
                        recycle_reason = "stall_inactivity_recycle"

            if active_n > 0:
                now_ts = time.time()
                cooldown = float(self.cfg.emergency_cancel_cooldown_sec)
                can_cancel = (now_ts - float(self._last_emergency_cancel_ts)) >= max(
                    0.0, cooldown
                )

                if can_cancel and (
                    self._used_frac_raw >= float(self.cfg.mem_crit_raw_frac)
                    or self._used_frac_eff >= float(self.cfg.mem_crit_frac)
                    or self._ramp_raw_frac_per_sec
                    >= float(self.cfg.mem_ramp_crit_frac_per_sec)
                ):
                    keep = max(0, int(self.cfg.min_active_keep))
                    budget = max(0, int(self.cfg.emergency_cancel_max))
                    cancellable = max(0, active_n - keep)
                    n_cancel = min(budget, cancellable)
                    if n_cancel > 0:
                        ranked: List[Tuple[float, str]] = []
                        now_mono = time.monotonic()
                        async with self._lock:
                            for cid in active_ids:
                                last_prog = self._company_last_progress_mono.get(
                                    cid, self._company_started_mono.get(cid, now_mono)
                                )
                                ranked.append((float(last_prog), cid))
                        ranked.sort(key=lambda x: x[0])
                        cancel_ids = [cid for (_, cid) in ranked[:n_cancel]]
                        self._last_emergency_cancel_ts = now_ts

                        for cid in cancel_ids:
                            await self.retry_store.record_scheduler_cancel(
                                str(cid),
                                reason="scheduler_mem_critical",
                                stage="scheduler_mem_critical",
                                md_done=None,
                                flush=True,
                            )
                            await self.retry_store.mark_failure(
                                str(cid),
                                cls="mem",
                                error=(
                                    f"mem critical used_raw={self._used_frac_raw:.3f} used_eff={self._used_frac_eff:.3f} "
                                    f"ramp_raw={self._ramp_raw_frac_per_sec:.3f}/s"
                                ),
                                stage="scheduler_mem_critical",
                                status_code=None,
                                nxdomain_like=False,
                                stall_kind_hint=None,
                                md_done=None,
                                override_allow=False,
                                flush=True,
                            )

                        logger.error(
                            "[AdaptiveScheduling] mem critical -> cancelling %s",
                            cancel_ids,
                        )
                        async with self._lock:
                            self._note_stall_cancel_locked(time.monotonic())

            if cancel_ids:
                async with self._lock:
                    self._cancel_inflight_mono.update(
                        {cid: time.monotonic() for cid in cancel_ids}
                    )
                self._request_cancel_companies(cancel_ids)

            if recycle_count > 0 and self._request_recycle_idle is not None:
                with contextlib.suppress(Exception):
                    await self._request_recycle_idle(int(recycle_count), recycle_reason)

    # ----------------------------
    # Memory sampling / estimation
    # ----------------------------
    def _sample_memory_locked(self, now_ts: float) -> None:
        total, used_eff, used_raw = self._read_memory_usage_bytes()
        self._total_mem_bytes = total
        self._used_bytes_eff = used_eff
        self._used_bytes_raw = used_raw
        self._used_frac_eff = (float(used_eff) / float(total)) if total > 0 else 0.0
        self._used_frac_raw = (float(used_raw) / float(total)) if total > 0 else 0.0
        self._proc_rss_bytes = (
            self._read_process_tree_rss_bytes()
            if self.cfg.use_process_tree_rss_for_estimate
            else 0
        )
        self._last_sample_ts = float(now_ts)

        now_mono = time.monotonic()
        self._update_ramp_locked(now_mono, self._used_frac_eff, self._used_frac_raw)

    def _update_ramp_locked(
        self, now_mono: float, frac_eff: float, frac_raw: float
    ) -> None:
        self._mem_frac_hist.append((float(now_mono), float(frac_eff), float(frac_raw)))
        window = max(0.5, float(self.cfg.mem_ramp_window_sec))
        cutoff = float(now_mono) - window
        while self._mem_frac_hist and self._mem_frac_hist[0][0] < cutoff:
            self._mem_frac_hist.popleft()

        if len(self._mem_frac_hist) < 2:
            self._ramp_eff_frac_per_sec = 0.0
            self._ramp_raw_frac_per_sec = 0.0
            return

        t0, e0, r0 = self._mem_frac_hist[0]
        t1, e1, r1 = self._mem_frac_hist[-1]
        dt = max(1e-6, float(t1 - t0))
        self._ramp_eff_frac_per_sec = float((e1 - e0) / dt)
        self._ramp_raw_frac_per_sec = float((r1 - r0) / dt)

    def _update_base_rss_locked(self, active_n: int) -> None:
        if not self.cfg.use_process_tree_rss_for_estimate:
            return
        if active_n > 1:
            return
        rss = float(self._proc_rss_bytes)
        if rss <= 0:
            return
        self._base_rss_samples += 1
        if self._base_rss_bytes <= 0:
            self._base_rss_bytes = rss
        else:
            alpha = 0.15
            self._base_rss_bytes = (1 - alpha) * self._base_rss_bytes + alpha * rss

    def _update_per_company_est_locked(self, active_n: int) -> None:
        lo = float(self.cfg.per_company_min_mb)
        hi = float(self.cfg.per_company_max_mb)
        if not self.cfg.use_process_tree_rss_for_estimate:
            self._per_company_est_mb = max(lo, min(self._per_company_est_mb, hi))
            return

        rss = float(self._proc_rss_bytes)
        base = float(self._base_rss_bytes)
        if rss <= 0:
            return

        if active_n <= 0:
            self._per_company_est_mb = max(lo, min(self._per_company_est_mb, hi))
            return

        per = max(0.0, (rss - base) / max(1, active_n))
        per_mb = (per / _MB) * float(self.cfg.per_company_safety_factor)
        per_mb = max(lo, min(per_mb, hi))

        alpha = 0.20
        self._per_company_est_mb = (1 - alpha) * float(
            self._per_company_est_mb
        ) + alpha * float(per_mb)

    def _update_target_parallel_locked(self) -> None:
        min_t = int(self.cfg.min_target)
        max_t = int(self.cfg.max_target)

        target = int(self._target_parallel)

        high = (self._used_frac_eff >= float(self.cfg.mem_high_frac)) or (
            self._used_frac_raw >= float(self.cfg.mem_high_raw_frac)
        )
        crit = (self._used_frac_eff >= float(self.cfg.mem_crit_frac)) or (
            self._used_frac_raw >= float(self.cfg.mem_crit_raw_frac)
        )
        ramp_high = self._ramp_raw_frac_per_sec >= float(
            self.cfg.mem_ramp_high_frac_per_sec
        )
        ramp_crit = self._ramp_raw_frac_per_sec >= float(
            self.cfg.mem_ramp_crit_frac_per_sec
        )

        if crit or ramp_crit:
            target = max(min_t, int(target * float(self.cfg.md_factor)))
        elif high or ramp_high:
            target = max(min_t, int(target * float(self.cfg.md_factor)))
        else:
            target = min(max_t, target + int(self.cfg.ai_step))

        self._target_parallel = max(min_t, min(max_t, int(target)))

    def _per_company_reservation_bytes_locked(self) -> int:
        mb = float(self._per_company_est_mb)
        return int(max(1.0, mb) * _MB)

    # ----------------------------
    # Memory reading (host or cgroup v2)
    # ----------------------------
    def _read_memory_usage_bytes(self) -> Tuple[int, int, int]:
        if self.cfg.prefer_cgroup_limits:
            cg = self._read_cgroup_v2_memory()
            if cg is not None:
                total, used_raw, inactive_file = cg
                used_eff = used_raw
                if self.cfg.cgroup_subtract_inactive_file and inactive_file is not None:
                    used_eff = max(0, used_raw - int(inactive_file))
                return int(total), int(used_eff), int(used_raw)

        vm = psutil.virtual_memory()
        total = int(vm.total)
        used_raw = int(total - vm.available)
        used_eff = int(used_raw)
        return total, used_eff, used_raw

    def _read_cgroup_v2_memory(self) -> Optional[Tuple[int, int, Optional[int]]]:
        base = Path("/sys/fs/cgroup")
        cur_p = base / "memory.current"
        max_p = base / "memory.max"
        stat_p = base / "memory.stat"
        if not (cur_p.exists() and max_p.exists()):
            return None

        try:
            cur = int(cur_p.read_text().strip())
            max_raw = max_p.read_text().strip()
            if max_raw == "max":
                vm = psutil.virtual_memory()
                total = int(vm.total)
            else:
                total = int(max_raw)

            inactive_file: Optional[int] = None
            if stat_p.exists():
                for line in stat_p.read_text().splitlines():
                    if line.startswith("inactive_file "):
                        inactive_file = int(line.split()[1])
                        break

            return total, cur, inactive_file
        except Exception:
            return None

    def _read_process_tree_rss_bytes(self) -> int:
        try:
            p = psutil.Process()
            rss = 0
            with contextlib.suppress(Exception):
                rss += int(p.memory_info().rss)
            with contextlib.suppress(Exception):
                for ch in p.children(recursive=True):
                    with contextlib.suppress(Exception):
                        rss += int(ch.memory_info().rss)
            return int(rss)
        except Exception:
            return 0

    # ----------------------------
    # Progress markers
    # ----------------------------
    def _mark_heartbeat(self) -> None:
        self._last_heartbeat_mono = time.monotonic()

    def _mark_progress(self) -> None:
        self._last_progress_mono = time.monotonic()
