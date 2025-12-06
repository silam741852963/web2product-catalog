from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

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


@dataclass
class AdaptiveConcurrencyConfig:
    """
    Configuration for adaptive concurrency control.

    This controller produces a single integer: current_limit.

    Semantics:

      - current_limit is the target number of concurrent *company* pipelines
        that the host should try to run.

      - It is always clamped to [min_concurrency, max_concurrency].

      - The controller does not know about task counts, inflight sets,
        admission budgets, or slots beyond this single scalar.

      - The runner (run.py) is responsible for enforcing:

            inflight_companies <= min(current_limit, --company-concurrency)

        and starting new companies immediately whenever there is room under
        that cap.

    Fields:

      max_concurrency:
          Hard upper bound (usually from --company-concurrency).

      min_concurrency:
          Lower bound.

      target_mem_low / target_mem_high:
          Memory usage band (fraction 0..1) that we aim to sit in.

      target_cpu_low / target_cpu_high:
          CPU usage band (fraction 0..1). We only scale up when both CPU
          and memory are below the lower band (if enabled). We scale down
          when either enabled dimension is above the upper band.

      sample_interval_sec:
          How often to sample CPU/memory and recompute the limit.

      smoothing_window_sec:
          Sliding window length for smoothing samples.

      log_path:
          Optional path to a line based JSON log for debugging decisions.

      use_cpu / use_mem:
          Whether to use CPU and or memory in decisions.

      scale_up_step / scale_down_step:
          How many slots to add or remove per scaling decision.

      scale_up_cooldown_sec / scale_down_cooldown_sec:
          Minimum time between scale up and scale down decisions.

      Global stall detection:

        The controller can detect a global under-utilization stall and
        temporarily ramp up faster than normal.

        stall_detection_window_sec:
            How far back in the sample history we look.

        stall_cpu_idle_threshold:
            Average CPU below this fraction is considered idle enough
            for stall detection.

        stall_mem_band_width:
            Maximum variation span for CPU and memory in the window to
            consider them flat.

        stall_release_cooldown_sec:
            After a stall boost ends, wait this long before allowing
            another boost.

        stall_boost_scale_up_multiplier:
            Multiply scale_up_step by this factor when a stall is active.
    """

    max_concurrency: int
    min_concurrency: int = 1

    target_mem_low: float = 0.80
    target_mem_high: float = 0.90
    target_cpu_low: float = 0.90
    target_cpu_high: float = 1.00

    sample_interval_sec: float = 1.0
    smoothing_window_sec: float = 10.0

    log_path: Optional[Path] = None

    use_cpu: bool = True
    use_mem: bool = True

    scale_up_step: int = 1
    scale_down_step: int = 1

    scale_up_cooldown_sec: float = 5.0
    scale_down_cooldown_sec: float = 0.0

    # Global stall detection
    stall_detection_window_sec: float = 30.0
    stall_cpu_idle_threshold: float = 0.5
    stall_mem_band_width: float = 0.05
    stall_release_cooldown_sec: float = 30.0
    stall_boost_scale_up_multiplier: float = 4.0


class AdaptiveConcurrencyController:
    """
    Adaptive controller that adjusts an integer concurrency limit based on
    smoothed CPU and memory usage.

    Important semantics:

      - The controller only produces a scalar: current_limit.

      - run.py should enforce:

            effective_limit = min(current_limit, --company-concurrency)
            while inflight_companies < effective_limit:
                start another company pipeline

        There is no can_start_new_worker, no admission scheduler, no local
        budgets. Slots are implied by the difference between effective_limit
        and the current number of inflight companies.

      - Once run.py decides to start another company (because there is
        a free slot under effective_limit), it should actually start it.
        AdaptiveConcurrencyController has already decided that extra load
        is acceptable.

    High level policy:

      - If mem >= target_mem_high or cpu >= target_cpu_high:
            scale down slowly (respecting scale_down_cooldown_sec)

      - Else if both mem <= target_mem_low and cpu <= target_cpu_low:
            scale up slowly (respecting scale_up_cooldown_sec)

      - Else:
            keep limit unchanged

      - Additionally, when the system is globally idle and flat
        (global stall), we temporarily ramp up more aggressively by
        multiplying scale_up_step by stall_boost_scale_up_multiplier.
    """

    def __init__(self, cfg: AdaptiveConcurrencyConfig) -> None:
        self.cfg = cfg

        # Initial limit is min_concurrency.
        self._current_limit: int = self._clamp_limit(cfg.min_concurrency)

        self._lock = asyncio.Lock()
        self._samples: List[Tuple[float, float, float]] = []  # (ts_mono, cpu_frac, mem_frac)
        self._task: Optional[asyncio.Task] = None

        # Last smoothed values
        self._last_avg_cpu: float = 0.0
        self._last_avg_mem: float = 0.0
        self._last_raw_cpu: float = 0.0
        self._last_raw_mem: float = 0.0
        self._last_update_ts: float = 0.0  # wall clock

        # Scale decision timestamps
        self._last_scale_up_ts: float = 0.0
        self._last_scale_down_ts: float = 0.0

        # Global stall boost state
        self._stall_boost_active: bool = False
        self._stall_boost_until_ts: float = 0.0
        self._last_stall_release_ts: float = 0.0

        # Memory pressure signals
        self._memory_pressure_hits: int = 0

        # Whether any real work has started
        self._has_seen_work: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self._task is not None:
            return
        interval = max(float(self.cfg.sample_interval_sec), 0.5)
        self._task = asyncio.create_task(self._loop(interval), name="adaptive-concurrency")
        logger.info(
            "[AdaptiveConcurrency] started (min=%d, max=%d, interval=%.2fs, psutil=%s)",
            self.cfg.min_concurrency,
            self.cfg.max_concurrency,
            interval,
            PSUTIL_AVAILABLE,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info(
            "[AdaptiveConcurrency] stopped (memory_pressure_hits=%d)",
            self._memory_pressure_hits,
        )

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                await self._sample_and_adjust()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[AdaptiveConcurrency] error in sampling loop")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ #
    # Public API for run.py
    # ------------------------------------------------------------------ #

    async def get_limit(self) -> int:
        async with self._lock:
            return self._current_limit

    async def set_limit_hard(self, value: int) -> None:
        async with self._lock:
            self._current_limit = self._clamp_limit(value)

    async def notify_work_started(self) -> None:
        async with self._lock:
            if not self._has_seen_work:
                self._has_seen_work = True
                logger.info("[AdaptiveConcurrency] first work observed; scale up enabled")

    async def on_memory_pressure(self) -> None:
        """
        Called when a critical memory pressure event is detected.

        This aggressively clamps the limit to min_concurrency and cancels
        any stall boost.
        """
        async with self._lock:
            self._memory_pressure_hits += 1
            self._current_limit = self._clamp_limit(self.cfg.min_concurrency)
            self._last_scale_down_ts = time.monotonic()
            self._stall_boost_active = False
            self._stall_boost_until_ts = 0.0
            self._last_stall_release_ts = time.monotonic()
            logger.warning(
                "[AdaptiveConcurrency] memory pressure event; forcing limit=%d (hits=%d)",
                self._current_limit,
                self._memory_pressure_hits,
            )

    async def get_state(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "current_limit": self._current_limit,
                "min_concurrency": self.cfg.min_concurrency,
                "max_concurrency": self.cfg.max_concurrency,
                "last_avg_cpu": self._last_avg_cpu,
                "last_avg_mem": self._last_avg_mem,
                "last_raw_cpu": self._last_raw_cpu,
                "last_raw_mem": self._last_raw_mem,
                "samples_count": len(self._samples),
                "last_update_ts": self._last_update_ts,
                "memory_pressure_hits": self._memory_pressure_hits,
                "stall_boost_active": self._stall_boost_active,
            }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _clamp_limit(self, value: int) -> int:
        return max(self.cfg.min_concurrency, min(self.cfg.max_concurrency, value))

    def _read_usage(self) -> Tuple[float, float]:
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0

        cpu_frac = 0.0
        mem_frac = 0.0

        try:
            cpu_pct = psutil.cpu_percent(interval=None)  # type: ignore[union-attr]
            cpu_frac = max(0.0, min(1.0, float(cpu_pct) / 100.0))
        except Exception:
            pass

        try:
            vm = psutil.virtual_memory()  # type: ignore[union-attr]
            mem_frac = max(0.0, min(1.0, float(vm.percent) / 100.0))
        except Exception:
            pass

        return cpu_frac, mem_frac

    def _maybe_detect_global_stall(
        self,
        now_mono: float,
        avg_cpu: float,
        avg_mem: float,
    ) -> None:
        """
        Detect global under utilization stall and activate or deactivate
        stall boost mode.

        Conditions for a stall:

          - We have seen work.
          - Limit significantly below max_concurrency.
          - CPU is mostly idle (avg_cpu <= stall_cpu_idle_threshold).
          - Memory is safe (below target_mem_high if memory is used).
          - Both CPU and memory are nearly flat over the window.
        """
        cfg = self.cfg

        # First, expire existing boost if needed.
        if self._stall_boost_active and now_mono >= self._stall_boost_until_ts:
            self._stall_boost_active = False
            self._stall_boost_until_ts = 0.0
            self._last_stall_release_ts = now_mono
            logger.info(
                "[AdaptiveConcurrency] stall boost period ended; returning to normal scaling"
            )

        # Cooldown after a boost before starting another.
        if (
            not self._stall_boost_active
            and now_mono - self._last_stall_release_ts < cfg.stall_release_cooldown_sec
        ):
            return

        if not self._has_seen_work:
            return
        if not PSUTIL_AVAILABLE:
            return

        window_len = max(float(cfg.stall_detection_window_sec), 0.0)
        if window_len <= 0.0:
            return

        cutoff = now_mono - window_len
        cpu_vals: List[float] = []
        mem_vals: List[float] = []

        for t, c, m in self._samples:
            if t >= cutoff:
                cpu_vals.append(c)
                mem_vals.append(m)

        if len(cpu_vals) < 2 or len(mem_vals) < 2:
            return

        cpu_span = max(cpu_vals) - min(cpu_vals)
        mem_span = max(mem_vals) - min(mem_vals)

        limit = self._current_limit
        under_provisioned = limit < int(self.cfg.max_concurrency * 0.9)
        cpu_idle = (not cfg.use_cpu) or avg_cpu <= cfg.stall_cpu_idle_threshold
        mem_safe = (not cfg.use_mem) or avg_mem <= cfg.target_mem_high
        flat_enough = cpu_span <= cfg.stall_mem_band_width and mem_span <= cfg.stall_mem_band_width

        if not (under_provisioned and cpu_idle and mem_safe and flat_enough):
            return

        if not self._stall_boost_active:
            self._stall_boost_active = True
            # Use a single stall window as boost period; you can tune this if needed.
            self._stall_boost_until_ts = now_mono + window_len
            logger.warning(
                "[AdaptiveConcurrency] global stall detected (limit=%d, max=%d, "
                "avg_cpu=%.3f, avg_mem=%.3f, cpu_span=%.3f, mem_span=%.3f); "
                "enabling stall boost.",
                limit,
                self.cfg.max_concurrency,
                avg_cpu,
                avg_mem,
                cpu_span,
                mem_span,
            )

    async def _sample_and_adjust(self) -> None:
        if not PSUTIL_AVAILABLE:
            async with self._lock:
                if self._current_limit != self.cfg.max_concurrency:
                    self._current_limit = self.cfg.max_concurrency
                    self._last_update_ts = time.time()
                    logger.debug(
                        "[AdaptiveConcurrency] psutil missing; forcing limit=%d",
                        self._current_limit,
                    )
            return

        raw_cpu, raw_mem = self._read_usage()
        now_mono = time.monotonic()
        wall_ts = time.time()
        cfg = self.cfg

        window = float(cfg.smoothing_window_sec) if cfg.smoothing_window_sec > 0.0 else 0.0

        async with self._lock:
            # Update sliding window
            self._samples.append((now_mono, raw_cpu, raw_mem))
            if window > 0.0:
                cutoff = now_mono - window
                self._samples = [
                    (t, c, m) for (t, c, m) in self._samples if t >= cutoff
                ]
            else:
                self._samples = self._samples[-1:]

            if self._samples:
                count = len(self._samples)
                sum_cpu = 0.0
                sum_mem = 0.0
                for _, c, m in self._samples:
                    sum_cpu += c
                    sum_mem += m
                avg_cpu = sum_cpu / count
                avg_mem = sum_mem / count
            else:
                avg_cpu = raw_cpu
                avg_mem = raw_mem

            # Stall detection can modify stall_boost flags.
            self._maybe_detect_global_stall(now_mono, avg_cpu, avg_mem)

            old_limit = self._current_limit
            new_limit = old_limit
            increased = False

            use_cpu = cfg.use_cpu
            use_mem = cfg.use_mem

            if not self._has_seen_work:
                new_limit = self._clamp_limit(cfg.min_concurrency)
            else:
                high_mem = use_mem and avg_mem >= cfg.target_mem_high
                high_cpu = use_cpu and avg_cpu >= cfg.target_cpu_high
                low_mem = (not use_mem) or avg_mem <= cfg.target_mem_low
                low_cpu = (not use_cpu) or avg_cpu <= cfg.target_cpu_low

                # Effective scale up step, optionally boosted during stall.
                scale_up_step = cfg.scale_up_step
                if self._stall_boost_active and scale_up_step > 0:
                    scale_up_step = max(
                        1,
                        int(round(scale_up_step * cfg.stall_boost_scale_up_multiplier)),
                    )

                now = now_mono

                # Scale down when overloaded
                if high_mem or high_cpu:
                    if (
                        cfg.scale_down_step > 0
                        and now - self._last_scale_down_ts >= cfg.scale_down_cooldown_sec
                    ):
                        new_limit = self._clamp_limit(old_limit - cfg.scale_down_step)
                        if new_limit != old_limit:
                            self._last_scale_down_ts = now

                # Scale up when comfortably low
                elif low_mem and low_cpu:
                    if (
                        scale_up_step > 0
                        and now - self._last_scale_up_ts >= cfg.scale_up_cooldown_sec
                    ):
                        new_limit = self._clamp_limit(old_limit + scale_up_step)
                        if new_limit != old_limit:
                            self._last_scale_up_ts = now
                            increased = True

                # Otherwise, keep limit unchanged

            self._current_limit = new_limit

            self._last_avg_cpu = avg_cpu
            self._last_avg_mem = avg_mem
            self._last_raw_cpu = raw_cpu
            self._last_raw_mem = raw_mem
            self._last_update_ts = wall_ts

            state_snapshot = {
                "ts": wall_ts,
                "monotonic": now_mono,
                "limit_old": old_limit,
                "limit_new": new_limit,
                "avg_cpu": avg_cpu,
                "avg_mem": avg_mem,
                "raw_cpu": raw_cpu,
                "raw_mem": raw_mem,
                "samples_window_sec": window,
                "samples_count": len(self._samples),
                "memory_pressure_hits": self._memory_pressure_hits,
                "stall_boost_active": self._stall_boost_active,
            }

        # Outside lock: logging
        self._maybe_log_state(state_snapshot)

        if state_snapshot["limit_new"] != state_snapshot["limit_old"]:
            logger.info(
                "[AdaptiveConcurrency] limit change %d -> %d "
                "(avg_cpu=%.3f, avg_mem=%.3f, raw_cpu=%.3f, raw_mem=%.3f, stall_boost=%s)",
                state_snapshot["limit_old"],
                state_snapshot["limit_new"],
                state_snapshot["avg_cpu"],
                state_snapshot["avg_mem"],
                state_snapshot["raw_cpu"],
                state_snapshot["raw_mem"],
                state_snapshot["stall_boost_active"],
            )
        else:
            logger.debug(
                "[AdaptiveConcurrency] limit=%d (avg_cpu=%.3f, avg_mem=%.3f, "
                "stall_boost=%s)",
                state_snapshot["limit_new"],
                state_snapshot["avg_cpu"],
                state_snapshot["avg_mem"],
                state_snapshot["stall_boost_active"],
            )

    def _maybe_log_state(self, state: Dict[str, Any]) -> None:
        if self.cfg.log_path is None:
            return

        try:
            self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.log_path.open("a", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            logger.debug(
                "[AdaptiveConcurrency] failed writing log to %s",
                self.cfg.log_path,
                exc_info=True,
            )


__all__ = [
    "AdaptiveConcurrencyConfig",
    "AdaptiveConcurrencyController",
]
