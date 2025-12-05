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

    max_concurrency:
        Hard upper bound (typically from --company-concurrency).

    min_concurrency:
        Minimum allowed concurrency. Usually 1.

    target_mem_low / target_mem_high:
        Memory usage band (fraction 0..1) that we aim to sit in.
        Example: 0.88 and 0.95 => 88 percent to 95 percent.

    target_cpu_low / target_cpu_high:
        CPU usage band (fraction 0..1). We only scale up when both
        CPU and memory are below the lower band (if enabled). We
        scale down when either enabled dimension is above the upper band.

    sample_interval_sec:
        How often to sample CPU/memory and recompute the limit.

    smoothing_window_sec:
        Sliding window length for smoothing the samples. The controller
        uses the average CPU and memory across this window.

    log_path:
        Optional path to a line based JSON log for debugging decisions.

    use_cpu / use_mem:
        Whether to take CPU and/or memory into account. This lets you
        run in CPU-only mode, mem-only mode, or disable both (which
        effectively drives concurrency toward max_concurrency).

    scale_up_step / scale_down_step:
        How many slots to add or remove per scale decision.

    scale_up_cooldown_sec / scale_down_cooldown_sec:
        Minimum time between consecutive scale up or scale down decisions.

    startup_mem_headroom / startup_cpu_headroom:
        Extra safety margin before allowing new workers to start.
        Kept for compatibility but the admission logic now uses
        explicit lower/upper thresholds instead.

    admission_cooldown_sec:
        Minimum gap between admitting new workers when resource
        usage is in the caution band (between lower and upper threshold).
        Below lower thresholds we admit freely (subject to limit).
        Above upper thresholds we stop admitting completely.

    caution_band_fraction:
        When memory is in the caution band (target_mem_low < mem < target_mem_high),
        we apply a local admission budget that limits how many new workers
        can be admitted in this band.

        The budget is computed as:

            band_budget = int(last_low_mem_limit * caution_band_fraction)

        where last_low_mem_limit is the concurrency limit recorded the last
        time smoothed memory was strictly below target_mem_low.

        The budget is:
          - reset whenever memory goes back below target_mem_low
          - recomputed whenever memory goes from below-low into the caution band

        This reduces the risk that you sit at mem just below target_mem_high
        and suddenly admit a large number of extra tasks that push memory to 100%.
    """

    max_concurrency: int
    min_concurrency: int = 1
    target_mem_low: float = 0.83
    target_mem_high: float = 0.90
    target_cpu_low: float = 0.90
    target_cpu_high: float = 1.00
    sample_interval_sec: float = 2.0
    smoothing_window_sec: float = 10.0
    log_path: Optional[Path] = None
    use_cpu: bool = True
    use_mem: bool = True

    # How fast we move when scaling up or down
    scale_up_step: int = 1
    scale_down_step: int = 1

    # Minimum time between consecutive scale up or scale down decisions
    scale_up_cooldown_sec: float = 5.0
    scale_down_cooldown_sec: float = 0.0

    # Extra safety margin before allowing new workers to start
    # (kept for backward compatibility but no longer central)
    startup_mem_headroom: float = 0.03
    startup_cpu_headroom: float = 0.05

    # How slowly to admit new workers when we enter the caution band
    # between target_mem_low/target_mem_high or target_cpu_low/target_cpu_high.
    admission_cooldown_sec: float = 20.0

    # New: local admission budget in the memory caution band.
    # See the class docstring for details.
    caution_band_fraction: float = 0.5


class AdaptiveConcurrencyController:
    """
    Adaptive controller that adjusts an integer concurrency limit based on
    smoothed CPU and memory usage.

    Policy (using averages over a sliding window):

      - If mem >= target_mem_high OR cpu >= target_cpu_high:
          - Do not admit new workers (can_start_new_worker returns False).
          - Potentially scale down the limit (down to min_concurrency).

      - Else if mem >= target_mem_low OR cpu >= target_cpu_low:
          - Admit new workers slowly:
              - at most one every admission_cooldown_sec
              - and, if memory is in the caution band (low < mem < high),
                at most 'band_budget' workers per entry into that band.

      - Else (both below their lower thresholds):
          - Admit new workers freely (subject to limit) and allow slow scale up
            controlled by scale_up_cooldown_sec.

    Host resource usage is sampled with psutil if available. If psutil is
    not available, the controller simply keeps the limit at max_concurrency
    and always admits new workers.
    """

    def __init__(self, cfg: AdaptiveConcurrencyConfig) -> None:
        self.cfg = cfg

        # Clamp initial limit into [min_concurrency, max_concurrency]
        self._current_limit: int = max(
            cfg.min_concurrency, min(cfg.max_concurrency, cfg.min_concurrency)
        )

        self._lock = asyncio.Lock()
        self._samples: List[Tuple[float, float, float]] = []  # (ts, cpu_frac, mem_frac)
        self._task: Optional[asyncio.Task] = None

        self._memory_pressure_hits: int = 0

        # Last smoothed values (for introspection and logging)
        self._last_avg_cpu: float = 0.0
        self._last_avg_mem: float = 0.0
        self._last_raw_cpu: float = 0.0
        self._last_raw_mem: float = 0.0
        self._last_update_ts: float = 0.0

        # Track last scale up / scale down moments (monotonic time)
        self._last_scale_up_ts: float = 0.0
        self._last_scale_down_ts: float = 0.0

        # Track when we last admitted a new worker
        self._last_admission_ts: float = 0.0

        # Do not scale up until at least one company has actually started work
        self._has_seen_work: bool = False

        # Memory zone and band admission control
        # zone_mem: "safe" (< low), "caution" (low <= mem < high),
        #           "high" (>= high), "unknown" (initial)
        self._zone_mem: str = "unknown"
        self._last_low_mem_limit: int = self._current_limit
        self._band_admission_budget: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Start the adaptive sampling loop as a background task.
        """
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="adaptive-concurrency")
        logger.info(
            "[AdaptiveConcurrency] started (min=%d, max=%d, interval=%.2fs, "
            "psutil_available=%s)",
            self.cfg.min_concurrency,
            self.cfg.max_concurrency,
            self.cfg.sample_interval_sec,
            PSUTIL_AVAILABLE,
        )

    async def stop(self) -> None:
        """
        Stop the background task.
        """
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

    async def _loop(self) -> None:
        """
        Periodic sampling loop.
        """
        interval = max(float(self.cfg.sample_interval_sec), 0.5)
        while True:
            try:
                await self._sample_and_adjust()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[AdaptiveConcurrency] error in sampling loop")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ #
    # Public API used by run.py
    # ------------------------------------------------------------------ #

    async def get_limit(self) -> int:
        """
        Get the current concurrency limit.
        """
        async with self._lock:
            return self._current_limit

    async def set_limit_hard(self, value: int) -> None:
        """
        Forcefully override the current concurrency limit (clamped to
        [min_concurrency, max_concurrency]).
        """
        async with self._lock:
            self._current_limit = self._clamp_limit(value)

    async def on_memory_pressure(self) -> None:
        """
        Called when a critical memory pressure event is detected (for example
        from MemoryGuard). Uses an aggressive backoff policy:

          - Increment memory_pressure_hits.
          - Drop the limit to min_concurrency.
        """
        async with self._lock:
            self._memory_pressure_hits += 1
            self._current_limit = self.cfg.min_concurrency
            # Treat this as a recent scale down so we do not immediately
            # climb back up.
            self._last_scale_down_ts = time.monotonic()
            # Also treat this like a high memory zone event.
            self._zone_mem = "high"
            self._band_admission_budget = 0
            logger.warning(
                "[AdaptiveConcurrency] memory pressure event; forcing limit=%d "
                "(hits=%d)",
                self._current_limit,
                self._memory_pressure_hits,
            )

    async def get_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of internal state for debugging or metrics.
        """
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
                "memory_pressure_hits": self._memory_pressure_hits,
                "last_update_ts": self._last_update_ts,
                "psutil_available": PSUTIL_AVAILABLE,
                "zone_mem": self._zone_mem,
                "last_low_mem_limit": self._last_low_mem_limit,
                "band_admission_budget": self._band_admission_budget,
            }

    async def notify_work_started(self) -> None:
        """
        Mark that at least one company has actually started doing work.
        Before this flag is set, the controller will keep the limit at
        min_concurrency and will not scale up based on idle CPU or RAM.
        """
        async with self._lock:
            if not self._has_seen_work:
                self._has_seen_work = True
                logger.info(
                    "[AdaptiveConcurrency] first work observed; enabling scale up."
                )

    async def can_start_new_worker(self) -> bool:
        """
        Decide if it is safe to start one more worker right now, based on
        the latest smoothed CPU and memory.

        - If avg mem or cpu is above the upper threshold, deny admission.
        - If any is above the lower threshold, allow admission only once
          every admission_cooldown_sec and, if memory is in the caution
          band (low < mem < high), also consume from a finite admission
          budget. When the budget reaches zero, further admissions are
          denied until memory returns to the safe zone again.
        - Otherwise, admit freely (subject to the current limit).

        This does not look at the active count, only at resource headroom.
        wait_for_adaptive_slot still enforces active_count < limit.
        """
        async with self._lock:
            # If no work has started yet, allow the first workers to bootstrap.
            if not self._has_seen_work:
                return True

            if not PSUTIL_AVAILABLE:
                # No resource signal, so we cannot do admission control based
                # on RAM or CPU. Fall back to always allowing new workers.
                return True

            avg_mem = self._last_avg_mem
            avg_cpu = self._last_avg_cpu
            use_mem = self.cfg.use_mem
            use_cpu = self.cfg.use_cpu

            now = time.monotonic()

            # If we have never recorded a sample yet, admit slowly to avoid
            # spiking before we have any feedback.
            if self._last_update_ts <= 0.0:
                cooldown = max(
                    self.cfg.admission_cooldown_sec, self.cfg.sample_interval_sec
                )
                if now - self._last_admission_ts < cooldown:
                    return False
                self._last_admission_ts = now
                return True

            high_mem = use_mem and avg_mem >= self.cfg.target_mem_high
            high_cpu = use_cpu and avg_cpu >= self.cfg.target_cpu_high

            # Upper threshold: stop admitting entirely until usage drops.
            if high_mem or high_cpu:
                return False

            # Caution band: above lower threshold on any enabled dimension.
            in_caution = False
            if use_mem and avg_mem >= self.cfg.target_mem_low:
                in_caution = True
            if use_cpu and avg_cpu >= self.cfg.target_cpu_low:
                in_caution = True

            mem_in_caution_band = (
                use_mem
                and avg_mem >= self.cfg.target_mem_low
                and avg_mem < self.cfg.target_mem_high
            )

            if in_caution:
                # If memory is in the caution band, enforce the local admission
                # budget based on the last safe limit.
                if (
                    mem_in_caution_band
                    and self.cfg.caution_band_fraction > 0.0
                    and self._band_admission_budget <= 0
                ):
                    # Budget exhausted for this band episode.
                    return False

                cooldown = max(
                    self.cfg.admission_cooldown_sec, self.cfg.sample_interval_sec
                )
                if now - self._last_admission_ts < cooldown:
                    return False

                self._last_admission_ts = now

                if mem_in_caution_band and self.cfg.caution_band_fraction > 0.0:
                    # Consume from the band admission budget.
                    self._band_admission_budget -= 1

                return True

            # Safe zone: both dimensions below their lower thresholds.
            # We still record admission time so that the bootstrap logic
            # in the "no samples yet" path behaves consistently.
            self._last_admission_ts = now
            return True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _clamp_limit(self, value: int) -> int:
        return max(self.cfg.min_concurrency, min(self.cfg.max_concurrency, value))

    def _read_usage(self) -> Tuple[float, float]:
        """
        Return (cpu_frac, mem_frac) where each is in [0.0, 1.0].

        cpu_frac is system wide CPU percent divided by 100.
        mem_frac is system memory used percent divided by 100.

        If psutil is not available or reading fails, returns (0.0, 0.0).
        """
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

    async def _sample_and_adjust(self) -> None:
        """
        Take one resource usage sample and adjust concurrency limit if needed.
        """
        # If psutil is not available, just keep max concurrency and do nothing.
        if not PSUTIL_AVAILABLE:
            async with self._lock:
                if self._current_limit != self.cfg.max_concurrency:
                    self._current_limit = self.cfg.max_concurrency
                    logger.debug(
                        "[AdaptiveConcurrency] psutil missing; forcing limit=%d",
                        self._current_limit,
                    )
            return

        raw_cpu, raw_mem = self._read_usage()
        now = time.monotonic()
        wall_ts = time.time()

        if self.cfg.smoothing_window_sec <= 0.0:
            window = 0.0
        else:
            window = float(self.cfg.smoothing_window_sec)

        # Compute new limit and smoothed averages under lock
        async with self._lock:
            self._samples.append((now, raw_cpu, raw_mem))

            if window > 0.0:
                cutoff = now - window
                self._samples = [
                    (t, c, m) for (t, c, m) in self._samples if t >= cutoff
                ]
            else:
                # Keep only the most recent sample if smoothing is disabled
                self._samples = self._samples[-1:]

            if self._samples:
                avg_cpu = sum(c for _, c, _ in self._samples) / len(self._samples)
                avg_mem = sum(m for _, _, m in self._samples) / len(self._samples)
            else:
                avg_cpu = raw_cpu
                avg_mem = raw_mem

            # Update memory zone and band admission budget based on smoothed memory
            prev_zone = self._zone_mem
            if not self.cfg.use_mem:
                # If memory is disabled, treat it as always safe
                new_zone = "safe"
            else:
                if avg_mem >= self.cfg.target_mem_high:
                    new_zone = "high"
                elif avg_mem >= self.cfg.target_mem_low:
                    new_zone = "caution"
                else:
                    new_zone = "safe"

            if new_zone != prev_zone:
                if new_zone == "safe":
                    # Record the limit that was stable at low memory and reset budget.
                    self._last_low_mem_limit = self._current_limit
                    self._band_admission_budget = 0
                elif new_zone == "caution":
                    # Entering caution band: compute new budget based on the
                    # last safe limit. If that is zero, fall back to current limit.
                    base = self._last_low_mem_limit or self._current_limit
                    frac = max(self.cfg.caution_band_fraction, 0.0)
                    self._band_admission_budget = int(base * frac)
                else:
                    # High memory: no new budget here.
                    self._band_admission_budget = 0

                self._zone_mem = new_zone
                logger.debug(
                    "[AdaptiveConcurrency] mem zone change %s -> %s "
                    "(avg_mem=%.3f, budget=%d, last_low_limit=%d)",
                    prev_zone,
                    new_zone,
                    avg_mem,
                    self._band_admission_budget,
                    self._last_low_mem_limit,
                )

            old_limit = self._current_limit
            new_limit = old_limit
            increased = False  # track if we actually scaled up

            use_cpu = self.cfg.use_cpu
            use_mem = self.cfg.use_mem

            if not self._has_seen_work:
                # Before any company has actually started work, do not scale up
                # just because the machine is idle. Keep the limit fixed at min.
                new_limit = self._clamp_limit(self.cfg.min_concurrency)
            else:
                # Interpret disabled dimensions as:
                #   - they never trigger "high" conditions
                #   - they are always considered "low" for scaling up decisions
                high_mem = use_mem and avg_mem >= self.cfg.target_mem_high
                high_cpu = use_cpu and avg_cpu >= self.cfg.target_cpu_high
                low_mem = (not use_mem) or avg_mem <= self.cfg.target_mem_low
                low_cpu = (not use_cpu) or avg_cpu <= self.cfg.target_cpu_low

                now_mono = now

                # Decrease if any enabled dimension is too high, but respect cooldown
                if high_mem or high_cpu:
                    if (
                        self.cfg.scale_down_step > 0
                        and now_mono - self._last_scale_down_ts
                        >= self.cfg.scale_down_cooldown_sec
                    ):
                        new_limit = self._clamp_limit(
                            old_limit - self.cfg.scale_down_step
                        )
                        if new_limit != old_limit:
                            self._last_scale_down_ts = now_mono

                # Increase if all enabled dimensions are comfortably low,
                # and enough time has passed since last scale up.
                elif low_mem and low_cpu:
                    if (
                        self.cfg.scale_up_step > 0
                        and now_mono - self._last_scale_up_ts
                        >= self.cfg.scale_up_cooldown_sec
                    ):
                        new_limit = self._clamp_limit(
                            old_limit + self.cfg.scale_up_step
                        )
                        if new_limit != old_limit:
                            self._last_scale_up_ts = now_mono
                            increased = True

            self._current_limit = new_limit

            # If we scaled up the limit, drop the admission cooldown so that
            # waiting workers can start immediately.
            if increased and new_limit > old_limit:
                self._last_admission_ts = 0.0

            self._last_avg_cpu = avg_cpu
            self._last_avg_mem = avg_mem
            self._last_raw_cpu = raw_cpu
            self._last_raw_mem = raw_mem
            self._last_update_ts = wall_ts

            state_snapshot = {
                "ts": wall_ts,
                "monotonic": now,
                "limit_old": old_limit,
                "limit_new": new_limit,
                "avg_cpu": avg_cpu,
                "avg_mem": avg_mem,
                "raw_cpu": raw_cpu,
                "raw_mem": raw_mem,
                "samples_window_sec": window,
                "samples_count": len(self._samples),
                "memory_pressure_hits": self._memory_pressure_hits,
                "zone_mem": self._zone_mem,
                "last_low_mem_limit": self._last_low_mem_limit,
                "band_admission_budget": self._band_admission_budget,
            }

        # Logging outside the lock
        self._maybe_log_state(state_snapshot)

        if state_snapshot["limit_new"] != state_snapshot["limit_old"]:
            logger.info(
                "[AdaptiveConcurrency] limit change %d -> %d "
                "(avg_cpu=%.3f, avg_mem=%.3f, raw_cpu=%.3f, raw_mem=%.3f, zone=%s, "
                "budget=%d, last_low_limit=%d)",
                state_snapshot["limit_old"],
                state_snapshot["limit_new"],
                state_snapshot["avg_cpu"],
                state_snapshot["avg_mem"],
                state_snapshot["raw_cpu"],
                state_snapshot["raw_mem"],
                state_snapshot["zone_mem"],
                state_snapshot["band_admission_budget"],
                state_snapshot["last_low_mem_limit"],
            )
        else:
            logger.debug(
                "[AdaptiveConcurrency] limit=%d (avg_cpu=%.3f, avg_mem=%.3f, "
                "zone=%s, budget=%d, last_low_limit=%d)",
                state_snapshot["limit_new"],
                state_snapshot["avg_cpu"],
                state_snapshot["avg_mem"],
                state_snapshot["zone_mem"],
                state_snapshot["band_admission_budget"],
                state_snapshot["last_low_mem_limit"],
            )

    def _maybe_log_state(self, state: Dict[str, Any]) -> None:
        """
        Append a single JSON line with the given state snapshot if log_path
        is configured.
        """
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
