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

    target_cpu_low / target_cpu_high:
        CPU usage band (fraction 0..1). We only scale up when both
        CPU and memory are below the lower band (if enabled). We
        scale down when either enabled dimension is above the upper band.

    sample_interval_sec:
        How often to sample CPU/memory and recompute the limit.

    smoothing_window_sec:
        Sliding window length for smoothing the samples.

    log_path:
        Optional path to a line based JSON log for debugging decisions.

    use_cpu / use_mem:
        Whether to take CPU and/or memory into account.

    scale_up_step / scale_down_step:
        How many slots to add or remove per scale decision.

    scale_up_cooldown_sec / scale_down_cooldown_sec:
        Minimum time between consecutive scale up or scale down decisions.
        NOTE: scale_up_cooldown_sec is now respected *strictly*; we no longer
        shrink it dynamically.

    startup_mem_headroom / startup_cpu_headroom:
        Extra safety margin before allowing new workers to start.

    admission_cooldown_sec:
        Minimum gap between admitting new workers when resource usage is in
        the caution band (between lower and upper threshold).

    caution_band_fraction:
        Local admission budget size as a fraction of the last safe limit.

    stall_*:
        Parameters for detecting a "caution-zone hurdle" where memory is in the
        caution band, CPU is idle and memory is flat. When detected, we relax
        caution limits temporarily.

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
    scale_up_cooldown_sec: float = 60.0
    scale_down_cooldown_sec: float = 0.0

    # Extra safety margin before allowing new workers to start
    startup_mem_headroom: float = 0.03
    startup_cpu_headroom: float = 0.06

    # How slowly to admit new workers when we enter the caution band
    admission_cooldown_sec: float = 60.0

    # Local admission budget in the memory caution band.
    caution_band_fraction: float = 0.2

    # Caution-zone stall detection / release
    stall_detection_window_sec: float = 30.0
    stall_cpu_idle_threshold: float = 0.05
    stall_mem_band_width: float = 0.05
    stall_release_cooldown_sec: float = 30.0


class AdaptiveConcurrencyController:
    """
    Adaptive controller that adjusts an integer concurrency limit based on
    smoothed CPU and memory usage.

    Policy (using averages over a sliding window):

      - If mem >= target_mem_high OR cpu >= target_cpu_high:
          - Do not admit new workers.
          - Potentially scale down the limit (down to min_concurrency).

      - Else if mem >= target_mem_low OR cpu >= target_cpu_low:
          - Admit new workers slowly:
              - at most one every admission_cooldown_sec
              - and, if memory is in the caution band (low < mem < high),
                at most 'band_budget' workers per entry into that band.

      - Else (both below their lower thresholds):
          - Admit new workers freely (subject to limit) and allow slow scale up
            controlled by *fixed* scale_up_cooldown_sec.

    Additionally, when we detect that we are stuck in the caution band with
    very low CPU and almost flat memory usage, we temporarily relax the
    caution-band admission limits so the run does not stall.
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
        self._zone_mem: str = "unknown"
        self._last_low_mem_limit: int = self._current_limit
        self._band_admission_budget: int = 0

        # Caution-zone stall override
        self._stall_override_active: bool = False
        self._last_stall_release_ts: float = 0.0

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
        Forcefully override the current concurrency limit.
        """
        async with self._lock:
            self._current_limit = self._clamp_limit(value)

    async def on_memory_pressure(self) -> None:
        """
        Called when a critical memory pressure event is detected.
        """
        async with self._lock:
            self._memory_pressure_hits += 1
            self._current_limit = self.cfg.min_concurrency
            self._last_scale_down_ts = time.monotonic()
            self._zone_mem = "high"
            self._band_admission_budget = 0
            self._stall_override_active = False
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
                "last_low_limit": self._last_low_mem_limit,
                "band_admission_budget": self._band_admission_budget,
                "stall_override_active": self._stall_override_active,
            }

    async def notify_work_started(self) -> None:
        """
        Mark that at least one company has actually started doing work.
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
        """
        async with self._lock:
            if not self._has_seen_work:
                return True

            if not PSUTIL_AVAILABLE:
                return True

            avg_mem = self._last_avg_mem
            avg_cpu = self._last_avg_cpu
            use_mem = self.cfg.use_mem
            use_cpu = self.cfg.use_cpu

            now = time.monotonic()

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

            if high_mem or high_cpu:
                return False

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
                if (
                    mem_in_caution_band
                    and self.cfg.caution_band_fraction > 0.0
                    and self._band_admission_budget <= 0
                    and not self._stall_override_active
                ):
                    return False

                cooldown = max(
                    self.cfg.admission_cooldown_sec, self.cfg.sample_interval_sec
                )

                if self._stall_override_active:
                    cooldown = self.cfg.sample_interval_sec

                if now - self._last_admission_ts < cooldown:
                    return False

                self._last_admission_ts = now

                if (
                    mem_in_caution_band
                    and self.cfg.caution_band_fraction > 0.0
                    and not self._stall_override_active
                ):
                    self._band_admission_budget -= 1

                return True

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

    def _maybe_detect_caution_stall(
        self,
        now_mono: float,
        avg_cpu: float,
        avg_mem: float,
    ) -> None:
        """
        Detect the "caution-zone hurdle" case and, if needed, enable an
        override that relaxes admission limits in the caution band.
        """
        cfg = self.cfg

        if not cfg.use_mem:
            self._stall_override_active = False
            return

        if self._zone_mem != "caution":
            if self._stall_override_active:
                self._stall_override_active = False
                self._last_stall_release_ts = now_mono
            return

        if cfg.use_cpu and avg_cpu > cfg.stall_cpu_idle_threshold:
            if self._stall_override_active:
                self._stall_override_active = False
                self._last_stall_release_ts = now_mono
            return

        if (
            self._stall_override_active is False
            and now_mono - self._last_stall_release_ts
            < cfg.stall_release_cooldown_sec
        ):
            return

        window_len = max(float(cfg.stall_detection_window_sec), 0.0)
        if window_len <= 0.0:
            return

        cutoff = now_mono - window_len
        mem_vals: List[float] = [m for (t, _, m) in self._samples if t >= cutoff]

        if len(mem_vals) < 2:
            return

        mem_span = max(mem_vals) - min(mem_vals)

        if mem_span <= cfg.stall_mem_band_width:
            if not self._stall_override_active:
                self._stall_override_active = True
                base = max(self._current_limit, self._last_low_mem_limit, 1)
                self._band_admission_budget = max(
                    self._band_admission_budget,
                    max(base, self.cfg.max_concurrency),
                )
                logger.warning(
                    "[AdaptiveConcurrency] detected caution-zone stall "
                    "(avg_cpu=%.3f, avg_mem=%.3f, mem_span=%.3f over %.1fs); "
                    "opening admission in caution band (budget=%d)",
                    avg_cpu,
                    avg_mem,
                    mem_span,
                    window_len,
                    self._band_admission_budget,
                )
        else:
            if self._stall_override_active:
                self._stall_override_active = False
                self._last_stall_release_ts = now_mono
                logger.info(
                    "[AdaptiveConcurrency] caution-zone stall cleared; "
                    "restoring normal admission controls."
                )

    async def _sample_and_adjust(self) -> None:
        """
        Take one resource usage sample and adjust concurrency limit if needed.
        """
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
        now_mono = time.monotonic()
        wall_ts = time.time()
        cfg = self.cfg

        window = float(cfg.smoothing_window_sec) if cfg.smoothing_window_sec > 0.0 else 0.0

        async with self._lock:
            # Sliding window update
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

            # Memory zone classification
            prev_zone = self._zone_mem
            if not cfg.use_mem:
                new_zone = "safe"
            else:
                if avg_mem >= cfg.target_mem_high:
                    new_zone = "high"
                elif avg_mem >= cfg.target_mem_low:
                    new_zone = "caution"
                else:
                    new_zone = "safe"

            if new_zone != prev_zone:
                if new_zone == "safe":
                    self._last_low_mem_limit = self._current_limit
                    self._band_admission_budget = 0
                elif new_zone == "caution":
                    base = self._last_low_mem_limit or self._current_limit
                    frac = max(cfg.caution_band_fraction, 0.0)
                    self._band_admission_budget = int(base * frac)
                else:
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

            # Caution-zone stall detection
            self._maybe_detect_caution_stall(now_mono, avg_cpu, avg_mem)

            # Scaling logic
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

                scale_up_step = cfg.scale_up_step
                scale_up_cooldown = cfg.scale_up_cooldown_sec

                # Scale down if needed (respecting cooldown)
                if high_mem or high_cpu:
                    if (
                        cfg.scale_down_step > 0
                        and now_mono - self._last_scale_down_ts
                        >= cfg.scale_down_cooldown_sec
                    ):
                        new_limit = self._clamp_limit(
                            old_limit - cfg.scale_down_step
                        )
                        if new_limit != old_limit:
                            self._last_scale_down_ts = now_mono

                # Scale up only when both dimensions are comfortably low,
                # and respecting the *fixed* scale_up_cooldown_sec.
                elif low_mem and low_cpu:
                    if (
                        scale_up_step > 0
                        and now_mono - self._last_scale_up_ts >= scale_up_cooldown
                    ):
                        new_limit = self._clamp_limit(old_limit + scale_up_step)
                        if new_limit != old_limit:
                            self._last_scale_up_ts = now_mono
                            increased = True

            self._current_limit = new_limit

            if increased and new_limit > old_limit:
                # Let waiting workers start right away.
                self._last_admission_ts = 0.0

            # Persist last readings
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
                "zone_mem": self._zone_mem,
                "last_low_limit": self._last_low_mem_limit,
                "band_admission_budget": self._band_admission_budget,
                "stall_override_active": self._stall_override_active,
            }

        self._maybe_log_state(state_snapshot)

        if state_snapshot["limit_new"] != state_snapshot["limit_old"]:
            logger.info(
                "[AdaptiveConcurrency] limit change %d -> %d "
                "(avg_cpu=%.3f, avg_mem=%.3f, raw_cpu=%.3f, raw_mem=%.3f, "
                "zone=%s, budget=%d, last_low_limit=%d, stall_override=%s)",
                state_snapshot["limit_old"],
                state_snapshot["limit_new"],
                state_snapshot["avg_cpu"],
                state_snapshot["avg_mem"],
                state_snapshot["raw_cpu"],
                state_snapshot["raw_mem"],
                state_snapshot["zone_mem"],
                state_snapshot["band_admission_budget"],
                state_snapshot["last_low_limit"],
                state_snapshot["stall_override_active"],
            )
        else:
            logger.debug(
                "[AdaptiveConcurrency] limit=%d (avg_cpu=%.3f, avg_mem=%.3f, "
                "zone=%s, budget=%d, last_low_limit=%d, stall_override=%s)",
                state_snapshot["limit_new"],
                state_snapshot["avg_cpu"],
                state_snapshot["avg_mem"],
                state_snapshot["zone_mem"],
                state_snapshot["band_admission_budget"],
                state_snapshot["last_low_limit"],
                state_snapshot["stall_override_active"],
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
