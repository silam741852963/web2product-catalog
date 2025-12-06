from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSchedulingConfig:
    """
    Configuration for adaptive scheduling based on host CPU and memory.

    This module does not expose any "limit" or slot concept.

    Semantics:

      - run.py owns:
          * The list of companies to run.
          * The creation and tracking of asyncio.Task per company.
          * The count of active company tasks (active_tasks).

      - AdaptiveScheduler:
          * Periodically samples CPU and memory and maintains smoothed values.
          * Classifies the current load into three bands:
                - low band:  both CPU and mem at or below target_*_low
                - cautious: between low and high thresholds
                - high:     CPU or mem at or above target_*_high
          * Decides whether it is safe to admit a single new company.
          * Enforces cooldowns:
                - between successful admissions
                - extra cooldown when in the cautious band
                - a recheck cooldown for the high band
          * Can enter a "rush" mode when CPU and memory are underused
            and stable, which speeds up admissions for a short period.
          * Enforces a hard cap via max_concurrency, using active_tasks.

      - The caller uses:
            ok, reason = await scheduler.can_start_new_company(active_tasks)

        If ok is True:
            - run.py should start exactly one new company task
            - and may call scheduler.notify_admitted() for extra logging.

        If ok is False:
            - run.py should not start a new company
            - and may log or inspect the returned reason.

    Fields:

      max_concurrency:
          Hard upper bound on concurrently running companies.

      target_mem_low / target_mem_high:
          Memory usage band (fraction 0..1).

      target_cpu_low / target_cpu_high:
          CPU usage band (fraction 0..1).

      sample_interval_sec:
          How often to sample CPU and memory in the background.

      smoothing_window_sec:
          Sliding window length for smoothing samples.

      log_path:
          Optional line based JSON log of sampling state.

      use_cpu / use_mem:
          Whether to consider CPU and/or memory in admission decisions.

      admission_cooldown_sec:
          Minimum time between any two successful admissions
          across all bands.

      cautious_admission_cooldown_sec:
          Extra minimum time between admissions that happen
          while in the cautious band. This makes the mid band
          slower than the low band.

      block_cooldown_sec:
          Minimum time to wait between high band rechecks.
          While in high band, no new companies are admitted.

      mem_pressure_block_sec:
          Extra block period applied when on_memory_pressure is invoked.

      rush_*:
          Settings for rush mode that speeds up admissions when
          CPU and memory are low and stable inside a ±5 percent band.
    """

    max_concurrency: int

    target_mem_low: float = 0.50
    target_mem_high: float = 0.80
    target_cpu_low: float = 0.85
    target_cpu_high: float = 0.95

    sample_interval_sec: float = 1.0
    smoothing_window_sec: float = 10.0

    log_path: Optional[Path] = None

    use_cpu: bool = True
    use_mem: bool = True

    # Cooldowns
    admission_cooldown_sec: float = 5.0
    cautious_admission_cooldown_sec: float = 15.0
    block_cooldown_sec: float = 30.0

    mem_pressure_block_sec: float = 15.0

    # Rush / plateau settings
    rush_enabled: bool = True
    rush_cpu_threshold: float = 0.50     # 50 percent
    rush_mem_threshold: float = 0.80     # 80 percent
    rush_band_width: float = 0.05        # ±5 percent band
    rush_min_samples: int = 4
    rush_duration_sec: float = 30.0


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class AdaptiveScheduler:
    """
    Adaptive scheduler that gates admission of new company tasks based
    on smoothed CPU and memory usage, three-band classification, rush
    mode, and cooldowns.

    It does not manage inflight sets, slots or limits. It only answers
    "is it ok to start one more company now" and leaves everything else
    to run.py.
    """

    def __init__(self, cfg: AdaptiveSchedulingConfig) -> None:
        self.cfg = cfg

        self._lock = asyncio.Lock()
        self._samples: List[Tuple[float, float, float]] = []  # (ts_mono, cpu_frac, mem_frac)
        self._task: Optional[asyncio.Task] = None

        self._last_avg_cpu: float = 0.0
        self._last_avg_mem: float = 0.0
        self._last_raw_cpu: float = 0.0
        self._last_raw_mem: float = 0.0
        self._last_update_ts: float = 0.0  # wall clock

        # Cooldown timestamps (monotonic)
        self._last_admission_ts: float = 0.0
        self._last_cautious_admission_ts: float = 0.0
        self._last_high_block_ts: float = 0.0
        self._blocked_until_ts: float = 0.0

        # Memory pressure tracking
        self._memory_pressure_hits: int = 0

        # Rush mode tracking (monotonic times)
        self._rush_active: bool = False
        self._rush_start_ts: float = 0.0
        self._rush_until_ts: float = 0.0
        self._rush_start_cpu: float = 0.0
        self._rush_start_mem: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Start background sampling of CPU and memory.
        """
        if self._task is not None:
            return
        interval = max(float(self.cfg.sample_interval_sec), 0.5)
        self._task = asyncio.create_task(
            self._loop(interval),
            name="adaptive-scheduling-sampler",
        )
        logger.info(
            "[AdaptiveScheduling] started "
            "(max_concurrency=%d, interval=%.2fs, psutil=%s)",
            self.cfg.max_concurrency,
            interval,
            PSUTIL_AVAILABLE,
        )

    async def stop(self) -> None:
        """
        Stop background sampling.
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
            "[AdaptiveScheduling] stopped (memory_pressure_hits=%d)",
            self._memory_pressure_hits,
        )

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                await self._sample_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[AdaptiveScheduling] error in sampling loop")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ #
    # Public API for run.py
    # ------------------------------------------------------------------ #

    @property
    def sample_interval_sec(self) -> float:
        """
        Convenience accessor for callers that want to align their waits
        with the sampling cadence.
        """
        return float(self.cfg.sample_interval_sec)

    async def can_start_new_company(self, active_tasks: int) -> Tuple[bool, str]:
        """
        Decide whether it is safe to admit a single new company task.

        Returns (ok, reason):

          - ok == True  -> caller should start exactly one new company.
          - ok == False -> caller should not start a new company now.

        Reasons include (non exhaustive):
          - "max_concurrency"
          - "mem_pressure_block"
          - "admission_cooldown"
          - "cautious_cooldown"
          - "high_band"
          - "high_band_cooldown"
          - "low_band"
          - "cautious_band"
          - "rush_mode"
          - "no_metrics"
        """
        now = time.monotonic()
        cfg = self.cfg

        # 1) Hard cap by active_tasks vs max_concurrency
        if active_tasks >= max(1, int(cfg.max_concurrency)):
            return False, "max_concurrency"

        async with self._lock:
            # 2) Global block due to memory pressure
            if now < self._blocked_until_ts:
                return False, "mem_pressure_block"

            avg_cpu = self._last_avg_cpu
            avg_mem = self._last_avg_mem

            # If we have no samples yet and psutil is missing, admit conservatively.
            if not self._samples and not PSUTIL_AVAILABLE:
                self._last_admission_ts = now
                logger.info(
                    "[AdaptiveScheduling] psutil missing and no prior samples; "
                    "admitting new company by default."
                )
                return True, "no_metrics"

            # If we have no samples yet but psutil exists, force a sample now.
            if not self._samples and PSUTIL_AVAILABLE:
                cpu_frac, mem_frac = self._read_usage()
                self._update_samples_locked(
                    now_mono=time.monotonic(),
                    cpu=cpu_frac,
                    mem=mem_frac,
                )
                avg_cpu = self._last_avg_cpu
                avg_mem = self._last_avg_mem

            # Classify the current band.
            band = self._classify_band_locked()

            # Rush mode: fast lane when underutilized plateau was detected.
            if self._rush_active and now < self._rush_until_ts:
                if band == "high":
                    # Even in rush, high band is not safe. Do not admit.
                    self._last_high_block_ts = now
                    logger.info(
                        "[AdaptiveScheduling] rush blocked by high band "
                        "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                        active_tasks,
                        avg_cpu,
                        avg_mem,
                    )
                    return False, "high_band"
                # In low or cautious band during rush, skip cooldowns and admit.
                self._last_admission_ts = now
                if band == "cautious":
                    self._last_cautious_admission_ts = now
                logger.info(
                    "[AdaptiveScheduling] rush admission granted "
                    "(active=%d, band=%s, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    band,
                    avg_cpu,
                    avg_mem,
                )
                return True, "rush_mode"

            # 3) Band specific logic when not in rush

            # High band: do not admit, respect a separate high band cooldown.
            if band == "high":
                if (
                    self._last_high_block_ts > 0.0
                    and now - self._last_high_block_ts < cfg.block_cooldown_sec
                ):
                    return False, "high_band_cooldown"

                self._last_high_block_ts = now
                logger.info(
                    "[AdaptiveScheduling] high band - admission denied "
                    "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    avg_cpu,
                    avg_mem,
                )
                return False, "high_band"

            # From here, band is either "low" or "cautious".
            # First, respect the global admission cooldown.
            if (
                self._last_admission_ts > 0.0
                and now - self._last_admission_ts < cfg.admission_cooldown_sec
            ):
                return False, "admission_cooldown"

            if band == "low":
                # Low band: only global admission cooldown applied.
                self._last_admission_ts = now
                logger.info(
                    "[AdaptiveScheduling] low band admission granted "
                    "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    avg_cpu,
                    avg_mem,
                )
                return True, "low_band"

            # Cautious band.
            # Apply an extra cooldown window inside this band.
            if (
                self._last_cautious_admission_ts > 0.0
                and now - self._last_cautious_admission_ts
                < cfg.cautious_admission_cooldown_sec
            ):
                logger.info(
                    "[AdaptiveScheduling] cautious band cooldown - "
                    "admission denied (active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                    active_tasks,
                    avg_cpu,
                    avg_mem,
                )
                return False, "cautious_cooldown"

            self._last_admission_ts = now
            self._last_cautious_admission_ts = now
            logger.info(
                "[AdaptiveScheduling] cautious band admission granted "
                "(active=%d, avg_cpu=%.3f, avg_mem=%.3f)",
                active_tasks,
                avg_cpu,
                avg_mem,
            )
            return True, "cautious_band"

    def notify_admitted(self, company_id: Optional[str] = None, reason: str = "ok") -> None:
        """
        Optional helper for callers that want to log an explicit admission
        event tied to a specific company. This does not affect the internal
        cooldown timestamps, which are already updated by can_start_new_company.
        """
        if company_id is not None:
            logger.info(
                "[AdaptiveScheduling] company admitted company_id=%s reason=%s",
                company_id,
                reason,
            )

    async def on_memory_pressure(self, severity: str = "hard") -> None:
        """
        Called when MemoryGuard detects a critical memory pressure event.

        This does not touch any "limit". Instead, it:

          - increments a counter for diagnostics
          - blocks all new admissions for mem_pressure_block_sec seconds
        """
        async with self._lock:
            self._memory_pressure_hits += 1
            now = time.monotonic()
            self._blocked_until_ts = max(
                self._blocked_until_ts,
                now + float(self.cfg.mem_pressure_block_sec),
            )
            logger.warning(
                "[AdaptiveScheduling] memory pressure event "
                "(severity=%s, hits=%d); blocking new admissions until %.1fs "
                "from monotonic zero.",
                severity,
                self._memory_pressure_hits,
                self._blocked_until_ts,
            )

    async def get_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of internal state for debugging.
        """
        async with self._lock:
            band = self._classify_band_locked() if self._samples else "unknown"
            return {
                "max_concurrency": self.cfg.max_concurrency,
                "last_avg_cpu": self._last_avg_cpu,
                "last_avg_mem": self._last_avg_mem,
                "last_raw_cpu": self._last_raw_cpu,
                "last_raw_mem": self._last_raw_mem,
                "samples_count": len(self._samples),
                "last_update_ts": self._last_update_ts,
                "last_admission_ts": self._last_admission_ts,
                "last_cautious_admission_ts": self._last_cautious_admission_ts,
                "last_high_block_ts": self._last_high_block_ts,
                "blocked_until_ts": self._blocked_until_ts,
                "memory_pressure_hits": self._memory_pressure_hits,
                "band": band,
                "rush_active": self._rush_active,
                "rush_start_ts": self._rush_start_ts,
                "rush_until_ts": self._rush_until_ts,
                "rush_start_cpu": self._rush_start_cpu,
                "rush_start_mem": self._rush_start_mem,
            }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

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

    async def _sample_once(self) -> None:
        """
        Sample CPU and memory once and update smoothed averages.
        Also updates rush mode based on plateau detection.
        """
        if not PSUTIL_AVAILABLE:
            # When psutil is missing, keep existing averages and simply
            # log once that we do not have usage metrics.
            async with self._lock:
                if self._last_update_ts == 0.0:
                    self._last_update_ts = time.time()
                    logger.warning(
                        "[AdaptiveScheduling] psutil not available; "
                        "admissions will not be based on real CPU/memory."
                    )
            return

        raw_cpu, raw_mem = self._read_usage()
        now_mono = time.monotonic()
        wall_ts = time.time()
        cfg = self.cfg

        window = float(cfg.smoothing_window_sec) if cfg.smoothing_window_sec > 0.0 else 0.0

        async with self._lock:
            self._update_samples_locked(now_mono=now_mono, cpu=raw_cpu, mem=raw_mem)
            avg_cpu = self._last_avg_cpu
            avg_mem = self._last_avg_mem
            self._last_raw_cpu = raw_cpu
            self._last_raw_mem = raw_mem
            self._last_update_ts = wall_ts

            # Possibly enter or exit rush mode based on plateau detection.
            self._update_rush_mode_locked(now_mono=now_mono)

            state_snapshot = {
                "ts": wall_ts,
                "monotonic": now_mono,
                "avg_cpu": avg_cpu,
                "avg_mem": avg_mem,
                "raw_cpu": raw_cpu,
                "raw_mem": raw_mem,
                "samples_window_sec": window,
                "samples_count": len(self._samples),
                "memory_pressure_hits": self._memory_pressure_hits,
                "rush_active": self._rush_active,
            }

        self._maybe_log_state(state_snapshot)

    def _update_samples_locked(self, now_mono: float, cpu: float, mem: float) -> None:
        """
        Update sliding window and averages. Caller must hold _lock.
        """
        window = float(self.cfg.smoothing_window_sec)
        self._samples.append((now_mono, cpu, mem))

        if window > 0.0:
            cutoff = now_mono - window
            self._samples = [
                (t, c, m) for (t, c, m) in self._samples if t >= cutoff
            ]
        else:
            # Only keep the latest sample when smoothing is disabled.
            self._samples = self._samples[-1:]

        if not self._samples:
            self._last_avg_cpu = cpu
            self._last_avg_mem = mem
            return

        sum_cpu = 0.0
        sum_mem = 0.0
        for _, c, m in self._samples:
            sum_cpu += c
            sum_mem += m
        count = len(self._samples)
        self._last_avg_cpu = sum_cpu / count
        self._last_avg_mem = sum_mem / count

    def _classify_band_locked(self) -> str:
        """
        Classify current load into 'low', 'cautious', or 'high'.
        Caller must hold _lock.
        """
        if not self._samples:
            return "low"

        cfg = self.cfg
        use_cpu = cfg.use_cpu
        use_mem = cfg.use_mem
        cpu = self._last_avg_cpu
        mem = self._last_avg_mem

        # High band: any metric at or above its high threshold.
        high = False
        if use_cpu and cpu >= cfg.target_cpu_high:
            high = True
        if use_mem and mem >= cfg.target_mem_high:
            high = True
        if high:
            return "high"

        # Low band: both metrics at or below low thresholds (if enabled).
        low_cpu_ok = (not use_cpu) or cpu <= cfg.target_cpu_low
        low_mem_ok = (not use_mem) or mem <= cfg.target_mem_low
        if low_cpu_ok and low_mem_ok:
            return "low"

        # Otherwise we are in the cautious band.
        return "cautious"

    def _update_rush_mode_locked(self, now_mono: float) -> None:
        """
        Update rush mode state based on plateau detection and duration.

        Rush mode is entered when CPU and memory are below configured
        thresholds and fluctuate inside a ±rush_band_width band for
        at least rush_min_samples samples.

        When rush mode ends, log how CPU and memory changed over that
        window.
        """
        cfg = self.cfg
        if not cfg.rush_enabled:
            return

        if not self._samples:
            return

        # First, handle rush window completion.
        if self._rush_active and now_mono >= self._rush_until_ts:
            avg_cpu = self._last_avg_cpu
            avg_mem = self._last_avg_mem
            cpu_delta = avg_cpu - self._rush_start_cpu
            mem_delta = avg_mem - self._rush_start_mem

            logger.info(
                "[AdaptiveScheduling] rush window complete: "
                "cpu %.3f -> %.3f (delta=%.3f), "
                "mem %.3f -> %.3f (delta=%.3f)",
                self._rush_start_cpu,
                avg_cpu,
                cpu_delta,
                self._rush_start_mem,
                avg_mem,
                mem_delta,
            )

            self._rush_active = False
            self._rush_start_ts = 0.0
            self._rush_until_ts = 0.0
            return

        # If already in rush and not finished, do not try to restart.
        if self._rush_active:
            return

        # Not in rush: check for underutilized plateau.
        if len(self._samples) < max(1, cfg.rush_min_samples):
            return

        cpus = [c for (_, c, _) in self._samples]
        mems = [m for (_, _, m) in self._samples]

        min_cpu = min(cpus)
        max_cpu = max(cpus)
        min_mem = min(mems)
        max_mem = max(mems)

        avg_cpu = self._last_avg_cpu
        avg_mem = self._last_avg_mem

        cpu_range = max_cpu - min_cpu
        mem_range = max_mem - min_mem

        plateau_cpu = (
            avg_cpu < cfg.rush_cpu_threshold and cpu_range <= cfg.rush_band_width
        )
        plateau_mem = (
            avg_mem < cfg.rush_mem_threshold and mem_range <= cfg.rush_band_width
        )

        if plateau_cpu and plateau_mem:
            # Enter rush mode.
            self._rush_active = True
            self._rush_start_ts = now_mono
            self._rush_until_ts = now_mono + float(cfg.rush_duration_sec)
            self._rush_start_cpu = avg_cpu
            self._rush_start_mem = avg_mem

            logger.info(
                "[AdaptiveScheduling] entering rush mode "
                "(avg_cpu=%.3f range=%.3f, avg_mem=%.3f range=%.3f, "
                "duration=%.1fs)",
                avg_cpu,
                cpu_range,
                avg_mem,
                mem_range,
                cfg.rush_duration_sec,
            )

    def _maybe_log_state(self, state: Dict[str, Any]) -> None:
        """
        Optional structured logging of sampling state.
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
                "[AdaptiveScheduling] failed writing log to %s",
                self.cfg.log_path,
                exc_info=True,
            )


__all__ = [
    "AdaptiveSchedulingConfig",
    "AdaptiveScheduler",
]
