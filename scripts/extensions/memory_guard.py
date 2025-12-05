from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("memory_guard")

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"


class CriticalMemoryPressure(RuntimeError):
    """
    Raised when memory pressure is high enough that we should abort
    the current company task.

    severity:
        "soft"      - host reached soft limit or per page critical marker
        "hard"      - host reached hard limit
        "emergency" - host reached emergency limit, run should exit
    """

    def __init__(self, message: str, severity: str = "emergency") -> None:
        super().__init__(message)
        self.severity = severity


@dataclass(slots=True)
class MemoryGuardConfig:
    """
    Configuration for MemoryGuard.

    - marker:
        String that Crawl4AI puts into page errors on critical pressure.

    - host_soft_limit:
        Fraction of RAM (0 to 1) at which we start reacting to memory pressure.
        At this level we may clamp concurrency and, after repeated hits,
        kill the triggering company.

    - host_hard_limit:
        Fraction of RAM at which we treat this as serious pressure and
        immediately kill the triggering company task and clamp concurrency
        more aggressively (policy is implemented in the on_critical callback).

    - host_emergency_limit:
        Fraction of RAM at which the process should gracefully abort and let
        the outer wrapper restart. This is the equivalent of your previous
        "hard" limit that exited immediately.

    - soft_kill_hits:
        Number of consecutive soft-limit hits required before we escalate
        soft pressure into an actual kill (raising CriticalMemoryPressure
        with severity="soft"). A value of 1 makes soft behave like a hard
        kill at the soft threshold; higher values make it more modest.

    - check_interval_sec:
        Minimum interval between host memory polls, to avoid spamming psutil.

    - on_critical:
        Optional callback(company_id, severity) when any of the limits is hit.
        It is responsible for clamping concurrency, global abort decisions,
        etc. The guard itself may still raise CriticalMemoryPressure to abort
        the current company.
    """

    marker: str = MEMORY_PRESSURE_MARKER
    host_soft_limit: float = 0.90
    host_hard_limit: float = 0.94
    host_emergency_limit: float = 0.98
    soft_kill_hits: int = 3
    check_interval_sec: float = 0.5
    on_critical: Optional[Callable[[str, str], None]] = None


class MemoryGuard:
    """
    Watches both page level errors and host memory usage.

    - If the Crawl4AI marker string is seen in `error`, marks the company
      and raises CriticalMemoryPressure(severity="soft") immediately.

    - Independently, it rate limits host memory checks via psutil and
      compares usage against soft, hard and emergency thresholds:

        soft       => on first few hits: clamp concurrency only;
                      after `soft_kill_hits` consecutive soft hits:
                      kill current company (severity="soft") and keep run alive.
        hard       => kill current company immediately (severity="hard"),
                      clamp concurrency more aggressively.
        emergency  => cancel all tasks via callback and ultimately exit 17
                      (severity="emergency").
    """

    def __init__(self, config: Optional[MemoryGuardConfig] = None) -> None:
        self.config = config or MemoryGuardConfig()
        self._last_check = 0.0
        # Count consecutive soft-limit hits while host is in the soft band.
        self._soft_hits: int = 0

    def _reset_soft_hits(self) -> None:
        self._soft_hits = 0

    def _classify_severity(self, used_frac: float) -> Optional[str]:
        cfg = self.config
        # Emergency limit is highest priority if configured
        if cfg.host_emergency_limit > 0.0 and used_frac >= cfg.host_emergency_limit:
            return "emergency"
        if cfg.host_hard_limit > 0.0 and used_frac >= cfg.host_hard_limit:
            return "hard"
        if cfg.host_soft_limit > 0.0 and used_frac >= cfg.host_soft_limit:
            return "soft"
        return None

    def _check_host_memory(
        self,
        *,
        company_id: str,
        url: Optional[str],
        mark_company_memory: Callable[[str], None],
    ) -> None:
        if psutil is None:
            return

        now = time.monotonic()
        if now - self._last_check < self.config.check_interval_sec:
            return
        self._last_check = now

        try:
            mem = psutil.virtual_memory()
        except Exception as e:
            logger.debug("MemoryGuard: psutil.virtual_memory failed: %s", e)
            return

        used_frac = mem.percent / 100.0
        severity = self._classify_severity(used_frac)

        if severity is None:
            # Back to safe zone; reset soft strike count.
            if self._soft_hits:
                logger.debug(
                    "MemoryGuard: host memory back below soft limit; "
                    "resetting soft hit counter (was %d).",
                    self._soft_hits,
                )
            self._reset_soft_hits()
            return

        cfg = self.config

        # --------- SOFT BAND: modest, but can kill after repeated hits ----------
        if severity == "soft":
            # Increment consecutive soft hits while we're in the soft band.
            self._soft_hits += 1
            hit = self._soft_hits
            kill_after = max(1, cfg.soft_kill_hits)

            if hit < kill_after:
                # First few hits: warning + concurrency clamp only.
                logger.warning(
                    "Host memory usage %.1f%% >= soft limit %.1f%%; "
                    "soft hit %d/%d for company_id=%s url=%s. "
                    "Clamping concurrency but not killing yet.",
                    mem.percent,
                    cfg.host_soft_limit * 100.0,
                    hit,
                    kill_after,
                    company_id,
                    url,
                )

                if cfg.on_critical is not None:
                    try:
                        cfg.on_critical(company_id, "soft")
                    except Exception:
                        logger.exception(
                            "MemoryGuard: on_critical callback failed for "
                            "company_id=%s severity=soft",
                            company_id,
                        )

                # Do NOT mark company or raise yet: let it continue, hoping
                # concurrency clamp will relieve pressure.
                return

            # Once we've seen `soft_kill_hits` consecutive soft events, we
            # escalate and kill the triggering company, but still keep the run.
            logger.error(
                "Host memory usage %.1f%% >= soft limit %.1f%%; "
                "soft limit exceeded %d times (threshold=%d). "
                "Killing triggering company_id=%s url=%s and clamping concurrency.",
                mem.percent,
                cfg.host_soft_limit * 100.0,
                hit,
                kill_after,
                company_id,
                url,
            )

            mark_company_memory(company_id)

            if cfg.on_critical is not None:
                try:
                    cfg.on_critical(company_id, "soft")
                except Exception:
                    logger.exception(
                        "MemoryGuard: on_critical callback failed for "
                        "company_id=%s severity=soft",
                        company_id,
                    )

            # Reset strike counter for the next soft-pressure episode.
            self._reset_soft_hits()

            raise CriticalMemoryPressure(
                f"Host memory usage {mem.percent:.1f}% exceeds soft limit "
                f"after {hit} consecutive hits",
                severity="soft",
            )

        # --------- HARD / EMERGENCY: immediate kill / abort ----------
        # Any non-soft severity resets the soft hit counter.
        self._reset_soft_hits()

        # Mark the triggering company for retry
        mark_company_memory(company_id)

        if severity == "hard":
            logger.error(
                "Host memory usage %.1f%% >= hard limit %.1f%%; "
                "killing triggering company_id=%s url=%s and tightening concurrency.",
                mem.percent,
                cfg.host_hard_limit * 100.0,
                company_id,
                url,
            )
        else:
            logger.critical(
                "Host memory usage %.1f%% >= emergency limit %.1f%%; "
                "run should abort so outer wrapper can restart (company_id=%s url=%s).",
                mem.percent,
                cfg.host_emergency_limit * 100.0,
                company_id,
                url,
            )

        if cfg.on_critical is not None:
            try:
                cfg.on_critical(company_id, severity)
            except Exception:
                logger.exception(
                    "MemoryGuard: on_critical callback failed for company_id=%s severity=%s",
                    company_id,
                    severity,
                )

        raise CriticalMemoryPressure(
            f"Host memory usage {mem.percent:.1f}% exceeds {severity} limit",
            severity=severity,
        )

    def check_host_only(
        self,
        *,
        company_id: str,
        url: Optional[str],
        mark_company_memory: Callable[[str], None],
    ) -> None:
        """
        Check host level memory pressure only, without looking at any page error.

        Use this in tight crawl loops or right after starting a browser session
        so we can bail out before the kernel OOM killer fires.
        """
        self._check_host_memory(
            company_id=company_id,
            url=url,
            mark_company_memory=mark_company_memory,
        )

    def check_page_error(
        self,
        *,
        error: Any,
        company_id: str,
        url: Optional[str],
        mark_company_memory: Callable[[str], None],
    ) -> None:
        """
        Inspect a page level `error` and host memory.

        - Always runs a rate limited host memory check.
        - If the configured marker is present in `error`, marks the company
          and raises CriticalMemoryPressure(severity="soft").
        """
        # First: host wide memory check (may raise CriticalMemoryPressure)
        self.check_host_only(
            company_id=company_id,
            url=url,
            mark_company_memory=mark_company_memory,
        )

        # Then: Crawl4AI-specific marker in page error
        if not isinstance(error, str):
            return

        if self.config.marker not in error:
            return

        logger.error(
            "Critical memory pressure marker detected for company_id=%s url=%s; "
            "marking for retry and signalling abort of this company.",
            company_id,
            url,
        )

        mark_company_memory(company_id)

        # Treat marker as soft level pressure: kill this company immediately,
        # but keep the run alive.
        if self.config.on_critical is not None:
            try:
                self.config.on_critical(company_id, "soft")
            except Exception:
                logger.exception(
                    "MemoryGuard: on_critical callback failed for company_id=%s severity=soft",
                    company_id,
                )

        # Marker always kills immediately; no soft-hit grace here because
        # the browser already reported critical pressure.
        raise CriticalMemoryPressure(
            f"Critical memory pressure for company_id={company_id} url={url!r}",
            severity="soft",
        )
