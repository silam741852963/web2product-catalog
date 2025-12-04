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
        Fraction of RAM (0 to 1) at which we start killing the triggering company,
        clamp concurrency and let the run continue.

    - host_hard_limit:
        Fraction of RAM at which we treat this as serious pressure, kill the
        triggering company task and clamp concurrency more aggressively
        (policy is implemented in the on_critical callback).

    - host_emergency_limit:
        Fraction of RAM at which the process should gracefully abort and let
        the outer wrapper restart. This is the equivalent of your previous
        "hard" limit that exited immediately.

    - check_interval_sec:
        Minimum interval between host memory polls, to avoid spamming psutil.

    - on_critical:
        Optional callback(company_id, severity) when any of the limits is hit.
        It is responsible for cancelling tasks, calling adaptive concurrency,
        etc.
    """

    marker: str = MEMORY_PRESSURE_MARKER
    host_soft_limit: float = 0.90
    host_hard_limit: float = 0.94
    host_emergency_limit: float = 0.98
    check_interval_sec: float = 0.5
    on_critical: Optional[Callable[[str, str], None]] = None


class MemoryGuard:
    """
    Watches both page level errors and host memory usage.

    - If the Crawl4AI marker string is seen in `error`, marks the company
      and raises CriticalMemoryPressure(severity="soft").

    - Independently, it rate limits host memory checks via psutil and
      compares usage against soft, hard and emergency thresholds:

        soft       => kill current company, clamp concurrency, keep run alive
        hard       => kill current company, clamp concurrency more
        emergency  => cancel all tasks via callback and ultimately exit 17
    """

    def __init__(self, config: Optional[MemoryGuardConfig] = None) -> None:
        self.config = config or MemoryGuardConfig()
        self._last_check = 0.0

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
            return

        # Mark the triggering company for retry
        mark_company_memory(company_id)

        if severity == "soft":
            logger.warning(
                "Host memory usage %.1f%% >= soft limit %.1f%%; "
                "killing triggering company_id=%s url=%s and clamping concurrency.",
                mem.percent,
                self.config.host_soft_limit * 100.0,
                company_id,
                url,
            )
        elif severity == "hard":
            logger.error(
                "Host memory usage %.1f%% >= hard limit %.1f%%; "
                "killing triggering company_id=%s url=%s and tightening concurrency.",
                mem.percent,
                self.config.host_hard_limit * 100.0,
                company_id,
                url,
            )
        else:
            logger.critical(
                "Host memory usage %.1f%% >= emergency limit %.1f%%; "
                "run should abort so outer wrapper can restart (company_id=%s url=%s).",
                mem.percent,
                self.config.host_emergency_limit * 100.0,
                company_id,
                url,
            )

        # Let the caller perform task cancellation or global abort.
        if self.config.on_critical is not None:
            try:
                self.config.on_critical(company_id, severity)
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
        # First: host wide memory check
        self._check_host_memory(
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

        # Treat marker as soft level pressure: kill the company, keep the run.
        if self.config.on_critical is not None:
            try:
                self.config.on_critical(company_id, "soft")
            except Exception:
                logger.exception(
                    "MemoryGuard: on_critical callback failed for company_id=%s severity=soft",
                    company_id,
                )

        raise CriticalMemoryPressure(
            f"Critical memory pressure for company_id={company_id} url={url!r}",
            severity="soft",
        )
