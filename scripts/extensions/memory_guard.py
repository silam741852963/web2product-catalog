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
    Raised when Crawl4AI reports critical memory pressure for a page/company
    or when host-level memory usage crosses the configured hard limit.
    """


@dataclass(slots=True)
class MemoryGuardConfig:
    """
    Configuration for MemoryGuard.

    - marker: string that Crawl4AI puts into page errors on critical pressure
    - host_soft_limit: fraction of RAM (0-1) at which to log warnings
    - host_hard_limit: fraction of RAM (0-1) at which to abort the run
    - check_interval_sec: minimum interval between host memory polls
    - on_critical: optional callback(company_id) when host_hard_limit is hit
    """
    marker: str = MEMORY_PRESSURE_MARKER
    host_soft_limit: float = 0.98
    host_hard_limit: float = 0.99
    check_interval_sec: float = 1.0
    on_critical: Optional[Callable[[str], None]] = None


class MemoryGuard:
    """
    Watches both page-level errors and host memory usage.

    - If the Crawl4AI marker string is seen in `error`, marks the company
      and raises CriticalMemoryPressure.

    - Independently, it periodically checks host memory via psutil. If
      usage >= host_hard_limit, it marks the company, calls `on_critical`
      (if set) so upper layers can cancel all tasks, then raises
      CriticalMemoryPressure.
    """

    def __init__(self, config: Optional[MemoryGuardConfig] = None) -> None:
        self.config = config or MemoryGuardConfig()
        self._last_check = 0.0

    def _check_host_memory(
        self,
        *,
        company_id: str,
        url: Optional[str],
        mark_company_memory: Callable[[str], None],
    ) -> None:
        if psutil is None:
            return

        hard = self.config.host_hard_limit
        soft = self.config.host_soft_limit
        if hard <= 0.0:
            # Host-level memory guard disabled
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

        if used_frac >= hard:
            logger.error(
                "Host memory usage %.1f%% >= hard limit %.1f%%; "
                "signalling CriticalMemoryPressure (company_id=%s url=%s)",
                mem.percent,
                hard * 100.0,
                company_id,
                url,
            )

            # Mark the triggering company for retry
            mark_company_memory(company_id)

            # Let the caller cancel all running tasks if desired
            if self.config.on_critical is not None:
                try:
                    self.config.on_critical(company_id)
                except Exception:
                    logger.exception(
                        "MemoryGuard: on_critical callback failed for company_id=%s",
                        company_id,
                    )

            raise CriticalMemoryPressure(
                f"Host memory usage {mem.percent:.1f}% >= hard limit {hard * 100.0:.1f}%"
            )

        if soft > 0.0 and used_frac >= soft:
            logger.warning(
                "Host memory usage high: %.1f%% (soft limit %.1f%%). "
                "Consider reducing concurrency.",
                mem.percent,
                soft * 100.0,
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
        Inspect a page-level `error` and host memory.

        - Always runs a rate-limited host memory check.
        - If the configured marker is present in `error`, marks the company
          and raises CriticalMemoryPressure.
        """
        # First: host-wide memory check
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
            "Critical memory pressure detected for company_id=%s url=%s; "
            "marking for retry and signalling abort.",
            company_id,
            url,
        )

        mark_company_memory(company_id)

        raise CriticalMemoryPressure(
            f"Critical memory pressure for company_id={company_id} url={url!r}"
        )
