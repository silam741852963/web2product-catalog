from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("memory_guard")

MEMORY_PRESSURE_MARKER = "Requeued due to critical memory pressure"

class CriticalMemoryPressure(RuntimeError):
    """Raised when Crawl4AI reports critical memory pressure for a page/company."""


@dataclass(slots=True)
class MemoryGuardConfig:
    """Configuration for MemoryGuard."""
    marker: str = MEMORY_PRESSURE_MARKER


class MemoryGuard:
    """
    Lightweight helper that watches page-level errors for Crawl4AI
    'critical memory pressure' signals.

    When the marker string is observed in a page error, this class:
      * calls the provided `mark_company_memory(company_id)` callback so the
        caller can add the company to its retry set / retry_companies.json
      * raises CriticalMemoryPressure to allow the caller to abort the
        current run and let an outer wrapper restart the process
    """

    def __init__(self, config: Optional[MemoryGuardConfig] = None) -> None:
        self.config = config or MemoryGuardConfig()

    def check_page_error(
        self,
        *,
        error: Any,
        company_id: str,
        url: Optional[str],
        mark_company_memory: Callable[[str], None],
    ) -> None:
        """
        Inspect a page-level `error` coming from Crawl4AI.

        If the configured marker is present, mark the company for retry and
        raise CriticalMemoryPressure so the caller can abort the run.
        """
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
        # Inform the caller that this company should be retried in the next run.
        mark_company_memory(company_id)

        # Raise so upper layers can choose to stop all crawling and exit
        # with the special retry exit code (so an outer wrapper can restart
        # the process and free all Playwright/Chromium resources).
        raise CriticalMemoryPressure(
            f"Critical memory pressure for company_id={company_id} url={url!r}"
        )