from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional

from .retry_state import RetryStateStore

_GOTO_TIMEOUT_PAT = re.compile(
    r"(acs-goto|page\.goto:\s*timeout|timeout\s*\d+\s*ms\s*exceeded|timeout\s*\d+ms\s*exceeded)",
    re.IGNORECASE,
)


class CriticalMemoryPressure(RuntimeError):
    def __init__(self, message: str, severity: str = "emergency") -> None:
        super().__init__(message)
        self.severity = severity


class CrawlerFatalError(RuntimeError):
    """Signals a browser / driver level fatal that should recycle crawler instance."""

    pass


class CrawlerTimeoutError(TimeoutError):
    def __init__(self, message: str, *, stage: str, company_id: str, url: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.company_id = company_id
        self.url = url


def is_goto_timeout_error(exc: BaseException) -> bool:
    return _GOTO_TIMEOUT_PAT.search(f"{type(exc).__name__}: {exc}") is not None


def is_playwright_driver_disconnect(exc: BaseException) -> bool:
    low = f"{type(exc).__name__}: {exc}".lower()
    return any(
        n in low
        for n in (
            "connection closed while reading from the driver",
            "browser has been closed",
            "target page, context or browser has been closed",
            "playwright connection closed",
            "pipe closed by peer",
        )
    )


@dataclass(frozen=True, slots=True)
class RetryEvent:
    cls: str  # "stall" | "net" | "mem" | "terminal"
    stage: str
    error: str
    nxdomain_like: bool = False
    status_code: Optional[int] = None


def _extract_status_code(exc: BaseException) -> Optional[int]:
    sc = getattr(exc, "status_code", None)
    if sc is None:
        return None
    try:
        return int(sc)
    except Exception:
        return None


def classify_failure(exc: BaseException, *, stage: str) -> RetryEvent:
    """
    Produces the RetryStateStore-compatible classification:
      - cls in {"stall","net","mem"}
      - plus nxdomain_like + optional status_code
    """
    status_code = _extract_status_code(exc)

    if isinstance(exc, CriticalMemoryPressure):
        return RetryEvent(
            cls="mem",
            stage=stage,
            error=str(exc),
            nxdomain_like=False,
            status_code=status_code,
        )

    if isinstance(exc, (asyncio.TimeoutError, CrawlerTimeoutError)):
        return RetryEvent(
            cls="stall",
            stage=stage,
            error=str(exc),
            nxdomain_like=False,
            status_code=status_code,
        )

    if isinstance(exc, CrawlerFatalError) or is_playwright_driver_disconnect(exc):
        return RetryEvent(
            cls="net",
            stage=stage,
            error=str(exc),
            nxdomain_like=RetryStateStore.classify_unreachable_error(str(exc)),
            status_code=status_code,
        )

    nxdomain_like = RetryStateStore.classify_unreachable_error(str(exc))
    if nxdomain_like:
        return RetryEvent(
            cls="net",
            stage=stage,
            error=str(exc),
            nxdomain_like=True,
            status_code=status_code,
        )

    return RetryEvent(
        cls="stall",
        stage=stage,
        error=str(exc),
        nxdomain_like=False,
        status_code=status_code,
    )


def should_fail_fast_on_goto(exc: BaseException, *, stage: str) -> bool:
    """
    Policy: if the failure looks like 'goto timeout' during initialization-ish stages,
    treat as fail-fast (typically terminalize on first-company attempt if no progress).
    """
    if stage in {"goto", "arun_init", "direct_fetch"} and is_goto_timeout_error(exc):
        return True
    return False
