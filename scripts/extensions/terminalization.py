from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .crawl_runner import PagePipelineSummary


@dataclass(frozen=True, slots=True)
class TerminalDecision:
    """
    action:
      - None: no special action
      - "mem": treat as memory failure (retry cls = mem)
      - "stall": treat as stall (retry cls = stall)
      - "terminal": mark company terminal-done (do not retry further)
    """

    action: Optional[str]
    reason: str


def decide_from_page_summary(summary: PagePipelineSummary) -> TerminalDecision:
    tp = max(0, int(getattr(summary, "total_pages", 0) or 0))
    ts = int(getattr(summary, "timeout_pages", 0) or 0)
    mp = int(getattr(summary, "memory_pressure_pages", 0) or 0)
    md_saved = int(getattr(summary, "markdown_saved", 0) or 0)

    if mp > 0:
        return TerminalDecision("mem", f"memory_pressure_pages={mp}/{tp}")

    if tp == 0:
        return TerminalDecision("stall", "0 pages")

    # No markdown and all timeouts -> very likely dead / blocked.
    if md_saved == 0 and ts >= tp and ts > 0:
        return TerminalDecision(
            "terminal", f"no_markdown_all_timeouts timeout_pages={ts}/{tp}"
        )

    # No markdown and overwhelmingly timeouts.
    if md_saved == 0 and ts > 0 and (ts / max(1, tp)) >= 0.90:
        return TerminalDecision(
            "terminal", f"no_markdown_timeout_dominant timeout_pages={ts}/{tp}"
        )

    # Timeout dominant / no markdown at all -> stall.
    if (md_saved == 0 and ts > 0) or (ts >= 3 and (ts / max(1, tp)) >= 0.60):
        return TerminalDecision("stall", f"timeout_dominant timeout_pages={ts}/{tp}")

    return TerminalDecision(None, "ok")
