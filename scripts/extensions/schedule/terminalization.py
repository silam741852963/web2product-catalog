from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from extensions.crawl.runner import PagePipelineSummary
from extensions.crawl.state import (
    COMPANY_STATUS_MD_DONE,
    crawl_runner_done_ok,
    urls_md_done,
    urls_total,
)


@dataclass(frozen=True, slots=True)
class TerminalDecision:
    """
    action:
      - None: no special action
      - "mem": treat as memory failure (retry cls = mem)
      - "stall": treat as stall (retry cls = stall)

    NOTE:
      This module MUST NOT decide "terminal" anymore.
      Terminalization is a state-truth decision (pending URLs, crawl_finished, etc.)
      and must be made in run.py after checking state truth.
    """

    action: Optional[str]
    reason: str


def decide_from_page_summary(summary: PagePipelineSummary) -> TerminalDecision:
    tp = max(0, int(getattr(summary, "total_pages", 0) or 0))
    ts = int(getattr(summary, "timeout_pages", 0) or 0)
    mp = int(getattr(summary, "memory_pressure_pages", 0) or 0)

    if mp > 0:
        return TerminalDecision("mem", f"memory_pressure_pages={mp}/{tp}")

    # IMPORTANT:
    # 0 pages is NOT a reliable stall signal:
    # - can mean "no new work" / "already exhausted" / "all URLs filtered/processed".
    # run.py/state is the source of truth (pending-md + crawl_finished).
    if tp == 0:
        return TerminalDecision(None, "0 pages")

    # Timeout dominant -> stall signal (NOT terminal).
    if ts > 0 and (ts / float(tp or 1)) >= 0.60:
        return TerminalDecision("stall", f"timeout_dominant timeout_pages={ts}/{tp}")

    return TerminalDecision(None, "ok")


# ---------------------------------------------------------------------------
# Sanity gating helpers (moved out of run.py)
# ---------------------------------------------------------------------------


async def stall_sanity_override_if_crawl_finished(
    state: Any,
    company: Any,
    decision: Any,
    logger: Any,
) -> bool:
    """
    If page decision says 'stall' but state truth says crawl finished OK,
    override the decision to 'ok'.
    """
    act = str(getattr(decision, "action", "") or "")
    if act != "stall":
        return False

    await state.recompute_company_from_index(
        str(getattr(company, "company_id")),
        name=getattr(company, "name", None),
        root_url=str(getattr(company, "domain_url")),
    )

    snap = await state.get_company_snapshot(
        str(getattr(company, "company_id")), recompute=True
    )
    if crawl_runner_done_ok(snap):
        logger.info(
            "stall overridden: crawl_finished OK; treating as completed company=%s urls_total=%d md_done=%d",
            str(getattr(company, "company_id")),
            urls_total(snap),
            urls_md_done(snap),
        )
        setattr(decision, "action", "ok")
        return True

    return False


async def terminal_sanity_gate_if_crawl_finished(
    state: Any,
    company: Any,
    term_dec: Any,
    page_decision: Any,
    logger: Any,
) -> bool:
    """
    If retry module wants terminalization but state truth says crawl finished OK,
    ignore terminalization and force page_decision action to 'ok'.
    """
    if not bool(getattr(term_dec, "should_terminalize", False)):
        return False

    await state.recompute_company_from_index(
        str(getattr(company, "company_id")),
        name=getattr(company, "name", None),
        root_url=str(getattr(company, "domain_url")),
    )

    snap = await state.get_company_snapshot(
        str(getattr(company, "company_id")), recompute=True
    )
    if crawl_runner_done_ok(snap):
        logger.info(
            "terminalization ignored (pre-gate): crawl_finished OK company=%s reason=%s urls_total=%s md_done=%s",
            str(getattr(company, "company_id")),
            str(getattr(term_dec, "reason", "") or ""),
            urls_total(snap),
            urls_md_done(snap),
        )
        setattr(page_decision, "action", "ok")
        return True

    return False


async def safe_mark_terminal(
    state: Any,
    company: Any,
    *,
    reason: str,
    details: Dict[str, Any],
    last_error: str,
    stage: str,
    logger: Any,
) -> bool:
    """
    Mark terminal only if state truth says crawl NOT finished OK.
    If crawl finished OK, refuse terminalization and (best-effort) mark MD_DONE.
    """
    await state.recompute_company_from_index(
        str(getattr(company, "company_id")),
        name=getattr(company, "name", None),
        root_url=str(getattr(company, "domain_url")),
    )

    snap = await state.get_company_snapshot(
        str(getattr(company, "company_id")), recompute=True
    )

    if crawl_runner_done_ok(snap):
        await state.upsert_company(
            str(getattr(company, "company_id")),
            status=COMPANY_STATUS_MD_DONE,
            last_error=None,
            name=getattr(company, "name", None),
            root_url=str(getattr(company, "domain_url")),
        )
        logger.info(
            "safe_terminal: refused terminalization (crawl_finished OK). stage=%s reason=%s urls_total=%s md_done=%s",
            stage,
            reason,
            urls_total(snap),
            urls_md_done(snap),
        )
        return False

    await state.mark_company_terminal(
        str(getattr(company, "company_id")),
        reason=reason,
        details=details,
        last_error=(last_error or reason)[:4000],
        name=getattr(company, "name", None),
        root_url=str(getattr(company, "domain_url")),
    )
    return True
