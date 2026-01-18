from __future__ import annotations

from typing import Any

from configs.models import (
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_TERMINAL_DONE,
)


def _status(snap: Any) -> str:
    return (
        str(getattr(snap, "status", "") or COMPANY_STATUS_PENDING).strip()
        or COMPANY_STATUS_PENDING
    )


def llm_precheck_ok(snap: Any) -> bool:
    """
    LLM precheck policy (status-based).

    Philosophy:
      - terminal_done is treated like markdown_done (crawl has "ended") and MUST NOT gate LLM.
      - last_error is NOT part of gating (LLM may run even if last_error is non-empty).
      - Disallow only when crawl is not finalized: pending / markdown_not_done.
    """
    st = _status(snap)
    return st in (
        COMPANY_STATUS_MD_DONE,
        COMPANY_STATUS_TERMINAL_DONE,
        COMPANY_STATUS_LLM_NOT_DONE,
        COMPANY_STATUS_LLM_DONE,
    )


def should_short_circuit_llm_due_to_no_markdown(
    snap: Any, *, llm_is_requested: bool
) -> bool:
    """
    Deterministic rule:
      - If LLM is requested but there are zero markdown pages, skip extraction.
      - Caller typically marks LLM_DONE in that case.
    """
    if not llm_is_requested:
        return False
    return int(getattr(snap, "urls_markdown_done", 0) or 0) <= 0


def crawl_or_terminal_done_ok(snap: Any) -> bool:
    """
    "Crawl is done enough to proceed / finalize" policy.

    Philosophy:
      - terminal_done is treated as done (even if the crawl runner didn't fully finish).
      - markdown_done / llm_done are obviously done.
      - markdown_not_done / pending are not done.
    """
    st = _status(snap)
    return st in (
        COMPANY_STATUS_TERMINAL_DONE,
        COMPANY_STATUS_MD_DONE,
        COMPANY_STATUS_LLM_NOT_DONE,
        COMPANY_STATUS_LLM_DONE,
    )
