from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Callable

from extensions.connectivity_guard import ConnectivityGuard
from extensions import md_gating
from extensions.output_paths import save_stage_output
from extensions.crawl_state import upsert_url_index_entry
from extensions.adaptive_scheduling import MemoryGuard, StallGuard

logger = logging.getLogger("page_pipeline")


async def process_page_result(
    page_result: Any,
    *,
    company: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    timeout_error_marker: str,
    stall_guard: Optional[StallGuard] = None,
    memory_guard: Optional[MemoryGuard] = None,
    mark_company_timeout_cb: Optional[Callable[[str], None]] = None,
    mark_company_memory_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Per page processing extracted from run.py.

    - Saves HTML and markdown.
    - Applies markdown gating.
    - Updates url_index via upsert_url_index_entry.
    - Notifies ConnectivityGuard, StallGuard, and MemoryGuard.
    - Optionally calls retry callbacks when provided.

    Parameters
    ----------
    page_result
        A Crawl4AI deep crawl result item (streamed or batched).
    company
        Object with at least .company_id attribute.
    guard
        ConnectivityGuard instance for tracking HTTP health.
    gating_cfg
        Markdown gating configuration (extensions.md_gating).
    timeout_error_marker
        String marker used in error messages to detect per page timeouts.
    stall_guard
        Optional StallGuard for heartbeat updates.
    memory_guard
        Optional MemoryGuard for memory marker detection at page level.
    mark_company_timeout_cb
        Optional callback mark_company_timeout(company_id: str).
    mark_company_memory_cb
        Optional callback mark_company_memory_pressure(company_id: str).
    """
    if guard is not None:
        await guard.wait_until_healthy()

    company_id = getattr(company, "company_id", None)
    if not company_id:
        logger.warning(
            "process_page_result called without company_id on company=%r", company
        )
        return

    _getattr = getattr

    requested_url = _getattr(page_result, "url", None)
    final_url = _getattr(page_result, "final_url", None) or requested_url
    if not final_url:
        logger.warning(
            "Page result missing URL; skipping entry (company_id=%s)", company_id
        )
        return
    url = final_url

    markdown = _getattr(page_result, "markdown", None)
    status_code = _getattr(page_result, "status_code", None)
    error = _getattr(page_result, "error", None) or _getattr(
        page_result,
        "error_message",
        None,
    )

    # Page level memory guard (marker only). Host level is handled by AdaptiveScheduler.
    if memory_guard is not None:

        def _mark_mem(cid: str) -> None:
            if mark_company_memory_cb is not None:
                mark_company_memory_cb(cid)

        memory_guard.check_page_error(
            error=error,
            company_id=company_id,
            url=url,
            mark_company_memory=_mark_mem,
        )

    timeout_exceeded = isinstance(error, str) and timeout_error_marker in error
    if timeout_exceeded and mark_company_timeout_cb is not None:
        mark_company_timeout_cb(company_id)

    html = _getattr(page_result, "html", None) or _getattr(
        page_result, "final_html", None
    )

    action, reason, stats = md_gating.evaluate_markdown(
        markdown or "",
        min_meaningful_words=gating_cfg.min_meaningful_words,
        cookie_max_fraction=gating_cfg.cookie_max_fraction,
        require_structure=gating_cfg.require_structure,
    )

    html_path: Optional[str] = None
    if html:
        try:
            p_html = save_stage_output(
                bvdid=company_id,
                url=url,
                data=html,
                stage="html",
            )
            if p_html is not None:
                html_path = str(p_html)
        except Exception as e:
            logger.error(
                "Failed to write HTML for %s (company=%s): %s",
                url,
                company_id,
                e,
            )

    # ConnectivityGuard update
    if guard is not None:
        try:
            code_int = int(status_code) if status_code is not None else None
        except Exception:
            code_int = None

        if error or (code_int is not None and code_int >= 500):
            guard.record_transport_error()
        else:
            guard.record_success()

    gating_accept = action == "save"
    md_path: Optional[str] = None
    md_status: str

    if gating_accept and markdown:
        try:
            p = save_stage_output(
                bvdid=company_id,
                url=url,
                data=markdown,
                stage="markdown",
            )
            if p is not None:
                md_path = str(p)
            md_status = "markdown_saved"
        except Exception as e:
            logger.error(
                "Failed to write markdown for %s (company=%s): %s",
                url,
                company_id,
                e,
            )
            md_status = "markdown_error"
    else:
        md_status = "markdown_suppressed"

    if timeout_exceeded:
        md_status = "timeout_page_exceeded"

    entry: Dict[str, Any] = {
        "url": url,
        "requested_url": requested_url,
        "status_code": status_code,
        "error": error,
        "depth": _getattr(page_result, "depth", None),
        "presence": 0,
        "extracted": 0,
        "gating_accept": gating_accept,
        "gating_action": action,
        "gating_reason": reason,
        "md_total_words": stats.get("total_words"),
        "status": md_status,
    }

    if timeout_exceeded:
        entry["timeout_page_exceeded"] = True
        entry["scheduled_retry"] = True

    if md_path is not None:
        entry["markdown_path"] = md_path
    if html_path is not None:
        entry["html_path"] = html_path

    upsert_url_index_entry(company_id, url, entry)

    if stall_guard is not None:
        stall_guard.record_heartbeat("page", company_id=company_id)
