from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Callable

from extensions.connectivity_guard import ConnectivityGuard
from extensions import md_gating
from extensions.output_paths import save_stage_output
from extensions.crawl_state import upsert_url_index_entry
from extensions.adaptive_scheduling import MEMORY_PRESSURE_MARKER

logger = logging.getLogger("page_pipeline")


async def process_page_result(
    page_result: Any,
    *,
    company: Any,
    guard: Optional[ConnectivityGuard],
    gating_cfg: md_gating.MarkdownGatingConfig,
    timeout_error_marker: str,
    mark_company_timeout_cb: Optional[Callable[[str], None]] = None,
    mark_company_memory_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Per page processing extracted from run.py.

    - Saves HTML and markdown.
    - Applies markdown gating.
    - Updates url_index via upsert_url_index_entry.
    - Notifies ConnectivityGuard.
    - Optionally calls timeout / memory callbacks when provided.

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

    # Error classification
    error_str = error if isinstance(error, str) else ""
    timeout_exceeded = bool(error_str and timeout_error_marker in error_str)
    memory_pressure = bool(error_str and MEMORY_PRESSURE_MARKER in error_str)

    # Company-level markers
    if timeout_exceeded and mark_company_timeout_cb is not None:
        mark_company_timeout_cb(company_id)

    if memory_pressure and mark_company_memory_cb is not None:
        mark_company_memory_cb(company_id)

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

    # Override status for timeout / memory pressure
    if timeout_exceeded:
        md_status = "timeout_page_exceeded"
    if memory_pressure:
        # Memory pressure takes precedence over timeout in the status label
        md_status = "memory_pressure"

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

    # Page-level flags for retry logic / analysis
    if timeout_exceeded:
        entry["timeout_page_exceeded"] = True
        entry["scheduled_retry"] = True

    if memory_pressure:
        entry["memory_pressure"] = True
        entry["scheduled_retry"] = True

    if md_path is not None:
        entry["markdown_path"] = md_path
    if html_path is not None:
        entry["html_path"] = html_path

    upsert_url_index_entry(company_id, url, entry)
