
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from configs.llm import parse_extracted_payload, parse_presence_result
from extensions.crawl_state import (
    get_crawl_state,
    load_url_index,
    upsert_url_index_entry,
)
from extensions.output_paths import save_stage_output
from extensions.stall_guard import StallGuard

logger = logging.getLogger("llm_passes")


# ---------------------------------------------------------------------------
# LLM presence second pass
# ---------------------------------------------------------------------------

async def run_presence_pass_for_company(
    company: Any,
    *,
    presence_strategy: Any,
    stall_guard: Optional[StallGuard] = None,
) -> None:
    """
    LLM presence pass for a single company.

    - Enumerates pending URLs for LLM presence.
    - Reads markdown for each.
    - Calls presence_strategy.extract(url, text) in a thread.
    - Parses result with parse_presence_result.
    - Updates url_index entries and recomputes company state.
    """
    state = get_crawl_state()
    pending_urls = await state.get_pending_urls_for_llm(company.company_id)
    if not pending_urls:
        logger.info(
            "LLM presence: no pending URLs for company_id=%s",
            company.company_id,
        )
        await state.recompute_company_from_index(
            company.company_id,
            name=None,
            root_url=company.domain_url,
        )
        return

    logger.info(
        "LLM presence: %d pending URLs for company_id=%s",
        len(pending_urls),
        company.company_id,
    )

    index = load_url_index(company.company_id) or {}
    updated = 0

    def _heartbeat() -> None:
        if stall_guard is not None:
            stall_guard.record_heartbeat("llm_presence", company_id=company.company_id)

    def _update(url: str, patch: Dict[str, Any], reason: str) -> None:
        nonlocal updated
        patch_full = {
            "presence": 0,
            "presence_checked": True,
            "status": "llm_extracted_empty",
            "llm_presence_reason": reason,
        }
        patch_full.update(patch)
        upsert_url_index_entry(company.company_id, url, patch_full)
        updated += 1
        _heartbeat()

    for url in pending_urls:
        ent = index.get(url) or {}
        md_path = ent.get("markdown_path")

        if not md_path:
            _update(url, {}, "no_markdown")
            continue

        try:
            text = Path(md_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.error(
                "LLM presence: failed reading markdown for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            _update(url, {}, "markdown_read_error")
            continue

        if not text.strip():
            _update(url, {}, "empty_markdown")
            continue

        try:
            raw_result = await asyncio.to_thread(
                presence_strategy.extract, url, text
            )
        except Exception as e:
            logger.exception(
                "LLM presence: error for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            await state.mark_url_failed(
                company.company_id,
                url,
                f"presence_error:{type(e).__name__}",
            )
            _update(url, {}, "presence_exception")
            continue

        has_offering, confidence, preview = parse_presence_result(
            raw_result,
            default=False,
        )

        patch: Dict[str, Any] = {
            "presence": 1 if has_offering else 0,
            "presence_checked": True,
            "status": "llm_extracted" if has_offering else "llm_extracted_empty",
        }
        if confidence is not None:
            patch["llm_presence_confidence"] = confidence
        if preview is not None:
            patch["llm_presence_preview"] = preview

        upsert_url_index_entry(company.company_id, url, patch)
        updated += 1
        _heartbeat()

    logger.info(
        "LLM presence: updated %d URLs for company_id=%s",
        updated,
        company.company_id,
    )

    await state.recompute_company_from_index(
        company.company_id,
        name=None,
        root_url=company.domain_url,
    )


# ---------------------------------------------------------------------------
# LLM full extraction second pass
# ---------------------------------------------------------------------------

async def run_full_pass_for_company(
    company: Any,
    *,
    full_strategy: Any,
    stall_guard: Optional[StallGuard] = None,
) -> None:
    """
    LLM full extraction pass for a single company.

    - Iterates url_index.
    - For each markdown that is not yet extracted:
      - reads markdown
      - calls full_strategy.extract(url, text) in a thread
      - parses payload with parse_extracted_payload
      - writes product JSON with save_stage_output
      - updates url_index entry
    """
    index = load_url_index(company.company_id)
    if not isinstance(index, dict) or not index:
        logger.info(
            "LLM full: no url_index entries for company_id=%s",
            company.company_id,
        )
        return

    updated = 0

    def _heartbeat() -> None:
        if stall_guard is not None:
            stall_guard.record_heartbeat("llm_full", company_id=company.company_id)

    for url, ent in index.items():
        if not isinstance(ent, dict):
            continue

        md_path = ent.get("markdown_path")
        if not md_path or ent.get("extracted"):
            continue

        try:
            text = Path(md_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.error(
                "LLM full: failed reading markdown for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {"extracted": 0, "status": "llm_full_markdown_read_error"},
            )
            _heartbeat()
            continue

        if not text.strip():
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    "extracted": 0,
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_full_empty_markdown",
                },
            )
            _heartbeat()
            continue

        try:
            raw_result = await asyncio.to_thread(full_strategy.extract, url, text)
        except Exception as e:
            logger.exception(
                "LLM full: error for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {"extracted": 0, "status": f"llm_full_error:{type(e).__name__}"},
            )
            _heartbeat()
            continue

        payload = parse_extracted_payload(raw_result)
        payload_dict = payload.model_dump()

        try:
            product_path = save_stage_output(
                bvdid=company.company_id,
                url=url,
                data=json.dumps(payload_dict, ensure_ascii=False),
                stage="product",
            )
        except Exception as e:
            logger.error(
                "LLM full: failed writing product JSON for %s (company=%s): %s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {"extracted": 0, "status": "llm_full_write_error"},
            )
            _heartbeat()
            continue

        presence_flag = 1 if payload.offerings else 0
        patch: Dict[str, Any] = {
            "extracted": 1,
            "presence": presence_flag,
            "presence_checked": True,
            "status": "llm_full_extracted",
        }
        if product_path is not None:
            patch["product_path"] = str(product_path)

        upsert_url_index_entry(company.company_id, url, patch)
        updated += 1
        _heartbeat()

    logger.info(
        "LLM full: wrote product JSON for %d URLs (company_id=%s)",
        updated,
        company.company_id,
    )

    state = get_crawl_state()
    await state.recompute_company_from_index(
        company.company_id,
        name=None,
        root_url=company.domain_url,
    )
