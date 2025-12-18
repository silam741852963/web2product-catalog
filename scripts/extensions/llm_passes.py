from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from configs.llm import call_llm_extract, parse_extracted_payload, parse_presence_result
from extensions.crawl_state import (
    get_crawl_state,
    load_url_index,
    upsert_url_index_entry,
)
from extensions.output_paths import save_stage_output

logger = logging.getLogger("llm_passes")


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return None


def _md_debug_stats(url: str, md_path: str, text: str) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    import hashlib

    ss = " ".join((text or "").split())
    sha1 = hashlib.sha1(ss.encode("utf-8", errors="ignore")).hexdigest()

    head = ss[:260] + ("…" if len(ss) > 260 else "")
    tail = ("…" if len(ss) > 260 else "") + ss[-260:]

    logger.debug(
        "[llm_passes] md_stats url=%s md_path=%s text_len=%d sha1=%s head=%s tail=%s",
        url,
        md_path,
        len(ss),
        sha1,
        head,
        tail,
    )


def _save_raw_stage(company_id: str, url: str, raw: Any, stage: str) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        if isinstance(raw, (dict, list)):
            data = json.dumps(raw, ensure_ascii=False)
        else:
            data = str(raw)
        save_stage_output(bvdid=company_id, url=url, data=data, stage=stage)
    except Exception as e:
        logger.debug(
            "[llm_passes] raw_save_failed stage=%s url=%s err=%s", stage, url, e
        )


# ---------------------------------------------------------------------------
# Presence pass
# ---------------------------------------------------------------------------


async def run_presence_pass_for_company(
    company: Any, *, presence_strategy: Any
) -> None:
    state = get_crawl_state()
    pending_urls = await state.get_pending_urls_for_llm(company.company_id)

    if not pending_urls:
        logger.info(
            "LLM presence: no pending URLs for company_id=%s", company.company_id
        )
        await state.recompute_company_from_index(
            company.company_id, name=None, root_url=company.domain_url
        )
        return

    logger.info(
        "LLM presence: %d pending URLs for company_id=%s",
        len(pending_urls),
        company.company_id,
    )

    index = load_url_index(company.company_id) or {}
    updated = 0

    def _update(url: str, patch: Dict[str, Any], reason: str) -> None:
        nonlocal updated
        patch_full = {
            "presence": 0,
            "presence_checked": True,
            "status": "llm_presence_empty",
            "llm_presence_reason": reason,
        }
        patch_full.update(patch)
        upsert_url_index_entry(company.company_id, url, patch_full)
        updated += 1

    for url in pending_urls:
        ent = index.get(url) or {}
        md_path = ent.get("markdown_path")

        if not md_path:
            _update(url, {}, "no_markdown_path")
            continue

        text = _read_text(md_path)
        if text is None:
            logger.error(
                "LLM presence: failed reading markdown url=%s company=%s path=%s",
                url,
                company.company_id,
                md_path,
            )
            _update(url, {}, "markdown_read_error")
            continue

        if not text.strip():
            _update(url, {}, "empty_markdown")
            continue

        _md_debug_stats(url, md_path, text)

        t0 = time.perf_counter()
        try:
            raw_result = await asyncio.to_thread(
                call_llm_extract, presence_strategy, url, text, kind="presence"
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "LLM presence: extract error url=%s company=%s elapsed_ms=%.1f err=%s",
                url,
                company.company_id,
                elapsed_ms,
                e,
            )
            await state.mark_url_failed(
                company.company_id, url, f"presence_error:{type(e).__name__}"
            )
            _update(url, {}, "presence_exception")
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _save_raw_stage(company.company_id, url, raw_result, stage="llm_presence_raw")

        has_offering, confidence, preview = parse_presence_result(
            raw_result, default=False
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm_passes] presence_result url=%s elapsed_ms=%.1f has=%s conf=%s preview=%s",
                url,
                elapsed_ms,
                has_offering,
                confidence,
                preview,
            )

        patch: Dict[str, Any] = {
            "presence": 1 if has_offering else 0,
            "presence_checked": True,
            "status": "llm_presence_yes" if has_offering else "llm_presence_no",
        }
        if confidence is not None:
            patch["llm_presence_confidence"] = confidence
        if preview is not None:
            patch["llm_presence_preview"] = preview

        upsert_url_index_entry(company.company_id, url, patch)
        updated += 1

    logger.info(
        "LLM presence: updated %d URLs for company_id=%s", updated, company.company_id
    )
    await state.recompute_company_from_index(
        company.company_id, name=None, root_url=company.domain_url
    )


# ---------------------------------------------------------------------------
# Full extraction pass
# ---------------------------------------------------------------------------


async def run_full_pass_for_company(company: Any, *, full_strategy: Any) -> None:
    index = load_url_index(company.company_id)
    if not isinstance(index, dict) or not index:
        logger.info(
            "LLM full: no url_index entries for company_id=%s", company.company_id
        )
        return

    updated = 0

    for url, ent in index.items():
        if not isinstance(ent, dict):
            continue

        md_path = ent.get("markdown_path")
        if not md_path:
            continue
        if ent.get("extracted"):
            continue

        text = _read_text(md_path)
        if text is None:
            logger.error(
                "LLM full: failed reading markdown url=%s company=%s path=%s",
                url,
                company.company_id,
                md_path,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {"extracted": 0, "status": "llm_full_markdown_read_error"},
            )
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
            continue

        _md_debug_stats(url, md_path, text)

        t0 = time.perf_counter()
        try:
            raw_result = await asyncio.to_thread(
                call_llm_extract, full_strategy, url, text, kind="full"
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "LLM full: extract error url=%s company=%s elapsed_ms=%.1f err=%s",
                url,
                company.company_id,
                elapsed_ms,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {"extracted": 0, "status": f"llm_full_error:{type(e).__name__}"},
            )
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _save_raw_stage(company.company_id, url, raw_result, stage="llm_full_raw")

        payload = parse_extracted_payload(raw_result)
        payload_dict = payload.model_dump()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm_passes] full_parsed url=%s elapsed_ms=%.1f offerings=%d",
                url,
                elapsed_ms,
                len(payload.offerings),
            )

        try:
            product_path = save_stage_output(
                bvdid=company.company_id,
                url=url,
                data=json.dumps(payload_dict, ensure_ascii=False),
                stage="product",
            )
        except Exception as e:
            logger.error(
                "LLM full: failed writing product JSON url=%s company=%s err=%s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {"extracted": 0, "status": "llm_full_write_error"},
            )
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

    logger.info(
        "LLM full: wrote product JSON for %d URLs (company_id=%s)",
        updated,
        company.company_id,
    )

    state = get_crawl_state()
    await state.recompute_company_from_index(
        company.company_id, name=None, root_url=company.domain_url
    )
