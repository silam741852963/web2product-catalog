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
# Metadata helpers (industry/profile + LLM provider/model)
# ---------------------------------------------------------------------------


def _industry_meta(company: Any) -> Dict[str, Any]:
    """
    Ensure every URL index entry knows which industry profile produced it.

    Expected fields on company (from run.py integration):
      - company.industry_code
      - company.industry_profile_id
      - company.industry_codes (optional, multi-code canonical "9|10|20")
      - company.industry_label (optional)
    """
    out: Dict[str, Any] = {}

    code = getattr(company, "industry_code", None)
    prof = getattr(company, "industry_profile_id", None)

    if code is not None:
        out["industry_code"] = code
    if prof is not None:
        out["industry_profile_id"] = prof

    # Optional but useful for analytics/debugging
    codes = getattr(company, "industry_codes", None)
    label = getattr(company, "industry_label", None)
    if codes is not None:
        out["industry_codes"] = codes
    if label is not None:
        out["industry_label"] = label

    return out


def _strategy_meta(strategy: Any, *, llm_mode: str) -> Dict[str, Any]:
    """
    Best-effort: capture model/provider + mode (presence/schema).
    Safe across Crawl4AI/LiteLLM variations.
    """
    out: Dict[str, Any] = {"llm_mode": llm_mode}

    llm_cfg = getattr(strategy, "llm_config", None)
    provider = None
    model = None

    if llm_cfg is not None:
        provider = getattr(llm_cfg, "provider", None)
        # some configs expose model separately; most encode it in provider
        model = getattr(llm_cfg, "model", None)

    if provider is None:
        provider = getattr(strategy, "provider", None) or getattr(
            strategy, "model", None
        )

    if provider is not None:
        out["llm_provider"] = str(provider)

    if model is not None:
        out["llm_model"] = str(model)

    # Also helpful if you later rotate profiles/instructions
    instr = getattr(strategy, "instruction", None)
    if isinstance(instr, str) and instr:
        # store hash only (keep url_index.json compact)
        import hashlib

        out["llm_instruction_sha1"] = hashlib.sha1(
            instr.encode("utf-8", errors="ignore")
        ).hexdigest()

    return out


def _patch_base(company: Any, strategy: Any, *, llm_mode: str) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    p.update(_industry_meta(company))
    p.update(_strategy_meta(strategy, llm_mode=llm_mode))
    return p


# ---------------------------------------------------------------------------
# Presence pass
# ---------------------------------------------------------------------------


async def run_presence_pass_for_company(
    company: Any, *, presence_strategy: Any
) -> None:
    state = get_crawl_state()

    base = _patch_base(company, presence_strategy, llm_mode="presence")

    # ---- SAFETY SWEEP (before pending_urls) ----
    index = load_url_index(company.company_id) or {}
    if not isinstance(index, dict):
        index = {}

    swept = 0
    for url, ent in list(index.items()):
        ent = ent if isinstance(ent, dict) else {}
        if ent.get("presence_checked") is True:
            continue
        if ent.get("markdown_path"):
            continue

        patch = {
            **base,
            "presence": 0,
            "presence_checked": True,
            "status": ent.get("status") or "llm_presence_skipped_no_markdown",
            "llm_presence_reason": "no_markdown_path",
        }
        upsert_url_index_entry(company.company_id, url, patch)
        swept += 1

    if swept:
        logger.info(
            "LLM presence: swept %d no-markdown URLs (company_id=%s)",
            swept,
            company.company_id,
        )

    pending_urls = await state.get_pending_urls_for_llm(company.company_id)
    if not pending_urls:
        logger.info(
            "LLM presence: no pending URLs for company_id=%s", company.company_id
        )
        await state.recompute_company_from_index(
            company.company_id,
            name=getattr(company, "name", None),
            root_url=company.domain_url,
        )
        return

    logger.info(
        "LLM presence: %d pending URLs for company_id=%s",
        len(pending_urls),
        company.company_id,
    )

    updated = 0

    def _update(url: str, patch: Dict[str, Any], reason: str) -> None:
        nonlocal updated
        patch_full = {
            **base,
            "presence": 0,
            "presence_checked": True,
            "status": "llm_presence_empty",
            "llm_presence_reason": reason,
        }
        patch_full.update(patch)
        upsert_url_index_entry(company.company_id, url, patch_full)
        updated += 1

    for url in pending_urls:
        # reload entry each time (avoid stale index issues)
        ent = (load_url_index(company.company_id) or {}).get(url) or {}
        ent = ent if isinstance(ent, dict) else {}

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
            # if your CrawlState supports it
            try:
                await state.mark_url_failed(
                    company.company_id, url, f"presence_error:{type(e).__name__}"
                )
            except Exception:
                pass
            _update(url, {"status": "llm_presence_exception"}, "presence_exception")
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
            **base,
            "presence": 1 if has_offering else 0,
            "presence_checked": True,
            "status": "llm_presence_yes" if has_offering else "llm_presence_no",
            "llm_presence_elapsed_ms": round(elapsed_ms, 1),
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
        company.company_id,
        name=getattr(company, "name", None),
        root_url=company.domain_url,
    )


# ---------------------------------------------------------------------------
# Full extraction pass
# ---------------------------------------------------------------------------


async def run_full_pass_for_company(company: Any, *, full_strategy: Any) -> None:
    state = get_crawl_state()

    base = _patch_base(company, full_strategy, llm_mode="schema")

    # ---- SAFETY SWEEP (before pending_urls) ----
    index0 = load_url_index(company.company_id)
    index0 = index0 if isinstance(index0, dict) else {}

    swept = 0
    for url, ent in list(index0.items()):
        ent = ent if isinstance(ent, dict) else {}
        if ent.get("extracted") == 1:
            continue
        if ent.get("markdown_path"):
            continue

        patch = {
            **base,
            "extracted": 1,
            "presence": 0,
            "presence_checked": True,
            "status": ent.get("status") or "llm_full_skipped_no_markdown",
            "llm_full_reason": "no_markdown_path",
        }
        upsert_url_index_entry(company.company_id, url, patch)
        swept += 1

    if swept:
        logger.info(
            "LLM full: swept %d no-markdown URLs (company_id=%s)",
            swept,
            company.company_id,
        )

    pending_urls = await state.get_pending_urls_for_llm(company.company_id)
    if not pending_urls:
        logger.info("LLM full: no pending URLs for company_id=%s", company.company_id)
        await state.recompute_company_from_index(
            company.company_id,
            name=getattr(company, "name", None),
            root_url=company.domain_url,
        )
        return

    logger.info(
        "LLM full: %d pending URLs for company_id=%s",
        len(pending_urls),
        company.company_id,
    )

    updated = 0

    for url in pending_urls:
        # reload entry each time (avoid stale index issues)
        ent = (load_url_index(company.company_id) or {}).get(url) or {}
        ent = ent if isinstance(ent, dict) else {}

        # If presence pass already said "no offerings", mark extracted=1 immediately.
        if ent.get("presence_checked") is True and int(ent.get("presence") or 0) == 0:
            patch = {
                **base,
                "extracted": 1,
                "presence": 0,
                "presence_checked": True,
                "status": ent.get("status") or "llm_full_skipped_presence_no",
                "llm_full_reason": "presence_no",
            }
            upsert_url_index_entry(company.company_id, url, patch)
            updated += 1
            continue

        md_path = ent.get("markdown_path")

        # URLs can be markdown-complete but have no markdown_path.
        if not md_path:
            patch = {
                **base,
                "extracted": 1,
                "presence": int(ent.get("presence") or 0),
                "presence_checked": bool(ent.get("presence_checked") is True),
                "llm_full_reason": "no_markdown_path",
            }
            if not ent.get("status"):
                patch["status"] = "llm_full_skipped_no_markdown"
            upsert_url_index_entry(company.company_id, url, patch)
            updated += 1
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
                {
                    **base,
                    "extracted": 0,
                    "status": "llm_full_markdown_read_error",
                    "llm_full_reason": "markdown_read_error",
                },
            )
            continue

        if not text.strip():
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    **base,
                    "extracted": 1,
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_full_empty_markdown",
                    "llm_full_reason": "empty_markdown",
                },
            )
            updated += 1
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
                {
                    **base,
                    "extracted": 0,
                    "status": f"llm_full_error:{type(e).__name__}",
                    "llm_full_reason": "extract_exception",
                    "llm_full_elapsed_ms": round(elapsed_ms, 1),
                },
            )
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _save_raw_stage(company.company_id, url, raw_result, stage="llm_full_raw")

        try:
            payload = parse_extracted_payload(raw_result)
            payload_dict = payload.model_dump()
        except Exception as e:
            logger.exception(
                "LLM full: parse error url=%s company=%s err=%s",
                url,
                company.company_id,
                e,
            )
            upsert_url_index_entry(
                company.company_id,
                url,
                {
                    **base,
                    "extracted": 0,
                    "status": "llm_full_parse_error",
                    "llm_full_reason": "parse_error",
                    "llm_full_elapsed_ms": round(elapsed_ms, 1),
                },
            )
            continue

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm_passes] full_parsed url=%s elapsed_ms=%.1f offerings=%d",
                url,
                elapsed_ms,
                len(payload.offerings),
            )

        # Embed context into saved product JSON
        payload_dict_with_meta = dict(payload_dict)
        payload_dict_with_meta.setdefault("company_id", company.company_id)
        payload_dict_with_meta.setdefault("source_url", url)
        payload_dict_with_meta.update(_industry_meta(company))
        payload_dict_with_meta.update(_strategy_meta(full_strategy, llm_mode="schema"))

        product_path = None
        try:
            product_path = save_stage_output(
                bvdid=company.company_id,
                url=url,
                data=json.dumps(payload_dict_with_meta, ensure_ascii=False),
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
                {
                    **base,
                    "extracted": 0,
                    "status": "llm_full_write_error",
                    "llm_full_reason": "write_error",
                    "llm_full_elapsed_ms": round(elapsed_ms, 1),
                },
            )
            continue

        presence_flag = 1 if payload.offerings else 0
        patch2: Dict[str, Any] = {
            **base,
            "extracted": 1,
            "presence": presence_flag,
            "presence_checked": True,
            "status": "llm_full_extracted",
            "llm_full_elapsed_ms": round(elapsed_ms, 1),
        }
        if product_path is not None:
            patch2["product_path"] = str(product_path)

        upsert_url_index_entry(company.company_id, url, patch2)
        updated += 1

    logger.info(
        "LLM full: updated %d URLs (company_id=%s)", updated, company.company_id
    )
    await state.recompute_company_from_index(
        company.company_id,
        name=getattr(company, "name", None),
        root_url=company.domain_url,
    )
