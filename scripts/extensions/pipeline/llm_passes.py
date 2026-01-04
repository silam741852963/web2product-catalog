from __future__ import annotations

import asyncio
import json
import logging
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from configs.llm import (
    IndustryContext,
    call_llm_extract,
    parse_extracted_payload,
    parse_presence_result,
    IndustryAwareStrategyCache,
)
from extensions.crawl.state import (
    get_crawl_state,
    load_url_index,
    upsert_url_index_entry,
)
from extensions.io.output_paths import save_stage_output

logger = logging.getLogger("llm_passes")


def llm_requested(
    args: argparse.Namespace,
    industry_llm_cache: Optional[IndustryAwareStrategyCache],
) -> bool:
    """
    LLM eligibility is request-based.
    """
    if bool(getattr(args, "finalize_in_progress_md", False)):
        return False
    if str(getattr(args, "llm_mode", "none")) not in ("presence", "full"):
        return False
    return industry_llm_cache is not None


def has_industry_context(company: Any) -> bool:
    """
    True if company has either:
      - industry_label, OR
      - both industry and nace ints
    """
    label = _get_str(company, "industry_label")
    if label:
        return True
    industry = _get_int(company, "industry")
    nace = _get_int(company, "nace")
    return industry is not None and nace is not None


def build_industry_context(company: Any) -> IndustryContext:
    """
    Always returns an IndustryContext (unknown/-1 fallback),
    so callers don't need branching.
    """
    label = _get_str(company, "industry_label") or "unknown"
    industry = _get_int(company, "industry")
    nace = _get_int(company, "nace")
    return IndustryContext(
        industry_label=label,
        industry=int(industry) if industry is not None else -1,
        nace=int(nace) if nace is not None else -1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _md_debug_stats(url: str, md_path: str, text: str) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    import hashlib

    ss = " ".join((text or "").split())
    sha1 = hashlib.sha1(ss.encode("utf-8", errors="ignore")).hexdigest()

    logger.debug(
        "[llm_passes] md_stats url=%s md_path=%s len=%d sha1=%s",
        url,
        md_path,
        len(ss),
        sha1,
    )


# ---------------------------------------------------------------------------
# RAW SAVE — PRESENCE ONLY
# ---------------------------------------------------------------------------


def _save_presence_raw(
    company_id: str,
    url: str,
    raw: Any,
) -> None:
    """
    Presence-only raw persistence.

    Full pass raw output MUST NOT be saved to metadata,
    to avoid duplicating product JSON.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    payload: Dict[str, Any] = {
        "kind": "llm_raw",
        "substage": "presence",
        "company_id": company_id,
        "source_url": url,
        "saved_at": _now_iso(),
        "raw": raw,
    }

    save_stage_output(
        bvdid=company_id,
        url=url,
        data=json.dumps(payload, ensure_ascii=False),
        stage="metadata",
    )


# ---------------------------------------------------------------------------
# Company helpers
# ---------------------------------------------------------------------------


def _get_int(obj: Any, attr: str) -> Optional[int]:
    v = getattr(obj, attr, None)
    return v if isinstance(v, int) else None


def _get_str(obj: Any, attr: str) -> Optional[str]:
    v = getattr(obj, attr, None)
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _company_id(company: Any) -> str:
    v = _get_str(company, "company_id")
    if not v:
        raise ValueError("company.company_id must be a non-empty str")
    return v


def _company_root_url(company: Any) -> str:
    v = _get_str(company, "domain_url")
    if not v:
        raise ValueError("company.domain_url must be a non-empty str")
    return v


def _try_industry_ctx(company: Any) -> Optional[IndustryContext]:
    if not has_industry_context(company):
        return None
    # Keep Optional behavior for metadata (“generic ctx” vs real)
    ctx = build_industry_context(company)
    # If it’s only label-based and missing ints, build_industry_context gives -1;
    # we still treat that as context-present, but metadata will reflect it.
    return ctx


def _effective_ctx_meta(ctx: Optional[IndustryContext]) -> Dict[str, Any]:
    if ctx is None:
        return {
            "ctx_is_generic": True,
            "industry_label_used": "",
            "industry_used": None,
            "nace_used": None,
        }

    return {
        "ctx_is_generic": not bool((ctx.industry_label or "").strip()),
        "industry_label_used": (ctx.industry_label or "").strip(),
        "industry_used": ctx.industry,
        "nace_used": ctx.nace,
    }


def _strategy_meta(strategy: Any, *, llm_mode: str) -> Dict[str, Any]:
    llm_cfg = getattr(strategy, "llm_config", None)
    if llm_cfg is None:
        raise ValueError("strategy.llm_config is required")

    provider = getattr(llm_cfg, "provider", None)
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("strategy.llm_config.provider must be non-empty")

    instr = getattr(strategy, "instruction", None)
    if not isinstance(instr, str) or not instr.strip():
        raise ValueError("strategy.instruction must be non-empty")

    import hashlib

    return {
        "llm_mode": llm_mode,
        "llm_provider": provider,
        "llm_instruction_sha1": hashlib.sha1(
            instr.encode("utf-8", errors="ignore")
        ).hexdigest(),
    }


def _patch_base(
    strategy: Any,
    *,
    llm_mode: str,
    ctx: Optional[IndustryContext],
) -> Dict[str, Any]:
    return {
        **_effective_ctx_meta(ctx),
        **_strategy_meta(strategy, llm_mode=llm_mode),
    }


# ---------------------------------------------------------------------------
# Presence pass
# ---------------------------------------------------------------------------


async def run_presence_pass_for_company(
    company: Any, *, presence_strategy: Any, repo_root: Path
) -> None:
    state = get_crawl_state()
    company_id = _company_id(company)
    root_url = _company_root_url(company)

    ctx = _try_industry_ctx(company)
    base = _patch_base(presence_strategy, llm_mode="presence", ctx=ctx)

    pending_urls = await state.get_pending_urls_for_llm(company_id)
    if not pending_urls:
        await state.recompute_company_from_index(
            company_id, name=getattr(company, "name", None), root_url=root_url
        )
        return

    index = load_url_index(company_id)
    updated = 0

    for url in pending_urls:
        ent = index.get(url, {}) if isinstance(index.get(url), dict) else {}
        md_path = ent.get("markdown_path")

        if not md_path:
            upsert_url_index_entry(
                company_id,
                url,
                {
                    **base,
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_presence_skipped_no_markdown",
                    "updated_at": _now_iso(),
                },
            )
            updated += 1
            continue

        text = _read_text(md_path)
        if not text.strip():
            upsert_url_index_entry(
                company_id,
                url,
                {
                    **base,
                    "presence": 0,
                    "presence_checked": True,
                    "status": "llm_presence_empty_markdown",
                    "updated_at": _now_iso(),
                },
            )
            updated += 1
            continue

        _md_debug_stats(url, md_path, text)

        t0 = time.perf_counter()
        raw = await asyncio.to_thread(
            call_llm_extract, presence_strategy, url, text, kind="presence"
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        _save_presence_raw(company_id, url, raw)

        has_offering, confidence, preview = parse_presence_result(raw, default=False)

        patch = {
            **base,
            "presence": 1 if has_offering else 0,
            "presence_checked": True,
            "status": "llm_presence_yes" if has_offering else "llm_presence_no",
            "llm_presence_elapsed_ms": round(elapsed_ms, 1),
            "updated_at": _now_iso(),
        }
        if confidence is not None:
            patch["llm_presence_confidence"] = confidence
        if preview is not None:
            patch["llm_presence_preview"] = preview

        upsert_url_index_entry(company_id, url, patch)
        updated += 1

    logger.info("LLM presence: updated %d URLs (company_id=%s)", updated, company_id)
    await state.recompute_company_from_index(
        company_id, name=getattr(company, "name", None), root_url=root_url
    )


# ---------------------------------------------------------------------------
# Full pass
# ---------------------------------------------------------------------------


async def run_full_pass_for_company(
    company: Any, *, full_strategy: Any, repo_root: Path
) -> None:
    state = get_crawl_state()
    company_id = _company_id(company)
    root_url = _company_root_url(company)

    ctx = _try_industry_ctx(company)
    base = _patch_base(full_strategy, llm_mode="full", ctx=ctx)

    pending_urls = await state.get_pending_urls_for_llm(company_id)
    if not pending_urls:
        await state.recompute_company_from_index(
            company_id, name=getattr(company, "name", None), root_url=root_url
        )
        return

    index = load_url_index(company_id)
    updated = 0

    for url in pending_urls:
        ent = index.get(url, {}) if isinstance(index.get(url), dict) else {}
        md_path = ent.get("markdown_path")

        if not md_path:
            upsert_url_index_entry(
                company_id,
                url,
                {
                    **base,
                    "extracted": 1,
                    "status": "llm_full_skipped_no_markdown",
                    "updated_at": _now_iso(),
                },
            )
            updated += 1
            continue

        text = _read_text(md_path)
        if not text.strip():
            upsert_url_index_entry(
                company_id,
                url,
                {
                    **base,
                    "extracted": 1,
                    "status": "llm_full_empty_markdown",
                    "updated_at": _now_iso(),
                },
            )
            updated += 1
            continue

        _md_debug_stats(url, md_path, text)

        t0 = time.perf_counter()
        raw = await asyncio.to_thread(
            call_llm_extract, full_strategy, url, text, kind="full"
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        payload = parse_extracted_payload(raw)
        product = payload.model_dump()

        product.update(
            {
                "company_id": company_id,
                "source_url": url,
                **_effective_ctx_meta(ctx),
                **_strategy_meta(full_strategy, llm_mode="full"),
            }
        )

        product_path = save_stage_output(
            bvdid=company_id,
            url=url,
            data=json.dumps(product, ensure_ascii=False),
            stage="product",
        )

        upsert_url_index_entry(
            company_id,
            url,
            {
                **base,
                "extracted": 1,
                "presence": 1 if payload.offerings else 0,
                "presence_checked": True,
                "status": "llm_full_extracted",
                "llm_full_elapsed_ms": round(elapsed_ms, 1),
                "product_path": str(product_path),
                "updated_at": _now_iso(),
            },
        )
        updated += 1

    logger.info("LLM full: updated %d URLs (company_id=%s)", updated, company_id)
    await state.recompute_company_from_index(
        company_id, name=getattr(company, "name", None), root_url=root_url
    )
