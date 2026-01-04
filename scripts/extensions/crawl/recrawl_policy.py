from __future__ import annotations

from contextlib import suppress
from typing import List, Optional

from configs.models import (
    Company,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_TERMINAL_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_LLM_DONE,
)

from extensions.crawl.state import CrawlState


async def apply_recrawl_policy(
    state: CrawlState,
    companies: List[Company],
    current_max_pages: int,
    prev_run_max_pages: Optional[int],
    force_full_recrawl: bool,
) -> int:
    touched = 0

    if force_full_recrawl:
        for c in companies:
            snap = await state.get_company_snapshot(c.company_id, recompute=False)
            st = snap.status or COMPANY_STATUS_PENDING
            if st == COMPANY_STATUS_TERMINAL_DONE:
                continue

            await state.upsert_company(
                c.company_id,
                status=COMPANY_STATUS_MD_NOT_DONE,
                last_error=None,
            )
            with suppress(Exception):
                await state.patch_company_meta(
                    c.company_id,
                    {"max_pages": int(current_max_pages), "recrawl_forced": True},
                    pretty=True,
                )
            touched += 1

        if touched:
            await state.recompute_all_in_progress(concurrency=32)
        return touched

    if prev_run_max_pages is None or current_max_pages <= prev_run_max_pages:
        for c in companies:
            with suppress(Exception):
                await state.patch_company_meta(
                    c.company_id,
                    {"max_pages": int(current_max_pages)},
                    pretty=True,
                )
        return 0

    old_cap = int(prev_run_max_pages)

    for c in companies:
        snap = await state.get_company_snapshot(c.company_id, recompute=False)
        st = snap.status or COMPANY_STATUS_PENDING
        if st == COMPANY_STATUS_TERMINAL_DONE:
            continue

        urls_total = int(snap.urls_total)
        if urls_total == old_cap and st in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
        ):
            await state.upsert_company(
                c.company_id,
                status=COMPANY_STATUS_MD_NOT_DONE,
            )
            with suppress(Exception):
                await state.patch_company_meta(
                    c.company_id,
                    {
                        "max_pages": int(current_max_pages),
                        "recrawl_reason": "max_pages_increased",
                        "prev_max_pages": int(old_cap),
                    },
                    pretty=True,
                )
            touched += 1
        else:
            with suppress(Exception):
                await state.patch_company_meta(
                    c.company_id,
                    {"max_pages": int(current_max_pages)},
                    pretty=True,
                )

    if touched:
        await state.recompute_all_in_progress(concurrency=32)
    return touched
