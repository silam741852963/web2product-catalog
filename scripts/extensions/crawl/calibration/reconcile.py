from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

from configs.models import (
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_TERMINAL_DONE,
)
from extensions.crawl.state import CrawlState
from extensions.io.output_paths import ensure_output_root

from .db_migration import rebuild_db_to_current_schema
from .types import ReconcileReport, TERMINAL_RECONCILE_FROM


async def reconcile_async(
    *,
    out_dir: Path,
    db_path: Path,
    write_global_state: bool = True,
    dry_run: bool = False,
    max_examples: int = 10,
    concurrency: int = 32,
) -> ReconcileReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    rebuild_db_to_current_schema(actual_db_path)

    conc = max(1, int(concurrency))

    state = CrawlState(db_path=actual_db_path)
    try:
        rows = await state._query_all(
            """
            SELECT company_id, status, done_reason
            FROM companies
            """,
            tuple(),
        )

        scanned = len(rows)

        # Parallel classification (safe: pure-python on row dicts)
        q: asyncio.Queue[dict] = asyncio.Queue()
        for r in rows:
            q.put_nowait(dict(r))

        lock = asyncio.Lock()
        invariant_bad: List[str] = []
        to_upgrade: List[str] = []

        async def worker() -> None:
            while True:
                try:
                    r = q.get_nowait()
                except asyncio.QueueEmpty:
                    return

                cid = str(r.get("company_id", ""))
                st = str(r.get("status") or "").strip()
                dr = r.get("done_reason")

                if st == COMPANY_STATUS_TERMINAL_DONE and (
                    dr is None or str(dr).strip() == ""
                ):
                    async with lock:
                        invariant_bad.append(cid)
                    q.task_done()
                    continue

                has_reason = dr is not None and str(dr).strip() != ""
                if has_reason and st in TERMINAL_RECONCILE_FROM:
                    async with lock:
                        to_upgrade.append(cid)

                q.task_done()

        workers = [
            asyncio.create_task(worker()) for _ in range(min(conc, scanned or 1))
        ]
        await asyncio.gather(*workers)

        if invariant_bad:
            preview = ", ".join(invariant_bad[: max(1, int(max_examples))])
            raise RuntimeError(
                "invariant violation: status=terminal_done but done_reason is NULL/empty "
                f"(count={len(invariant_bad)} examples={preview})"
            )

        upgraded = 0
        if to_upgrade and not dry_run:
            placeholders = ",".join(["?"] * len(to_upgrade))
            sql = f"""
            UPDATE companies
            SET status=?
            WHERE company_id IN ({placeholders})
              AND done_reason IS NOT NULL
              AND TRIM(COALESCE(done_reason,'')) != ''
              AND status IN ('pending','in_progress','{COMPANY_STATUS_MD_NOT_DONE}')
            """
            await state._exec(sql, (COMPANY_STATUS_TERMINAL_DONE, *to_upgrade))
            upgraded = len(to_upgrade)
        else:
            upgraded = len(to_upgrade)

        wrote = False
        if write_global_state and not dry_run:
            await state.write_global_state_from_db_only(pretty=False)
            wrote = True

        return ReconcileReport(
            out_dir=str(out_dir),
            db_path=str(actual_db_path),
            dry_run=bool(dry_run),
            scanned_companies=int(scanned),
            upgraded_to_terminal_done=int(upgraded),
            invariant_errors=0,
            wrote_global_state=bool(wrote),
            example_company_ids=to_upgrade[: max(0, int(max_examples))],
        )
    finally:
        state.close()


def reconcile(
    *,
    out_dir: Path,
    db_path: Path,
    write_global_state: bool = True,
    dry_run: bool = False,
    max_examples: int = 10,
    concurrency: int = 32,
) -> ReconcileReport:
    return asyncio.run(
        reconcile_async(
            out_dir=out_dir,
            db_path=db_path,
            write_global_state=write_global_state,
            dry_run=dry_run,
            max_examples=max_examples,
            concurrency=concurrency,
        )
    )
