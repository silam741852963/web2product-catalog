from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import List

from extensions.crawl.state import CrawlState
from extensions.io.output_paths import ensure_output_root

from .db_migration import rebuild_db_to_current_schema
from .json_io import (
    bytes_head_hex,
    bytes_text_preview,
    read_bytes,
    validate_json_bytes,
)
from .paths import url_index_candidates
from .types import CorruptJsonFile, CorruptionFixReport, CorruptionReport


def _collect_corrupt_for_company(
    out_dir: Path, company_id: str
) -> List[CorruptJsonFile]:
    out: List[CorruptJsonFile] = []
    for p in url_index_candidates(out_dir, company_id):
        if not (p.exists() and p.is_file()):
            continue

        ok, reason = validate_json_bytes(p)
        if ok:
            continue

        b = read_bytes(p)
        if b is None:
            b = b""

        out.append(
            CorruptJsonFile(
                company_id=company_id,
                path=str(p),
                size_bytes=int(len(b)),
                reason=str(reason),
                head_bytes_hex=bytes_head_hex(b, 80),
                head_text_preview=bytes_text_preview(b, 200),
            )
        )
    return out


def _quarantine_file(path: Path) -> None:
    """
    Rename corrupted file aside:
      url_index.json -> url_index.json.corrupt.<ts>.<pid>

    Raises on failure.
    """
    if not path.exists():
        return
    ts = int(time.time())
    dst = Path(str(path) + f".corrupt.{ts}.{os.getpid()}")
    path.rename(dst)


async def _unmark_latest_run_done(state: CrawlState, company_id: str) -> int:
    """
    Remove (latest_run_id, company_id) from run_company_done so the scheduler won't skip it.
    Returns 1 if a row existed and was deleted, else 0.
    """
    row = await state._query_one(
        "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1",
        tuple(),
    )
    if row is None:
        return 0

    rid = str(row["run_id"])
    existed = await state._query_one(
        "SELECT 1 AS x FROM run_company_done WHERE run_id=? AND company_id=? LIMIT 1",
        (rid, company_id),
    )
    if existed is None:
        return 0

    await state._exec(
        "DELETE FROM run_company_done WHERE run_id=? AND company_id=?",
        (rid, company_id),
    )
    return 1


async def scan_corrupt_url_indexes_async(
    *,
    out_dir: Path,
    db_path: Path,
    max_examples: int = 1,
    concurrency: int = 32,
) -> CorruptionReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    rebuild_db_to_current_schema(actual_db_path)

    conc = max(1, int(concurrency))

    state = CrawlState(db_path=actual_db_path)
    try:
        rows = await state._query_all("SELECT company_id FROM companies", tuple())
        db_ids = [str(r["company_id"]) for r in rows]

        examples: List[CorruptJsonFile] = []
        affected_company_set: set[str] = set()
        affected_files = 0

        lock = asyncio.Lock()
        q: asyncio.Queue[str] = asyncio.Queue()
        for cid in db_ids:
            q.put_nowait(cid)

        async def worker() -> None:
            nonlocal affected_files
            while True:
                try:
                    cid = q.get_nowait()
                except asyncio.QueueEmpty:
                    return

                files = await asyncio.to_thread(
                    _collect_corrupt_for_company, out_dir, cid
                )
                if files:
                    async with lock:
                        affected_company_set.add(cid)
                        affected_files += len(files)
                        if len(examples) < max_examples:
                            examples.extend(files[: max_examples - len(examples)])

                q.task_done()

        workers = [
            asyncio.create_task(worker()) for _ in range(min(conc, len(db_ids) or 1))
        ]
        await asyncio.gather(*workers)

        return CorruptionReport(
            out_dir=str(out_dir),
            db_path=str(actual_db_path),
            scanned_companies=int(len(db_ids)),
            affected_companies=int(len(affected_company_set)),
            affected_files=int(affected_files),
            examples=examples,
        )
    finally:
        state.close()


def scan_corrupt_url_indexes(
    *,
    out_dir: Path,
    db_path: Path,
    max_examples: int = 1,
    concurrency: int = 32,
) -> CorruptionReport:
    return asyncio.run(
        scan_corrupt_url_indexes_async(
            out_dir=out_dir,
            db_path=db_path,
            max_examples=max_examples,
            concurrency=concurrency,
        )
    )


async def fix_corrupt_url_indexes_async(
    *,
    out_dir: Path,
    db_path: Path,
    max_examples: int = 1,
    mark_pending: bool = True,
    quarantine: bool = True,
    unmark_run_done: bool = True,
    dry_run: bool = False,
    concurrency: int = 32,
) -> CorruptionFixReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    rebuild_db_to_current_schema(actual_db_path)

    conc = max(1, int(concurrency))

    state = CrawlState(db_path=actual_db_path)
    db_lock = asyncio.Lock()
    try:
        rows = await state._query_all("SELECT company_id FROM companies", tuple())
        db_ids = [str(r["company_id"]) for r in rows]

        examples: List[CorruptJsonFile] = []
        affected_company_set: set[str] = set()
        quarantined_files = 0
        marked_pending_count = 0
        run_done_unmarked = 0

        lock = asyncio.Lock()
        q: asyncio.Queue[str] = asyncio.Queue()
        for cid in db_ids:
            q.put_nowait(cid)

        async def worker() -> None:
            nonlocal quarantined_files, marked_pending_count, run_done_unmarked
            while True:
                try:
                    cid = q.get_nowait()
                except asyncio.QueueEmpty:
                    return

                files = await asyncio.to_thread(
                    _collect_corrupt_for_company, out_dir, cid
                )
                if files:
                    async with lock:
                        affected_company_set.add(cid)
                        if len(examples) < max_examples:
                            examples.extend(files[: max_examples - len(examples)])

                    if quarantine:
                        for f in files:
                            if dry_run:
                                async with lock:
                                    quarantined_files += 1
                            else:
                                await asyncio.to_thread(_quarantine_file, Path(f.path))
                                async with lock:
                                    quarantined_files += 1

                    if mark_pending:
                        if dry_run:
                            async with lock:
                                marked_pending_count += 1
                        else:
                            async with db_lock:
                                await state.upsert_company(
                                    cid,
                                    status="pending",
                                    crawl_finished=False,
                                    urls_total=0,
                                    urls_markdown_done=0,
                                    urls_llm_done=0,
                                    done_reason=None,
                                    done_details=None,
                                    done_at=None,
                                    last_error="corrupt_url_index_json",
                                    write_meta=True,
                                )
                            async with lock:
                                marked_pending_count += 1

                    if unmark_run_done:
                        if dry_run:
                            async with lock:
                                run_done_unmarked += 1
                        else:
                            async with db_lock:
                                n = await _unmark_latest_run_done(state, cid)
                            async with lock:
                                run_done_unmarked += int(n)

                q.task_done()

        workers = [
            asyncio.create_task(worker()) for _ in range(min(conc, len(db_ids) or 1))
        ]
        await asyncio.gather(*workers)

        return CorruptionFixReport(
            out_dir=str(out_dir),
            db_path=str(actual_db_path),
            dry_run=bool(dry_run),
            scanned_companies=int(len(db_ids)),
            affected_companies=int(len(affected_company_set)),
            quarantined_files=int(quarantined_files),
            marked_pending=int(marked_pending_count),
            run_done_unmarked=int(run_done_unmarked),
            examples=examples,
        )
    finally:
        state.close()


def fix_corrupt_url_indexes(
    *,
    out_dir: Path,
    db_path: Path,
    max_examples: int = 1,
    mark_pending: bool = True,
    quarantine: bool = True,
    unmark_run_done: bool = True,
    dry_run: bool = False,
    concurrency: int = 32,
) -> CorruptionFixReport:
    return asyncio.run(
        fix_corrupt_url_indexes_async(
            out_dir=out_dir,
            db_path=db_path,
            max_examples=max_examples,
            mark_pending=mark_pending,
            quarantine=quarantine,
            unmark_run_done=unmark_run_done,
            dry_run=dry_run,
            concurrency=concurrency,
        )
    )
