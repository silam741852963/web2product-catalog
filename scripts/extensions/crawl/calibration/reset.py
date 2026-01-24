from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from configs.models import (
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_PENDING,
)
from extensions.crawl.state import CrawlState
from extensions.io.output_paths import ensure_output_root

from .db_migration import rebuild_db_to_current_schema
from .json_io import read_json_file, validate_json_bytes
from .paths import (
    company_output_dir,
    delete_company_output_dir,
    first_existing,
    url_index_candidates,
)
from .types import ResetCandidate, ResetReport, safe_norm_company_id_list
from .url_index import is_url_index_empty_semantic


def _atomic_write_json_compact(path: Path, data: object) -> None:
    """
    Deterministic atomic JSON write:
      - UTF-8
      - compact separators
      - atomic replace
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        Path(tmp_name).replace(path)
    except Exception:
        with contextlib.suppress(Exception):
            Path(tmp_name).unlink(missing_ok=True)  # py3.8+ compat in pathlib
        raise


def _load_json_dict_or_empty(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    obj = read_json_file(path)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise RuntimeError(f"retry-store file must be a JSON object: {path}")
    # normalize keys to str deterministically
    out: Dict[str, object] = {}
    for k, v in obj.items():
        out[str(k)] = v
    return out


def _clear_retry_store_for_company_ids(
    *, out_dir: Path, company_ids: List[str], dry_run: bool
) -> Tuple[int, int]:
    """
    Remove selected company_ids from persistent retry store:
      - out_dir/_retry/quarantine.json
      - out_dir/_retry/retry_state.json

    Returns (removed_quarantine, removed_state).
    """
    if not company_ids:
        return (0, 0)

    base = (Path(out_dir) / "_retry").resolve()
    quarantine_path = base / "quarantine.json"
    state_path = base / "retry_state.json"

    quarantine = _load_json_dict_or_empty(quarantine_path)
    state = _load_json_dict_or_empty(state_path)

    cid_set = set(company_ids)

    removed_quarantine = 0
    removed_state = 0

    for cid in list(quarantine.keys()):
        if cid in cid_set:
            removed_quarantine += 1
            if not dry_run:
                quarantine.pop(cid, None)

    for cid in list(state.keys()):
        if cid in cid_set:
            removed_state += 1
            if not dry_run:
                state.pop(cid, None)

    if not dry_run:
        # Only write if something changed to reduce churn
        if removed_quarantine > 0:
            _atomic_write_json_compact(quarantine_path, quarantine)
        if removed_state > 0:
            _atomic_write_json_compact(state_path, state)

    return (removed_quarantine, removed_state)


async def _reset_select_candidates_async(
    *,
    out_dir: Path,
    state: CrawlState,
    targets: Set[str],
    include_company_ids: Optional[Iterable[str]],
    exclude_company_ids: Optional[Iterable[str]],
    scan_concurrency: int = 64,
) -> List[ResetCandidate]:
    """
    Deterministic selection:
      - OR across targets
      - apply include/exclude last
      - return sorted by company_id
    Also records match reasons for audit.
    """
    rows = await state._query_all(
        """
        SELECT
            company_id,
            status,
            crawl_finished,
            done_reason
        FROM companies
        """,
        tuple(),
    )

    base_reasons: Dict[str, List[str]] = defaultdict(list)

    for r in rows:
        cid = str(r["company_id"])
        status = str(r["status"] or "").strip()
        crawl_finished = int(r["crawl_finished"] or 0)
        done_reason = str(r["done_reason"] or "").strip()

        if "crawl_not_finished" in targets and crawl_finished == 0:
            base_reasons[cid].append("crawl_not_finished")

        if "pending" in targets and status == COMPANY_STATUS_PENDING:
            base_reasons[cid].append("pending")

        if "markdown_not_done" in targets and status == COMPANY_STATUS_MD_NOT_DONE:
            base_reasons[cid].append("markdown_not_done")

        if "quarantined" in targets and done_reason == "quarantined":
            base_reasons[cid].append("quarantined")

    file_targets = {
        "url_index_corrupt",
        "url_index_empty",
        "missing_output_dir",
    }
    need_fs = bool(targets.intersection(file_targets))
    if need_fs:
        sem = asyncio.Semaphore(max(1, int(scan_concurrency)))

        async def _one(cid: str) -> None:
            async with sem:
                base = company_output_dir(out_dir, cid)

                if "missing_output_dir" in targets:
                    if not base.exists():
                        base_reasons[cid].append("missing_output_dir")

                if "url_index_corrupt" in targets or "url_index_empty" in targets:
                    p = first_existing(url_index_candidates(out_dir, cid))
                    if p is None:
                        return

                    ok, _ = validate_json_bytes(p)
                    if "url_index_corrupt" in targets and not ok:
                        base_reasons[cid].append("url_index_corrupt")
                        return

                    if "url_index_empty" in targets and ok:
                        idx = read_json_file(p)
                        if isinstance(idx, dict) and is_url_index_empty_semantic(idx):
                            base_reasons[cid].append("url_index_empty")

        cids = [str(r["company_id"]) for r in rows]
        batch = 512
        for i in range(0, len(cids), batch):
            await asyncio.gather(*(_one(cid) for cid in cids[i : i + batch]))

    selected: Dict[str, ResetCandidate] = {}
    for cid, rs in base_reasons.items():
        if not rs:
            continue
        reasons_sorted = sorted(set(rs))
        selected[cid] = ResetCandidate(company_id=cid, reasons=reasons_sorted)

    inc = set(safe_norm_company_id_list(include_company_ids))
    exc = set(safe_norm_company_id_list(exclude_company_ids))

    if inc:
        selected = {cid: c for cid, c in selected.items() if cid in inc}
    if exc:
        selected = {cid: c for cid, c in selected.items() if cid not in exc}

    return [selected[cid] for cid in sorted(selected.keys())]


def _list_tables_with_company_id(conn: sqlite3.Connection) -> List[str]:
    """
    Best-effort schema inspection:
    return all user tables (excluding sqlite_*) that contain a column named 'company_id'.
    """
    tables: List[str] = []
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    for (name,) in rows:
        try:
            cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            col_names = {str(c[1]) for c in cols}
            if "company_id" in col_names:
                tables.append(str(name))
        except Exception:
            # ignore any weird PRAGMA/table name edge cases
            continue
    # stable order
    return sorted(set(tables))


async def reset_async(
    *,
    out_dir: Path,
    db_path: Path,
    targets: Set[str],
    include_company_ids: Optional[Iterable[str]] = None,
    exclude_company_ids: Optional[Iterable[str]] = None,
    write_global_state: bool = True,
    dry_run: bool = False,
    max_examples: int = 200,
    scan_concurrency: int = 64,
    clear_retry_store: bool = False,
    delete_db_rows: bool = False,
) -> ResetReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    rebuild_db_to_current_schema(actual_db_path)

    state = CrawlState(db_path=actual_db_path)
    try:
        candidates = await _reset_select_candidates_async(
            out_dir=Path(out_dir),
            state=state,
            targets=set(targets),
            include_company_ids=include_company_ids,
            exclude_company_ids=exclude_company_ids,
            scan_concurrency=int(scan_concurrency),
        )

        selected_ids = [c.company_id for c in candidates]
        scanned_rows = await state._query_one(
            "SELECT COUNT(1) AS n FROM companies", tuple()
        )
        scanned = int(scanned_rows["n"] if scanned_rows else 0)

        deleted_dirs = 0
        missing_dirs = 0
        if selected_ids:
            for cid in selected_ids:
                if dry_run:
                    base = company_output_dir(Path(out_dir), cid)
                    if base.exists():
                        deleted_dirs += 1
                    else:
                        missing_dirs += 1
                else:
                    deleted, missing = await asyncio.to_thread(
                        delete_company_output_dir, Path(out_dir), cid
                    )
                    if deleted:
                        deleted_dirs += 1
                    if missing:
                        missing_dirs += 1

        # db_rows_reset:
        # - if delete_db_rows=False: number of companies rows UPDATED
        # - if delete_db_rows=True:  number of companies rows DELETED
        db_rows_reset = 0
        run_done_deleted = 0

        if selected_ids and not dry_run:
            conn = sqlite3.connect(
                str(actual_db_path), isolation_level=None, timeout=30.0
            )
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute("BEGIN")

                placeholders = ",".join(["?"] * len(selected_ids))

                # always delete run_company_done entries (same as before)
                before = conn.total_changes
                conn.execute(
                    f"DELETE FROM run_company_done WHERE company_id IN ({placeholders})",
                    tuple(selected_ids),
                )
                after = conn.total_changes
                run_done_deleted = int(after - before)

                if delete_db_rows:
                    # delete from any other table that has company_id (best-effort), excluding companies/run_company_done
                    tables = _list_tables_with_company_id(conn)
                    for t in tables:
                        if t in ("companies", "run_company_done"):
                            continue
                        # Some tables might be views or have triggers; ignore failures but keep transaction consistent.
                        # If a delete fails, we abort (raise) because partial deletion is worse than none.
                        conn.execute(
                            f"DELETE FROM {t} WHERE company_id IN ({placeholders})",
                            tuple(selected_ids),
                        )

                    # finally delete from companies
                    before2 = conn.total_changes
                    conn.execute(
                        f"DELETE FROM companies WHERE company_id IN ({placeholders})",
                        tuple(selected_ids),
                    )
                    after2 = conn.total_changes
                    db_rows_reset = int(after2 - before2)
                else:
                    # reset companies row fields (original behavior)
                    before2 = conn.total_changes
                    conn.execute(
                        f"""
                        UPDATE companies
                        SET
                            status=?,
                            crawl_finished=0,
                            urls_total=0,
                            urls_markdown_done=0,
                            urls_llm_done=0,
                            done_reason=NULL,
                            done_details=NULL,
                            done_at=NULL,
                            last_error=NULL,
                            last_crawled_at=NULL,

                            retry_next_eligible_at=0.0,
                            retry_attempts=0,
                            retry_same_error_streak=0,
                            retry_last_error='',
                            retry_last_stage='',
                            retry_last_error_sig='',
                            retry_last_error_sig_updated_at=0.0,

                            retry_net_attempts=0,
                            retry_stall_attempts=0,
                            retry_mem_attempts=0,
                            retry_other_attempts=0,
                            retry_mem_hits=0,
                            retry_last_stall_kind='unknown',
                            retry_last_progress_md_done=0,
                            retry_last_seen_md_done=0
                        WHERE company_id IN ({placeholders})
                        """,
                        (COMPANY_STATUS_PENDING, *selected_ids),
                    )
                    after2 = conn.total_changes
                    db_rows_reset = int(after2 - before2)

                conn.execute("COMMIT")
            except Exception:
                with contextlib.suppress(Exception):
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

        elif selected_ids and dry_run:
            placeholders = ",".join(["?"] * len(selected_ids))

            # companies rows "affected" (count only)
            r1 = await state._query_one(
                f"SELECT COUNT(1) AS n FROM companies WHERE company_id IN ({placeholders})",
                tuple(selected_ids),
            )
            db_rows_reset = int(r1["n"] if r1 else 0)

            # run_company_done rows that would be deleted
            r2 = await state._query_one(
                f"SELECT COUNT(1) AS n FROM run_company_done WHERE company_id IN ({placeholders})",
                tuple(selected_ids),
            )
            run_done_deleted = int(r2["n"] if r2 else 0)

        # --- clear persistent retry store entries (quarantine + retry_state) ---
        retry_quarantine_rows_deleted = 0
        retry_state_rows_deleted = 0
        if clear_retry_store and selected_ids:
            q_del, s_del = await asyncio.to_thread(
                _clear_retry_store_for_company_ids,
                out_dir=Path(out_dir),
                company_ids=list(selected_ids),
                dry_run=bool(dry_run),
            )
            retry_quarantine_rows_deleted = int(q_del)
            retry_state_rows_deleted = int(s_del)

        wrote = False
        if write_global_state and not dry_run:
            await state.write_global_state_from_db_only(pretty=False)
            wrote = True

        rep_candidates = candidates[: max(0, int(max_examples))]

        return ResetReport(
            out_dir=str(out_dir),
            db_path=str(actual_db_path),
            dry_run=bool(dry_run),
            targets=sorted(list(targets)),
            scanned_companies=int(scanned),
            selected_companies=int(len(selected_ids)),
            deleted_dirs=int(deleted_dirs),
            missing_dirs=int(missing_dirs),
            db_rows_reset=int(db_rows_reset),
            run_done_rows_deleted=int(run_done_deleted),
            retry_quarantine_rows_deleted=int(retry_quarantine_rows_deleted),
            retry_state_rows_deleted=int(retry_state_rows_deleted),
            wrote_global_state=bool(wrote),
            candidates=rep_candidates,
        )
    finally:
        state.close()


def reset(
    *,
    out_dir: Path,
    db_path: Path,
    targets: Set[str],
    include_company_ids: Optional[Iterable[str]] = None,
    exclude_company_ids: Optional[Iterable[str]] = None,
    write_global_state: bool = True,
    dry_run: bool = False,
    max_examples: int = 200,
    scan_concurrency: int = 64,
    clear_retry_store: bool = False,
    delete_db_rows: bool = False,
) -> ResetReport:
    return asyncio.run(
        reset_async(
            out_dir=out_dir,
            db_path=db_path,
            targets=targets,
            include_company_ids=include_company_ids,
            exclude_company_ids=exclude_company_ids,
            write_global_state=write_global_state,
            dry_run=dry_run,
            max_examples=max_examples,
            scan_concurrency=scan_concurrency,
            clear_retry_store=clear_retry_store,
            delete_db_rows=delete_db_rows,
        )
    )
