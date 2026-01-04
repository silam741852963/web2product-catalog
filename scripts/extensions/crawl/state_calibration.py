from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from extensions.crawl.state import (
    CrawlState,
    CompanySnapshot,
    load_crawl_meta,
    load_url_index,
    patch_crawl_meta,
    patch_url_index_meta,
)


@dataclass(frozen=True)
class CalibrationSample:
    bvdid: str
    db_snapshot: CompanySnapshot
    crawl_meta: Dict[str, Any]
    url_index_meta: Dict[str, Any]


@dataclass(frozen=True)
class CalibrationReport:
    db_path: str
    touched_companies: int
    wrote_global_state: bool
    sample_before: CalibrationSample
    sample_after: CalibrationSample


def asyncio_run(coro):
    return asyncio.run(coro)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # row tuple: (cid, name, type, notnull, dflt_value, pk)
    return [str(r[1]) for r in rows]


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _rebuild_db_to_current_schema(db_path: Path) -> None:
    """
    Rebuilds legacy DBs into the CURRENT crawl.state.py schema.

    Current companies columns (authoritative):
      bvdid TEXT PK,
      name TEXT,
      root_url TEXT,
      status TEXT,
      urls_total INTEGER,
      urls_markdown_done INTEGER,
      urls_llm_done INTEGER,
      last_error TEXT,
      done_reason TEXT,
      done_details TEXT,
      done_at TEXT,
      created_at TEXT,
      updated_at TEXT,
      industry INTEGER,
      nace INTEGER,
      industry_label TEXT,
      industry_source TEXT

    Current runs columns:
      run_id TEXT PK,
      pipeline TEXT,
      version TEXT,
      args_hash TEXT,
      crawl4ai_cache_base_dir TEXT,
      crawl4ai_cache_mode TEXT,
      started_at TEXT,
      total_companies INTEGER,
      completed_companies INTEGER,
      last_company_bvdid TEXT,
      last_updated TEXT

    Current run_company_done columns unchanged:
      run_id TEXT, bvdid TEXT, done_at TEXT
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")

        have_companies = _table_exists(conn, "companies")
        have_runs = _table_exists(conn, "runs")
        have_done = _table_exists(conn, "run_company_done")

        if not have_companies and not have_runs and not have_done:
            return

        old_companies_cols = _table_columns(conn, "companies") if have_companies else []
        old_runs_cols = _table_columns(conn, "runs") if have_runs else []
        old_done_cols = _table_columns(conn, "run_company_done") if have_done else []

        # New tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS companies_new (
                bvdid TEXT PRIMARY KEY,
                name TEXT,
                root_url TEXT,
                status TEXT,
                urls_total INTEGER NOT NULL DEFAULT 0,
                urls_markdown_done INTEGER NOT NULL DEFAULT 0,
                urls_llm_done INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                done_reason TEXT,
                done_details TEXT,
                done_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                industry INTEGER,
                nace INTEGER,
                industry_label TEXT,
                industry_source TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs_new (
                run_id TEXT PRIMARY KEY,
                pipeline TEXT,
                version TEXT,
                args_hash TEXT,
                crawl4ai_cache_base_dir TEXT,
                crawl4ai_cache_mode TEXT,
                started_at TEXT,
                total_companies INTEGER NOT NULL DEFAULT 0,
                completed_companies INTEGER NOT NULL DEFAULT 0,
                last_company_bvdid TEXT,
                last_updated TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_company_done_new (
                run_id TEXT NOT NULL,
                bvdid TEXT NOT NULL,
                done_at TEXT,
                PRIMARY KEY (run_id, bvdid)
            )
            """
        )

        def _cols_in(old_cols: Sequence[str], wanted: Sequence[str]) -> List[str]:
            return [c for c in wanted if c in set(old_cols)]

        def _insert_select(
            new_table: str,
            new_cols: Sequence[str],
            old_table: str,
            old_cols: Sequence[str],
        ) -> None:
            if not new_cols:
                return
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {new_table} ({", ".join(new_cols)})
                SELECT {", ".join(old_cols)} FROM {old_table}
                """
            )

        # 1) Copy runs (direct mapping where possible)
        if have_runs:
            wanted_runs = [
                "run_id",
                "pipeline",
                "version",
                "args_hash",
                "crawl4ai_cache_base_dir",
                "crawl4ai_cache_mode",
                "started_at",
                "total_companies",
                "completed_companies",
                "last_company_bvdid",
                "last_updated",
            ]
            cols = _cols_in(old_runs_cols, wanted_runs)
            _insert_select("runs_new", cols, "runs", cols)

        # 2) Copy run_company_done
        if have_done:
            wanted_done = ["run_id", "bvdid", "done_at"]
            cols = _cols_in(old_done_cols, wanted_done)
            _insert_select("run_company_done_new", cols, "run_company_done", cols)

        # 3) Copy companies with best-effort mapping for legacy columns.
        #    We do a row-wise migration to handle field renames:
        #      legacy industry_code (TEXT) -> industry (INTEGER) if parseable
        #      legacy industry_codes ignored
        if have_companies:
            # Figure out which legacy columns exist.
            has = set(old_companies_cols)

            select_cols: List[str] = [
                "bvdid",
                "name",
                "root_url",
                "status",
                "urls_total",
                "urls_markdown_done",
                "urls_llm_done",
                "last_error",
                "done_reason",
                "done_details",
                "done_at",
                "created_at",
                "updated_at",
            ]
            # optional legacy/current fields
            if "industry" in has:
                select_cols.append("industry")
            elif "industry_code" in has:
                select_cols.append("industry_code")
            else:
                select_cols.append("NULL AS industry_code")

            if "nace" in has:
                select_cols.append("nace")
            else:
                select_cols.append("NULL AS nace")

            if "industry_label" in has:
                select_cols.append("industry_label")
            else:
                select_cols.append("NULL AS industry_label")

            if "industry_source" in has:
                select_cols.append("industry_source")
            else:
                select_cols.append("NULL AS industry_source")

            rows = conn.execute(
                f"SELECT {', '.join(select_cols)} FROM companies"
            ).fetchall()

            # Build inserts.
            for r in rows:
                # indices:
                # 0..12 fixed, then industry-ish, nace, label, source
                bvdid = r[0]
                name = r[1]
                root_url = r[2]
                status = r[3]
                urls_total = r[4]
                urls_md = r[5]
                urls_llm = r[6]
                last_error = r[7]
                done_reason = r[8]
                done_details = r[9]
                done_at = r[10]
                created_at = r[11]
                updated_at = r[12]

                industry_raw = r[13]
                nace_raw = r[14]
                industry_label = r[15]
                industry_source = r[16]

                industry_val = _to_int_or_none(industry_raw)
                nace_val = _to_int_or_none(nace_raw)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO companies_new (
                        bvdid, name, root_url, status,
                        urls_total, urls_markdown_done, urls_llm_done,
                        last_error, done_reason, done_details, done_at,
                        created_at, updated_at,
                        industry, nace, industry_label, industry_source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bvdid,
                        name,
                        root_url,
                        status,
                        int(urls_total or 0),
                        int(urls_md or 0),
                        int(urls_llm or 0),
                        last_error,
                        done_reason,
                        done_details,
                        done_at,
                        created_at,
                        updated_at,
                        industry_val,
                        nace_val,
                        industry_label,
                        industry_source,
                    ),
                )

        # Swap tables
        if have_companies:
            conn.execute("DROP TABLE companies")
        if have_runs:
            conn.execute("DROP TABLE runs")
        if have_done:
            conn.execute("DROP TABLE run_company_done")

        conn.execute("ALTER TABLE companies_new RENAME TO companies")
        conn.execute("ALTER TABLE runs_new RENAME TO runs")
        conn.execute("ALTER TABLE run_company_done_new RENAME TO run_company_done")

    finally:
        conn.close()


def _pick_sample_bvdid(state: CrawlState, requested: Optional[str]) -> str:
    if requested is not None and requested.strip() != "":
        return requested.strip()

    row = (
        sqlite3.connect(str(state.db_path))
        .execute("SELECT bvdid FROM companies ORDER BY bvdid ASC LIMIT 1")
        .fetchone()
    )
    if row is None:
        raise RuntimeError("No companies found in DB to sample.")
    return str(row[0])


def _sample(state: CrawlState, bvdid: str) -> CalibrationSample:
    snap = asyncio_run(state.get_company_snapshot(bvdid, recompute=False))
    meta = load_crawl_meta(bvdid)
    idx = load_url_index(bvdid)
    idx_meta = idx.get("__meta__") if isinstance(idx, dict) else {}
    if idx_meta is None:
        idx_meta = {}
    if not isinstance(idx_meta, dict):
        raise ValueError("url_index.__meta__ must be a JSON object.")
    return CalibrationSample(
        bvdid=bvdid,
        db_snapshot=snap,
        crawl_meta=(meta if isinstance(meta, dict) else {}),
        url_index_meta=idx_meta,
    )


def calibrate(
    *,
    db_path: Optional[Path] = None,
    sample_bvdid: Optional[str] = None,
    write_global_state: bool = True,
) -> CalibrationReport:
    state = CrawlState(db_path=db_path)
    try:
        bvdid = _pick_sample_bvdid(state, sample_bvdid)
        sample_before = _sample(state, bvdid)

        # 1) Ensure DB schema matches current crawl.state.py
        _rebuild_db_to_current_schema(Path(state.db_path))

        # Re-open with current schema available
        state2 = CrawlState(db_path=Path(state.db_path))
        try:
            conn = sqlite3.connect(
                str(state2.db_path), isolation_level=None, timeout=30.0
            )
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=30000")

                # Pull the authoritative fields we mirror into crawl_meta.json
                rows = conn.execute(
                    """
                    SELECT
                        bvdid, name, root_url,
                        industry, nace, industry_label, industry_source
                    FROM companies
                    """
                ).fetchall()
            finally:
                conn.close()

            touched = 0
            for r in rows:
                b = str(r[0])
                name = r[1]
                root_url = r[2]
                industry = r[3]
                nace = r[4]
                industry_label = r[5]
                industry_source = r[6]

                patch: Dict[str, Any] = {}

                if name is not None:
                    patch["company_name"] = str(name)
                if root_url is not None:
                    patch["root_url"] = str(root_url)

                # These are the NEW canonical keys
                patch["industry"] = _to_int_or_none(industry)
                patch["nace"] = _to_int_or_none(nace)
                patch["industry_label"] = (
                    str(industry_label) if industry_label is not None else None
                )
                patch["industry_source"] = (
                    str(industry_source) if industry_source is not None else None
                )

                patch_crawl_meta(b, patch, pretty=True)

                # Ensure url_index has __meta__ object (even if empty),
                # so downstream code relying on meta fields is stable.
                idx = load_url_index(b)
                if isinstance(idx, dict) and idx:
                    patch_url_index_meta(b, {})

                touched += 1

            if write_global_state:
                asyncio_run(state2.write_global_state_from_db_only())

            sample_after = _sample(state2, bvdid)

            return CalibrationReport(
                db_path=str(state2.db_path),
                touched_companies=int(touched),
                wrote_global_state=bool(write_global_state),
                sample_before=sample_before,
                sample_after=sample_after,
            )
        finally:
            state2.close()
    finally:
        state.close()


def check(
    *,
    db_path: Optional[Path] = None,
    sample_bvdid: Optional[str] = None,
) -> CalibrationSample:
    state = CrawlState(db_path=db_path)
    try:
        bvdid = _pick_sample_bvdid(state, sample_bvdid)
        return _sample(state, bvdid)
    finally:
        state.close()
