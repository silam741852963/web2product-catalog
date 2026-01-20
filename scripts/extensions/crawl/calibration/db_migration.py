from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

from .types import now_iso


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _table_info(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(f"PRAGMA table_info({table})").fetchall()


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [str(r["name"]) for r in _table_info(conn, table)]


def _pk_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [str(r["name"]) for r in _table_info(conn, table) if int(r["pk"] or 0) > 0]


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    return s if s.strip() else None


def _should_rebuild_schema(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "companies"):
        return False

    cols = set(_table_columns(conn, "companies"))
    if "company_id" in cols:
        pks = _pk_columns(conn, "companies")
        return not (len(pks) == 1 and pks[0] == "company_id")

    if "bvdid" in cols:
        return True
    return True


def _create_new_schema_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS companies_new (
            company_id TEXT PRIMARY KEY,
            root_url TEXT,
            name TEXT,
            metadata_json TEXT,

            industry INTEGER,
            nace INTEGER,
            industry_label TEXT,
            industry_label_source TEXT,

            status TEXT,
            crawl_finished INTEGER DEFAULT 0,

            urls_total INTEGER DEFAULT 0,
            urls_markdown_done INTEGER DEFAULT 0,
            urls_llm_done INTEGER DEFAULT 0,

            last_error TEXT,
            done_reason TEXT,
            done_details TEXT,
            done_at TEXT,

            created_at TEXT,
            updated_at TEXT,
            last_crawled_at TEXT,

            max_pages INTEGER,

            retry_cls TEXT,
            retry_attempts INTEGER DEFAULT 0,
            retry_next_eligible_at REAL DEFAULT 0.0,
            retry_updated_at REAL DEFAULT 0.0,
            retry_last_error TEXT,
            retry_last_stage TEXT,

            retry_net_attempts INTEGER DEFAULT 0,
            retry_stall_attempts INTEGER DEFAULT 0,
            retry_mem_attempts INTEGER DEFAULT 0,
            retry_other_attempts INTEGER DEFAULT 0,

            retry_mem_hits INTEGER DEFAULT 0,
            retry_last_stall_kind TEXT,

            retry_last_progress_md_done INTEGER DEFAULT 0,
            retry_last_seen_md_done INTEGER DEFAULT 0,

            retry_last_error_sig TEXT,
            retry_same_error_streak INTEGER DEFAULT 0,
            retry_last_error_sig_updated_at REAL DEFAULT 0.0
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
            total_companies INTEGER DEFAULT 0,
            completed_companies INTEGER DEFAULT 0,
            last_company_id TEXT,
            last_updated TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_company_done_new (
            run_id TEXT NOT NULL,
            company_id TEXT NOT NULL,
            done_at TEXT,
            PRIMARY KEY (run_id, company_id)
        )
        """
    )


def _row_get(row: sqlite3.Row, key: str) -> Any:
    try:
        return row[key]
    except Exception:
        return None


def rebuild_db_to_current_schema(db_path: Path) -> None:
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")

        if not _should_rebuild_schema(conn):
            return

        have_companies = _table_exists(conn, "companies")
        have_runs = _table_exists(conn, "runs")
        have_done = _table_exists(conn, "run_company_done")

        _create_new_schema_tables(conn)

        now = now_iso()

        if have_companies:
            rows = conn.execute("SELECT * FROM companies").fetchall()
            for r in rows:
                cid = (
                    _to_str_or_none(_row_get(r, "company_id"))
                    or _to_str_or_none(_row_get(r, "bvdid"))
                    or _to_str_or_none(_row_get(r, "id"))
                )
                if not cid:
                    continue

                root_url = (
                    _to_str_or_none(_row_get(r, "root_url"))
                    or _to_str_or_none(_row_get(r, "domain_url"))
                    or _to_str_or_none(_row_get(r, "url"))
                    or ""
                )

                name = (
                    _to_str_or_none(_row_get(r, "name"))
                    or _to_str_or_none(_row_get(r, "company_name"))
                    or _to_str_or_none(_row_get(r, "company"))
                )

                metadata_json = _row_get(r, "metadata_json")
                if metadata_json is None:
                    metadata_json = _row_get(r, "metadata")
                if metadata_json is not None and not isinstance(metadata_json, str):
                    try:
                        metadata_json = json.dumps(
                            dict(metadata_json), ensure_ascii=False
                        )
                    except Exception:
                        metadata_json = None

                industry = _to_int_or_none(_row_get(r, "industry"))
                if industry is None:
                    industry = _to_int_or_none(_row_get(r, "industry_code"))

                nace = _to_int_or_none(_row_get(r, "nace"))
                industry_label = _to_str_or_none(_row_get(r, "industry_label"))
                industry_label_source = _to_str_or_none(
                    _row_get(r, "industry_label_source")
                ) or _to_str_or_none(_row_get(r, "industry_source"))

                status = _to_str_or_none(_row_get(r, "status")) or "pending"
                crawl_finished = int(
                    _to_int_or_none(_row_get(r, "crawl_finished")) or 0
                )

                urls_total = int(_to_int_or_none(_row_get(r, "urls_total")) or 0)
                urls_md = int(_to_int_or_none(_row_get(r, "urls_markdown_done")) or 0)
                urls_llm = int(_to_int_or_none(_row_get(r, "urls_llm_done")) or 0)

                last_error = _to_str_or_none(_row_get(r, "last_error"))
                done_reason = _to_str_or_none(_row_get(r, "done_reason"))
                done_details = _to_str_or_none(_row_get(r, "done_details"))
                done_at = _to_str_or_none(_row_get(r, "done_at"))

                created_at = _to_str_or_none(_row_get(r, "created_at")) or now
                updated_at = _to_str_or_none(_row_get(r, "updated_at")) or now
                last_crawled_at = _to_str_or_none(_row_get(r, "last_crawled_at"))

                max_pages = _to_int_or_none(_row_get(r, "max_pages"))

                retry_cls = _to_str_or_none(_row_get(r, "retry_cls"))
                retry_attempts = int(
                    _to_int_or_none(_row_get(r, "retry_attempts")) or 0
                )
                retry_next_eligible_at = float(
                    _to_float_or_none(_row_get(r, "retry_next_eligible_at")) or 0.0
                )
                retry_updated_at = float(
                    _to_float_or_none(_row_get(r, "retry_updated_at")) or 0.0
                )
                retry_last_error = (
                    _to_str_or_none(_row_get(r, "retry_last_error")) or ""
                )
                retry_last_stage = (
                    _to_str_or_none(_row_get(r, "retry_last_stage")) or ""
                )

                retry_net_attempts = int(
                    _to_int_or_none(_row_get(r, "retry_net_attempts")) or 0
                )
                retry_stall_attempts = int(
                    _to_int_or_none(_row_get(r, "retry_stall_attempts")) or 0
                )
                retry_mem_attempts = int(
                    _to_int_or_none(_row_get(r, "retry_mem_attempts")) or 0
                )
                retry_other_attempts = int(
                    _to_int_or_none(_row_get(r, "retry_other_attempts")) or 0
                )

                retry_mem_hits = int(
                    _to_int_or_none(_row_get(r, "retry_mem_hits")) or 0
                )
                retry_last_stall_kind = (
                    _to_str_or_none(_row_get(r, "retry_last_stall_kind")) or "unknown"
                )

                retry_last_progress_md_done = int(
                    _to_int_or_none(_row_get(r, "retry_last_progress_md_done")) or 0
                )
                retry_last_seen_md_done = int(
                    _to_int_or_none(_row_get(r, "retry_last_seen_md_done")) or 0
                )

                retry_last_error_sig = (
                    _to_str_or_none(_row_get(r, "retry_last_error_sig")) or ""
                )
                retry_same_error_streak = int(
                    _to_int_or_none(_row_get(r, "retry_same_error_streak")) or 0
                )
                retry_last_error_sig_updated_at = float(
                    _to_float_or_none(_row_get(r, "retry_last_error_sig_updated_at"))
                    or 0.0
                )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO companies_new (
                        company_id, root_url, name, metadata_json,
                        industry, nace, industry_label, industry_label_source,
                        status, crawl_finished,
                        urls_total, urls_markdown_done, urls_llm_done,
                        last_error, done_reason, done_details, done_at,
                        created_at, updated_at, last_crawled_at,
                        max_pages,
                        retry_cls, retry_attempts, retry_next_eligible_at, retry_updated_at,
                        retry_last_error, retry_last_stage,
                        retry_net_attempts, retry_stall_attempts, retry_mem_attempts, retry_other_attempts,
                        retry_mem_hits, retry_last_stall_kind,
                        retry_last_progress_md_done, retry_last_seen_md_done,
                        retry_last_error_sig, retry_same_error_streak, retry_last_error_sig_updated_at
                    )
                    VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?
                    )
                    """,
                    (
                        cid,
                        root_url,
                        name,
                        metadata_json,
                        industry,
                        nace,
                        industry_label,
                        industry_label_source,
                        status,
                        crawl_finished,
                        urls_total,
                        urls_md,
                        urls_llm,
                        last_error,
                        done_reason,
                        done_details,
                        done_at,
                        created_at,
                        updated_at,
                        last_crawled_at,
                        max_pages,
                        retry_cls,
                        retry_attempts,
                        retry_next_eligible_at,
                        retry_updated_at,
                        retry_last_error,
                        retry_last_stage,
                        retry_net_attempts,
                        retry_stall_attempts,
                        retry_mem_attempts,
                        retry_other_attempts,
                        retry_mem_hits,
                        retry_last_stall_kind,
                        retry_last_progress_md_done,
                        retry_last_seen_md_done,
                        retry_last_error_sig,
                        retry_same_error_streak,
                        retry_last_error_sig_updated_at,
                    ),
                )

        if have_runs:
            old_cols = set(_table_columns(conn, "runs"))
            rows = conn.execute("SELECT * FROM runs").fetchall()
            for r in rows:
                run_id = _to_str_or_none(_row_get(r, "run_id"))
                if not run_id:
                    continue
                last_company_id = _to_str_or_none(_row_get(r, "last_company_id"))
                if last_company_id is None and "last_company_bvdid" in old_cols:
                    last_company_id = _to_str_or_none(_row_get(r, "last_company_bvdid"))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO runs_new (
                        run_id, pipeline, version, args_hash,
                        crawl4ai_cache_base_dir, crawl4ai_cache_mode,
                        started_at, total_companies, completed_companies,
                        last_company_id, last_updated
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        _to_str_or_none(_row_get(r, "pipeline")),
                        _to_str_or_none(_row_get(r, "version")),
                        _to_str_or_none(_row_get(r, "args_hash")),
                        _to_str_or_none(_row_get(r, "crawl4ai_cache_base_dir")),
                        _to_str_or_none(_row_get(r, "crawl4ai_cache_mode")),
                        _to_str_or_none(_row_get(r, "started_at")),
                        int(_to_int_or_none(_row_get(r, "total_companies")) or 0),
                        int(_to_int_or_none(_row_get(r, "completed_companies")) or 0),
                        last_company_id,
                        _to_str_or_none(_row_get(r, "last_updated")),
                    ),
                )

        if have_done:
            old_cols = set(_table_columns(conn, "run_company_done"))
            rows = conn.execute("SELECT * FROM run_company_done").fetchall()
            for r in rows:
                run_id = _to_str_or_none(_row_get(r, "run_id"))
                if not run_id:
                    continue
                cid = _to_str_or_none(_row_get(r, "company_id"))
                if cid is None and "bvdid" in old_cols:
                    cid = _to_str_or_none(_row_get(r, "bvdid"))
                if not cid:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO run_company_done_new (run_id, company_id, done_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        run_id,
                        cid,
                        _to_str_or_none(_row_get(r, "done_at")),
                    ),
                )

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
