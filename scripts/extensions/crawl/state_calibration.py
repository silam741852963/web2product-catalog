from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from configs.models import Company, UrlIndexEntry, UrlIndexMeta, URL_INDEX_META_KEY
from extensions.crawl.state import CrawlState
from extensions.io.load_source import (
    DEFAULT_INDUSTRY_FALLBACK_PATH,
    DEFAULT_NACE_INDUSTRY_PATH,
    IndustryEnrichmentConfig,
    load_companies_from_source_with_industry,
)
from extensions.io.output_paths import ensure_output_root, sanitize_bvdid
from extensions.utils.versioning import safe_version_metadata

_VERSION_META_KEY = "version_metadata"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    company_id: str
    db_snapshot: Company
    crawl_meta: Dict[str, Any]
    url_index_meta: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    out_dir: str
    db_path: str
    touched_companies: int
    wrote_global_state: bool
    source_companies_loaded: int
    source_companies_used: int
    sample_before: CalibrationSample
    sample_after: CalibrationSample


@dataclass(frozen=True, slots=True)
class CorruptJsonFile:
    company_id: str
    path: str
    size_bytes: int
    reason: str
    head_bytes_hex: str
    head_text_preview: str


@dataclass(frozen=True, slots=True)
class CorruptionReport:
    out_dir: str
    db_path: str
    scanned_companies: int
    affected_companies: int
    affected_files: int
    examples: List[CorruptJsonFile]


@dataclass(frozen=True, slots=True)
class CorruptionFixReport:
    out_dir: str
    db_path: str
    dry_run: bool
    scanned_companies: int
    affected_companies: int
    quarantined_files: int
    marked_pending: int
    run_done_unmarked: int
    examples: List[CorruptJsonFile]


# -----------------------------------------------------------------------------
# Output-path resolution
# -----------------------------------------------------------------------------


def _company_base(out_dir: Path, company_id: str) -> Path:
    return out_dir / sanitize_bvdid(company_id)


def _crawl_meta_candidates(out_dir: Path, company_id: str) -> List[Path]:
    base = _company_base(out_dir, company_id)
    return [
        base / "meta" / "crawl_meta.json",
        base / "metadata" / "crawl_meta.json",
    ]


def _url_index_candidates(out_dir: Path, company_id: str) -> List[Path]:
    base = _company_base(out_dir, company_id)
    return [
        base / "url_index.json",
        base / "metadata" / "url_index.json",
    ]


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


# -----------------------------------------------------------------------------
# JSON helpers (robust corruption detection)
# -----------------------------------------------------------------------------


def _read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except Exception:
        return None


def _bytes_head_hex(b: bytes, n: int = 80) -> str:
    h = b[:n]
    return h.hex()


def _bytes_text_preview(b: bytes, n: int = 200) -> str:
    # best-effort preview
    try:
        s = b[:n].decode("utf-8", errors="replace")
    except Exception:
        return ""
    # keep control chars visible-ish
    return s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _is_probably_binary_nul(b: bytes) -> bool:
    # Your observed failure: starts with NULs
    if not b:
        return False
    # consider "corrupt" if it starts with many NULs OR contains NUL before any printable JSON char
    if b[:16] == b"\x00" * 16:
        return True
    # also if there are NUL bytes very early
    early = b[:256]
    if b"\x00" in early:
        # if the first non-whitespace is NUL, that's not JSON text
        for ch in early:
            if ch in (9, 10, 13, 32):  # whitespace
                continue
            return ch == 0
        return True
    return False


def _validate_json_bytes(path: Path) -> Tuple[bool, str]:
    """
    Returns (ok, reason_if_bad).
    """
    b = _read_bytes(path)
    if b is None:
        return False, "unreadable"
    if len(b) == 0:
        return False, "empty"
    if _is_probably_binary_nul(b):
        return False, "nul_bytes_prefix_or_early"
    try:
        txt = b.decode("utf-8")
    except Exception:
        return False, "non_utf8_bytes"
    if txt.strip() == "":
        return False, "whitespace_only"
    try:
        obj = json.loads(txt)
    except Exception:
        return False, "json_parse_error"
    if not isinstance(obj, dict):
        return False, "json_not_dict"
    return True, ""


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    # legacy helper used by sampling/calibration; keep permissive
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        obj = json.loads(txt)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _write_json_file(path: Path, obj: Mapping[str, Any], *, pretty: bool) -> None:
    # Intentionally only touches existing files (calibration should not invent new files)
    if not path.exists():
        return
    if pretty:
        data = json.dumps(obj, ensure_ascii=False, indent=2)
    else:
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(data, encoding="utf-8")


# -----------------------------------------------------------------------------
# URL index normalization (reads/writes existing url_index.json only)
# -----------------------------------------------------------------------------


def _normalize_url_index_file(
    out_dir: Path,
    company_id: str,
    *,
    version_meta: Dict[str, Any],
) -> None:
    p = _first_existing(_url_index_candidates(out_dir, company_id))
    if p is None:
        return

    # If it's corrupted, do not touch it here.
    ok, _ = _validate_json_bytes(p)
    if not ok:
        return

    idx = _read_json_file(p)
    if not isinstance(idx, dict) or not idx:
        return

    out: Dict[str, Any] = {}
    for k, raw in idx.items():
        if str(k).startswith("__"):
            continue
        url = str(k)
        ent = raw if isinstance(raw, Mapping) else {}
        ent_dict = dict(ent)
        ent_dict.setdefault("company_id", company_id)
        ent_dict.setdefault("url", url)
        normalized = UrlIndexEntry.from_dict(ent_dict, company_id=company_id, url=url)
        out[url] = normalized.to_dict()

    meta_raw = idx.get(URL_INDEX_META_KEY)
    meta_dict = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
    meta_dict.setdefault("company_id", company_id)
    meta_dict.setdefault("created_at", _now_iso())
    meta_dict["updated_at"] = _now_iso()
    meta_dict[_VERSION_META_KEY] = version_meta

    meta_norm = UrlIndexMeta.from_dict(meta_dict, company_id=company_id)
    out[URL_INDEX_META_KEY] = meta_norm.to_dict()

    _write_json_file(p, out, pretty=False)


# -----------------------------------------------------------------------------
# crawl_meta.json canonical rewrite (allowlist; removes legacy keys)
# -----------------------------------------------------------------------------


def _patch_crawl_meta_file(
    out_dir: Path,
    company_id: str,
    *,
    src: Optional[Company],
    db_snap: Company,
    version_meta: Dict[str, Any],
) -> None:
    p = _first_existing(_crawl_meta_candidates(out_dir, company_id))
    if p is None:
        return

    meta: Dict[str, Any] = {}

    meta["company_id"] = db_snap.company_id
    meta["root_url"] = db_snap.root_url
    meta["name"] = db_snap.name

    md = db_snap.metadata if isinstance(db_snap.metadata, dict) else {}
    meta["metadata"] = md

    meta["industry"] = db_snap.industry
    meta["nace"] = db_snap.nace
    meta["industry_label"] = db_snap.industry_label
    meta["industry_label_source"] = db_snap.industry_label_source

    meta["status"] = db_snap.status
    meta["crawl_finished"] = bool(db_snap.crawl_finished)

    meta["urls_total"] = int(db_snap.urls_total or 0)
    meta["urls_markdown_done"] = int(db_snap.urls_markdown_done or 0)
    meta["urls_llm_done"] = int(db_snap.urls_llm_done or 0)

    meta["created_at"] = db_snap.created_at
    meta["updated_at"] = db_snap.updated_at
    meta["last_crawled_at"] = db_snap.last_crawled_at

    meta["retry_cls"] = db_snap.retry_cls
    meta["retry_attempts"] = int(db_snap.retry_attempts or 0)
    meta["retry_next_eligible_at"] = float(db_snap.retry_next_eligible_at or 0.0)
    meta["retry_updated_at"] = float(db_snap.retry_updated_at or 0.0)
    meta["retry_last_error"] = db_snap.retry_last_error or ""
    meta["retry_last_stage"] = db_snap.retry_last_stage or ""

    meta["retry_net_attempts"] = int(db_snap.retry_net_attempts or 0)
    meta["retry_stall_attempts"] = int(db_snap.retry_stall_attempts or 0)
    meta["retry_mem_attempts"] = int(db_snap.retry_mem_attempts or 0)
    meta["retry_other_attempts"] = int(db_snap.retry_other_attempts or 0)

    meta["retry_mem_hits"] = int(db_snap.retry_mem_hits or 0)
    meta["retry_last_stall_kind"] = db_snap.retry_last_stall_kind or "unknown"

    meta["retry_last_progress_md_done"] = int(db_snap.retry_last_progress_md_done or 0)
    meta["retry_last_seen_md_done"] = int(db_snap.retry_last_seen_md_done or 0)

    meta["retry_last_error_sig"] = db_snap.retry_last_error_sig or ""
    meta["retry_same_error_streak"] = int(db_snap.retry_same_error_streak or 0)
    meta["retry_last_error_sig_updated_at"] = float(
        db_snap.retry_last_error_sig_updated_at or 0.0
    )

    meta[_VERSION_META_KEY] = version_meta
    meta["max_pages"] = db_snap.max_pages

    _write_json_file(p, meta, pretty=True)


# -----------------------------------------------------------------------------
# DB schema migration (kept as-is from your file)
# -----------------------------------------------------------------------------


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


def _rebuild_db_to_current_schema(db_path: Path) -> None:
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

        now = _now_iso()

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


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


def _pick_sample_company_id(state: CrawlState, requested: Optional[str]) -> str:
    if requested is not None and requested.strip():
        return requested.strip()

    row = (
        sqlite3.connect(str(state.db_path))
        .execute("SELECT company_id FROM companies ORDER BY company_id ASC LIMIT 1")
        .fetchone()
    )
    if row is None:
        raise RuntimeError("No companies found in DB to sample.")
    return str(row[0])


async def _sample(
    out_dir: Path, state: CrawlState, company_id: str
) -> CalibrationSample:
    snap = await state.get_company_snapshot(company_id, recompute=False)

    cm_path = _first_existing(_crawl_meta_candidates(out_dir, company_id))
    cm = _read_json_file(cm_path) if cm_path else None

    idx_path = _first_existing(_url_index_candidates(out_dir, company_id))
    idx = _read_json_file(idx_path) if idx_path else None

    idx_meta: Dict[str, Any] = {}
    if isinstance(idx, dict):
        raw_meta = idx.get(URL_INDEX_META_KEY)
        if isinstance(raw_meta, dict):
            idx_meta = raw_meta

    return CalibrationSample(
        company_id=company_id,
        db_snapshot=snap,
        crawl_meta=(cm if isinstance(cm, dict) else {}),
        url_index_meta=(idx_meta if isinstance(idx_meta, dict) else {}),
    )


# -----------------------------------------------------------------------------
# Source enrichment (LOAD ONLY; NEVER inserts into DB)
# -----------------------------------------------------------------------------


def _industry_cfg(
    *,
    industry_nace_path: Optional[Path],
    industry_fallback_path: Optional[Path],
) -> IndustryEnrichmentConfig:
    return IndustryEnrichmentConfig(
        nace_path=industry_nace_path
        if industry_nace_path is not None
        else DEFAULT_NACE_INDUSTRY_PATH,
        fallback_path=industry_fallback_path
        if industry_fallback_path is not None
        else DEFAULT_INDUSTRY_FALLBACK_PATH,
        enabled=True,
    )


def _load_source_company_map(
    *,
    dataset_file: Optional[Path],
    company_file: Optional[Path],
    industry_nace_path: Optional[Path],
    industry_fallback_path: Optional[Path],
) -> Tuple[Dict[str, Company], int]:
    if dataset_file is None and company_file is None:
        return {}, 0

    cfg = _industry_cfg(
        industry_nace_path=industry_nace_path,
        industry_fallback_path=industry_fallback_path,
    )

    paths: List[Path] = []
    if dataset_file is not None:
        paths.append(Path(dataset_file))
    if company_file is not None:
        paths.append(Path(company_file))

    all_companies: List[Company] = []
    for p in paths:
        all_companies.extend(
            load_companies_from_source_with_industry(
                p,
                industry_config=cfg,
                encoding="utf-8",
                limit=None,
                aggregate_same_url=True,
                interleave_domains=True,
            )
        )

    out: Dict[str, Company] = {}
    for c in all_companies:
        out[c.company_id] = c.normalized()

    return out, len(all_companies)


def _filter_source_map_to_db(
    *, db_company_ids: List[str], src_map: Mapping[str, Company]
) -> Dict[str, Company]:
    if not src_map:
        return {}
    have = set(db_company_ids)
    return {cid: src_map[cid] for cid in have if cid in src_map}


# -----------------------------------------------------------------------------
# Corruption scan & fix
# -----------------------------------------------------------------------------


def _collect_corrupt_for_company(
    out_dir: Path, company_id: str
) -> List[CorruptJsonFile]:
    out: List[CorruptJsonFile] = []
    for p in _url_index_candidates(out_dir, company_id):
        if not p.exists() or not p.is_file():
            continue
        ok, reason = _validate_json_bytes(p)
        if ok:
            continue
        b = _read_bytes(p) or b""
        out.append(
            CorruptJsonFile(
                company_id=company_id,
                path=str(p),
                size_bytes=int(len(b)),
                reason=reason,
                head_bytes_hex=_bytes_head_hex(b, 80),
                head_text_preview=_bytes_text_preview(b, 200),
            )
        )
    return out


async def scan_corrupt_url_indexes_async(
    *,
    out_dir: Path,
    db_path: Path,
    max_examples: int = 1,
) -> CorruptionReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    _rebuild_db_to_current_schema(actual_db_path)

    state = CrawlState(db_path=actual_db_path)
    try:
        rows = await state._query_all("SELECT company_id FROM companies", tuple())
        db_ids = [str(r["company_id"]) for r in rows]

        sem = asyncio.Semaphore(64)
        examples: List[CorruptJsonFile] = []
        affected_company_set: set[str] = set()
        affected_files = 0

        async def _one(cid: str) -> None:
            nonlocal affected_files
            async with sem:
                files = await asyncio.to_thread(
                    _collect_corrupt_for_company, out_dir, cid
                )
                if files:
                    affected_company_set.add(cid)
                    affected_files += len(files)
                    # Keep a few examples for review
                    if len(examples) < max_examples:
                        examples.extend(files[: max_examples - len(examples)])

        batch = 512
        for i in range(0, len(db_ids), batch):
            await asyncio.gather(*(_one(cid) for cid in db_ids[i : i + batch]))

        return CorruptionReport(
            out_dir=str(out_dir),
            db_path=str(actual_db_path),
            scanned_companies=len(db_ids),
            affected_companies=len(affected_company_set),
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
) -> CorruptionReport:
    return asyncio.run(
        scan_corrupt_url_indexes_async(
            out_dir=out_dir,
            db_path=db_path,
            max_examples=max_examples,
        )
    )


def _quarantine_file(path: Path) -> bool:
    """
    Rename corrupted file aside: url_index.json -> url_index.json.corrupt.<ts>.<pid>
    """
    if not path.exists():
        return False
    ts = int(time.time())
    suffix = f".corrupt.{ts}.{os.getpid()}"
    dst = Path(str(path) + suffix)
    try:
        path.rename(dst)
        return True
    except Exception:
        return False


async def _unmark_latest_run_done(state: CrawlState, company_id: str) -> int:
    """
    Remove (latest_run_id, company_id) from run_company_done so the scheduler won't skip it.
    Returns number of rows deleted (0/1).
    """
    row = await state._query_one(
        "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1",
        tuple(),
    )
    if row is None:
        return 0
    rid = str(row["run_id"])
    # execute delete
    await state._exec(
        "DELETE FROM run_company_done WHERE run_id=? AND company_id=?",
        (rid, company_id),
    )
    # sqlite changes() is per-connection; easiest is to re-check existence
    row2 = await state._query_one(
        "SELECT 1 AS x FROM run_company_done WHERE run_id=? AND company_id=? LIMIT 1",
        (rid, company_id),
    )
    return 0 if row2 is not None else 1


async def fix_corrupt_url_indexes_async(
    *,
    out_dir: Path,
    db_path: Path,
    max_examples: int = 1,
    mark_pending: bool = True,
    quarantine: bool = True,
    unmark_run_done: bool = True,
    dry_run: bool = False,
) -> CorruptionFixReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    _rebuild_db_to_current_schema(actual_db_path)

    state = CrawlState(db_path=actual_db_path)
    try:
        rows = await state._query_all("SELECT company_id FROM companies", tuple())
        db_ids = [str(r["company_id"]) for r in rows]

        sem = asyncio.Semaphore(64)
        examples: List[CorruptJsonFile] = []
        affected_company_set: set[str] = set()
        quarantined = 0
        marked = 0
        unmarked = 0

        async def _one(cid: str) -> None:
            nonlocal quarantined, marked, unmarked
            async with sem:
                files = await asyncio.to_thread(
                    _collect_corrupt_for_company, out_dir, cid
                )
                if not files:
                    return

                affected_company_set.add(cid)

                # keep examples
                if len(examples) < max_examples:
                    examples.extend(files[: max_examples - len(examples)])

                # quarantine corrupted files
                if quarantine:
                    for f in files:
                        p = Path(f.path)
                        if dry_run:
                            quarantined += 1
                        else:
                            ok = await asyncio.to_thread(_quarantine_file, p)
                            if ok:
                                quarantined += 1

                # mark pending in DB
                if mark_pending:
                    if not dry_run:
                        # Reset progress so the crawler redoes it from scratch.
                        # Note: leave retry counters intact; they help throttling.
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
                    marked += 1

                # remove run_done for latest run (optional)
                if unmark_run_done:
                    if dry_run:
                        unmarked += 1
                    else:
                        unmarked += await _unmark_latest_run_done(state, cid)

        batch = 512
        for i in range(0, len(db_ids), batch):
            await asyncio.gather(*(_one(cid) for cid in db_ids[i : i + batch]))

        return CorruptionFixReport(
            out_dir=str(out_dir),
            db_path=str(actual_db_path),
            dry_run=bool(dry_run),
            scanned_companies=len(db_ids),
            affected_companies=len(affected_company_set),
            quarantined_files=int(quarantined),
            marked_pending=int(marked),
            run_done_unmarked=int(unmarked),
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
        )
    )


# -----------------------------------------------------------------------------
# Public API (existing calibrate/check, unchanged behavior unless you call corrupt funcs)
# -----------------------------------------------------------------------------


async def calibrate_async(
    *,
    out_dir: Path,
    db_path: Path,
    sample_company_id: Optional[str] = None,
    write_global_state: bool = True,
    concurrency: int = 32,
    dataset_file: Optional[Path] = None,
    company_file: Optional[Path] = None,
    industry_nace_path: Optional[Path] = None,
    industry_fallback_path: Optional[Path] = None,
) -> CalibrationReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()

    _rebuild_db_to_current_schema(actual_db_path)

    version_meta = safe_version_metadata(
        component="state_calibration", start_path=Path(__file__)
    )
    if not isinstance(version_meta, dict):
        version_meta = {
            "component": "state_calibration",
            "available": False,
            "reason": "unavailable",
        }

    state = CrawlState(db_path=actual_db_path)
    try:
        rows = await state._query_all("SELECT company_id FROM companies", tuple())
        db_ids = [str(r["company_id"]) for r in rows]

        src_map, src_loaded_rows = _load_source_company_map(
            dataset_file=dataset_file,
            company_file=company_file,
            industry_nace_path=industry_nace_path,
            industry_fallback_path=industry_fallback_path,
        )
        src_map = _filter_source_map_to_db(db_company_ids=db_ids, src_map=src_map)
        src_used = int(len(src_map))

        company_id = _pick_sample_company_id(state, sample_company_id)
        sample_before = await _sample(out_dir, state, company_id)

        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def _one(cid: str) -> None:
            async with sem:
                src = src_map.get(cid)

                if src is not None:
                    await state.upsert_company(
                        cid,
                        root_url=src.root_url,
                        name=src.name,
                        metadata={},
                        industry=src.industry,
                        nace=src.nace,
                        industry_label=src.industry_label,
                        industry_label_source=src.industry_label_source,
                    )

                db_snap = await state.get_company_snapshot(cid, recompute=False)

                await asyncio.to_thread(
                    _normalize_url_index_file, out_dir, cid, version_meta=version_meta
                )
                await asyncio.to_thread(
                    _patch_crawl_meta_file,
                    out_dir,
                    cid,
                    src=src,
                    db_snap=db_snap,
                    version_meta=version_meta,
                )

        batch = max(64, int(concurrency) * 8)
        for i in range(0, len(db_ids), batch):
            await asyncio.gather(*(_one(cid) for cid in db_ids[i : i + batch]))

        if write_global_state:
            await state.write_global_state_from_db_only(pretty=False)

        sample_after = await _sample(out_dir, state, company_id)

        return CalibrationReport(
            out_dir=str(out_dir),
            db_path=str(state.db_path),
            touched_companies=int(len(db_ids)),
            wrote_global_state=bool(write_global_state),
            source_companies_loaded=int(src_loaded_rows),
            source_companies_used=int(src_used),
            sample_before=sample_before,
            sample_after=sample_after,
        )
    finally:
        state.close()


async def check_async(
    *, out_dir: Path, db_path: Path, sample_company_id: Optional[str] = None
) -> CalibrationSample:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    _rebuild_db_to_current_schema(actual_db_path)

    state = CrawlState(db_path=actual_db_path)
    try:
        cid = _pick_sample_company_id(state, sample_company_id)
        return await _sample(out_dir, state, cid)
    finally:
        state.close()


def calibrate(
    *,
    out_dir: Path,
    db_path: Path,
    sample_company_id: Optional[str] = None,
    write_global_state: bool = True,
    concurrency: int = 32,
    dataset_file: Optional[Path] = None,
    company_file: Optional[Path] = None,
    industry_nace_path: Optional[Path] = None,
    industry_fallback_path: Optional[Path] = None,
) -> CalibrationReport:
    return asyncio.run(
        calibrate_async(
            out_dir=out_dir,
            db_path=db_path,
            sample_company_id=sample_company_id,
            write_global_state=write_global_state,
            concurrency=concurrency,
            dataset_file=dataset_file,
            company_file=company_file,
            industry_nace_path=industry_nace_path,
            industry_fallback_path=industry_fallback_path,
        )
    )


def check(
    *, out_dir: Path, db_path: Path, sample_company_id: Optional[str] = None
) -> CalibrationSample:
    return asyncio.run(
        check_async(
            out_dir=out_dir, db_path=db_path, sample_company_id=sample_company_id
        )
    )
