from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from configs.models import (
    Company,
    CompanyStatus,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_TERMINAL_DONE,
    URL_INDEX_META_KEY,
    UrlIndexEntry,
    UrlIndexEntryStatus,
    UrlIndexMeta,
)

from extensions.io import output_paths
from extensions.io.output_paths import ensure_company_dirs, sanitize_bvdid
from extensions.utils.versioning import safe_version_metadata

META_NAME = "crawl_meta.json"
URL_INDEX_NAME = "url_index.json"
GLOBAL_STATE_NAME = "crawl_global_state.json"

URL_INDEX_RESERVED_PREFIX = "__"

_VERSION_META_KEY = "version_metadata"


_STATUS_RANK: Dict[str, int] = {
    COMPANY_STATUS_PENDING: 0,
    COMPANY_STATUS_MD_NOT_DONE: 1,
    COMPANY_STATUS_MD_DONE: 2,
    COMPANY_STATUS_LLM_NOT_DONE: 3,
    COMPANY_STATUS_LLM_DONE: 4,
    COMPANY_STATUS_TERMINAL_DONE: 2,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_root() -> Path:
    root = output_paths.get_output_root()
    return Path(root).resolve()


def default_db_path() -> Path:
    return _output_root() / "crawl_state.sqlite3"


class _DefaultDBPathProxy(os.PathLike):
    def __fspath__(self) -> str:
        return os.fspath(default_db_path())

    def __str__(self) -> str:
        return str(default_db_path())

    def __repr__(self) -> str:
        return f"_DefaultDBPathProxy({default_db_path()!r})"

    @property
    def path(self) -> Path:
        return default_db_path()


DEFAULT_DB = _DefaultDBPathProxy()

_FILE_LOCKS_LOCK = threading.Lock()
_FILE_LOCKS: Dict[Path, threading.Lock] = {}


def _get_file_lock(path: Path) -> threading.Lock:
    with _FILE_LOCKS_LOCK:
        lk = _FILE_LOCKS.get(path)
        if lk is None:
            lk = threading.Lock()
            _FILE_LOCKS[path] = lk
        return lk


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    tmp = Path(f"{str(path)}.tmp.{stamp}")
    try:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with suppress(Exception):
            if tmp.exists() and tmp != path:
                tmp.unlink()


def _json_load(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    if raw.strip() == "":
        raise ValueError(f"Empty/whitespace JSON file: {path} (len={len(raw)})")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {path} (len={len(raw)}; head={raw[:8000]!r})"
        ) from e


def _json_dumps(obj: Any, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_json(path: Path, obj: Any, *, pretty: bool) -> None:
    _atomic_write_text(path, _json_dumps(obj, pretty=pretty), "utf-8")


def _meta_path_for(company_id: str) -> Path:
    # writes go to sanitized location
    return _company_metadata_dir_write(company_id) / META_NAME


def _url_index_path_for(company_id: str) -> Path:
    # writes go to sanitized location
    return _company_metadata_dir_write(company_id) / URL_INDEX_NAME


def _global_state_path() -> Path:
    return _output_root() / GLOBAL_STATE_NAME


def _company_metadata_dir_write(company_id: str) -> Path:
    """
    Always write into the sanitized/company_dirs location (Windows-safe).
    """
    dirs = ensure_company_dirs(company_id)
    meta = dirs.get("metadata") or dirs.get("checkpoints")
    if meta is None:
        safe = sanitize_bvdid(company_id)
        p = _output_root() / safe / "metadata"
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path(meta)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _company_metadata_dir_legacy(company_id: str) -> Path:
    """
    Legacy path that may exist if someone constructed paths using raw company_id.
    We DO NOT write here, but we may read/migrate from it.
    """
    # IMPORTANT: do not sanitize here
    return _output_root() / str(company_id) / "metadata"


def _resolve_company_meta_file_for_read(company_id: str, filename: str) -> Path:
    """
    Read-path resolution:
      1) Prefer sanitized location.
      2) If missing, fall back to legacy raw path (when it exists).
    """
    p_new = _company_metadata_dir_write(company_id) / filename
    if p_new.exists():
        return p_new

    p_old = _company_metadata_dir_legacy(company_id) / filename
    if p_old.exists():
        return p_old

    return p_new


def _maybe_migrate_legacy_meta_file(company_id: str, filename: str) -> None:
    """
    If a legacy raw-path file exists but the sanitized file doesn't,
    copy it to the sanitized location (best-effort, atomic).
    """
    p_new = _company_metadata_dir_write(company_id) / filename
    if p_new.exists():
        return

    p_old = _company_metadata_dir_legacy(company_id) / filename
    if not p_old.exists():
        return

    # best-effort migration
    with suppress(Exception):
        raw = p_old.read_text(encoding="utf-8")
        _atomic_write_text(p_new, raw, encoding="utf-8")


_VERSION_META_LOCK = threading.Lock()
_VERSION_META_CACHED: Optional[Dict[str, object]] = None
_VERSION_META_CACHED_AT: float = 0.0
_VERSION_META_TTL_SEC = float(os.getenv("CRAWLSTATE_VERSION_META_TTL_SEC", "30"))


def _version_metadata() -> Dict[str, object]:
    global _VERSION_META_CACHED, _VERSION_META_CACHED_AT

    now = time.time()
    with _VERSION_META_LOCK:
        if (
            _VERSION_META_CACHED is not None
            and (now - _VERSION_META_CACHED_AT) < _VERSION_META_TTL_SEC
        ):
            return dict(_VERSION_META_CACHED)

    md = safe_version_metadata(component="crawl_state", start_path=Path(__file__))
    if not isinstance(md, dict):
        md = {
            "component": "crawl_state",
            "available": False,
            "reason": "git metadata unavailable",
        }
    else:
        md = dict(md)
        md["available"] = True

    with _VERSION_META_LOCK:
        _VERSION_META_CACHED = dict(md)
        _VERSION_META_CACHED_AT = now

    return md


def _normalize_status(st: Optional[str]) -> CompanyStatus:
    s = (st or "").strip()
    if s in _STATUS_RANK:
        return s  # type: ignore[return-value]
    return COMPANY_STATUS_PENDING


def _prefer_higher_status(cur: Optional[str], derived: Optional[str]) -> CompanyStatus:
    c = _normalize_status(cur)
    d = _normalize_status(derived)
    return c if _STATUS_RANK.get(c, 0) >= _STATUS_RANK.get(d, 0) else d


def load_crawl_meta(company_id: str) -> Dict[str, Any]:
    p = _resolve_company_meta_file_for_read(company_id, META_NAME)
    if not p.exists():
        return {}
    obj = _json_load(p)
    return obj if isinstance(obj, dict) else {}


def patch_crawl_meta(
    company_id: str, patch: Dict[str, Any], *, pretty: bool = True
) -> None:
    # If legacy exists, migrate once so subsequent reads/writes are consistent.
    _maybe_migrate_legacy_meta_file(company_id, META_NAME)

    p = _meta_path_for(company_id)  # sanitized write path
    lk = _get_file_lock(p)

    with lk:
        base: Dict[str, Any] = {}
        if p.exists():
            loaded = _json_load(p)
            if isinstance(loaded, dict):
                base = loaded

        if isinstance(patch, dict):
            base.update(patch)

        if not str(base.get("company_id") or "").strip():
            base["company_id"] = company_id

        if not str(base.get("created_at") or "").strip():
            base["created_at"] = _now_iso()

        base[_VERSION_META_KEY] = _version_metadata()
        base["updated_at"] = _now_iso()

        _atomic_write_json(p, base, pretty=pretty)


def patch_company_meta(
    company_id: str, patch: Dict[str, Any], *, pretty: bool = True
) -> None:
    # keep behavior identical, just ensure we don't lose legacy meta
    _maybe_migrate_legacy_meta_file(company_id, META_NAME)
    patch_crawl_meta(company_id, patch, pretty=pretty)


def _company_to_meta_dict(
    company: Company,
    *,
    company_ctx: Optional[Dict[str, Any]] = None,
    set_last_crawled_at: bool = True,
) -> Dict[str, Any]:
    c = company.normalized()
    now = _now_iso()

    payload: Dict[str, Any] = {
        "company_id": c.company_id,
        "root_url": c.root_url,
        "name": c.name,
        "metadata": dict(c.metadata) if isinstance(c.metadata, dict) else {},
        "industry": c.industry,
        "nace": c.nace,
        "industry_label": c.industry_label,
        "industry_label_source": c.industry_label_source,
        "status": c.status,
        "crawl_finished": bool(c.crawl_finished),
        "urls_total": int(c.urls_total),
        "urls_markdown_done": int(c.urls_markdown_done),
        "urls_llm_done": int(c.urls_llm_done),
        "last_error": c.last_error,
        "done_reason": c.done_reason,
        "done_details": c.done_details if isinstance(c.done_details, dict) else None,
        "done_at": c.done_at,
        "created_at": c.created_at,
        "updated_at": now,
        "last_crawled_at": (now if set_last_crawled_at else c.last_crawled_at),
        "max_pages": c.max_pages,
        "retry_cls": c.retry_cls,
        "retry_attempts": int(c.retry_attempts),
        "retry_next_eligible_at": float(c.retry_next_eligible_at),
        "retry_updated_at": float(c.retry_updated_at),
        "retry_last_error": c.retry_last_error,
        "retry_last_stage": c.retry_last_stage,
        "retry_net_attempts": int(c.retry_net_attempts),
        "retry_stall_attempts": int(c.retry_stall_attempts),
        "retry_mem_attempts": int(c.retry_mem_attempts),
        "retry_other_attempts": int(c.retry_other_attempts),
        "retry_mem_hits": int(c.retry_mem_hits),
        "retry_last_stall_kind": c.retry_last_stall_kind,
        "retry_last_progress_md_done": int(c.retry_last_progress_md_done),
        "retry_last_seen_md_done": int(c.retry_last_seen_md_done),
        "retry_last_error_sig": c.retry_last_error_sig,
        "retry_same_error_streak": int(c.retry_same_error_streak),
        "retry_last_error_sig_updated_at": float(c.retry_last_error_sig_updated_at),
        _VERSION_META_KEY: _version_metadata(),
    }

    if isinstance(company_ctx, dict) and company_ctx:
        for k, v in company_ctx.items():
            if k in (_VERSION_META_KEY, "updated_at"):
                continue
            payload[k] = v

    return {k: v for k, v in payload.items() if v is not None}


def _write_company_meta_snapshot_sync(
    company_id: str,
    company: Company,
    *,
    pretty: bool = True,
    company_ctx: Optional[Dict[str, Any]] = None,
    set_last_crawled_at: bool = True,
) -> Dict[str, Any]:
    p = _meta_path_for(company_id)
    lk = _get_file_lock(p)

    with lk:
        base: Dict[str, Any] = {}
        if p.exists():
            loaded = _json_load(p)
            if isinstance(loaded, dict):
                base = loaded

        patch = _company_to_meta_dict(
            company, company_ctx=company_ctx, set_last_crawled_at=set_last_crawled_at
        )
        base.update(patch)

        if not str(base.get("company_id") or "").strip():
            base["company_id"] = company_id
        if not str(base.get("created_at") or "").strip():
            base["created_at"] = _now_iso()

        base[_VERSION_META_KEY] = _version_metadata()
        base["updated_at"] = _now_iso()

        _atomic_write_json(p, base, pretty=pretty)
        return base


def load_url_index(company_id: str) -> Dict[str, Any]:
    p = _resolve_company_meta_file_for_read(company_id, URL_INDEX_NAME)
    if not p.exists():
        return {}
    obj = _json_load(p)
    return obj if isinstance(obj, dict) else {}


def upsert_url_index_entry(company_id: str, url: str, patch: Dict[str, Any]) -> None:
    # If legacy exists, migrate once so we stop "missing index" due to raw-path folders.
    _maybe_migrate_legacy_meta_file(company_id, URL_INDEX_NAME)

    idx_path = _url_index_path_for(company_id)  # sanitized write path
    lk = _get_file_lock(idx_path)

    with lk:
        existing: Dict[str, Any] = {}
        if idx_path.exists():
            loaded = _json_load(idx_path)
            if isinstance(loaded, dict):
                existing = loaded

        ent = existing.get(url)
        ent_dict: Dict[str, Any] = dict(ent) if isinstance(ent, dict) else {}

        ent_dict.setdefault("company_id", company_id)
        ent_dict.setdefault("url", url)
        ent_dict.setdefault("created_at", _now_iso())

        if isinstance(patch, dict):
            patch_dict = {k: v for k, v in patch.items() if v is not None}

            patch_dict.setdefault("company_id", company_id)
            patch_dict.setdefault("url", url)
            patch_dict.setdefault(
                "created_at", ent_dict.get("created_at") or _now_iso()
            )

            if "updated_at" not in patch_dict:
                patch_dict["updated_at"] = _now_iso()

            ent_dict.update(patch_dict)

        normalized = UrlIndexEntry.from_dict(ent_dict, company_id=company_id, url=url)
        existing[url] = normalized.to_dict()

        _atomic_write_text(
            idx_path,
            json.dumps(existing, ensure_ascii=False, separators=(",", ":")),
            "utf-8",
        )


def patch_url_index_meta(company_id: str, patch: Dict[str, Any]) -> None:
    # If legacy exists, migrate once so we stop "missing index" due to raw-path folders.
    _maybe_migrate_legacy_meta_file(company_id, URL_INDEX_NAME)

    idx_path = _url_index_path_for(company_id)  # sanitized write path
    lk = _get_file_lock(idx_path)

    with lk:
        existing: Dict[str, Any] = {}
        if idx_path.exists():
            loaded = _json_load(idx_path)
            if isinstance(loaded, dict):
                existing = loaded

        meta = existing.get(URL_INDEX_META_KEY)
        meta_dict: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}

        meta_dict.setdefault("company_id", company_id)
        meta_dict.setdefault("created_at", _now_iso())

        if isinstance(patch, dict):
            patch_dict = {k: v for k, v in patch.items() if v is not None}
            patch_dict.setdefault("company_id", company_id)
            patch_dict.setdefault(
                "created_at", meta_dict.get("created_at") or _now_iso()
            )

            if "updated_at" not in patch_dict:
                patch_dict["updated_at"] = _now_iso()

            meta_dict.update(patch_dict)

        normalized = UrlIndexMeta.from_dict(meta_dict, company_id=company_id)
        existing[URL_INDEX_META_KEY] = normalized.to_dict()

        _atomic_write_text(
            idx_path,
            json.dumps(existing, ensure_ascii=False, separators=(",", ":")),
            "utf-8",
        )


def _is_reserved_url_index_key(k: Any) -> bool:
    return str(k).startswith(URL_INDEX_RESERVED_PREFIX)


def _index_crawl_finished(index: Dict[str, Any]) -> bool:
    meta = index.get(URL_INDEX_META_KEY)
    return bool(isinstance(meta, dict) and meta.get("crawl_finished"))


def _classify_url_entry(ent: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Classify whether a URL is markdown-done and/or llm-done from a url_index entry.

    IMPORTANT: be tolerant to historical / alternative keys.
      - Some pipelines used "md_path" instead of "markdown_path".
      - Some pipelines may mark markdown completion via flags/status only.
    """
    status: UrlIndexEntryStatus = str(ent.get("status") or "").strip()

    # Be tolerant to legacy keys
    md_path = ent.get("markdown_path") or ent.get("md_path") or ent.get("md_file")
    has_md_path = bool(md_path)

    has_llm_artifact = bool(
        ent.get("json_path")
        or ent.get("extraction_path")
        or ent.get("product_path")
        or ent.get("products_path")
    )

    md_flag = bool(ent.get("markdown_done"))
    llm_flag = bool(ent.get("llm_done"))

    status_md_done = status == "markdown_done"
    status_llm_done = status == "llm_done"

    # If LLM artifacts exist, markdown must have existed too (even if md flag missing).
    markdown_done = bool(
        md_flag
        or has_md_path
        or status_md_done
        or llm_flag
        or has_llm_artifact
        or status_llm_done
    )
    llm_done = bool(llm_flag or has_llm_artifact or status_llm_done)
    return markdown_done, llm_done


def _compute_company_stage_from_index(
    index: Dict[str, Any],
) -> Tuple[CompanyStatus, bool, int, int, int]:
    if not index:
        return COMPANY_STATUS_PENDING, False, 0, 0, 0

    crawl_finished = _index_crawl_finished(index)

    urls_total = 0
    md_done = 0
    llm_done = 0

    for url, raw_ent in index.items():
        if _is_reserved_url_index_key(url):
            continue
        urls_total += 1
        if isinstance(raw_ent, dict):
            m_done, l_done = _classify_url_entry(raw_ent)
            if m_done:
                md_done += 1
            if l_done:
                llm_done += 1

    if urls_total == 0:
        return COMPANY_STATUS_PENDING, crawl_finished, 0, 0, 0

    if not crawl_finished:
        if llm_done > 0:
            return (
                COMPANY_STATUS_LLM_NOT_DONE,
                crawl_finished,
                urls_total,
                md_done,
                llm_done,
            )
        if md_done > 0:
            return (
                COMPANY_STATUS_MD_NOT_DONE,
                crawl_finished,
                urls_total,
                md_done,
                llm_done,
            )
        return COMPANY_STATUS_PENDING, crawl_finished, urls_total, md_done, llm_done

    if llm_done == urls_total:
        status: CompanyStatus = COMPANY_STATUS_LLM_DONE
    elif llm_done > 0:
        status = COMPANY_STATUS_LLM_NOT_DONE
    elif md_done == urls_total:
        status = COMPANY_STATUS_MD_DONE
    else:
        status = COMPANY_STATUS_MD_NOT_DONE

    return status, crawl_finished, urls_total, md_done, llm_done


def _pending_urls_for_stage(
    index: Dict[str, Any], stage: Literal["markdown", "llm"]
) -> List[str]:
    pending: List[str] = []
    for url, raw_ent in index.items():
        if _is_reserved_url_index_key(url):
            continue
        ent = raw_ent if isinstance(raw_ent, dict) else {}
        md_done, llm_done = _classify_url_entry(ent)
        if stage == "markdown":
            if not md_done:
                pending.append(url)
        else:
            if md_done and not llm_done:
                pending.append(url)
    return pending


def log_snapshot(clog: logging.Logger, *, label: str, snap: Company) -> None:
    s = snap.normalized()
    clog.debug(
        "snapshot[%s] company_id=%s status=%s crawl_finished=%s urls_total=%d md_done=%d llm_done=%d last_error=%r done_reason=%r done_at=%r updated_at=%r last_crawled_at=%r",
        label,
        s.company_id,
        s.status,
        bool(s.crawl_finished),
        int(s.urls_total),
        int(s.urls_markdown_done),
        int(s.urls_llm_done),
        s.last_error,
        s.done_reason,
        s.done_at,
        s.updated_at,
        s.last_crawled_at,
    )


def urls_total(snap: Company) -> int:
    return int(snap.urls_total)


def urls_md_done(snap: Company) -> int:
    return int(snap.urls_markdown_done)


def urls_llm_done(snap: Company) -> int:
    return int(snap.urls_llm_done)


def crawl_finished(snap: Company) -> bool:
    return bool(snap.crawl_finished)


def snap_last_error(snap: Company) -> str:
    return (snap.last_error or "").strip()


def crawl_runner_done_ok(snap: Company) -> bool:
    s = snap.normalized()
    return bool(s.crawl_finished) and (not (s.last_error or "").strip())


class CrawlState:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = (
            Path(db_path) if db_path is not None else default_db_path()
        ).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = self._connect(self.db_path)
        self._init_schema()

        self._global_state_write_lock = asyncio.Lock()
        self._last_global_write_ts = 0.0

    def _connect(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema_locked(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS companies (
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

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
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

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_company_done (
                run_id TEXT NOT NULL,
                company_id TEXT NOT NULL,
                done_at TEXT,
                PRIMARY KEY (run_id, company_id)
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_company_attempts (
                run_id TEXT NOT NULL,
                company_id TEXT NOT NULL,
                attempt_no INTEGER NOT NULL,
                started_at TEXT,
                PRIMARY KEY (run_id, company_id, attempt_no)
            )
            """
        )

        self._ensure_columns_locked()

    def _ensure_columns_locked(self) -> None:
        rows = self._conn.execute("PRAGMA table_info(companies)").fetchall()
        cols = {r["name"] for r in rows}

        def _add(col_ddl: str, col_name: str) -> None:
            if col_name in cols:
                return
            self._conn.execute(f"ALTER TABLE companies ADD COLUMN {col_ddl}")

        _add("metadata_json TEXT", "metadata_json")
        _add("industry INTEGER", "industry")
        _add("nace INTEGER", "nace")
        _add("industry_label TEXT", "industry_label")
        _add("industry_label_source TEXT", "industry_label_source")
        _add("crawl_finished INTEGER DEFAULT 0", "crawl_finished")
        _add("done_reason TEXT", "done_reason")
        _add("done_details TEXT", "done_details")
        _add("done_at TEXT", "done_at")
        _add("last_crawled_at TEXT", "last_crawled_at")
        _add("max_pages INTEGER", "max_pages")

        _add("retry_cls TEXT", "retry_cls")
        _add("retry_attempts INTEGER DEFAULT 0", "retry_attempts")
        _add("retry_next_eligible_at REAL DEFAULT 0.0", "retry_next_eligible_at")
        _add("retry_updated_at REAL DEFAULT 0.0", "retry_updated_at")
        _add("retry_last_error TEXT", "retry_last_error")
        _add("retry_last_stage TEXT", "retry_last_stage")

        _add("retry_net_attempts INTEGER DEFAULT 0", "retry_net_attempts")
        _add("retry_stall_attempts INTEGER DEFAULT 0", "retry_stall_attempts")
        _add("retry_mem_attempts INTEGER DEFAULT 0", "retry_mem_attempts")
        _add("retry_other_attempts INTEGER DEFAULT 0", "retry_other_attempts")

        _add("retry_mem_hits INTEGER DEFAULT 0", "retry_mem_hits")
        _add("retry_last_stall_kind TEXT", "retry_last_stall_kind")

        _add(
            "retry_last_progress_md_done INTEGER DEFAULT 0",
            "retry_last_progress_md_done",
        )
        _add("retry_last_seen_md_done INTEGER DEFAULT 0", "retry_last_seen_md_done")

        _add("retry_last_error_sig TEXT", "retry_last_error_sig")
        _add("retry_same_error_streak INTEGER DEFAULT 0", "retry_same_error_streak")
        _add(
            "retry_last_error_sig_updated_at REAL DEFAULT 0.0",
            "retry_last_error_sig_updated_at",
        )

        rows2 = self._conn.execute("PRAGMA table_info(runs)").fetchall()
        cols2 = {r["name"] for r in rows2}

        def _add_run(col_ddl: str, col_name: str) -> None:
            if col_name in cols2:
                return
            self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col_ddl}")

        _add_run("crawl4ai_cache_base_dir TEXT", "crawl4ai_cache_base_dir")
        _add_run("crawl4ai_cache_mode TEXT", "crawl4ai_cache_mode")
        _add_run("args_hash TEXT", "args_hash")
        _add_run("last_company_id TEXT", "last_company_id")

    def _init_schema(self) -> None:
        with self._lock:
            self._init_schema_locked()

    def close(self) -> None:
        with suppress(Exception):
            with self._lock:
                self._conn.close()

    async def _exec(self, sql: str, args: Tuple[Any, ...]) -> None:
        def _run() -> None:
            with self._lock:
                self._conn.execute(sql, args)

        await asyncio.to_thread(_run)

    async def _query_one(
        self, sql: str, args: Tuple[Any, ...]
    ) -> Optional[sqlite3.Row]:
        def _run() -> Optional[sqlite3.Row]:
            with self._lock:
                return self._conn.execute(sql, args).fetchone()

        return await asyncio.to_thread(_run)

    async def _query_all(self, sql: str, args: Tuple[Any, ...]) -> List[sqlite3.Row]:
        def _run() -> List[sqlite3.Row]:
            with self._lock:
                return self._conn.execute(sql, args).fetchall()

        return await asyncio.to_thread(_run)

    @staticmethod
    def _in_progress_statuses() -> Tuple[str, ...]:
        return (
            COMPANY_STATUS_PENDING,
            COMPANY_STATUS_MD_NOT_DONE,
            COMPANY_STATUS_LLM_NOT_DONE,
        )

    @staticmethod
    def _done_statuses_for_progress(*, llm_requested: bool) -> Tuple[str, ...]:
        if llm_requested:
            return (COMPANY_STATUS_LLM_DONE,)
        return (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            COMPANY_STATUS_TERMINAL_DONE,
        )

    async def get_db_progress_counts(self, *, llm_requested: bool) -> Tuple[int, int]:
        done_sts = self._done_statuses_for_progress(llm_requested=llm_requested)
        placeholders = ",".join("?" for _ in done_sts)

        def _run() -> Tuple[int, int]:
            with self._lock:
                total_row = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM companies"
                ).fetchone()
                done_row = self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM companies WHERE status IN ({placeholders})",
                    tuple(done_sts),
                ).fetchone()
            total = int(total_row["c"] or 0) if total_row is not None else 0
            done = int(done_row["c"] or 0) if done_row is not None else 0
            return done, total

        return await asyncio.to_thread(_run)

    async def mark_company_attempt_started(self, run_id: str, company_id: str) -> int:
        now = _now_iso()

        def _run() -> int:
            with self._lock:
                prev_row = self._conn.execute(
                    """
                    SELECT COALESCE(MAX(attempt_no), 0) AS m
                      FROM run_company_attempts
                     WHERE run_id=? AND company_id=?
                    """,
                    (run_id, company_id),
                ).fetchone()
                prev = int(prev_row["m"] or 0) if prev_row is not None else 0
                attempt_no = prev + 1
                self._conn.execute(
                    """
                    INSERT INTO run_company_attempts (run_id, company_id, attempt_no, started_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, company_id, attempt_no, now),
                )
                return attempt_no

        return await asyncio.to_thread(_run)

    def _row_to_company(self, row: sqlite3.Row) -> Company:
        md: Dict[str, Any] = {}
        mj = row["metadata_json"] if "metadata_json" in row.keys() else None
        if isinstance(mj, str) and mj.strip():
            try:
                obj = json.loads(mj)
                if isinstance(obj, dict):
                    md = obj
            except Exception:
                md = {}

        done_details_obj: Optional[Dict[str, Any]] = None
        dd = row["done_details"] if "done_details" in row.keys() else None
        if isinstance(dd, str) and dd.strip():
            try:
                obj2 = json.loads(dd)
                if isinstance(obj2, dict):
                    done_details_obj = obj2
            except Exception:
                done_details_obj = None

        c = Company(
            company_id=str(row["company_id"]),
            root_url=str(row["root_url"] or ""),
            name=(str(row["name"]) if row["name"] is not None else None),
            metadata=md,
            industry=(int(row["industry"]) if row["industry"] is not None else None),
            nace=(int(row["nace"]) if row["nace"] is not None else None),
            industry_label=(
                str(row["industry_label"])
                if row["industry_label"] is not None
                else None
            ),
            industry_label_source=(
                str(row["industry_label_source"])
                if row["industry_label_source"] is not None
                else None
            ),
            status=_normalize_status(
                str(row["status"]) if row["status"] is not None else None
            ),
            crawl_finished=bool(int(row["crawl_finished"] or 0)),
            urls_total=int(row["urls_total"] or 0),
            urls_markdown_done=int(row["urls_markdown_done"] or 0),
            urls_llm_done=int(row["urls_llm_done"] or 0),
            last_error=(
                str(row["last_error"]) if row["last_error"] is not None else None
            ),
            done_reason=(
                str(row["done_reason"]) if row["done_reason"] is not None else None
            ),
            done_details=done_details_obj,
            done_at=(str(row["done_at"]) if row["done_at"] is not None else None),
            created_at=(
                str(row["created_at"]) if row["created_at"] is not None else None
            ),
            updated_at=(
                str(row["updated_at"]) if row["updated_at"] is not None else None
            ),
            last_crawled_at=(
                str(row["last_crawled_at"])
                if row["last_crawled_at"] is not None
                else None
            ),
            max_pages=(int(row["max_pages"]) if row["max_pages"] is not None else None),
            retry_cls=(
                str(row["retry_cls"]) if row["retry_cls"] is not None else "net"
            ),
            retry_attempts=int(row["retry_attempts"] or 0),
            retry_next_eligible_at=float(row["retry_next_eligible_at"] or 0.0),
            retry_updated_at=float(row["retry_updated_at"] or 0.0),
            retry_last_error=(
                str(row["retry_last_error"])
                if row["retry_last_error"] is not None
                else ""
            ),
            retry_last_stage=(
                str(row["retry_last_stage"])
                if row["retry_last_stage"] is not None
                else ""
            ),
            retry_net_attempts=int(row["retry_net_attempts"] or 0),
            retry_stall_attempts=int(row["retry_stall_attempts"] or 0),
            retry_mem_attempts=int(row["retry_mem_attempts"] or 0),
            retry_other_attempts=int(row["retry_other_attempts"] or 0),
            retry_mem_hits=int(row["retry_mem_hits"] or 0),
            retry_last_stall_kind=(
                str(row["retry_last_stall_kind"])
                if row["retry_last_stall_kind"] is not None
                else "unknown"
            ),
            retry_last_progress_md_done=int(row["retry_last_progress_md_done"] or 0),
            retry_last_seen_md_done=int(row["retry_last_seen_md_done"] or 0),
            retry_last_error_sig=(
                str(row["retry_last_error_sig"])
                if row["retry_last_error_sig"] is not None
                else ""
            ),
            retry_same_error_streak=int(row["retry_same_error_streak"] or 0),
            retry_last_error_sig_updated_at=float(
                row["retry_last_error_sig_updated_at"] or 0.0
            ),
        )
        return c.normalized()

    async def write_company_meta_snapshot(
        self,
        company_id: str,
        company: Company,
        *,
        pretty: bool = True,
        company_ctx: Optional[Dict[str, Any]] = None,
        set_last_crawled_at: bool = True,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            _write_company_meta_snapshot_sync,
            company_id,
            company.normalized(),
            pretty=pretty,
            company_ctx=company_ctx,
            set_last_crawled_at=set_last_crawled_at,
        )

    async def patch_company_meta(
        self, company_id: str, patch: Dict[str, Any], *, pretty: bool = True
    ) -> None:
        await asyncio.to_thread(patch_company_meta, company_id, patch, pretty=pretty)

    async def has_in_progress_companies(self) -> bool:
        sts = self._in_progress_statuses()
        placeholders = ",".join("?" for _ in sts)
        row = await self._query_one(
            f"SELECT 1 AS x FROM companies WHERE status IN ({placeholders}) LIMIT 1",
            tuple(sts),
        )
        return row is not None

    async def get_in_progress_company_ids(self, limit: int = 2000) -> List[str]:
        limit = max(1, int(limit))
        sts = self._in_progress_statuses()
        placeholders = ",".join("?" for _ in sts)
        rows = await self._query_all(
            f"""
            SELECT company_id FROM companies
             WHERE status IN ({placeholders})
             ORDER BY updated_at DESC
             LIMIT ?
            """,
            tuple(sts) + (limit,),
        )
        return [str(r["company_id"]) for r in rows]

    async def recompute_all_in_progress(self, *, concurrency: int = 32) -> None:
        c = max(1, int(concurrency))
        ids = await self.get_in_progress_company_ids(limit=1_000_000)
        if not ids:
            return

        sem = asyncio.Semaphore(c)

        async def _one(cid: str) -> None:
            async with sem:
                await self.recompute_company_from_index(cid)

        batch = max(64, c * 8)
        for i in range(0, len(ids), batch):
            await asyncio.gather(*(_one(cid) for cid in ids[i : i + batch]))

    @staticmethod
    def _parse_run_args_max_pages(args_hash: Optional[str]) -> Optional[int]:
        if not args_hash:
            return None
        s = str(args_hash).strip()
        if not s:
            return None

        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except Exception:
                return None
            if isinstance(obj, dict) and "max_pages" in obj:
                v = obj.get("max_pages")
                try:
                    iv = int(str(v).strip())
                except Exception:
                    return None
                return iv if iv > 0 else None
            return None

        if s.startswith("max_pages="):
            tail = s[len("max_pages=") :].strip()
            if not tail:
                return None
            try:
                iv = int(tail)
            except Exception:
                return None
            return iv if iv > 0 else None

        return None

    async def start_run(
        self,
        pipeline: str,
        *,
        version: Optional[str] = None,
        args_hash: Optional[str] = None,
        run_id: Optional[str] = None,
        crawl4ai_cache_base_dir: Optional[str] = None,
        crawl4ai_cache_mode: Optional[str] = None,
    ) -> str:
        rid = run_id or f"run-{int(time.time())}-{os.getpid()}"
        now = _now_iso()
        await self._exec(
            """
            INSERT INTO runs (
                run_id, pipeline, version, args_hash,
                crawl4ai_cache_base_dir, crawl4ai_cache_mode,
                started_at, total_companies, completed_companies,
                last_company_id, last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                pipeline=excluded.pipeline,
                version=excluded.version,
                args_hash=excluded.args_hash,
                crawl4ai_cache_base_dir=excluded.crawl4ai_cache_base_dir,
                crawl4ai_cache_mode=excluded.crawl4ai_cache_mode,
                started_at=excluded.started_at,
                last_updated=excluded.last_updated
            """,
            (
                rid,
                pipeline,
                version or "",
                args_hash or "",
                crawl4ai_cache_base_dir,
                crawl4ai_cache_mode,
                now,
                now,
            ),
        )
        return rid

    async def update_run_totals(
        self,
        run_id: str,
        *,
        total_companies: Optional[int] = None,
        completed_companies: Optional[int] = None,
        last_company_id: Optional[str] = None,
    ) -> None:
        now = _now_iso()
        sets = ["last_updated=?"]
        args: List[Any] = [now]

        if total_companies is not None:
            sets.append("total_companies=?")
            args.append(int(total_companies))
        if completed_companies is not None:
            sets.append("completed_companies=?")
            args.append(int(completed_companies))
        if last_company_id is not None:
            sets.append("last_company_id=?")
            args.append(last_company_id)

        args.append(run_id)
        sql = f"UPDATE runs SET {', '.join(sets)} WHERE run_id=?"
        await self._exec(sql, tuple(args))

    async def mark_company_completed(self, run_id: str, company_id: str) -> None:
        now = _now_iso()

        def _run() -> None:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO run_company_done (run_id, company_id, done_at)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, company_id, now),
                )
                inserted = int(
                    (self._conn.execute("SELECT changes()").fetchone() or [0])[0]
                )
                if inserted > 0:
                    self._conn.execute(
                        """
                        UPDATE runs
                           SET completed_companies = COALESCE(completed_companies, 0) + 1,
                               last_company_id = ?,
                               last_updated = ?,
                               total_companies = CASE
                                   WHEN total_companies < (COALESCE(completed_companies, 0) + 1)
                                       THEN (COALESCE(completed_companies, 0) + 1)
                                   ELSE total_companies
                               END
                         WHERE run_id = ?
                        """,
                        (company_id, now, run_id),
                    )
                else:
                    self._conn.execute(
                        "UPDATE runs SET last_company_id=?, last_updated=? WHERE run_id=?",
                        (company_id, now, run_id),
                    )

        await asyncio.to_thread(_run)

    async def get_latest_run_max_pages(self) -> Optional[int]:
        row = await self._query_one(
            "SELECT args_hash FROM runs ORDER BY started_at DESC LIMIT 1",
            tuple(),
        )
        if row is None:
            return None
        v = row["args_hash"] if "args_hash" in row.keys() else None
        return self._parse_run_args_max_pages(str(v) if v is not None else None)

    async def upsert_company(
        self,
        company_id: str,
        *,
        root_url: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        industry: Optional[int] = None,
        nace: Optional[int] = None,
        industry_label: Optional[str] = None,
        industry_label_source: Optional[str] = None,
        industry_source: Optional[str] = None,
        status: Optional[CompanyStatus] = None,
        crawl_finished: Optional[bool] = None,
        urls_total: Optional[int] = None,
        urls_markdown_done: Optional[int] = None,
        urls_llm_done: Optional[int] = None,
        last_error: Optional[str] = None,
        done_reason: Optional[str] = None,
        done_details: Optional[Dict[str, Any]] = None,
        done_at: Optional[str] = None,
        last_crawled_at: Optional[str] = None,
        max_pages: Optional[int] = None,
        retry_cls: Optional[str] = None,
        retry_attempts: Optional[int] = None,
        retry_next_eligible_at: Optional[float] = None,
        retry_updated_at: Optional[float] = None,
        retry_last_error: Optional[str] = None,
        retry_last_stage: Optional[str] = None,
        retry_net_attempts: Optional[int] = None,
        retry_stall_attempts: Optional[int] = None,
        retry_mem_attempts: Optional[int] = None,
        retry_other_attempts: Optional[int] = None,
        retry_mem_hits: Optional[int] = None,
        retry_last_stall_kind: Optional[str] = None,
        retry_last_progress_md_done: Optional[int] = None,
        retry_last_seen_md_done: Optional[int] = None,
        retry_last_error_sig: Optional[str] = None,
        retry_same_error_streak: Optional[int] = None,
        retry_last_error_sig_updated_at: Optional[float] = None,
        write_meta: bool = False,
        company_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _now_iso()

        md_json: Optional[str] = None
        if metadata is not None:
            md_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

        dd_json: Optional[str] = None
        if done_details is not None:
            dd_json = json.dumps(
                done_details, ensure_ascii=False, separators=(",", ":")
            )

        await self._exec(
            """
            INSERT OR IGNORE INTO companies (
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
                NULL, NULL, NULL, NULL,
                ?, 0,
                0, 0, 0,
                NULL, NULL, NULL, NULL,
                ?, ?, NULL,
                NULL,
                NULL, 0, 0.0, 0.0,
                '', '',
                0, 0, 0, 0,
                0, 'unknown',
                0, 0,
                '', 0, 0.0
            )
            """,
            (
                company_id,
                root_url or "",
                name,
                md_json,
                status or COMPANY_STATUS_PENDING,
                now,
                now,
            ),
        )

        eff_industry_label_source = (
            industry_label_source
            if industry_label_source is not None
            else industry_source
        )

        sets: List[str] = ["updated_at=?"]
        args: List[Any] = [now]

        if root_url is not None:
            sets.append("root_url=?")
            args.append(root_url)
        if name is not None:
            sets.append("name=?")
            args.append(name)
        if metadata is not None:
            sets.append("metadata_json=?")
            args.append(md_json)
        if industry is not None:
            sets.append("industry=?")
            args.append(int(industry))
        if nace is not None:
            sets.append("nace=?")
            args.append(int(nace))
        if industry_label is not None:
            sets.append("industry_label=?")
            args.append(industry_label)
        if eff_industry_label_source is not None:
            sets.append("industry_label_source=?")
            args.append(eff_industry_label_source)

        if status is not None:
            sets.append("status=?")
            args.append(status)
        if crawl_finished is not None:
            sets.append("crawl_finished=?")
            args.append(1 if crawl_finished else 0)
        if urls_total is not None:
            sets.append("urls_total=?")
            args.append(int(urls_total))
        if urls_markdown_done is not None:
            sets.append("urls_markdown_done=?")
            args.append(int(urls_markdown_done))
        if urls_llm_done is not None:
            sets.append("urls_llm_done=?")
            args.append(int(urls_llm_done))

        if last_error is not None:
            sets.append("last_error=?")
            args.append((last_error or "")[:4000] or None)
        if done_reason is not None:
            sets.append("done_reason=?")
            args.append((done_reason or "")[:256] or None)
        if done_details is not None:
            sets.append("done_details=?")
            args.append(dd_json)
        if done_at is not None:
            sets.append("done_at=?")
            args.append(done_at)

        if last_crawled_at is not None:
            sets.append("last_crawled_at=?")
            args.append(last_crawled_at)

        if max_pages is not None:
            sets.append("max_pages=?")
            args.append(int(max_pages))

        if retry_cls is not None:
            sets.append("retry_cls=?")
            args.append(retry_cls)
        if retry_attempts is not None:
            sets.append("retry_attempts=?")
            args.append(int(retry_attempts))
        if retry_next_eligible_at is not None:
            sets.append("retry_next_eligible_at=?")
            args.append(float(retry_next_eligible_at))
        if retry_updated_at is not None:
            sets.append("retry_updated_at=?")
            args.append(float(retry_updated_at))
        if retry_last_error is not None:
            sets.append("retry_last_error=?")
            args.append(str(retry_last_error or ""))
        if retry_last_stage is not None:
            sets.append("retry_last_stage=?")
            args.append(str(retry_last_stage or ""))

        if retry_net_attempts is not None:
            sets.append("retry_net_attempts=?")
            args.append(int(retry_net_attempts))
        if retry_stall_attempts is not None:
            sets.append("retry_stall_attempts=?")
            args.append(int(retry_stall_attempts))
        if retry_mem_attempts is not None:
            sets.append("retry_mem_attempts=?")
            args.append(int(retry_mem_attempts))
        if retry_other_attempts is not None:
            sets.append("retry_other_attempts=?")
            args.append(int(retry_other_attempts))

        if retry_mem_hits is not None:
            sets.append("retry_mem_hits=?")
            args.append(int(retry_mem_hits))
        if retry_last_stall_kind is not None:
            sets.append("retry_last_stall_kind=?")
            args.append(str(retry_last_stall_kind or "unknown"))

        if retry_last_progress_md_done is not None:
            sets.append("retry_last_progress_md_done=?")
            args.append(int(retry_last_progress_md_done))
        if retry_last_seen_md_done is not None:
            sets.append("retry_last_seen_md_done=?")
            args.append(int(retry_last_seen_md_done))

        if retry_last_error_sig is not None:
            sets.append("retry_last_error_sig=?")
            args.append(str(retry_last_error_sig or ""))
        if retry_same_error_streak is not None:
            sets.append("retry_same_error_streak=?")
            args.append(int(retry_same_error_streak))
        if retry_last_error_sig_updated_at is not None:
            sets.append("retry_last_error_sig_updated_at=?")
            args.append(float(retry_last_error_sig_updated_at))

        args.append(company_id)
        await self._exec(
            f"UPDATE companies SET {', '.join(sets)} WHERE company_id=?", tuple(args)
        )

        if write_meta:
            snap = await self.get_company_snapshot(company_id, recompute=False)
            await self.write_company_meta_snapshot(
                company_id,
                snap,
                pretty=True,
                company_ctx=company_ctx,
                set_last_crawled_at=True,
            )

    async def get_company_snapshot(
        self, company_id: str, *, recompute: bool = True
    ) -> Company:
        row = await self._query_one(
            "SELECT * FROM companies WHERE company_id=?", (company_id,)
        )
        if row is not None:
            c = self._row_to_company(row)
            if recompute:
                return await self.recompute_company_from_index(company_id)
            return c

        now = _now_iso()
        await self._exec(
            """
            INSERT OR IGNORE INTO companies (
                company_id, root_url, name, metadata_json,
                status, crawl_finished,
                urls_total, urls_markdown_done, urls_llm_done,
                created_at, updated_at
            )
            VALUES (?, '', NULL, NULL, ?, 0, 0, 0, 0, ?, ?)
            """,
            (company_id, COMPANY_STATUS_PENDING, now, now),
        )
        row2 = await self._query_one(
            "SELECT * FROM companies WHERE company_id=?", (company_id,)
        )
        c2 = self._row_to_company(row2)  # type: ignore[arg-type]
        if recompute:
            return await self.recompute_company_from_index(company_id)
        return c2

    async def recompute_company_from_index(
        self,
        company_id: str,
        *,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
        write_meta: bool = False,
        company_ctx: Optional[Dict[str, Any]] = None,
    ) -> Company:
        cur = await self._query_one(
            """
            SELECT status, crawl_finished, urls_total, urls_markdown_done, urls_llm_done
              FROM companies
             WHERE company_id=?
            """,
            (company_id,),
        )

        cur_status = (
            _normalize_status(cur["status"])
            if cur is not None
            else COMPANY_STATUS_PENDING
        )

        cur_crawl_finished = (
            bool(int(cur["crawl_finished"] or 0)) if cur is not None else False
        )
        cur_total = int(cur["urls_total"] or 0) if cur is not None else 0
        cur_md = int(cur["urls_markdown_done"] or 0) if cur is not None else 0
        cur_llm = int(cur["urls_llm_done"] or 0) if cur is not None else 0

        index = await asyncio.to_thread(load_url_index, company_id)
        derived_status, derived_crawl_finished, total_i, md_i, llm_i = (
            _compute_company_stage_from_index(index if isinstance(index, dict) else {})
        )

        new_status = _prefer_higher_status(cur_status, derived_status)

        new_total = max(cur_total, int(total_i))
        new_md = max(cur_md, int(md_i))
        new_llm = max(cur_llm, int(llm_i))

        new_total = max(new_total, new_md, new_llm)
        new_md = min(new_md, new_total)
        new_llm = min(new_llm, new_total)

        if new_llm > 0:
            new_md = max(new_md, new_llm)

        if new_status == COMPANY_STATUS_LLM_DONE:
            new_md = new_total
            new_llm = new_total

        new_crawl_finished = bool(cur_crawl_finished or derived_crawl_finished)

        await self.upsert_company(
            company_id,
            name=name,
            root_url=root_url,
            status=new_status,
            crawl_finished=new_crawl_finished,
            urls_total=new_total,
            urls_markdown_done=new_md,
            urls_llm_done=new_llm,
            write_meta=False,
        )

        snap = await self.get_company_snapshot(company_id, recompute=False)
        if write_meta:
            await self.write_company_meta_snapshot(
                company_id, snap, pretty=True, company_ctx=company_ctx
            )
        return snap

    async def get_pending_urls_for_markdown(self, company_id: str) -> List[str]:
        index = await asyncio.to_thread(load_url_index, company_id)
        return _pending_urls_for_stage(
            index if isinstance(index, dict) else {}, "markdown"
        )

    async def get_pending_urls_for_llm(self, company_id: str) -> List[str]:
        index = await asyncio.to_thread(load_url_index, company_id)
        return _pending_urls_for_stage(index if isinstance(index, dict) else {}, "llm")

    async def mark_company_terminal(
        self,
        company_id: str,
        *,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        last_error: Optional[str] = None,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
        write_meta: bool = True,
        company_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _now_iso()

        index = await asyncio.to_thread(load_url_index, company_id)
        _, _, total, md_done, llm_done = _compute_company_stage_from_index(
            index if isinstance(index, dict) else {}
        )

        last_error_trim = (last_error or "")[:4000] if last_error is not None else None

        await self.upsert_company(
            company_id,
            name=name,
            root_url=root_url,
            status=COMPANY_STATUS_TERMINAL_DONE,
            crawl_finished=True,
            urls_total=int(total),
            urls_markdown_done=int(md_done),
            urls_llm_done=min(int(llm_done), int(total)),
            last_error=last_error_trim,
            done_reason=(reason or "")[:256],
            done_details=details if isinstance(details, dict) else None,
            done_at=now,
            write_meta=False,
        )

        if write_meta:
            snap = await self.get_company_snapshot(company_id, recompute=False)
            await self.write_company_meta_snapshot(
                company_id,
                snap,
                pretty=True,
                company_ctx=company_ctx,
                set_last_crawled_at=True,
            )

    async def clear_company_terminal(
        self, company_id: str, *, keep_status: bool = False
    ) -> None:
        row = await self._query_one(
            "SELECT status FROM companies WHERE company_id=?", (company_id,)
        )
        if row is None:
            return

        cur = _normalize_status(
            str(row["status"]) if row["status"] is not None else None
        )
        new_status: CompanyStatus = cur
        if (not keep_status) and cur == COMPANY_STATUS_TERMINAL_DONE:
            new_status = COMPANY_STATUS_PENDING

        now = _now_iso()
        if keep_status:
            await self._exec(
                "UPDATE companies SET updated_at=? WHERE company_id=?",
                (now, company_id),
            )
            return

        await self._exec(
            """
            UPDATE companies
               SET status=?,
                   done_reason=NULL,
                   done_details=NULL,
                   done_at=NULL,
                   crawl_finished=0,
                   updated_at=?
             WHERE company_id=?
            """,
            (new_status, now, company_id),
        )

    async def write_global_state_from_db_only(
        self, *, max_ids: int = 200, pretty: bool = False
    ) -> Dict[str, Any]:
        def _compute() -> Dict[str, Any]:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT company_id, status, crawl_finished,
                           urls_total, urls_markdown_done, urls_llm_done,
                           done_reason
                      FROM companies
                    """
                ).fetchall()

                run_row = self._conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
                ).fetchone()

                run_done_count: Optional[int] = None
                if run_row is not None:
                    rrid = run_row["run_id"]
                    c = self._conn.execute(
                        "SELECT COUNT(*) AS c FROM run_company_done WHERE run_id=?",
                        (rrid,),
                    ).fetchone()
                    run_done_count = int(c["c"] or 0) if c is not None else None

            total = len(rows)

            pending_ids: List[str] = []
            in_progress_ids: List[str] = []
            done_ids: List[str] = []

            pending_companies = 0
            md_done_companies = 0
            llm_done_companies = 0
            terminal_done_companies = 0
            terminal_done_reasons: Dict[str, int] = {}

            urls_total_sum = 0
            urls_md_done_sum = 0
            urls_llm_done_sum = 0
            crawl_finished_companies = 0

            for r in rows:
                st = _normalize_status(r["status"])
                cid = str(r["company_id"])

                if st == COMPANY_STATUS_PENDING:
                    pending_companies += 1
                    if len(pending_ids) < max_ids:
                        pending_ids.append(cid)
                elif st in (COMPANY_STATUS_MD_NOT_DONE, COMPANY_STATUS_LLM_NOT_DONE):
                    if len(in_progress_ids) < max_ids:
                        in_progress_ids.append(cid)
                else:
                    if len(done_ids) < max_ids:
                        done_ids.append(cid)

                if st in (
                    COMPANY_STATUS_MD_DONE,
                    COMPANY_STATUS_LLM_DONE,
                    COMPANY_STATUS_TERMINAL_DONE,
                ):
                    md_done_companies += 1
                if st == COMPANY_STATUS_LLM_DONE:
                    llm_done_companies += 1
                if st == COMPANY_STATUS_TERMINAL_DONE:
                    terminal_done_companies += 1
                    dr = (r["done_reason"] or "unknown").strip()
                    terminal_done_reasons[dr] = terminal_done_reasons.get(dr, 0) + 1

                if bool(int(r["crawl_finished"] or 0)):
                    crawl_finished_companies += 1

                urls_total_sum += int(r["urls_total"] or 0)
                urls_md_done_sum += int(r["urls_markdown_done"] or 0)
                urls_llm_done_sum += int(r["urls_llm_done"] or 0)

            crawled_companies = total - pending_companies

            md_done_pct = (md_done_companies / total * 100.0) if total else 0.0
            llm_done_pct = (llm_done_companies / total * 100.0) if total else 0.0
            crawl_finished_pct = (
                (crawl_finished_companies / total * 100.0) if total else 0.0
            )

            latest_run: Optional[Dict[str, Any]] = None
            if run_row is not None:
                run_total = int(run_row["total_companies"] or 0)
                run_completed = int(run_row["completed_companies"] or 0)

                if run_done_count is not None and run_done_count > 0:
                    run_completed = int(run_done_count)

                if run_total <= 0 or run_total != total:
                    run_total = total
                run_completed = min(int(run_completed), int(run_total))

                latest_run = {
                    "run_id": run_row["run_id"],
                    "pipeline": run_row["pipeline"],
                    "version": run_row["version"],
                    "args_hash": run_row["args_hash"],
                    "crawl4ai_cache_base_dir": run_row["crawl4ai_cache_base_dir"],
                    "crawl4ai_cache_mode": run_row["crawl4ai_cache_mode"],
                    "started_at": run_row["started_at"],
                    "total_companies": run_total,
                    "completed_companies": run_completed,
                    "last_company_id": run_row["last_company_id"],
                    "last_updated": run_row["last_updated"],
                }

            payload: Dict[str, Any] = {
                "output_root": str(_output_root()),
                "db_path": str(self.db_path),
                "updated_at": _now_iso(),
                "total_companies": int(total),
                "crawled_companies": int(crawled_companies),
                "crawl_finished_companies": int(crawl_finished_companies),
                "md_done_companies": int(md_done_companies),
                "llm_done_companies": int(llm_done_companies),
                "terminal_done_companies": int(terminal_done_companies),
                "crawl_finished_pct": round(crawl_finished_pct, 2),
                "md_done_pct": round(md_done_pct, 2),
                "llm_done_pct": round(llm_done_pct, 2),
                "company_ids_sample": {
                    "pending": pending_ids,
                    "in_progress": in_progress_ids,
                    "done": done_ids,
                },
                "urls_total_sum": int(urls_total_sum),
                "urls_markdown_done_sum": int(urls_md_done_sum),
                "urls_llm_done_sum": int(urls_llm_done_sum),
                "latest_run": latest_run,
            }

            if terminal_done_reasons:
                payload["terminal_done_reasons"] = terminal_done_reasons

            _atomic_write_json(_global_state_path(), payload, pretty=pretty)
            return payload

        return await asyncio.to_thread(_compute)

    async def write_global_state_throttled(
        self, min_interval_sec: float = 2.0, *, max_ids: int = 200, pretty: bool = False
    ) -> Dict[str, Any]:
        mi = max(0.05, float(min_interval_sec))
        async with self._global_state_write_lock:
            now = time.time()
            if (now - self._last_global_write_ts) < mi:
                return await self.write_global_state_from_db_only(
                    max_ids=max_ids, pretty=pretty
                )
            payload = await self.write_global_state_from_db_only(
                max_ids=max_ids, pretty=pretty
            )
            self._last_global_write_ts = time.time()
            return payload


_STATE_LOCK = threading.Lock()
_STATE: Optional[CrawlState] = None


def get_crawl_state(db_path: Optional[Path] = None) -> CrawlState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = CrawlState(db_path=db_path)
        return _STATE


__all__ = [
    "DEFAULT_DB",
    "CrawlState",
    "get_crawl_state",
    "CompanyStatus",
    "load_url_index",
    "upsert_url_index_entry",
    "patch_url_index_meta",
    "load_crawl_meta",
    "patch_crawl_meta",
    "patch_company_meta",
    "log_snapshot",
    "crawl_runner_done_ok",
    "snap_last_error",
    "urls_total",
    "urls_md_done",
    "urls_llm_done",
    "crawl_finished",
]
