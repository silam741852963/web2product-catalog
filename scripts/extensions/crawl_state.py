from __future__ import annotations

import asyncio
import errno
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from extensions import output_paths
from extensions.output_paths import (
    ensure_company_dirs,
    sanitize_bvdid,
)

# ---------------------------------------------------------------------------
# Atomic file I/O + bounded JSON cache + per-path locks
# ---------------------------------------------------------------------------

_JSON_CACHE_LOCK = threading.Lock()
_FILE_LOCKS_LOCK = threading.Lock()
_DIR_CACHE_LOCK = threading.Lock()

# Bounded caches (avoid unbounded growth when crawling tens of thousands of companies)
_JSON_CACHE_MAX = int(os.getenv("CRAWLSTATE_JSON_CACHE_SIZE", "2048"))
_DIR_CACHE_MAX = int(os.getenv("CRAWLSTATE_DIR_CACHE_SIZE", "4096"))

# Path -> (mtime_ns, obj) as LRU
_JSON_CACHE: "OrderedDict[Path, Tuple[int, Any]]" = OrderedDict()
# bvdid -> metadata_dir as LRU
_DIR_CACHE: "OrderedDict[str, Path]" = OrderedDict()
# Path -> Lock (for read-modify-write JSON files like url_index.json)
_FILE_LOCKS: Dict[Path, threading.Lock] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_root() -> Path:
    """
    IMPORTANT: output_paths.OUTPUT_ROOT is mutated by run.py at runtime.
    Never capture it at import-time. Always resolve dynamically here.
    """
    try:
        root = getattr(output_paths, "OUTPUT_ROOT", None)
        if root is None:
            return Path("outputs").resolve()
        return Path(root).resolve()
    except Exception:
        return Path("outputs").resolve()


def default_db_path() -> Path:
    return _output_root() / "crawl_state.sqlite3"


def _retry_emfile(fn, attempts: int = 6, base_delay: float = 0.15):
    """
    Retry helper for transient file-descriptor exhaustion.
    """
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENFILE) or "Too many open files" in str(
                e
            ):
                time.sleep(base_delay * (2**i))
                continue
            raise


def _get_file_lock(path: Path) -> threading.Lock:
    # Per-file lock prevents lost updates for read-modify-write JSON patterns.
    with _FILE_LOCKS_LOCK:
        lk = _FILE_LOCKS.get(path)
        if lk is None:
            lk = threading.Lock()
            _FILE_LOCKS[path] = lk
        return lk


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """
    Robust atomic write:
      * Writes to a unique temp file in the same dir
      * fsyncs it (best-effort)
      * Replaces the target
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    tmp = Path(f"{str(path)}.tmp.{stamp}")

    def _write() -> None:
        try:
            with open(tmp, "w", encoding=encoding, newline="") as f:
                f.write(data)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp, path)
        finally:
            with suppress(Exception):
                if tmp.exists() and tmp != path:
                    tmp.unlink()

    _retry_emfile(_write)


def _json_dumps(obj: Any, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_cache_get(path: Path, mtime_ns: int) -> Optional[Any]:
    with _JSON_CACHE_LOCK:
        entry = _JSON_CACHE.get(path)
        if entry is None:
            return None
        cached_mtime_ns, obj = entry
        if cached_mtime_ns != mtime_ns:
            return None
        _JSON_CACHE.move_to_end(path, last=True)
        return obj


def _json_cache_put(path: Path, mtime_ns: int, obj: Any) -> None:
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[path] = (mtime_ns, obj)
        _JSON_CACHE.move_to_end(path, last=True)
        while len(_JSON_CACHE) > _JSON_CACHE_MAX:
            _JSON_CACHE.popitem(last=False)


def _atomic_write_json(
    path: Path, obj: Any, *, cache: bool = True, pretty: bool = True
) -> None:
    payload = _json_dumps(obj, pretty=pretty)
    _atomic_write_text(path, payload, "utf-8")
    if cache:
        with suppress(OSError):
            mtime_ns = path.stat().st_mtime_ns
            _json_cache_put(path, mtime_ns, obj)


def _json_load_cached(path: Path) -> Any:
    """
    Load JSON with a bounded LRU cache keyed by path+mtime_ns.
    Intended for small files (e.g., crawl_meta.json). Do NOT use for url_index.json.
    """
    try:
        st = path.stat()
        mtime_ns = st.st_mtime_ns
    except OSError:
        return {}

    cached = _json_cache_get(path, mtime_ns)
    if cached is not None:
        return cached

    def _read() -> Any:
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return {}

    obj = _retry_emfile(_read)
    _json_cache_put(path, mtime_ns, obj)
    return obj


def _json_load_nocache(path: Path) -> Any:
    """
    Load JSON without touching the global cache.
    Intended for large per-company manifests like url_index.json.
    """

    def _read() -> Any:
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return {}

    return _retry_emfile(_read)


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

META_NAME = "crawl_meta.json"
URL_INDEX_NAME = "url_index.json"
GLOBAL_STATE_NAME = "crawl_global_state.json"

# Keep for backwards compat imports (but DO NOT rely on it being dynamic).
DEFAULT_DB = default_db_path()

CompanyStatus = Literal[
    "pending",
    "markdown_not_done",
    "markdown_done",
    "llm_not_done",
    "llm_done",
    "terminal_done",
]

COMPANY_STATUS_PENDING: CompanyStatus = "pending"
COMPANY_STATUS_MD_NOT_DONE: CompanyStatus = "markdown_not_done"
COMPANY_STATUS_MD_DONE: CompanyStatus = "markdown_done"
COMPANY_STATUS_LLM_NOT_DONE: CompanyStatus = "llm_not_done"
COMPANY_STATUS_LLM_DONE: CompanyStatus = "llm_done"
COMPANY_STATUS_TERMINAL_DONE: CompanyStatus = "terminal_done"

# Per-URL status semantics derived from run_utils.upsert_url_index
_MARKDOWN_COMPLETE_STATUSES = {"markdown_saved", "markdown_suppressed"}
_LLM_COMPLETE_STATUSES = {
    "llm_extracted",
    "llm_extracted_empty",
    "llm_full_extracted",  # treat full extraction as LLM-complete as well
}


def _company_metadata_dir(bvdid: str) -> Path:
    """
    Returns the company-level metadata directory (outputs/{safe}/metadata).
    Uses ensure_company_dirs so it cooperates with the rest of the system.
    Uses a bounded LRU to reduce repeated fs ops within a run.
    """
    with _DIR_CACHE_LOCK:
        cached = _DIR_CACHE.get(bvdid)
        if cached is not None:
            _DIR_CACHE.move_to_end(bvdid, last=True)
            return cached

    dirs = ensure_company_dirs(bvdid)
    meta = dirs.get("metadata") or dirs.get("checkpoints")
    if meta is None:
        safe = sanitize_bvdid(bvdid)
        meta = _output_root() / safe / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
    else:
        meta = Path(meta)
        meta.mkdir(parents=True, exist_ok=True)

    with _DIR_CACHE_LOCK:
        _DIR_CACHE[bvdid] = meta
        _DIR_CACHE.move_to_end(bvdid, last=True)
        while len(_DIR_CACHE) > _DIR_CACHE_MAX:
            _DIR_CACHE.popitem(last=False)

    return meta


def _meta_path_for(bvdid: str) -> Path:
    return _company_metadata_dir(bvdid) / META_NAME


def _url_index_path_for(bvdid: str) -> Path:
    return _company_metadata_dir(bvdid) / URL_INDEX_NAME


def _global_state_path() -> Path:
    # IMPORTANT: dynamic output root
    return _output_root() / GLOBAL_STATE_NAME


# ---------------------------------------------------------------------------
# Dataclasses: snapshots for callers (plugin-style, immutable views)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanySnapshot:
    bvdid: str
    name: Optional[str] = None
    root_url: Optional[str] = None
    status: CompanyStatus = COMPANY_STATUS_PENDING
    urls_total: int = 0
    urls_markdown_done: int = 0
    urls_llm_done: int = 0
    last_error: Optional[str] = None
    # terminal-done details (distinct from md/llm done)
    done_reason: Optional[str] = None
    done_details: Optional[Dict[str, Any]] = None
    done_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    pipeline: Optional[str]
    version: Optional[str]
    started_at: str
    total_companies: int
    completed_companies: int
    last_company_bvdid: Optional[str]
    last_updated: Optional[str]


@dataclass(frozen=True)
class _Stmt:
    sql: str
    args: Tuple[Any, ...] = tuple()


def _safe_json_load(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _row_to_company_snapshot(row: sqlite3.Row) -> CompanySnapshot:
    return CompanySnapshot(
        bvdid=row["bvdid"],
        name=row["name"],
        root_url=row["root_url"],
        status=(row["status"] or COMPANY_STATUS_PENDING),
        urls_total=int(row["urls_total"] or 0),
        urls_markdown_done=int(row["urls_markdown_done"] or 0),
        urls_llm_done=int(row["urls_llm_done"] or 0),
        last_error=row["last_error"],
        done_reason=row["done_reason"] if "done_reason" in row.keys() else None,
        done_details=_safe_json_load(
            row["done_details"] if "done_details" in row.keys() else None
        ),
        done_at=row["done_at"] if "done_at" in row.keys() else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_run_snapshot(row: sqlite3.Row) -> RunSnapshot:
    return RunSnapshot(
        run_id=row["run_id"],
        pipeline=row["pipeline"],
        version=row["version"],
        started_at=row["started_at"],
        total_companies=int(row["total_companies"] or 0),
        completed_companies=int(row["completed_companies"] or 0),
        last_company_bvdid=row["last_company_bvdid"],
        last_updated=row["last_updated"],
    )


# ---------------------------------------------------------------------------
# URL-index helpers
# ---------------------------------------------------------------------------


def load_url_index(bvdid: str) -> Dict[str, Any]:
    """
    Public helper to read url_index.json from outputs/{safe}/metadata/url_index.json.
    """
    p = _url_index_path_for(bvdid)
    if not p.exists():
        return {}
    obj = _json_load_nocache(p)
    return obj if isinstance(obj, dict) else {}


def upsert_url_index_entry(bvdid: str, url: str, patch: Dict[str, Any]) -> None:
    """
    Central upsert helper for url_index.json.

    - Merges `patch` into the per-URL entry.
    - Uses a per-file lock to avoid lost updates under concurrency.
    - Writes compact JSON (faster/smaller) and avoids caching the full dict.
    """
    idx_path = _url_index_path_for(bvdid)
    lk = _get_file_lock(idx_path)

    def _update() -> None:
        with lk:
            existing: Dict[str, Any] = {}
            if idx_path.exists():
                loaded = _json_load_nocache(idx_path)
                if isinstance(loaded, dict):
                    existing = loaded

            ent = existing.get(url)
            ent_dict = dict(ent) if isinstance(ent, dict) else {}
            ent_dict.update(patch)
            existing[url] = ent_dict

            _atomic_write_json(idx_path, existing, cache=False, pretty=False)

    _update()


def _classify_url_entry(ent: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    For a single url_index entry, decide:
      - markdown_done: True if Markdown is considered done
      - llm_done:      True if LLM / presence is considered done

    IMPORTANT FIX:
      - Treat legacy/existing `"extracted": 1` as LLM-complete.
        Older url_index writers sometimes set extracted without setting
        status/product_path consistently; global state must still become llm_done.
    """
    status = (ent.get("status") or "").lower()
    has_md_path = bool(ent.get("markdown_path"))

    # Legacy + current LLM artifact keys
    has_llm_artifact = bool(ent.get("json_path") or ent.get("product_path"))

    # Presence-only completion
    presence_checked = bool(ent.get("presence_checked"))

    # Legacy full-extraction boolean (critical for your reported bug)
    extracted_flag = bool(ent.get("extracted"))

    markdown_done = has_md_path or status in _MARKDOWN_COMPLETE_STATUSES
    llm_done = (
        extracted_flag
        or has_llm_artifact
        or status in _LLM_COMPLETE_STATUSES
        or presence_checked
    )
    return markdown_done, llm_done


def _compute_company_stage_from_index(
    index: Dict[str, Any],
) -> Tuple[CompanyStatus, int, int, int]:
    if not index:
        return COMPANY_STATUS_PENDING, 0, 0, 0

    urls_total = 0
    md_done = 0
    llm_done = 0

    for raw_ent in index.values():
        urls_total += 1
        if isinstance(raw_ent, dict):
            m_done, l_done = _classify_url_entry(raw_ent)
            if m_done:
                md_done += 1
            if l_done:
                llm_done += 1

    if urls_total == 0:
        return COMPANY_STATUS_PENDING, 0, 0, 0

    if llm_done == urls_total:
        status: CompanyStatus = COMPANY_STATUS_LLM_DONE
    elif llm_done > 0:
        status = COMPANY_STATUS_LLM_NOT_DONE
    elif md_done == urls_total:
        status = COMPANY_STATUS_MD_DONE
    else:
        status = COMPANY_STATUS_MD_NOT_DONE

    return status, urls_total, md_done, llm_done


def _pending_urls_for_stage(
    index: Dict[str, Any], stage: Literal["markdown", "llm"]
) -> List[str]:
    pending: List[str] = []
    for url, raw_ent in index.items():
        ent = raw_ent if isinstance(raw_ent, dict) else {}
        md_done, llm_done = _classify_url_entry(ent)
        if stage == "markdown":
            if not md_done:
                pending.append(url)
        else:
            if md_done and not llm_done:
                pending.append(url)
    return pending


# ---------------------------------------------------------------------------
# SQLite-backed global state (runs + per-company high level)
# ---------------------------------------------------------------------------


class CrawlState:
    """
    Unified state backend.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.db_path = self.db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
            timeout=5.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def close(self) -> None:
        with suppress(Exception):
            with self._lock:
                self._conn.close()

    def _ensure_company_columns(self) -> None:
        """
        Best-effort schema migration for older DBs.
        """
        with self._lock:
            cols = set()
            try:
                rows = self._conn.execute("PRAGMA table_info(companies)").fetchall()
                cols = {r["name"] for r in rows}
            except Exception:
                cols = set()

            def _add(col_ddl: str, col_name: str) -> None:
                if col_name in cols:
                    return
                with suppress(Exception):
                    self._conn.execute(f"ALTER TABLE companies ADD COLUMN {col_ddl}")

            _add("done_reason TEXT", "done_reason")
            _add("done_details TEXT", "done_details")
            _add("done_at TEXT", "done_at")

    def _init_schema(self) -> None:
        with self._lock:
            with suppress(Exception):
                self._conn.execute("PRAGMA journal_mode=WAL")
            with suppress(Exception):
                self._conn.execute("PRAGMA synchronous=NORMAL")
            with suppress(Exception):
                self._conn.execute("PRAGMA busy_timeout=5000")

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS companies (
                    bvdid TEXT PRIMARY KEY,
                    name TEXT,
                    root_url TEXT,
                    status TEXT,
                    urls_total INTEGER DEFAULT 0,
                    urls_markdown_done INTEGER DEFAULT 0,
                    urls_llm_done INTEGER DEFAULT 0,
                    last_error TEXT,
                    done_reason TEXT,
                    done_details TEXT,
                    done_at TEXT,
                    created_at TEXT,
                    updated_at TEXT
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
                    started_at TEXT,
                    total_companies INTEGER DEFAULT 0,
                    completed_companies INTEGER DEFAULT 0,
                    last_company_bvdid TEXT,
                    last_updated TEXT
                )
                """
            )

        # ensure added columns for older DBs
        self._ensure_company_columns()

    async def _exec(self, stmt: _Stmt) -> None:
        def _run() -> None:
            with self._lock:
                self._conn.execute(stmt.sql, stmt.args)

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

    # ---------- terminal "done" API (sync-safe for retry_state.py) ----------

    def mark_company_terminal_sync(
        self,
        bvdid: str,
        *,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        last_error: Optional[str] = None,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
    ) -> None:
        """
        Mark a company as terminal_done so it stays out of the pending queue.

        This is intentionally synchronous so callers outside asyncio (e.g. retry_state.py)
        can mark terminal without messing with event loops.

        - status           := terminal_done
        - done_reason      := short, stable code / reason (human readable preferred)
        - done_details     := JSON object for debugging/audit (class, attempts, stage, etc.)
        - done_at          := ISO timestamp
        """
        now = _now_iso()
        details_s = (
            json.dumps(details, ensure_ascii=False, separators=(",", ":"))
            if isinstance(details, dict)
            else None
        )
        last_error_trim = (last_error or "")[:4000] if last_error is not None else None

        with self._lock:
            # ensure row exists
            self._conn.execute(
                """
                INSERT OR IGNORE INTO companies (
                    bvdid, name, root_url, status,
                    urls_total, urls_markdown_done, urls_llm_done,
                    last_error, done_reason, done_details, done_at,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, NULL, NULL, NULL, NULL, ?, ?)
                """,
                (bvdid, name, root_url, COMPANY_STATUS_PENDING, now, now),
            )

            sets = [
                "status=?",
                "done_reason=?",
                "done_details=?",
                "done_at=?",
                "updated_at=?",
            ]
            args: List[Any] = [
                COMPANY_STATUS_TERMINAL_DONE,
                (reason or "")[:256],
                details_s,
                now,
                now,
            ]

            if last_error_trim is not None:
                sets.append("last_error=?")
                args.append(last_error_trim)
            if name is not None:
                sets.append("name=?")
                args.append(name)
            if root_url is not None:
                sets.append("root_url=?")
                args.append(root_url)

            args.append(bvdid)
            self._conn.execute(
                f"UPDATE companies SET {', '.join(sets)} WHERE bvdid=?",
                tuple(args),
            )

    async def mark_company_terminal(
        self,
        bvdid: str,
        *,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        last_error: Optional[str] = None,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            self.mark_company_terminal_sync,
            bvdid,
            reason=reason,
            details=details,
            last_error=last_error,
            name=name,
            root_url=root_url,
        )

    def clear_company_terminal_sync(
        self, bvdid: str, *, keep_status: bool = False
    ) -> None:
        """
        Clear terminal_done marker. If current status is terminal_done and keep_status=False,
        reset status to pending.
        """
        now = _now_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM companies WHERE bvdid=?",
                (bvdid,),
            ).fetchone()
            if row is None:
                return
            cur_status = (row["status"] or "").strip()
            new_status = cur_status
            if (not keep_status) and cur_status == COMPANY_STATUS_TERMINAL_DONE:
                new_status = COMPANY_STATUS_PENDING

            self._conn.execute(
                """
                UPDATE companies
                   SET status=?,
                       done_reason=NULL,
                       done_details=NULL,
                       done_at=NULL,
                       updated_at=?
                 WHERE bvdid=?
                """,
                (new_status, now, bvdid),
            )

    async def clear_company_terminal(
        self, bvdid: str, *, keep_status: bool = False
    ) -> None:
        await asyncio.to_thread(
            self.clear_company_terminal_sync, bvdid, keep_status=keep_status
        )

    # ---------- run-level API ----------

    async def start_run(
        self,
        pipeline: str,
        *,
        version: Optional[str] = None,
        args_hash: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        rid = run_id or f"run-{int(time.time())}-{os.getpid()}"
        now = _now_iso()
        await self._exec(
            _Stmt(
                """
                INSERT INTO runs (
                    run_id, pipeline, version, args_hash,
                    started_at, total_companies, completed_companies,
                    last_company_bvdid, last_updated
                )
                VALUES (?, ?, ?, ?, ?, 0, 0, NULL, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    pipeline=excluded.pipeline,
                    version=excluded.version,
                    args_hash=excluded.args_hash,
                    started_at=excluded.started_at,
                    last_updated=excluded.last_updated
                """,
                (rid, pipeline, version or "", args_hash or "", now, now),
            )
        )
        return rid

    async def update_run_totals(
        self,
        run_id: str,
        *,
        total_companies: Optional[int] = None,
        completed_companies: Optional[int] = None,
        last_company_bvdid: Optional[str] = None,
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
        if last_company_bvdid is not None:
            sets.append("last_company_bvdid=?")
            args.append(last_company_bvdid)

        args.append(run_id)
        sql = f"UPDATE runs SET {', '.join(sets)} WHERE run_id=?"
        await self._exec(_Stmt(sql, tuple(args)))

    async def mark_company_completed(self, run_id: str, bvdid: str) -> None:
        now = _now_iso()
        await self._exec(
            _Stmt(
                """
                UPDATE runs
                   SET completed_companies = COALESCE(completed_companies, 0) + 1,
                       last_company_bvdid = ?,
                       last_updated = ?,
                       total_companies = CASE
                           WHEN total_companies < (COALESCE(completed_companies, 0) + 1)
                               THEN (COALESCE(completed_companies, 0) + 1)
                           ELSE total_companies
                       END
                 WHERE run_id = ?
                """,
                (bvdid, now, run_id),
            )
        )

    async def get_run_snapshot(self, run_id: str) -> Optional[RunSnapshot]:
        row = await self._query_one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return _row_to_run_snapshot(row) if row else None

    # ---------- company-level API ----------

    async def upsert_company(
        self,
        bvdid: str,
        *,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
        status: Optional[CompanyStatus] = None,
        urls_total: Optional[int] = None,
        urls_markdown_done: Optional[int] = None,
        urls_llm_done: Optional[int] = None,
        last_error: Optional[str] = None,
        # terminal fields (optional)
        done_reason: Optional[str] = None,
        done_details: Optional[Dict[str, Any]] = None,
        done_at: Optional[str] = None,
    ) -> None:
        now = _now_iso()

        await self._exec(
            _Stmt(
                """
                INSERT OR IGNORE INTO companies (
                    bvdid, name, root_url, status,
                    urls_total, urls_markdown_done, urls_llm_done,
                    last_error, done_reason, done_details, done_at,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, NULL, NULL, NULL, NULL, ?, ?)
                """,
                (bvdid, name, root_url, status or COMPANY_STATUS_PENDING, now, now),
            )
        )

        sets = ["updated_at=?"]
        args: List[Any] = [now]

        if name is not None:
            sets.append("name=?")
            args.append(name)
        if root_url is not None:
            sets.append("root_url=?")
            args.append(root_url)
        if status is not None:
            sets.append("status=?")
            args.append(status)
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
            args.append((last_error or "")[:4000])
        if done_reason is not None:
            sets.append("done_reason=?")
            args.append((done_reason or "")[:256])
        if done_details is not None:
            sets.append("done_details=?")
            args.append(
                json.dumps(done_details, ensure_ascii=False, separators=(",", ":"))
            )
        if done_at is not None:
            sets.append("done_at=?")
            args.append(done_at)

        args.append(bvdid)
        await self._exec(
            _Stmt(f"UPDATE companies SET {', '.join(sets)} WHERE bvdid=?", tuple(args))
        )

    async def recompute_company_from_index(
        self,
        bvdid: str,
        *,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
    ) -> CompanySnapshot:
        # Do NOT overwrite terminal_done (keeps company out of pending queue)
        cur = await self._query_one(
            "SELECT status FROM companies WHERE bvdid=?",
            (bvdid,),
        )
        if cur is not None and (cur["status"] or "") == COMPANY_STATUS_TERMINAL_DONE:
            # still allow updating name/root_url if provided
            if name is not None or root_url is not None:
                await self.upsert_company(bvdid, name=name, root_url=root_url)
            return await self.get_company_snapshot(bvdid, recompute=False)

        index = await asyncio.to_thread(load_url_index, bvdid)
        status, urls_total, md_done, llm_done = _compute_company_stage_from_index(index)
        now = _now_iso()

        await self._exec(
            _Stmt(
                """
                INSERT INTO companies (
                    bvdid, name, root_url, status,
                    urls_total, urls_markdown_done, urls_llm_done,
                    last_error, done_reason, done_details, done_at,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(bvdid) DO UPDATE SET
                    name      = COALESCE(excluded.name, companies.name),
                    root_url  = COALESCE(excluded.root_url, companies.root_url),
                    status    = CASE
                        WHEN companies.status = 'terminal_done' THEN companies.status
                        ELSE excluded.status
                    END,
                    urls_total = excluded.urls_total,
                    urls_markdown_done = excluded.urls_markdown_done,
                    urls_llm_done = excluded.urls_llm_done,
                    updated_at = excluded.updated_at
                """,
                (
                    bvdid,
                    name,
                    root_url,
                    status,
                    urls_total,
                    md_done,
                    llm_done,
                    now,
                    now,
                ),
            )
        )
        return await self.get_company_snapshot(bvdid, recompute=False)

    async def set_company_error(self, bvdid: str, reason: str) -> None:
        await self.upsert_company(bvdid, last_error=reason)

    async def get_company_snapshot(
        self, bvdid: str, *, recompute: bool = True
    ) -> CompanySnapshot:
        row = await self._query_one("SELECT * FROM companies WHERE bvdid=?", (bvdid,))
        if row is not None:
            # terminal_done should always short-circuit recompute
            if (row["status"] or "") == COMPANY_STATUS_TERMINAL_DONE:
                return _row_to_company_snapshot(row)
            if recompute:
                # optionally recompute (but won't overwrite terminal due to checks)
                return await self.recompute_company_from_index(bvdid)
            return _row_to_company_snapshot(row)

        if recompute:
            return await self.recompute_company_from_index(bvdid)

        # create minimal row
        now = _now_iso()
        await self._exec(
            _Stmt(
                """
                INSERT OR IGNORE INTO companies (
                    bvdid, status, urls_total, urls_markdown_done,
                    urls_llm_done, created_at, updated_at
                )
                VALUES (?, ?, 0, 0, 0, ?, ?)
                """,
                (bvdid, COMPANY_STATUS_PENDING, now, now),
            )
        )
        row = await self._query_one("SELECT * FROM companies WHERE bvdid=?", (bvdid,))
        return _row_to_company_snapshot(row)  # type: ignore[arg-type]

    async def list_companies(self) -> List[CompanySnapshot]:
        rows = await self._query_all(
            "SELECT * FROM companies ORDER BY created_at, bvdid", tuple()
        )
        return [_row_to_company_snapshot(r) for r in rows]

    async def get_companies_by_ids(self, bvdids: List[str]) -> List[CompanySnapshot]:
        if not bvdids:
            return []

        placeholders = ",".join("?" for _ in bvdids)
        rows = await self._query_all(
            f"SELECT * FROM companies WHERE bvdid IN ({placeholders})",
            tuple(bvdids),
        )
        by_id = {r["bvdid"]: _row_to_company_snapshot(r) for r in rows}
        return [by_id[b] for b in bvdids if b in by_id]

    async def recompute_global_state(self) -> Dict[str, Any]:
        """
        Recompute and persist a global JSON overview of crawl progress.

        Output file: OUTPUT_ROOT / 'crawl_global_state.json'
        """
        path = _global_state_path()

        def _compute_and_write() -> Dict[str, Any]:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT bvdid, status, urls_total, urls_markdown_done, urls_llm_done,
                           done_reason, done_at
                      FROM companies
                    """
                ).fetchall()
                run_row = self._conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
                ).fetchone()

            total = len(rows)
            by_status: Dict[str, int] = {
                COMPANY_STATUS_PENDING: 0,
                COMPANY_STATUS_MD_NOT_DONE: 0,
                COMPANY_STATUS_MD_DONE: 0,
                COMPANY_STATUS_LLM_NOT_DONE: 0,
                COMPANY_STATUS_LLM_DONE: 0,
                COMPANY_STATUS_TERMINAL_DONE: 0,
            }

            pending: List[str] = []
            in_progress: List[str] = []
            done: List[str] = []
            terminal_done: List[str] = []

            terminal_reasons: Dict[str, int] = {}

            for r in rows:
                st = r["status"] or COMPANY_STATUS_PENDING
                b = r["bvdid"]
                by_status[st] = by_status.get(st, 0) + 1

                if st == COMPANY_STATUS_PENDING:
                    pending.append(b)
                elif st in (COMPANY_STATUS_MD_NOT_DONE, COMPANY_STATUS_LLM_NOT_DONE):
                    in_progress.append(b)
                elif st == COMPANY_STATUS_TERMINAL_DONE:
                    done.append(b)
                    terminal_done.append(b)
                    dr = (r["done_reason"] or "unknown").strip()
                    terminal_reasons[dr] = terminal_reasons.get(dr, 0) + 1
                else:
                    done.append(b)

            crawled = total - len(pending)
            completed = len(done)
            percentage_completed = (completed / total * 100.0) if total else 0.0

            latest_run: Optional[Dict[str, Any]] = None
            if run_row is not None:
                latest_run = {
                    "run_id": run_row["run_id"],
                    "pipeline": run_row["pipeline"],
                    "version": run_row["version"],
                    "started_at": run_row["started_at"],
                    "total_companies": int(run_row["total_companies"] or 0),
                    "completed_companies": int(run_row["completed_companies"] or 0),
                    "last_company_bvdid": run_row["last_company_bvdid"],
                    "last_updated": run_row["last_updated"],
                }

            payload: Dict[str, Any] = {
                "generated_at": _now_iso(),
                "output_root": str(_output_root()),
                "db_path": str(self.db_path),
                "total_companies": total,
                "crawled_companies": crawled,
                "completed_companies": completed,
                "percentage_completed": percentage_completed,
                "by_status": by_status,
                "pending_companies": pending,
                "in_progress_companies": in_progress,
                "done_companies": done,
                "terminal_done_companies": terminal_done,
                "terminal_done_by_reason": terminal_reasons,
                "latest_run": latest_run,
            }

            _atomic_write_json(path, payload, cache=False, pretty=True)
            return payload

        return await asyncio.to_thread(_compute_and_write)

    async def get_pending_urls_for_markdown(self, bvdid: str) -> List[str]:
        snap = await self.get_company_snapshot(bvdid, recompute=False)
        if snap.status == COMPANY_STATUS_TERMINAL_DONE:
            return []
        index = await asyncio.to_thread(load_url_index, bvdid)
        return (
            _pending_urls_for_stage(index, "markdown")
            if isinstance(index, dict)
            else []
        )

    async def get_pending_urls_for_llm(self, bvdid: str) -> List[str]:
        snap = await self.get_company_snapshot(bvdid, recompute=False)
        if snap.status == COMPANY_STATUS_TERMINAL_DONE:
            return []
        index = await asyncio.to_thread(load_url_index, bvdid)
        return _pending_urls_for_stage(index, "llm") if isinstance(index, dict) else []

    async def mark_url_failed(self, bvdid: str, url: str, reason: str) -> None:
        def _update() -> None:
            patch: Dict[str, Any] = {
                "failed": True,
                "failed_reason": reason,
                "failed_at": _now_iso(),
            }
            upsert_url_index_entry(bvdid, url, patch)

        await asyncio.to_thread(_update)

    async def read_last_crawl_date(self, bvdid: str) -> Optional[datetime]:
        meta_p = _meta_path_for(bvdid)

        def _load() -> Optional[datetime]:
            if not meta_p.exists():
                return None
            try:
                obj = _json_load_cached(meta_p)
                if not isinstance(obj, dict):
                    return None
                v = obj.get("last_crawled_at")
                return datetime.fromisoformat(v) if v else None
            except Exception:
                return None

        return await asyncio.to_thread(_load)

    async def write_last_crawl_date(
        self, bvdid: str, dt: Optional[datetime] = None
    ) -> None:
        meta_p = _meta_path_for(bvdid)

        def _write() -> None:
            new_ts = (dt or datetime.now(timezone.utc)).isoformat()
            payload: Dict[str, Any] = {"last_crawled_at": new_ts}
            if meta_p.exists():
                try:
                    existing = _json_load_cached(meta_p)
                    if isinstance(existing, dict):
                        existing["last_crawled_at"] = new_ts
                        payload = existing
                except Exception:
                    pass
            _atomic_write_json(meta_p, payload, cache=True, pretty=True)

        await asyncio.to_thread(_write)


# ---------------------------------------------------------------------------
# Plugin-style accessor (single backend instance)
# ---------------------------------------------------------------------------

_default_crawl_state: Optional[CrawlState] = None


def get_crawl_state(db_path: Optional[Path] = None) -> CrawlState:
    """
    Return the default CrawlState backend (singleton).

    IMPORTANT FIX:
      - If db_path is provided and differs from the current singleton DB,
        close and recreate the backend. This avoids cross-outdir corruption.
      - If db_path is None, compute default dynamically from output_paths.OUTPUT_ROOT.
    """
    global _default_crawl_state

    want = Path(db_path) if db_path is not None else default_db_path()
    want = want.resolve()

    if _default_crawl_state is None:
        _default_crawl_state = CrawlState(want)
        return _default_crawl_state

    try:
        cur = Path(getattr(_default_crawl_state, "db_path", "")).resolve()
    except Exception:
        cur = want

    if cur != want:
        with suppress(Exception):
            _default_crawl_state.close()
        _default_crawl_state = CrawlState(want)

    return _default_crawl_state


__all__ = [
    "CrawlState",
    "CompanySnapshot",
    "RunSnapshot",
    "CompanyStatus",
    "COMPANY_STATUS_PENDING",
    "COMPANY_STATUS_MD_NOT_DONE",
    "COMPANY_STATUS_MD_DONE",
    "COMPANY_STATUS_LLM_NOT_DONE",
    "COMPANY_STATUS_LLM_DONE",
    "COMPANY_STATUS_TERMINAL_DONE",
    "get_crawl_state",
    "load_url_index",
    "upsert_url_index_entry",
    "default_db_path",
    "DEFAULT_DB",
]
