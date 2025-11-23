from __future__ import annotations

import asyncio
import errno
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from extensions.output_paths import (
    OUTPUT_ROOT as OUTPUTS_DIR,
    sanitize_bvdid,
    ensure_company_dirs,
)

# ---------------------------------------------------------------------------
# Atomic file I/O + small JSON cache (shared for meta + url_index + global)
# ---------------------------------------------------------------------------

_JSON_CACHE: Dict[Path, Tuple[float, Any]] = {}


def _retry_emfile(fn, attempts: int = 6, base_delay: float = 0.15):
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            if e.errno == errno.EMFILE or "Too many open files" in str(e):
                time.sleep(base_delay * (2 ** i))
                continue
            raise


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """
    Robust atomic write that:
      * Writes to a unique temp file in the same dir
      * fsyncs it (if available)
      * Replaces the target
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    tmp = Path(f"{str(path)}.tmp.{stamp}")

    def _write() -> None:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(data)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # fsync not always available; best-effort
                pass
        os.replace(tmp, path)

    _retry_emfile(_write)


def _atomic_write_json(path: Path, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    _atomic_write_text(path, payload, "utf-8")
    _json_update_cache(path, obj)


def _atomic_write_json_nocache(path: Path, obj: Any) -> None:
    """
    Atomic JSON write that does NOT update _JSON_CACHE.
    Use for large, frequently-updated files such as url_index.json.
    """
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    _atomic_write_text(path, payload, "utf-8")
    # intentionally do NOT call _json_update_cache


def _json_load_cached(path: Path) -> Any:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}

    cached = _JSON_CACHE.get(path)
    if cached is not None:
        cached_mtime, obj = cached
        if cached_mtime == mtime:
            return obj

    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
    except Exception:
        obj = {}

    _JSON_CACHE[path] = (mtime, obj)
    return obj


def _json_load_nocache(path: Path) -> Any:
    """
    Load JSON from disk without touching the global _JSON_CACHE.
    Intended for large per-company manifests like url_index.json.
    """

    def _read() -> Any:
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return {}

    return _retry_emfile(_read)


def _json_update_cache(path: Path, obj: Any) -> None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return
    _JSON_CACHE[path] = (mtime, obj)


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

META_NAME = "crawl_meta.json"
URL_INDEX_NAME = "url_index.json"
GLOBAL_STATE_NAME = "crawl_global_state.json"
DEFAULT_DB = OUTPUTS_DIR / "crawl_state.sqlite3"

CompanyStatus = Literal[
    "pending",
    "markdown_not_done",
    "markdown_done",
    "llm_not_done",
    "llm_done",
]

COMPANY_STATUS_PENDING: CompanyStatus = "pending"
COMPANY_STATUS_MD_NOT_DONE: CompanyStatus = "markdown_not_done"
COMPANY_STATUS_MD_DONE: CompanyStatus = "markdown_done"
COMPANY_STATUS_LLM_NOT_DONE: CompanyStatus = "llm_not_done"
COMPANY_STATUS_LLM_DONE: CompanyStatus = "llm_done"

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
    """
    dirs = ensure_company_dirs(bvdid)
    meta = dirs.get("metadata") or dirs.get("checkpoints")
    if meta is None:
        safe = sanitize_bvdid(bvdid)
        meta = OUTPUTS_DIR / safe / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
    return meta


def _meta_path_for(bvdid: str) -> Path:
    return _company_metadata_dir(bvdid) / META_NAME


def _url_index_path_for(bvdid: str) -> Path:
    return _company_metadata_dir(bvdid) / URL_INDEX_NAME


def _global_state_path() -> Path:
    return OUTPUTS_DIR / GLOBAL_STATE_NAME


# ---------------------------------------------------------------------------
# Dataclasses: snapshots for callers (plugin-style, immutable views)
# ---------------------------------------------------------------------------


@dataclass
class CompanySnapshot:
    bvdid: str
    name: Optional[str] = None
    root_url: Optional[str] = None
    status: CompanyStatus = COMPANY_STATUS_PENDING
    urls_total: int = 0
    urls_markdown_done: int = 0
    urls_llm_done: int = 0
    last_error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class RunSnapshot:
    run_id: str
    pipeline: Optional[str]
    version: Optional[str]
    started_at: str
    total_companies: int
    completed_companies: int
    last_company_bvdid: Optional[str]
    last_updated: Optional[str]


@dataclass
class _Stmt:
    sql: str
    args: Tuple[Any, ...] = tuple()


# ---------------------------------------------------------------------------
# URL-index helpers
# ---------------------------------------------------------------------------


def load_url_index(bvdid: str) -> Dict[str, Any]:
    """
    Public helper to read url_index.json from outputs/{safe}/metadata/url_index.json.
    This is the single entry point in the state module for URL manifest access.
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
    - Creates url_index.json or the URL entry if missing.
    - Intended to be used by:
        * page processing (HTML/markdown)
        * LLM passes
        * filters/scorers (BM25, etc.)
    """
    idx_path = _url_index_path_for(bvdid)

    def _update() -> None:
        existing: Dict[str, Any] = {}
        if idx_path.exists():
            try:
                # NOTE: do NOT use _json_load_cached here;
                # url_index.json may be huge and we don't want it in _JSON_CACHE.
                existing = _json_load_nocache(idx_path)
            except Exception:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}

        ent = dict(existing.get(url, {}))
        ent.update(patch)
        existing[url] = ent

        # Use the no-cache writer to avoid pinning this large dict in memory
        _atomic_write_json_nocache(idx_path, existing)

    _update()


def _classify_url_entry(ent: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    For a single url_index entry, decide:
      - markdown_done: True if Markdown is considered done
      - llm_done:      True if LLM / presence is considered done

    Based on fields used in run_utils.upsert_url_index.
    """
    status = (ent.get("status") or "").lower()
    has_md_path = bool(ent.get("markdown_path"))
    # Support both legacy 'json_path' and current 'product_path' key
    has_llm_artifact = bool(ent.get("json_path") or ent.get("product_path"))
    presence_checked = bool(ent.get("presence_checked"))

    markdown_done = has_md_path or status in _MARKDOWN_COMPLETE_STATUSES
    # We treat either full LLM extraction or presence-only as LLM stage completion.
    llm_done = has_llm_artifact or status in _LLM_COMPLETE_STATUSES or presence_checked
    return markdown_done, llm_done


def _compute_company_stage_from_index(
    index: Dict[str, Any]
) -> Tuple[CompanyStatus, int, int, int]:
    """
    Given the full url_index dict for a company, compute:
      - company-level status in the limited enum
      - urls_total
      - urls_markdown_done
      - urls_llm_done
    """
    if not index:
        return COMPANY_STATUS_PENDING, 0, 0, 0

    urls_total = 0
    md_done = 0
    llm_done = 0

    for _, raw_ent in index.items():
        if not isinstance(raw_ent, dict):
            # tolerate legacy / unexpected shapes
            urls_total += 1
            continue
        urls_total += 1
        m_done, l_done = _classify_url_entry(raw_ent)
        if m_done:
            md_done += 1
        if l_done:
            llm_done += 1

    if urls_total == 0:
        return COMPANY_STATUS_PENDING, 0, 0, 0

    # Stage precedence:
    # 1) llm_done (all URLs)
    # 2) llm_not_done (some but not all)
    # 3) markdown_done (all URLs markdown-complete but no llm)
    # 4) markdown_not_done (anything else with URLs discovered)
    if llm_done == urls_total:
        status: CompanyStatus = COMPANY_STATUS_LLM_DONE
    elif llm_done > 0:
        status = COMPANY_STATUS_LLM_NOT_DONE
    elif md_done == urls_total:
        status = COMPANY_STATUS_MD_DONE
    elif md_done > 0 or urls_total > 0:
        # URLs discovered but markdown pipeline not fully completed
        status = COMPANY_STATUS_MD_NOT_DONE
    else:
        status = COMPANY_STATUS_PENDING

    return status, urls_total, md_done, llm_done


def _pending_urls_for_stage(
    index: Dict[str, Any], stage: Literal["markdown", "llm"]
) -> List[str]:
    """
    Compute URLs that are *not done* for the given stage, using url_index only.

    - For 'markdown': URLs where markdown is not considered done.
    - For 'llm':      URLs where markdown is done but llm/presence is not.
    """
    pending: List[str] = []
    for url, raw_ent in index.items():
        if not isinstance(raw_ent, dict):
            raw_ent = {}
        md_done, llm_done = _classify_url_entry(raw_ent)
        if stage == "markdown":
            if not md_done:
                pending.append(url)
        else:
            # For LLM we only consider URLs that are already markdown-complete
            if md_done and not llm_done:
                pending.append(url)
    return pending


# ---------------------------------------------------------------------------
# SQLite-backed global state (runs + per-company high level)
# ---------------------------------------------------------------------------


class CrawlState:
    """
    Unified state backend (plugin-style):

    * Owns per-company high-level state (status + URL counts) derived from url_index.json.
    * Owns global run-level state (run_id, counters, timings, version).
    * Exposes a clean async API for callers in the pipeline.

    All detailed per-URL status is still persisted in url_index.json (written elsewhere),
    but this module is the *only* place that interprets that for global planning.
    """

    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    # ---------- schema ----------

    def _init_schema(self) -> None:
        with self._lock:
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
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")

    # ---------- tiny async executors ----------

    async def _exec(self, stmt: _Stmt) -> None:
        def _run() -> None:
            with self._lock:
                self._conn.execute(stmt.sql, stmt.args)

        await asyncio.to_thread(_run)

    async def _query_one(
        self, sql: str, args: Tuple[Any, ...]
    ) -> Optional[sqlite3.Row]:
        def _run():
            with self._lock:
                cur = self._conn.execute(sql, args)
                return cur.fetchone()

        return await asyncio.to_thread(_run)

    async def _query_all(self, sql: str, args: Tuple[Any, ...]) -> List[sqlite3.Row]:
        def _run():
            with self._lock:
                cur = self._conn.execute(sql, args)
                return cur.fetchall()

        return await asyncio.to_thread(_run)

    # ---------- run-level API ----------

    async def start_run(
        self,
        pipeline: str,
        *,
        version: Optional[str] = None,
        args_hash: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Create (or re-init) a run row. Returns the run_id.
        """
        rid = run_id or f"run-{int(time.time())}-{os.getpid()}"
        now = datetime.now(timezone.utc).isoformat()
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
        now = datetime.now(timezone.utc).isoformat()
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
        """
        Convenience: increment completed_companies and set last_company_bvdid.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _run() -> None:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT completed_companies FROM runs WHERE run_id=?", (run_id,)
                )
                row = cur.fetchone()
                completed = int(row["completed_companies"] or 0) + 1 if row else 1
                self._conn.execute(
                    """
                    UPDATE runs
                       SET completed_companies=?,
                           last_company_bvdid=?,
                           last_updated=?,
                           total_companies = CASE
                               WHEN total_companies < ? THEN ?
                               ELSE total_companies
                           END
                     WHERE run_id=?
                    """,
                    (completed, bvdid, now, completed, completed, run_id),
                )

        await asyncio.to_thread(_run)

    async def get_run_snapshot(self, run_id: str) -> Optional[RunSnapshot]:
        row = await self._query_one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not row:
            return None
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
    ) -> None:
        """
        Lightweight upsert that updates only provided fields.
        Use `recompute_company_from_index` when you want the status derived
        from url_index.json instead of setting it manually.
        """
        now = datetime.now(timezone.utc).isoformat()
        # Build dynamic update
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
            args.append(last_error)

        sets_clause = ", ".join(sets)

        async def _ensure_row() -> None:
            # Ensure row exists first
            existing = await self._query_one(
                "SELECT bvdid FROM companies WHERE bvdid=?", (bvdid,)
            )
            if not existing:
                created_at = now
                await self._exec(
                    _Stmt(
                        """
                        INSERT INTO companies (
                            bvdid, name, root_url, status,
                            urls_total, urls_markdown_done, urls_llm_done,
                            last_error, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, 0, 0, 0, NULL, ?, ?)
                        """,
                        (
                            bvdid,
                            name,
                            root_url,
                            status or COMPANY_STATUS_PENDING,
                            created_at,
                            now,
                        ),
                    )
                )

        await _ensure_row()
        args.append(bvdid)
        await self._exec(
            _Stmt(f"UPDATE companies SET {sets_clause} WHERE bvdid=?", tuple(args))
        )

    async def recompute_company_from_index(
        self,
        bvdid: str,
        *,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
    ) -> CompanySnapshot:
        """
        Read url_index.json, derive (status, url counts) and upsert into the DB.
        This is the canonical way to sync company-level status with per-URL state.
        """
        index = await asyncio.to_thread(load_url_index, bvdid)
        status, urls_total, md_done, llm_done = _compute_company_stage_from_index(index)
        now = datetime.now(timezone.utc).isoformat()

        await self._exec(
            _Stmt(
                """
                INSERT INTO companies (
                    bvdid, name, root_url, status,
                    urls_total, urls_markdown_done, urls_llm_done,
                    last_error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(bvdid) DO UPDATE SET
                    name      = COALESCE(?, companies.name),
                    root_url  = COALESCE(?, companies.root_url),
                    status    = excluded.status,
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
                    # DO UPDATE COALESCE params
                    name,
                    root_url,
                ),
            )
        )
        snap = await self.get_company_snapshot(bvdid, recompute=False)
        # get_company_snapshot always returns non-None for an existing row
        return snap  # type: ignore[return-value]

    async def set_company_error(self, bvdid: str, reason: str) -> None:
        await self.upsert_company(bvdid, last_error=reason)

    async def get_company_snapshot(
        self,
        bvdid: str,
        *,
        recompute: bool = True,
    ) -> CompanySnapshot:
        """
        Get an immutable snapshot of company state.

        If no DB row exists and recompute=True, we compute status from url_index.json.
        If still nothing, we create a minimal 'pending' row.
        """
        row = await self._query_one("SELECT * FROM companies WHERE bvdid=?", (bvdid,))
        if row is None and recompute:
            # This also ensures a row exists afterwards
            return await self.recompute_company_from_index(bvdid)

        if row is None:
            now = datetime.now(timezone.utc).isoformat()
            await self._exec(
                _Stmt(
                    """
                    INSERT INTO companies (
                        bvdid, status, urls_total, urls_markdown_done,
                        urls_llm_done, created_at, updated_at
                    )
                    VALUES (?, ?, 0, 0, 0, ?, ?)
                    """,
                    (bvdid, COMPANY_STATUS_PENDING, now, now),
                )
            )
            row = await self._query_one(
                "SELECT * FROM companies WHERE bvdid=?", (bvdid,)
            )

        return CompanySnapshot(
            bvdid=row["bvdid"],
            name=row["name"],
            root_url=row["root_url"],
            status=(row["status"] or COMPANY_STATUS_PENDING),
            urls_total=int(row["urls_total"] or 0),
            urls_markdown_done=int(row["urls_markdown_done"] or 0),
            urls_llm_done=int(row["urls_llm_done"] or 0),
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def list_companies(self) -> List[CompanySnapshot]:
        """
        Return all known companies in the DB as snapshots.
        """
        rows = await self._query_all(
            "SELECT * FROM companies ORDER BY created_at, bvdid", tuple()
        )
        out: List[CompanySnapshot] = []
        for row in rows:
            out.append(
                CompanySnapshot(
                    bvdid=row["bvdid"],
                    name=row["name"],
                    root_url=row["root_url"],
                    status=(row["status"] or COMPANY_STATUS_PENDING),
                    urls_total=int(row["urls_total"] or 0),
                    urls_markdown_done=int(row["urls_markdown_done"] or 0),
                    urls_llm_done=int(row["urls_llm_done"] or 0),
                    last_error=row["last_error"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return out

    async def recompute_global_state(self) -> Dict[str, Any]:
        """
        Recompute and persist a global JSON overview of crawl progress.

        Output file: OUTPUT_ROOT / 'crawl_global_state.json'

        Includes:
          - total_companies
          - crawled_companies (status != pending)
          - completed_companies (markdown_done or llm_done)
          - percentage_completed
          - per-status counts
          - lists of pending / in-progress / done bvdid
          - latest run summary (from runs table, if any)

        NOTE: This is always a snapshot of the *current* DB state.
        It overwrites the JSON file each time, but does NOT reset any DB state,
        so progress from previous runs is preserved.
        """

        path = _global_state_path()

        def _compute_and_write() -> Dict[str, Any]:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT bvdid, status, urls_total, "
                    "urls_markdown_done, urls_llm_done "
                    "FROM companies"
                )
                rows = cur.fetchall()
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
            }

            pending: List[str] = []
            in_progress: List[str] = []
            done: List[str] = []

            for r in rows:
                st = (r["status"] or COMPANY_STATUS_PENDING)
                b = r["bvdid"]
                if st not in by_status:
                    by_status[st] = by_status.get(st, 0) + 1
                else:
                    by_status[st] += 1

                if st == COMPANY_STATUS_PENDING:
                    pending.append(b)
                elif st in (COMPANY_STATUS_MD_NOT_DONE, COMPANY_STATUS_LLM_NOT_DONE):
                    in_progress.append(b)
                elif st in (COMPANY_STATUS_MD_DONE, COMPANY_STATUS_LLM_DONE):
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
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_companies": total,
                "crawled_companies": crawled,
                "completed_companies": completed,
                "percentage_completed": percentage_completed,
                "by_status": by_status,
                "pending_companies": pending,
                "in_progress_companies": in_progress,
                "done_companies": done,
                "latest_run": latest_run,
            }

            _atomic_write_json(path, payload)
            return payload

        return await asyncio.to_thread(_compute_and_write)

    # ---------- URL-level helpers (derived from url_index) ----------

    async def get_pending_urls_for_markdown(self, bvdid: str) -> List[str]:
        """
        URLs that still need Markdown (markdown_not_done).
        """
        index = await asyncio.to_thread(load_url_index, bvdid)
        if not isinstance(index, dict):
            return []
        return _pending_urls_for_stage(index, "markdown")

    async def get_pending_urls_for_llm(self, bvdid: str) -> List[str]:
        """
        URLs that still need LLM/presence (llm_not_done):
          - Markdown considered done
          - LLM/presence not yet considered done
        """
        index = await asyncio.to_thread(load_url_index, bvdid)
        if not isinstance(index, dict):
            return []
        return _pending_urls_for_stage(index, "llm")

    async def mark_url_failed(self, bvdid: str, url: str, reason: str) -> None:
        """
        Best-effort helper to mirror a failure into url_index.json.
        Does *not* modify the main status field; it only adds dedicated failure
        metadata so you can inspect or filter later.
        """

        def _update() -> None:
            patch: Dict[str, Any] = {
                "failed": True,
                "failed_reason": reason,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            upsert_url_index_entry(bvdid, url, patch)

        await asyncio.to_thread(_update)

    # ---------- last_crawled_at helpers (company-level meta) ----------

    async def read_last_crawl_date(self, bvdid: str) -> Optional[datetime]:
        """
        Read last_crawled_at from crawl_meta.json for the company, if present.
        """
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
        """
        Update last_crawled_at in crawl_meta.json. Other keys are preserved.
        """
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
            _atomic_write_json(meta_p, payload)

        await asyncio.to_thread(_write)


# ---------------------------------------------------------------------------
# Plugin-style accessor (single backend instance)
# ---------------------------------------------------------------------------

_default_crawl_state: Optional[CrawlState] = None


def get_crawl_state(db_path: Optional[Path] = None) -> CrawlState:
    """
    Return the default CrawlState backend (singleton).
    Other modules should depend on this function rather than constructing
    CrawlState directly, to keep the plugin-style boundary clean.
    """
    global _default_crawl_state
    if _default_crawl_state is None:
        _default_crawl_state = CrawlState(db_path or DEFAULT_DB)
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
    "get_crawl_state",
    "load_url_index",
    "upsert_url_index_entry",
]