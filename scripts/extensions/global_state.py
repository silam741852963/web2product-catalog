from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List

DEFAULT_DB = Path("outputs") / "global_state.sqlite3"


@dataclass
class _Stmt:
    sql: str
    args: Tuple[Any, ...] = tuple()


@dataclass
class ResumePlan:
    # URL source decision
    use_index: bool
    skip_seeding_entirely: bool
    url_source: str  # "index" | "network"

    # Stage skipping
    skip_html: bool
    skip_markdown: bool
    skip_llm: bool

    # Hints (for logging/planning)
    local_html_expected: bool
    local_md_expected: bool

    # Explanation for logs
    reason: str = "no-skip"


class GlobalState:
    """
    Small async-friendly SQLite layer storing per-company state used to decide resume/skip
    without scanning company folders.

    Table: companies
      bvdid TEXT PRIMARY KEY
      name TEXT
      root_url TEXT
      stage TEXT
      status TEXT                   -- pending | running | paused_net | done | failed
      created_at TEXT               -- ISO
      updated_at TEXT               -- ISO
      last_seen TEXT                -- ISO
      fail_reason TEXT              -- nullable
      resume_mode TEXT              -- 'url_index' | 'artifacts' | NULL

      -- Totals for planning/skips
      urls_total INTEGER DEFAULT 0  -- total URLs we intend to process (from last plan)
      urls_done  INTEGER DEFAULT 0
      urls_failed INTEGER DEFAULT 0

      -- URL index (seed manifest) knowledge
      has_url_index INTEGER DEFAULT 0  -- 1 if url_index.json exists (even if it has 0 URLs)
      seeded_urls   INTEGER DEFAULT 0  -- number of URLs in url_index.json (last recorded)

      -- Artifact counters (rolled-up across runs; used for skip decisions of HTML/LLM)
      saved_html_total INTEGER DEFAULT 0
      saved_md_total   INTEGER DEFAULT 0
      saved_json_total INTEGER DEFAULT 0
    """

    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    # ---------- schema ----------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    bvdid TEXT PRIMARY KEY,
                    name TEXT,
                    root_url TEXT,
                    stage TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    last_seen TEXT,
                    fail_reason TEXT,
                    urls_total INTEGER DEFAULT 0,
                    urls_done INTEGER DEFAULT 0,
                    urls_failed INTEGER DEFAULT 0,
                    resume_mode TEXT,
                    has_url_index INTEGER DEFAULT 0,
                    seeded_urls INTEGER DEFAULT 0,
                    saved_html_total INTEGER DEFAULT 0,
                    saved_md_total INTEGER DEFAULT 0,
                    saved_json_total INTEGER DEFAULT 0
                )
            """)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # Defensive migrations if table already existed
            self._migrate_columns({
                "root_url": "TEXT",
                "stage": "TEXT",
                "status": "TEXT",
                "created_at": "TEXT",
                "updated_at": "TEXT",
                "last_seen": "TEXT",
                "fail_reason": "TEXT",
                "urls_total": "INTEGER DEFAULT 0",
                "urls_done": "INTEGER DEFAULT 0",
                "urls_failed": "INTEGER DEFAULT 0",
                "resume_mode": "TEXT",
                "has_url_index": "INTEGER DEFAULT 0",
                "seeded_urls": "INTEGER DEFAULT 0",
                "saved_html_total": "INTEGER DEFAULT 0",
                "saved_md_total": "INTEGER DEFAULT 0",
                "saved_json_total": "INTEGER DEFAULT 0",
            })

    def _migrate_columns(self, required: Dict[str, str]) -> None:
        cur = self._conn.execute("PRAGMA table_info(companies)")
        existing = {row["name"] for row in cur.fetchall()}
        for col, coltype in required.items():
            if col not in existing:
                self._conn.execute(f"ALTER TABLE companies ADD COLUMN {col} {coltype}")

    # ---------- tiny async executor ----------

    async def _exec(self, stmt: _Stmt) -> None:
        def _run():
            with self._lock:
                self._conn.execute(stmt.sql, stmt.args)
        await asyncio.to_thread(_run)

    async def _query_one(self, sql: str, args: Tuple[Any, ...]) -> Optional[sqlite3.Row]:
        def _run():
            with self._lock:
                cur = self._conn.execute(sql, args)
                return cur.fetchone()
        return await asyncio.to_thread(_run)

    # ---------- public API (mutations) ----------

    async def upsert_company(self, bvdid: str, name: str, root_url: str, *, stage: str, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._exec(_Stmt("""
            INSERT INTO companies (bvdid, name, root_url, stage, status, created_at, updated_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bvdid) DO UPDATE SET
                name=excluded.name,
                root_url=excluded.root_url,
                stage=excluded.stage,
                status=excluded.status,
                updated_at=excluded.updated_at,
                last_seen=excluded.last_seen
        """, (bvdid, name, root_url, stage, status, now, now, now)))

    async def mark_in_progress(self, bvdid: str, *, stage: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if stage:
            await self._exec(_Stmt("UPDATE companies SET status='running', stage=?, updated_at=?, last_seen=? WHERE bvdid=?",
                                   (stage, now, now, bvdid)))
        else:
            await self._exec(_Stmt("UPDATE companies SET status='running', updated_at=?, last_seen=? WHERE bvdid=?",
                                   (now, now, bvdid)))

    async def mark_paused_net(self, bvdid: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._exec(_Stmt("UPDATE companies SET status='paused_net', updated_at=?, last_seen=? WHERE bvdid=?",
                               (now, now, bvdid)))

    async def mark_done(self, bvdid: str, stage: Optional[str] = None, *, presence_only: bool = False) -> None:
        """
        Mark the company as done. If `stage` is provided, update the stored stage too.
        Also clear any resume_mode sentinel so future runs won't incorrectly prefer
        the url_index-only resume path once we're done.

        If the completed stage includes 'llm' and presence_only is False, we also
        set all artifact counters to match urls_total (so next runs know html/markdown/llm
        are all complete). If presence_only is True (i.e. the run was presence-only),
        we do NOT mark the llm artifacts as complete — leave saved_json_total alone
        so a later non-presence LLM run will actually run.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Lookup current totals (best-effort)
        row = await self._query_one("SELECT urls_total FROM companies WHERE bvdid=?", (bvdid,))
        urls_total = int((row["urls_total"] if row and row["urls_total"] is not None else 0))

        set_all_complete = False
        set_md_only = False
        if stage:
            stage_lower = stage.lower()
            if "llm" in stage_lower:
                # If this was an LLM pipeline run, only set all-complete if it was a
                # full LLM run (not presence-only). If presence_only is True, only
                # mark up to markdown completion.
                if presence_only:
                    set_md_only = True
                else:
                    set_all_complete = True

        if set_all_complete:
            await self._exec(_Stmt(""" 
                UPDATE companies
                   SET status='done',
                       stage=?,
                       resume_mode=NULL,
                       has_url_index=1,
                       saved_html_total=?,
                       saved_md_total=?,
                       saved_json_total=?,
                       updated_at=?,
                       last_seen=?
                 WHERE bvdid=?
            """, (stage, urls_total, urls_total, urls_total, now, now, bvdid)))
        elif set_md_only:
            # Mark html/markdown as complete, but leave saved_json_total as-is.
            await self._exec(_Stmt("""
                UPDATE companies
                   SET status='done',
                       stage=?,
                       resume_mode=NULL,
                       has_url_index=1,
                       saved_html_total=?,
                       saved_md_total=?,
                       updated_at=?,
                       last_seen=?
                 WHERE bvdid=?
            """, (stage, urls_total, urls_total, now, now, bvdid)))
        elif stage:
            await self._exec(_Stmt("""
                UPDATE companies
                   SET status='done',
                       stage=?,
                       resume_mode=NULL,
                       updated_at=?,
                       last_seen=?
                 WHERE bvdid=?
            """, (stage, now, now, bvdid)))
        else:
            await self._exec(_Stmt("""
                UPDATE companies
                   SET status='done',
                       resume_mode=NULL,
                       updated_at=?,
                       last_seen=?
                 WHERE bvdid=?
            """, (now, now, bvdid)))

    async def mark_failed(self, bvdid: str, reason: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._exec(_Stmt(
            "UPDATE companies SET status='failed', fail_reason=?, updated_at=?, last_seen=? WHERE bvdid=?",
            (reason, now, now, bvdid)
        ))

    async def set_resume_mode(self, bvdid: str, mode: Optional[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._exec(_Stmt("UPDATE companies SET resume_mode=?, updated_at=?, last_seen=? WHERE bvdid=?",
                               (mode, now, now, bvdid)))

    async def set_urls_total(self, bvdid: str, n: int) -> None:
        await self.update_counts(bvdid, urls_total=n)

    async def update_counts(
        self,
        bvdid: str,
        *,
        urls_total: Optional[int] = None,
        urls_done: Optional[int] = None,
        urls_failed: Optional[int] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sets = ["updated_at=?", "last_seen=?"]
        args: list[Any] = [now, now]
        if urls_total is not None:
            sets.append("urls_total=?"); args.append(int(urls_total))
        if urls_done is not None:
            sets.append("urls_done=?"); args.append(int(urls_done))
        if urls_failed is not None:
            sets.append("urls_failed=?"); args.append(int(urls_failed))
        sets_clause = ", ".join(sets)
        await self._exec(_Stmt(f"UPDATE companies SET {sets_clause} WHERE bvdid=?", tuple(args + [bvdid])))

    async def update_artifacts(
        self,
        bvdid: str,
        *,
        saved_html_total: Optional[int] = None,
        saved_md_total: Optional[int] = None,
        saved_json_total: Optional[int] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sets = ["updated_at=?", "last_seen=?"]
        args: list[Any] = [now, now]
        if saved_html_total is not None:
            sets.append("saved_html_total=?"); args.append(int(saved_html_total))
        if saved_md_total is not None:
            sets.append("saved_md_total=?"); args.append(int(saved_md_total))
        if saved_json_total is not None:
            sets.append("saved_json_total=?"); args.append(int(saved_json_total))
        sets_clause = ", ".join(sets)
        await self._exec(_Stmt(f"UPDATE companies SET {sets_clause} WHERE bvdid=?", tuple(args + [bvdid])))

    async def record_url_index(self, bvdid: str, count: int) -> None:
        """Mark that url_index.json exists (even if empty) and record number of URLs."""
        now = datetime.now(timezone.utc).isoformat()
        await self._exec(_Stmt("""
            UPDATE companies
               SET has_url_index=1,
                   seeded_urls=?,
                   updated_at=?, last_seen=?
             WHERE bvdid=?""", (int(count), now, now, bvdid)))

    # ---------- public API (reads) ----------

    async def get_company(self, bvdid: str) -> Optional[Dict[str, Any]]:
        row = await self._query_one("SELECT * FROM companies WHERE bvdid=?", (bvdid,))
        if not row:
            return None
        return {k: row[k] for k in row.keys()}

    async def recommend_resume(
        self,
        bvdid: str,
        *,
        requested_pipeline: Iterable[str],
        force_seeder_cache: bool = False,
        bypass_local: bool = False,
    ) -> ResumePlan:
        """
        Return a high-level plan to skip stages based solely on DB knowledge.

        NEW: First, if the DB row indicates the company is already DONE and the stored
        `stage` includes the completion stage requested, we immediately return a plan
        that skips the requested completion stage(s). This provides a deterministic
        stage-based skip (no artifact-count heuristics) and avoids racey behavior.

        Fallback: if no DB sentinel present, fall back to the existing artifact-count logic.
        """
        row = await self.get_company(bvdid)
        req = [s.strip().lower() for s in requested_pipeline if s and s.strip()]
        req_set = set(req)
        completion_stage = req[-1] if req else ""

        if row is None:
            # Unknown company in DB: safest plan (no skips)
            return ResumePlan(
                use_index=False,
                skip_seeding_entirely=False,
                url_source="network",
                skip_html=False,
                skip_markdown=False,
                skip_llm=False,
                local_html_expected=False,
                local_md_expected=False,
                reason="no-db-row",
            )

        # --- NEW: deterministic stage-based skip
        db_status = (row.get("status") or "").lower()
        db_stage = (row.get("stage") or "").lower()
        if db_status == "done" and completion_stage:
            # If the recorded stage string includes the requested completion stage,
            # treat the pipeline as already complete.
            if completion_stage.lower() in db_stage.split(","):
                # craft a ResumePlan that skips requested stage(s)
                skip_html = ("html" in req_set) or ("markdown" in req_set) or ("llm" in req_set)
                skip_markdown = ("markdown" in req_set) or ("llm" in req_set)
                skip_llm = ("llm" in req_set)
                # Reason for logging
                reason = f"stage-done({db_stage})"
                return ResumePlan(
                    use_index=True,                      # safe: prefer index on skip (harmless)
                    skip_seeding_entirely=True,
                    url_source="index" if bool(int(row.get("has_url_index") or 0)) else "network",
                    skip_html=skip_html,
                    skip_markdown=skip_markdown,
                    skip_llm=skip_llm,
                    local_html_expected=bool(int(row.get("saved_html_total") or 0)) and not bypass_local,
                    local_md_expected=(bool(int(row.get("saved_md_total") or 0)) or bool(int(row.get("urls_done") or 0))) and not bypass_local,
                    reason=reason,
                )
        # --- End NEW check

        # Existing artifact-count-based conservative logic (unchanged)
        urls_total = int(row.get("urls_total") or 0)
        urls_done = int(row.get("urls_done") or 0)
        has_index = bool(int(row.get("has_url_index") or 0))
        saved_html_total = int(row.get("saved_html_total") or 0)
        saved_md_total = int(row.get("saved_md_total") or 0)
        saved_json_total = int(row.get("saved_json_total") or 0)

        # Decide url source
        use_index = has_index and not force_seeder_cache
        skip_seeding_entirely = use_index

        # Stage completion inference
        html_done = urls_total > 0 and saved_html_total >= urls_total
        md_done = urls_total > 0 and urls_done >= urls_total
        llm_done = urls_total > 0 and saved_json_total >= urls_total

        # Cascade
        if llm_done:
            md_done = True
            html_done = True
        elif md_done:
            html_done = True

        skip_html = html_done and not bypass_local
        skip_markdown = md_done and not bypass_local
        skip_llm = llm_done

        local_html_expected = (saved_html_total > 0) and not bypass_local
        local_md_expected = (saved_md_total > 0 or urls_done > 0) and not bypass_local

        reason_bits: List[str] = []
        if use_index: reason_bits.append("reuse-index")
        if html_done: reason_bits.append("html-complete")
        if md_done: reason_bits.append("md-complete")
        if llm_done: reason_bits.append("llm-complete")
        if not reason_bits: reason_bits.append("no-skip")
        reason = ",".join(reason_bits)

        return ResumePlan(
            use_index=use_index,
            skip_seeding_entirely=skip_seeding_entirely,
            url_source=("index" if use_index else "network"),
            skip_html=skip_html,
            skip_markdown=skip_markdown,
            skip_llm=skip_llm,
            local_html_expected=local_html_expected,
            local_md_expected=local_md_expected,
            reason=reason,
        )