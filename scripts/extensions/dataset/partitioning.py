from __future__ import annotations

import csv
import errno
import json
import logging
import os
import random
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from extensions.io.output_paths import sanitize_bvdid

# Import canonical status constants from crawl_state.
from extensions.crawl.state import (
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_TERMINAL_DONE,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

GLOBAL_STATE_NAME = "crawl_global_state.json"
CRAWL_DB_NAME = "crawl_state.sqlite3"

RETRY_DIR_NAME = "_retry"
RETRY_STATE_NAME = "retry_state.json"
QUARANTINE_NAME = "quarantine.json"
LEDGER_NAME = "failure_ledger.jsonl"

URL_INDEX_NAME = "url_index.json"
URL_INDEX_META_KEY = "__meta__"
URL_INDEX_RESERVED_PREFIX = "__"

# url_index entry status semantics
_MARKDOWN_COMPLETE_STATUSES = {
    "markdown_saved",
    "markdown_suppressed",
    "markdown_done",
    "md_done",
    "md_saved",
    "saved_markdown",
}
_LLM_COMPLETE_STATUSES = {
    "llm_extracted",
    "llm_extracted_empty",
    "llm_full_extracted",
    "llm_done",
    "extracted",
    "product_saved",
    "products_saved",
    "json_saved",
    "presence_done",
}

_KNOWN_COMPANY_STATUSES: Set[str] = {
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_TERMINAL_DONE,
}

# --------------------------------------------------------------------------------------
# Time / atomic I/O
# --------------------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry_emfile(fn, attempts: int = 6, base_delay: float = 0.15):
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
    raise RuntimeError("unreachable")


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
    tmp = Path(f"{str(path)}.tmp.{stamp}")

    def _write() -> None:
        try:
            with open(tmp, "w", encoding=encoding, newline="") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists() and tmp != path:
                tmp.unlink()

    _retry_emfile(_write)


def _atomic_write_json(path: Path, obj: Any, *, pretty: bool = True) -> None:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    _atomic_write_text(path, payload, "utf-8")


def _read_json_strict(path: Path) -> Any:
    if not path.exists():
        return None

    def _read() -> Any:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)

    return _retry_emfile(_read)


# --------------------------------------------------------------------------------------
# Dataset (CSV/TSV) handling (preserve columns)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    header: List[str]
    rows: List[Dict[str, str]]
    id_col: str
    url_col: Optional[str]
    delimiter: str


_ID_CANDIDATES = ["bvdid", "company_id", "id", "BVDID", "BvDID", "BVD_ID", "bvd_id"]
_URL_CANDIDATES = ["url", "domain_url", "website", "web", "homepage", "root_url"]


def _detect_delimiter(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".tsv", ".tab"):
        return "\t"
    return ","


def _find_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        hit = lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def read_table_preserve(path: Path) -> Table:
    delim = _detect_delimiter(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError(f"Dataset file has no header: {path}")

        id_col = _find_col(header, _ID_CANDIDATES)
        if not id_col:
            raise ValueError(
                f"Could not detect company id column. Tried: {_ID_CANDIDATES}. Columns: {header}"
            )

        url_col = _find_col(header, _URL_CANDIDATES)

        rows: List[Dict[str, str]] = []
        for r in reader:
            rr: Dict[str, str] = {}
            for k in header:
                v = r.get(k)
                rr[k] = "" if v is None else str(v)
            rows.append(rr)

    return Table(
        header=header, rows=rows, id_col=id_col, url_col=url_col, delimiter=delim
    )


def write_table_preserve(path: Path, table: Table, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table.header, delimiter=table.delimiter)
        w.writeheader()
        for r in rows:
            out = {
                k: ("" if r.get(k) is None else str(r.get(k) or ""))
                for k in table.header
            }
            w.writerow(out)


def split_table_by_ids(
    table: Table, move_ids: Set[str]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    moved: List[Dict[str, str]] = []
    remain: List[Dict[str, str]] = []
    for r in table.rows:
        cid = (r.get(table.id_col) or "").strip()
        if cid and cid in move_ids:
            moved.append(r)
        else:
            remain.append(r)
    return remain, moved


def merge_tables_dedupe(
    base: Table,
    base_rows: List[Dict[str, str]],
    other: Table,
    other_rows: List[Dict[str, str]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    id_col_base = base.id_col
    id_col_other = other.id_col

    header_union = list(base.header)
    for c in other.header:
        if c not in header_union:
            header_union.append(c)

    seen: Set[str] = set()
    out_rows: List[Dict[str, str]] = []

    def _emit(r: Dict[str, str], *, src_header: List[str], src_id_col: str) -> None:
        cid = (r.get(src_id_col) or "").strip()
        if not cid or cid in seen:
            return
        seen.add(cid)
        merged = {k: "" for k in header_union}
        for k in src_header:
            merged[k] = "" if r.get(k) is None else str(r.get(k) or "")
        if src_id_col != id_col_base:
            merged[id_col_base] = cid
        out_rows.append(merged)

    for r in base_rows:
        _emit(r, src_header=base.header, src_id_col=id_col_base)
    for r in other_rows:
        _emit(r, src_header=other.header, src_id_col=id_col_other)

    return header_union, out_rows


# --------------------------------------------------------------------------------------
# Run-root paths / company dirs
# --------------------------------------------------------------------------------------


def retry_dir(root: Path) -> Path:
    return root / RETRY_DIR_NAME


def crawl_db_path(root: Path) -> Path:
    return root / CRAWL_DB_NAME


def global_state_path(root: Path) -> Path:
    return root / GLOBAL_STATE_NAME


def find_company_dir(root: Path, company_id: str) -> Optional[Path]:
    raw = root / company_id
    if raw.exists() and raw.is_dir():
        return raw
    safe = root / sanitize_bvdid(company_id)
    if safe.exists() and safe.is_dir():
        return safe
    return None


def expected_company_dir(root: Path, company_id: str) -> Path:
    return root / sanitize_bvdid(company_id)


def url_index_path_in_company_dir(company_dir: Path) -> Path:
    p1 = company_dir / "metadata" / URL_INDEX_NAME
    if p1.exists():
        return p1
    return company_dir / "checkpoints" / URL_INDEX_NAME


# --------------------------------------------------------------------------------------
# SQLite schema (match crawl.state.py)
# --------------------------------------------------------------------------------------

COMPANIES_SCHEMA_SQL = """
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
    updated_at TEXT,
    industry INTEGER,
    nace INTEGER,
    industry_label TEXT,
    industry_source TEXT
);
"""

RUNS_SCHEMA_SQL = """
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
    last_company_bvdid TEXT,
    last_updated TEXT
);
"""

RUN_COMPANY_DONE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS run_company_done (
    run_id TEXT NOT NULL,
    bvdid TEXT NOT NULL,
    done_at TEXT,
    PRIMARY KEY (run_id, bvdid)
);
"""


def _open_db(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_crawl_db(db: Path) -> None:
    conn = _open_db(db)
    try:
        conn.execute(COMPANIES_SCHEMA_SQL)
        conn.execute(RUNS_SCHEMA_SQL)
        conn.execute(RUN_COMPANY_DONE_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def read_companies_rows(db: Path) -> Dict[str, Dict[str, Any]]:
    if not db.exists():
        return {}
    conn = _open_db(db)
    try:
        rows = conn.execute("SELECT * FROM companies").fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = dict(r)
            cid = str(d.get("bvdid") or "").strip()
            if cid:
                out[cid] = d
        return out
    finally:
        conn.close()


def read_latest_run_row(db: Path) -> Optional[Dict[str, Any]]:
    if not db.exists():
        return None
    conn = _open_db(db)
    try:
        row = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def read_run_company_done_count(db: Path, run_id: str) -> Optional[int]:
    if not db.exists():
        return None
    conn = _open_db(db)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM run_company_done WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            return 0
        return int(row["c"] or 0)
    finally:
        conn.close()


def delete_company_rows(db: Path, company_ids: Sequence[str]) -> int:
    if (not db.exists()) or (not company_ids):
        return 0
    conn = _open_db(db)
    try:
        n = 0
        chunk = 500
        for i in range(0, len(company_ids), chunk):
            part = list(company_ids[i : i + chunk])
            q = ",".join("?" for _ in part)
            cur = conn.execute(
                f"DELETE FROM companies WHERE bvdid IN ({q})", tuple(part)
            )
            n += int(cur.rowcount or 0)
        conn.commit()
        return n
    finally:
        conn.close()


def write_companies_rows(db: Path, rows: Dict[str, Dict[str, Any]]) -> None:
    init_crawl_db(db)
    conn = _open_db(db)
    try:
        cols = [
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
            "industry",
            "nace",
            "industry_label",
            "industry_source",
        ]
        placeholders = ",".join("?" for _ in cols)

        conn.execute("BEGIN;")
        for cid, r in rows.items():
            payload = dict(r)
            payload["bvdid"] = cid
            conn.execute(
                f"INSERT OR REPLACE INTO companies ({', '.join(cols)}) VALUES ({placeholders})",
                tuple(payload.get(k) for k in cols),
            )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# url_index.json classification (mirror crawl.state.py semantics)
# --------------------------------------------------------------------------------------


def _is_reserved_url_index_key(k: Any) -> bool:
    return str(k).startswith(URL_INDEX_RESERVED_PREFIX)


def _index_crawl_finished(index: Dict[str, Any]) -> bool:
    meta = index.get(URL_INDEX_META_KEY)
    return bool(isinstance(meta, dict) and meta.get("crawl_finished"))


def _status_has_any(s: str, needles: Tuple[str, ...]) -> bool:
    return any(n in s for n in needles)


def _classify_url_entry(ent: Dict[str, Any]) -> Tuple[bool, bool]:
    status = str(ent.get("status") or "").strip().lower()

    has_md_path = bool(ent.get("markdown_path") or ent.get("md_path"))
    has_llm_artifact = bool(
        ent.get("json_path")
        or ent.get("product_path")
        or ent.get("products_path")
        or ent.get("llm_json_path")
        or ent.get("extraction_path")
    )

    presence_checked = bool(ent.get("presence_checked") or ent.get("presence_done"))
    extracted_flag = bool(ent.get("extracted") or ent.get("llm_extracted"))

    status_md_done = (
        status in _MARKDOWN_COMPLETE_STATUSES
        or _status_has_any(
            status, ("markdown_done", "markdown_saved", "md_done", "md_saved")
        )
        or (
            ("markdown" in status or status == "md")
            and _status_has_any(status, ("done", "saved", "complete", "suppressed"))
        )
    )
    status_llm_done = (
        status in _LLM_COMPLETE_STATUSES
        or _status_has_any(
            status,
            (
                "llm_done",
                "llm_extracted",
                "full_extracted",
                "product_saved",
                "products_saved",
                "json_saved",
            ),
        )
        or (
            ("llm" in status or "extract" in status)
            and _status_has_any(status, ("done", "saved", "complete", "extracted"))
        )
    )

    markdown_done = bool(has_md_path or status_md_done)
    if has_llm_artifact or extracted_flag or status_llm_done or presence_checked:
        markdown_done = True

    llm_done = bool(
        extracted_flag or has_llm_artifact or status_llm_done or presence_checked
    )
    return markdown_done, llm_done


def compute_company_stage_from_url_index(
    index: Dict[str, Any],
) -> Tuple[str, int, int, int]:
    """
    Match crawl.state.py:

    - Reserved keys (starting with "__") are ignored as URLs.
    - Never conclude markdown_done/llm_done unless __meta__.crawl_finished == True.
    """
    if not index:
        return COMPANY_STATUS_PENDING, 0, 0, 0

    crawl_finished = _index_crawl_finished(index)

    urls_total = 0
    md_done = 0
    llm_done = 0

    for url, raw_ent in index.items():
        if _is_reserved_url_index_key(url):
            continue
        urls_total += 1
        ent = raw_ent if isinstance(raw_ent, dict) else {}
        m, l = _classify_url_entry(ent)
        if m:
            md_done += 1
        if l:
            llm_done += 1

    if urls_total == 0:
        return COMPANY_STATUS_PENDING, 0, 0, 0

    if not crawl_finished:
        if llm_done > 0:
            return COMPANY_STATUS_LLM_NOT_DONE, urls_total, md_done, llm_done
        if md_done > 0:
            return COMPANY_STATUS_MD_NOT_DONE, urls_total, md_done, llm_done
        return COMPANY_STATUS_PENDING, urls_total, md_done, llm_done

    if llm_done == urls_total:
        return COMPANY_STATUS_LLM_DONE, urls_total, md_done, llm_done
    if llm_done > 0:
        return COMPANY_STATUS_LLM_NOT_DONE, urls_total, md_done, llm_done
    if md_done == urls_total:
        return COMPANY_STATUS_MD_DONE, urls_total, md_done, llm_done
    return COMPANY_STATUS_MD_NOT_DONE, urls_total, md_done, llm_done


def read_url_index_file(path: Path) -> Dict[str, Any]:
    obj = _read_json_strict(path)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"url_index.json is not a JSON object: {path}")
    return obj


# --------------------------------------------------------------------------------------
# Retry state subset/merge (raw JSON files)
# --------------------------------------------------------------------------------------


def _load_retry_state(
    root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path, Path]:
    rdir = retry_dir(root)
    sp = rdir / RETRY_STATE_NAME
    qp = rdir / QUARANTINE_NAME
    lp = rdir / LEDGER_NAME

    state_raw = _read_json_strict(sp) if sp.exists() else None
    quarantine_raw = _read_json_strict(qp) if qp.exists() else None

    state = state_raw if isinstance(state_raw, dict) else {}
    quarantine = quarantine_raw if isinstance(quarantine_raw, dict) else {}
    return state, quarantine, sp, qp, lp


def write_retry_state(
    root: Path, state: Dict[str, Any], quarantine: Dict[str, Any]
) -> None:
    rdir = retry_dir(root)
    rdir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(rdir / RETRY_STATE_NAME, state, pretty=False)
    _atomic_write_json(rdir / QUARANTINE_NAME, quarantine, pretty=False)


def filter_ledger(src_ledger: Path, dst_ledger: Path, keep_ids: Set[str]) -> int:
    if not src_ledger.exists():
        return 0
    dst_ledger.parent.mkdir(parents=True, exist_ok=True)
    kept = 0

    def _iter_lines() -> Iterator[str]:
        with open(src_ledger, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line

    with open(dst_ledger, "a", encoding="utf-8", newline="") as out:
        for line in _iter_lines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line in ledger: %s", src_ledger)
                continue
            if not isinstance(obj, dict):
                continue
            cid = str(obj.get("company_id") or obj.get("bvdid") or "").strip()
            if cid and cid in keep_ids:
                out.write(
                    json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                kept += 1
    return kept


def merge_retry_states(
    a_state: Dict[str, Any], b_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge retry_state.json dicts. For conflicts, pick the entry with higher updated_at,
    fallback to higher attempts.
    """
    out = dict(a_state)
    for cid, ent in b_state.items():
        if cid not in out:
            out[cid] = ent
            continue
        ea = out.get(cid)
        eb = ent
        if not isinstance(ea, dict) or not isinstance(eb, dict):
            out[cid] = eb
            continue
        ua = float(ea.get("updated_at") or 0.0)
        ub = float(eb.get("updated_at") or 0.0)
        if ub > ua:
            out[cid] = eb
            continue
        if ua > ub:
            continue
        aa = int(ea.get("attempts") or 0)
        ab = int(eb.get("attempts") or 0)
        if ab > aa:
            out[cid] = eb
    return out


def merge_quarantine(a_q: Dict[str, Any], b_q: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a_q)
    for cid, ent in b_q.items():
        if cid not in out:
            out[cid] = ent
            continue
        ea = out.get(cid)
        eb = ent
        if not isinstance(ea, dict) or not isinstance(eb, dict):
            out[cid] = eb
            continue

        def _ts(x: Dict[str, Any]) -> float:
            for k in ("updated_at", "quarantined_at", "at", "ts"):
                if k in x:
                    return float(x.get(k) or 0.0)
            return 0.0

        if _ts(eb) >= _ts(ea):
            out[cid] = eb
    return out


# --------------------------------------------------------------------------------------
# File-tree sync and url_index merge
# --------------------------------------------------------------------------------------


def _file_better(src: Path, dst: Path) -> bool:
    ss = src.stat()
    ds = dst.stat()
    if ss.st_size != ds.st_size:
        return ss.st_size > ds.st_size
    return ss.st_mtime_ns > ds.st_mtime_ns


def merge_url_index_files(dst_path: Path, src_path: Path) -> None:
    """
    Merge url_index.json dicts with per-URL preference:
      score(llm_done)=2, score(md_done)=1

    Reserved __meta__ is merged:
      - crawl_finished becomes OR of both.
      - other keys: prefer non-empty src values, else keep dst.
    """
    dst = read_url_index_file(dst_path) if dst_path.exists() else {}
    src = read_url_index_file(src_path) if src_path.exists() else {}

    if not src:
        return
    if not dst:
        _atomic_write_json(dst_path, src, pretty=False)
        return

    def _score(ent: Dict[str, Any]) -> int:
        md, llm = _classify_url_entry(ent if isinstance(ent, dict) else {})
        return (2 if llm else 0) + (1 if md else 0)

    merged: Dict[str, Any] = dict(dst)

    # Merge __meta__ first (if any)
    if URL_INDEX_META_KEY in src or URL_INDEX_META_KEY in dst:
        m_dst = dst.get(URL_INDEX_META_KEY)
        m_src = src.get(URL_INDEX_META_KEY)
        mdict: Dict[str, Any] = dict(m_dst) if isinstance(m_dst, dict) else {}
        sdict: Dict[str, Any] = dict(m_src) if isinstance(m_src, dict) else {}

        if bool(mdict.get("crawl_finished")) or bool(sdict.get("crawl_finished")):
            mdict["crawl_finished"] = True
        else:
            mdict["crawl_finished"] = False

        for k, v in sdict.items():
            if k == "crawl_finished":
                continue
            if k not in mdict or mdict.get(k) in ("", None, [], {}):
                mdict[k] = v

        merged[URL_INDEX_META_KEY] = mdict

    for url, raw_ent in src.items():
        if _is_reserved_url_index_key(url):
            continue
        ent = raw_ent if isinstance(raw_ent, dict) else {}
        if url not in merged:
            merged[url] = ent
            continue
        cur = merged.get(url)
        cur_ent = cur if isinstance(cur, dict) else {}
        s_new = _score(ent)
        s_cur = _score(cur_ent)
        if s_new > s_cur:
            merged[url] = ent
        elif s_new == s_cur:
            out_ent = dict(cur_ent)
            for k, v in ent.items():
                if k not in out_ent or out_ent.get(k) in ("", None, [], {}):
                    out_ent[k] = v
            merged[url] = out_ent

    _atomic_write_json(dst_path, merged, pretty=False)


def sync_tree(src: Path, dst: Path, *, move: bool, merge: bool) -> None:
    """
    Sync directory src into dst.

    - If dst doesn't exist: move/copy whole tree.
    - If exists and merge=True: merge contents, with special handling for url_index.json.
    """
    if not src.exists():
        return

    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copytree(str(src), str(dst), dirs_exist_ok=False)
        return

    if not merge:
        shutil.rmtree(dst)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copytree(str(src), str(dst), dirs_exist_ok=False)
        return

    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for d in dirs:
            (out_dir / d).mkdir(parents=True, exist_ok=True)

        for fn in files:
            sp = Path(root) / fn
            dp = out_dir / fn

            if fn == URL_INDEX_NAME:
                merge_url_index_files(dp, sp)
                continue

            if not dp.exists():
                shutil.copy2(sp, dp)
            else:
                if _file_better(sp, dp):
                    shutil.copy2(sp, dp)

    if move:
        shutil.rmtree(src, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Global state (DB-only) writer (match CrawlState.write_global_state_from_db_only keys)
# --------------------------------------------------------------------------------------


def write_global_state_from_db_only(root: Path) -> Dict[str, Any]:
    db = crawl_db_path(root)
    if not db.exists():
        payload: Dict[str, Any] = {
            "output_root": str(root.resolve()),
            "db_path": str(db.resolve()),
            "updated_at": _now_iso(),
            "total_companies": 0,
            "crawled_companies": 0,
            "completed_companies": 0,
            "percentage_completed": 0.0,
            "by_status": {k: 0 for k in sorted(_KNOWN_COMPANY_STATUSES)},
            "unknown_statuses": {},
            "pending_company_ids": [],
            "in_progress_company_ids": [],
            "done_company_ids": [],
            "terminal_done_company_ids": [],
            "terminal_done_reasons": {},
            "urls_total_sum": 0,
            "urls_markdown_done_sum": 0,
            "urls_llm_done_sum": 0,
            "latest_run": None,
        }
        _atomic_write_json(global_state_path(root), payload, pretty=True)
        return payload

    conn = _open_db(db)
    try:
        rows = conn.execute(
            """
            SELECT bvdid, status, urls_total, urls_markdown_done, urls_llm_done,
                   done_reason, done_at
              FROM companies
            """
        ).fetchall()
        run_row = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        run_done_count: Optional[int] = None
        if run_row is not None:
            rrid = str(run_row["run_id"])
            c = conn.execute(
                "SELECT COUNT(*) AS c FROM run_company_done WHERE run_id=?",
                (rrid,),
            ).fetchone()
            run_done_count = int(c["c"] or 0) if c is not None else 0
    finally:
        conn.close()

    total = len(rows)
    by_status: Dict[str, int] = {k: 0 for k in _KNOWN_COMPANY_STATUSES}
    unknown_statuses: Dict[str, int] = {}

    pending: List[str] = []
    in_progress: List[str] = []
    done: List[str] = []
    terminal_done: List[str] = []
    terminal_reasons: Dict[str, int] = {}

    urls_total_sum = 0
    urls_md_done_sum = 0
    urls_llm_done_sum = 0

    for r in rows:
        b = str(r["bvdid"])
        raw_st = r["status"]
        st_norm = (
            str(raw_st).strip().lower() if raw_st is not None else ""
        ) or COMPANY_STATUS_PENDING

        if st_norm in _KNOWN_COMPANY_STATUSES:
            by_status[st_norm] = by_status.get(st_norm, 0) + 1
            st_effective = st_norm
        else:
            unknown_statuses[st_norm] = unknown_statuses.get(st_norm, 0) + 1
            by_status[COMPANY_STATUS_PENDING] = (
                by_status.get(COMPANY_STATUS_PENDING, 0) + 1
            )
            st_effective = COMPANY_STATUS_PENDING

        if st_effective == COMPANY_STATUS_PENDING:
            pending.append(b)
        elif st_effective in (COMPANY_STATUS_MD_NOT_DONE, COMPANY_STATUS_LLM_NOT_DONE):
            in_progress.append(b)
        elif st_effective == COMPANY_STATUS_TERMINAL_DONE:
            done.append(b)
            terminal_done.append(b)
            dr = (r["done_reason"] or "unknown").strip()
            terminal_reasons[dr] = terminal_reasons.get(dr, 0) + 1
        else:
            done.append(b)

        urls_total_sum += int(r["urls_total"] or 0)
        urls_md_done_sum += int(r["urls_markdown_done"] or 0)
        urls_llm_done_sum += int(r["urls_llm_done"] or 0)

    crawled = total - len(pending)
    completed = len(done)
    percentage_completed = (completed / total * 100.0) if total else 0.0

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
            "last_company_bvdid": run_row["last_company_bvdid"],
            "last_updated": run_row["last_updated"],
        }

    payload = {
        "output_root": str(root.resolve()),
        "db_path": str(db.resolve()),
        "updated_at": _now_iso(),
        "total_companies": total,
        "crawled_companies": int(crawled),
        "completed_companies": int(completed),
        "percentage_completed": round(percentage_completed, 2),
        "by_status": by_status,
        "unknown_statuses": unknown_statuses,
        "pending_company_ids": pending[:200],
        "in_progress_company_ids": in_progress[:200],
        "done_company_ids": done[:200],
        "terminal_done_company_ids": terminal_done[:200],
        "terminal_done_reasons": terminal_reasons,
        "urls_total_sum": int(urls_total_sum),
        "urls_markdown_done_sum": int(urls_md_done_sum),
        "urls_llm_done_sum": int(urls_llm_done_sum),
        "latest_run": latest_run,
    }

    _atomic_write_json(global_state_path(root), payload, pretty=True)
    return payload


# --------------------------------------------------------------------------------------
# High-level operations: split and merge
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPlan:
    move_ids: List[str]
    remain_ids: List[str]


def plan_split(
    *,
    root: Path,
    dataset_table: Table,
    move_count: Optional[int],
    move_ids: Optional[Set[str]],
    seed: int,
    only_not_done: bool,
) -> SplitPlan:
    db_rows = read_companies_rows(crawl_db_path(root))

    dataset_ids: List[str] = []
    for r in dataset_table.rows:
        cid = (r.get(dataset_table.id_col) or "").strip()
        if cid:
            dataset_ids.append(cid)

    candidates: List[str] = []
    for cid in dataset_ids:
        row = db_rows.get(cid)
        st = (
            str((row or {}).get("status") or COMPANY_STATUS_PENDING).strip().lower()
            or COMPANY_STATUS_PENDING
        )
        if only_not_done and st in (
            COMPANY_STATUS_MD_DONE,
            COMPANY_STATUS_LLM_DONE,
            COMPANY_STATUS_TERMINAL_DONE,
        ):
            continue
        candidates.append(cid)

    if move_ids is not None:
        chosen = [cid for cid in candidates if cid in move_ids]
    else:
        n = int(move_count or 0)
        if n <= 0:
            raise ValueError("move_count must be > 0 when move_ids not provided.")
        rnd = random.Random(int(seed))
        pool = list(candidates)
        rnd.shuffle(pool)
        chosen = pool[: min(n, len(pool))]

    move_set = set(chosen)
    remain = [cid for cid in dataset_ids if cid not in move_set]
    return SplitPlan(move_ids=chosen, remain_ids=remain)


def _prefer_non_empty(dst: Dict[str, Any], src: Dict[str, Any], key: str) -> None:
    if dst.get(key) in ("", None, [], {}):
        if src.get(key) not in ("", None, [], {}):
            dst[key] = src.get(key)


def rebuild_db_from_outputs(
    root: Path,
    company_ids: Sequence[str],
    *,
    prefer_existing_rows: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Build a fresh crawl_state.sqlite3 in root from url_index.json under each company dir.

    - Preserves name/root_url/industry/nace/industry_label/industry_source and terminal fields from prefer_existing_rows.
    - Status is derived from url_index unless existing status is terminal_done.
    """
    prefer_existing_rows = prefer_existing_rows or {}

    tmp = root / f"{CRAWL_DB_NAME}.tmp.{int(time.time())}"
    init_crawl_db(tmp)

    out_rows: Dict[str, Dict[str, Any]] = {}
    now_iso = _now_iso()

    for cid in company_ids:
        base = prefer_existing_rows.get(cid, {})

        name = base.get("name")
        root_url = base.get("root_url")

        industry = base.get("industry")
        nace = base.get("nace")
        industry_label = base.get("industry_label")
        industry_source = base.get("industry_source")

        existing_status = (
            str(base.get("status") or "").strip().lower() or COMPANY_STATUS_PENDING
        )
        keep_terminal = existing_status == COMPANY_STATUS_TERMINAL_DONE

        status = COMPANY_STATUS_PENDING
        ut = 0
        md = 0
        llm = 0

        cdir = find_company_dir(root, cid)
        if cdir is not None:
            ip = url_index_path_in_company_dir(cdir)
            if ip.exists():
                idx = read_url_index_file(ip)
                status, ut, md, llm = compute_company_stage_from_url_index(idx)

        # Normalize counts like crawl.state recompute_company_from_index does.
        total = max(int(ut), int(md), int(llm))
        md_done = min(int(md), total)
        llm_done = min(int(llm), total)

        if keep_terminal:
            status = COMPANY_STATUS_TERMINAL_DONE

        row = {
            "bvdid": cid,
            "name": name,
            "root_url": root_url,
            "status": status,
            "urls_total": int(total),
            "urls_markdown_done": int(md_done),
            "urls_llm_done": int(llm_done),
            "last_error": base.get("last_error"),
            "done_reason": base.get("done_reason"),
            "done_details": base.get("done_details"),
            "done_at": base.get("done_at"),
            "created_at": base.get("created_at") or now_iso,
            "updated_at": now_iso,
            "industry": industry,
            "nace": nace,
            "industry_label": industry_label,
            "industry_source": industry_source,
        }
        out_rows[cid] = row

    write_companies_rows(tmp, out_rows)

    final = crawl_db_path(root)
    if final.exists():
        final.unlink()
    os.replace(tmp, final)


def split_run_root(
    *,
    root: Path,
    dataset_path: Path,
    bundle_root: Path,
    remain_dataset_out: Path,
    moved_dataset_out: Path,
    move_count: Optional[int],
    move_ids_file: Optional[Path],
    seed: int,
    only_not_done: bool,
    move_mode: str,  # "move" or "copy"
    include_ledger: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    root = root.resolve()
    bundle_root = bundle_root.resolve()

    table = read_table_preserve(dataset_path)

    move_ids: Optional[Set[str]] = None
    if move_ids_file is not None:
        lines = move_ids_file.read_text(encoding="utf-8").splitlines()
        move_ids = {ln.strip() for ln in lines if ln.strip()}

    plan = plan_split(
        root=root,
        dataset_table=table,
        move_count=move_count,
        move_ids=move_ids,
        seed=seed,
        only_not_done=only_not_done,
    )

    move_set = set(plan.move_ids)
    remain_rows, moved_rows = split_table_by_ids(table, move_set)

    report: Dict[str, Any] = {
        "root": str(root),
        "bundle_root": str(bundle_root),
        "dataset": str(dataset_path),
        "move_mode": move_mode,
        "only_not_done": bool(only_not_done),
        "move_count": len(plan.move_ids),
        "remain_count": len(plan.remain_ids),
    }

    if dry_run:
        return report

    bundle_root.mkdir(parents=True, exist_ok=True)

    # 1) Write dataset files
    write_table_preserve(remain_dataset_out, table, remain_rows)
    write_table_preserve(moved_dataset_out, table, moved_rows)
    write_table_preserve(bundle_root / dataset_path.name, table, moved_rows)

    # 2) Move/copy company dirs
    do_move = move_mode.strip().lower() == "move"
    for cid in plan.move_ids:
        src_dir = find_company_dir(root, cid)
        if src_dir is None:
            continue
        dst_dir = expected_company_dir(bundle_root, cid)
        sync_tree(src_dir, dst_dir, move=do_move, merge=True)

    # 3) Retry state subset into bundle; if move, remove from source.
    src_state, src_q, _, _, src_ledger = _load_retry_state(root)
    bundle_state = {cid: v for cid, v in src_state.items() if cid in move_set}
    bundle_q = {cid: v for cid, v in src_q.items() if cid in move_set}
    write_retry_state(bundle_root, bundle_state, bundle_q)

    if include_ledger:
        kept = filter_ledger(src_ledger, retry_dir(bundle_root) / LEDGER_NAME, move_set)
        report["bundle_ledger_rows"] = kept

    if do_move:
        remain_state = {cid: v for cid, v in src_state.items() if cid not in move_set}
        remain_q = {cid: v for cid, v in src_q.items() if cid not in move_set}
        write_retry_state(root, remain_state, remain_q)

    # 4) Rebuild bundle DB from outputs (authoritative), preserving fields from source DB.
    src_db = crawl_db_path(root)
    src_rows = read_companies_rows(src_db)

    prefer_for_bundle: Dict[str, Dict[str, Any]] = {}
    for cid in plan.move_ids:
        if cid in src_rows:
            prefer_for_bundle[cid] = dict(src_rows[cid])

    rebuild_db_from_outputs(
        bundle_root, plan.move_ids, prefer_existing_rows=prefer_for_bundle
    )
    write_global_state_from_db_only(bundle_root)

    # 5) If move, rebuild source DB from remaining outputs (authoritative), preserving remaining fields.
    if do_move:
        prefer_for_source: Dict[str, Dict[str, Any]] = {}
        for cid in plan.remain_ids:
            if cid in src_rows:
                prefer_for_source[cid] = dict(src_rows[cid])
        rebuild_db_from_outputs(
            root, plan.remain_ids, prefer_existing_rows=prefer_for_source
        )
        write_global_state_from_db_only(root)

    else:
        # copy mode: source unchanged; still refresh global state file to reflect DB as-is
        write_global_state_from_db_only(root)

    return report


def merge_run_roots(
    *,
    target_root: Path,
    source_roots: Sequence[Path],
    move_mode: str,  # "move" or "copy"
    include_ledger: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    target_root = target_root.resolve()
    srcs = [Path(p).resolve() for p in source_roots]

    report: Dict[str, Any] = {
        "target_root": str(target_root),
        "sources": [str(s) for s in srcs],
        "move_mode": move_mode,
    }

    if dry_run:
        return report

    target_root.mkdir(parents=True, exist_ok=True)
    do_move = move_mode.strip().lower() == "move"

    all_company_ids: Set[str] = set()
    prefer_rows: Dict[str, Dict[str, Any]] = {}

    # 1) Merge company dirs; collect prefer rows from source DBs
    per_source_ids: Dict[Path, List[str]] = {}
    for src_root in srcs:
        rows = read_companies_rows(crawl_db_path(src_root))
        ids = list(rows.keys())
        per_source_ids[src_root] = ids

        for cid, r in rows.items():
            all_company_ids.add(cid)
            if cid not in prefer_rows:
                prefer_rows[cid] = dict(r)
            else:
                cur = prefer_rows[cid]
                # Preserve first non-empty for these fields.
                for k in (
                    "name",
                    "root_url",
                    "industry",
                    "nace",
                    "industry_label",
                    "industry_source",
                    "done_reason",
                    "done_details",
                    "done_at",
                    "last_error",
                    "status",
                    "created_at",
                ):
                    _prefer_non_empty(cur, r, k)

        for cid in ids:
            src_dir = find_company_dir(src_root, cid)
            if src_dir is None:
                continue
            dst_dir = expected_company_dir(target_root, cid)
            sync_tree(src_dir, dst_dir, move=do_move, merge=True)

    report["company_ids_merged"] = len(all_company_ids)

    # 2) Merge retry state (and optionally ledgers)
    merged_state: Dict[str, Any] = {}
    merged_q: Dict[str, Any] = {}
    ledgers_to_append: List[Path] = []

    for src_root in srcs:
        st, q, _, _, lp = _load_retry_state(src_root)
        merged_state = merge_retry_states(merged_state, st)
        merged_q = merge_quarantine(merged_q, q)
        if include_ledger and lp.exists():
            ledgers_to_append.append(lp)

    write_retry_state(target_root, merged_state, merged_q)

    if include_ledger and ledgers_to_append:
        out_ledger = retry_dir(target_root) / LEDGER_NAME
        out_ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(out_ledger, "a", encoding="utf-8", newline="") as out:
            for lp in ledgers_to_append:
                with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                    shutil.copyfileobj(f, out)

    # 3) Rebuild target DB from merged outputs (authoritative)
    rebuild_db_from_outputs(
        target_root, sorted(all_company_ids), prefer_existing_rows=prefer_rows
    )
    write_global_state_from_db_only(target_root)

    # 4) If move, also scrub moved IDs from each source DB/retry and refresh source global state
    if do_move:
        for src_root in srcs:
            moved_ids = per_source_ids.get(src_root, [])
            if not moved_ids:
                continue

            # Retry state: remove moved IDs
            st, q, _, _, _ = _load_retry_state(src_root)
            st2 = {cid: v for cid, v in st.items() if cid not in set(moved_ids)}
            q2 = {cid: v for cid, v in q.items() if cid not in set(moved_ids)}
            write_retry_state(src_root, st2, q2)

            # DB: rebuild from remaining company dirs by scanning remaining ids in db after deletion.
            src_db = crawl_db_path(src_root)
            delete_company_rows(src_db, moved_ids)
            remaining_rows = read_companies_rows(src_db)
            rebuild_db_from_outputs(
                src_root,
                sorted(remaining_rows.keys()),
                prefer_existing_rows=remaining_rows,
            )
            write_global_state_from_db_only(src_root)

    return report
