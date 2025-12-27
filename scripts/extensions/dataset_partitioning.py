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
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from extensions.output_paths import sanitize_bvdid

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants / schema mirrors (crawl_state.py)
# --------------------------------------------------------------------------------------

GLOBAL_STATE_NAME = "crawl_global_state.json"
CRAWL_DB_NAME = "crawl_state.sqlite3"
RETRY_DIR_NAME = "_retry"
RETRY_STATE_NAME = "retry_state.json"
QUARANTINE_NAME = "quarantine.json"
LEDGER_NAME = "failure_ledger.jsonl"

STATUS_PENDING = "pending"
STATUS_MD_NOT_DONE = "markdown_not_done"
STATUS_MD_DONE = "markdown_done"
STATUS_LLM_NOT_DONE = "llm_not_done"
STATUS_LLM_DONE = "llm_done"
STATUS_TERMINAL_DONE = "terminal_done"

_DONE_STATUSES_DEFAULT = {STATUS_MD_DONE, STATUS_LLM_DONE, STATUS_TERMINAL_DONE}
_NOT_DONE_STATUSES_DEFAULT = {STATUS_PENDING, STATUS_MD_NOT_DONE, STATUS_LLM_NOT_DONE}

# Best-effort tolerant sets (mirrors crawl_state heuristics)
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


# --------------------------------------------------------------------------------------
# Utility: retries for EMFILE / transient FS errors
# --------------------------------------------------------------------------------------


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


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
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
            try:
                if tmp.exists() and tmp != path:
                    tmp.unlink()
            except Exception:
                pass

    _retry_emfile(_write)


def _atomic_write_json(path: Path, obj: Any, *, pretty: bool = True) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None)
    _atomic_write_text(path, payload, "utf-8")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None

    def _read() -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    return _retry_emfile(_read)


# --------------------------------------------------------------------------------------
# Dataset (CSV/TSV) handling with “compat as much as possible”
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
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def read_table_preserve(path: Path) -> Table:
    """
    Reads CSV/TSV preserving *all columns*. For non-CSV/TSV, you should convert first.
    """
    delim = _detect_delimiter(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError(f"Dataset file has no header: {path}")

        id_col = _find_col(header, _ID_CANDIDATES)
        if not id_col:
            raise ValueError(
                f"Could not detect company id column. Tried: {_ID_CANDIDATES}. "
                f"Columns: {header}"
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
                k: (r.get(k, "") if r.get(k, "") is not None else "")
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
    """
    Merge rows, dedupe by company id. Header becomes union (base header first).
    """
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
            if k in merged:
                merged[k] = "" if r.get(k) is None else str(r.get(k) or "")
        # normalize id into base id_col key if different
        if src_id_col != id_col_base:
            merged[id_col_base] = cid
        out_rows.append(merged)

    for r in base_rows:
        _emit(r, src_header=base.header, src_id_col=id_col_base)
    for r in other_rows:
        _emit(r, src_header=other.header, src_id_col=id_col_other)

    return header_union, out_rows


# --------------------------------------------------------------------------------------
# Run-root paths
# --------------------------------------------------------------------------------------


def retry_dir(root: Path) -> Path:
    return root / RETRY_DIR_NAME


def crawl_db_path(root: Path) -> Path:
    return root / CRAWL_DB_NAME


def global_state_path(root: Path) -> Path:
    return root / GLOBAL_STATE_NAME


def find_company_dir(root: Path, company_id: str) -> Optional[Path]:
    """
    Try both raw and sanitized.
    """
    raw = root / company_id
    if raw.exists() and raw.is_dir():
        return raw
    safe = root / sanitize_bvdid(company_id)
    if safe.exists() and safe.is_dir():
        return safe
    return None


def expected_company_dir(root: Path, company_id: str) -> Path:
    """
    Where we will write to (sanitized) when creating new dirs.
    """
    return root / sanitize_bvdid(company_id)


def url_index_path_in_company_dir(company_dir: Path) -> Path:
    # Your ensure_company_dirs uses /metadata or /checkpoints; crawl_state prefers metadata then checkpoints.
    p1 = company_dir / "metadata" / "url_index.json"
    if p1.exists():
        return p1
    p2 = company_dir / "checkpoints" / "url_index.json"
    return p2


# --------------------------------------------------------------------------------------
# crawl_state.sqlite3 low-level read/write
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
    updated_at TEXT
);
"""

RUNS_SCHEMA_SQL = """
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
        conn.commit()
    finally:
        conn.close()


def read_companies_rows(db: Path) -> Dict[str, Dict[str, Any]]:
    if not db.exists():
        return {}
    conn = _open_db(db)
    try:
        try:
            rows = conn.execute("SELECT * FROM companies").fetchall()
        except sqlite3.OperationalError:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = dict(r)
            cid = str(d.get("bvdid") or "").strip()
            if cid:
                out[cid] = d
        return out
    finally:
        conn.close()


def delete_company_rows(db: Path, company_ids: Sequence[str]) -> int:
    if not db.exists() or not company_ids:
        return 0
    conn = _open_db(db)
    try:
        n = 0
        # chunk to avoid SQLITE max vars
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
        ]

        def _val(r: Dict[str, Any], k: str) -> Any:
            return r.get(k)

        conn.execute("BEGIN;")
        for cid, r in rows.items():
            payload = {k: _val(r, k) for k in cols}
            payload["bvdid"] = cid
            placeholders = ",".join("?" for _ in cols)
            conn.execute(
                f"INSERT OR REPLACE INTO companies ({', '.join(cols)}) VALUES ({placeholders})",
                tuple(payload.get(k) for k in cols),
            )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# url_index classification (rebuild statuses from artifacts)
# --------------------------------------------------------------------------------------


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


def compute_company_status_from_url_index(
    index: Dict[str, Any],
) -> Tuple[str, int, int, int]:
    if not index:
        return STATUS_PENDING, 0, 0, 0

    urls_total = 0
    md_done = 0
    llm_done = 0
    for raw_ent in index.values():
        urls_total += 1
        ent = raw_ent if isinstance(raw_ent, dict) else {}
        m, l = _classify_url_entry(ent)
        if m:
            md_done += 1
        if l:
            llm_done += 1

    if urls_total == 0:
        return STATUS_PENDING, 0, 0, 0

    if llm_done == urls_total:
        return STATUS_LLM_DONE, urls_total, md_done, llm_done
    if llm_done > 0:
        return STATUS_LLM_NOT_DONE, urls_total, md_done, llm_done
    if md_done == urls_total:
        return STATUS_MD_DONE, urls_total, md_done, llm_done
    return STATUS_MD_NOT_DONE, urls_total, md_done, llm_done


def read_url_index_file(path: Path) -> Dict[str, Any]:
    obj = _read_json(path)
    return obj if isinstance(obj, dict) else {}


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

    sraw = _read_json(sp)
    qraw = _read_json(qp)
    state = sraw if isinstance(sraw, dict) else {}
    quarantine = qraw if isinstance(qraw, dict) else {}
    return state, quarantine, sp, qp, lp


def write_retry_state(
    root: Path, state: Dict[str, Any], quarantine: Dict[str, Any]
) -> None:
    rdir = retry_dir(root)
    rdir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(rdir / RETRY_STATE_NAME, state, pretty=False)
    _atomic_write_json(rdir / QUARANTINE_NAME, quarantine, pretty=False)
    # ledger is optional and append-only; we do not create by default


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
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = str(obj.get("company_id") or obj.get("bvdid") or "").strip()
            if cid and cid in keep_ids:
                out.write(
                    json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                kept += 1
    return kept


def merge_retry_states(
    a_state: Dict[str, Any],
    b_state: Dict[str, Any],
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


def merge_quarantine(
    a_q: Dict[str, Any],
    b_q: Dict[str, Any],
) -> Dict[str, Any]:
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

        # prefer later "updated_at" / "quarantined_at" if exists
        def _ts(x: Dict[str, Any]) -> float:
            for k in ("updated_at", "quarantined_at", "at", "ts"):
                if k in x:
                    try:
                        return float(x.get(k) or 0.0)
                    except Exception:
                        pass
            return 0.0

        if _ts(eb) >= _ts(ea):
            out[cid] = eb
    return out


# --------------------------------------------------------------------------------------
# File-tree sync and url_index merge
# --------------------------------------------------------------------------------------


def _file_better(src: Path, dst: Path) -> bool:
    """
    Decide if src should overwrite dst.
    Prefer larger file; tie-breaker newer mtime.
    """
    try:
        ss = src.stat()
        ds = dst.stat()
        if ss.st_size != ds.st_size:
            return ss.st_size > ds.st_size
        return ss.st_mtime_ns > ds.st_mtime_ns
    except Exception:
        return True


def merge_url_index_files(dst_path: Path, src_path: Path) -> None:
    """
    Merge url_index.json dicts with per-URL preference:
      score(llm_done)=2, score(md_done)=1
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

    merged = dict(dst)
    for url, raw_ent in src.items():
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
            # tie: shallow-merge preferring non-empty fields from src
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
        # overwrite
        if dst.exists():
            shutil.rmtree(dst)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copytree(str(src), str(dst), dirs_exist_ok=False)
        return

    # merge mode
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for d in dirs:
            (out_dir / d).mkdir(parents=True, exist_ok=True)

        for fn in files:
            sp = Path(root) / fn
            dp = out_dir / fn

            if fn == "url_index.json":
                try:
                    merge_url_index_files(dp, sp)
                    continue
                except Exception:
                    # fallback to overwrite decision
                    pass

            if not dp.exists():
                shutil.copy2(sp, dp)
            else:
                if _file_better(sp, dp):
                    shutil.copy2(sp, dp)

    if move:
        shutil.rmtree(src, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Global state recompute (DB-only, minimal mirror)
# --------------------------------------------------------------------------------------


def recompute_global_state_from_db(root: Path) -> Dict[str, Any]:
    db = crawl_db_path(root)
    if not db.exists():
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_root": str(root.resolve()),
            "db_path": str(db.resolve()),
            "total_companies": 0,
            "crawled_companies": 0,
            "completed_companies": 0,
            "percentage_completed": 0.0,
            "by_status": {},
            "unknown_statuses": {},
            "pending_companies": [],
            "in_progress_companies": [],
            "done_companies": [],
            "terminal_done_companies": [],
            "terminal_done_by_reason": {},
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
            "SELECT bvdid,status,urls_total,urls_markdown_done,urls_llm_done,done_reason FROM companies"
        ).fetchall()
        run_row = None
        try:
            run_row = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        except Exception:
            run_row = None
    finally:
        conn.close()

    known = {
        STATUS_PENDING,
        STATUS_MD_NOT_DONE,
        STATUS_MD_DONE,
        STATUS_LLM_NOT_DONE,
        STATUS_LLM_DONE,
        STATUS_TERMINAL_DONE,
    }
    by_status: Dict[str, int] = {k: 0 for k in known}
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
        cid = str(r["bvdid"])
        st_raw = r["status"]
        st = (
            str(st_raw).strip().lower() if st_raw is not None else ""
        ) or STATUS_PENDING

        urls_total_sum += int(r["urls_total"] or 0)
        urls_md_done_sum += int(r["urls_markdown_done"] or 0)
        urls_llm_done_sum += int(r["urls_llm_done"] or 0)

        if st in known:
            by_status[st] = by_status.get(st, 0) + 1
        else:
            unknown_statuses[st] = unknown_statuses.get(st, 0) + 1
            st = STATUS_PENDING
            by_status[st] = by_status.get(st, 0) + 1

        if st == STATUS_PENDING:
            pending.append(cid)
        elif st in (STATUS_MD_NOT_DONE, STATUS_LLM_NOT_DONE):
            in_progress.append(cid)
        elif st == STATUS_TERMINAL_DONE:
            done.append(cid)
            terminal_done.append(cid)
            dr = (r["done_reason"] or "unknown").strip()
            terminal_reasons[dr] = terminal_reasons.get(dr, 0) + 1
        else:
            done.append(cid)

    total = len(rows)
    crawled = total - len(pending)
    completed = len(done)
    pct = (completed / total * 100.0) if total else 0.0

    latest_run = None
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

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(root.resolve()),
        "db_path": str(db.resolve()),
        "total_companies": total,
        "crawled_companies": crawled,
        "completed_companies": completed,
        "percentage_completed": pct,
        "by_status": by_status,
        "unknown_statuses": unknown_statuses,
        "pending_companies": pending,
        "in_progress_companies": in_progress,
        "done_companies": done,
        "terminal_done_companies": terminal_done,
        "terminal_done_by_reason": terminal_reasons,
        "urls_total_sum": urls_total_sum,
        "urls_markdown_done_sum": urls_md_done_sum,
        "urls_llm_done_sum": urls_llm_done_sum,
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
        st = str((row or {}).get("status") or STATUS_PENDING).strip().lower()
        if only_not_done and st in _DONE_STATUSES_DEFAULT:
            continue
        candidates.append(cid)

    if move_ids is not None:
        chosen = [cid for cid in candidates if cid in move_ids]
    else:
        n = int(move_count or 0)
        if n <= 0:
            raise ValueError("move_count must be > 0 when move_ids not provided.")
        # deterministic shuffle for fair distribution
        rnd = random.Random(int(seed))
        pool = list(candidates)
        rnd.shuffle(pool)
        chosen = pool[: min(n, len(pool))]

    move_set = set(chosen)
    remain = [cid for cid in dataset_ids if cid not in move_set]
    return SplitPlan(move_ids=chosen, remain_ids=remain)


def rebuild_db_from_outputs(
    root: Path,
    company_ids: Sequence[str],
    *,
    prefer_existing_rows: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Build a fresh crawl_state.sqlite3 in root from url_index.json under each company dir.
    Preserves name/root_url if available in prefer_existing_rows.
    """
    prefer_existing_rows = prefer_existing_rows or {}

    tmp = root / f"{CRAWL_DB_NAME}.tmp.{int(time.time())}"
    init_crawl_db(tmp)

    out_rows: Dict[str, Dict[str, Any]] = {}

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for cid in company_ids:
        base = prefer_existing_rows.get(cid, {})
        name = base.get("name")
        root_url = base.get("root_url")

        cdir = find_company_dir(root, cid)
        status = STATUS_PENDING
        ut = 0
        md = 0
        llm = 0

        if cdir is not None:
            ip = url_index_path_in_company_dir(cdir)
            if ip.exists():
                idx = read_url_index_file(ip)
                status, ut, md, llm = compute_company_status_from_url_index(idx)

        row = {
            "bvdid": cid,
            "name": name,
            "root_url": root_url,
            "status": status,
            "urls_total": int(ut),
            "urls_markdown_done": int(md),
            "urls_llm_done": int(llm),
            "last_error": base.get("last_error"),
            "done_reason": base.get("done_reason"),
            "done_details": base.get("done_details"),
            "done_at": base.get("done_at"),
            "created_at": base.get("created_at") or now_iso,
            "updated_at": now_iso,
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
        s: Set[str] = set()
        for line in move_ids_file.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t:
                s.add(t)
        move_ids = s

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

    # Prepare bundle root
    bundle_root.mkdir(parents=True, exist_ok=True)

    # 1) Write new dataset files
    write_table_preserve(remain_dataset_out, table, remain_rows)
    write_table_preserve(moved_dataset_out, table, moved_rows)

    # Also drop a copy into bundle_root for convenience
    bundle_dataset_default = bundle_root / dataset_path.name
    try:
        write_table_preserve(bundle_dataset_default, table, moved_rows)
    except Exception:
        pass

    # 2) Move/copy company dirs
    do_move = move_mode.strip().lower() == "move"
    for cid in plan.move_ids:
        src_dir = find_company_dir(root, cid)
        if src_dir is None:
            continue
        dst_dir = expected_company_dir(bundle_root, cid)
        sync_tree(src_dir, dst_dir, move=do_move, merge=True)

    # 3) Subset retry state into bundle and (optionally) remove from source
    src_state, src_q, _, _, src_ledger = _load_retry_state(root)
    move_state = {cid: v for cid, v in src_state.items() if cid in move_set}
    move_q = {cid: v for cid, v in src_q.items() if cid in move_set}

    write_retry_state(bundle_root, move_state, move_q)

    if include_ledger:
        kept = filter_ledger(src_ledger, retry_dir(bundle_root) / LEDGER_NAME, move_set)
        report["bundle_ledger_rows"] = kept

    if do_move:
        # remove moved retry entries from source
        remain_state = {cid: v for cid, v in src_state.items() if cid not in move_set}
        remain_q = {cid: v for cid, v in src_q.items() if cid not in move_set}
        write_retry_state(root, remain_state, remain_q)
        # (ledger stays append-only in source; we do not delete)

    # 4) Subset crawl_state sqlite into bundle and remove from source (if move)
    src_db = crawl_db_path(root)
    src_rows = read_companies_rows(src_db)

    bundle_rows = {cid: src_rows[cid] for cid in plan.move_ids if cid in src_rows}
    # If some moved IDs have no row, create a minimal pending row
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for cid in plan.move_ids:
        if cid not in bundle_rows:
            bundle_rows[cid] = {
                "bvdid": cid,
                "name": None,
                "root_url": None,
                "status": STATUS_PENDING,
                "urls_total": 0,
                "urls_markdown_done": 0,
                "urls_llm_done": 0,
                "last_error": None,
                "done_reason": None,
                "done_details": None,
                "done_at": None,
                "created_at": now_iso,
                "updated_at": now_iso,
            }

    write_companies_rows(crawl_db_path(bundle_root), bundle_rows)

    if do_move:
        deleted = delete_company_rows(src_db, plan.move_ids)
        report["source_db_deleted_rows"] = deleted

    # 5) Rebuild bundle DB from copied outputs (more accurate), and recompute globals
    try:
        rebuild_db_from_outputs(
            bundle_root, plan.move_ids, prefer_existing_rows=bundle_rows
        )
    except Exception as e:
        report["bundle_db_rebuild_error"] = str(e)

    recompute_global_state_from_db(bundle_root)
    recompute_global_state_from_db(root)

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

    # 1) Merge company dirs
    all_company_ids: Set[str] = set()
    prefer_rows: Dict[str, Dict[str, Any]] = {}

    for src_root in srcs:
        # collect ids from db if present
        rows = read_companies_rows(crawl_db_path(src_root))
        for cid, r in rows.items():
            all_company_ids.add(cid)
            # keep first non-null name/root_url
            if cid not in prefer_rows:
                prefer_rows[cid] = dict(r)
            else:
                cur = prefer_rows[cid]
                if not cur.get("name") and r.get("name"):
                    cur["name"] = r.get("name")
                if not cur.get("root_url") and r.get("root_url"):
                    cur["root_url"] = r.get("root_url")

        # merge dirs by iterating ids (limits what we copy)
        for cid in rows.keys():
            src_dir = find_company_dir(src_root, cid)
            if src_dir is None:
                continue
            dst_dir = expected_company_dir(target_root, cid)
            sync_tree(src_dir, dst_dir, move=do_move, merge=True)

    report["company_ids_merged"] = len(all_company_ids)

    # 2) Merge retry state
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
    recompute_global_state_from_db(target_root)

    # 4) If moving, sources still have their db/global/retry; we intentionally do not delete those
    # (user might want to keep their original run roots as backups).
    return report
