from __future__ import annotations

import csv
import errno
import json
import logging
import os
import random
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from configs.models import (
    CompanyStatus,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_TERMINAL_DONE,
    URL_INDEX_META_KEY,
    UrlIndexEntry,
    UrlIndexMeta,
)
from extensions.crawl.state import CrawlState
from extensions.io.output_paths import sanitize_bvdid

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

GLOBAL_STATE_NAME = "crawl_global_state.json"
CRAWL_DB_NAME = "crawl_state.sqlite3"

RETRY_DIR_NAME = "_retry"
RETRY_STATE_NAME = "retry_state.json"
QUARANTINE_NAME = "quarantine.json"

# Support both layouts:
#   <root>/failure_ledger.jsonl
#   <root>/_retry/failure_ledger.jsonl
LEDGER_NAME = "failure_ledger.jsonl"

URL_INDEX_NAME = "url_index.json"
URL_INDEX_RESERVED_PREFIX = "__"

_STATUS_RANK: Dict[str, int] = {
    COMPANY_STATUS_PENDING: 0,
    COMPANY_STATUS_MD_NOT_DONE: 1,
    COMPANY_STATUS_MD_DONE: 2,
    COMPANY_STATUS_LLM_NOT_DONE: 3,
    COMPANY_STATUS_LLM_DONE: 4,
    COMPANY_STATUS_TERMINAL_DONE: 5,
}

# Matches crawl.state.py’s “in progress” notion.
IN_PROGRESS_STATUSES: Tuple[CompanyStatus, ...] = (
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
)

DONE_STATUSES: Tuple[CompanyStatus, ...] = (
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_TERMINAL_DONE,
)

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
    raise RuntimeError("Too many open files (EMFILE/ENFILE) persisted across retries")


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
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
                tmp.unlink(missing_ok=True)

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
# Concurrency helpers
# --------------------------------------------------------------------------------------


def _default_workers() -> int:
    # IO-bound; don’t go crazy by default.
    c = os.cpu_count() or 4
    return max(1, min(16, c * 2))


def _resolve_workers(workers: Optional[int]) -> int:
    if workers is None:
        return _default_workers()
    w = int(workers)
    if w <= 0:
        return _default_workers()
    return w


def _run_threaded(label: str, items: Sequence[Any], fn, *, workers: int) -> None:
    """
    Run fn(item) for each item concurrently with ThreadPoolExecutor.

    - Raises first exception, after all futures completed.
    - Keeps concurrency bounded.
    """
    if not items:
        return
    w = max(1, int(workers))

    # Avoid oversubscribing for tiny worklists.
    w = min(w, len(items))

    t0 = time.time()
    logger.info("%s: starting %d tasks with workers=%d", label, len(items), w)

    errors: List[BaseException] = []
    with ThreadPoolExecutor(max_workers=w, thread_name_prefix="partition") as ex:
        futs = [ex.submit(fn, it) for it in items]
        for fut in as_completed(futs):
            try:
                fut.result()
            except BaseException as e:
                errors.append(e)

    dt = time.time() - t0
    if errors:
        logger.error(
            "%s: %d/%d tasks failed (elapsed=%.2fs)", label, len(errors), len(items), dt
        )
        raise errors[0]
    logger.info("%s: completed %d tasks (elapsed=%.2fs)", label, len(items), dt)


# --------------------------------------------------------------------------------------
# Timestamp parsing (DB rows may store float seconds OR ISO8601 strings)
# --------------------------------------------------------------------------------------


def _to_epoch_seconds(v: Any) -> float:
    """
    Best-effort normalization for CrawlState DB row timestamps.

    Supported:
      - None / "" -> 0.0
      - int/float -> float(v)
      - numeric strings -> float(...)
      - ISO8601 strings (with/without timezone) -> epoch seconds
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    if not s:
        return 0.0

    # numeric string
    try:
        return float(s)
    except ValueError:
        pass

    # ISO8601
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return 0.0

    if dt.tzinfo is None:
        # treat naive as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


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


_ID_CANDIDATES = ["company_id", "bvdid", "id", "BVDID", "BvDID", "BVD_ID", "bvd_id"]
_URL_CANDIDATES = ["root_url", "domain_url", "url", "website", "web", "homepage"]


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


def read_table_preserve(path: Path, *, id_col_override: Optional[str] = None) -> Table:
    delim = _detect_delimiter(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError(f"Dataset file has no header: {path}")

        if id_col_override is not None:
            if id_col_override not in header:
                raise ValueError(
                    f"--id-col={id_col_override!r} not found in header: {header}"
                )
            id_col = id_col_override
        else:
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


def group_rows_by_company_id(table: Table) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in table.rows:
        cid = (r.get(table.id_col) or "").strip()
        if not cid:
            continue
        groups.setdefault(cid, []).append(r)
    return groups


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
    add: Table,
    add_rows: List[Dict[str, str]],
) -> Tuple[Table, List[Dict[str, str]]]:
    """
    Merge two tables and dedupe by base.id_col.

    - Output delimiter/header order follows base.
    - Header is base.header + new columns from add.header (in add order).
    - Deduping keeps first-seen company_id from base_rows, then fills missing ids from add_rows.
    """
    base_id = base.id_col
    header = list(base.header)
    seen = set(header)
    for c in add.header:
        if c not in seen:
            header.append(c)
            seen.add(c)

    def _normalize_row(r: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k in header:
            v = r.get(k)
            out[k] = "" if v is None else str(v)
        return out

    out_rows: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()

    for r in base_rows:
        cid = (r.get(base_id) or "").strip()
        if not cid:
            continue
        if cid in seen_ids:
            continue
        out_rows.append(_normalize_row(r))
        seen_ids.add(cid)

    add_id_col = add.id_col
    for r in add_rows:
        cid = (r.get(add_id_col) or "").strip()
        if not cid:
            continue
        if cid in seen_ids:
            continue
        out_rows.append(_normalize_row(r))
        seen_ids.add(cid)

    merged = Table(
        header=header,
        rows=out_rows,
        id_col=base.id_col,
        url_col=base.url_col,
        delimiter=base.delimiter,
    )
    return merged, out_rows


# --------------------------------------------------------------------------------------
# Partitioning helpers (deterministic, balanced)
# --------------------------------------------------------------------------------------


def sample_ids(ids: Sequence[str], k: int, *, seed: int) -> List[str]:
    uniq = list(dict.fromkeys([str(x).strip() for x in ids if str(x).strip()]))
    if k < 0:
        raise ValueError("k must be >= 0")
    if k > len(uniq):
        raise ValueError(f"Requested k={k} but only {len(uniq)} ids are available")
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    return uniq[:k]


# --------------------------------------------------------------------------------------
# Run-root paths / company dirs
# --------------------------------------------------------------------------------------


def crawl_db_path(root: Path) -> Path:
    return root / CRAWL_DB_NAME


def global_state_path(root: Path) -> Path:
    return root / GLOBAL_STATE_NAME


def retry_dir(root: Path) -> Path:
    return root / RETRY_DIR_NAME


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


def _company_meta_dir(company_dir: Path) -> Path:
    m1 = company_dir / "metadata"
    if m1.exists():
        return m1
    m2 = company_dir / "checkpoints"
    if m2.exists():
        return m2
    return company_dir / "metadata"


def url_index_path_in_company_dir(company_dir: Path) -> Path:
    return _company_meta_dir(company_dir) / URL_INDEX_NAME


def _ledger_paths(root: Path) -> List[Path]:
    # currently we only support the canonical layout used by the crawler:
    p = retry_dir(root) / LEDGER_NAME
    out: List[Path] = []
    if p.exists() and p not in out:
        out.append(p)
    return out


def _target_ledger_path(root: Path) -> Path:
    return retry_dir(root) / LEDGER_NAME


# --------------------------------------------------------------------------------------
# url_index.json merge (uses configs.models + crawl.state semantics)
# --------------------------------------------------------------------------------------


def _is_reserved_url_index_key(k: Any) -> bool:
    return str(k).startswith(URL_INDEX_RESERVED_PREFIX)


def _classify_url_entry_like_crawl_state(ent: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Mirrors crawl.state.py::_classify_url_entry.

    markdown_done = md_flag OR markdown_path OR status==markdown_done OR llm_flag OR llm_artifacts OR status==llm_done
    llm_done = llm_flag OR llm_artifacts OR status==llm_done
    """
    status = str(ent.get("status") or "").strip()
    has_md_path = bool(ent.get("markdown_path"))
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


def _json_obj(path: Path) -> Dict[str, Any]:
    obj = _read_json_strict(path)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object")
    return obj


def merge_url_index_files(dst_path: Path, src_path: Path) -> None:
    """
    Merge url_index.json dicts with per-URL preference:
      score(llm_done)=2, score(md_done)=1
    __meta__ is normalized via UrlIndexMeta.
    Entries are normalized via UrlIndexEntry to preserve extra fields.
    """
    src = _json_obj(src_path) if src_path.exists() else {}
    if not src:
        return

    dst = _json_obj(dst_path) if dst_path.exists() else {}
    if not dst:
        _atomic_write_json(dst_path, src, pretty=False)
        return

    merged: Dict[str, Any] = dict(dst)

    # __meta__
    if URL_INDEX_META_KEY in src or URL_INDEX_META_KEY in dst:
        md = dst.get(URL_INDEX_META_KEY)
        ms = src.get(URL_INDEX_META_KEY)
        mdict = dict(md) if isinstance(md, dict) else {}
        sdict = dict(ms) if isinstance(ms, dict) else {}

        # OR crawl_finished
        mdict["crawl_finished"] = bool(mdict.get("crawl_finished")) or bool(
            sdict.get("crawl_finished")
        )

        for k, v in sdict.items():
            if k == "crawl_finished":
                continue
            if k not in mdict or mdict.get(k) in ("", None, [], {}):
                mdict[k] = v

        cid = str(mdict.get("company_id") or sdict.get("company_id") or "").strip()
        normalized_meta = UrlIndexMeta.from_dict(mdict, company_id=cid or "unknown")
        merged[URL_INDEX_META_KEY] = normalized_meta.to_dict()

    def _score(ent: Dict[str, Any]) -> int:
        md_done, llm_done = _classify_url_entry_like_crawl_state(ent)
        return (2 if llm_done else 0) + (1 if md_done else 0)

    for url, raw_ent in src.items():
        if _is_reserved_url_index_key(url):
            continue
        ent = raw_ent if isinstance(raw_ent, dict) else {}

        if url not in merged:
            cid = str(ent.get("company_id") or "").strip()
            merged[url] = UrlIndexEntry.from_dict(
                ent, company_id=cid or "unknown", url=str(url)
            ).to_dict()
            continue

        cur_raw = merged.get(url)
        cur_ent = cur_raw if isinstance(cur_raw, dict) else {}

        s_new = _score(ent)
        s_cur = _score(cur_ent)

        if s_new > s_cur:
            cid = str(ent.get("company_id") or cur_ent.get("company_id") or "").strip()
            merged[url] = UrlIndexEntry.from_dict(
                ent, company_id=cid or "unknown", url=str(url)
            ).to_dict()
            continue

        if s_new == s_cur:
            out_ent = dict(cur_ent)
            for k, v in ent.items():
                if k not in out_ent or out_ent.get(k) in ("", None, [], {}):
                    out_ent[k] = v
            cid = str(out_ent.get("company_id") or "").strip()
            merged[url] = UrlIndexEntry.from_dict(
                out_ent, company_id=cid or "unknown", url=str(url)
            ).to_dict()

    _atomic_write_json(dst_path, merged, pretty=False)


# --------------------------------------------------------------------------------------
# File-tree sync (url_index aware)
# --------------------------------------------------------------------------------------


def _file_better(src: Path, dst: Path) -> bool:
    ss = src.stat()
    ds = dst.stat()
    if ss.st_size != ds.st_size:
        return ss.st_size > ds.st_size
    return ss.st_mtime_ns > ds.st_mtime_ns


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
# Retry state subset/merge (raw JSON files)
# --------------------------------------------------------------------------------------


def _load_retry_state(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    rdir = retry_dir(root)
    sp = rdir / RETRY_STATE_NAME
    qp = rdir / QUARANTINE_NAME

    state_raw = _read_json_strict(sp) if sp.exists() else None
    quarantine_raw = _read_json_strict(qp) if qp.exists() else None

    state = state_raw if isinstance(state_raw, dict) else {}
    quarantine = quarantine_raw if isinstance(quarantine_raw, dict) else {}
    return state, quarantine, sp, qp


def write_retry_state(
    root: Path, state: Dict[str, Any], quarantine: Dict[str, Any]
) -> None:
    rdir = retry_dir(root)
    rdir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(rdir / RETRY_STATE_NAME, state, pretty=False)
    _atomic_write_json(rdir / QUARANTINE_NAME, quarantine, pretty=False)


def filter_ledger_lines(src_ledger: Path, dst_ledger: Path, keep_ids: Set[str]) -> int:
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


def append_ledger_all(src_ledger: Path, dst_ledger: Path) -> int:
    if not src_ledger.exists():
        return 0
    dst_ledger.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with (
        open(src_ledger, "r", encoding="utf-8", errors="ignore") as f_in,
        open(dst_ledger, "a", encoding="utf-8", newline="") as f_out,
    ):
        for line in f_in:
            if not line.strip():
                continue
            f_out.write(line if line.endswith("\n") else (line + "\n"))
            n += 1
    return n


def merge_retry_states(
    a_state: Dict[str, Any], b_state: Dict[str, Any]
) -> Dict[str, Any]:
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

        ua = _to_epoch_seconds(ea.get("updated_at"))
        ub = _to_epoch_seconds(eb.get("updated_at"))
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

    def _ts(x: Dict[str, Any]) -> float:
        for k in ("updated_at", "quarantined_at", "at", "ts"):
            if k in x:
                return _to_epoch_seconds(x.get(k))
        return 0.0

    for cid, ent in b_q.items():
        if cid not in out:
            out[cid] = ent
            continue
        ea = out.get(cid)
        eb = ent
        if not isinstance(ea, dict) or not isinstance(eb, dict):
            out[cid] = eb
            continue
        if _ts(eb) >= _ts(ea):
            out[cid] = eb

    return out


# --------------------------------------------------------------------------------------
# SQLite access (schema matches extensions.crawl.state.CrawlState)
# --------------------------------------------------------------------------------------

COMPANIES_COLS: Tuple[str, ...] = (
    "company_id",
    "root_url",
    "name",
    "metadata_json",
    "industry",
    "nace",
    "industry_label",
    "industry_label_source",
    "status",
    "crawl_finished",
    "urls_total",
    "urls_markdown_done",
    "urls_llm_done",
    "last_error",
    "done_reason",
    "done_details",
    "done_at",
    "created_at",
    "updated_at",
    "last_crawled_at",
    "max_pages",
    "retry_cls",
    "retry_attempts",
    "retry_next_eligible_at",
    "retry_updated_at",
    "retry_last_error",
    "retry_last_stage",
    "retry_net_attempts",
    "retry_stall_attempts",
    "retry_mem_attempts",
    "retry_other_attempts",
    "retry_mem_hits",
    "retry_last_stall_kind",
    "retry_last_progress_md_done",
    "retry_last_seen_md_done",
    "retry_last_error_sig",
    "retry_same_error_streak",
    "retry_last_error_sig_updated_at",
)

RUNS_COLS: Tuple[str, ...] = (
    "run_id",
    "pipeline",
    "version",
    "args_hash",
    "crawl4ai_cache_base_dir",
    "crawl4ai_cache_mode",
    "started_at",
    "total_companies",
    "completed_companies",
    "last_company_id",
    "last_updated",
)

RUN_COMPANY_DONE_COLS: Tuple[str, ...] = (
    "run_id",
    "company_id",
    "done_at",
)


def _open_db(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def ensure_db_initialized(db: Path) -> None:
    cs = CrawlState(db_path=db)
    cs.close()


def read_company_status_map(
    db: Path,
    *,
    include_statuses: Optional[Sequence[str]] = None,
    exclude_statuses: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    if not db.exists():
        return {}

    inc = [str(x).strip() for x in (include_statuses or []) if str(x).strip()]
    exc = [str(x).strip() for x in (exclude_statuses or []) if str(x).strip()]

    sql = "SELECT company_id, status FROM companies"
    args: List[Any] = []
    where: List[str] = []

    if inc:
        where.append(f"status IN ({','.join('?' for _ in inc)})")
        args.extend(inc)
    if exc:
        where.append(f"status NOT IN ({','.join('?' for _ in exc)})")
        args.extend(exc)

    if where:
        sql += " WHERE " + " AND ".join(where)

    conn = _open_db(db)
    try:
        rows = conn.execute(sql, tuple(args)).fetchall()
        out: Dict[str, str] = {}
        for r in rows:
            cid = str(r["company_id"] or "").strip()
            if cid:
                out[cid] = str(r["status"] or "").strip()
        return out
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
            cid = str(d.get("company_id") or "").strip()
            if cid:
                out[cid] = d
        return out
    finally:
        conn.close()


def read_runs_rows(db: Path) -> List[Dict[str, Any]]:
    if not db.exists():
        return []
    conn = _open_db(db)
    try:
        rows = conn.execute("SELECT * FROM runs").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def read_run_company_done_rows(db: Path) -> List[Dict[str, Any]]:
    if not db.exists():
        return []
    conn = _open_db(db)
    try:
        rows = conn.execute("SELECT * FROM run_company_done").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def write_companies_rows(db: Path, rows: Dict[str, Dict[str, Any]]) -> None:
    ensure_db_initialized(db)
    conn = _open_db(db)
    try:
        placeholders = ",".join("?" for _ in COMPANIES_COLS)
        conn.execute("BEGIN;")
        for cid, r in rows.items():
            payload = dict(r)
            payload["company_id"] = cid
            conn.execute(
                f"INSERT OR REPLACE INTO companies ({', '.join(COMPANIES_COLS)}) VALUES ({placeholders})",
                tuple(payload.get(k) for k in COMPANIES_COLS),
            )
        conn.commit()
    finally:
        conn.close()


def write_runs_rows(db: Path, runs: List[Dict[str, Any]]) -> None:
    ensure_db_initialized(db)
    conn = _open_db(db)
    try:
        placeholders = ",".join("?" for _ in RUNS_COLS)
        conn.execute("BEGIN;")
        for r in runs:
            payload = dict(r)
            conn.execute(
                f"INSERT OR IGNORE INTO runs ({', '.join(RUNS_COLS)}) VALUES ({placeholders})",
                tuple(payload.get(k) for k in RUNS_COLS),
            )
        conn.commit()
    finally:
        conn.close()


def write_run_company_done_rows(db: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_db_initialized(db)
    conn = _open_db(db)
    try:
        placeholders = ",".join("?" for _ in RUN_COMPANY_DONE_COLS)
        conn.execute("BEGIN;")
        for r in rows:
            payload = dict(r)
            conn.execute(
                f"INSERT OR IGNORE INTO run_company_done ({', '.join(RUN_COMPANY_DONE_COLS)}) VALUES ({placeholders})",
                tuple(payload.get(k) for k in RUN_COMPANY_DONE_COLS),
            )
        conn.commit()
    finally:
        conn.close()


def delete_company_rows(db: Path, company_ids: Sequence[str]) -> int:
    if (not db.exists()) or (not company_ids):
        return 0
    ids = [str(x).strip() for x in company_ids if str(x).strip()]
    if not ids:
        return 0

    conn = _open_db(db)
    try:
        n = 0
        chunk = 500
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            q = ",".join("?" for _ in part)
            cur = conn.execute(
                f"DELETE FROM companies WHERE company_id IN ({q})", tuple(part)
            )
            n += int(cur.rowcount or 0)
        conn.commit()
        return n
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# Global state writer (DB-only) for arbitrary run roots
#   IMPORTANT: matches crawl.state.py payload shape (NO by_status)
# --------------------------------------------------------------------------------------


def _normalize_status(st: Any) -> CompanyStatus:
    s = str(st or "").strip()
    if s in _STATUS_RANK:
        return s  # type: ignore[return-value]
    return COMPANY_STATUS_PENDING


def write_global_state_from_db_only(
    root: Path, *, max_ids: int = 200, pretty: bool = False
) -> Dict[str, Any]:
    """
    Write <root>/crawl_global_state.json using ONLY the DB content in <root>/crawl_state.sqlite3.

    Payload shape intentionally matches CrawlState.write_global_state_from_db_only in crawl.state.py:
      - NO "by_status"
      - includes company_ids_sample, *_pct, urls_*_sum, latest_run, terminal_done_reasons (optional)
    """
    db = crawl_db_path(root)
    ensure_db_initialized(db)

    conn = _open_db(db)
    try:
        rows = conn.execute(
            """
            SELECT company_id, status, crawl_finished,
                   urls_total, urls_markdown_done, urls_llm_done,
                   done_reason
              FROM companies
            """
        ).fetchall()

        run_row = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

        run_done_count: Optional[int] = None
        if run_row is not None:
            rrid = run_row["run_id"]
            c = conn.execute(
                "SELECT COUNT(*) AS c FROM run_company_done WHERE run_id=?",
                (rrid,),
            ).fetchone()
            run_done_count = int(c["c"] or 0) if c is not None else None
    finally:
        conn.close()

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
        cid = str(r["company_id"] or "").strip()
        if not cid:
            continue

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
            dr = str(r["done_reason"] or "unknown").strip() or "unknown"
            terminal_done_reasons[dr] = terminal_done_reasons.get(dr, 0) + 1

        if bool(int(r["crawl_finished"] or 0)):
            crawl_finished_companies += 1

        urls_total_sum += int(r["urls_total"] or 0)
        urls_md_done_sum += int(r["urls_markdown_done"] or 0)
        urls_llm_done_sum += int(r["urls_llm_done"] or 0)

    crawled_companies = total - pending_companies

    md_done_pct = (md_done_companies / total * 100.0) if total else 0.0
    llm_done_pct = (llm_done_companies / total * 100.0) if total else 0.0
    crawl_finished_pct = (crawl_finished_companies / total * 100.0) if total else 0.0

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
        "output_root": str(root.resolve()),
        "db_path": str(crawl_db_path(root).resolve()),
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

    _atomic_write_json(global_state_path(root), payload, pretty=pretty)
    return payload


# --------------------------------------------------------------------------------------
# Split / Merge run roots (workflows from usage doc)
# --------------------------------------------------------------------------------------


def _status_rank(st: Any) -> int:
    s = str(st or "").strip()
    return _STATUS_RANK.get(s, -1)


def _pick_better_company_row(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer higher status rank, then higher updated_at.

    updated_at may be:
      - float seconds
      - ISO8601 string (e.g. '2026-01-04T15:28:31.779492+00:00')
    """
    sa = _status_rank(a.get("status"))
    sb = _status_rank(b.get("status"))
    if sb > sa:
        return b
    if sa > sb:
        return a

    ua = _to_epoch_seconds(a.get("updated_at"))
    ub = _to_epoch_seconds(b.get("updated_at"))
    if ub > ua:
        return b
    return a


def _read_ids_file(path: Path) -> List[str]:
    ids: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s)
    return list(dict.fromkeys(ids))  # keep order, dedupe


def _validate_run_root(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Run root not found: {root}")
    db = crawl_db_path(root)
    if not db.exists():
        raise ValueError(f"Run root missing crawl_state.sqlite3: {db}")


def split_run_root(
    *,
    root: Path,
    dataset: Path,
    bundle_root: Path,
    remaining_dataset_out: Path,
    moved_dataset_out: Path,
    move_count: Optional[int],
    move_ids_file: Optional[Path],
    seed: int = 7,
    only_not_done: bool = False,
    move_mode: str = "move",  # move|copy
    include_ledger: bool = False,
    dry_run: bool = False,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Split a run root into a bundle root + remaining/moved dataset files.

    Concurrency:
      - Company directory sync is executed in parallel with `workers`.
    """
    root = Path(root)
    dataset = Path(dataset)
    bundle_root = Path(bundle_root)
    remaining_dataset_out = Path(remaining_dataset_out)
    moved_dataset_out = Path(moved_dataset_out)

    _validate_run_root(root)
    if move_mode not in ("move", "copy"):
        raise ValueError("--move-mode must be move or copy")

    if (move_count is None) == (move_ids_file is None):
        raise ValueError("Exactly one of --move-count or --move-ids-file is required")

    table = read_table_preserve(dataset)
    dataset_ids = sorted(
        {
            (r.get(table.id_col) or "").strip()
            for r in table.rows
            if (r.get(table.id_col) or "").strip()
        }
    )

    db = crawl_db_path(root)
    status_map = read_company_status_map(db)
    db_ids = set(status_map.keys())

    candidates = [cid for cid in dataset_ids if cid in db_ids]
    missing_in_db = [cid for cid in dataset_ids if cid not in db_ids]

    if only_not_done:
        candidates = [
            cid for cid in candidates if status_map.get(cid) not in DONE_STATUSES
        ]

    if move_ids_file is not None:
        requested = _read_ids_file(Path(move_ids_file))
        unknown = [cid for cid in requested if cid not in set(candidates)]
        if unknown:
            raise ValueError(
                f"Some requested ids are not eligible candidates: {unknown[:20]}"
            )
        chosen = requested
    else:
        assert move_count is not None
        chosen = sample_ids(candidates, int(move_count), seed=int(seed))

    chosen_set = set(chosen)

    remaining_rows, moved_rows = split_table_by_ids(table, chosen_set)

    moved_row_ids = {
        (r.get(table.id_col) or "").strip()
        for r in moved_rows
        if (r.get(table.id_col) or "").strip()
    }
    missing_in_dataset_rows = [cid for cid in chosen if cid not in moved_row_ids]
    if missing_in_dataset_rows:
        raise ValueError(
            f"Selected ids missing from dataset rows: {missing_in_dataset_rows[:20]}"
        )

    plan = {
        "root": str(root),
        "dataset": str(dataset),
        "bundle_root": str(bundle_root),
        "move_mode": move_mode,
        "only_not_done": bool(only_not_done),
        "seed": int(seed),
        "selected_count": len(chosen),
        "selected_ids_sample": chosen[:20],
        "remaining_rows": len(remaining_rows),
        "moved_rows": len(moved_rows),
        "missing_ids_in_db_from_dataset_sample": missing_in_db[:20],
        "workers": _resolve_workers(workers),
    }

    if dry_run:
        return {"cmd": "split", "dry_run": True, "plan": plan}

    write_table_preserve(remaining_dataset_out, table, remaining_rows)
    write_table_preserve(moved_dataset_out, table, moved_rows)

    bundle_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(moved_dataset_out, bundle_root / moved_dataset_out.name)

    move_dirs = move_mode == "move"
    w = _resolve_workers(workers)

    def _sync_one(cid: str) -> None:
        sdir = find_company_dir(root, cid)
        if sdir is None:
            raise ValueError(f"Company directory missing for selected id: {cid}")
        ddir = expected_company_dir(bundle_root, cid)
        sync_tree(sdir, ddir, move=move_dirs, merge=True)

    _run_threaded("split:sync_company_dirs", chosen, _sync_one, workers=w)
    moved_company_dirs = len(chosen)

    ensure_db_initialized(crawl_db_path(bundle_root))
    src_companies = read_companies_rows(crawl_db_path(root))
    subset = {cid: src_companies[cid] for cid in chosen if cid in src_companies}
    write_companies_rows(crawl_db_path(bundle_root), subset)

    write_runs_rows(crawl_db_path(bundle_root), read_runs_rows(crawl_db_path(root)))
    write_run_company_done_rows(
        crawl_db_path(bundle_root), read_run_company_done_rows(crawl_db_path(root))
    )

    src_retry, src_quarantine, _, _ = _load_retry_state(root)
    dst_retry = {cid: src_retry.get(cid) for cid in chosen if cid in src_retry}
    dst_quarantine = {
        cid: src_quarantine.get(cid) for cid in chosen if cid in src_quarantine
    }
    write_retry_state(bundle_root, dst_retry, dst_quarantine)

    ledger_kept = 0
    if include_ledger:
        dst_ledger = _target_ledger_path(bundle_root)
        for lp in _ledger_paths(root):
            ledger_kept += filter_ledger_lines(lp, dst_ledger, chosen_set)

    if move_dirs:
        delete_company_rows(crawl_db_path(root), chosen)
        new_retry = {k: v for k, v in src_retry.items() if str(k) not in chosen_set}
        new_quarantine = {
            k: v for k, v in src_quarantine.items() if str(k) not in chosen_set
        }
        write_retry_state(root, new_retry, new_quarantine)

    write_global_state_from_db_only(bundle_root, pretty=False)
    write_global_state_from_db_only(root, pretty=False)

    return {
        "cmd": "split",
        "dry_run": False,
        "plan": plan,
        "outputs": {
            "remaining_dataset_out": str(remaining_dataset_out),
            "moved_dataset_out": str(moved_dataset_out),
            "bundle_moved_dataset_copy": str(bundle_root / moved_dataset_out.name),
        },
        "bundle": {
            "moved_company_dirs": moved_company_dirs,
            "ledger_lines_copied": ledger_kept,
        },
    }


def merge_run_roots(
    *,
    target_root: Path,
    source_roots: Sequence[Path],
    move_mode: str = "copy",  # copy|move
    include_ledger: bool = False,
    dry_run: bool = False,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Merge multiple run roots into one target root.

    Concurrency:
      - For each source root, company directory sync runs in parallel with `workers`.
      - Source roots are processed sequentially to avoid races when multiple sources contain the same company_id.
    """
    target_root = Path(target_root)
    srcs = [Path(s) for s in source_roots]
    if move_mode not in ("move", "copy"):
        raise ValueError("--move-mode must be move or copy")
    if not srcs:
        raise ValueError("--source-roots requires at least one path")

    for s in srcs:
        _validate_run_root(s)

    w = _resolve_workers(workers)

    if dry_run:
        return {
            "cmd": "merge",
            "dry_run": True,
            "target_root": str(target_root),
            "source_roots": [str(s) for s in srcs],
            "move_mode": move_mode,
            "include_ledger": bool(include_ledger),
            "workers": w,
        }

    target_root.mkdir(parents=True, exist_ok=True)
    ensure_db_initialized(crawl_db_path(target_root))

    tgt_db = crawl_db_path(target_root)
    tgt_companies = read_companies_rows(tgt_db)
    tgt_runs = read_runs_rows(tgt_db)
    tgt_done = read_run_company_done_rows(tgt_db)

    merged_company_ids: Set[str] = set()
    ledger_appended = 0

    # Load target retry/quarantine once and update in-memory; write after each source for resilience.
    tgt_retry, tgt_quarantine, _, _ = _load_retry_state(target_root)

    for sroot in srcs:
        sdb = crawl_db_path(sroot)
        src_companies = read_companies_rows(sdb)
        src_company_ids = list(src_companies.keys())

        # 1) company dirs + url_index merge (parallel per source)
        def _sync_one(cid: str) -> None:
            sdir = find_company_dir(sroot, cid)
            if sdir is None:
                logger.warning(
                    "Source company dir missing: root=%s company_id=%s", sroot, cid
                )
                return
            ddir = expected_company_dir(target_root, cid)
            sync_tree(sdir, ddir, move=(move_mode == "move"), merge=True)

        _run_threaded(
            f"merge:sync_company_dirs[{sroot.name}]",
            src_company_ids,
            _sync_one,
            workers=w,
        )
        merged_company_ids.update(src_company_ids)

        # 2) merge DB companies rows (prefer better)
        for cid, row in src_companies.items():
            if cid not in tgt_companies:
                tgt_companies[cid] = row
            else:
                tgt_companies[cid] = _pick_better_company_row(tgt_companies[cid], row)

        # 3) union runs + done rows
        tgt_runs.extend(read_runs_rows(sdb))
        tgt_done.extend(read_run_company_done_rows(sdb))

        # 4) merge retry/quarantine in-memory
        src_retry, src_quarantine, _, _ = _load_retry_state(sroot)
        tgt_retry = merge_retry_states(tgt_retry, src_retry)
        tgt_quarantine = merge_quarantine(tgt_quarantine, src_quarantine)
        write_retry_state(target_root, tgt_retry, tgt_quarantine)

        # 5) ledger
        if include_ledger:
            dst_ledger = _target_ledger_path(target_root)
            for lp in _ledger_paths(sroot):
                ledger_appended += append_ledger_all(lp, dst_ledger)

        # 6) scrub source root db/retry (if move-mode move)
        if move_mode == "move":
            delete_company_rows(sdb, src_company_ids)
            write_retry_state(sroot, {}, {})
            write_global_state_from_db_only(sroot, pretty=False)

    write_companies_rows(tgt_db, tgt_companies)
    write_runs_rows(tgt_db, tgt_runs)
    write_run_company_done_rows(tgt_db, tgt_done)

    write_global_state_from_db_only(target_root, pretty=False)

    return {
        "cmd": "merge",
        "dry_run": False,
        "target_root": str(target_root),
        "source_roots": [str(s) for s in srcs],
        "move_mode": move_mode,
        "include_ledger": bool(include_ledger),
        "workers": w,
        "merged_company_ids": len(merged_company_ids),
        "ledger_lines_appended": ledger_appended,
    }


__all__ = [
    # tables
    "Table",
    "read_table_preserve",
    "write_table_preserve",
    "merge_tables_dedupe",
    "split_table_by_ids",
    "group_rows_by_company_id",
    # statuses
    "IN_PROGRESS_STATUSES",
    "DONE_STATUSES",
    # roots
    "crawl_db_path",
    "global_state_path",
    "find_company_dir",
    "expected_company_dir",
    "write_global_state_from_db_only",
    # workflows
    "split_run_root",
    "merge_run_roots",
]
