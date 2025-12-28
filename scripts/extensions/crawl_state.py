from __future__ import annotations

import asyncio
import csv
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
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from extensions import output_paths
from extensions.output_paths import ensure_company_dirs, sanitize_bvdid

from configs.llm_industry import (
    industry_label as _industry_label,
    normalize_industry_all as _normalize_industry_all,
    normalize_industry_code as _normalize_industry_code,
)

# ---------------------------------------------------------------------------
# Atomic file I/O + bounded JSON cache + per-path locks
# ---------------------------------------------------------------------------

_JSON_CACHE_LOCK = threading.Lock()
_FILE_LOCKS_LOCK = threading.Lock()
_DIR_CACHE_LOCK = threading.Lock()

_JSON_CACHE_MAX = int(os.getenv("CRAWLSTATE_JSON_CACHE_SIZE", "2048"))
_DIR_CACHE_MAX = int(os.getenv("CRAWLSTATE_DIR_CACHE_SIZE", "4096"))

_JSON_CACHE: "OrderedDict[Path, Tuple[int, Any]]" = OrderedDict()
# IMPORTANT: key includes output_root to avoid stale cache when OUTPUT_ROOT changes.
_DIR_CACHE: "OrderedDict[str, Path]" = OrderedDict()
_FILE_LOCKS: Dict[Path, threading.Lock] = {}

# Reserved / meta keys inside url_index.json
URL_INDEX_META_KEY = "__meta__"
URL_INDEX_RESERVED_PREFIX = "__"

# Files
META_NAME = "crawl_meta.json"
URL_INDEX_NAME = "url_index.json"
GLOBAL_STATE_NAME = "crawl_global_state.json"

# ---------------------------------------------------------------------------
# Time / paths
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_root() -> Path:
    """
    IMPORTANT: output_paths.OUTPUT_ROOT is a mutable proxy possibly mutated by run.py at runtime.
    Never capture it at import-time. Always resolve dynamically here.
    """
    try:
        fn = getattr(output_paths, "get_output_root", None)
        if callable(fn):
            root = fn()
            return Path(root).resolve()
    except Exception:
        pass

    try:
        root = getattr(output_paths, "OUTPUT_ROOT", None)
        if root is None:
            return Path("outputs").resolve()
        return Path(root).resolve()
    except Exception:
        return Path("outputs").resolve()


def default_db_path() -> Path:
    return _output_root() / "crawl_state.sqlite3"


class _DefaultDBPathProxy(os.PathLike):
    """
    Dynamic PathLike proxy for default DB path.
    This prevents stale DEFAULT_DB paths when OUTPUT_ROOT changes at runtime.
    """

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


def _json_load_nocache(path: Path) -> Any:
    def _read() -> Any:
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return {}

    return _retry_emfile(_read)


# ---------------------------------------------------------------------------
# Company dirs / paths
# ---------------------------------------------------------------------------


def _company_metadata_dir(bvdid: str) -> Path:
    # Cache key includes current output_root to avoid stale paths if OUTPUT_ROOT changes.
    root = str(_output_root())
    key = f"{root}::{bvdid}"

    with _DIR_CACHE_LOCK:
        cached = _DIR_CACHE.get(key)
        if cached is not None:
            _DIR_CACHE.move_to_end(key, last=True)
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
        _DIR_CACHE[key] = meta
        _DIR_CACHE.move_to_end(key, last=True)
        while len(_DIR_CACHE) > _DIR_CACHE_MAX:
            _DIR_CACHE.popitem(last=False)

    return meta


def _meta_path_for(bvdid: str) -> Path:
    return _company_metadata_dir(bvdid) / META_NAME


def _url_index_path_for(bvdid: str) -> Path:
    return _company_metadata_dir(bvdid) / URL_INDEX_NAME


def _global_state_path() -> Path:
    return _output_root() / GLOBAL_STATE_NAME


# ---------------------------------------------------------------------------
# Company status semantics (existing behavior preserved)
# ---------------------------------------------------------------------------

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

_KNOWN_COMPANY_STATUSES = {
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_TERMINAL_DONE,
}

# Sticky precedence: never downgrade; only allow upgrades.
_COMPANY_STATUS_RANK: Dict[str, int] = {
    COMPANY_STATUS_PENDING: 0,
    COMPANY_STATUS_MD_NOT_DONE: 1,
    COMPANY_STATUS_MD_DONE: 2,
    COMPANY_STATUS_LLM_NOT_DONE: 3,
    COMPANY_STATUS_LLM_DONE: 4,
    COMPANY_STATUS_TERMINAL_DONE: 5,
}

_LEGACY_COMPANY_STATUS_MAP: Dict[str, CompanyStatus] = {
    "done": COMPANY_STATUS_LLM_DONE,
    "completed": COMPANY_STATUS_LLM_DONE,
    "complete": COMPANY_STATUS_LLM_DONE,
    "llmcomplete": COMPANY_STATUS_LLM_DONE,
    "llm_complete": COMPANY_STATUS_LLM_DONE,
    "terminal": COMPANY_STATUS_TERMINAL_DONE,
    "terminalized": COMPANY_STATUS_TERMINAL_DONE,
    "terminated": COMPANY_STATUS_TERMINAL_DONE,
    "md_done": COMPANY_STATUS_MD_DONE,
    "markdown_saved": COMPANY_STATUS_MD_DONE,
    "markdown_complete": COMPANY_STATUS_MD_DONE,
    "markdowncomplete": COMPANY_STATUS_MD_DONE,
}

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


def _normalize_company_status(st: Optional[str]) -> str:
    s = (st or "").strip().lower()
    if not s:
        return COMPANY_STATUS_PENDING
    if s in _KNOWN_COMPANY_STATUSES:
        return s
    if s in _LEGACY_COMPANY_STATUS_MAP:
        return _LEGACY_COMPANY_STATUS_MAP[s]
    return s


def _status_rank(st: Optional[str]) -> int:
    s = _normalize_company_status(st)
    return _COMPANY_STATUS_RANK.get(s, -1)


def _prefer_higher_status(current: Optional[str], derived: Optional[str]) -> str:
    c = _normalize_company_status(current)
    d = _normalize_company_status(derived)
    if _status_rank(c) >= _status_rank(d):
        return c
    return d


# ---------------------------------------------------------------------------
# Industry helpers (canonicalization for meta + DB)
# ---------------------------------------------------------------------------


def _canon_industry_codes(raw: Any) -> Optional[str]:
    """
    Canonical multi-code string: "9|10|20" (sorted, unique) or None.
    Uses configs.llm_industry.normalize_industry_all
    """
    return _normalize_industry_all(raw)


def _primary_industry_code(raw_codes: Optional[str]) -> Optional[str]:
    """
    Primary code stored as industry_code:
      - If codes is "9|10|20" => "9"
      - If None => None
    """
    if not raw_codes:
        return None
    parts = [p for p in str(raw_codes).split("|") if p]
    return parts[0] if parts else None


def normalize_company_industry_fields(
    raw: Any,
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns (industry_code, industry_label, industry_codes)

    - industry_codes: canonical multi-code "a|b|c" or None
    - industry_code: primary code (first in industry_codes) or None
    - industry_label: label of industry_code or "Unclassified"
    """
    codes = _canon_industry_codes(raw)
    code = _primary_industry_code(codes)
    # if missing, also allow a single raw code to become primary
    if code is None and raw is not None:
        c2 = _normalize_industry_code(raw)
        if c2 and c2 != "default":
            code = c2
            codes = c2  # single-code canonical
    label = _industry_label(code)
    return code, label, codes


# ---------------------------------------------------------------------------
# crawl_meta.json helpers (industry fields added)
# ---------------------------------------------------------------------------


def load_crawl_meta(bvdid: str) -> Dict[str, Any]:
    p = _meta_path_for(bvdid)
    if not p.exists():
        return {}
    obj = _json_load_nocache(p)
    return obj if isinstance(obj, dict) else {}


def patch_crawl_meta(
    bvdid: str,
    patch: Dict[str, Any],
    *,
    pretty: bool = True,
) -> None:
    """
    Atomic patch of crawl_meta.json. Ensures the 3 industry fields exist if provided:
      - industry_code
      - industry_label
      - industry_codes
    """
    p = _meta_path_for(bvdid)
    lk = _get_file_lock(p)

    def _update() -> None:
        with lk:
            base: Dict[str, Any] = {}
            if p.exists():
                loaded = _json_load_nocache(p)
                if isinstance(loaded, dict):
                    base = loaded
            if isinstance(patch, dict):
                base.update(patch)
            _atomic_write_json(p, base, cache=False, pretty=pretty)

    _update()


def ensure_crawl_meta_has_industry(
    bvdid: str,
    *,
    industry_code: Optional[str],
    industry_label: Optional[str],
    industry_codes: Optional[str],
) -> None:
    """
    Write the 3 industry fields to crawl_meta.json (idempotent).
    """
    patch: Dict[str, Any] = {}
    patch["industry_code"] = industry_code
    patch["industry_label"] = industry_label or _industry_label(industry_code)
    patch["industry_codes"] = industry_codes
    patch_crawl_meta(bvdid, patch, pretty=True)


# ---------------------------------------------------------------------------
# url_index.json helpers
# ---------------------------------------------------------------------------


def _is_reserved_url_index_key(k: Any) -> bool:
    try:
        return str(k).startswith(URL_INDEX_RESERVED_PREFIX)
    except Exception:
        return False


def _index_crawl_finished(index: Dict[str, Any]) -> bool:
    meta = index.get(URL_INDEX_META_KEY)
    return bool(isinstance(meta, dict) and meta.get("crawl_finished"))


def load_url_index(bvdid: str) -> Dict[str, Any]:
    p = _url_index_path_for(bvdid)
    if not p.exists():
        return {}
    obj = _json_load_nocache(p)
    return obj if isinstance(obj, dict) else {}


def upsert_url_index_entry(bvdid: str, url: str, patch: Dict[str, Any]) -> None:
    """
    Upsert a URL entry (non-meta). Does not touch __meta__.
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

            payload = json.dumps(existing, ensure_ascii=False, separators=(",", ":"))
            _atomic_write_text(idx_path, payload, "utf-8")

    _update()


def patch_url_index_meta(bvdid: str, patch: Dict[str, Any]) -> None:
    """
    Safely patch url_index.json __meta__ dictionary atomically.
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

            meta = existing.get(URL_INDEX_META_KEY)
            meta_dict = dict(meta) if isinstance(meta, dict) else {}
            if isinstance(patch, dict):
                meta_dict.update(patch)
            existing[URL_INDEX_META_KEY] = meta_dict

            payload = json.dumps(existing, ensure_ascii=False, separators=(",", ":"))
            _atomic_write_text(idx_path, payload, "utf-8")

    _update()


def _status_has_any(s: str, needles: Tuple[str, ...]) -> bool:
    return any(n in s for n in needles)


def _classify_url_entry(ent: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Return (markdown_done, llm_done).
    """
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
            ("markdown" in status or "md" == status)
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


def _compute_company_stage_from_index(
    index: Dict[str, Any],
) -> Tuple[CompanyStatus, int, int, int]:
    """
    IMPORTANT SEMANTICS FIX:

    - url_index.json can contain __meta__ (reserved key).
    - We NEVER conclude markdown_done / llm_done unless __meta__.crawl_finished == True.
      This prevents Ctrl+C (partial discovery) from looking "done".
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
        if isinstance(raw_ent, dict):
            m_done, l_done = _classify_url_entry(raw_ent)
            if m_done:
                md_done += 1
            if l_done:
                llm_done += 1

    if urls_total == 0:
        return COMPANY_STATUS_PENDING, 0, 0, 0

    # Crawl not finished => NEVER return *_done.
    if not crawl_finished:
        if llm_done > 0:
            return COMPANY_STATUS_LLM_NOT_DONE, urls_total, md_done, llm_done
        if md_done > 0:
            return COMPANY_STATUS_MD_NOT_DONE, urls_total, md_done, llm_done
        return COMPANY_STATUS_PENDING, urls_total, md_done, llm_done

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


@dataclass(frozen=True)
class URLStats:
    urls_total: int
    urls_markdown_done: int
    urls_llm_done: int


def compute_url_stats_from_index_dict(index: Dict[str, Any]) -> URLStats:
    _, total, md, llm = _compute_company_stage_from_index(index)
    return URLStats(int(total), int(md), int(llm))


def compute_url_stats_for_company(bvdid: str) -> URLStats:
    idx = load_url_index(bvdid)
    if not isinstance(idx, dict) or not idx:
        return URLStats(0, 0, 0)
    return compute_url_stats_from_index_dict(idx)


# ---------------------------------------------------------------------------
# Snapshots / helpers
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
    done_reason: Optional[str] = None
    done_details: Optional[Dict[str, Any]] = None
    done_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # NEW: industry fields persisted in DB (and mirrored in crawl_meta.json)
    industry_code: Optional[str] = None
    industry_label: Optional[str] = None
    industry_codes: Optional[str] = None


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
    st = _normalize_company_status(row["status"] if "status" in row.keys() else None)
    if st not in _KNOWN_COMPANY_STATUSES:
        st = COMPANY_STATUS_PENDING

    def _col(name: str) -> Optional[str]:
        if name not in row.keys():
            return None
        v = row[name]
        return str(v) if v is not None else None

    return CompanySnapshot(
        bvdid=row["bvdid"],
        name=row["name"],
        root_url=row["root_url"],
        status=(st or COMPANY_STATUS_PENDING),  # type: ignore[assignment]
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
        industry_code=_col("industry_code"),
        industry_label=_col("industry_label"),
        industry_codes=_col("industry_codes"),
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
# SQLite-backed global state
# ---------------------------------------------------------------------------


class CrawlState:
    """
    Adds industry persistence:

      - DB columns: industry_code, industry_label, industry_codes
      - crawl_meta.json fields: industry_code, industry_label, industry_codes

    Canonical semantics:
      - industry_codes: canonical multi-code "a|b|c" sorted unique (or None)
      - industry_code: primary code = first element of industry_codes (or None)
      - industry_label: label of industry_code (or "Unclassified")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.db_path = self.db_path.resolve()

        try:
            self._dynamic_db = self.db_path == default_db_path().resolve()
        except Exception:
            self._dynamic_db = False

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._connect(self.db_path)
        self._init_schema()

        self._global_state_write_lock = asyncio.Lock()
        self._last_db_only_global_write_ts = 0.0

    # -------------------- connection / schema --------------------

    def _connect(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
        with suppress(Exception):
            conn.execute("PRAGMA journal_mode=WAL")
        with suppress(Exception):
            conn.execute("PRAGMA synchronous=NORMAL")
        with suppress(Exception):
            conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_company_columns_locked(self) -> None:
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

        # existing migrations
        _add("done_reason TEXT", "done_reason")
        _add("done_details TEXT", "done_details")
        _add("done_at TEXT", "done_at")

        # NEW: industry migrations
        _add("industry_code TEXT", "industry_code")
        _add("industry_label TEXT", "industry_label")
        _add("industry_codes TEXT", "industry_codes")

    def _init_schema_locked(self) -> None:
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
                updated_at TEXT,
                industry_code TEXT,
                industry_label TEXT,
                industry_codes TEXT
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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_company_done (
                run_id TEXT NOT NULL,
                bvdid TEXT NOT NULL,
                done_at TEXT,
                PRIMARY KEY (run_id, bvdid)
            )
            """
        )
        self._ensure_company_columns_locked()

    def _init_schema(self) -> None:
        with self._lock:
            with suppress(Exception):
                self._conn.execute("PRAGMA journal_mode=WAL")
            with suppress(Exception):
                self._conn.execute("PRAGMA synchronous=NORMAL")
            with suppress(Exception):
                self._conn.execute("PRAGMA busy_timeout=5000")
            self._init_schema_locked()

    def _ensure_db_current_locked(self) -> None:
        if not getattr(self, "_dynamic_db", False):
            return

        try:
            want = default_db_path().resolve()
        except Exception:
            return

        if want == self.db_path:
            return

        try:
            with suppress(Exception):
                self._conn.close()
        finally:
            self.db_path = want
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = self._connect(self.db_path)
            self._init_schema_locked()

    def close(self) -> None:
        with suppress(Exception):
            with self._lock:
                self._conn.close()

    # -------------------- basic async db helpers --------------------

    async def _exec(self, stmt: _Stmt) -> None:
        def _run() -> None:
            with self._lock:
                self._ensure_db_current_locked()
                self._conn.execute(stmt.sql, stmt.args)

        await asyncio.to_thread(_run)

    async def _query_one(
        self, sql: str, args: Tuple[Any, ...]
    ) -> Optional[sqlite3.Row]:
        def _run() -> Optional[sqlite3.Row]:
            with self._lock:
                self._ensure_db_current_locked()
                return self._conn.execute(sql, args).fetchone()

        return await asyncio.to_thread(_run)

    async def _query_all(self, sql: str, args: Tuple[Any, ...]) -> List[sqlite3.Row]:
        def _run() -> List[sqlite3.Row]:
            with self._lock:
                self._ensure_db_current_locked()
                return self._conn.execute(sql, args).fetchall()

        return await asyncio.to_thread(_run)

    # ---------- in-progress DB APIs ----------

    @staticmethod
    def _in_progress_statuses() -> Tuple[str, ...]:
        return (
            COMPANY_STATUS_PENDING,
            COMPANY_STATUS_MD_NOT_DONE,
            COMPANY_STATUS_LLM_NOT_DONE,
        )

    async def has_in_progress_companies(self) -> bool:
        sts = self._in_progress_statuses()
        placeholders = ",".join("?" for _ in sts)
        row = await self._query_one(
            f"SELECT 1 AS x FROM companies WHERE status IN ({placeholders}) LIMIT 1",
            tuple(sts),
        )
        return row is not None

    async def count_in_progress_companies(self) -> int:
        sts = self._in_progress_statuses()
        placeholders = ",".join("?" for _ in sts)
        row = await self._query_one(
            f"SELECT COUNT(*) AS c FROM companies WHERE status IN ({placeholders})",
            tuple(sts),
        )
        try:
            return int(row["c"] or 0) if row is not None else 0
        except Exception:
            return 0

    async def get_in_progress_company_ids(self, limit: int = 2000) -> List[str]:
        limit = max(1, int(limit))
        sts = self._in_progress_statuses()
        placeholders = ",".join("?" for _ in sts)
        rows = await self._query_all(
            f"""
            SELECT bvdid FROM companies
             WHERE status IN ({placeholders})
             ORDER BY updated_at DESC
             LIMIT ?
            """,
            tuple(sts) + (limit,),
        )
        return [str(r["bvdid"]) for r in rows]

    async def recompute_all_in_progress(self, *, concurrency: int = 32) -> None:
        c = max(1, int(concurrency))
        ids = await self.get_in_progress_company_ids(limit=1_000_000)
        if not ids:
            return

        sem = asyncio.Semaphore(c)

        async def _one(cid: str) -> None:
            async with sem:
                with suppress(Exception):
                    await self.recompute_company_from_index(cid)

        batch = max(64, c * 8)
        for i in range(0, len(ids), batch):
            await asyncio.gather(*(_one(cid) for cid in ids[i : i + batch]))

    # ---------- terminal "done" API ----------

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
        now = _now_iso()
        details_s = (
            json.dumps(details, ensure_ascii=False, separators=(",", ":"))
            if isinstance(details, dict)
            else None
        )
        last_error_trim = (last_error or "")[:4000] if last_error is not None else None

        try:
            stats = compute_url_stats_for_company(bvdid)
        except Exception:
            stats = URLStats(0, 0, 0)

        with self._lock:
            self._ensure_db_current_locked()

            self._conn.execute(
                """
                INSERT OR IGNORE INTO companies (
                    bvdid, name, root_url, status,
                    urls_total, urls_markdown_done, urls_llm_done,
                    last_error, done_reason, done_details, done_at,
                    created_at, updated_at,
                    industry_code, industry_label, industry_codes
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, NULL)
                """,
                (bvdid, name, root_url, COMPANY_STATUS_PENDING, now, now),
            )

            sets = [
                "status=?",
                "done_reason=?",
                "done_details=?",
                "done_at=?",
                "updated_at=?",
                "urls_total=?",
                "urls_markdown_done=?",
                "urls_llm_done=?",
            ]
            args: List[Any] = [
                COMPANY_STATUS_TERMINAL_DONE,
                (reason or "")[:256],
                details_s,
                now,
                now,
                int(stats.urls_total),
                int(stats.urls_markdown_done),
                int(stats.urls_llm_done),
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
        now = _now_iso()
        with self._lock:
            self._ensure_db_current_locked()

            row = self._conn.execute(
                "SELECT status FROM companies WHERE bvdid=?",
                (bvdid,),
            ).fetchone()
            if row is None:
                return

            st = _normalize_company_status(row["status"])
            new_status = st
            if (not keep_status) and st == COMPANY_STATUS_TERMINAL_DONE:
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

        def _run() -> None:
            with self._lock:
                self._ensure_db_current_locked()

                self._conn.execute(
                    "INSERT OR IGNORE INTO run_company_done (run_id, bvdid, done_at) VALUES (?, ?, ?)",
                    (run_id, bvdid, now),
                )
                try:
                    inserted = int(
                        (self._conn.execute("SELECT changes()").fetchone() or [0])[0]
                    )
                except Exception:
                    inserted = 0

                if inserted > 0:
                    self._conn.execute(
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
                else:
                    self._conn.execute(
                        "UPDATE runs SET last_company_bvdid=?, last_updated=? WHERE run_id=?",
                        (bvdid, now, run_id),
                    )

        await asyncio.to_thread(_run)

    async def get_run_snapshot(self, run_id: str) -> Optional[RunSnapshot]:
        row = await self._query_one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return _row_to_run_snapshot(row) if row else None

    # -----------------------------------------------------------------------
    # Company-level API (industry fields added)
    # -----------------------------------------------------------------------

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
        done_reason: Optional[str] = None,
        done_details: Optional[Dict[str, Any]] = None,
        done_at: Optional[str] = None,
        # NEW:
        industry_code: Optional[str] = None,
        industry_label: Optional[str] = None,
        industry_codes: Optional[str] = None,
        # If True, write crawl_meta.json too (industry + basics)
        write_meta: bool = False,
    ) -> None:
        now = _now_iso()

        await self._exec(
            _Stmt(
                """
                INSERT OR IGNORE INTO companies (
                    bvdid, name, root_url, status,
                    urls_total, urls_markdown_done, urls_llm_done,
                    last_error, done_reason, done_details, done_at,
                    created_at, updated_at,
                    industry_code, industry_label, industry_codes
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, NULL)
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

        # NEW: industry fields (normalize once here if caller gives raw)
        if (
            (industry_code is not None)
            or (industry_label is not None)
            or (industry_codes is not None)
        ):
            # Caller is expected to pass canonical values already.
            # Still keep consistency: if label missing, derive; if codes missing but code exists, set codes=code.
            c = industry_code
            codes = industry_codes
            if codes is None and c is not None:
                codes = c
            lbl = industry_label or _industry_label(c)
            sets.append("industry_code=?")
            args.append(c)
            sets.append("industry_label=?")
            args.append(lbl)
            sets.append("industry_codes=?")
            args.append(codes)

        args.append(bvdid)
        await self._exec(
            _Stmt(f"UPDATE companies SET {', '.join(sets)} WHERE bvdid=?", tuple(args))
        )

        if write_meta:
            patch: Dict[str, Any] = {}
            if name is not None:
                patch["company_name"] = name
            if root_url is not None:
                patch["root_url"] = root_url

            if (
                (industry_code is not None)
                or (industry_label is not None)
                or (industry_codes is not None)
            ):
                c = industry_code
                codes = (
                    industry_codes
                    if industry_codes is not None
                    else (c if c is not None else None)
                )
                lbl = industry_label or _industry_label(c)
                patch["industry_code"] = c
                patch["industry_label"] = lbl
                patch["industry_codes"] = codes

            if patch:
                patch_crawl_meta(bvdid, patch, pretty=True)

    async def set_company_industry(
        self,
        bvdid: str,
        *,
        raw_industry: Any,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
        write_meta: bool = True,
    ) -> None:
        """
        Canonical, single entrypoint: set industry fields from any raw input.
        Also updates crawl_meta.json by default.
        """
        code, label, codes = normalize_company_industry_fields(raw_industry)
        await self.upsert_company(
            bvdid,
            name=name,
            root_url=root_url,
            industry_code=code,
            industry_label=label,
            industry_codes=codes,
            write_meta=write_meta,
        )

    async def get_company_snapshot(
        self, bvdid: str, *, recompute: bool = True
    ) -> CompanySnapshot:
        row = await self._query_one("SELECT * FROM companies WHERE bvdid=?", (bvdid,))
        if row is not None:
            if _normalize_company_status(row["status"]) == COMPANY_STATUS_TERMINAL_DONE:
                return _row_to_company_snapshot(row)
            if recompute:
                return await self.recompute_company_from_index(bvdid)
            return _row_to_company_snapshot(row)

        if recompute:
            return await self.recompute_company_from_index(bvdid)

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
        row2 = await self._query_one("SELECT * FROM companies WHERE bvdid=?", (bvdid,))
        return _row_to_company_snapshot(row2)  # type: ignore[arg-type]

    async def recompute_company_from_index(
        self,
        bvdid: str,
        *,
        name: Optional[str] = None,
        root_url: Optional[str] = None,
    ) -> CompanySnapshot:
        cur = await self._query_one(
            "SELECT status, urls_total, urls_markdown_done, urls_llm_done FROM companies WHERE bvdid=?",
            (bvdid,),
        )

        cur_status = _normalize_company_status(cur["status"]) if cur is not None else ""
        cur_total = int(cur["urls_total"] or 0) if cur is not None else 0
        cur_md = int(cur["urls_markdown_done"] or 0) if cur is not None else 0
        cur_llm = int(cur["urls_llm_done"] or 0) if cur is not None else 0

        if cur_status == COMPANY_STATUS_TERMINAL_DONE:
            if name is not None or root_url is not None:
                await self.upsert_company(bvdid, name=name, root_url=root_url)
            return await self.get_company_snapshot(bvdid, recompute=False)

        index = await asyncio.to_thread(load_url_index, bvdid)
        derived_status, urls_total, md_done, llm_done = (
            _compute_company_stage_from_index(index)
        )

        new_status = _prefer_higher_status(cur_status, derived_status)

        new_total = max(cur_total, int(urls_total))
        new_md = max(cur_md, int(md_done))
        new_llm = max(cur_llm, int(llm_done))

        new_total = max(new_total, new_md, new_llm)
        new_md = min(new_md, new_total)
        new_llm = min(new_llm, new_total)

        await self.upsert_company(
            bvdid,
            name=name,
            root_url=root_url,
            status=new_status if new_status in _KNOWN_COMPANY_STATUSES else None,
            urls_total=new_total,
            urls_markdown_done=new_md,
            urls_llm_done=new_llm,
        )
        return await self.get_company_snapshot(bvdid, recompute=False)

    # -----------------------------------------------------------------------
    # LLM stage helpers (used by extensions/llm_passes.py)
    # -----------------------------------------------------------------------

    async def get_pending_urls_for_markdown(self, bvdid: str) -> List[str]:
        index = await asyncio.to_thread(load_url_index, bvdid)
        return _pending_urls_for_stage(
            index if isinstance(index, dict) else {}, "markdown"
        )

    async def get_pending_urls_for_llm(self, bvdid: str) -> List[str]:
        index = await asyncio.to_thread(load_url_index, bvdid)
        return _pending_urls_for_stage(index if isinstance(index, dict) else {}, "llm")

    async def mark_url_failed(self, bvdid: str, url: str, reason: str) -> None:
        upsert_url_index_entry(
            bvdid,
            url,
            {
                "status": f"url_failed:{(reason or 'unknown')[:120]}",
                "last_error": (reason or "unknown")[:4000],
                "updated_at": _now_iso(),
            },
        )
        await self.recompute_company_from_index(bvdid)

    # -----------------------------------------------------------------------
    # Industry bulk update utility (for run.py optional flag)
    # -----------------------------------------------------------------------

    async def update_industry_from_csv(
        self,
        csv_path: str | Path,
        *,
        bvdid_column: str = "bvdid",
        industry_column: str = "industry",
        # optional: sometimes the new csv might contain "industry_codes" already
        industry_codes_column: Optional[str] = None,
        name_column: Optional[str] = None,
        root_url_column: Optional[str] = None,
        strict_header: bool = True,
        write_meta: bool = True,
        batch_size: int = 500,
    ) -> int:
        """
        Update DB industry fields for existing companies using a new CSV that contains the same bvdid.

        Canonical behavior:
          - prefer industry_codes_column if provided and present in csv
          - else use industry_column
          - normalize to (industry_code, industry_label, industry_codes)
          - update only rows that exist in DB (INSERT OR IGNORE is not used here)

        Returns: number of companies updated (attempted updates for existing rows).
        """
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        def _read_rows() -> List[Dict[str, str]]:
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV has no header row.")
                fields = [x.strip() for x in reader.fieldnames if x is not None]
                have = set(fields)

                need = {bvdid_column}
                if industry_codes_column:
                    need.add(industry_codes_column)
                need.add(industry_column)
                if name_column:
                    need.add(name_column)
                if root_url_column:
                    need.add(root_url_column)

                if strict_header:
                    missing = [c for c in need if c not in have]
                    if missing:
                        raise ValueError(f"CSV missing required columns: {missing}")

                out: List[Dict[str, str]] = []
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    out.append(
                        {k: (v if v is not None else "") for k, v in row.items()}
                    )
                return out

        rows = await asyncio.to_thread(_read_rows)
        if not rows:
            return 0

        # Determine which industry source to use per-row
        def _pick_industry_value(r: Dict[str, str]) -> Any:
            if (
                industry_codes_column
                and industry_codes_column in r
                and str(r.get(industry_codes_column) or "").strip()
            ):
                return r.get(industry_codes_column)
            return r.get(industry_column)

        # Fetch existing bvdids once (avoid per-row SELECT)
        bvdids = [str(r.get(bvdid_column) or "").strip() for r in rows]
        bvdids = [b for b in bvdids if b]
        if not bvdids:
            return 0

        def _existing_set() -> set[str]:
            with self._lock:
                self._ensure_db_current_locked()
                # chunk IN queries to avoid SQLite limits
                exist: set[str] = set()
                chunk = 900
                for i in range(0, len(bvdids), chunk):
                    part = bvdids[i : i + chunk]
                    placeholders = ",".join("?" for _ in part)
                    q = f"SELECT bvdid FROM companies WHERE bvdid IN ({placeholders})"
                    for rr in self._conn.execute(q, tuple(part)).fetchall():
                        exist.add(str(rr["bvdid"]))
                return exist

        existing = await asyncio.to_thread(_existing_set)
        if not existing:
            return 0

        updates: List[
            Tuple[str, Optional[str], str, Optional[str], Optional[str], Optional[str]]
        ] = []
        # tuple: (bvdid, industry_code, industry_label, industry_codes, name, root_url)
        for r in rows:
            b = str(r.get(bvdid_column) or "").strip()
            if not b or b not in existing:
                continue
            raw_ind = _pick_industry_value(r)
            code, label, codes = normalize_company_industry_fields(raw_ind)

            nm = str(r.get(name_column) or "").strip() if name_column else None
            ru = str(r.get(root_url_column) or "").strip() if root_url_column else None
            nm = nm if nm else None
            ru = ru if ru else None

            updates.append((b, code, label, codes, nm, ru))

        if not updates:
            return 0

        def _apply_batch(
            batch: List[
                Tuple[
                    str, Optional[str], str, Optional[str], Optional[str], Optional[str]
                ]
            ],
        ) -> int:
            with self._lock:
                self._ensure_db_current_locked()
                n = 0
                now = _now_iso()
                for b, code, label, codes, nm, ru in batch:
                    # keep codes consistent
                    if codes is None and code is not None:
                        codes = code
                    label2 = label or _industry_label(code)

                    sets = [
                        "industry_code=?",
                        "industry_label=?",
                        "industry_codes=?",
                        "updated_at=?",
                    ]
                    args: List[Any] = [code, label2, codes, now]

                    if nm is not None:
                        sets.append("name=?")
                        args.append(nm)
                    if ru is not None:
                        sets.append("root_url=?")
                        args.append(ru)

                    args.append(b)
                    self._conn.execute(
                        f"UPDATE companies SET {', '.join(sets)} WHERE bvdid=?",
                        tuple(args),
                    )
                    n += 1

            # meta writes are outside DB lock
            if write_meta:
                for b, code, label, codes, nm, ru in batch:
                    patch: Dict[str, Any] = {
                        "industry_code": code,
                        "industry_label": (label or _industry_label(code)),
                        "industry_codes": (
                            codes
                            if codes is not None
                            else (code if code is not None else None)
                        ),
                    }
                    if nm is not None:
                        patch["company_name"] = nm
                    if ru is not None:
                        patch["root_url"] = ru
                    patch_crawl_meta(b, patch, pretty=True)

            return n

        total_updated = 0
        bs = max(1, int(batch_size))
        for i in range(0, len(updates), bs):
            total_updated += await asyncio.to_thread(_apply_batch, updates[i : i + bs])

        return total_updated

    # -----------------------------------------------------------------------
    # Global-state writer (DB-only) — kept (existing behavior)
    # -----------------------------------------------------------------------

    async def write_global_state_from_db_only(self) -> Dict[str, Any]:
        def _compute_and_write() -> Dict[str, Any]:
            with self._lock:
                self._ensure_db_current_locked()

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

                run_done_count: Optional[int] = None
                if run_row is not None:
                    try:
                        rrid = run_row["run_id"]
                        c = self._conn.execute(
                            "SELECT COUNT(*) AS c FROM run_company_done WHERE run_id=?",
                            (rrid,),
                        ).fetchone()
                        if c is not None:
                            run_done_count = int(c["c"] or 0)
                    except Exception:
                        run_done_count = None

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
                raw_st = r["status"]
                st_norm = _normalize_company_status(raw_st)
                b = str(r["bvdid"])

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
                elif st_effective in (
                    COMPANY_STATUS_MD_NOT_DONE,
                    COMPANY_STATUS_LLM_NOT_DONE,
                ):
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
                    "started_at": run_row["started_at"],
                    "total_companies": run_total,
                    "completed_companies": run_completed,
                    "last_company_bvdid": run_row["last_company_bvdid"],
                    "last_updated": run_row["last_updated"],
                }

            payload: Dict[str, Any] = {
                "output_root": str(_output_root()),
                "db_path": str(self.db_path),
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

            _atomic_write_json(_global_state_path(), payload, cache=False, pretty=True)
            return payload

        return await asyncio.to_thread(_compute_and_write)

    async def write_global_state_throttled(
        self, min_interval_sec: float = 2.0
    ) -> Dict[str, Any]:
        mi = max(0.05, float(min_interval_sec))
        async with self._global_state_write_lock:
            now = time.time()
            if (now - self._last_db_only_global_write_ts) < mi:
                # return best-effort snapshot without rewriting
                return await self.write_global_state_from_db_only()
            payload = await self.write_global_state_from_db_only()
            self._last_db_only_global_write_ts = time.time()
            return payload


# ---------------------------------------------------------------------------
# Singleton accessor (used across extensions)
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
_STATE: Optional[CrawlState] = None


def get_crawl_state(db_path: Optional[Path] = None) -> CrawlState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = CrawlState(db_path=db_path)
        return _STATE


# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_DB",
    "CrawlState",
    "get_crawl_state",
    "CompanySnapshot",
    "RunSnapshot",
    "URLStats",
    "load_url_index",
    "upsert_url_index_entry",
    "patch_url_index_meta",
    "load_crawl_meta",
    "patch_crawl_meta",
    "ensure_crawl_meta_has_industry",
    "normalize_company_industry_fields",
]
