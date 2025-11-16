from __future__ import annotations

import os
import asyncio
import errno
import json
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from extensions.output_paths import OUTPUT_ROOT as OUTPUTS_DIR, sanitize_bvdid

CHECKPOINTS_DIR = "checkpoints"
CRAWL_META_NAME = "crawl_meta.json"
SESSION_NAME = "session_checkpoint.json"
URL_INDEX_NAME = "url_index.json"  # manifest used for resume


# ----------------------------
# Resilient atomic writers
# ----------------------------

def _retry_emfile(fn, attempts: int = 6, base_delay: float = 0.15):
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            # Only EMFILE handled here; AccessDenied is handled in _atomic_write_text
            if e.errno == errno.EMFILE or "Too many open files" in str(e):
                time.sleep(base_delay * (2 ** i))
                continue
            raise


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8"):
    """
    Robust atomic write:
      1) Write to a unique temp file in the same directory.
      2) fsync to ensure the data hits disk.
      3) Retry os.replace(...) on Windows 'Access is denied' / sharing violations.
      4) Skip write if content identical to avoid churn.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: skip if unchanged (reduces file locking pressure a lot)
    try:
        if path.exists():
            prev = path.read_text(encoding=encoding)
            if prev == data:
                return
    except Exception:
        # If we can't read, just proceed with the write.
        pass

    # Unique temp name to avoid clashes with other writes
    stamp = f"{int(time.time()*1000)}-{os.getpid()}-{threading.get_ident()}"
    tmp = Path(f"{str(path)}.tmp.{stamp}")

    def _write():
        # 1) Write the temp file
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(data)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # fsync might not be available on some FS; safe to skip.
                pass

        # 2) Replace with retries for Windows locking
        last_err = None
        for i in range(12):  # ~ (0.05 * (2^11)) ≈ 102s max; usually succeeds in < 1s
            try:
                os.replace(tmp, path)  # atomic on Windows/NTFS
                return
            except PermissionError as e:
                last_err = e
            except OSError as e:
                # Handle common Windows lock scenarios: EACCES, EPERM, winerror 5/32
                winerr = getattr(e, "winerror", 0)
                if e.errno in (errno.EACCES, errno.EPERM) or winerr in (5, 32):
                    last_err = e
                else:
                    # Unexpected; clean temp and re-raise
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise
            # Backoff (0.05, 0.1, 0.2, ... seconds)
            time.sleep(0.05 * (2 ** i))

        # If we got here, we never managed to swap. Keep .tmp for inspection and raise.
        raise last_err  # type: ignore[misc]

    _retry_emfile(_write)


def _atomic_write_json(path: Path, obj: object):
    import json as _json
    _atomic_write_text(path, _json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


# ----------------------------
# Company checkpoint
# ----------------------------

@dataclass
class CompanyProgress:
    bvdid: str
    company_name: Optional[str] = None

    stage: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    urls_total: int = 0
    urls_done: int = 0
    urls_failed: int = 0
    last_url: Optional[str] = None

    # Seeding block (optional)
    seeding: Dict[str, Any] = None  # discovered_total, filtered_total, seed_roots, etc.

    # Save stats (optional)
    saves: Dict[str, Any] = None  # saved_html_total, saved_md_total, saved_json_total, md_suppressed_total
    resumed_skips: int = 0

    # Resume metadata
    resume_mode: Optional[str] = None   # 'url_index' | 'artifacts' | None
    notes: list = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("seeding") is None:
            d["seeding"] = {}
        if d.get("saves") is None:
            d["saves"] = {}
        if d.get("notes") is None:
            d["notes"] = []
        return d


class CompanyCheckpoint:
    """
    Async-aware helper around:
      outputs/{safe_bvdid}/checkpoints/crawl_meta.json

    This replaced the old 'progress.json' — all progress/state is now written to crawl_meta.json.
    Atomic writes, EMFILE retries, and Windows-safe bvdid via sanitize_bvdid.
    """
    def __init__(self, bvdid: str, *, company_name: Optional[str] = None) -> None:
        self.bvdid_original = str(bvdid)
        self.bvdid_safe = sanitize_bvdid(bvdid)
        self.root = OUTPUTS_DIR / self.bvdid_safe / CHECKPOINTS_DIR
        self.path = self.root / CRAWL_META_NAME
        self._lock = asyncio.Lock()
        self.data = CompanyProgress(bvdid=self.bvdid_original, company_name=company_name).as_dict()

    async def load(self) -> None:
        async with self._lock:
            if self.path.exists():
                try:
                    text = self.path.read_text(encoding="utf-8")
                    obj = json.loads(text)
                    # merge what's in disk into in-memory data (disk takes precedence for fields present)
                    if isinstance(obj, dict):
                        self.data.update(obj)
                except Exception:
                    # ignore corrupt / partial file; rewrite on next save
                    pass

    async def save(self) -> None:
        """
        Save self.data to crawl_meta.json atomically.

        Preserve any existing 'last_crawled_at' in the on-disk crawl_meta.json so we
        do not clobber the last-crawled timestamp written by helpers that only
        update that key (write_last_crawl_date). This makes writes idempotent and
        avoids races where one writer would erase another writer's keys.
        """
        async with self._lock:
            payload_dict = dict(self.data)
            # preserve last_crawled_at if present on disk and not in payload_dict
            if self.path.exists():
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict):
                        if "last_crawled_at" in existing and "last_crawled_at" not in payload_dict:
                            payload_dict["last_crawled_at"] = existing["last_crawled_at"]
                except Exception:
                    # if read fails, proceed and write payload as-is
                    pass

            payload = json.dumps(payload_dict, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_atomic_write_text, self.path, payload, "utf-8")

    # ---- marks ----

    async def mark_start(self, stage: str, total_urls: int = 0) -> None:
        self.data.update({
            "stage": stage,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "urls_total": int(total_urls or 0),
            "urls_done": 0,
            "urls_failed": 0,
            "last_url": None,
        })
        await self.save()

    async def mark_finished(self) -> None:
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        await self.save()

    async def set_total(self, total: int) -> None:
        self.data["urls_total"] = int(total)
        await self.save()

    async def mark_url_done(self, url: str) -> None:
        self.data["urls_done"] = int(self.data.get("urls_done", 0)) + 1
        self.data["last_url"] = url
        await self.save()

    async def mark_url_failed(self, url: str, reason: str) -> None:
        self.data["urls_failed"] = int(self.data.get("urls_failed", 0)) + 1
        self.data["last_url"] = url
        errs = self.data.setdefault("errors", [])
        errs.append({"url": url, "reason": reason})
        await self.save()

    async def add_note(self, text: str) -> None:
        notes = self.data.setdefault("notes", [])
        notes.append({"time": datetime.now(timezone.utc).isoformat(), "note": text})
        await self.save()

    async def set_company_name(self, name: str) -> None:
        self.data["company_name"] = (str(name).strip() or None) if name is not None else None
        await self.save()

    # ---- seeding/saves blocks ----

    async def set_seeding_stats(self, stats: Dict[str, Any]) -> None:
        blk = self.data.setdefault("seeding", {})
        blk.update(stats or {})
        await self.save()

    async def add_saves_delta(self, **kwargs: int) -> None:
        blk = self.data.setdefault("saves", {})
        for k, v in kwargs.items():
            blk[k] = int(blk.get(k, 0)) + int(v or 0)
        await self.save()

    async def set_resume_mode(self, mode: Optional[str]) -> None:
        self.data["resume_mode"] = mode
        await self.save()

    # ---- convenience ----

    def is_finished(self) -> bool:
        return bool(self.data.get("finished_at"))

    def progress_ratio(self) -> float:
        tot = int(self.data.get("urls_total") or 0) or 1
        return float(self.data.get("urls_done", 0)) / float(tot)

    # ---- paths ----

    def checkpoints_dir(self) -> Path:
        return self.root

    def url_index_path(self) -> Path:
        return self.root / URL_INDEX_NAME


# ----------------------------
# Session-level (one JSON file)
# ----------------------------

class CheckpointManager:
    """
    Maintains in-memory map of CompanyCheckpoint and a single
    session file (outputs/session_checkpoint.json) for the current run.

    New saved format for session_checkpoint.json (extended with global progress):

      {
        "started_at": "...",
        "companies": {
           "<bvdid>": "<company name>|\"\"|null",
           ...
        },
        "total_companies": 58200,
        "completed_companies": 1240,
        "completion_pct": 2.13,
        "last_company_bvdid": "US123456789",
        "last_updated": "..."
      }

    The loader is tolerant to the older format and will
    migrate entries into the new structure on update.
    """
    def __init__(self, outputs_dir: Path = OUTPUTS_DIR) -> None:
        self.outputs_dir = outputs_dir
        self.session_path = self.outputs_dir / SESSION_NAME
        self._checkpoints: Dict[str, CompanyCheckpoint] = {}
        self._lock = asyncio.Lock()

    # ---- internal helpers ----

    def _load_session_locked(self) -> Dict[str, Any]:
        """
        Load and normalize the session JSON.
        Must be called with self._lock held.
        """
        cur: Dict[str, Any] = {}
        if self.session_path.exists():
            try:
                cur = json.loads(self.session_path.read_text(encoding="utf-8"))
            except Exception:
                cur = {}

        companies_field = cur.get("companies")
        company_names_map = cur.get("company_names", {})

        # Migrate older list-based format into dict[{bvdid: name}]
        if isinstance(companies_field, list):
            migrated: Dict[str, Optional[str]] = {}
            for bid in companies_field:
                name = company_names_map.get(bid) or None
                migrated[bid] = name
            cur["companies"] = migrated
            cur.pop("company_names", None)
        elif companies_field is None:
            cur["companies"] = {}
        elif isinstance(companies_field, dict):
            # already in new format
            pass
        else:
            # unknown shape -> reset
            cur["companies"] = {}

        return cur

    # ---- per-company checkpoints ----

    async def get(self, bvdid: str, company_name: Optional[str] = None) -> CompanyCheckpoint:
        """
        Get or create a CompanyCheckpoint. If company_name is provided,
        store it in the checkpoint (and persist).
        """
        async with self._lock:
            cp = self._checkpoints.get(bvdid)
            if cp is None:
                cp = CompanyCheckpoint(bvdid, company_name=company_name)
                await cp.load()
                if company_name and not cp.data.get("company_name"):
                    await cp.set_company_name(company_name)
                self._checkpoints[bvdid] = cp
            else:
                # Upgrade missing name on an existing cp
                if company_name and not cp.data.get("company_name"):
                    await cp.set_company_name(company_name)
            return cp

    # ---- session: company list & names ----

    async def append_company(self, bvdid: str, company_name: Optional[str] = None, **_ignored) -> None:
        """
        Append a company ID (and optionally its name) to session_checkpoint.json.
        Accepts **_ignored to be tolerant to older call sites passing extra kwargs.

        The session file now stores "companies" as a dict: { bvdid: company_name }.
        This function will also migrate older list-based files into the new dict form.
        """
        async with self._lock:
            cur = self._load_session_locked()

            # ensure started_at exists (keep existing if present)
            cur.setdefault("started_at", datetime.now(timezone.utc).isoformat())

            # update/insert the current bvdid entry
            if company_name is not None:
                cur["companies"][bvdid] = company_name
            else:
                # if no name provided and entry missing, set empty string (so presence is recorded)
                cur["companies"].setdefault(bvdid, "")

            payload = json.dumps(cur, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    async def mark_global_start(self) -> None:
        """
        Initialize / reset the session checkpoint for a new run.
        This mirrors the old behavior (fresh structure) but now also zeros
        global progress counters so completion_pct starts from 0.
        """
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            cur: Dict[str, Any] = {
                "started_at": now,
                "companies": {},
                "total_companies": 0,
                "completed_companies": 0,
                "completion_pct": 0.0,
                "last_company_bvdid": None,
                "last_updated": now,
            }
            payload = json.dumps(cur, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    # ---- session: global progress (percentage) ----

    async def set_total_companies(self, total: int) -> None:
        """
        Set the total number of companies in this run.
        Safe to call multiple times; last value wins.
        """
        async with self._lock:
            cur = self._load_session_locked()
            now = datetime.now(timezone.utc).isoformat()

            cur.setdefault("started_at", now)
            cur["total_companies"] = int(total)
            completed = int(cur.get("completed_companies") or 0)

            if cur["total_companies"] > 0:
                cur["completion_pct"] = (completed / cur["total_companies"]) * 100.0
            else:
                cur["completion_pct"] = 0.0

            cur["last_updated"] = now

            payload = json.dumps(cur, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    async def mark_company_completed(self, bvdid: str) -> None:
        """
        Increment the global completed_companies counter and recompute completion_pct.
        Intended to be called once per company when its crawl pipeline finishes
        (regardless of success/failure, as long as it's 'done').
        """
        async with self._lock:
            cur = self._load_session_locked()
            now = datetime.now(timezone.utc).isoformat()

            total = int(cur.get("total_companies") or 0)
            completed = int(cur.get("completed_companies") or 0) + 1

            cur["completed_companies"] = completed
            cur.setdefault("total_companies", total)

            if cur["total_companies"] > 0:
                cur["completion_pct"] = (completed / cur["total_companies"]) * 100.0
            else:
                # If total is unknown, leave pct as 0 or None; caller can still log (done, total)
                cur["completion_pct"] = 0.0

            cur["last_company_bvdid"] = bvdid
            cur["last_updated"] = now

            payload = json.dumps(cur, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    async def get_session_progress(self) -> Dict[str, Any]:
        """
        Convenience helper to read the latest global progress snapshot.
        This reads directly from disk, without relying on in-memory state.
        """
        async with self._lock:
            cur = self._load_session_locked()
            return {
                "total_companies": int(cur.get("total_companies") or 0),
                "completed_companies": int(cur.get("completed_companies") or 0),
                "completion_pct": float(cur.get("completion_pct") or 0.0),
                "last_company_bvdid": cur.get("last_company_bvdid"),
                "started_at": cur.get("started_at"),
                "last_updated": cur.get("last_updated"),
            }

    # ---- misc summary of in-memory company checkpoints ----

    async def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for bvdid, cp in self._checkpoints.items():
            d = cp.data
            out[bvdid] = {
                "company_name": d.get("company_name"),
                "done": d.get("urls_done", 0),
                "failed": d.get("urls_failed", 0),
                "total": d.get("urls_total", 0),
                "ratio": cp.progress_ratio(),
                "finished": cp.is_finished(),
            }
        return out


# ----------------------------
# url_index helpers (for seed-only stage)
# ----------------------------

def url_index_path_for(bvdid: str) -> Path:
    """Public helper to get outputs/{safe}/checkpoints/url_index.json."""
    safe = sanitize_bvdid(bvdid)
    return OUTPUTS_DIR / safe / CHECKPOINTS_DIR / URL_INDEX_NAME


def write_url_index_seed_only(bvdid: str, urls: List[str]) -> None:
    """
    Write url_index.json during a seed-only pipeline so later stages
    can resume without re-seeding. Keeps minimal entries {url: {}}.
    Idempotent and atomic.
    """
    idx_path = url_index_path_for(bvdid)
    existing: Dict[str, Any] = {}
    if idx_path.exists():
        try:
            existing = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    for u in urls:
        if not u:
            continue
        existing.setdefault(u, {})

    _atomic_write_json(idx_path, existing)


async def write_url_index_seed_only_async(bvdid: str, urls: List[str]) -> None:
    await asyncio.to_thread(write_url_index_seed_only, bvdid, urls)