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
import contextlib
from contextlib import contextmanager


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

def _update_url_index_failed(bvdid: str, url: str, reason: str) -> None:
    """
    Best-effort helper to mark a URL as failed in url_index.json.
    We *do not* touch the main 'status' field used for pipeline stages;
    instead we add dedicated failure fields so we can filter on them.
    """
    idx_path = url_index_path_for(bvdid)
    existing: Dict[str, Any] = {}
    if idx_path.exists():
        try:
            existing = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    ent = existing.get(url) or {}
    ent["failed"] = True
    ent["failed_reason"] = reason
    ent["failed_at"] = datetime.now(timezone.utc).isoformat()
    existing[url] = ent

    _atomic_write_json(idx_path, existing)

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

    def _update_url_index_failed(bvdid: str, url: str, reason: str) -> None:
        """
        Best-effort helper to mark a URL as failed in url_index.json.
        We *do not* touch the main 'status' field used for pipeline stages;
        instead we add dedicated failure fields so we can filter on them.
        """
        idx_path = url_index_path_for(bvdid)
        existing: Dict[str, Any] = {}
        if idx_path.exists():
            try:
                existing = json.loads(idx_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        ent = existing.get(url) or {}
        ent["failed"] = True
        ent["failed_reason"] = reason
        ent["failed_at"] = datetime.now(timezone.utc).isoformat()
        existing[url] = ent

        _atomic_write_json(idx_path, existing)

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
        """
        Increment failure counters, append a timestamped error entry, and
        mirror the failure into url_index.json so future runs can decide
        to skip or retry this URL based on the --retry-failed flag.
        """
        self.data["urls_failed"] = int(self.data.get("urls_failed", 0)) + 1
        self.data["last_url"] = url

        errs = self.data.setdefault("errors", [])
        errs.append({
            "url": url,
            "reason": reason,
            "time": datetime.now(timezone.utc).isoformat(),
        })

        # Best-effort: reflect this failure into url_index.json
        try:
            await asyncio.to_thread(
                _update_url_index_failed,
                self.bvdid_original,
                url,
                reason,
            )
        except Exception:
            # Don't let url_index issues break checkpoint updates
            pass

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

@contextmanager
def _session_file_lock(path: Path):
    """
    Cross-process advisory lock for session_checkpoint.json.

    Uses a sidecar *.lock file and fcntl/msvcrt where available.
    Best-effort: if locking fails, we still proceed – but when it works,
    it serializes readers/writers across processes.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Always open in text mode; we only care about the handle for locking.
    f = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl  # type: ignore[attr-defined]
        except Exception:
            fcntl = None

        if fcntl is not None:
            # POSIX advisory lock
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        else:
            # Windows best-effort lock
            try:
                import msvcrt  # type: ignore
                # Ensure file length > 0 or locking length 0 will fail
                try:
                    if lock_path.stat().st_size == 0:
                        f.write("0")
                        f.flush()
                except Exception:
                    pass
                length = max(1, lock_path.stat().st_size)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, length)
            except Exception:
                pass

        yield
    finally:
        try:
            if "fcntl" in locals() and fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            else:
                try:
                    import msvcrt  # type: ignore
                    length = max(1, lock_path.stat().st_size)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, length)
                except Exception:
                    pass
        finally:
            f.close()

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

    def _load_session_unlocked(self) -> Dict[str, Any]:
        """
        Load and normalize the session JSON without taking the asyncio lock.
        Callers are responsible for holding self._lock and (for multi-process
        safety) _session_file_lock(self.session_path).
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

        Now uses a cross-process file lock and always merges with existing
        content, so multi-instance runs don't clobber each other.
        """
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            with _session_file_lock(self.session_path):
                cur = self._load_session_unlocked()

                # ensure started_at exists (keep existing if present)
                cur.setdefault("started_at", cur.get("started_at") or now)

                if not isinstance(cur.get("companies"), dict):
                    cur["companies"] = {}
                companies: Dict[str, Any] = cur["companies"]

                # update/insert the current bvdid entry
                if company_name is not None:
                    companies[bvdid] = company_name
                else:
                    companies.setdefault(bvdid, "")

                # total_companies: number of distinct companies we've ever seen
                existing_total = int(cur.get("total_companies") or 0)
                cur["total_companies"] = max(existing_total, len(companies))

                # last_updated + last_company
                cur["last_company_bvdid"] = bvdid
                cur["last_updated"] = now

                completed = int(cur.get("completed_companies") or 0)
                if cur["total_companies"] > 0:
                    cur["completion_pct"] = (completed / cur["total_companies"]) * 100.0
                else:
                    cur["completion_pct"] = 0.0

                payload = json.dumps(cur, indent=2, ensure_ascii=False)
                await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    async def mark_global_start(self) -> None:
        """
        Initialize the session checkpoint in a non-destructive way.

        If the file already exists, we *reuse* its companies and counters,
        only normalizing the structure and updating last_updated. This prevents
        later runs from wiping earlier progress while still keeping a single
        shared session file.
        """
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            with _session_file_lock(self.session_path):
                cur = self._load_session_unlocked()

                if not cur:
                    # First-time initialization
                    cur = {
                        "started_at": now,
                        "companies": {},
                        "total_companies": 0,
                        "completed_companies": 0,
                        "completion_pct": 0.0,
                        "last_company_bvdid": None,
                        "last_updated": now,
                    }
                else:
                    cur.setdefault("started_at", cur.get("started_at") or now)
                    if not isinstance(cur.get("companies"), dict):
                        cur["companies"] = {}

                    companies: Dict[str, Any] = cur["companies"]
                    total_companies = max(int(cur.get("total_companies") or 0), len(companies))
                    cur["total_companies"] = total_companies

                    completed = int(cur.get("completed_companies") or 0)
                    cur["completed_companies"] = completed

                    if total_companies > 0:
                        cur["completion_pct"] = (completed / total_companies) * 100.0
                    else:
                        cur["completion_pct"] = 0.0

                    cur["last_updated"] = now

                payload = json.dumps(cur, indent=2, ensure_ascii=False)
                await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    # ---- session: global progress (percentage) ----

    async def set_total_companies(self, total: int) -> None:
        """
        Set (or raise) the total number of companies in this session.

        In multi-instance scenarios, we only ever move this value upward and
        also respect the actual number of distinct company IDs recorded so far.
        """
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            with _session_file_lock(self.session_path):
                cur = self._load_session_unlocked()
                cur.setdefault("started_at", cur.get("started_at") or now)

                if not isinstance(cur.get("companies"), dict):
                    cur["companies"] = {}
                companies: Dict[str, Any] = cur["companies"]

                existing_total = max(int(cur.get("total_companies") or 0), len(companies))
                new_total = max(existing_total, int(total))
                cur["total_companies"] = new_total

                completed = int(cur.get("completed_companies") or 0)
                cur["completed_companies"] = completed

                if new_total > 0:
                    cur["completion_pct"] = (completed / new_total) * 100.0
                else:
                    cur["completion_pct"] = 0.0

                cur["last_updated"] = now

                payload = json.dumps(cur, indent=2, ensure_ascii=False)
                await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    async def mark_company_completed(self, bvdid: str) -> None:
        """
        Increment the global completed_companies counter and recompute completion_pct.

        Safe under multi-instance: we always re-read the current snapshot under a
        cross-process lock and only bump the counter + last_company_bvdid.
        """
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            with _session_file_lock(self.session_path):
                cur = self._load_session_unlocked()
                cur.setdefault("started_at", cur.get("started_at") or now)

                if not isinstance(cur.get("companies"), dict):
                    cur["companies"] = {}
                companies: Dict[str, Any] = cur["companies"]

                total = max(int(cur.get("total_companies") or 0), len(companies))
                completed = int(cur.get("completed_companies") or 0) + 1

                cur["total_companies"] = total
                cur["completed_companies"] = completed
                cur["last_company_bvdid"] = bvdid
                cur["last_updated"] = now

                if total > 0:
                    cur["completion_pct"] = (completed / total) * 100.0
                else:
                    cur["completion_pct"] = 0.0

                payload = json.dumps(cur, indent=2, ensure_ascii=False)
                await asyncio.to_thread(_atomic_write_text, self.session_path, payload, "utf-8")

    async def get_session_progress(self) -> Dict[str, Any]:
        """
        Convenience helper to read the latest global progress snapshot.

        Reads from disk under a cross-process file lock and normalizes
        total_companies based on the companies map (if present).
        """
        async with self._lock:
            with _session_file_lock(self.session_path):
                cur = self._load_session_unlocked()

        companies = cur.get("companies")
        if not isinstance(companies, dict):
            companies = {}

        total_companies = max(int(cur.get("total_companies") or 0), len(companies))
        completed = int(cur.get("completed_companies") or 0)

        if total_companies > 0:
            completion_pct = (completed / total_companies) * 100.0
        else:
            completion_pct = 0.0

        return {
            "total_companies": total_companies,
            "completed_companies": completed,
            "completion_pct": completion_pct,
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