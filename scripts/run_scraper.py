from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Iterable, Optional, Dict, Set, Tuple
from urllib.parse import urlparse, urlunparse, ParseResult
import os
import tempfile
import time
import shutil
from threading import Lock, Event, Thread
from collections import deque
import contextlib
import queue as _queue
import gzip
from logging.handlers import (
    QueueHandler,
    QueueListener,
    MemoryHandler,
    TimedRotatingFileHandler,
)

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scraper.browser import init_browser, shutdown_browser
from scraper.crawler import SiteCrawler
from scraper.config import load_config
from scraper.utils import (
    prune_html_for_markdown, html_to_markdown, clean_markdown, save_markdown,
    TransientHTTPError, is_http_url, normalize_url, same_site,
    looks_non_product_url, is_meaningful_markdown,
    get_base_domain, is_producty_url
)

log = logging.getLogger("scripts.run_scraper")

# ================================
# High-throughput logging setup
# ================================

class _GzipTimedRotator(TimedRotatingFileHandler):
    """
    Timed rotating handler that gzips old segments.
    Rolls over at the configured time (e.g., midnight) and keeps backupCount files.
    """
    def __init__(self,
                 filename: str,
                 when: str = "midnight",
                 interval: int = 1,
                 backupCount: int = 14,
                 encoding: str | None = "utf-8"):
        super().__init__(filename, when=when, interval=interval,
                         backupCount=backupCount, encoding=encoding, delay=True)
        # Make rotated names end with .gz
        self.namer = lambda default_name: f"{default_name}.gz"
        self.rotator = self._gzip_rotator

    @staticmethod
    def _gzip_rotator(source: str, dest: str) -> None:
        # dest already has ".gz" because of self.namer
        try:
            with open(source, "rb") as sf, gzip.open(dest, "wb", compresslevel=6) as df:
                shutil.copyfileobj(sf, df)
        finally:
            with contextlib.suppress(Exception):
                os.remove(source)


class LogManager:
    """
    One queue, one listener thread, one MemoryHandler flushing to a gzipped timed-rotating file.
    - Producers attach a QueueHandler to the root.
    - MemoryHandler flushes immediately for WARNING+.
    - A small timer thread flushes the buffer every flush_interval_s for INFO/DEBUG.
    """
    def __init__(
        self,
        log_path: Path,
        *,
        level: int = logging.INFO,
        flush_interval_s: int = 5,
        memory_capacity: int = 2000,
        console_level: int = logging.WARNING,
        rotation_when: str = "midnight",
        rotation_interval: int = 1,
        rotation_backup_count: int = 14,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler with daily rotation + gzip
        self.file_handler = _GzipTimedRotator(
            str(self.log_path),
            when=rotation_when,
            interval=rotation_interval,
            backupCount=rotation_backup_count,
            encoding="utf-8",
        )
        file_fmt = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.file_handler.setFormatter(file_fmt)
        self.file_handler.setLevel(logging.DEBUG)  # accept all; MemoryHandler decides flush

        # Memory buffer: flush on WARNING+, or when explicitly flushed
        self.memory_handler = MemoryHandler(
            capacity=memory_capacity,
            flushLevel=logging.WARNING,
            target=self.file_handler,
        )

        # Optional console (kept minimal to reduce I/O)
        self.console_handler = logging.StreamHandler(stream=sys.stderr)
        self.console_handler.setLevel(console_level)
        self.console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        # Queue + listener serialize disk writes
        self.queue: _queue.Queue = _queue.Queue(maxsize=10000)
        self.queue_handler = QueueHandler(self.queue)
        self.listener = QueueListener(
            self.queue,
            self.memory_handler,   # buffered to file
            self.console_handler,  # direct to console
            respect_handler_level=True,
        )

        # Root logger wiring
        root = logging.getLogger()
        root.setLevel(level)
        # Remove any existing handlers (avoid double logging when re-running)
        for h in list(root.handlers):
            root.removeHandler(h)
        root.addHandler(self.queue_handler)

        # Background periodic flusher for INFO/DEBUG
        self._stop_evt = Event()
        self._flush_interval = max(1, int(flush_interval_s))
        self._flusher = Thread(target=self._flush_loop, name="log-flusher", daemon=True)

    def start(self) -> None:
        self.listener.start()
        self._flusher.start()

    def _flush_loop(self) -> None:
        # Periodically flush buffered INFO/DEBUG
        while not self._stop_evt.wait(self._flush_interval):
            try:
                self.memory_handler.flush()
            except Exception:
                # Avoid crash on logging failure
                pass

    def shutdown(self) -> None:
        # Stop timer
        self._stop_evt.set()
        with contextlib.suppress(Exception):
            self._flusher.join(timeout=3)

        # Final flushes
        with contextlib.suppress(Exception):
            self.memory_handler.flush()
        with contextlib.suppress(Exception):
            self.listener.stop()
        with contextlib.suppress(Exception):
            self.file_handler.flush()
        with contextlib.suppress(Exception):
            self.file_handler.close()

        # Detach queue handler
        root = logging.getLogger()
        with contextlib.suppress(Exception):
            root.removeHandler(self.queue_handler)


# ---------------------------- Data Models ----------------------------

@dataclass(frozen=True)
class CompanyRow:
    hojin_id: str
    company_name: str
    url: str
    hightechflag: str
    us_flag: str

    @property
    def key(self) -> str:
        return f"{self.hojin_id}:{self.url.strip()}"


# Fuse tracking (per-company, in-memory only; cfg stays frozen)
@dataclass
class CompanyRunState:
    forbidden_count: int = 0  # sum of 401 + 403 observed by crawler
    fp_window: int = 8
    pending_fingerprints: deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def set_fp_window(self, n: int) -> None:
        # Rebuild deque to adopt new maxlen, preserving items if any
        items = list(self.pending_fingerprints)
        self.pending_fingerprints = deque(items[-n:], maxlen=max(1, n))
        self.fp_window = max(1, n)


# ----------------------------- IO helpers ----------------------------

def _discover_input_files(*, arg_glob: Optional[str], cfg) -> list[Path]:
    candidates: list[str] = []
    if arg_glob:
        candidates = glob(arg_glob)
        log.info("Using --input-glob=%s → %d file(s) found", arg_glob, len(candidates))
    else:
        if getattr(cfg, "input_glob", None):
            candidates = glob(cfg.input_glob)
            log.info("Using cfg.input_glob=%s → %d file(s) found", cfg.input_glob, len(candidates))
    files: list[Path] = [Path(p) for p in candidates if Path(p).exists()]
    if not files and getattr(cfg, "input_urls_csv", None) and Path(cfg.input_urls_csv).exists():
        files = [Path(cfg.input_urls_csv)]
        log.info("Falling back to single file: %s", cfg.input_urls_csv)
    return sorted(files)


def _read_rows(csv_path: Path) -> Iterable[CompanyRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_url = (row.get("url") or "").strip()
            url = normalize_url(raw_url) if raw_url else ""
            yield CompanyRow(
                hojin_id=str(row.get("hojin_id", "")).strip(),
                company_name=str(row.get("company_name", "")).strip(),
                url=url,
                hightechflag=str(row.get("hightechflag", "")).strip(),
                us_flag=str(row.get("us_flag", "")).strip(),
            )


def _csv_checkpoint_path(checkpoints_dir: Path, csv_input: Path) -> Path:
    return checkpoints_dir / f"{csv_input.stem}.json"


def _load_csv_checkpoint(checkpoints_dir: Path, csv_input: Path) -> set[str]:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _csv_checkpoint_path(checkpoints_dir, csv_input)
    if ckpt.exists():
        try:
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            return set(data.get("done", []))
        except Exception:
            log.warning("CSV checkpoint %s was unreadable; starting fresh.", ckpt)
    return set()


def _save_csv_checkpoint(checkpoints_dir: Path, csv_input: Path, done_keys: set[str]) -> None:
    ckpt = _csv_checkpoint_path(checkpoints_dir, csv_input)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    body = {"done": sorted(done_keys)}

    fd, tmp_name = tempfile.mkstemp(prefix=ckpt.stem + "_", suffix=".tmp", dir=str(ckpt.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        log.exception("Failed writing CSV checkpoint temp %s", ckpt)
        return

    with _state_lock(ckpt):
        backoff = 0.05
        for _ in range(8):
            try:
                os.replace(tmp_name, ckpt)
                return
            except PermissionError:
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.8)
            except Exception:
                log.exception("CSV checkpoint replace failed for %s", ckpt)
                break

    try:
        bak = ckpt.with_suffix(ckpt.suffix + ".bak")
        shutil.move(tmp_name, bak)
        log.warning("CSV checkpoint replace failing for %s; wrote backup %s", ckpt, bak)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        log.exception("CSV checkpoint backup move also failed for %s", ckpt)


def _company_state_path(checkpoints_dir: Path, row: CompanyRow) -> Path:
    host = (urlparse(row.url).hostname or "unknown-host").lower()
    if host.startswith("www."):
        host = host[4:]
    h = hashlib.sha1(row.url.strip().encode("utf-8")).hexdigest()[:10]
    return checkpoints_dir / "companies" / f"{row.hojin_id}_{host}_{h}.json"


def _load_company_state(path: Path, row: CompanyRow | None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Best-effort cleanup of any stale temp file from prior crash
    tmp_glob = f"{path.name}*.tmp"
    try:
        for p in path.parent.glob(tmp_glob):
            # Only delete small json temps; ignore unrelated files
            if p.is_file() and p.stat().st_size < 10_000_000:
                p.unlink(missing_ok=True)
    except Exception:
        pass

    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Company state %s unreadable; recreating.", path)
    if row is None:
        return {"hojin_id": "", "company": "", "homepage": "", "visited": [], "pending": [], "done": False}
    homepage = normalize_url(row.url) if row.url else ""
    pending = [homepage] if homepage else []
    return {
        "hojin_id": row.hojin_id,
        "company": row.company_name,
        "homepage": homepage,
        "visited": [],
        "pending": pending,
        "done": False,
    }


def _save_company_state(path: Path, state: dict) -> None:
    """
    Robust atomic-ish write:
      - write to a temp file in the same directory
      - flush + fsync
      - try os.replace with exponential backoff on Windows AV/locking
      - never raise; log and keep a .bak if we couldn't replace
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use a unique temp file (not a fixed .json.tmp) to avoid races
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # ensure file descriptor closed on json dump error
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        log.exception("Failed writing temp state for %s", path)
        return

    # Replace with retries under a lock (avoid concurrent writers)
    with _state_lock(path):
        backoff = 0.05  # 50ms
        for attempt in range(8):  # ~ (0.05 + 0.1 + 0.2 + ... ~1.5s total)
            try:
                os.replace(tmp_name, path)
                return
            except PermissionError as e:
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.8)
            except Exception:
                log.exception("State replace failed for %s", path)
                break

    # Last resort: keep a backup to avoid total loss
    try:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.move(tmp_name, bak)
        log.warning("State replace still failing for %s; wrote backup %s", path, bak)
    except Exception:
        # If even backup move fails, remove temp to avoid clutter
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        log.exception("State backup move also failed for %s", path)


_MAX_SKIPPED_PER_REASON = 200  # cap samples per company per reason to keep files small

def _state_add_skips(state: dict, page_skips: list[tuple[str, str]]) -> None:
    if not page_skips:
        return
    sk = state.setdefault("skipped", {"reasons": {}, "samples": {}})
    reasons = sk.setdefault("reasons", {})
    samples = sk.setdefault("samples", {})
    for url, reason in page_skips:
        reason = str(reason)
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        arr = samples.setdefault(reason, [])
        if len(arr) < _MAX_SKIPPED_PER_REASON:
            arr.append(url)

def _summarize_skips(state: dict, top_k: int = 3) -> str:
    try:
        reasons = (state.get("skipped", {}) or {}).get("reasons", {}) or {}
        if not reasons:
            return "skips: none"
        # top-k by count desc, then key
        top = sorted(reasons.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:top_k]
        return "skips: " + ", ".join(f"{k}={int(v)}" for k, v in top)
    except Exception:
        return "skips: n/a"

# ---------------------------- URL helpers ----------------------------

def _norm(u: str) -> str:
    try:
        return normalize_url(u)
    except Exception:
        return u or ""


def _with_base_domain(u: str, new_base: str) -> str:
    """Replace the eTLD+1 of u with new_base (keep scheme, path, query, fragment)."""
    try:
        pu = urlparse(u)
        if not pu.hostname:
            return u
        # keep subdomain? Migration adopts base; we normalize to just new_base (no subdomain).
        new_netloc = new_base
        if pu.port:
            new_netloc = f"{new_netloc}:{pu.port}"
        return urlunparse(ParseResult(
            scheme=pu.scheme or "https",
            netloc=new_netloc,
            path=pu.path or "/",
            params=pu.params,
            query=pu.query,
            fragment=pu.fragment,
        ))
    except Exception:
        return u


def _pop_pending_equiv(state: dict, u: str) -> None:
    """Remove any pending entry equivalent to u once normalized."""
    nu = _norm(u)
    pending = state.get("pending", [])
    keep = []
    removed = False
    for p in pending:
        if _norm(p) == nu:
            removed = True
            continue
        keep.append(p)
    if removed:
        state["pending"] = keep


def _rebuild_frontier_after_migration(
    state: dict,
    *,
    new_base: str,
    allow_subdomains: bool,
    old_base: str | None = None,
) -> None:
    """Drop off-domain; map old base→new base when path-equivalent; de-dupe."""
    # Use provided old_base from pre-migration homepage if available; otherwise derive.
    if not old_base:
        home = _norm(state.get("homepage", ""))
        old_base = get_base_domain((urlparse(home).hostname or "")) if home else ""

    visited = {_norm(u) for u in state.get("visited", []) if u}
    new_pending: set[str] = set()

    for u in state.get("pending", []):
        if not u:
            continue
        nu = _norm(u)
        host = (urlparse(nu).hostname or "").lower()
        base = get_base_domain(host) if host else ""

        if base == old_base:
            # map to new base (preserve path/query/fragment)
            mapped = _norm(_with_base_domain(nu, new_base))
            # Only require that the mapped URL is actually on the new base and not visited
            m_host = (urlparse(mapped).hostname or "").lower()
            if get_base_domain(m_host) == new_base and mapped not in visited:
                new_pending.add(mapped)
        else:
            # keep only if already on the new base
            if get_base_domain(host) == new_base and nu not in visited:
                new_pending.add(nu)

    state["pending"] = sorted(new_pending)

# ---------------------------- Redirect/Canonical handlers ----------------------------

def _record_redirect(cfg, state_path: Path, state: dict, src: str, dst: str, code: int, is_homepage: bool) -> None:
    """Handle 3xx (and JS) redirects observed by crawler."""
    try:
        old_host = (urlparse(src).hostname or "").lower()
        new_host = (urlparse(dst).hostname or "").lower()
        if not old_host or not new_host:
            return
        old_base = get_base_domain(old_host)
        new_base = get_base_domain(new_host)
        same = (old_base == new_base)

        log.debug("REDIR %s %s → %s (same-site=%s, homepage=%s)", code, src, dst, same, is_homepage)

        # count off-site targets
        if not same:
            forbid = set(getattr(cfg, "migration_forbid_hosts", ()) or ())
            if new_base in forbid:
                return
            rc = state.setdefault("redirect_counter", {})
            rc[new_base] = int(rc.get(new_base, 0)) + 1
            _save_company_state(state_path, state)

            # 301/308 → migration candidate
            if code in (301, 308):
                seen = rc[new_base]
                need = int(getattr(cfg, "migration_threshold", 2))
                log.info('301/308 migrate candidate: %s → %s (homepage=%s, seen=%d/%d)', old_base, new_base, is_homepage, seen, need)
                if is_homepage or seen >= need:
                    _adopt_migration(cfg, state_path, state, new_base, homepage_dst=dst)
                    return

            # 302/307 on homepage → probation: allow seed to be crawled this pass
            if code in (302, 307) and is_homepage:
                tmp = set(state.get("temp_allowed_hosts", []) or [])
                if new_base not in tmp:
                    tmp.add(new_base)
                    state["temp_allowed_hosts"] = sorted(tmp)
                # enqueue destination as a seed explicitly
                pend = set(state.get("pending", []) or [])
                pend.add(_norm(dst))
                state["pending"] = sorted(pend)
                log.info("[PROBATION] 302/307: homepage redirected to %s; enqueued for this pass.", new_base)
                _save_company_state(state_path, state)

    except Exception as e:
        log.debug("redirect handler failed: %s", e)


def _record_canonical(cfg, state_path: Path, state: dict, src: str, dst: str, is_homepage: bool) -> None:
    """Handle 200 + rel=canonical pointing off-site."""
    try:
        src_host = (urlparse(src).hostname or "").lower()
        dst_host = (urlparse(dst).hostname or "").lower()
        if not src_host or not dst_host:
            return
        src_base = get_base_domain(src_host)
        dst_base = get_base_domain(dst_host)
        if src_base == dst_base:
            return

        forbid = set(getattr(cfg, "migration_forbid_hosts", ()) or ())
        if dst_base in forbid:
            return

        rc = state.setdefault("redirect_counter", {})
        rc[dst_base] = int(rc.get(dst_base, 0)) + 1
        need = int(getattr(cfg, "migration_threshold", 2))
        log.info('CANON migrate candidate: %s → %s (homepage=%s, seen=%d/%d)',
                 src_base, dst_base, is_homepage, rc[dst_base], need)

        if is_homepage or rc[dst_base] >= need:
            _adopt_migration(cfg, state_path, state, dst_base, homepage_dst=dst)
    except Exception as e:
        log.debug("canonical handler failed: %s", e)


def _note_backoff(cfg, state_path: Path, state: dict, host: str) -> None:
    """Remember that we hit 429; runner can slow down next pass."""
    try:
        b = state.setdefault("backoff_hits", 0) + 1
        state["backoff_hits"] = b
        _save_company_state(state_path, state)
        log.info("company backoff triggered ×%d", b)
    except Exception:
        pass


def _note_server_error(cfg, state_path: Path, state: dict, host: str, path: str, status: int) -> None:
    """Tag a sick prefix candidate (for future pass-level suppression)."""
    try:
        pref = path.split("/")
        prefix = "/" + (pref[1] if len(pref) > 1 else "")
        sick = set(state.get("sick_prefixes", []) or [])
        if prefix and prefix != "/":
            if prefix not in sick:
                log.info("prefix %s tagged sick", prefix)
            sick.add(prefix)
            state["sick_prefixes"] = sorted(sick)[:20]
            _save_company_state(state_path, state)
    except Exception:
        pass

def _adopt_migration(cfg, state_path: Path, state: dict, new_base: str, *, homepage_dst: Optional[str] = None) -> None:
    """Adopt a new eTLD+1 as primary for this company; rebuild frontier."""
    old_home = _norm(state.get("homepage", ""))
    old_host = (urlparse(old_home).hostname or "").lower()
    old_base = get_base_domain(old_host) if old_host else ""

    # guard: no-op if already migrated
    if get_base_domain((urlparse(old_home).hostname or "")) == new_base and not homepage_dst:
        return

    aliases = set(state.get("alias_domains", []) or [])
    if old_base and old_base != new_base:
        aliases.add(old_base)
    state["alias_domains"] = sorted(aliases)[:3]

    # set homepage to target (prefer full redirected dst if given)
    if homepage_dst:
        state["homepage"] = _norm(homepage_dst)
    else:
        state["homepage"] = _norm(_with_base_domain(old_home, new_base))
    state["migrated_to"] = new_base

    log.info("[MIGRATE] Domain migration adopted for company: base=%s (aliases: %s)",
             new_base, ", ".join(state.get("alias_domains", [])))

    # Rebuild frontier using the PRE-migration base
    _rebuild_frontier_after_migration(
        state,
        new_base=new_base,
        allow_subdomains=bool(getattr(cfg, "allow_subdomains", False)),
        old_base=old_base,
    )
    _save_company_state(state_path, state)

# ---------------------------- Core routines ----------------------------

# ---- cross-thread/process safe-ish save helpers ----
_STATE_LOCKS: dict[str, Lock] = {}

def _state_lock(path: Path) -> Lock:
    key = str(path.resolve())
    lock = _STATE_LOCKS.get(key)
    if lock is None:
        lock = Lock()
        _STATE_LOCKS[key] = lock
    return lock


async def _crawl_seed_pass(
    cfg,
    crawler: SiteCrawler,
    seed_url: str,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    max_pages_for_this_seed: Optional[int],
) -> list:
    snaps = await crawler.crawl_site(
        homepage=seed_url,
        max_pages=max_pages_for_this_seed if max_pages_for_this_seed is not None else getattr(cfg, "max_pages_per_company", None),
        url_allow_regex=allow_regex,
        url_deny_regex=deny_regex,
    )
    return snaps


def _render_markdown_from_html(cfg, url: str, html: str) -> str:
    try:
        cleaned_html = prune_html_for_markdown(html)
    except Exception as e:
        log.warning("prune_html_for_markdown failed for %s: %s", url, e)
        cleaned_html = html
    try:
        md = clean_markdown(html_to_markdown(cleaned_html))
    except Exception as e:
        log.warning("HTML→Markdown failed for %s: %s", url, e)
        md = cleaned_html
    return md


async def _save_markdown_for_snaps(cfg, crawler: SiteCrawler, snaps: list, *, retried_dynamic: set[str]) -> None:
    total = len(snaps)
    if total == 0:
        return
    dyn_budget = int(getattr(cfg, "dynamic_retry_budget_per_pass", 8))
    dyn_used = 0

    for idx, snap in enumerate(snaps, start=1):
        if not getattr(snap, "html_path", None):
            continue
        p = Path(snap.html_path)
        if not p.exists():
            continue

        if idx == 1 or idx % 10 == 0 or idx == total:
            log.info("Rendering markdown for batch: %d/%d (dynamic budget remaining=%d)", idx, total, max(0, dyn_budget - dyn_used))

        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        md = _render_markdown_from_html(cfg, snap.url, html)

        want_dynamic_retry = (
            bool(getattr(cfg, "enable_static_first", False)) and
            snap.url not in retried_dynamic and
            dyn_used < dyn_budget and
            not is_meaningful_markdown(md,
                                       min_chars=int(getattr(cfg, "md_min_chars", 400)),
                                       min_words=int(getattr(cfg, "md_min_words", 80)),
                                       min_uniq_words=int(getattr(cfg, "md_min_uniq", 50)),
                                       min_lines=int(getattr(cfg, "md_min_lines", 8)))
        )

        if want_dynamic_retry:
            try:
                dyn_snap = await crawler.fetch_dynamic_only(snap.url)
                if dyn_snap.html_path and Path(dyn_snap.html_path).exists():
                    html = Path(dyn_snap.html_path).read_text(encoding="utf-8", errors="ignore")
                    md = _render_markdown_from_html(cfg, snap.url, html)
                    retried_dynamic.add(snap.url)
                    snap.out_links = dyn_snap.out_links
                    snap.html_path = dyn_snap.html_path
                    snap.title = dyn_snap.title
                    dyn_used += 1
                    log.info("Dynamic retry ✓ %s (now %d/%d; budget left=%d)", snap.url, idx, total, max(0, dyn_budget - dyn_used))
            except Exception as e:
                log.debug("Dynamic retry failed for %s: %s", snap.url, e)

        parsed = urlparse(snap.url)
        host = (parsed.hostname or "unknown-host")
        url_path = parsed.path or "/"
        try:
            save_markdown(cfg.markdown_dir, host, url_path, snap.url, md)
        except Exception as e:
            log.warning("save_markdown failed for %s: %s", snap.url, e)


def _update_state_from_snaps(state: dict, snaps: list, homepage_url: str, allow_subdomains: bool) -> None:
    visited = set(_norm(u) for u in state.get("visited", []) if u)
    pending = set(_norm(u) for u in state.get("pending", []) if u)
    home_norm = _norm(homepage_url)

    # --- per-URL metadata bucket ---
    page_meta = state.setdefault("page_meta", {})
    import time as _time
    now = _time.time()

    # Mark visited & update meta (sha1 if available; always update ts)
    for s in snaps:
        su = _norm(s.url)
        if su:
            visited.add(su)
            if su in pending:
                pending.discard(su)
            else:
                _pop_pending_equiv(state, s.url)

            meta = page_meta.get(su, {})
            sha = getattr(s, "content_sha1", None)
            if sha:
                meta["sha1"] = sha
            meta["ts"] = now
            page_meta[su] = meta

    # Grow frontier only from pages we actually fetched content/links from
    for s in snaps:
        for link in getattr(s, "out_links", []) or []:
            lu = _norm(link)
            if not lu:
                continue
            if lu in visited or lu in pending:
                continue
            if looks_non_product_url(lu):
                continue
            if same_site(home_norm, lu, allow_subdomains):
                pending.add(lu)

    state["visited"] = sorted(visited)
    state["pending"] = sorted(pending)
    # company-level crawl timestamp (coarse)
    state["company_last_crawl"] = now


def _frontier_fingerprint(pending_sorted: list[str], cap: int = 10) -> str:
    """Stable, compact fingerprint of pending frontier."""
    if not pending_sorted:
        return "0:"
    sample = ",".join(pending_sorted[:cap])
    return f"{len(pending_sorted)}:{sample}"


async def _drain_company(
    cfg,
    crawler: SiteCrawler,
    state_path: Path,
    state: dict,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    max_pages_per_seed: Optional[int],
    seed_batch: int,
    run_state: CompanyRunState,          # fuse state
) -> bool:
    homepage = _norm(state.get("homepage") or "")
    allow_subdomains = bool(getattr(cfg, "allow_subdomains", False))

    cap_total = int(getattr(cfg, "max_pages_per_company", 300) or 300)
    already = len({_norm(u) for u in state.get("visited", []) if u})
    remaining = max(0, cap_total - already)
    if remaining == 0:
        state["done"] = True
        state.setdefault("done_reason", "cap_reached")
        _save_company_state(state_path, state)
        log.debug("Company cap already reached (%d). Marking done.", cap_total)
        return True

    # Fuses (config-driven; cfg stays frozen)
    forbidden_thresh = int(getattr(cfg, "forbidden_done_threshold", 0) or 0)
    stall_pending_max = int(getattr(cfg, "stall_pending_max", 2))
    stall_repeat_passes = int(getattr(cfg, "stall_repeat_passes", 3))
    stall_fp_window = int(getattr(cfg, "stall_fingerprint_window", max(3, stall_repeat_passes)))
    if run_state.fp_window != stall_fp_window:
        run_state.set_fp_window(stall_fp_window)

    progress = False
    retried_dynamic: set[str] = set()

    while state["pending"] and remaining > 0:
        # Fuse A: 401/403 count
        if forbidden_thresh > 0 and run_state.forbidden_count >= forbidden_thresh:
            state["done"] = True
            state["done_reason"] = "403_fuse"
            state["blocked_reason"] = "forbidden"
            state["done_meta"] = {"count": run_state.forbidden_count, "threshold": forbidden_thresh}
            _save_company_state(state_path, state)
            log.info("Fuse trip: 401/403 %d ≥ %d → mark done.", run_state.forbidden_count, forbidden_thresh)
            return True

        pend_set = {_norm(u) for u in state["pending"] if u}
        # product-first sort (0 for producty)
        pending_list = sorted(pend_set, key=lambda u: 0 if is_producty_url(u) else 1)
        state["pending"] = pending_list
        seeds = pending_list[:seed_batch]
        if not seeds:
            break

        # Fuse B: stall detector (small, stable frontier)
        if len(pending_list) <= stall_pending_max:
            fp = _frontier_fingerprint(pending_list, cap=10)
            run_state.pending_fingerprints.append(fp)
            if len(run_state.pending_fingerprints) >= stall_repeat_passes:
                tail = list(run_state.pending_fingerprints)[-stall_repeat_passes:]
                if len(set(tail)) == 1:
                    state["done"] = True
                    state["done_reason"] = "stall_fuse"
                    state["blocked_reason"] = "stalled"
                    state["done_meta"] = {"pending": len(pending_list), "repeat": stall_repeat_passes}
                    _save_company_state(state_path, state)
                    log.info("Fuse trip: stall (pending=%d, repeat=%d) → mark done.",
                             len(pending_list), stall_repeat_passes)
                    return True
        else:
            fp = _frontier_fingerprint(pending_list, cap=10)
            run_state.pending_fingerprints.append(fp)

        denom = max(1, len(seeds))
        share = max(1, remaining // denom)
        if max_pages_per_seed is not None:
            share = min(share, max_pages_per_seed)

        log.debug(
            "Draining %s: remaining=%d cap=%d batch=%d share/seed=%d (visited=%d)",
            state.get("company", ""), remaining, cap_total, len(seeds), share, already
        )

        tasks = [
            asyncio.create_task(
                _crawl_seed_pass(
                    cfg, crawler, u,
                    allow_regex=allow_regex,
                    deny_regex=deny_regex,
                    max_pages_for_this_seed=share,
                )
            )
            for u in seeds
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # remove attempted seeds from pending
        current_pending = {_norm(u) for u in state.get("pending", []) if u}
        for s_url in seeds:
            current_pending.discard(s_url)
        state["pending"] = sorted(current_pending)

        for seed, res in zip(seeds, results):
            if isinstance(res, Exception):
                # keep failed seed for another pass
                if seed not in state["pending"]:
                    state["pending"].append(seed)
                _save_company_state(state_path, state)
                log.debug("Seed failed, re-queued: %s", seed)
                continue

            snaps = res or []
            prev_visited = len(state.get("visited", []))

            await _save_markdown_for_snaps(cfg, crawler, snaps, retried_dynamic=retried_dynamic)

            _update_state_from_snaps(
                state, snaps,
                homepage_url=homepage,
                allow_subdomains=allow_subdomains
            )

            for s in snaps:
                if getattr(s, "dropped", None):
                    _state_add_skips(state, s.dropped)

            _save_company_state(state_path, state)

            now_visited = len(state.get("visited", []))
            gained = max(0, now_visited - prev_visited)
            remaining = max(0, remaining - gained)
            already += gained

            if gained > 0:
                progress = True
                log.debug(
                    "Progress: +%d pages; visited=%d pending=%d remaining=%d",
                    gained, now_visited, len(state["pending"]), remaining
                )

        if remaining == 0:
            log.info("Company cap reached (%d). Stopping company.", cap_total)
            break

        if not progress:
            _save_company_state(state_path, state)
            log.info("No progress this pass for %s (%s).",
                     state.get("company", ""), _summarize_skips(state))
            # After a no-progress pass, re-check fuses immediately
            pending_list = sorted({_norm(u) for u in state.get("pending", []) if u},
                                  key=lambda u: 0 if is_producty_url(u) else 1)
            fp = _frontier_fingerprint(pending_list, cap=10)
            run_state.pending_fingerprints.append(fp)
            if forbidden_thresh > 0 and run_state.forbidden_count >= forbidden_thresh:
                state["done"] = True
                state["done_reason"] = "403_fuse"
                state["blocked_reason"] = "forbidden"
                state["done_meta"] = {"count": run_state.forbidden_count, "threshold": forbidden_thresh}
                _save_company_state(state_path, state)
                log.info("Fuse trip: 401/403 %d ≥ %d → mark done.",
                         run_state.forbidden_count, forbidden_thresh)
                return True
            if len(pending_list) <= stall_pending_max and len(run_state.pending_fingerprints) >= stall_repeat_passes:
                tail = list(run_state.pending_fingerprints)[-stall_repeat_passes:]
                if len(set(tail)) == 1:
                    state["done"] = True
                    state["done_reason"] = "stall_fuse"
                    state["blocked_reason"] = "stalled"
                    state["done_meta"] = {"pending": len(pending_list), "repeat": stall_repeat_passes}
                    _save_company_state(state_path, state)
                    log.info("Fuse trip: stall (pending=%d, repeat=%d) → mark done.",
                             len(pending_list), stall_repeat_passes)
                    return True
            continue

        progress = False

    finished = bool(state.get("done", False) or remaining == 0 or len(state.get("pending", [])) == 0)
    state["done"] = finished
    if finished and "done_reason" not in state:
        state["done_reason"] = "frontier_exhausted" if len(state.get("pending", [])) == 0 else "cap_reached"
    _save_company_state(state_path, state)

    if finished:
        log.info("Company DONE: %s (visited=%d, pending=%d) — %s",
                 state.get("company",""),
                 len(state.get("visited",[])),
                 len(state.get("pending",[])),
                 _summarize_skips(state))
    else:
        log.info("Company PASS complete: %s (visited=%d, pending=%d, remaining=%d)",
                 state.get("company",""),
                 len(state.get("visited",[])),
                 len(state.get("pending",[])),
                 remaining)

    try:
        sk = (state.get("skipped", {}) or {}).get("reasons", {}) or {}
        log.debug("End-of-company counters: visited=%d, pending=%d, skipped_total=%d (by reason: %s)",
                  len(state.get("visited", [])),
                  len(state.get("pending", [])),
                  sum(int(v) for v in sk.values()),
                  ", ".join(f"{k}={int(v)}" for k, v in sorted(sk.items())))
    except Exception:
        pass

    return finished


async def _process_company(
    cfg,
    context,
    row: CompanyRow,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    max_pages_per_seed: Optional[int],
    seed_batch: int,
    company_attempts: int,
) -> bool:
    state_path = _company_state_path(cfg.checkpoints_dir, row)
    state = _load_company_state(state_path, row)
    if state.get("done"):
        return True

    _save_company_state(state_path, state)
    log.debug("Company %s: starting with pending=%d visited=%d",
              row.company_name, len(state["pending"]), len(state["visited"]))

    # ---- wire crawler with stateful callbacks ----
    def cb_redirect(src: str, dst: str, code: int, is_home: bool):
        _record_redirect(cfg, state_path, state, src, dst, code, is_home)

    def cb_canonical(src: str, dst: str, is_home: bool):
        _record_canonical(cfg, state_path, state, src, dst, is_home)

    def cb_backoff(host: str):
        _note_backoff(cfg, state_path, state, host)

    def cb_server_error(host: str, path: str, status: int):
        _note_server_error(cfg, state_path, state, host, path, status)

    run_state = CompanyRunState()
    run_state.set_fp_window(int(getattr(cfg, "stall_fingerprint_window", 4)))

    def cb_http_status(host: str, path: str, status: int):
        # count both 401 and 403
        if status in (401, 403):
            run_state.forbidden_count += 1

    crawler = SiteCrawler(
        cfg, context,
        on_redirect=cb_redirect,
        on_canonical=cb_canonical,
        on_backoff=cb_backoff,
        on_server_error=cb_server_error,
        on_http_status=cb_http_status,   # increment 401/403
    )

    # --- feed per-URL last-seen (for If-Modified-Since) ---
    last_seen: dict[str, float] = {}
    try:
        for u, meta in (state.get("page_meta", {}) or {}).items():
            ts = meta.get("ts")
            if isinstance(ts, (int, float)) and ts > 0:
                last_seen[normalize_url(u)] = float(ts)
    except Exception:
        pass
    setattr(crawler, "_ims", last_seen)

    # (Optional) Product-first order for company seeds
    try:
        pend = [p for p in state.get("pending", []) if p]
        if pend:
            pend_sorted = sorted({normalize_url(p) for p in pend},
                                 key=lambda u: 0 if is_producty_url(u) else 1)
            state["pending"] = pend_sorted
            _save_company_state(state_path, state)
    except Exception:
        pass

    for attempt in range(1, max(1, company_attempts) + 1):
        try:
            finished = await _drain_company(
                cfg, crawler, state_path, state,
                allow_regex=allow_regex,
                deny_regex=deny_regex,
                max_pages_per_seed=max_pages_per_seed,
                seed_batch=seed_batch,
                run_state=run_state,
            )
            return finished
        except TransientHTTPError as e:
            log.warning("Transient company-level error (%s) on %s attempt %d; retrying...",
                        e, row.url, attempt)
            await asyncio.sleep(0.8 * attempt)
        except Exception:
            log.exception("Hard failure for company %s (%s); will not mark done.", row.company_name, row.url)
            break

    return False

# ------------- (kept for reference/back-compat; no longer used in main) -------------
async def _process_csv(
    cfg,
    csv_path: Path,
    *,
    context,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    resume: bool,
    limit: Optional[int],
    max_pages_per_seed: Optional[int],
    seed_batch: int,
    company_attempts: int,
) -> None:
    """
    Legacy per-file parallelizer (retained for back-compat or focused runs).
    The new main() uses a global queue across all files instead.
    """
    done = _load_csv_checkpoint(cfg.checkpoints_dir, csv_path)
    rows_all = list(_read_rows(csv_path))

    rows_all = [
        r for r in rows_all
        if r.url and is_http_url(r.url) and (urlparse(r.url).hostname or "").strip()
    ]
    if limit:
        rows_all = rows_all[:limit]

    log.info("CSV %s: loaded %d rows", csv_path.name, len(rows_all))
    if resume:
        rows = [r for r in rows_all if r.key not in done]
        log.info("CSV %s: %d already complete (skipped), %d to process",
                 csv_path.name, len(rows_all) - len(rows), len(rows))
    else:
        rows = rows_all
        log.info("CSV %s: resume disabled; processing %d rows", csv_path.name, len(rows))

    if not rows:
        return

    sem = asyncio.Semaphore(cfg.max_companies_parallel)
    tasks: list[asyncio.Task] = []

    async def worker(row: CompanyRow):
        if resume and row.key in done:
            return
        async with sem:
            finished = await _process_company(
                cfg,
                context,
                row,
                allow_regex=allow_regex,
                deny_regex=deny_regex,
                max_pages_per_seed=max_pages_per_seed,
                seed_batch=seed_batch,
                company_attempts=max(2, company_attempts),
            )
            if finished:
                done.add(row.key)
                _save_csv_checkpoint(cfg.checkpoints_dir, csv_path, done)

    for row in rows:
        tasks.append(asyncio.create_task(worker(row)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
# ------------------------------------------------------------------------------------


def _parse_args(argv: list[str]):
    import argparse

    p = argparse.ArgumentParser(description="Run the batch scraper with true intra-company resume.")
    p.add_argument("--input-glob", default=None,
                   help='Glob for input CSVs (default: cfg.input_glob, e.g. "data/input/us/*.csv")')
    p.add_argument("--pattern", default=None,
                   help="Deprecated: use --input-glob instead.")
    p.add_argument("--limit", type=int, default=None, help="Limit companies per CSV (debugging)")
    # Resume controls
    p.add_argument("--max-pages-per-seed", type=int, default=40,
                   help="Cap pages fetched per seed pass (default: 40); company-wide cap enforced too")
    p.add_argument("--seed-batch", type=int, default=6,
                   help="How many seeds to crawl concurrently per company (default: 6)")
    p.add_argument("--allow", type=str, default=None, help="Allow regex for URLs (e.g. /products|/solutions)")
    p.add_argument("--deny", type=str, default=None, help="Deny regex for URLs (e.g. /blog|/careers)")
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoints and start fresh")
    p.add_argument("--company-attempts", type=int, default=3, help="Attempts to recover a company on transient errors (default: 3)")
    p.add_argument("--debug", action="store_true", help="Set logging level to DEBUG for this run")
    return p.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    cfg = load_config()

    # -------- Optimized logging --------
    # - Daily rotation + gzip
    # - Buffered writes flushed every 5s or on WARNING+
    log_mgr = LogManager(
        log_path=cfg.log_file,
        level=(logging.DEBUG if args.debug else logging.INFO),
        flush_interval_s=5,          # periodic batch flush
        memory_capacity=4000,        # buffer size before forced flush (also flushes on WARNING+)
        console_level=(logging.INFO if args.debug else logging.WARNING),
        rotation_when="midnight",
        rotation_interval=1,
        rotation_backup_count=14,    # keep two weeks of gz logs
    )
    log_mgr.start()

    input_glob = args.input_glob or args.pattern or getattr(cfg, "input_glob", None)

    files = _discover_input_files(arg_glob=input_glob, cfg=cfg)
    if not files:
        logging.getLogger(__name__).error(
            "No input CSVs found. Checked --input-glob (%s), cfg.input_glob (%s), and fallback single file %s",
            args.input_glob or args.pattern, getattr(cfg, "input_glob", None), getattr(cfg, "input_urls_csv", None),
        )
        log_mgr.shutdown()
        return

    log.info("Discovered %d input file(s). Example: %s", len(files), files[0])

    # Quick sample count (best-effort)
    total_rows = 0
    for pth in files[:5]:
        try:
            total_rows += sum(1 for _ in _read_rows(pth))
        except Exception:
            pass
    log.info("Sampling shows at least ~%d rows in first %d file(s).", total_rows, min(5, len(files)))

    # -------- Global Work Queue across all files --------
    # 1) Preload per-file done sets and candidate rows (respect --limit per file)
    per_file_done: Dict[Path, Set[str]] = {}
    per_file_rows: Dict[Path, list[CompanyRow]] = {}
    total_enqueued = 0

    resume = not args.no_resume

    for csv_path in files:
        done = _load_csv_checkpoint(cfg.checkpoints_dir, csv_path)
        per_file_done[csv_path] = done

        rows_all = [
            r for r in _read_rows(csv_path)
            if r.url and is_http_url(r.url) and (urlparse(r.url).hostname or "").strip()
        ]
        if args.limit:
            rows_all = rows_all[:args.limit]

        if resume:
            rows = [r for r in rows_all if r.key not in done]
            log.info("CSV %s: %d already complete (skipped), %d to process",
                     csv_path.name, len(rows_all) - len(rows), len(rows))
        else:
            rows = rows_all
            log.info("CSV %s: resume disabled; processing %d rows", csv_path.name, len(rows))

        per_file_rows[csv_path] = rows
        total_enqueued += len(rows)

    if total_enqueued == 0:
        log.info("Nothing to do — all companies already completed across %d file(s).", len(files))
        log_mgr.shutdown()
        return

    # 2) Build the global queue of (csv_path, CompanyRow)
    work_q: asyncio.Queue[Tuple[Path, CompanyRow] | None] = asyncio.Queue()
    for csv_path, rows in per_file_rows.items():
        for row in rows:
            work_q.put_nowait((csv_path, row))

    log.info("Global queue initialized with %d companies from %d file(s).", total_enqueued, len(files))

    # 3) Spin up cfg.max_companies_parallel workers that pull any row
    pw, browser, context = await init_browser(cfg)

    # -------- Pressure-aware throttling --------
    concurrency = max(1, int(getattr(cfg, "max_companies_parallel", 6)))
    throttle_sem = asyncio.Semaphore(concurrency)

    inflight_companies = 0
    inflight_lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def _inflight_guard():
        nonlocal inflight_companies
        async with inflight_lock:
            inflight_companies += 1
        try:
            yield
        finally:
            async with inflight_lock:
                inflight_companies -= 1

    nearcap_streak = 0
    max_pages_cap = int(getattr(cfg, "max_global_pages_open", 256))
    nearcap_threshold = max(1, int(max_pages_cap * 0.95))

    monitor_interval = max(30, int(getattr(cfg, "watchdog_interval_seconds", 30)))
    idle_hold_seconds = monitor_interval  # how long to idle some workers

    idling_task: Optional[asyncio.Task] = None

    async def _pressure_monitor():
        nonlocal nearcap_streak, idling_task
        try:
            while True:
                await asyncio.sleep(monitor_interval)
                try:
                    pages_now = len(context.pages)
                except Exception:
                    pages_now = 0

                log.info(
                    "Pressure: pages_open=%d/%d; inflight_companies=%d; nearcap_streak=%d",
                    pages_now, max_pages_cap, inflight_companies, nearcap_streak
                )

                if pages_now >= nearcap_threshold:
                    nearcap_streak += 1
                else:
                    nearcap_streak = 0

                if nearcap_streak >= 3 and idling_task is None:
                    idle_n = max(1, concurrency // 4)
                    log.warning(
                        "High pressure: %d pages ≥ %d (95%% cap). Idling %d worker(s) for %ds.",
                        pages_now, nearcap_threshold, idle_n, idle_hold_seconds
                    )

                    async def _idle_for_a_bit(to_idle: int, duration: int):
                        acquired = 0
                        for _ in range(to_idle):
                            try:
                                await throttle_sem.acquire()
                                acquired += 1
                            except Exception:
                                break
                        try:
                            await asyncio.sleep(duration)
                        finally:
                            for _ in range(acquired):
                                throttle_sem.release()

                    idling_task = asyncio.create_task(_idle_for_a_bit(idle_n, idle_hold_seconds))

                    def _clear(_):
                        nonlocal idling_task
                        idling_task = None

                    idling_task.add_done_callback(_clear)
        except asyncio.CancelledError:
            pass

    pressure_task = asyncio.create_task(_pressure_monitor())

    try:
        async def worker_loop(wid: int):
            while True:
                item = await work_q.get()
                if item is None:
                    work_q.task_done()
                    return

                await throttle_sem.acquire()
                try:
                    csv_path, row = item
                    try:
                        async with _inflight_guard():
                            finished = await _process_company(
                                cfg,
                                context,
                                row,
                                allow_regex=args.allow,
                                deny_regex=args.deny,
                                max_pages_per_seed=args.max_pages_per_seed,
                                seed_batch=args.seed_batch,
                                company_attempts=max(2, args.company_attempts),
                            )
                        if finished:
                            per_file_done[csv_path].add(row.key)
                            _save_csv_checkpoint(cfg.checkpoints_dir, csv_path, per_file_done[csv_path])
                    except Exception:
                        log.exception("Worker %d: unhandled error while processing %s (%s)", wid, row.company_name, row.url)
                    finally:
                        work_q.task_done()
                finally:
                    throttle_sem.release()

        workers = [asyncio.create_task(worker_loop(i + 1)) for i in range(concurrency)]

        await work_q.join()

        for _ in workers:
            work_q.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)

    finally:
        try:
            if 'pressure_task' in locals() and pressure_task:
                pressure_task.cancel()
                with contextlib.suppress(Exception):
                    await pressure_task
        except Exception:
            pass

        try:
            if 'idling_task' in locals() and idling_task:
                idling_task.cancel()
                with contextlib.suppress(Exception):
                    await idling_task
        except Exception:
            pass

        await shutdown_browser(pw, browser, context)
        # Ensure all logs are flushed & files closed
        log_mgr.shutdown()


def main() -> None:
    asyncio.run(main_async())

if __name__ == "__main__":
    main()