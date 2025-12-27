from __future__ import annotations

import errno
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from extensions import output_paths

# --------------------------------------------------------------------------------------
# RetryState: compact company-level failure state + append-only failure ledger
# --------------------------------------------------------------------------------------

RetryClass = str  # "net" | "stall" | "mem" | "other" | "permanent"
StallKind = str  # "goto" | "no_yield" | "rate_limit" | "unknown"

# Defaults (net/stall LIMITED; mem/other UNLIMITED)
DEFAULT_MAX_ATTEMPTS_NET = 3
DEFAULT_MAX_ATTEMPTS_STALL = 4
DEFAULT_MAX_ATTEMPTS_STALL_FAST = 2  # goto/no_yield
DEFAULT_NXDOMAIN_CUTOFF = 1

DEFAULT_JITTER_FRAC = 0.15
DEFAULT_FLUSH_MIN_INTERVAL_SEC = 1.0

# A "far future" timestamp that is always > now (year 3000)
_FAR_FUTURE_TS = 32503680000.0

# Backoff schedules (seconds)
_NET_SCHEDULE = [20, 60, 300]
_STALL_SCHEDULE = [60, 300, 1200, 3600]  # "real" stalls like 429/overloaded
_STALL_FAST_SCHEDULE = [30, 120]  # goto/no_yield: quick decision then quarantine
_MEM_SCHEDULE = [300, 1800, 7200, 21600, 86400]
_OTHER_SCHEDULE = [60, 300, 1800, 7200, 21600, 86400]


def _output_root() -> Path:
    """Resolve output root dynamically. output_paths.OUTPUT_ROOT may be mutated at runtime."""
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


def _retry_base_dir() -> Path:
    """Canonical retry directory under output root. Prefer output_paths.retry_base_dir() if present."""
    fn = getattr(output_paths, "retry_base_dir", None)
    if callable(fn):
        try:
            p = fn()
            return Path(p).resolve()
        except Exception:
            pass
    return (_output_root() / "_retry").resolve()


def _global_path(*parts: str) -> Path:
    """Prefer output_paths.global_path if present (keeps all modules consistent)."""
    fn = getattr(output_paths, "global_path", None)
    if callable(fn):
        try:
            p = fn(*parts)
            return Path(p).resolve()
        except Exception:
            pass
    return _output_root().joinpath(*parts).resolve()


def _retry_emfile(fn, attempts: int = 6, base_delay: float = 0.12):
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
            try:
                if tmp.exists() and tmp != path:
                    tmp.unlink()
            except Exception:
                pass

    _retry_emfile(_write)


def _atomic_write_json_compact(path: Path, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    _atomic_write_text(path, payload, "utf-8")


def _json_load_nocache(path: Path) -> Any:
    def _read() -> Any:
        try:
            if not path.exists():
                return None
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return None

    return _retry_emfile(_read)


def _append_jsonl_many(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Append many JSONL rows in a single open() for speed. Best-effort fsync."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
    )

    def _write() -> None:
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(lines)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass

    _retry_emfile(_write)


def _now_ts() -> float:
    return time.time()


def _jitter(seconds: float, frac: float = DEFAULT_JITTER_FRAC) -> float:
    if seconds <= 0:
        return 0.0
    r = random.random()
    delta = (r * 2.0 - 1.0) * frac
    return max(0.0, seconds * (1.0 + delta))


def _schedule_pick(attempts: int, schedule: Sequence[float]) -> float:
    a = max(1, int(attempts))
    idx = min(a, len(schedule)) - 1
    return float(schedule[idx])


def _backoff_seconds(
    attempts: int, *, kind: RetryClass, stall_kind: StallKind = "unknown"
) -> float:
    """
    Backoff policy:
      - net: quick retries then quarantine
      - stall:
          - goto/no_yield: short schedule (fewer attempts)
          - rate_limit/unknown: longer schedule
      - mem/other: unlimited, increasingly slow
    """
    if kind == "net":
        return _schedule_pick(attempts, _NET_SCHEDULE)
    if kind == "stall":
        if stall_kind in ("goto", "no_yield"):
            return _schedule_pick(attempts, _STALL_FAST_SCHEDULE)
        return _schedule_pick(attempts, _STALL_SCHEDULE)
    if kind == "mem":
        return _schedule_pick(attempts, _MEM_SCHEDULE)
    if kind == "other":
        return _schedule_pick(attempts, _OTHER_SCHEDULE)
    return _schedule_pick(attempts, _OTHER_SCHEDULE)


def _is_transient_message(msg: str) -> bool:
    """
    Transient events (do NOT count as attempts; do NOT quarantine).
    Keep this conservative: only known scheduler-cancel / lifecycle noise.
    """
    m = (msg or "").lower()
    if not m:
        return False
    needles = [
        "cancelled by scheduler",
        "canceled by scheduler",
        "scheduler_cancel",
        "scheduler cancel",
        "cancel_reason=scheduler",
        "task cancelled",
        "task canceled",
        "cancellederror",
        "asyncio.cancellederror",
    ]
    return any(n in m for n in needles)


# --------------------------------------------------------------------------------------
# Datamodel
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class CompanyRetryState:
    cls: RetryClass = "net"
    attempts: int = 0
    next_eligible_at: float = 0.0
    updated_at: float = 0.0
    last_error: str = ""
    last_stage: str = ""

    # per-class counters (used for policy / quarantine decisions)
    net_attempts: int = 0
    stall_attempts: int = 0
    mem_attempts: int = 0
    other_attempts: int = 0

    # informational
    mem_hits: int = 0
    last_stall_kind: StallKind = "unknown"

    # progress tracking (used to avoid stall quarantine while progress is being made)
    last_progress_md_done: int = 0  # highest md_done observed
    last_seen_md_done: int = 0  # last md_done reported by caller

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CompanyRetryState":
        cls = str(d.get("cls") or "net")
        if cls == "mem_heavy":
            cls = "mem"
        return CompanyRetryState(
            cls=cls,
            attempts=int(d.get("attempts") or 0),
            next_eligible_at=float(d.get("next_eligible_at") or 0.0),
            updated_at=float(d.get("updated_at") or 0.0),
            last_error=str(d.get("last_error") or ""),
            last_stage=str(d.get("last_stage") or ""),
            net_attempts=int(d.get("net_attempts") or 0),
            stall_attempts=int(d.get("stall_attempts") or 0),
            mem_attempts=int(d.get("mem_attempts") or 0),
            other_attempts=int(d.get("other_attempts") or 0),
            mem_hits=int(d.get("mem_hits") or 0),
            last_stall_kind=str(d.get("last_stall_kind") or "unknown"),
            last_progress_md_done=int(d.get("last_progress_md_done") or 0),
            last_seen_md_done=int(d.get("last_seen_md_done") or 0),
        )


# --------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------


class RetryStateStore:
    """
    Company-level retry state store:
      - retry_state.json : compact snapshot (atomic rewrite)
      - failure_ledger.jsonl : append-only audit trail
      - quarantine.json : terminal quarantines (small set)

    POLICY:
      - net/stall are LIMITED retries
      - mem/other are UNLIMITED retries
      - permanent quarantines immediately

    IMPORTANT:
      - Scheduler cancellations are TRANSIENT (do not count as failures / attempts).
      - Stall quarantine is progress-aware (if md_done increased, do not quarantine).
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        *,
        flush_min_interval_sec: float = DEFAULT_FLUSH_MIN_INTERVAL_SEC,
        default_max_attempts_net: int = DEFAULT_MAX_ATTEMPTS_NET,
        default_max_attempts_stall: int = DEFAULT_MAX_ATTEMPTS_STALL,
        default_nxdomain_cutoff: int = DEFAULT_NXDOMAIN_CUTOFF,
        default_max_attempts_stall_fast: int = DEFAULT_MAX_ATTEMPTS_STALL_FAST,
    ) -> None:
        if base_dir is None:
            base_dir = _retry_base_dir()
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.base_dir / "retry_state.json"
        self.ledger_path = self.base_dir / "failure_ledger.jsonl"
        self.quarantine_path = self.base_dir / "quarantine.json"

        self._lock = threading.Lock()
        self._state: Dict[str, CompanyRetryState] = {}
        self._quarantine: Dict[str, Dict[str, Any]] = {}
        self._dirty_state = False
        self._dirty_quarantine = False
        self._last_flush_ts = 0.0
        self._flush_min_interval_sec = max(0.0, float(flush_min_interval_sec))

        # store-level defaults (used for clamp)
        self._default_max_attempts_net = max(0, int(default_max_attempts_net))
        self._default_max_attempts_stall = max(0, int(default_max_attempts_stall))
        self._default_nxdomain_cutoff = max(0, int(default_nxdomain_cutoff))
        self._default_max_attempts_stall_fast = max(
            0, int(default_max_attempts_stall_fast)
        )

        self._load()
        # Ensure files exist even if empty (reduces “missing file” ambiguity)
        self.flush(force=True)
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.ledger_path.exists():
                _atomic_write_text(self.ledger_path, "")
        except Exception:
            pass

    # ------------------ persistence ------------------

    def _load(self) -> None:
        with self._lock:
            raw = _json_load_nocache(self.state_path)
            if isinstance(raw, dict):
                st: Dict[str, CompanyRetryState] = {}
                for cid, v in raw.items():
                    if isinstance(v, dict):
                        st[str(cid)] = CompanyRetryState.from_dict(v)
                self._state = st
            else:
                self._state = {}

            qraw = _json_load_nocache(self.quarantine_path)
            if isinstance(qraw, dict):
                self._quarantine = {
                    str(k): (v if isinstance(v, dict) else {}) for k, v in qraw.items()
                }
            else:
                self._quarantine = {}

            self._dirty_state = False
            self._dirty_quarantine = False
            self._last_flush_ts = _now_ts()

    def flush(self, *, force: bool = False) -> None:
        """Flush only when dirty, and no more often than flush_min_interval_sec unless force=True."""
        now = _now_ts()
        with self._lock:
            if not force:
                if not (self._dirty_state or self._dirty_quarantine):
                    return
                if (
                    self._flush_min_interval_sec > 0
                    and (now - self._last_flush_ts) < self._flush_min_interval_sec
                ):
                    return

            snapshot = {cid: s.to_dict() for cid, s in self._state.items()}
            quarantine = dict(self._quarantine)
            self._dirty_state = False
            self._dirty_quarantine = False
            self._last_flush_ts = now

        # Best-effort persistence; do not crash run loop if disk hiccups.
        try:
            _atomic_write_json_compact(self.state_path, snapshot)
        except Exception:
            pass
        try:
            _atomic_write_json_compact(self.quarantine_path, quarantine)
        except Exception:
            pass

    # ------------------ crawl_state integration (best-effort) ------------------

    def _crawl_state_db_path(self) -> Path:
        return _global_path("crawl_state.sqlite3")

    def _try_mark_terminal_done(
        self,
        company_id: str,
        *,
        done_reason: str,
        details: Dict[str, Any],
        last_error: str,
    ) -> None:
        """Best-effort integration: mark terminal_done in crawl_state.sqlite3. Must NEVER raise."""
        try:
            from extensions.crawl_state import get_crawl_state  # type: ignore

            cs = get_crawl_state(db_path=self._crawl_state_db_path())
            fn = getattr(cs, "mark_company_terminal_sync", None)
            if callable(fn):
                fn(
                    company_id,
                    reason=done_reason,
                    details=details,
                    last_error=last_error,
                )
        except Exception:
            return

    def _try_clear_terminal_done(self, company_id: str) -> None:
        """Best-effort: clear terminal marker if someone later succeeds or re-runs."""
        try:
            from extensions.crawl_state import get_crawl_state  # type: ignore

            cs = get_crawl_state(db_path=self._crawl_state_db_path())
            fn = getattr(cs, "clear_company_terminal_sync", None)
            if callable(fn):
                fn(company_id, keep_status=False)
        except Exception:
            return

    # ------------------ classification helpers ------------------

    @staticmethod
    def classify_unreachable_error(error: str) -> bool:
        if not error:
            return False
        e = error.lower()
        needles = [
            "name or service not known",
            "nxdomain",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "no address associated with hostname",
        ]
        return any(n in e for n in needles)

    @staticmethod
    def _looks_rate_limited(error: str, status_code: Optional[int]) -> bool:
        e = (error or "").lower()
        if status_code == 429:
            return True
        needles = [
            "rate limit",
            "too many requests",
            "quota",
            "overloaded",
            "try again later",
        ]
        return any(n in e for n in needles)

    @staticmethod
    def _looks_gateway_or_transport(status_code: Optional[int], error: str) -> bool:
        if status_code in (502, 503, 504, 520, 521, 522, 523, 524):
            return True
        e = (error or "").lower()
        needles = [
            "connection refused",
            "connection reset",
            "network is unreachable",
            "no route to host",
            "server disconnected",
            "socket hang up",
            "broken pipe",
            "timed out",
            "read timeout",
            "connect timeout",
            "dns",
            "nxdomain",
            "name resolution",
        ]
        return any(n in e for n in needles)

    def normalize_class(
        self,
        cls_hint: RetryClass,
        *,
        error: str,
        stage: str,
        status_code: Optional[int],
    ) -> RetryClass:
        cls = str(cls_hint or "").strip().lower()
        if cls == "permanent":
            return "permanent"
        if cls in ("mem", "mem_heavy"):
            return "mem"

        if cls in ("net", "stall"):
            if self._looks_rate_limited(error, status_code):
                return "stall"
            if self._looks_gateway_or_transport(status_code, error):
                return "net"
            return cls

        if self._looks_rate_limited(error, status_code):
            return "stall"
        if self._looks_gateway_or_transport(status_code, error):
            return "net"
        return "other"

    # ------------------ stall kind inference ------------------

    @staticmethod
    def _infer_stall_kind(
        *,
        stall_kind_hint: Optional[StallKind],
        error: str,
        stage: str,
        status_code: Optional[int],
    ) -> StallKind:
        """
        Keep retry class as 'stall' (interface stable) but derive stall_kind:

          - goto: Playwright/page goto/navigation stalls (harsher)
          - no_yield: scheduler inactivity / no-yield stalls (harsher)
          - rate_limit: 429/quota/overloaded (more patient)
          - unknown: default
        """
        if stall_kind_hint:
            sk = str(stall_kind_hint).strip().lower()
            if sk in ("goto", "no_yield", "rate_limit", "unknown"):
                return sk

        e = (error or "").lower()
        stg = (stage or "").lower()

        if (
            "scheduler" in stg
            or "inactivity" in stg
            or "no_yield" in e
            or "no yield" in e
        ):
            return "no_yield"

        if (
            "goto" in e
            or "page.goto" in e
            or "navigation" in e
            or "waiting for navigation" in e
        ):
            return "goto"
        if "goto" in stg:
            return "goto"

        if status_code == 429:
            return "rate_limit"
        needles = [
            "rate limit",
            "too many requests",
            "quota",
            "overloaded",
            "try again later",
        ]
        if any(n in e for n in needles):
            return "rate_limit"

        return "unknown"

    # ------------------ policy clamp ------------------

    def _clamp_policy(
        self,
        *,
        max_attempts_net: Optional[int],
        max_attempts_stall: Optional[int],
        nxdomain_cutoff: Optional[int],
        override_allow: bool,
    ) -> Tuple[int, int, int]:
        """
        Prevent callers from silently making retries more lenient:
          - allow caller to reduce values (stricter) freely
          - if caller tries to increase above store defaults, clamp unless override_allow=True
        """
        max_net = self._default_max_attempts_net
        max_stall = self._default_max_attempts_stall
        nx_cutoff = self._default_nxdomain_cutoff

        if max_attempts_net is not None:
            v = max(0, int(max_attempts_net))
            max_net = v if override_allow else min(v, self._default_max_attempts_net)

        if max_attempts_stall is not None:
            v = max(0, int(max_attempts_stall))
            max_stall = (
                v if override_allow else min(v, self._default_max_attempts_stall)
            )

        if nxdomain_cutoff is not None:
            v = max(0, int(nxdomain_cutoff))
            nx_cutoff = v if override_allow else min(v, self._default_nxdomain_cutoff)

        return max_net, max_stall, nx_cutoff

    # ------------------ fast queries ------------------

    def pending_ids(self, *, exclude_quarantined: bool = True) -> List[str]:
        with self._lock:
            if not exclude_quarantined:
                return list(self._state.keys())
            if not self._quarantine:
                return list(self._state.keys())
            return [cid for cid in self._state.keys() if cid not in self._quarantine]

    def pending_total(self, *, exclude_quarantined: bool = True) -> int:
        with self._lock:
            if not exclude_quarantined:
                return len(self._state)
            if not self._quarantine:
                return len(self._state)
            return sum(1 for cid in self._state.keys() if cid not in self._quarantine)

    def pending_eligible_total(self, now: Optional[float] = None) -> int:
        if now is None:
            now = _now_ts()
        with self._lock:
            n = 0
            for cid, st in self._state.items():
                if cid in self._quarantine:
                    continue
                if float(st.next_eligible_at) <= float(now):
                    n += 1
            return n

    def is_quarantined(self, company_id: str) -> bool:
        with self._lock:
            return company_id in self._quarantine

    def quarantine_info(self, company_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            info = self._quarantine.get(company_id)
            return dict(info) if isinstance(info, dict) else None

    def get(self, company_id: str) -> Optional[CompanyRetryState]:
        with self._lock:
            s = self._state.get(company_id)
            return CompanyRetryState.from_dict(s.to_dict()) if s is not None else None

    def next_eligible_at(self, company_id: str) -> float:
        with self._lock:
            s = self._state.get(company_id)
            return float(s.next_eligible_at) if s is not None else 0.0

    def min_next_eligible_at(self, *, exclude_quarantined: bool = True) -> float:
        """
        Return the minimum next_eligible_at across pending items.
        0.0 means "eligible now or no pending".
        """
        now = _now_ts()
        with self._lock:
            best: Optional[float] = None
            for cid, st in self._state.items():
                if exclude_quarantined and cid in self._quarantine:
                    continue
                t = float(st.next_eligible_at)
                if t <= now:
                    return 0.0
                if best is None or t < best:
                    best = t
            return float(best) if best is not None else 0.0

    def sleep_hint_sec(
        self, *, now: Optional[float] = None, cap_sec: float = 60.0
    ) -> float:
        """
        Suggested sleep when no eligible items exist. Never negative.
        """
        if now is None:
            now = _now_ts()
        t = self.min_next_eligible_at(exclude_quarantined=True)
        if t <= 0:
            return 0.0
        return max(0.0, min(float(cap_sec), float(t) - float(now)))

    def is_eligible(self, company_id: str, now: Optional[float] = None) -> bool:
        if now is None:
            now = _now_ts()
        with self._lock:
            if company_id in self._quarantine:
                return False
            s = self._state.get(company_id)
            if s is None:
                return True
            return s.next_eligible_at <= now

    def eligible_mask(
        self, company_ids: Sequence[str], now: Optional[float] = None
    ) -> Dict[str, bool]:
        if now is None:
            now = _now_ts()
        out: Dict[str, bool] = {}
        with self._lock:
            q = self._quarantine
            st = self._state
            for cid in company_ids:
                if cid in q:
                    out[cid] = False
                    continue
                s = st.get(cid)
                out[cid] = True if s is None else (s.next_eligible_at <= now)
        return out

    def next_eligible_at_many(self, company_ids: Sequence[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        with self._lock:
            st = self._state
            for cid in company_ids:
                s = st.get(cid)
                out[cid] = float(s.next_eligible_at) if s is not None else 0.0
        return out

    def eligible_ids(self, *, now: Optional[float] = None, limit: int = 0) -> List[str]:
        """
        Return eligible pending IDs (excluding quarantined).
        If limit<=0, return all.
        """
        if now is None:
            now = _now_ts()
        out: List[str] = []
        with self._lock:
            for cid, st in self._state.items():
                if cid in self._quarantine:
                    continue
                if float(st.next_eligible_at) <= float(now):
                    out.append(cid)
                    if limit > 0 and len(out) >= limit:
                        break
        return out

    # ------------------ helper APIs ------------------

    def class_attempts(self, company_id: str) -> Dict[str, Any]:
        with self._lock:
            s = self._state.get(company_id)
            if s is None:
                return {
                    "attempts_total": 0,
                    "net_attempts": 0,
                    "stall_attempts": 0,
                    "mem_attempts": 0,
                    "other_attempts": 0,
                    "last_class": None,
                    "last_stage": None,
                    "last_stall_kind": "unknown",
                    "next_eligible_at": 0.0,
                    "quarantined": company_id in self._quarantine,
                    "last_progress_md_done": 0,
                    "last_seen_md_done": 0,
                }
            return {
                "attempts_total": int(s.attempts),
                "net_attempts": int(s.net_attempts),
                "stall_attempts": int(s.stall_attempts),
                "mem_attempts": int(s.mem_attempts),
                "other_attempts": int(s.other_attempts),
                "last_class": str(s.cls),
                "last_stage": str(s.last_stage),
                "last_stall_kind": str(s.last_stall_kind),
                "next_eligible_at": float(s.next_eligible_at),
                "quarantined": company_id in self._quarantine,
                "last_progress_md_done": int(s.last_progress_md_done),
                "last_seen_md_done": int(s.last_seen_md_done),
            }

    # ------------------ transient APIs ------------------

    def mark_transient(
        self,
        company_id: str,
        *,
        reason: str,
        stage: str = "transient",
        next_eligible_delay_sec: float = 0.0,
        md_done: Optional[int] = None,
        flush: bool = True,
    ) -> None:
        """
        Record a transient event: does NOT increment attempts, does NOT quarantine.
        Sets next_eligible_at to now (or now+delay).
        """
        now = _now_ts()
        msg = (reason or "transient").strip()

        try:
            _append_jsonl_many(
                self.ledger_path,
                [
                    {
                        "ts": now,
                        "company_id": company_id,
                        "event": "transient",
                        "stage": stage,
                        "reason": msg,
                        "md_done": int(md_done) if md_done is not None else None,
                    }
                ],
            )
        except Exception:
            pass

        with self._lock:
            s = self._state.get(company_id)
            if s is None:
                s = CompanyRetryState()
                self._state[company_id] = s

            s.updated_at = now
            s.last_error = msg[:2000]
            s.last_stage = (stage or "")[:128]

            if md_done is not None:
                mdv = int(md_done)
                s.last_seen_md_done = mdv
                if mdv > int(s.last_progress_md_done):
                    s.last_progress_md_done = mdv

            delay = max(0.0, float(next_eligible_delay_sec))
            s.next_eligible_at = float(now + delay)

            self._dirty_state = True

        if flush:
            self.flush()

    def mark_transient_cancel(
        self,
        company_id: str,
        *,
        reason: str = "cancelled by scheduler",
        stage: str = "scheduler_cancel",
        md_done: Optional[int] = None,
        flush: bool = True,
    ) -> None:
        self.mark_transient(
            company_id,
            reason=reason,
            stage=stage,
            next_eligible_delay_sec=0.0,
            md_done=md_done,
            flush=flush,
        )

    def record_scheduler_cancel(
        self,
        company_id: str,
        reason: str,
        *,
        error: str = "",
        stage: str = "scheduler_inactivity",
        status_code: Optional[int] = None,
        stall_kind_hint: Optional[StallKind] = None,
        md_done: Optional[int] = None,
        flush: bool = True,
    ) -> None:
        """
        Scheduler helper: record a scheduler-driven cancellation.

        IMPORTANT: cancellation is TRANSIENT (no attempts / no quarantine).
        """
        _ = status_code, stall_kind_hint  # kept for interface compatibility
        msg = (error or reason or "cancelled by scheduler").strip()
        self.mark_transient_cancel(
            company_id, reason=msg, stage=stage, md_done=md_done, flush=flush
        )

    # ------------------ core transitions ------------------

    def mark_success(
        self, company_id: str, *, stage: str = "", note: str = "", flush: bool = True
    ) -> None:
        now = _now_ts()
        try:
            _append_jsonl_many(
                self.ledger_path,
                [
                    {
                        "ts": now,
                        "company_id": company_id,
                        "event": "success",
                        "stage": stage,
                        "note": note,
                    }
                ],
            )
        except Exception:
            pass

        with self._lock:
            if company_id in self._state:
                del self._state[company_id]
                self._dirty_state = True
            if company_id in self._quarantine:
                del self._quarantine[company_id]
                self._dirty_quarantine = True

        if flush:
            self.flush()
        self._try_clear_terminal_done(company_id)

    def mark_success_many(
        self,
        company_ids: Sequence[str],
        *,
        stage: str = "",
        note: str = "",
        flush: bool = True,
    ) -> int:
        if not company_ids:
            return 0
        now = _now_ts()
        try:
            rows = [
                {
                    "ts": now,
                    "company_id": cid,
                    "event": "success",
                    "stage": stage,
                    "note": note,
                }
                for cid in company_ids
            ]
            _append_jsonl_many(self.ledger_path, rows)
        except Exception:
            pass

        cleared = 0
        with self._lock:
            for cid in company_ids:
                if cid in self._state:
                    del self._state[cid]
                    self._dirty_state = True
                    cleared += 1
                if cid in self._quarantine:
                    del self._quarantine[cid]
                    self._dirty_quarantine = True
        if flush:
            self.flush()
        for cid in company_ids:
            self._try_clear_terminal_done(cid)
        return cleared

    def quarantine_company(
        self,
        company_id: str,
        *,
        reason: str,
        stage: str = "",
        status_code: Optional[int] = None,
        error: str = "",
        details: Optional[Dict[str, Any]] = None,
        flush: bool = True,
    ) -> None:
        """
        Explicit quarantine API (useful from finalize). Also marks crawl_state terminal_done best-effort.
        """
        now = _now_ts()
        meta = {
            "reason": (reason or "quarantined"),
            "ts": now,
            "stall_kind": "unknown",
            "last_stage": (stage or "")[:128],
            "status_code": status_code,
            "last_error": (error or "")[:2000],
            "details": details or {},
        }

        with self._lock:
            s = self._state.get(company_id) or CompanyRetryState()
            s.cls = "permanent"
            s.updated_at = now
            s.last_stage = meta["last_stage"]
            s.last_error = meta["last_error"]
            s.next_eligible_at = _FAR_FUTURE_TS
            self._state[company_id] = s
            self._quarantine[company_id] = meta
            self._dirty_state = True
            self._dirty_quarantine = True

        try:
            _append_jsonl_many(
                self.ledger_path,
                [
                    {
                        "ts": now,
                        "company_id": company_id,
                        "event": "quarantine",
                        "stage": stage,
                        "reason": meta["reason"],
                        "status_code": status_code,
                        "error": meta["last_error"],
                        "details": meta["details"],
                    }
                ],
            )
        except Exception:
            pass

        if flush:
            self.flush()

        done_reason = f"retry_quarantined:{meta['reason']}"
        self._try_mark_terminal_done(
            company_id,
            done_reason=done_reason,
            details={"quarantine": meta},
            last_error=meta["last_error"],
        )

    def unquarantine_company(self, company_id: str, *, flush: bool = True) -> bool:
        """
        Remove quarantine marker and make company eligible immediately (keeps retry counters).
        Returns True if changed.
        """
        changed = False
        with self._lock:
            if company_id in self._quarantine:
                del self._quarantine[company_id]
                self._dirty_quarantine = True
                changed = True
            s = self._state.get(company_id)
            if s is not None and s.next_eligible_at >= _FAR_FUTURE_TS:
                s.next_eligible_at = 0.0
                s.cls = "other" if s.cls == "permanent" else s.cls
                self._dirty_state = True
                changed = True
        if flush and changed:
            self.flush()
        if changed:
            self._try_clear_terminal_done(company_id)
        return changed

    def mark_failure(
        self,
        company_id: str,
        *,
        cls: RetryClass,
        error: str,
        stage: str = "",
        status_code: Optional[int] = None,
        permanent_reason: str = "",
        nxdomain_like: bool = False,
        stall_kind_hint: Optional[StallKind] = None,
        md_done: Optional[int] = None,
        # policy inputs (clamp upward unless override_allow=True)
        max_attempts_net: Optional[int] = None,
        max_attempts_stall: Optional[int] = None,
        nxdomain_cutoff: Optional[int] = None,
        override_allow: bool = False,
        flush: bool = True,
    ) -> Tuple[CompanyRetryState, bool]:
        # IMPORTANT: treat scheduler cancels as transient
        msg = (error or "").strip()
        if _is_transient_message(msg):
            self.mark_transient(
                company_id,
                reason=msg,
                stage=stage or "transient",
                md_done=md_done,
                flush=flush,
            )
            st = self.get(company_id) or CompanyRetryState()
            return st, False

        st, quarantined, ledger_row, terminal_row = self._apply_failure_one(
            company_id=company_id,
            cls=cls,
            error=error,
            stage=stage,
            status_code=status_code,
            permanent_reason=permanent_reason,
            nxdomain_like=nxdomain_like,
            stall_kind_hint=stall_kind_hint,
            md_done=md_done,
            max_attempts_net=max_attempts_net,
            max_attempts_stall=max_attempts_stall,
            nxdomain_cutoff=nxdomain_cutoff,
            override_allow=override_allow,
        )

        try:
            _append_jsonl_many(self.ledger_path, [ledger_row])
        except Exception:
            pass
        if flush:
            self.flush()

        if terminal_row is not None:
            self._try_mark_terminal_done(
                company_id,
                done_reason=terminal_row["done_reason"],
                details=terminal_row["details"],
                last_error=terminal_row["last_error"],
            )
        return st, quarantined

    def mark_failure_many(
        self, events: Sequence[Dict[str, Any]], *, flush: bool = True
    ) -> Tuple[int, int]:
        if not events:
            return 0, 0

        ledger_rows: List[Dict[str, Any]] = []
        terminal_rows: List[Dict[str, Any]] = []
        quarantined_count = 0

        for ev in events:
            cid = str(ev.get("company_id") or "")
            if not cid:
                continue
            cls = str(ev.get("cls") or "other")
            error = str(ev.get("error") or "")
            stage = str(ev.get("stage") or "")
            status_code = ev.get("status_code", None)
            try:
                status_code = int(status_code) if status_code is not None else None
            except Exception:
                status_code = None

            # transient guard
            if _is_transient_message(error):
                self.mark_transient(
                    cid,
                    reason=error,
                    stage=stage or "transient",
                    md_done=int(ev.get("md_done"))
                    if ev.get("md_done") is not None
                    else None,
                    flush=False,
                )
                continue

            permanent_reason = str(ev.get("permanent_reason") or "")
            nxdomain_like = bool(ev.get("nxdomain_like") or False)
            stall_kind_hint = ev.get("stall_kind", None)
            md_done = ev.get("md_done", None)
            try:
                md_done = int(md_done) if md_done is not None else None
            except Exception:
                md_done = None

            max_attempts_net = ev.get("max_attempts_net", None)
            max_attempts_stall = ev.get("max_attempts_stall", None)
            nxdomain_cutoff = ev.get("nxdomain_cutoff", None)
            override_allow = bool(ev.get("override_allow") or False)

            _st, quarantined, ledger_row, terminal_row = self._apply_failure_one(
                company_id=cid,
                cls=cls,
                error=error,
                stage=stage,
                status_code=status_code,
                permanent_reason=permanent_reason,
                nxdomain_like=nxdomain_like,
                stall_kind_hint=stall_kind_hint,
                md_done=md_done,
                max_attempts_net=max_attempts_net,
                max_attempts_stall=max_attempts_stall,
                nxdomain_cutoff=nxdomain_cutoff,
                override_allow=override_allow,
            )
            ledger_rows.append(ledger_row)
            if quarantined:
                quarantined_count += 1
            if terminal_row is not None:
                terminal_rows.append(terminal_row)

        if ledger_rows:
            try:
                _append_jsonl_many(self.ledger_path, ledger_rows)
            except Exception:
                pass
        if flush:
            self.flush()

        for tr in terminal_rows:
            try:
                self._try_mark_terminal_done(
                    tr["company_id"],
                    done_reason=tr["done_reason"],
                    details=tr["details"],
                    last_error=tr["last_error"],
                )
            except Exception:
                continue

        return len(ledger_rows), quarantined_count

    # ------------------ internal: apply one failure under lock ------------------

    def _apply_failure_one(
        self,
        *,
        company_id: str,
        cls: RetryClass,
        error: str,
        stage: str,
        status_code: Optional[int],
        permanent_reason: str,
        nxdomain_like: bool,
        stall_kind_hint: Optional[StallKind],
        md_done: Optional[int],
        max_attempts_net: Optional[int],
        max_attempts_stall: Optional[int],
        nxdomain_cutoff: Optional[int],
        override_allow: bool,
    ) -> Tuple[CompanyRetryState, bool, Dict[str, Any], Optional[Dict[str, Any]]]:
        now = _now_ts()

        max_net, max_stall, nx_cutoff = self._clamp_policy(
            max_attempts_net=max_attempts_net,
            max_attempts_stall=max_attempts_stall,
            nxdomain_cutoff=nxdomain_cutoff,
            override_allow=override_allow,
        )

        cls_canon = self.normalize_class(
            cls, error=error or "", stage=stage or "", status_code=status_code
        )

        stall_kind: StallKind = "unknown"
        if cls_canon == "stall":
            stall_kind = self._infer_stall_kind(
                stall_kind_hint=stall_kind_hint,
                error=error or "",
                stage=stage or "",
                status_code=status_code,
            )

        effective_max_stall = max_stall
        if cls_canon == "stall" and stall_kind in ("goto", "no_yield"):
            effective_max_stall = min(
                effective_max_stall, self._default_max_attempts_stall_fast
            )

        quarantined = False
        quarantine_meta: Dict[str, Any] = {}
        terminal_row: Optional[Dict[str, Any]] = None

        progress_increased = False
        mdv: Optional[int] = None
        if md_done is not None:
            try:
                mdv = int(md_done)
            except Exception:
                mdv = None

        with self._lock:
            prev = self._state.get(company_id) or CompanyRetryState()
            s = CompanyRetryState.from_dict(prev.to_dict())

            # progress markers update BEFORE quarantine decision
            if mdv is not None:
                s.last_seen_md_done = mdv
                if mdv > int(s.last_progress_md_done):
                    s.last_progress_md_done = mdv
                    progress_increased = True

            s.cls = cls_canon
            s.attempts = int(s.attempts) + 1
            s.updated_at = now
            s.last_error = (error or "")[:2000]
            s.last_stage = (stage or "")[:128]
            if cls_canon == "stall":
                s.last_stall_kind = stall_kind

            # per-class attempts
            class_attempts = s.attempts
            if cls_canon == "net":
                s.net_attempts = int(s.net_attempts) + 1
                class_attempts = s.net_attempts
            elif cls_canon == "stall":
                s.stall_attempts = int(s.stall_attempts) + 1
                class_attempts = s.stall_attempts
            elif cls_canon == "mem":
                s.mem_attempts = int(s.mem_attempts) + 1
                class_attempts = s.mem_attempts
                s.mem_hits = int(s.mem_hits) + 1
            else:
                s.other_attempts = int(s.other_attempts) + 1
                class_attempts = s.other_attempts

            # If progress increased, do NOT allow stall quarantine pressure to accumulate aggressively.
            # Keep it minimal: reduce stall_attempts by 1 (never below 0) and skip stall quarantine for this event.
            if progress_increased and cls_canon == "stall":
                s.stall_attempts = max(0, int(s.stall_attempts) - 1)
                class_attempts = s.stall_attempts

            # quarantine decisions
            if cls_canon == "permanent":
                quarantined = True
                quarantine_meta = {
                    "reason": permanent_reason or "permanent",
                    "ts": now,
                    "stall_kind": stall_kind if cls_canon == "stall" else "unknown",
                    "last_stage": s.last_stage,
                    "status_code": status_code,
                    "last_error": s.last_error,
                    "counters": {
                        "attempts_total": s.attempts,
                        "net_attempts": s.net_attempts,
                        "stall_attempts": s.stall_attempts,
                        "mem_attempts": s.mem_attempts,
                        "other_attempts": s.other_attempts,
                        "mem_hits": s.mem_hits,
                    },
                    "progress": {
                        "last_progress_md_done": int(s.last_progress_md_done),
                        "last_seen_md_done": int(s.last_seen_md_done),
                    },
                }
            else:
                inferred_nx = self.classify_unreachable_error(error or "")
                nx = bool(nxdomain_like or inferred_nx)

                if (not quarantined) and nx and s.net_attempts >= nx_cutoff:
                    quarantined = True
                    quarantine_meta = {
                        "reason": "nxdomain_like",
                        "ts": now,
                        "stall_kind": stall_kind if cls_canon == "stall" else "unknown",
                        "last_stage": s.last_stage,
                        "status_code": status_code,
                        "last_error": s.last_error,
                        "policy": {"nxdomain_cutoff": nx_cutoff},
                        "counters": {
                            "attempts_total": s.attempts,
                            "net_attempts": s.net_attempts,
                            "stall_attempts": s.stall_attempts,
                            "mem_attempts": s.mem_attempts,
                            "other_attempts": s.other_attempts,
                            "mem_hits": s.mem_hits,
                        },
                        "progress": {
                            "last_progress_md_done": int(s.last_progress_md_done),
                            "last_seen_md_done": int(s.last_seen_md_done),
                        },
                    }

                if (
                    (not quarantined)
                    and cls_canon == "net"
                    and s.net_attempts >= max_net
                ):
                    quarantined = True
                    quarantine_meta = {
                        "reason": "max_attempts_net",
                        "ts": now,
                        "stall_kind": "unknown",
                        "last_stage": s.last_stage,
                        "status_code": status_code,
                        "last_error": s.last_error,
                        "policy": {"max_attempts_net": max_net},
                        "counters": {
                            "attempts_total": s.attempts,
                            "net_attempts": s.net_attempts,
                            "stall_attempts": s.stall_attempts,
                            "mem_attempts": s.mem_attempts,
                            "other_attempts": s.other_attempts,
                            "mem_hits": s.mem_hits,
                        },
                        "progress": {
                            "last_progress_md_done": int(s.last_progress_md_done),
                            "last_seen_md_done": int(s.last_seen_md_done),
                        },
                    }

                # IMPORTANT: stall quarantine requires "no progress" for this event
                if (
                    (not quarantined)
                    and cls_canon == "stall"
                    and (not progress_increased)
                    and s.stall_attempts >= effective_max_stall
                ):
                    quarantined = True
                    quarantine_meta = {
                        "reason": "max_attempts_stall",
                        "ts": now,
                        "stall_kind": stall_kind,
                        "last_stage": s.last_stage,
                        "status_code": status_code,
                        "last_error": s.last_error,
                        "policy": {"max_attempts_stall": effective_max_stall},
                        "counters": {
                            "attempts_total": s.attempts,
                            "net_attempts": s.net_attempts,
                            "stall_attempts": s.stall_attempts,
                            "mem_attempts": s.mem_attempts,
                            "other_attempts": s.other_attempts,
                            "mem_hits": s.mem_hits,
                        },
                        "progress": {
                            "last_progress_md_done": int(s.last_progress_md_done),
                            "last_seen_md_done": int(s.last_seen_md_done),
                        },
                    }

            if quarantined:
                s.cls = "permanent"
                s.next_eligible_at = _FAR_FUTURE_TS
                self._quarantine[company_id] = quarantine_meta
                self._dirty_quarantine = True
            else:
                base = _backoff_seconds(
                    class_attempts, kind=cls_canon, stall_kind=stall_kind
                )
                s.next_eligible_at = now + _jitter(base, frac=DEFAULT_JITTER_FRAC)

            self._state[company_id] = s
            self._dirty_state = True

        ledger_row: Dict[str, Any] = {
            "ts": now,
            "company_id": company_id,
            "event": "failure",
            "class_hint": str(cls or ""),
            "class": cls_canon,
            "stall_kind": stall_kind if cls_canon == "stall" else None,
            "attempts_total": s.attempts,
            "net_attempts": s.net_attempts,
            "stall_attempts": s.stall_attempts,
            "mem_attempts": s.mem_attempts,
            "other_attempts": s.other_attempts,
            "next_eligible_at": s.next_eligible_at,
            "stage": stage,
            "status_code": status_code,
            "error": (error or "")[:4000],
            "quarantined": quarantined,
            "permanent_reason": permanent_reason,
            "mem_hits": s.mem_hits,
            "progress": {
                "md_done": mdv,
                "progress_increased": bool(progress_increased),
                "last_progress_md_done": int(s.last_progress_md_done),
                "last_seen_md_done": int(s.last_seen_md_done),
            },
            "policy": {
                "effective": {
                    "max_attempts_net": max_net,
                    "max_attempts_stall": (
                        effective_max_stall if cls_canon == "stall" else max_stall
                    ),
                    "nxdomain_cutoff": nx_cutoff,
                },
                "store_defaults": {
                    "DEFAULT_MAX_ATTEMPTS_NET": self._default_max_attempts_net,
                    "DEFAULT_MAX_ATTEMPTS_STALL": self._default_max_attempts_stall,
                    "DEFAULT_MAX_ATTEMPTS_STALL_FAST": self._default_max_attempts_stall_fast,
                    "DEFAULT_NXDOMAIN_CUTOFF": self._default_nxdomain_cutoff,
                },
                "override_allow": bool(override_allow),
                "requested": {
                    "max_attempts_net": max_attempts_net,
                    "max_attempts_stall": max_attempts_stall,
                    "nxdomain_cutoff": nxdomain_cutoff,
                },
            },
            "quarantine_meta": quarantine_meta if quarantined else None,
        }

        if quarantined:
            done_reason = (
                f"retry_quarantined:{(quarantine_meta.get('reason') or 'unknown')}"
            )
            details = {
                "retry_class_hint": str(cls or ""),
                "retry_class": cls_canon,
                "stall_kind": stall_kind if cls_canon == "stall" else None,
                "attempts_total": s.attempts,
                "net_attempts": s.net_attempts,
                "stall_attempts": s.stall_attempts,
                "mem_attempts": s.mem_attempts,
                "other_attempts": s.other_attempts,
                "stage": stage,
                "status_code": status_code,
                "quarantine_reason": quarantine_meta.get("reason"),
                "permanent_reason": permanent_reason,
                "next_eligible_at": s.next_eligible_at,
                "policy_effective": {
                    "max_attempts_net": max_net,
                    "max_attempts_stall": (
                        effective_max_stall if cls_canon == "stall" else max_stall
                    ),
                    "nxdomain_cutoff": nx_cutoff,
                },
                "progress": {
                    "md_done": mdv,
                    "last_progress_md_done": int(s.last_progress_md_done),
                    "last_seen_md_done": int(s.last_seen_md_done),
                },
                "quarantine_meta": quarantine_meta,
            }
            terminal_row = {
                "company_id": company_id,
                "done_reason": done_reason,
                "details": details,
                "last_error": s.last_error,
            }

        return s, quarantined, ledger_row, terminal_row
