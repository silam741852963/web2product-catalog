from __future__ import annotations

import errno
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from extensions import output_paths

# --------------------------------------------------------------------------------------
# RetryState: compact company-level failure state + append-only failure ledger
# --------------------------------------------------------------------------------------

RetryClass = str  # "net" | "stall" | "mem" | "other" | "permanent"

DEFAULT_MAX_ATTEMPTS_NET = 6
DEFAULT_MAX_ATTEMPTS_STALL = 8
DEFAULT_NXDOMAIN_CUTOFF = 2
DEFAULT_JITTER_FRAC = 0.20


def _output_root() -> Path:
    """
    output_paths.OUTPUT_ROOT may be mutated at runtime; resolve dynamically.
    """
    try:
        root = getattr(output_paths, "OUTPUT_ROOT", None)
        if root is None:
            return Path("outputs").resolve()
        return Path(root).resolve()
    except Exception:
        return Path("outputs").resolve()


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
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception:
            return None

    return _retry_emfile(_read)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    Append-only JSONL writer. Best-effort fsync to avoid loss on abrupt exit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"

    def _write() -> None:
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(line)
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


def _backoff_schedule_seconds(attempts: int, *, kind: RetryClass) -> float:
    a = max(1, int(attempts))

    if kind == "net":
        schedule = [60, 300, 1200, 3600, 10800, 28800]
    elif kind == "stall":
        schedule = [600, 1800, 7200, 21600, 43200, 86400]
    elif kind == "mem":
        schedule = [900, 3600, 21600, 86400]
    elif kind == "other":
        schedule = [300, 1200, 7200, 43200, 86400]
    elif kind == "permanent":
        return 10**9
    else:
        schedule = [300, 1200, 7200, 43200, 86400]

    idx = min(a, len(schedule)) - 1
    return float(schedule[idx])


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

    net_attempts: int = 0
    stall_attempts: int = 0
    mem_attempts: int = 0
    other_attempts: int = 0

    mem_hits: int = 0

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
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            base_dir = _output_root() / "_retry"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.base_dir / "retry_state.json"
        self.ledger_path = self.base_dir / "failure_ledger.jsonl"
        self.quarantine_path = self.base_dir / "quarantine.json"

        self._lock = threading.Lock()
        self._state: Dict[str, CompanyRetryState] = {}
        self._quarantine: Dict[str, Dict[str, Any]] = {}

        self._load()
        # Ensure files exist even if empty (reduces “missing file” ambiguity)
        self.flush()

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

    def flush(self) -> None:
        with self._lock:
            snapshot = {cid: s.to_dict() for cid, s in self._state.items()}
            quarantine = dict(self._quarantine)

        _atomic_write_json_compact(self.state_path, snapshot)
        _atomic_write_json_compact(self.quarantine_path, quarantine)

    # ------------------ crawl_state integration (best-effort) ------------------

    def _crawl_state_db_path(self) -> Path:
        return _output_root() / "crawl_state.sqlite3"

    def _try_mark_terminal_done(
        self,
        company_id: str,
        *,
        done_reason: str,
        details: Dict[str, Any],
        last_error: str,
    ) -> None:
        """
        Best-effort integration: mark terminal_done in crawl_state.sqlite3.
        Must NEVER raise.
        """
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
        """
        Best-effort: clear terminal marker if someone later succeeds or re-runs.
        """
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

    # ------------------ queries ------------------

    def is_quarantined(self, company_id: str) -> bool:
        with self._lock:
            return company_id in self._quarantine

    def get(self, company_id: str) -> Optional[CompanyRetryState]:
        with self._lock:
            s = self._state.get(company_id)
            return CompanyRetryState.from_dict(s.to_dict()) if s is not None else None

    def next_eligible_at(self, company_id: str) -> float:
        with self._lock:
            s = self._state.get(company_id)
            return float(s.next_eligible_at) if s is not None else 0.0

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

    # ------------------ core transitions ------------------

    def mark_success(self, company_id: str, *, stage: str = "", note: str = "") -> None:
        now = _now_ts()
        _append_jsonl(
            self.ledger_path,
            {
                "ts": now,
                "company_id": company_id,
                "event": "success",
                "stage": stage,
                "note": note,
            },
        )

        with self._lock:
            if company_id in self._state:
                del self._state[company_id]
            if company_id in self._quarantine:
                del self._quarantine[company_id]

        self.flush()
        self._try_clear_terminal_done(company_id)

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
        max_attempts_net: Optional[int] = None,
        max_attempts_stall: Optional[int] = None,
        nxdomain_cutoff: Optional[int] = None,
    ) -> Tuple[CompanyRetryState, bool]:
        now = _now_ts()

        max_net = (
            int(max_attempts_net)
            if max_attempts_net is not None
            else DEFAULT_MAX_ATTEMPTS_NET
        )
        max_stall = (
            int(max_attempts_stall)
            if max_attempts_stall is not None
            else DEFAULT_MAX_ATTEMPTS_STALL
        )
        nx_cutoff = (
            int(nxdomain_cutoff)
            if nxdomain_cutoff is not None
            else DEFAULT_NXDOMAIN_CUTOFF
        )

        cls_canon = self.normalize_class(
            cls,
            error=error or "",
            stage=stage or "",
            status_code=status_code,
        )

        quarantined = False
        quarantine_meta: Dict[str, Any] = {}

        with self._lock:
            prev = self._state.get(company_id) or CompanyRetryState()
            s = CompanyRetryState.from_dict(prev.to_dict())

            s.cls = cls_canon
            s.attempts = int(s.attempts) + 1
            s.updated_at = now
            s.last_error = (error or "")[:2000]
            s.last_stage = (stage or "")[:128]

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

            if cls_canon == "permanent":
                quarantined = True
                quarantine_meta = {
                    "reason": permanent_reason or "permanent",
                    "ts": now,
                    "stage": stage,
                    "last_error": s.last_error,
                    "status_code": status_code,
                }
            else:
                inferred_nx = self.classify_unreachable_error(error or "")
                nx = bool(nxdomain_like or inferred_nx)

                if (not quarantined) and nx and s.net_attempts >= nx_cutoff:
                    quarantined = True
                    quarantine_meta = {
                        "reason": "nxdomain_like",
                        "ts": now,
                        "stage": stage,
                        "last_error": s.last_error,
                        "status_code": status_code,
                        "cutoff": nx_cutoff,
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
                        "stage": stage,
                        "last_error": s.last_error,
                        "status_code": status_code,
                        "max": max_net,
                    }

                if (
                    (not quarantined)
                    and cls_canon == "stall"
                    and s.stall_attempts >= max_stall
                ):
                    quarantined = True
                    quarantine_meta = {
                        "reason": "max_attempts_stall",
                        "ts": now,
                        "stage": stage,
                        "last_error": s.last_error,
                        "status_code": status_code,
                        "max": max_stall,
                    }

            if quarantined:
                s.cls = "permanent"
                s.next_eligible_at = 10**9
                self._quarantine[company_id] = quarantine_meta
            else:
                base = _backoff_schedule_seconds(class_attempts, kind=cls_canon)
                s.next_eligible_at = now + _jitter(base, frac=DEFAULT_JITTER_FRAC)

            self._state[company_id] = s

        _append_jsonl(
            self.ledger_path,
            {
                "ts": now,
                "company_id": company_id,
                "event": "failure",
                "class_hint": str(cls or ""),
                "class": cls_canon,
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
                "policy": {
                    "max_attempts_net": max_net,
                    "max_attempts_stall": max_stall,
                    "nxdomain_cutoff": nx_cutoff,
                    "defaults": {
                        "DEFAULT_MAX_ATTEMPTS_NET": DEFAULT_MAX_ATTEMPTS_NET,
                        "DEFAULT_MAX_ATTEMPTS_STALL": DEFAULT_MAX_ATTEMPTS_STALL,
                        "DEFAULT_NXDOMAIN_CUTOFF": DEFAULT_NXDOMAIN_CUTOFF,
                    },
                },
                "quarantine_meta": quarantine_meta if quarantined else None,
            },
        )

        self.flush()

        if quarantined:
            # Avoid repeated DB writes if already quarantined
            already_quarantined = self.is_quarantined(company_id)
            if already_quarantined:
                done_reason = (
                    f"retry_quarantined:{(quarantine_meta.get('reason') or 'unknown')}"
                )
                details = {
                    "retry_class_hint": str(cls or ""),
                    "retry_class": cls_canon,
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
                    "policy": {
                        "max_attempts_net": max_net,
                        "max_attempts_stall": max_stall,
                        "nxdomain_cutoff": nx_cutoff,
                    },
                }
                self._try_mark_terminal_done(
                    company_id,
                    done_reason=done_reason,
                    details=details,
                    last_error=s.last_error,
                )

        return s, quarantined
