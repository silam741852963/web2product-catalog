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

# Classes are intentionally coarse and stable
RetryClass = str  # "net" | "stall" | "mem" | "mem_heavy" | "permanent"


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


def _jitter(seconds: float, frac: float = 0.20) -> float:
    """
    Add +/- frac jitter (default +/-20%).
    """
    if seconds <= 0:
        return 0.0
    r = random.random()  # 0..1
    delta = (r * 2.0 - 1.0) * frac
    return max(0.0, seconds * (1.0 + delta))


def _backoff_schedule_seconds(
    attempts: int,
    *,
    kind: RetryClass,
) -> float:
    """
    Conservative defaults that behave well on droplets.
    """
    a = max(1, int(attempts))

    if kind == "net":
        # 1m, 5m, 20m, 1h, 3h, 8h
        schedule = [60, 300, 1200, 3600, 10800, 28800]
    elif kind == "stall":
        # 10m, 30m, 2h, 6h
        schedule = [600, 1800, 7200, 21600]
    elif kind in ("mem", "mem_heavy"):
        # 15m, 1h, 6h, 24h
        schedule = [900, 3600, 21600, 86400]
    elif kind == "permanent":
        return 10**9  # effectively never (quarantine handles this)
    else:
        # unknown class: treat as net
        schedule = [60, 300, 1200, 3600, 10800, 28800]

    idx = min(a, len(schedule)) - 1
    return float(schedule[idx])


# --------------------------------------------------------------------------------------
# Datamodel
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class CompanyRetryState:
    cls: RetryClass = "net"
    attempts: int = 0  # total attempts (all classes)
    next_eligible_at: float = 0.0  # epoch seconds; 0 means eligible now
    updated_at: float = 0.0
    last_error: str = ""
    last_stage: str = ""
    # per-class counters (so "reasonable retries" can be class-specific)
    net_attempts: int = 0
    stall_attempts: int = 0
    mem_attempts: int = 0
    # mem-specific escalation signal (kept for backward compatibility)
    mem_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CompanyRetryState":
        return CompanyRetryState(
            cls=str(d.get("cls") or "net"),
            attempts=int(d.get("attempts") or 0),
            next_eligible_at=float(d.get("next_eligible_at") or 0.0),
            updated_at=float(d.get("updated_at") or 0.0),
            last_error=str(d.get("last_error") or ""),
            last_stage=str(d.get("last_stage") or ""),
            net_attempts=int(d.get("net_attempts") or 0),
            stall_attempts=int(d.get("stall_attempts") or 0),
            mem_attempts=int(d.get("mem_attempts") or 0),
            mem_hits=int(d.get("mem_hits") or 0),
        )


# --------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------


class RetryStateStore:
    """
    A small, bounded company-level retry state store:
      - retry_state.json : compact snapshot (atomic rewrite)
      - failure_ledger.jsonl : append-only audit trail
      - quarantine.json : permanently failed companies (small set)

    NEW:
      - When a company is quarantined, we best-effort mark it as terminal_done
        in crawl_state.sqlite3 so it stays out of the pending queue.
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

    # ------------------ persistence ------------------

    def _load(self) -> None:
        with self._lock:
            # retry_state.json
            raw = _json_load_nocache(self.state_path)
            if isinstance(raw, dict):
                st: Dict[str, CompanyRetryState] = {}
                for cid, v in raw.items():
                    if isinstance(v, dict):
                        st[str(cid)] = CompanyRetryState.from_dict(v)
                self._state = st
            else:
                self._state = {}

            # quarantine.json
            qraw = _json_load_nocache(self.quarantine_path)
            if isinstance(qraw, dict):
                self._quarantine = {
                    str(k): (v if isinstance(v, dict) else {}) for k, v in qraw.items()
                }
            else:
                self._quarantine = {}

    def flush(self) -> None:
        """
        Persist snapshot files. JSONL ledger is append-only and doesn't need flushing.
        """
        with self._lock:
            snapshot = {cid: s.to_dict() for cid, s in self._state.items()}
            quarantine = dict(self._quarantine)

        _atomic_write_json_compact(self.state_path, snapshot)
        _atomic_write_json_compact(self.quarantine_path, quarantine)

    # ------------------ crawl_state integration ------------------

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
        Must NEVER raise (retry bookkeeping must be resilient).
        """
        try:
            # Lazy import avoids cycles / heavy imports at module load time
            from crawl_state import get_crawl_state  # type: ignore

            cs = get_crawl_state()
            cs.mark_company_terminal_sync(
                company_id,
                reason=done_reason,
                details=details,
                last_error=last_error,
            )
        except Exception:
            return

    def _try_clear_terminal_done(self, company_id: str) -> None:
        """
        Best-effort: clear terminal marker if someone manually re-runs / overrides.
        """
        try:
            from crawl_state import get_crawl_state  # type: ignore

            cs = get_crawl_state()
            cs.clear_company_terminal_sync(company_id, keep_status=False)
        except Exception:
            return

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
        """
        Company succeeded: clear retry state and remove from quarantine (if any).
        Also best-effort clear terminal_done marker in crawl_state.
        """
        now = _now_ts()
        event = {
            "ts": now,
            "company_id": company_id,
            "event": "success",
            "stage": stage,
            "note": note,
        }
        _append_jsonl(self.ledger_path, event)

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
        # policy knobs (reasonable retry limits)
        max_attempts_net: int = 6,
        max_attempts_stall: int = 4,
        max_attempts_mem: int = 6,
        max_attempts_mem_heavy: int = 4,
        nxdomain_like: bool = False,
    ) -> Tuple[CompanyRetryState, bool]:
        """
        Record a company-level failure and compute next eligibility with backoff.

        Returns: (new_state, quarantined_bool)

        NEW:
          - When quarantined, mark terminal_done in crawl_state with clear reason/details.
        """
        now = _now_ts()
        cls = str(cls or "net")

        quarantined = False
        quarantine_meta: Dict[str, Any] = {}

        with self._lock:
            prev = self._state.get(company_id) or CompanyRetryState()
            s = CompanyRetryState.from_dict(prev.to_dict())

            # Update total counters
            s.cls = cls
            s.attempts = int(s.attempts) + 1
            s.updated_at = now
            s.last_error = (error or "")[:2000]
            s.last_stage = (stage or "")[:128]

            # Update per-class counters and compute class_attempts (for backoff and limits)
            class_attempts = s.attempts
            if cls == "net":
                s.net_attempts = int(s.net_attempts) + 1
                class_attempts = s.net_attempts
            elif cls == "stall":
                s.stall_attempts = int(s.stall_attempts) + 1
                class_attempts = s.stall_attempts
            elif cls in ("mem", "mem_heavy"):
                s.mem_attempts = int(s.mem_attempts) + 1
                class_attempts = s.mem_attempts
                s.mem_hits = int(s.mem_hits) + 1
                # escalate after repeated hits
                if s.mem_hits >= 3:
                    s.cls = "mem_heavy"
                    cls = "mem_heavy"

            # Decide quarantine (terminal)
            if cls == "permanent":
                quarantined = True
                quarantine_meta = {
                    "reason": permanent_reason or "permanent",
                    "ts": now,
                    "stage": stage,
                    "last_error": s.last_error,
                    "status_code": status_code,
                }
            else:
                # NXDOMAIN-like errors should be cut off quickly (net-like)
                if (not quarantined) and nxdomain_like and s.net_attempts >= 2:
                    quarantined = True
                    quarantine_meta = {
                        "reason": "nxdomain_like",
                        "ts": now,
                        "stage": stage,
                        "last_error": s.last_error,
                        "status_code": status_code,
                    }

                # class-specific "reasonable" caps
                if (
                    (not quarantined)
                    and cls == "net"
                    and s.net_attempts >= int(max_attempts_net)
                ):
                    quarantined = True
                    quarantine_meta = {
                        "reason": "max_attempts_net",
                        "ts": now,
                        "stage": stage,
                        "last_error": s.last_error,
                        "status_code": status_code,
                    }

                if (
                    (not quarantined)
                    and cls == "stall"
                    and s.stall_attempts >= int(max_attempts_stall)
                ):
                    quarantined = True
                    quarantine_meta = {
                        "reason": "max_attempts_stall",
                        "ts": now,
                        "stage": stage,
                        "last_error": s.last_error,
                        "status_code": status_code,
                    }

                if (not quarantined) and cls in ("mem", "mem_heavy"):
                    cap = (
                        int(max_attempts_mem_heavy)
                        if cls == "mem_heavy"
                        else int(max_attempts_mem)
                    )
                    if s.mem_attempts >= cap:
                        quarantined = True
                        quarantine_meta = {
                            "reason": "max_attempts_mem_heavy"
                            if cls == "mem_heavy"
                            else "max_attempts_mem",
                            "ts": now,
                            "stage": stage,
                            "last_error": s.last_error,
                            "status_code": status_code,
                        }

            if quarantined:
                s.cls = "permanent"
                s.next_eligible_at = 10**9  # effectively never
                self._quarantine[company_id] = quarantine_meta
            else:
                base = _backoff_schedule_seconds(class_attempts, kind=cls)
                s.next_eligible_at = now + _jitter(base, frac=0.20)

            self._state[company_id] = s

        # ledger entry outside lock
        event = {
            "ts": now,
            "company_id": company_id,
            "event": "failure",
            "class": cls,
            "attempts_total": s.attempts,
            "net_attempts": s.net_attempts,
            "stall_attempts": s.stall_attempts,
            "mem_attempts": s.mem_attempts,
            "next_eligible_at": s.next_eligible_at,
            "stage": stage,
            "status_code": status_code,
            "error": (error or "")[:4000],
            "quarantined": quarantined,
            "permanent_reason": permanent_reason,
            "mem_hits": s.mem_hits,
            "quarantine_meta": quarantine_meta if quarantined else None,
        }
        _append_jsonl(self.ledger_path, event)

        self.flush()

        # If quarantined => mark terminal_done in CrawlState so it won't be queued again
        if quarantined:
            done_reason = (
                f"retry_quarantined:{(quarantine_meta.get('reason') or 'unknown')}"
            )
            details = {
                "retry_class": cls,
                "attempts_total": s.attempts,
                "net_attempts": s.net_attempts,
                "stall_attempts": s.stall_attempts,
                "mem_attempts": s.mem_attempts,
                "stage": stage,
                "status_code": status_code,
                "quarantine_reason": quarantine_meta.get("reason"),
                "permanent_reason": permanent_reason,
                "next_eligible_at": s.next_eligible_at,
            }
            self._try_mark_terminal_done(
                company_id,
                done_reason=done_reason,
                details=details,
                last_error=s.last_error,
            )

        return s, quarantined

    # ------------------ optional helpers ------------------

    @staticmethod
    def classify_unreachable_error(error: str) -> bool:
        """
        Conservative "NXDOMAIN-like" detector.
        """
        if not error:
            return False
        e = error.lower()
        needles = [
            "name or service not known",
            "nxdomain",
            "temporary failure in name resolution",  # can be transient, but retry policy handles repeats
            "nodename nor servname provided",
            "no address associated with hostname",
        ]
        return any(n in e for n in needles)

    @staticmethod
    def classify_tls_error(error: str) -> bool:
        if not error:
            return False
        e = error.lower()
        needles = [
            "ssl",
            "tls",
            "certificate verify failed",
            "handshake",
        ]
        return any(n in e for n in needles)
