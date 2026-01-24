from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from configs.models import Company
from extensions.crawl.state import get_crawl_state
from extensions.io import output_paths

logger = logging.getLogger("extensions.retry")

_RETRY_DEBUG = os.getenv("RETRY_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _dbg(msg: str, *args: Any) -> None:
    if _RETRY_DEBUG:
        logger.info(msg, *args)
    else:
        logger.debug(msg, *args)


_ERROR_CUT_MARKERS = (
    "\n\nCall log:",
    "\nCall log:",
    "\n\nCode context:",
    "\nCode context:",
    "\n\nStack trace:",
    "\nStack trace:",
)

_SIG_WS_PAT = re.compile(r"\s+")
_SIG_DIGITS_PAT = re.compile(r"\d+")
_SIG_HEXLIKE_PAT = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)


def _strip_verbose_blocks(text: str) -> str:
    t = text or ""
    if not t:
        return ""
    cut = len(t)
    for m in _ERROR_CUT_MARKERS:
        i = t.find(m)
        if i != -1:
            cut = min(cut, i)
    return t[:cut]


def _one_line(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return _SIG_WS_PAT.sub(" ", t)


def _compact_error_text(text: str, *, max_chars: int = 600) -> str:
    t = _one_line(_strip_verbose_blocks(text or ""))
    if len(t) > max_chars:
        t = t[: max_chars - 3] + "..."
    return t


def _exc_to_compact_string(exc: BaseException, *, max_chars: int = 600) -> str:
    return _compact_error_text(f"{type(exc).__name__}: {exc}", max_chars=max_chars)


_GOTO_TIMEOUT_PAT = re.compile(
    r"(acs-goto|page\.goto:\s*timeout|timeout\s*\d+\s*ms\s*exceeded|timeout\s*\d+ms\s*exceeded)",
    re.IGNORECASE,
)

DEFAULT_SAME_ERROR_STREAK_QUARANTINE_GOTO = 2
DEFAULT_SAME_ERROR_STREAK_QUARANTINE_GENERIC = 4

_ALLOWED_RETRY_CLASSES = {"net", "stall", "mem", "other", "permanent"}
_ALLOWED_STALL_KINDS = {"goto", "no_yield", "rate_limit", "unknown"}


class CriticalMemoryPressure(RuntimeError):
    def __init__(self, message: str, severity: str = "emergency") -> None:
        super().__init__(message)
        self.severity = severity


class CrawlerFatalError(RuntimeError):
    pass


class CrawlerTimeoutError(TimeoutError):
    def __init__(self, message: str, *, stage: str, company_id: str, url: str) -> None:
        super().__init__(_compact_error_text(message, max_chars=800))
        self.stage = stage
        self.company_id = company_id
        self.url = url


def is_goto_timeout_error(exc: BaseException) -> bool:
    return (
        _GOTO_TIMEOUT_PAT.search(_exc_to_compact_string(exc, max_chars=1200))
        is not None
    )


def is_playwright_driver_disconnect(exc: BaseException) -> bool:
    low = _exc_to_compact_string(exc, max_chars=1200).lower()
    return any(
        n in low
        for n in (
            "connection closed while reading from the driver",
            "browser has been closed",
            "target page, context or browser has been closed",
            "playwright connection closed",
            "pipe closed by peer",
        )
    )


@dataclass(frozen=True, slots=True)
class RetryEvent:
    cls: str
    stage: str
    error: str
    nxdomain_like: bool = False
    status_code: Optional[int] = None
    stall_kind: Optional[str] = None


def _extract_status_code(exc: BaseException) -> Optional[int]:
    sc = getattr(exc, "status_code", None)
    if sc is None:
        return None
    return int(sc)


def classify_failure(exc: BaseException, *, stage: str) -> RetryEvent:
    status_code = _extract_status_code(exc)
    compact_err = _compact_error_text(str(exc), max_chars=800)

    if is_goto_timeout_error(exc):
        return RetryEvent(
            cls="stall",
            stage=stage,
            error=compact_err,
            status_code=status_code,
            stall_kind="goto",
        )

    if isinstance(exc, CriticalMemoryPressure):
        return RetryEvent(
            cls="mem", stage=stage, error=compact_err, status_code=status_code
        )

    if isinstance(exc, (asyncio.TimeoutError, CrawlerTimeoutError)):
        return RetryEvent(
            cls="stall", stage=stage, error=compact_err, status_code=status_code
        )

    if isinstance(exc, CrawlerFatalError) or is_playwright_driver_disconnect(exc):
        nx = RetryStateStore.classify_unreachable_error(compact_err)
        return RetryEvent(
            cls="net",
            stage=stage,
            error=compact_err,
            nxdomain_like=nx,
            status_code=status_code,
        )

    nxdomain_like = RetryStateStore.classify_unreachable_error(compact_err)
    if nxdomain_like:
        return RetryEvent(
            cls="net",
            stage=stage,
            error=compact_err,
            nxdomain_like=True,
            status_code=status_code,
        )

    return RetryEvent(
        cls="stall", stage=stage, error=compact_err, status_code=status_code
    )


def should_fail_fast_on_goto(exc: BaseException, *, stage: str) -> bool:
    return stage in {"goto", "arun_init", "direct_fetch"} and is_goto_timeout_error(exc)


RetryClass = str
StallKind = str

DEFAULT_MAX_ATTEMPTS_NET = 3
DEFAULT_MAX_ATTEMPTS_STALL = 4
DEFAULT_MAX_ATTEMPTS_STALL_FAST = 2
DEFAULT_NXDOMAIN_CUTOFF = 1

DEFAULT_JITTER_FRAC = 0.15
DEFAULT_FLUSH_MIN_INTERVAL_SEC = 1.0

_NET_SCHEDULE = [20, 60, 300]
_STALL_SCHEDULE = [60, 300, 1200, 3600]
_STALL_FAST_SCHEDULE = [30, 120]
_MEM_SCHEDULE = [300, 1800, 7200, 21600, 86400]
_OTHER_SCHEDULE = [60, 300, 1800, 7200, 21600, 86400]


def _output_root() -> Path:
    root = output_paths.get_output_root()
    if not root:
        raise RuntimeError("output_paths.get_output_root() returned empty.")
    return Path(root).resolve()


def _retry_base_dir() -> Path:
    return (_output_root() / "_retry").resolve()


def _retry_emfile(
    fn: Callable[[], Any], attempts: int = 6, base_delay: float = 0.12
) -> Any:
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            last = e
            if e.errno in (errno.EMFILE, errno.ENFILE) or "Too many open files" in str(
                e
            ):
                time.sleep(base_delay * (2**i))
                continue
            raise
    raise OSError("EMFILE/ENFILE retry budget exhausted") from last


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    tmp = Path(f"{str(path)}.tmp.{stamp}")

    def _write() -> None:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    _retry_emfile(_write)


def _atomic_write_json_compact(path: Path, obj: Any) -> None:
    _atomic_write_text(
        path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")), "utf-8"
    )


def _json_load_optional(path: Path) -> Any:
    if not path.exists():
        return None

    def _read() -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    return _retry_emfile(_read)


def _append_jsonl_many(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
    )

    def _write() -> None:
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(lines)
            f.flush()
            os.fsync(f.fileno())

    _retry_emfile(_write)


async def _atomic_write_json_compact_async(path: Path, obj: Any) -> None:
    await asyncio.to_thread(_atomic_write_json_compact, path, obj)


async def _append_jsonl_many_async(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    await asyncio.to_thread(_append_jsonl_many, path, rows)


def _now_ts() -> float:
    return time.time()


def _jitter(seconds: float, frac: float = DEFAULT_JITTER_FRAC) -> float:
    if seconds <= 0:
        return 0.0
    delta = (random.random() * 2.0 - 1.0) * frac
    return max(0.0, seconds * (1.0 + delta))


def _schedule_pick(attempts: int, schedule: Sequence[float]) -> float:
    a = max(1, int(attempts))
    idx = min(a, len(schedule)) - 1
    return float(schedule[idx])


def _backoff_seconds(
    attempts: int, *, kind: RetryClass, stall_kind: StallKind = "unknown"
) -> float:
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
    m = (msg or "").lower()
    if not m:
        return False
    needles = (
        "cancelled by scheduler",
        "canceled by scheduler",
        "scheduler_cancel",
        "scheduler cancel",
        "cancel_reason=scheduler",
        "task cancelled",
        "task canceled",
        "cancellederror",
        "asyncio.cancellederror",
    )
    return any(n in m for n in needles)


def _normalize_error_for_signature(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    t = _SIG_HEXLIKE_PAT.sub("<hex>", t)
    t = _SIG_DIGITS_PAT.sub("<n>", t)
    t = _SIG_WS_PAT.sub(" ", t)
    return t[:240]


def _error_signature(
    *,
    cls_norm: RetryClass,
    stall_kind: StallKind,
    nxdomain_like: bool,
    error: str,
    stage: str,
    status_code: Optional[int],
) -> str:
    err_sig_src = _compact_error_text(error or "", max_chars=1200)
    if (
        _GOTO_TIMEOUT_PAT.search(f"{stage}: {err_sig_src}") is not None
        or _GOTO_TIMEOUT_PAT.search(err_sig_src) is not None
    ):
        return "goto_timeout"
    if nxdomain_like:
        return "nxdomain"
    if cls_norm == "stall" and stall_kind == "rate_limit":
        return "rate_limit"
    if cls_norm == "mem":
        return "mem_pressure"
    base = f"{cls_norm}:{stall_kind}:{status_code or ''}:{stage}:{err_sig_src}"
    return _normalize_error_for_signature(base) or "unknown"


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
    last_stall_kind: StallKind = "unknown"

    last_progress_md_done: int = 0
    last_seen_md_done: int = 0

    last_error_sig: str = ""
    same_error_streak: int = 0
    last_error_sig_updated_at: float = 0.0

    force_ready_hits: int = 0
    last_force_ready_at: float = 0.0

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
            last_error_sig=str(d.get("last_error_sig") or ""),
            same_error_streak=int(d.get("same_error_streak") or 0),
            last_error_sig_updated_at=float(d.get("last_error_sig_updated_at") or 0.0),
            force_ready_hits=int(d.get("force_ready_hits") or 0),
            last_force_ready_at=float(d.get("last_force_ready_at") or 0.0),
        )


CompanyRef = Union[str, Company]


def _company_id(ref: CompanyRef) -> str:
    if isinstance(ref, Company):
        return ref.company_id
    return str(ref)


class RetryStateStore:
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
        self.base_dir = (
            Path(base_dir).resolve() if base_dir is not None else _retry_base_dir()
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.base_dir / "retry_state.json"
        self.ledger_path = self.base_dir / "failure_ledger.jsonl"
        self.quarantine_path = self.base_dir / "quarantine.json"

        self._lock = asyncio.Lock()
        self._state: Dict[str, CompanyRetryState] = {}
        self._quarantine: Dict[str, Dict[str, Any]] = {}
        self._dirty_state = False
        self._dirty_quarantine = False
        self._last_flush_ts = 0.0
        self._flush_min_interval_sec = max(0.0, float(flush_min_interval_sec))

        self._default_max_attempts_net = max(0, int(default_max_attempts_net))
        self._default_max_attempts_stall = max(0, int(default_max_attempts_stall))
        self._default_nxdomain_cutoff = max(0, int(default_nxdomain_cutoff))
        self._default_max_attempts_stall_fast = max(
            0, int(default_max_attempts_stall_fast)
        )

        self._load_sync()
        if not self.ledger_path.exists():
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(self.ledger_path, "")

        self._crawl_state = get_crawl_state()

    def _load_sync(self) -> None:
        raw = _json_load_optional(self.state_path)
        if raw is None:
            self._state = {}
        elif not isinstance(raw, dict):
            raise ValueError(f"retry_state.json must be an object: {self.state_path}")
        else:
            self._state = {
                str(cid): CompanyRetryState.from_dict(v)
                for cid, v in raw.items()
                if isinstance(v, dict)
            }

        qraw = _json_load_optional(self.quarantine_path)
        if qraw is None:
            self._quarantine = {}
        elif not isinstance(qraw, dict):
            raise ValueError(
                f"quarantine.json must be an object: {self.quarantine_path}"
            )
        else:
            self._quarantine = {
                str(k): (v if isinstance(v, dict) else {}) for k, v in qraw.items()
            }

        self._dirty_state = False
        self._dirty_quarantine = False
        self._last_flush_ts = _now_ts()

    async def flush(self, *, force: bool = False) -> None:
        now = _now_ts()
        async with self._lock:
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

        await _atomic_write_json_compact_async(self.state_path, snapshot)
        await _atomic_write_json_compact_async(self.quarantine_path, quarantine)

    async def _mark_terminal_done(
        self,
        company_id: str,
        *,
        done_reason: str,
        details: Dict[str, Any],
        last_error: str,
    ) -> None:
        await self._crawl_state.mark_company_terminal(
            company_id,
            reason=done_reason,
            details=details,
            last_error=last_error,
            write_meta=True,
        )

    async def _clear_terminal_done(self, company_id: str) -> None:
        # Preserve Company.status when clearing terminal markers; retry bookkeeping
        # must not reset terminal_done back to pending.
        await self._crawl_state.clear_company_terminal(company_id, keep_status=True)

    @staticmethod
    def classify_unreachable_error(error: str) -> bool:
        e = (error or "").lower()
        return any(
            n in e
            for n in (
                "name or service not known",
                "nxdomain",
                "temporary failure in name resolution",
                "nodename nor servname provided",
                "no address associated with hostname",
            )
        )

    @staticmethod
    def _looks_rate_limited(error: str, status_code: Optional[int]) -> bool:
        e = (error or "").lower()
        if status_code == 429:
            return True
        return any(
            n in e
            for n in (
                "rate limit",
                "too many requests",
                "quota",
                "overloaded",
                "try again later",
            )
        )

    @staticmethod
    def _looks_gateway_or_transport(status_code: Optional[int], error: str) -> bool:
        if status_code in (502, 503, 504, 520, 521, 522, 523, 524):
            return True
        e = (error or "").lower()
        return any(
            n in e
            for n in (
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
            )
        )

    def normalize_class(
        self,
        cls_hint: RetryClass,
        *,
        error: str,
        stage: str,
        status_code: Optional[int],
    ) -> RetryClass:
        cls = str(cls_hint or "").strip().lower()
        if cls == "mem_heavy":
            cls = "mem"
        if cls and cls not in _ALLOWED_RETRY_CLASSES:
            raise ValueError(f"normalize_class: invalid cls_hint={cls_hint!r}")
        if cls == "permanent":
            return "permanent"
        if cls == "mem":
            return "mem"
        if (
            _GOTO_TIMEOUT_PAT.search(f"{stage}: {error}") is not None
            or _GOTO_TIMEOUT_PAT.search(error) is not None
        ):
            return "stall"
        if cls in ("net", "stall", "other"):
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

    @staticmethod
    def _infer_stall_kind(
        *,
        stall_kind_hint: Optional[StallKind],
        error: str,
        stage: str,
        status_code: Optional[int],
    ) -> StallKind:
        if stall_kind_hint is not None:
            sk = str(stall_kind_hint).strip().lower()
            if sk not in _ALLOWED_STALL_KINDS:
                raise ValueError(
                    f"_infer_stall_kind: invalid stall_kind_hint={stall_kind_hint!r}"
                )
            return sk

        if (
            _GOTO_TIMEOUT_PAT.search(f"{stage}: {error}") is not None
            or _GOTO_TIMEOUT_PAT.search(error or "") is not None
        ):
            return "goto"

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
        if any(
            n in e
            for n in (
                "rate limit",
                "too many requests",
                "quota",
                "overloaded",
                "try again later",
            )
        ):
            return "rate_limit"

        return "unknown"

    def _clamp_policy(
        self,
        *,
        max_attempts_net: Optional[int],
        max_attempts_stall: Optional[int],
        nxdomain_cutoff: Optional[int],
        override_allow: bool,
    ) -> Tuple[int, int, int]:
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

    async def pending_ids(self, *, exclude_quarantined: bool = True) -> List[str]:
        async with self._lock:
            if not exclude_quarantined or not self._quarantine:
                return list(self._state.keys())
            return [cid for cid in self._state.keys() if cid not in self._quarantine]

    async def pending_total(self, *, exclude_quarantined: bool = True) -> int:
        async with self._lock:
            if not exclude_quarantined or not self._quarantine:
                return len(self._state)
            return sum(1 for cid in self._state.keys() if cid not in self._quarantine)

    async def pending_eligible_total(
        self, now: Optional[float] = None, *, exclude_quarantined: bool = True
    ) -> int:
        nowv = _now_ts() if now is None else float(now)
        async with self._lock:
            return sum(
                1
                for cid, st in self._state.items()
                if (not exclude_quarantined or cid not in self._quarantine)
                and float(st.next_eligible_at) <= nowv
            )

    def pending_eligible_total_sync(
        self, now: Optional[float] = None, *, exclude_quarantined: bool = True
    ) -> int:
        """
        Sync helper for code paths that cannot await (e.g., exit-code decision).

        Reads persisted state/quarantine from disk (no asyncio.Lock usage).
        """
        nowv = _now_ts() if now is None else float(now)

        raw_state = _json_load_optional(self.state_path)
        if not isinstance(raw_state, dict):
            return 0

        raw_quarantine = _json_load_optional(self.quarantine_path)
        quarantine = raw_quarantine if isinstance(raw_quarantine, dict) else {}

        n = 0
        for cid, st in raw_state.items():
            if exclude_quarantined and cid in quarantine:
                continue
            if not isinstance(st, dict):
                continue
            t = float(st.get("next_eligible_at") or 0.0)
            if t <= nowv:
                n += 1
        return n

    async def is_quarantined(self, company: CompanyRef) -> bool:
        cid = _company_id(company)
        async with self._lock:
            return cid in self._quarantine

    async def quarantine_info(self, company: CompanyRef) -> Optional[Dict[str, Any]]:
        cid = _company_id(company)
        async with self._lock:
            info = self._quarantine.get(cid)
            return dict(info) if isinstance(info, dict) else None

    get_quarantine_info = quarantine_info

    async def get(self, company: CompanyRef) -> Optional[CompanyRetryState]:
        cid = _company_id(company)
        async with self._lock:
            s = self._state.get(cid)
            return CompanyRetryState.from_dict(s.to_dict()) if s is not None else None

    async def next_eligible_at(self, company: CompanyRef) -> float:
        cid = _company_id(company)
        async with self._lock:
            s = self._state.get(cid)
            return float(s.next_eligible_at) if s is not None else 0.0

    async def min_next_eligible_at(self, *, exclude_quarantined: bool = True) -> float:
        now = _now_ts()
        async with self._lock:
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

    async def sleep_hint_sec(
        self,
        *,
        now: Optional[float] = None,
        cap_sec: float = 60.0,
        exclude_quarantined: bool = True,
    ) -> float:
        nowv = _now_ts() if now is None else float(now)
        t = await self.min_next_eligible_at(exclude_quarantined=exclude_quarantined)
        if t <= 0:
            return 0.0
        return max(0.0, min(float(cap_sec), float(t) - nowv))

    async def is_eligible(
        self,
        company: CompanyRef,
        now: Optional[float] = None,
        *,
        ignore_quarantine: bool = False,
    ) -> bool:
        """
        Eligibility gate.
        - By default, quarantined => NOT eligible
        - If ignore_quarantine=True, quarantine is NOT a blocker (scheduler-level override)
        """
        nowv = _now_ts() if now is None else float(now)
        cid = _company_id(company)
        async with self._lock:
            if (not ignore_quarantine) and (cid in self._quarantine):
                return False
            s = self._state.get(cid)
            if s is None:
                return True
            return float(s.next_eligible_at) <= nowv

    async def unquarantine_company(
        self,
        company_id: str,
        *,
        reason: str = "",
        stage: str = "operator_force_requeue",
        flush: bool = True,
    ) -> bool:
        cid = (company_id or "").strip()
        if not cid:
            return False

        removed: Optional[Dict[str, Any]] = None
        async with self._lock:
            removed = self._quarantine.pop(cid, None)
            if removed is None:
                return False
            self._dirty_quarantine = True

        logger.warning(
            "[RetryStateStore] unquarantined cid=%s stage=%s reason=%s prev=%s",
            cid,
            stage,
            reason or "unspecified",
            (removed or {}).get("reason") or "",
        )

        # Keep terminal_done as-is unless your policy wants to clear it too.
        # Force requeue is about scheduler admission; terminal markers are separate.
        if flush:
            await self.flush(force=True)
        return True

    async def force_eligible_now_many(
        self,
        companies: Sequence[CompanyRef],
        *,
        reason: str = "scheduler_force_ready",
        stage: str = "scheduler_force_ready",
        flush: bool = True,
        max_force_ready_hits_per_company: int = 3,
        force_ready_cooldown_sec: float = 60.0,
        quarantine_on_exceed: bool = True,
        ignore_quarantine: bool = False,
    ) -> int:
        """
        Force companies to become eligible immediately.

        Semantics:
          - Sets next_eligible_at = now for retry-pending entries.
          - If ignore_quarantine=True, quarantined companies are ALSO updated (quarantine is not removed).
          - If ignore_quarantine=False, quarantined companies are skipped (legacy behavior).
          - Enforces per-company cooldown and hit-count escalation (optional quarantine).
        """
        if not companies:
            return 0

        now = _now_ts()
        cids = [_company_id(c) for c in companies if _company_id(c).strip()]
        if not cids:
            return 0

        msg = _compact_error_text(reason or "scheduler_force_ready", max_chars=600)
        stg = (stage or "scheduler_force_ready")[:128]

        # ledger (best-effort; no lock needed since it's append-only)
        await _append_jsonl_many_async(
            self.ledger_path,
            [
                {
                    "ts": now,
                    "company_id": cid,
                    "event": "force_ready",
                    "stage": stg,
                    "reason": msg,
                    "ignore_quarantine": bool(ignore_quarantine),
                }
                for cid in cids
            ],
        )

        changed = 0
        skipped_quarantine = 0
        skipped_no_state = 0
        skipped_cooldown = 0
        quarantined_on_exceed = 0

        max_hits = max(0, int(max_force_ready_hits_per_company))
        cooldown = max(0.0, float(force_ready_cooldown_sec))

        async with self._lock:
            for cid in cids:
                # If quarantine is respected, skip quarantined.
                if (not ignore_quarantine) and (cid in self._quarantine):
                    skipped_quarantine += 1
                    continue

                st = self._state.get(cid)
                if st is None:
                    # No retry state => already eligible; nothing to force in store.
                    skipped_no_state += 1
                    continue

                # Cooldown
                last_force = float(st.last_force_ready_at or 0.0)
                if cooldown > 0 and (now - last_force) < cooldown:
                    skipped_cooldown += 1
                    continue

                # Hit counter
                hits = int(st.force_ready_hits or 0) + 1
                st.force_ready_hits = hits
                st.last_force_ready_at = float(now)

                if max_hits > 0 and hits > max_hits and quarantine_on_exceed:
                    # Escalate to quarantine (do not remove retry state; just quarantine it).
                    self._quarantine[cid] = {
                        "ts": float(now),
                        "reason": _compact_error_text(
                            f"force_ready_hits_exceeded hits={hits} max={max_hits}: {msg}",
                            max_chars=600,
                        ),
                        "stage": stg,
                        "cls": "permanent",
                    }
                    self._dirty_quarantine = True
                    quarantined_on_exceed += 1
                    # Do NOT also mark it eligible now; keep quarantine as the controlling state.
                    continue

                # Force eligible now (even if quarantined when ignore_quarantine=True)
                st.next_eligible_at = float(now)
                st.updated_at = float(now)
                st.last_stage = stg
                st.last_error = msg
                self._dirty_state = True
                changed += 1

        if (changed or quarantined_on_exceed) and flush:
            await self.flush(force=True)

        logger.info(
            "retry_force_ready_many changed=%d quarantined_on_exceed=%d skipped_quarantine=%d skipped_no_state=%d skipped_cooldown=%d ignore_quarantine=%s stage=%s",
            int(changed),
            int(quarantined_on_exceed),
            int(skipped_quarantine),
            int(skipped_no_state),
            int(skipped_cooldown),
            bool(ignore_quarantine),
            stg,
        )
        return int(changed)

    async def is_doomed_repeat_goto(
        self,
        company: CompanyRef,
        threshold: int = DEFAULT_SAME_ERROR_STREAK_QUARANTINE_GOTO,
    ) -> bool:
        cid = _company_id(company)
        thr = max(1, int(threshold))
        async with self._lock:
            s = self._state.get(cid)
            if s is None:
                return False
            return (
                (s.last_stall_kind == "goto")
                and (s.last_error_sig == "goto_timeout")
                and (int(s.same_error_streak) >= thr)
            )

    async def mark_transient(
        self,
        company: CompanyRef,
        *,
        reason: str,
        stage: str = "transient",
        next_eligible_delay_sec: float = 0.0,
        md_done: Optional[int] = None,
        flush: bool = True,
    ) -> None:
        cid = _company_id(company)
        now = _now_ts()
        msg = _compact_error_text((reason or "transient").strip(), max_chars=600)

        await _append_jsonl_many_async(
            self.ledger_path,
            [
                {
                    "ts": now,
                    "company_id": cid,
                    "event": "transient",
                    "stage": stage,
                    "reason": msg,
                    "md_done": int(md_done) if md_done is not None else None,
                }
            ],
        )

        async with self._lock:
            s = self._state.get(cid)
            if s is None:
                s = CompanyRetryState()
                self._state[cid] = s

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
            await self.flush()

    async def mark_transient_cancel(
        self,
        company: CompanyRef,
        *,
        reason: str = "cancelled by scheduler",
        stage: str = "scheduler_cancel",
        md_done: Optional[int] = None,
        flush: bool = True,
    ) -> None:
        await self.mark_transient(
            company,
            reason=reason,
            stage=stage,
            next_eligible_delay_sec=0.0,
            md_done=md_done,
            flush=flush,
        )

    record_scheduler_cancel = mark_transient_cancel

    async def mark_success(
        self,
        company: CompanyRef,
        *,
        stage: str = "",
        note: str = "",
        flush: bool = True,
    ) -> None:
        cid = _company_id(company)
        now = _now_ts()

        await _append_jsonl_many_async(
            self.ledger_path,
            [
                {
                    "ts": now,
                    "company_id": cid,
                    "event": "success",
                    "stage": stage,
                    "note": note,
                }
            ],
        )

        async with self._lock:
            if cid in self._state:
                del self._state[cid]
                self._dirty_state = True
            if cid in self._quarantine:
                del self._quarantine[cid]
                self._dirty_quarantine = True

        if flush:
            await self.flush()

        await self._clear_terminal_done(cid)

    async def mark_success_many(
        self,
        companies: Sequence[CompanyRef],
        *,
        stage: str = "",
        note: str = "",
        flush: bool = True,
    ) -> int:
        if not companies:
            return 0
        cids = [_company_id(c) for c in companies]
        now = _now_ts()

        await _append_jsonl_many_async(
            self.ledger_path,
            [
                {
                    "ts": now,
                    "company_id": cid,
                    "event": "success",
                    "stage": stage,
                    "note": note,
                }
                for cid in cids
            ],
        )

        async with self._lock:
            for cid in cids:
                if cid in self._state:
                    del self._state[cid]
                    self._dirty_state = True
                if cid in self._quarantine:
                    del self._quarantine[cid]
                    self._dirty_quarantine = True

        if flush:
            await self.flush()

        for cid in cids:
            await self._clear_terminal_done(cid)

        return len(cids)

    async def quarantine_company(
        self,
        company: CompanyRef,
        *,
        reason: str,
        stage: str,
        error: str,
        cls: RetryClass = "permanent",
        status_code: Optional[int] = None,
        nxdomain_like: bool = False,
        md_done: Optional[int] = None,
        flush: bool = True,
        also_mark_terminal_done: bool = True,
    ) -> None:
        cid = _company_id(company)
        now = _now_ts()
        err_compact = _compact_error_text(
            error or reason or "quarantined", max_chars=1200
        )

        info = {
            "ts": now,
            "reason": _compact_error_text(reason or "", max_chars=600)[:2000],
            "stage": (stage or "")[:128],
            "error": err_compact[:4000],
            "cls": cls,
            "status_code": status_code,
            "nxdomain_like": bool(nxdomain_like),
            "md_done": int(md_done) if md_done is not None else None,
        }

        await _append_jsonl_many_async(
            self.ledger_path,
            [{"ts": now, "company_id": cid, "event": "quarantine", **info}],
        )

        async with self._lock:
            self._quarantine[cid] = info
            s = self._state.get(cid)
            if s is None:
                s = CompanyRetryState(cls="permanent")
                self._state[cid] = s
            s.cls = "permanent"
            s.updated_at = now
            s.last_stage = (stage or "")[:128]
            s.last_error = err_compact[:2000]
            self._dirty_state = True
            self._dirty_quarantine = True

        logger.warning(
            "quarantine_company: cid=%s reason=%s stage=%s status=%s nxdomain=%s md_done=%s error=%r",
            cid,
            info["reason"],
            stage,
            status_code,
            bool(nxdomain_like),
            md_done,
            err_compact[:240],
        )

        if flush:
            await self.flush()

        if also_mark_terminal_done:
            await self._mark_terminal_done(
                cid,
                done_reason="quarantined",
                details={"retry": info},
                last_error=info["error"] or info["reason"] or "quarantined",
            )

    async def clear_quarantine(
        self, company: CompanyRef, *, flush: bool = True
    ) -> bool:
        cid = _company_id(company)
        async with self._lock:
            existed = cid in self._quarantine
            if existed:
                del self._quarantine[cid]
                self._dirty_quarantine = True
        if existed:
            if flush:
                await self.flush()
            await self._clear_terminal_done(cid)
        return existed

    async def mark_failure(
        self,
        company: CompanyRef,
        *,
        cls: RetryClass,
        error: str,
        stage: str,
        status_code: Optional[int] = None,
        nxdomain_like: bool = False,
        stall_kind_hint: Optional[StallKind] = None,
        md_done: Optional[int] = None,
        max_attempts_net: Optional[int] = None,
        max_attempts_stall: Optional[int] = None,
        nxdomain_cutoff: Optional[int] = None,
        override_allow: bool = False,
        flush: bool = True,
    ) -> None:
        cid = _company_id(company)
        now = _now_ts()
        err = _compact_error_text((error or "").strip(), max_chars=1200)
        stg = (stage or "").strip()
        st_code = status_code

        if _is_transient_message(err) or _is_transient_message(stg):
            await self.mark_transient_cancel(
                cid,
                reason=err or "transient_cancel",
                stage=stg or "transient",
                md_done=md_done,
                flush=flush,
            )
            return

        goto_hit = (
            _GOTO_TIMEOUT_PAT.search(f"{stg}: {err}") is not None
            or _GOTO_TIMEOUT_PAT.search(err) is not None
        )
        if goto_hit:
            cls_norm: RetryClass = "stall"
            stall_kind: StallKind = "goto"
        else:
            cls_norm = self.normalize_class(
                cls, error=err, stage=stg, status_code=st_code
            )
            stall_kind = self._infer_stall_kind(
                stall_kind_hint=stall_kind_hint,
                error=err,
                stage=stg,
                status_code=st_code,
            )

        if cls_norm not in _ALLOWED_RETRY_CLASSES:
            raise ValueError(f"mark_failure: invalid normalized cls={cls_norm!r}")
        if stall_kind not in _ALLOWED_STALL_KINDS:
            raise ValueError(f"mark_failure: invalid stall_kind={stall_kind!r}")

        max_net, max_stall, nx_cutoff = self._clamp_policy(
            max_attempts_net=max_attempts_net,
            max_attempts_stall=max_attempts_stall,
            nxdomain_cutoff=nxdomain_cutoff,
            override_allow=override_allow,
        )

        sig = _error_signature(
            cls_norm=cls_norm,
            stall_kind=stall_kind,
            nxdomain_like=bool(nxdomain_like),
            error=err,
            stage=stg,
            status_code=st_code,
        )

        await _append_jsonl_many_async(
            self.ledger_path,
            [
                {
                    "ts": now,
                    "company_id": cid,
                    "event": "failure",
                    "cls": cls_norm,
                    "stage": stg,
                    "error": err[:4000],
                    "status_code": st_code,
                    "nxdomain_like": bool(nxdomain_like),
                    "stall_kind": stall_kind,
                    "md_done": int(md_done) if md_done is not None else None,
                    "error_sig": sig,
                }
            ],
        )

        quarantine_now = False
        quarantine_reason = ""

        prev_streak = 0
        new_streak = 0
        progressed = False
        prev_progress = 0
        prev_seen = 0

        net_attempts = stall_attempts = mem_attempts = other_attempts = 0
        attempts_total = 0
        next_at = 0.0

        async with self._lock:
            s = self._state.get(cid)
            if s is None:
                s = CompanyRetryState()
                self._state[cid] = s

            prev_progress = int(s.last_progress_md_done)
            prev_seen = int(s.last_seen_md_done)

            if md_done is not None:
                mdv = int(md_done)
                s.last_seen_md_done = mdv
                if mdv > int(s.last_progress_md_done):
                    s.last_progress_md_done = mdv

            prev_sig = s.last_error_sig or ""
            prev_streak = int(s.same_error_streak)
            if sig and sig == prev_sig:
                s.same_error_streak = int(s.same_error_streak) + 1
            else:
                s.same_error_streak = 1 if sig else 0
                s.last_error_sig = sig
            s.last_error_sig_updated_at = now
            new_streak = int(s.same_error_streak)

            s.updated_at = now
            s.attempts += 1
            s.cls = cls_norm
            s.last_stage = stg[:128]
            s.last_error = err[:2000]
            s.last_stall_kind = stall_kind

            if cls_norm == "net":
                s.net_attempts += 1
            elif cls_norm == "stall":
                s.stall_attempts += 1
            elif cls_norm == "mem":
                s.mem_attempts += 1
                s.mem_hits += 1
            else:
                s.other_attempts += 1

            net_attempts = int(s.net_attempts)
            stall_attempts = int(s.stall_attempts)
            mem_attempts = int(s.mem_attempts)
            other_attempts = int(s.other_attempts)
            attempts_total = int(s.attempts)

            progressed = False
            if md_done is not None:
                mdv = int(md_done)
                if cls_norm == "stall" and stall_kind == "goto":
                    progressed = mdv > prev_progress
                else:
                    progressed = mdv > prev_progress or mdv > prev_seen

            if cls_norm == "stall" and stall_kind == "goto":
                if (
                    s.last_error_sig == "goto_timeout"
                    and s.same_error_streak >= DEFAULT_SAME_ERROR_STREAK_QUARANTINE_GOTO
                ):
                    quarantine_now = True
                    quarantine_reason = "repeat_goto_timeout"
            else:
                if (
                    s.same_error_streak >= DEFAULT_SAME_ERROR_STREAK_QUARANTINE_GENERIC
                    and (not progressed)
                    and sig
                ):
                    quarantine_now = True
                    quarantine_reason = "repeat_same_error"

            if not quarantine_now:
                if cls_norm == "permanent":
                    quarantine_now = True
                    quarantine_reason = "permanent"
                elif nxdomain_like and nx_cutoff > 0 and s.net_attempts >= nx_cutoff:
                    quarantine_now = True
                    quarantine_reason = "nxdomain_like_cutoff"
                elif cls_norm == "net" and max_net > 0 and s.net_attempts >= max_net:
                    quarantine_now = True
                    quarantine_reason = "max_net_attempts"
                elif cls_norm == "stall":
                    limit = max_stall
                    if stall_kind in ("goto", "no_yield"):
                        limit = (
                            min(limit, self._default_max_attempts_stall_fast)
                            if limit > 0
                            else self._default_max_attempts_stall_fast
                        )
                    if limit > 0 and s.stall_attempts >= limit and not progressed:
                        quarantine_now = True
                        quarantine_reason = f"max_stall_attempts[{stall_kind}]"

            next_delay = _backoff_seconds(
                attempts=s.attempts if cls_norm != "permanent" else 1,
                kind=cls_norm,
                stall_kind=stall_kind,
            )
            s.next_eligible_at = float(now + _jitter(next_delay, DEFAULT_JITTER_FRAC))
            next_at = float(s.next_eligible_at)
            self._dirty_state = True

        _dbg(
            "mark_failure: cid=%s cls=%s stall=%s sig=%s streak=%d->%d progressed=%s md_done=%s "
            "attempts_total=%d net=%d stall=%d mem=%d other=%d quarantine=%s reason=%s next_in=%.1fs",
            cid,
            cls_norm,
            stall_kind,
            sig,
            prev_streak,
            new_streak,
            progressed,
            md_done,
            attempts_total,
            net_attempts,
            stall_attempts,
            mem_attempts,
            other_attempts,
            quarantine_now,
            quarantine_reason,
            max(0.0, next_at - now),
        )

        if quarantine_now:
            await self.quarantine_company(
                cid,
                reason=quarantine_reason,
                stage=stg or "failure",
                error=err or quarantine_reason,
                cls="permanent",
                status_code=st_code,
                nxdomain_like=nxdomain_like,
                md_done=md_done,
                flush=flush,
            )
            return

        if flush:
            await self.flush()

    async def company_is_quarantined(self, company: CompanyRef) -> bool:
        return await self.is_quarantined(company)

    async def has_pending(self) -> bool:
        return (await self.pending_total(exclude_quarantined=True)) > 0

    async def company_has_pending(self, company: CompanyRef) -> bool:
        cid = _company_id(company)
        async with self._lock:
            if cid in self._quarantine:
                return False
            return cid in self._state

    async def is_pending(self, company: CompanyRef) -> bool:
        return await self.company_has_pending(company)


@dataclass(slots=True)
class AttemptOutcome:
    ok: bool
    stage: str
    event: Optional[RetryEvent] = None
    md_done: Optional[int] = None

    terminalized: bool = False
    terminal_reason: Optional[str] = None
    terminal_last_error: Optional[str] = None

    should_mark_success: bool = False


@dataclass(slots=True)
class TerminalizationDecision:
    should_terminalize: bool
    reason: str
    last_error: str
    as_retry_class: RetryClass = "stall"
    stall_kind: Optional[StallKind] = None


@dataclass(slots=True)
class RequeueDecision:
    should_requeue: bool
    delay_sec: float
    reason: str


@dataclass(slots=True)
class RecordResult:
    recorded: bool
    quarantined: bool
    note: str = ""


def decide_terminalization(
    *,
    page_summary_decision: Optional[Any],
    exception: Optional[BaseException],
    stage: str,
    urls_md_done: int,
    attempt_index: int,
) -> TerminalizationDecision:
    if urls_md_done > 0:
        return TerminalizationDecision(
            should_terminalize=False,
            reason="has_progress",
            last_error="has_progress",
            as_retry_class="stall",
            stall_kind="unknown",
        )

    if page_summary_decision is not None:
        action = getattr(page_summary_decision, "action", None)
        reason = str(getattr(page_summary_decision, "reason", "") or "")

        if action == "terminal":
            if attempt_index == 0:
                return TerminalizationDecision(
                    should_terminalize=False,
                    reason=f"page_summary_terminal_downgraded:{reason}",
                    last_error=_compact_error_text(
                        reason or "page_summary_terminal", max_chars=400
                    ),
                    as_retry_class="stall",
                    stall_kind="unknown",
                )
            return TerminalizationDecision(
                should_terminalize=True,
                reason=f"page_summary_terminal:{reason}",
                last_error=_compact_error_text(
                    reason or "page_summary_terminal", max_chars=400
                ),
                as_retry_class="permanent",
                stall_kind=None,
            )

        if action == "mem":
            return TerminalizationDecision(
                should_terminalize=False,
                reason=f"page_summary_mem:{reason}",
                last_error=_compact_error_text(reason or "mem", max_chars=400),
                as_retry_class="mem",
                stall_kind=None,
            )

        if action == "stall":
            return TerminalizationDecision(
                should_terminalize=False,
                reason=f"page_summary_stall:{reason}",
                last_error=_compact_error_text(reason or "stall", max_chars=400),
                as_retry_class="stall",
                stall_kind="unknown",
            )

    if exception is not None:
        msg = _exc_to_compact_string(exception, max_chars=800)
        if should_fail_fast_on_goto(exception, stage=stage):
            return TerminalizationDecision(
                should_terminalize=False,
                reason=f"goto_fail_fast_downgraded:{msg}",
                last_error=msg,
                as_retry_class="stall",
                stall_kind="goto",
            )

        ev = classify_failure(exception, stage=stage)
        if ev.nxdomain_like:
            return TerminalizationDecision(
                should_terminalize=False,
                reason=f"nxdomain_like:{ev.error}",
                last_error=ev.error,
                as_retry_class="net",
                stall_kind=None,
            )

        if ev.stall_kind == "goto":
            return TerminalizationDecision(
                should_terminalize=False,
                reason=f"goto_timeout:{ev.error}",
                last_error=ev.error,
                as_retry_class="stall",
                stall_kind="goto",
            )

    return TerminalizationDecision(
        should_terminalize=False,
        reason="no_terminalize",
        last_error="",
        as_retry_class="stall",
        stall_kind="unknown",
    )


async def record_attempt(
    store: RetryStateStore,
    company: CompanyRef,
    outcome: AttemptOutcome,
    *,
    flush: bool = True,
    never_clear_quarantine_on_success: bool = True,
) -> RecordResult:
    cid = _company_id(company)

    if outcome.terminalized:
        await store.mark_failure(
            cid,
            cls="permanent",
            error=_compact_error_text(
                str(
                    outcome.terminal_last_error
                    or outcome.terminal_reason
                    or "terminalize"
                ),
                max_chars=800,
            ),
            stage="terminalize",
            status_code=None,
            nxdomain_like=False,
            stall_kind_hint=None,
            md_done=outcome.md_done,
            override_allow=False,
            flush=flush,
        )
        return RecordResult(recorded=True, quarantined=True, note="terminalized")

    if outcome.ok and outcome.should_mark_success:
        if never_clear_quarantine_on_success and await store.is_quarantined(cid):
            return RecordResult(
                recorded=True, quarantined=True, note="success_but_quarantined_kept"
            )
        await store.mark_success(cid, stage=outcome.stage, note="ok", flush=flush)
        return RecordResult(recorded=True, quarantined=False, note="success")

    if outcome.stage in ("scheduler_cancel", "cancelled", "cancel"):
        reason = outcome.event.error if outcome.event is not None else "cancelled"
        await store.mark_transient_cancel(
            cid,
            reason=_compact_error_text(reason, max_chars=600),
            stage=outcome.stage,
            md_done=outcome.md_done,
            flush=flush,
        )
        return RecordResult(
            recorded=True,
            quarantined=await store.is_quarantined(cid),
            note="transient_cancel",
        )

    if outcome.event is None:
        raise ValueError(
            "record_attempt: outcome.event is required for non-success, non-cancel outcomes"
        )

    ev = outcome.event
    await store.mark_failure(
        cid,
        cls=ev.cls,
        error=_compact_error_text(ev.error, max_chars=1200),
        stage=ev.stage,
        status_code=ev.status_code,
        nxdomain_like=ev.nxdomain_like,
        stall_kind_hint=(ev.stall_kind or None),
        md_done=outcome.md_done,
        override_allow=False,
        flush=flush,
    )
    return RecordResult(
        recorded=True, quarantined=await store.is_quarantined(cid), note="failure"
    )


async def decide_requeue(
    *,
    store: RetryStateStore,
    company: CompanyRef,
    is_runnable: bool,
    stop_requested: bool,
) -> RequeueDecision:
    cid = _company_id(company)

    if stop_requested:
        return RequeueDecision(False, 0.0, "stop_requested")
    if not is_runnable:
        return RequeueDecision(False, 0.0, "not_runnable")
    if await store.is_quarantined(cid):
        return RequeueDecision(False, 0.0, "quarantined")
    if await store.is_doomed_repeat_goto(cid):
        return RequeueDecision(False, 0.0, "doomed_repeat_goto")

    delay = max(0.0, float(await store.next_eligible_at(cid)) - time.time())
    return RequeueDecision(True, delay, "retry_policy_next_eligible")


def decide_exit_code(
    *,
    forced_exit_code: Optional[int],
    retry_exit_code: int,
    scheduler_pending_total: int,
    retry_pending_total: int,
    db_in_progress: bool,
    in_progress_payload: bool,
) -> int:
    if forced_exit_code is not None:
        return int(forced_exit_code)
    if (
        scheduler_pending_total > 0
        or retry_pending_total > 0
        or db_in_progress
        or in_progress_payload
    ):
        return int(retry_exit_code)
    return 0


def traceback_enabled_from_env() -> bool:
    def _truthy(v: str) -> bool:
        return (v or "").strip().lower() in {"1", "true", "yes", "on"}

    return _truthy(os.getenv("RUN_TRACEBACK", "")) or _truthy(
        os.getenv("RETRY_DEBUG", "")
    )


TRACEBACK_ENABLED: bool = traceback_enabled_from_env()


def short_exc(e: BaseException, limit: int = 900) -> str:
    msg = f"{type(e).__name__}: {e}"
    msg = _SIG_WS_PAT.sub(" ", (msg or "").strip())
    if len(msg) > int(limit):
        return msg[: int(limit) - 1] + "…"
    return msg


def log_attempt_failure(
    clog: logging.Logger,
    *,
    prefix: str,
    attempt_index: int,
    stage: str,
    event: Optional["RetryEvent"],
    exc: BaseException,
    traceback_enabled: bool,
) -> None:
    cls = getattr(event, "cls", None)
    sk = getattr(event, "stall_kind", None)
    sc = getattr(event, "status_code", None)
    nx = getattr(event, "nxdomain_like", None)

    if traceback_enabled:
        clog.exception(
            "%s attempt=%d stage=%s cls=%s stall=%s status=%s nx=%s err=%s",
            prefix,
            attempt_index,
            stage,
            cls,
            sk,
            sc,
            nx,
            short_exc(exc),
        )
        return

    clog.error(
        "%s attempt=%d stage=%s cls=%s stall=%s status=%s nx=%s err=%s",
        prefix,
        attempt_index,
        stage,
        cls,
        sk,
        sc,
        nx,
        short_exc(exc),
        exc_info=False,
    )


__all__ = [
    "RetryEvent",
    "CriticalMemoryPressure",
    "CrawlerFatalError",
    "CrawlerTimeoutError",
    "classify_failure",
    "should_fail_fast_on_goto",
    "is_goto_timeout_error",
    "is_playwright_driver_disconnect",
    "RetryStateStore",
    "CompanyRetryState",
    "AttemptOutcome",
    "TerminalizationDecision",
    "RequeueDecision",
    "RecordResult",
    "decide_terminalization",
    "record_attempt",
    "decide_requeue",
    "decide_exit_code",
    "traceback_enabled_from_env",
    "TRACEBACK_ENABLED",
    "short_exc",
    "log_attempt_failure",
]
