from __future__ import annotations

import logging
from collections import deque
from contextvars import ContextVar
from pathlib import Path
from typing import Deque, Dict, Optional

from .output_paths import ensure_company_dirs, sanitize_bvdid


# Context: current company in this task
_CURRENT_BVDID: ContextVar[Optional[str]] = ContextVar("_CURRENT_BVDID", default=None)


def _as_path(x: object) -> Optional[Path]:
    if x is None:
        return None
    if isinstance(x, Path):
        return x
    if isinstance(x, str):
        return Path(x)
    return None


def _pick_company_root_dir(dirs: Dict[str, object]) -> Optional[Path]:
    """
    ensure_company_dirs() implementations can vary over time.
    We try common keys first, then fall back to the first path-like value.
    """
    for k in (
        "company",
        "company_dir",
        "root",
        "base",
        "out",
        "output",
        "data",
        "metadata",
        "checkpoints",
        "pages",
    ):
        p = _as_path(dirs.get(k))
        if p is not None:
            return p

    for v in dirs.values():
        p = _as_path(v)
        if p is not None:
            return p

    return None


def _resolve_logs_dir_for_company(bvdid: str) -> Path:
    """
    Robustly resolve a per-company logs directory.

    - Preferred: dirs["logs"] if present.
    - Else: <company_root>/logs if we can infer a company root.
    - Else: ./logs (last resort).
    """
    dirs = ensure_company_dirs(bvdid)

    logs_dir = _as_path(dirs.get("log"))
    if logs_dir is None:
        root = _pick_company_root_dir(dirs)
        if root is not None:
            logs_dir = Path(root) / "log"
        else:
            logs_dir = Path("log")

    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


class _BvdidFilter(logging.Filter):
    """
    Filter that only lets through records for the currently-bound BVDID
    (via the ContextVar) or records logged under the company-specific logger.
    """

    __slots__ = ("bvdid",)

    def __init__(self, bvdid: str) -> None:
        super().__init__()
        self.bvdid = str(bvdid)

    def filter(self, record: logging.LogRecord) -> bool:
        current = _CURRENT_BVDID.get()
        if current == self.bvdid:
            return True
        name = getattr(record, "name", "") or ""
        return name.startswith(f"company.{self.bvdid}")


class LoggingExtension:
    """
    Logging plugin/extension.

    Responsibilities:
      - Installs a console handler (single, shared).
      - Provides per-company file handlers, but keeps at most `max_open_company_logs`
        open at a time (LRU).
      - You MUST call `close_company(bvdid)` after a company run to free its handle early.
    """

    __slots__ = (
        "log_dir",
        "global_level",
        "per_company_level",
        "max_open",
        "_company_handlers",
        "_lru",
        "_session_handler",
    )

    def __init__(
        self,
        log_dir: Path = Path("logs"),  # kept for compatibility; not relied on
        *,
        global_level: int = logging.INFO,
        per_company_level: Optional[int] = None,
        max_open_company_logs: int = 128,
        enable_session_log: bool = False,
        session_log_path: Optional[Path] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.global_level = int(global_level)
        self.per_company_level = int(
            per_company_level if per_company_level is not None else global_level
        )
        self.max_open: int = max(8, int(max_open_company_logs))
        self._company_handlers: Dict[str, logging.Handler] = {}
        self._lru: Deque[str] = deque()  # bvdid order, most-recent at the right
        self._session_handler: Optional[logging.Handler] = None

        self._install_console(self.global_level)

        if enable_session_log and session_log_path:
            session_log_path = Path(session_log_path)
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
            sh = logging.FileHandler(session_log_path, mode="a", encoding="utf-8")
            sh.setLevel(self.global_level)
            sh.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logging.getLogger().addHandler(sh)
            self._session_handler = sh

        # Root permissive; handler levels do the filtering
        logging.getLogger().setLevel(logging.DEBUG)

    # ---------------- Console ----------------

    def _install_console(self, level: int) -> None:
        """
        Install a single console handler for the root logger, replacing any
        existing plain StreamHandler. Keeps console logging deterministic.
        """
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                root.removeHandler(h)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(ch)

    # ---------------- LRU helpers ----------------

    def _touch(self, bvdid: str) -> None:
        try:
            self._lru.remove(bvdid)
        except ValueError:
            pass
        self._lru.append(bvdid)

    def _evict_if_needed(self) -> None:
        """
        Ensure we keep at most `self.max_open` company log file handlers open.
        Evicts least-recently-used handlers first.
        """
        root = logging.getLogger()
        while len(self._company_handlers) > self.max_open:
            old = self._lru.popleft()
            h = self._company_handlers.pop(old, None)
            if h is None:
                continue
            try:
                root.removeHandler(h)
            except Exception:
                pass
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass

    # ---------------- Company logger ----------------

    def get_company_logger(self, bvdid: str) -> logging.Logger:
        """
        Return a logger dedicated to a specific company (bvdid).
        Logger name is "company.{bvdid}" and writes to a per-company file.
        """
        safe = sanitize_bvdid(bvdid)
        logs_dir = _resolve_logs_dir_for_company(bvdid)
        company_log_path = logs_dir / f"{safe}.log"

        root = logging.getLogger()

        if bvdid not in self._company_handlers:
            fh = logging.FileHandler(company_log_path, mode="a", encoding="utf-8")
            fh.setLevel(self.per_company_level)
            fh.addFilter(_BvdidFilter(bvdid))
            fh.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(fh)
            self._company_handlers[bvdid] = fh
            self._touch(bvdid)
            self._evict_if_needed()
        else:
            self._touch(bvdid)

        clog = logging.getLogger(f"company.{bvdid}")
        clog.setLevel(logging.DEBUG)
        clog.propagate = True
        return clog

    # ---------------- Context helpers ----------------

    def set_company_context(self, bvdid: str):
        """Bind current company context to this task/flow. Returns token to reset."""
        return _CURRENT_BVDID.set(str(bvdid))

    def reset_company_context(self, token) -> None:
        try:
            _CURRENT_BVDID.reset(token)
        except Exception:
            pass

    # ---------------- Close per company (free FD early) ----------------

    def close_company(self, bvdid: str) -> None:
        """Close this company's file handler to free a descriptor immediately."""
        root = logging.getLogger()
        h = self._company_handlers.pop(bvdid, None)
        if h is not None:
            try:
                root.removeHandler(h)
            except Exception:
                pass
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
        try:
            self._lru.remove(bvdid)
        except ValueError:
            pass

    # ---------------- Cleanup ----------------

    def close(self) -> None:
        """
        Close all company-specific handlers and clear internal LRU state.
        Console handler remains. Session handler is also closed if created.
        """
        root = logging.getLogger()
        for fh in list(self._company_handlers.values()):
            try:
                root.removeHandler(fh)
            except Exception:
                pass
            try:
                fh.flush()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass
        self._company_handlers.clear()
        self._lru.clear()

        if self._session_handler is not None:
            try:
                root.removeHandler(self._session_handler)
            except Exception:
                pass
            try:
                self._session_handler.flush()
            except Exception:
                pass
            try:
                self._session_handler.close()
            except Exception:
                pass
            self._session_handler = None
