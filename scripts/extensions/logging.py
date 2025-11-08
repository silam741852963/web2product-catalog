from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, Optional
from contextvars import ContextVar

from .output_paths import ensure_company_dirs, sanitize_bvdid

# Context: current company in this task
_CURRENT_BVDID: ContextVar[Optional[str]] = ContextVar("_CURRENT_BVDID", default=None)


class _BvdidFilter(logging.Filter):
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
    - Installs a console handler.
    - Provides per-company file handlers, but keeps at most `max_open_company_logs`
      open at a time (LRU). Older ones are closed to avoid EMFILE on Windows.
    - You MUST call `close_company(bvdid)` after a company run to free its handle early.
    """
    def __init__(
        self,
        log_dir: Path = Path("logs"),  # not used; kept for compat
        *,
        global_level: int = logging.INFO,
        per_company_level: Optional[int] = None,
        max_open_company_logs: int = 128,   # <- NEW: hard cap of simultaneously open files
        enable_session_log: bool = False,   # optional: single global file log
        session_log_path: Optional[Path] = Path("logs/session.log"),
    ) -> None:
        self.log_dir = log_dir
        self.global_level = global_level
        self.per_company_level = per_company_level if per_company_level is not None else global_level
        self.max_open = max(8, int(max_open_company_logs))
        self._company_handlers: Dict[str, logging.Handler] = {}
        self._lru: deque[str] = deque()  # bvdid order, most-recent at the right

        # Console handler
        self._install_console(self.global_level)

        # Optional single session log (one FD only)
        if enable_session_log and session_log_path:
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
            sh = logging.FileHandler(session_log_path, mode="a", encoding="utf-8")
            sh.setLevel(self.global_level)
            sh.setFormatter(logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logging.getLogger().addHandler(sh)

        # Root permissive; handler levels do the filtering
        logging.getLogger().setLevel(logging.DEBUG)

    # ---------------- Console ----------------

    def _install_console(self, level: int) -> None:
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler):
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
        while len(self._company_handlers) > self.max_open:
            old = self._lru.popleft()
            h = self._company_handlers.pop(old, None)
            if h is not None:
                try:
                    logging.getLogger().removeHandler(h)
                except Exception:
                    pass
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass

    # ---------------- Company logger ----------------

    def get_company_logger(self, bvdid: str) -> logging.Logger:
        dirs = ensure_company_dirs(bvdid)
        safe = sanitize_bvdid(bvdid)
        company_log_path = dirs["logs"] / f"{safe}.log"

        if bvdid not in self._company_handlers:
            fh = logging.FileHandler(company_log_path, mode="a", encoding="utf-8")
            fh.setLevel(self.per_company_level)
            fh.addFilter(_BvdidFilter(bvdid))
            fh.setFormatter(logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logging.getLogger().addHandler(fh)
            self._company_handlers[bvdid] = fh
            self._touch(bvdid)
            self._evict_if_needed()
        else:
            self._touch(bvdid)

        logger = logging.getLogger(f"company.{bvdid}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
        return logger

    # ---------------- Context helpers ----------------

    def set_company_context(self, bvdid: str):
        return _CURRENT_BVDID.set(str(bvdid))

    def reset_company_context(self, token) -> None:
        try:
            _CURRENT_BVDID.reset(token)
        except Exception:
            pass

    # ---------------- Close per company (free FD early) ----------------

    def close_company(self, bvdid: str) -> None:
        """Close this company's file handler to free a descriptor immediately."""
        h = self._company_handlers.pop(bvdid, None)
        if h is not None:
            try:
                logging.getLogger().removeHandler(h)
            except Exception:
                pass
            try:
                h.flush()
                h.close()
            except Exception:
                pass
        try:
            self._lru.remove(bvdid)
        except ValueError:
            pass

    # ---------------- Cleanup ----------------

    def close(self):
        root = logging.getLogger()
        for fh in list(self._company_handlers.values()):
            try:
                root.removeHandler(fh)
            except Exception:
                pass
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
        self._company_handlers.clear()
        self._lru.clear()