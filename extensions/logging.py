from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional
from contextvars import ContextVar

from .output_paths import ensure_company_dirs

# Per-task context: which hojin_id are we processing right now?
_CURRENT_HOJIN_ID: ContextVar[Optional[str]] = ContextVar("_CURRENT_HOJIN_ID", default=None)


class _HojinFilter(logging.Filter):
    """
    Allow records if they belong to the current company context OR
    if their logger name starts with company.<hojin_id>.
    This lets us attach the handler high (root) and still isolate per company.
    """
    def __init__(self, hojin_id: str) -> None:
        super().__init__()
        self.hojin_id = str(hojin_id)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        current = _CURRENT_HOJIN_ID.get()
        if current == self.hojin_id:
            return True
        name = getattr(record, "name", "") or ""
        return name.startswith(f"company.{self.hojin_id}")


class LoggingExtension:
    def __init__(
        self,
        log_dir: Path = Path("logs"),  # global (console) logs can still go here if you want
        *,
        global_level: int = logging.INFO,
        per_company_level: Optional[int] = None,  # default to global_level if None
    ) -> None:
        self.log_dir = log_dir
        self.global_level = global_level
        self.per_company_level = per_company_level if per_company_level is not None else global_level
        self._company_handlers: Dict[str, logging.Handler] = {}

        # Console formatter/handler on root
        self._install_console(self.global_level)

        # Make root permissive; rely on handler levels to filter.
        logging.getLogger().setLevel(logging.DEBUG)

    # ---------------- Console ----------------

    def _install_console(self, level: int) -> None:
        root = logging.getLogger()
        # Remove any default handlers (e.g., from basicConfig)
        for h in list(root.handlers):
            root.removeHandler(h)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(ch)

    # ---------------- Company logger ----------------

    def get_company_logger(self, hojin_id: str) -> logging.Logger:
        """
        Return a company-scoped logger. Also ensures a per-company
        file handler is attached high at root with a filter that routes
        only the current company's logs into that file.
        """
        # 1) Ensure outputs/{hojin_id}/logs exists and create file handler
        dirs = ensure_company_dirs(hojin_id)
        company_log_path = dirs["logs"] / f"{hojin_id}.log"

        if hojin_id not in self._company_handlers:
            fh = logging.FileHandler(company_log_path, mode="a", encoding="utf-8")
            fh.setLevel(self.per_company_level)  # <- respects --log-level, includes DEBUG if set
            fh.addFilter(_HojinFilter(hojin_id))
            fh.setFormatter(logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            # 2) Attach to ROOT so it can see records from all modules
            logging.getLogger().addHandler(fh)
            self._company_handlers[hojin_id] = fh

        # 3) Return a named logger for convenience (child logs always pass the filter)
        logger = logging.getLogger(f"company.{hojin_id}")
        logger.setLevel(logging.DEBUG)  # be permissive locally
        logger.propagate = True         # bubble to root (file/console handlers live there)
        return logger

    # ---------------- Context helpers ----------------

    def set_company_context(self, hojin_id: str):
        """
        Activate the per-task company context so any module logger emits into
        that company's file (via the root-attached filtered handler).
        Returns a token you must reset when done.
        """
        return _CURRENT_HOJIN_ID.set(str(hojin_id))

    def reset_company_context(self, token) -> None:
        try:
            _CURRENT_HOJIN_ID.reset(token)
        except Exception:
            # Safe guard; context might already be cleared
            pass

    # ---------------- Cleanup ----------------

    def close(self):
        root = logging.getLogger()
        for fh in self._company_handlers.values():
            try:
                root.removeHandler(fh)
                fh.flush()
                fh.close()
            except Exception:
                pass
        self._company_handlers.clear()