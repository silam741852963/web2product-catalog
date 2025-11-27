from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryTrackerConfig:
    """
    Configuration for retry tracking.

    out_dir: base output directory (usually args.out_dir)
    filename: name of the JSON file (default: retry_companies.json)
    flush_threshold: number of changes before auto flush
    """
    out_dir: Path
    filename: str = "retry_companies.json"
    flush_threshold: int = 1


class RetryTracker:
    """
    Centralized tracker for companies that need retry due to:
      - stalls
      - per page timeouts
      - critical memory pressure

    It also tracks:
      - total_companies for this run
      - attempted_total (count of companies that started a pipeline)
      - all_attempted (attempted_total >= total_companies)

    The JSON written is compatible with your existing format:
      {
        "generated_at": ...,
        "stalled_companies": [...],
        "timeout_companies": [...],
        "memory_companies": [...],
        "retry_companies": [...],
        "total_companies": N,
        "attempted_total": K,
        "all_attempted": true | false
      }
    """

    def __init__(self, cfg: RetryTrackerConfig) -> None:
        self.cfg = cfg
        self.path: Path = cfg.out_dir / cfg.filename
        self.flush_threshold: int = cfg.flush_threshold

        self._stalled: Set[str] = set()
        self._timeout: Set[str] = set()
        self._memory: Set[str] = set()
        self._total_companies: int = 0
        self._attempted_companies: Set[str] = set()
        self._dirty: int = 0

    # ---------------- Existing file handling ----------------

    def load_existing(self) -> Dict[str, Any]:
        """
        Best effort load of an existing retry_companies.json.
        Used by run.py to support retry_mode (all, skip-retry, only-retry).

        Returns an empty dict on any error and logs a warning.
        """
        try:
            if not self.path.exists():
                return {}
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Failed to load existing retry file %s: %s", self.path, e)
            return {}

    def get_previous_retry_ids(self) -> Set[str]:
        """
        Convenience helper to get the set of company_ids from the
        previous run's retry_companies.json.
        """
        data = self.load_existing()
        retry_list = data.get("retry_companies") or []
        if not isinstance(retry_list, list):
            return set()
        return {str(x) for x in retry_list}

    # ---------------- Run level stats ----------------

    def set_total_companies(self, total: int) -> None:
        """
        Set the total number of companies that this run intends to process.
        Used to compute all_attempted flag in the JSON payload.
        """
        self._total_companies = max(0, int(total))

    def record_attempt(self, company_id: str) -> None:
        """
        Mark that the pipeline for this company at least started.
        """
        self._attempted_companies.add(company_id)
        # This affects the summary so we mark dirty too.
        self._dirty += 1
        self.flush()

    # ---------------- Mark functions ----------------

    def mark_timeout(self, company_id: str) -> None:
        if company_id in self._timeout:
            return
        self._timeout.add(company_id)
        self._dirty += 1
        self.flush()

    def mark_stalled(self, company_id: str) -> None:
        if company_id in self._stalled:
            return
        self._stalled.add(company_id)
        self._dirty += 1
        self.flush()

    def mark_memory(self, company_id: str) -> None:
        if company_id in self._memory:
            return
        self._memory.add(company_id)
        self._dirty += 1
        self.flush()

    # ---------------- Flush and summary ----------------

    def flush(self, force: bool = False) -> None:
        """
        Persist current retry state atomically to disk if needed.
        """
        if not force and self._dirty < self.flush_threshold:
            return

        stalled_ids = sorted(self._stalled)
        timeout_ids = sorted(self._timeout)
        memory_ids = sorted(self._memory)
        retry_ids = sorted(set(stalled_ids) | set(timeout_ids) | set(memory_ids))

        total = self._total_companies or 0
        attempted_total = len(self._attempted_companies)
        all_attempted = bool(total and attempted_total >= total)

        payload: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stalled_companies": stalled_ids,
            "timeout_companies": timeout_ids,
            "memory_companies": memory_ids,
            "retry_companies": retry_ids,
            "total_companies": total,
            "attempted_total": attempted_total,
            "all_attempted": all_attempted,
        }

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
        self._dirty = 0

    def summary(self) -> Dict[str, Any]:
        stalled_ids = sorted(self._stalled)
        timeout_ids = sorted(self._timeout)
        memory_ids = sorted(self._memory)
        retry_ids = sorted(set(stalled_ids) | set(timeout_ids) | set(memory_ids))

        total = self._total_companies or 0
        attempted_total = len(self._attempted_companies)
        all_attempted = bool(total and attempted_total >= total)

        return {
            "stalled": stalled_ids,
            "timeout": timeout_ids,
            "memory": memory_ids,
            "retry_companies": retry_ids,
            "total_companies": total,
            "attempted_total": attempted_total,
            "all_attempted": all_attempted,
        }

    def has_retries(self) -> bool:
        return bool(self._stalled or self._timeout or self._memory)

    # ---------------- Finalization helper ----------------

    def finalize_and_exit_code(self, retry_exit_code: int) -> int:
        """
        Flush the latest state and decide what exit code the run should use.
        Returns 0 if there are no retries, or retry_exit_code otherwise.
        """
        self.flush(force=True)
        s = self.summary()
        retry_ids = s["retry_companies"]

        if retry_ids:
            logger.error(
                "Detected %d companies requiring retry (stall=%d, timeout=%d, memory=%d). "
                "See %s.",
                len(retry_ids),
                len(s["stalled"]),
                len(s["timeout"]),
                len(s["memory"]),
                self.path,
            )
            return retry_exit_code

        logger.info(
            "No stalled companies, page timeouts, or memory pressure companies detected."
        )
        return 0