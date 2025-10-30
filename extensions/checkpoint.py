from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from extensions.output_paths import ensure_company_dirs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Checkpoint schema and helpers
# ---------------------------------------------------------------------------

class CompanyCheckpoint:
    """
    Represents per-company progress.
    Stored at outputs/{hojin_id}/checkpoints/progress.json
    """

    def __init__(self, hojin_id: str):
        self.hojin_id = str(hojin_id)
        dirs = ensure_company_dirs(hojin_id)
        self.path = dirs["checkpoints"] / "progress.json"
        self.data: Dict[str, Any] = {
            "hojin_id": self.hojin_id,
            "started_at": None,
            "finished_at": None,
            "stage": None,  # html | markdown | llm
            "urls_total": 0,
            "urls_done": 0,
            "urls_failed": 0,
            "last_url": None,
            "errors": [],
            "notes": [],
        }
        self._lock = asyncio.Lock()

    # ---------------------- Core methods ----------------------

    async def load(self) -> None:
        """Load existing checkpoint if exists."""
        async with self._lock:
            if self.path.exists():
                try:
                    text = self.path.read_text(encoding="utf-8")
                    self.data.update(json.loads(text))
                    logger.info(f"[checkpoint] Loaded checkpoint for {self.hojin_id}")
                except Exception as e:
                    logger.warning(f"[checkpoint] Failed to load: {e}")

    async def save(self) -> None:
        """Persist current checkpoint to disk."""
        async with self._lock:
            try:
                self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.error(f"[checkpoint] Save failed for {self.hojin_id}: {e}")

    async def mark_start(self, stage: str, total_urls: int = 0) -> None:
        self.data.update({
            "started_at": datetime.utcnow().isoformat(),
            "stage": stage,
            "urls_total": total_urls,
            "urls_done": 0,
            "urls_failed": 0,
            "errors": [],
            "notes": [],
        })
        await self.save()

    async def mark_url_done(self, url: str) -> None:
        self.data["urls_done"] += 1
        self.data["last_url"] = url
        await self.save()

    async def mark_url_failed(self, url: str, reason: str) -> None:
        self.data["urls_failed"] += 1
        self.data["last_url"] = url
        self.data.setdefault("errors", []).append({"url": url, "reason": reason})
        await self.save()

    async def add_note(self, text: str) -> None:
        self.data.setdefault("notes", []).append({
            "time": datetime.utcnow().isoformat(),
            "note": text
        })
        await self.save()

    async def mark_finished(self) -> None:
        self.data["finished_at"] = datetime.utcnow().isoformat()
        await self.save()

    # ---------------------- Convenience accessors ----------------------

    def is_finished(self) -> bool:
        return self.data.get("finished_at") is not None

    def progress_ratio(self) -> float:
        total = self.data.get("urls_total", 0) or 1
        return self.data.get("urls_done", 0) / total

# ---------------------------------------------------------------------------
#  Global checkpoint manager for multi-company runs
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Keeps track of multiple CompanyCheckpoint instances.
    Supports concurrent updates and persistence.
    """

    def __init__(self, base_dir: Path = Path("outputs")):
        self.base_dir = base_dir
        self._checkpoints: Dict[str, CompanyCheckpoint] = {}
        self._lock = asyncio.Lock()

    async def get(self, hojin_id: str) -> CompanyCheckpoint:
        async with self._lock:
            if hojin_id not in self._checkpoints:
                cp = CompanyCheckpoint(hojin_id)
                await cp.load()
                self._checkpoints[hojin_id] = cp
            return self._checkpoints[hojin_id]

    async def mark_global_start(self) -> None:
        self.session_path = self.base_dir / "session_checkpoint.json"
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "started_at": datetime.utcnow().isoformat(),
            "companies": [],
        }
        self.session_path.write_text(json.dumps(meta, indent=2))

    async def append_company(self, hojin_id: str) -> None:
        self.session_path = self.base_dir / "session_checkpoint.json"
        try:
            data = json.loads(self.session_path.read_text())
        except Exception:
            data = {"started_at": datetime.utcnow().isoformat(), "companies": []}
        if hojin_id not in data["companies"]:
            data["companies"].append(hojin_id)
        self.session_path.write_text(json.dumps(data, indent=2))

    async def summary(self) -> Dict[str, Any]:
        """Return a summary of all checkpoints."""
        out = {}
        for hid, cp in self._checkpoints.items():
            out[hid] = {
                "done": cp.data["urls_done"],
                "failed": cp.data["urls_failed"],
                "total": cp.data["urls_total"],
                "ratio": cp.progress_ratio(),
                "finished": cp.is_finished(),
            }
        return out
