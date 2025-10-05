#!/usr/bin/env python3
"""
Resume the US batch scraper from checkpoints + per-company frontier.
This is a convenience wrapper around run_scraper with resume on.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_scraper import main as run_main  # reuse same entrypoint

if __name__ == "__main__":
    run_main()
