from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from configs.models import URL_INDEX_META_KEY
from .paths import crawl_meta_candidates, first_existing, url_index_candidates
from .json_io import read_json_file
from .types import CalibrationSample

if TYPE_CHECKING:
    from extensions.crawl.state import CrawlState


def pick_sample_company_id(state: "CrawlState", requested: Optional[str]) -> str:
    if requested is not None and requested.strip():
        return requested.strip()

    row = (
        sqlite3.connect(str(state.db_path))
        .execute("SELECT company_id FROM companies ORDER BY company_id ASC LIMIT 1")
        .fetchone()
    )
    if row is None:
        raise RuntimeError("No companies found in DB to sample.")
    return str(row[0])


async def sample(
    out_dir: Path, state: "CrawlState", company_id: str
) -> CalibrationSample:
    snap = await state.get_company_snapshot(company_id, recompute=False)

    cm_path = first_existing(crawl_meta_candidates(out_dir, company_id))
    cm = read_json_file(cm_path) if cm_path else None

    idx_path = first_existing(url_index_candidates(out_dir, company_id))
    idx = read_json_file(idx_path) if idx_path else None

    idx_meta: Dict[str, Any] = {}
    if isinstance(idx, dict):
        raw_meta = idx.get(URL_INDEX_META_KEY)
        if isinstance(raw_meta, dict):
            idx_meta = raw_meta

    return CalibrationSample(
        company_id=company_id,
        db_snapshot=snap,
        crawl_meta=(cm if isinstance(cm, dict) else {}),
        url_index_meta=(idx_meta if isinstance(idx_meta, dict) else {}),
    )
