from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from configs.models import Company

from .json_io import write_json_file
from .paths import crawl_meta_candidates, first_existing
from .types import VERSION_META_KEY


def patch_crawl_meta_file(
    out_dir: Path,
    company_id: str,
    *,
    db_snap: Company,
    version_meta: Dict[str, Any],
) -> None:
    p = first_existing(crawl_meta_candidates(out_dir, company_id))
    if p is None:
        return

    meta: Dict[str, Any] = {}

    meta["company_id"] = db_snap.company_id
    meta["root_url"] = db_snap.root_url
    meta["name"] = db_snap.name

    md = db_snap.metadata if isinstance(db_snap.metadata, dict) else {}
    meta["metadata"] = md

    meta["industry"] = db_snap.industry
    meta["nace"] = db_snap.nace
    meta["industry_label"] = db_snap.industry_label
    meta["industry_label_source"] = db_snap.industry_label_source

    meta["status"] = db_snap.status
    meta["crawl_finished"] = bool(db_snap.crawl_finished)

    meta["urls_total"] = int(db_snap.urls_total or 0)
    meta["urls_markdown_done"] = int(db_snap.urls_markdown_done or 0)
    meta["urls_llm_done"] = int(db_snap.urls_llm_done or 0)

    meta["created_at"] = db_snap.created_at
    meta["updated_at"] = db_snap.updated_at
    meta["last_crawled_at"] = db_snap.last_crawled_at

    meta["retry_cls"] = db_snap.retry_cls
    meta["retry_attempts"] = int(db_snap.retry_attempts or 0)
    meta["retry_next_eligible_at"] = float(db_snap.retry_next_eligible_at or 0.0)
    meta["retry_updated_at"] = float(db_snap.retry_updated_at or 0.0)
    meta["retry_last_error"] = db_snap.retry_last_error or ""
    meta["retry_last_stage"] = db_snap.retry_last_stage or ""

    meta["retry_net_attempts"] = int(db_snap.retry_net_attempts or 0)
    meta["retry_stall_attempts"] = int(db_snap.retry_stall_attempts or 0)
    meta["retry_mem_attempts"] = int(db_snap.retry_mem_attempts or 0)
    meta["retry_other_attempts"] = int(db_snap.retry_other_attempts or 0)

    meta["retry_mem_hits"] = int(db_snap.retry_mem_hits or 0)
    meta["retry_last_stall_kind"] = db_snap.retry_last_stall_kind or "unknown"

    meta["retry_last_progress_md_done"] = int(db_snap.retry_last_progress_md_done or 0)
    meta["retry_last_seen_md_done"] = int(db_snap.retry_last_seen_md_done or 0)

    meta["retry_last_error_sig"] = db_snap.retry_last_error_sig or ""
    meta["retry_same_error_streak"] = int(db_snap.retry_same_error_streak or 0)
    meta["retry_last_error_sig_updated_at"] = float(
        db_snap.retry_last_error_sig_updated_at or 0.0
    )

    meta[VERSION_META_KEY] = version_meta
    meta["max_pages"] = db_snap.max_pages

    write_json_file(p, meta, pretty=True)
