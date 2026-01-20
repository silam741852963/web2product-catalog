from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from configs.models import COMPANY_STATUS_PENDING, COMPANY_STATUS_MD_NOT_DONE

VERSION_META_KEY = "version_metadata"

TERMINAL_RECONCILE_FROM: Set[str] = {
    COMPANY_STATUS_PENDING,
    "in_progress",
    COMPANY_STATUS_MD_NOT_DONE,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_norm_company_id_list(xs: Optional[Iterable[str]]) -> List[str]:
    if not xs:
        return []
    out: List[str] = []
    for x in xs:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        out.append(s)

    # dedup while preserving deterministic order
    seen: set[str] = set()
    uniq: List[str] = []
    for cid in out:
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(cid)
    return uniq


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    company_id: str
    db_snapshot: Any  # configs.models.Company
    crawl_meta: Dict[str, Any]
    url_index_meta: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    out_dir: str
    db_path: str
    touched_companies: int
    wrote_global_state: bool
    source_companies_loaded: int
    source_companies_used: int
    sample_before: CalibrationSample
    sample_after: CalibrationSample


@dataclass(frozen=True, slots=True)
class CorruptJsonFile:
    company_id: str
    path: str
    size_bytes: int
    reason: str
    head_bytes_hex: str
    head_text_preview: str


@dataclass(frozen=True, slots=True)
class CorruptionReport:
    out_dir: str
    db_path: str
    scanned_companies: int
    affected_companies: int
    affected_files: int
    examples: List[CorruptJsonFile]


@dataclass(frozen=True, slots=True)
class CorruptionFixReport:
    out_dir: str
    db_path: str
    dry_run: bool
    scanned_companies: int
    affected_companies: int
    quarantined_files: int
    marked_pending: int
    run_done_unmarked: int
    examples: List[CorruptJsonFile]


# -----------------------------------------------------------------------------
# Reconcile / Reset reports + selection
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    out_dir: str
    db_path: str
    dry_run: bool
    scanned_companies: int
    upgraded_to_terminal_done: int
    invariant_errors: int
    wrote_global_state: bool
    example_company_ids: List[str]


@dataclass(frozen=True, slots=True)
class ResetCandidate:
    company_id: str
    reasons: List[str]


@dataclass(frozen=True, slots=True)
class ResetReport:
    out_dir: str
    db_path: str
    dry_run: bool
    targets: List[str]
    scanned_companies: int
    selected_companies: int

    deleted_dirs: int
    missing_dirs: int
    db_rows_reset: int
    run_done_rows_deleted: int

    # NEW: persistent retry-store cleanup (out_dir/_retry/*.json)
    retry_quarantine_rows_deleted: int
    retry_state_rows_deleted: int

    wrote_global_state: bool
    candidates: List[ResetCandidate]


# -----------------------------------------------------------------------------
# Enforce max pages (post-crawl hard cap + deterministic suppression)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnforcedCompany:
    company_id: str
    max_pages: int
    candidates: int
    kept: int
    overflow: int
    moved_md: int
    moved_html: int
    skipped_reason: Optional[str]
    example_overflow_urls: List[str]


@dataclass(frozen=True, slots=True)
class EnforceMaxPagesReport:
    out_dir: str
    db_path: str
    dry_run: bool
    selector: str
    scanned_companies: int
    applied_companies: int
    skipped_companies: int
    total_candidates: int
    total_kept: int
    total_overflow: int
    wrote_global_state: bool
    companies: List[EnforcedCompany]
