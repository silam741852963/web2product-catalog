from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .data_loader import AnalysisDataset, CompanyRecord


@dataclass(frozen=True, slots=True)
class IndustryView:
    key: str
    label: str
    companies: List[CompanyRecord]


def _lv1_key(industry: Optional[int]) -> str:
    return f"industry:{industry}" if industry is not None else "industry:unknown"


def _lv2_key(industry: Optional[int], nace: Optional[int]) -> str:
    i = industry if industry is not None else "unknown"
    n = nace if nace is not None else "unknown"
    return f"industry:{i}|nace:{n}"


def _lv1_label(c: CompanyRecord) -> str:
    if c.industry is None:
        return "(unclassified)"
    if c.industry_label_lv1:
        return f"{c.industry} — {c.industry_label_lv1}"
    return str(c.industry)


def _lv2_label(c: CompanyRecord) -> str:
    # Prefer crawl_meta-derived industry_label when source is industry+nace
    if c.industry is None or c.nace is None:
        return "(unclassified lv2)"
    if c.industry_label_lv2:
        return f"{c.industry}.{c.nace} — {c.industry_label_lv2}"
    # fallback: show ids only
    return f"{c.industry}.{c.nace}"


def build_industry_views(
    ds: AnalysisDataset,
    *,
    mode: str = "global",  # global | lv1 | lv1-all | lv2-all
    industry_id: Optional[int] = None,
) -> Dict[str, IndustryView]:
    """
    Default: only global view.
    mode=lv1: only one lv1 industry view (requires industry_id)
    mode=lv1-all: all lv1 views
    mode=lv2-all: all lv2 views
    """
    companies = sorted(ds.companies, key=lambda x: x.company_id)

    if mode == "global":
        return {
            "__global__": IndustryView(
                key="__global__",
                label="Global",
                companies=companies,
            )
        }

    if mode == "lv1":
        if industry_id is None:
            raise ValueError("mode=lv1 requires industry_id")
        subset = [c for c in companies if c.industry == industry_id]
        # Deterministic, even if empty
        label = f"{industry_id}"
        # Try to find a label from any matching company (lv1 table-derived)
        for c in subset:
            if c.industry_label_lv1:
                label = f"{industry_id} — {c.industry_label_lv1}"
                break
        return {
            _lv1_key(industry_id): IndustryView(
                key=_lv1_key(industry_id),
                label=label,
                companies=subset,
            )
        }

    if mode == "lv1-all":
        groups: Dict[str, List[CompanyRecord]] = {}
        labels: Dict[str, str] = {}

        for c in companies:
            k = _lv1_key(c.industry)
            groups.setdefault(k, []).append(c)
            # set a stable label for this group
            if k not in labels:
                labels[k] = _lv1_label(c)

        for k in list(groups.keys()):
            groups[k] = sorted(groups[k], key=lambda x: x.company_id)

        return {
            k: IndustryView(key=k, label=labels.get(k, k), companies=groups[k])
            for k in sorted(groups.keys())
        }

    if mode == "lv2-all":
        groups2: Dict[str, List[CompanyRecord]] = {}
        labels2: Dict[str, str] = {}

        for c in companies:
            # Only include true lv2 when nace exists (keyed by industry+nace)
            if c.industry is None or c.nace is None:
                continue
            k = _lv2_key(c.industry, c.nace)
            groups2.setdefault(k, []).append(c)
            if k not in labels2:
                labels2[k] = _lv2_label(c)

        for k in list(groups2.keys()):
            groups2[k] = sorted(groups2[k], key=lambda x: x.company_id)

        return {
            k: IndustryView(key=k, label=labels2.get(k, k), companies=groups2[k])
            for k in sorted(groups2.keys())
        }

    raise ValueError(f"Unknown mode: {mode}")
