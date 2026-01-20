from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from configs.models import Company
from extensions.io.load_source import (
    DEFAULT_INDUSTRY_FALLBACK_PATH,
    DEFAULT_NACE_INDUSTRY_PATH,
    IndustryEnrichmentConfig,
    load_companies_from_source_with_industry,
)


def _industry_cfg(
    *,
    industry_nace_path: Optional[Path],
    industry_fallback_path: Optional[Path],
) -> IndustryEnrichmentConfig:
    return IndustryEnrichmentConfig(
        nace_path=industry_nace_path
        if industry_nace_path is not None
        else DEFAULT_NACE_INDUSTRY_PATH,
        fallback_path=industry_fallback_path
        if industry_fallback_path is not None
        else DEFAULT_INDUSTRY_FALLBACK_PATH,
        enabled=True,
    )


def load_source_company_map(
    *,
    dataset_file: Optional[Path],
    company_file: Optional[Path],
    industry_nace_path: Optional[Path],
    industry_fallback_path: Optional[Path],
) -> Tuple[Dict[str, Company], int]:
    if dataset_file is None and company_file is None:
        return {}, 0

    cfg = _industry_cfg(
        industry_nace_path=industry_nace_path,
        industry_fallback_path=industry_fallback_path,
    )

    paths: List[Path] = []
    if dataset_file is not None:
        paths.append(Path(dataset_file))
    if company_file is not None:
        paths.append(Path(company_file))

    all_companies: List[Company] = []
    for p in paths:
        all_companies.extend(
            load_companies_from_source_with_industry(
                p,
                industry_config=cfg,
                encoding="utf-8",
                limit=None,
                aggregate_same_url=True,
                interleave_domains=True,
            )
        )

    out: Dict[str, Company] = {}
    for c in all_companies:
        out[c.company_id] = c.normalized()

    return out, len(all_companies)


def filter_source_map_to_db(
    *, db_company_ids: List[str], src_map: Mapping[str, Company]
) -> Dict[str, Company]:
    if not src_map:
        return {}
    have = set(db_company_ids)
    return {cid: src_map[cid] for cid in have if cid in src_map}
