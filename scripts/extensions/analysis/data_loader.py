from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from extensions.io.output_paths import ensure_output_root
from extensions.io.load_source import (
    DEFAULT_INDUSTRY_FALLBACK_PATH,
    IndustryEnrichmentConfig,
    load_industry_lookup_tables,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompanyPaths:
    crawl_meta_path: Path
    url_index_path: Path
    company_profile_path: Path
    markdown_dir: Path
    html_dir: Path
    product_dir: Path


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    company_id: str
    company_dir: Path
    root_url: str | None

    # Industry fields
    industry: Optional[int]  # lv1 key
    nace: Optional[int]  # lv2 detail key
    industry_label_lv1: Optional[str]
    industry_label_lv2: Optional[str]
    industry_label_source: Optional[str]

    # This is what grouping/view code uses (depends on selected mode)
    industry_group_key: str

    crawl_meta: Optional[dict]
    url_index: Optional[dict]
    company_profile: Optional[dict]
    paths: CompanyPaths


@dataclass(frozen=True, slots=True)
class AnalysisDataset:
    output_root: Path
    crawl_global_state: Optional[dict]
    industry_tables: Optional[Any]
    companies: List[CompanyRecord]


def _read_json_if_exists(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed reading JSON: %s", path)
        return None


def _looks_like_company_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    candidates = [
        p / "metadata" / "crawl_meta.json",
        p / "metadata" / "url_index.json",
        p / "markdown",
        p / "html",
        p / "product",
    ]
    return any(x.exists() for x in candidates)


def _parse_int(x: object) -> Optional[int]:
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return None


def _clean_str(x: object) -> Optional[str]:
    if isinstance(x, str):
        s = x.strip()
        return s or None
    return None


def _resolve_lv2_from_crawl_meta(
    crawl_meta: Optional[dict],
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Return (nace, industry_label_lv2, industry_label_source) when present.
    lv2 label is only considered valid when crawl_meta indicates it is from "industry+nace".
    """
    if not crawl_meta:
        return None, None, None

    nace = _parse_int(crawl_meta.get("nace"))
    src = _clean_str(crawl_meta.get("industry_label_source"))
    lbl = _clean_str(crawl_meta.get("industry_label"))

    # In your example:
    #   industry_label_source == "industry+nace"
    #   industry_label == "Computer programming activities"
    # => that's lv2 label (nace-resolved)
    if src == "industry+nace" and nace is not None and lbl:
        return nace, lbl, src

    # If nace exists but no label/source, still expose nace; label remains None.
    if nace is not None:
        return nace, None, src

    return None, None, src


def _resolve_lv1_label_from_crawl_meta(crawl_meta: Optional[dict]) -> Optional[str]:
    """
    If crawl_meta already contains an lv1 industry label (not the nace-derived lv2),
    return it. We only trust it when the source indicates lv1.
    """
    if not crawl_meta:
        return None

    src = _clean_str(crawl_meta.get("industry_label_source"))
    lbl = _clean_str(crawl_meta.get("industry_label"))

    # We should NOT treat "industry+nace" as lv1 label.
    if not lbl:
        return None

    # Accept a few plausible lv1 sources used across versions.
    # (If your pipeline only ever writes "industry+nace", this will be None, and we’ll
    #  fall back to the lookup tables.)
    if src in {"industry", "fallback", "industry_only", "lv1"}:
        return lbl

    return None


def _resolve_lv1_label_from_tables(
    industry_tables: Optional[Any],
    *,
    industry: Optional[int],
    nace: Optional[int],
) -> Optional[str]:
    """
    Robustly resolve lv1 label using IndustryLookupTables / enrichment tables.
    We try:
      - resolve_industry_label with multiple call patterns
      - known dict-like attributes
    """
    if industry is None or industry_tables is None:
        return None

    resolver = getattr(industry_tables, "resolve_industry_label", None)
    if callable(resolver):
        # Try the most common signatures seen in codebases.
        # 1) resolve_industry_label(industry, nace)
        try:
            v = resolver(industry, nace)  # type: ignore[misc]
            s = _clean_str(v)
            if s:
                return s
        except Exception:
            pass

        # 2) resolve_industry_label(industry, None)
        try:
            v = resolver(industry, None)  # type: ignore[misc]
            s = _clean_str(v)
            if s:
                return s
        except Exception:
            pass

        # 3) resolve_industry_label(industry_id=..., nace=...)
        try:
            v = resolver(industry_id=industry, nace=nace)  # type: ignore[call-arg]
            s = _clean_str(v)
            if s:
                return s
        except Exception:
            pass

        # 4) resolve_industry_label(industry_id=...)
        try:
            v = resolver(industry_id=industry)  # type: ignore[call-arg]
            s = _clean_str(v)
            if s:
                return s
        except Exception:
            pass

    # Dict-like fallbacks (wider net than before)
    dict_attrs = (
        "industry_id_to_label",
        "industry_to_label",
        "industry_labels",
        "industry_map",
        "fallback_industry_id_to_label",
        "fallback_industry_to_label",
    )
    for attr in dict_attrs:
        m = getattr(industry_tables, attr, None)
        if isinstance(m, dict):
            v = m.get(industry)
            s = _clean_str(v)
            if s:
                return s

    # Some implementations keep tables nested: industry_tables.fallback.industry_id_to_label
    fallback = getattr(industry_tables, "fallback", None)
    if fallback is not None:
        for attr in ("industry_id_to_label", "industry_to_label", "labels", "map"):
            m = getattr(fallback, attr, None)
            if isinstance(m, dict):
                v = m.get(industry)
                s = _clean_str(v)
                if s:
                    return s

    return None


def load_analysis_dataset(
    *, out_dir: str | Path, max_companies: Optional[int] = None
) -> AnalysisDataset:
    out_dir_p = Path(out_dir)
    output_root = ensure_output_root(out_dir_p)

    crawl_global_state = _read_json_if_exists(output_root / "crawl_global_state.json")

    industry_tables: Optional[Any] = None
    try:
        # You set nace_path=None, but load_industry_lookup_tables() apparently uses
        # its own DEFAULT_NACE_PATH internally (as shown in your log).
        cfg = IndustryEnrichmentConfig(
            nace_path=None,
            fallback_path=DEFAULT_INDUSTRY_FALLBACK_PATH,
            enabled=True,
        )
        industry_tables = load_industry_lookup_tables(cfg)
    except Exception:
        logger.exception(
            "Failed to load industry tables from %s", DEFAULT_INDUSTRY_FALLBACK_PATH
        )
        industry_tables = None

    global_company_profile_path = output_root / "company_profile.json"
    global_profile = _read_json_if_exists(global_company_profile_path)

    dirs = sorted(
        [p for p in output_root.iterdir() if _looks_like_company_dir(p)],
        key=lambda p: p.name,
    )
    if max_companies is not None:
        dirs = dirs[:max_companies]

    companies: List[CompanyRecord] = []
    for cdir in dirs:
        company_id = cdir.name

        crawl_meta_path = cdir / "metadata" / "crawl_meta.json"
        url_index_path = cdir / "metadata" / "url_index.json"
        crawl_meta = _read_json_if_exists(crawl_meta_path)
        url_index = _read_json_if_exists(url_index_path)

        per_company_profile_path = cdir / "company_profile.json"
        per_company_profile = _read_json_if_exists(per_company_profile_path)
        company_profile = per_company_profile or global_profile

        root_url: str | None = None
        if crawl_meta and isinstance(crawl_meta.get("root_url"), str):
            root_url = str(crawl_meta.get("root_url") or "") or None
        elif isinstance((company_profile or {}).get("root_url"), str):
            root_url = str((company_profile or {}).get("root_url") or "") or None

        industry = _parse_int((crawl_meta or {}).get("industry"))
        nace, industry_label_lv2, industry_label_source = _resolve_lv2_from_crawl_meta(
            crawl_meta
        )

        # lv1 label resolution:
        #   1) trust crawl_meta only if it explicitly says it is lv1
        #   2) otherwise resolve from tables (industry.ods)
        industry_label_lv1 = _resolve_lv1_label_from_crawl_meta(crawl_meta)
        if industry_label_lv1 is None:
            industry_label_lv1 = _resolve_lv1_label_from_tables(
                industry_tables,
                industry=industry,
                nace=nace,
            )

        paths = CompanyPaths(
            crawl_meta_path=crawl_meta_path,
            url_index_path=url_index_path,
            company_profile_path=(
                per_company_profile_path
                if per_company_profile_path.exists()
                else global_company_profile_path
            ),
            markdown_dir=cdir / "markdown",
            html_dir=cdir / "html",
            product_dir=cdir / "product",
        )

        # Default group key is lv1-ish; industry_view.py may override grouping later.
        if industry is not None:
            group_key = f"industry:{industry}"
        else:
            group_key = "industry:unknown"

        companies.append(
            CompanyRecord(
                company_id=company_id,
                company_dir=cdir,
                root_url=root_url,
                industry=industry,
                nace=nace,
                industry_label_lv1=industry_label_lv1,
                industry_label_lv2=industry_label_lv2,
                industry_label_source=industry_label_source,
                industry_group_key=group_key,
                crawl_meta=crawl_meta,
                url_index=url_index,
                company_profile=company_profile,
                paths=paths,
            )
        )

    # Helpful one-line summary in loader logs (keeps analysis logs simpler)
    total = len(companies)
    have_ind = sum(1 for c in companies if isinstance(c.industry, int))
    have_lv1 = sum(
        1
        for c in companies
        if isinstance(c.industry_label_lv1, str) and c.industry_label_lv1.strip()
    )
    have_nace = sum(1 for c in companies if isinstance(c.nace, int))
    have_lv2 = sum(
        1
        for c in companies
        if isinstance(c.industry_label_lv2, str) and c.industry_label_lv2.strip()
    )
    logger.info(
        "[analysis_loader] companies=%d have_industry=%d have_lv1_label=%d have_nace=%d have_lv2_label=%d",
        total,
        have_ind,
        have_lv1,
        have_nace,
        have_lv2,
    )

    return AnalysisDataset(
        output_root=output_root,
        crawl_global_state=crawl_global_state,
        industry_tables=industry_tables,
        companies=companies,
    )
