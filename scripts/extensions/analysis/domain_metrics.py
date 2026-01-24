from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .domain_utils import (
    extract_site_key,
    normalized_site_key,
    root_site_key_from_url,
    top_k_pairs,
)
from .industry_views import IndustryView
from .utils_stats import summarize_distribution


def _iter_url_index_urls(url_index: dict | None) -> List[str]:
    if not isinstance(url_index, dict):
        return []
    return [str(u) for u in url_index.keys() if u != "__meta__"]


def _root_site_for_company(crawl_meta: dict | None, fallback_root_url: str) -> str:
    if isinstance(crawl_meta, dict):
        ru = crawl_meta.get("root_url")
        if isinstance(ru, str) and ru.strip():
            return ru.strip()
    return (fallback_root_url or "").strip()


@dataclass(frozen=True, slots=True)
class CompanyDomainInput:
    company_id: str
    root_url: str
    crawl_meta: dict | None
    url_index: dict | None


@dataclass(frozen=True, slots=True)
class CompanyDomainOutput:
    company_id: str

    # Required for scatter/debug
    total_pages: int
    offsite_pages: int
    offsite_pct: float

    # Convenience counts (onsite pages)
    onsite_pages: int

    # Offsite domain frequency (for top-k)
    offsite_domains: Counter[str]


def extract_company_domain_metrics(inp: CompanyDomainInput) -> CompanyDomainOutput:
    """
    Deterministic offsite extraction:
      - total_pages comes from url_index URL keys (excluding __meta__)
      - offsite_pages counts urls whose normalized site key != root site key
      - root site key normalization is IDENTICAL to top offsite domain normalization
    """
    urls = _iter_url_index_urls(inp.url_index)

    root_url = _root_site_for_company(inp.crawl_meta, inp.root_url)
    root_key = root_site_key_from_url(root_url)

    total_pages = int(len(urls))
    offsite_pages = 0
    onsite_pages = 0

    offsite_domains: Counter[str] = Counter()

    for u in urls:
        u_key = normalized_site_key(u)
        if not u_key:
            # treat non-http/invalid as offsite (conservative)
            offsite_pages += 1
            offsite_domains["(non-http)"] += 1
            continue

        if root_key and (u_key == root_key):
            onsite_pages += 1
        else:
            offsite_pages += 1
            offsite_domains[extract_site_key(u)] += 1

    den = max(1, total_pages)
    offsite_pct = round((offsite_pages / den) * 100.0, 4)

    return CompanyDomainOutput(
        company_id=str(inp.company_id),
        total_pages=int(total_pages),
        offsite_pages=int(offsite_pages),
        offsite_pct=float(offsite_pct),
        onsite_pages=int(onsite_pages),
        offsite_domains=offsite_domains,
    )


def aggregate_domain_sections_for_view(
    *,
    company_ids_in_view: Iterable[str],
    by_company: Dict[str, CompanyDomainOutput],
    k_top: int = 20,
) -> Dict[str, Any]:
    """
    IMPORTANT: vectors are aligned with company_ids_in_view order.
    This avoids the "0 everywhere" bug and makes scatter wiring trivial.
    """
    ids = list(company_ids_in_view)

    offsite_pct_by_company: List[float] = []
    offsite_pages_by_company: List[int] = []
    total_pages_by_company: List[int] = []

    all_offsite_domains: Counter[str] = Counter()

    for cid in ids:
        r = by_company.get(cid)
        if r is None:
            offsite_pct_by_company.append(0.0)
            offsite_pages_by_company.append(0)
            total_pages_by_company.append(0)
            continue

        offsite_pct_by_company.append(float(r.offsite_pct))
        offsite_pages_by_company.append(int(r.offsite_pages))
        total_pages_by_company.append(int(r.total_pages))
        all_offsite_domains.update(r.offsite_domains)

    top_rows = top_k_pairs(all_offsite_domains, k=int(k_top))

    return {
        "domains": {
            # aligned vectors (for scatter)
            "company_ids_in_view": ids,
            "offsite_pct_by_company": offsite_pct_by_company,
            "offsite_pages_by_company": offsite_pages_by_company,
            "total_pages_by_company": total_pages_by_company,
            # summary of just the pct values
            "offsite_pct_summary": summarize_distribution(
                [float(x) for x in offsite_pct_by_company],
                include_minmax=True,
            ),
            # top-k bar input
            "top_offsite_domains": {
                "k": int(k_top),
                "rows": top_rows,  # [[domain, count], ...]
            },
            "charts": {
                "offsite_pct_hist": "offsite_pct_hist",
                "top_offsite_domains_bar": "top_offsite_domains_bar",
            },
        }
    }


# -------------------------------------------------------------------
# Backwards-compatible wrapper (kept)
# -------------------------------------------------------------------
def compute_domain_sections(view: IndustryView, *, k_top: int = 20) -> Dict[str, Any]:
    by_company: Dict[str, CompanyDomainOutput] = {}
    for c in view.companies:
        inp = CompanyDomainInput(
            company_id=c.company_id,
            root_url=getattr(c, "root_url", "") or "",
            crawl_meta=getattr(c, "crawl_meta", None),
            url_index=getattr(c, "url_index", None),
        )
        by_company[c.company_id] = extract_company_domain_metrics(inp)

    return aggregate_domain_sections_for_view(
        company_ids_in_view=[c.company_id for c in view.companies],
        by_company=by_company,
        k_top=int(k_top),
    )
