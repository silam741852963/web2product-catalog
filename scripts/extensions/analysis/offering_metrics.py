from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils_stats import summarize_distribution


@dataclass(frozen=True, slots=True)
class CompanyOfferingInput:
    company_id: str
    industry_group_key: str
    company_profile: dict | None


@dataclass(frozen=True, slots=True)
class CompanyOfferingOutput:
    company_id: str
    industry_group_key: str
    product: int
    service: int
    total: int
    unique_total: int  # best-effort uniqueness count (if extractable)


def _offering_counts_from_profile(profile: dict | None) -> Tuple[int, int, int]:
    """
    Returns (product_count, service_count, unique_total) from company_profile.offerings[*].
    Uniqueness is best-effort: uses (type, name/title/label) if present; otherwise falls back to total.
    """
    if not isinstance(profile, dict):
        return (0, 0, 0)

    offerings = profile.get("offerings")
    if not isinstance(offerings, list):
        return (0, 0, 0)

    p = 0
    s = 0
    uniq = set()

    for o in offerings:
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        if t == "product":
            p += 1
        elif t == "service":
            s += 1

        # best-effort key
        name = ""
        for k in ("name", "title", "label"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                name = v.strip().lower()
                break
        uniq.add((str(t or ""), name))

    total = int(p + s)
    unique_total = int(len([x for x in uniq if x != ("", "")])) if uniq else int(total)
    if unique_total <= 0:
        unique_total = int(total)

    return (p, s, unique_total)


def extract_company_offering_metrics(
    inp: CompanyOfferingInput,
) -> CompanyOfferingOutput:
    prod, serv, uniq_total = _offering_counts_from_profile(inp.company_profile)
    tot = int(prod + serv)
    return CompanyOfferingOutput(
        company_id=inp.company_id,
        industry_group_key=inp.industry_group_key,
        product=int(prod),
        service=int(serv),
        total=int(tot),
        unique_total=int(uniq_total),
    )


def aggregate_offering_sections_for_view(
    *,
    company_ids_in_view: Iterable[str],
    by_company: Dict[str, CompanyOfferingOutput],
    md_tokens_by_company: Dict[str, int],
    llm_tokens_by_company: Dict[str, int],
    # Optional extras (don’t break old callers)
    total_pages_by_company: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Adds scatter-ready metrics:
      - offering_count (total)
      - offering_density = total / max(1,total_pages) (if total_pages_by_company provided)
      - offering_per_md_token = total / max(1, md_tokens)
      - offering_per_llm_token = total / max(1, llm_tokens)
      - unique_offering_ratio = unique_total / max(1,total)
    """
    ids = list(company_ids_in_view)

    rows: List[CompanyOfferingOutput] = []
    for cid in ids:
        r = by_company.get(cid)
        if r is not None:
            rows.append(r)

    # stable ordering by company_id (for tables), but vectors must align to ids (view order)
    rows_by_id: Dict[str, CompanyOfferingOutput] = {r.company_id: r for r in rows}

    # per-company vectors aligned to ids
    offering_count: List[int] = []
    offering_density: List[float] = []
    offering_per_md_token: List[float] = []
    offering_per_llm_token: List[float] = []
    unique_offering_ratio: List[float] = []

    for cid in ids:
        r = rows_by_id.get(cid)
        tot = int(r.total) if r else 0
        uniq = int(r.unique_total) if r else 0

        md_tok = int(md_tokens_by_company.get(cid, 0) or 0)
        llm_tok = int(llm_tokens_by_company.get(cid, 0) or 0)
        pages = int((total_pages_by_company or {}).get(cid, 0) or 0)

        offering_count.append(tot)
        offering_per_md_token.append(tot / max(1.0, float(md_tok)))
        offering_per_llm_token.append(tot / max(1.0, float(llm_tok)))
        unique_offering_ratio.append(float(uniq) / max(1.0, float(tot)))

        if total_pages_by_company is not None:
            offering_density.append(float(tot) / max(1.0, float(pages)))
        else:
            offering_density.append(0.0)

    # Aggregate by industry within the view (for stacked bars)
    by_industry: Dict[str, Dict[str, int]] = {}
    for r in rows:
        agg = by_industry.setdefault(
            r.industry_group_key, {"product": 0, "service": 0, "total": 0}
        )
        agg["product"] += int(r.product)
        agg["service"] += int(r.service)
        agg["total"] += int(r.total)

    per_industry_rows = [
        {
            "industry": k,
            "product": int(v["product"]),
            "service": int(v["service"]),
            "total": int(v["total"]),
        }
        for k, v in sorted(by_industry.items(), key=lambda kv: kv[0])
    ]

    # Table-ish rows (sorted by company_id for readability)
    rows_sorted = sorted(rows, key=lambda r: r.company_id)
    per_company_rows = [
        {
            "company_id": r.company_id,
            "product": int(r.product),
            "service": int(r.service),
            "total": int(r.total),
            "unique_total": int(r.unique_total),
        }
        for r in rows_sorted
    ]

    return {
        "offerings": {
            "company_ids_in_view": ids,
            "per_company": {
                "rows": per_company_rows,
                "summary_total": summarize_distribution(
                    [float(x) for x in offering_count], include_minmax=True
                ),
                "charts": {
                    "stacked_bar": "offering_company_stacked_bar",
                    "hist_total": "offering_total_hist",
                },
            },
            "per_industry": {
                "rows": per_industry_rows,
                "charts": {"stacked_bar": "offering_industry_stacked_bar"},
            },
            "features": {
                "offering_count": offering_count,
                "offering_density": offering_density,  # 0.0 if pages not provided
                "offering_per_md_token": offering_per_md_token,
                "offering_per_llm_token": offering_per_llm_token,
                "unique_offering_ratio": unique_offering_ratio,
                "summaries": {
                    "offering_per_md_token": summarize_distribution(
                        offering_per_md_token, include_minmax=False
                    ),
                    "offering_per_llm_token": summarize_distribution(
                        offering_per_llm_token, include_minmax=False
                    ),
                    "unique_offering_ratio": summarize_distribution(
                        unique_offering_ratio, include_minmax=False
                    ),
                    "offering_density": summarize_distribution(
                        offering_density, include_minmax=False
                    ),
                },
            },
            "scatter_candidates": [
                {
                    "id": "md_tokens_vs_offering_count",
                    "x": "md_tokens",
                    "y": "offering_count",
                    "chart_id": "scatter_md_tokens_offering_count",
                },
                {
                    "id": "llm_tokens_vs_offering_count",
                    "x": "llm_tokens",
                    "y": "offering_count",
                    "chart_id": "scatter_llm_tokens_offering_count",
                },
                {
                    "id": "md_tokens_vs_offering_efficiency",
                    "x": "md_tokens",
                    "y": "offering_per_md_token",
                    "chart_id": "scatter_md_tokens_offering_efficiency",
                },
            ],
        }
    }
