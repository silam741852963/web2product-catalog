from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .error_utils import error_signature, top_k_pairs, truncate_label
from .industry_views import IndustryView
from .utils_stats import summarize_distribution


_CRAWL_META_ERROR_KEYS: Tuple[str, ...] = (
    "error",
    "error_text",
    "last_error",
    "exception",
    "exception_text",
    "failure_reason",
    "fail_reason",
    "reason",
    "traceback",
    "errors",  # sometimes list-like
)

_URL_ENTRY_ERROR_KEYS: Tuple[str, ...] = (
    "error",
    "error_text",
    "exception",
    "failure_reason",
    "fail_reason",
    "reason",
    "status_reason",
    "traceback",
)

_ERROR_STATUSES: Tuple[str, ...] = (
    "error",
    "failed",
    "crawl_failed",
    "fetch_failed",
    "http_error",
    "blocked",
)


def _collect_error_texts_from_any(v: object) -> List[str]:
    out: List[str] = []
    if v is None:
        return out

    if isinstance(v, str):
        t = v.strip()
        if t:
            out.append(t)
        return out

    if isinstance(v, dict):
        for k in ("message", "error", "error_text", "exception", "reason", "traceback"):
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                out.append(vv.strip())
        return out

    if isinstance(v, list):
        for item in v:
            out.extend(_collect_error_texts_from_any(item))
        return out

    return out


def _extract_company_error_texts(
    crawl_meta: dict | None, url_index: dict | None
) -> List[str]:
    texts: List[str] = []

    if isinstance(crawl_meta, dict):
        for k in _CRAWL_META_ERROR_KEYS:
            if k in crawl_meta:
                texts.extend(_collect_error_texts_from_any(crawl_meta.get(k)))

    if isinstance(url_index, dict):
        for url, entry in url_index.items():
            if url == "__meta__":
                continue
            if not isinstance(entry, dict):
                continue

            status = entry.get("status")
            status_is_error = (
                isinstance(status, str) and status.strip().lower() in _ERROR_STATUSES
            )
            has_err_field = any(
                isinstance(entry.get(k), str) and str(entry.get(k)).strip()
                for k in _URL_ENTRY_ERROR_KEYS
            )

            if status_is_error or has_err_field:
                for k in _URL_ENTRY_ERROR_KEYS:
                    texts.extend(_collect_error_texts_from_any(entry.get(k)))

    # Dedup deterministic
    seen = set()
    uniq: List[str] = []
    for t in texts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


@dataclass(frozen=True, slots=True)
class CompanyErrorInput:
    company_id: str
    crawl_meta: dict | None
    url_index: dict | None


@dataclass(frozen=True, slots=True)
class CompanyErrorOutput:
    company_id: str
    sig_counts: Counter[str]
    error_count_total: int


def extract_company_error_metrics(inp: CompanyErrorInput) -> CompanyErrorOutput:
    texts = _extract_company_error_texts(inp.crawl_meta, inp.url_index)

    sigs: Counter[str] = Counter()
    for t in texts:
        sig = error_signature(t)
        if sig:
            sigs[sig] += 1

    total = int(sum(sigs.values()))
    return CompanyErrorOutput(
        company_id=str(inp.company_id),
        sig_counts=sigs,
        error_count_total=int(total),
    )


def aggregate_error_sections_for_view(
    *,
    company_ids_in_view: Iterable[str],
    by_company: Dict[str, CompanyErrorOutput],
    k_top: int = 20,
    label_max_len: int = 60,
) -> Dict[str, Any]:
    ids = list(company_ids_in_view)

    all_sigs: Counter[str] = Counter()
    errors_per_company: List[int] = []

    # Optional: per-company row objects (useful for hover)
    per_company_rows: List[Dict[str, Any]] = []

    for cid in ids:
        r = by_company.get(cid)
        if r is None:
            errors_per_company.append(0)
            per_company_rows.append(
                {"company_id": cid, "error_count_total": 0, "top_signature": ""}
            )
            continue

        errors_per_company.append(int(r.error_count_total))
        all_sigs.update(r.sig_counts)

        top_sig = ""
        if r.sig_counts:
            items = sorted(
                r.sig_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
            )
            top_sig = str(items[0][0])

        per_company_rows.append(
            {
                "company_id": r.company_id,
                "error_count_total": int(r.error_count_total),
                "top_signature": top_sig,
            }
        )

    # top signatures with truncated labels
    raw_top = top_k_pairs(all_sigs, k=int(k_top))  # [[sig, count], ...]
    top_sig_rows: List[Dict[str, Any]] = []
    for sig, cnt in raw_top:
        full = str(sig)
        top_sig_rows.append(
            {
                "label": truncate_label(full, max_len=int(label_max_len)),
                "full_label": full,
                "count": int(cnt),
            }
        )

    return {
        "errors": {
            "company_ids_in_view": ids,
            "errors_per_company": errors_per_company,  # aligned vector
            "errors_summary": summarize_distribution(
                [float(x) for x in errors_per_company], include_minmax=True
            ),
            "per_company": {
                "rows": per_company_rows,  # stable in view order
            },
            "top_signatures": {
                "k": int(k_top),
                "rows": top_sig_rows,  # [{label, full_label, count}]
            },
            "charts": {
                "errors_total_hist": "errors_total_hist",
                "top_signatures_bar": "errors_top_signatures_bar",
            },
        }
    }


# -------------------------------------------------------------------
# Backwards-compatible wrapper (kept)
# -------------------------------------------------------------------
def compute_error_sections(view: IndustryView, *, k_top: int = 20) -> Dict[str, Any]:
    by_company: Dict[str, CompanyErrorOutput] = {}
    for c in view.companies:
        inp = CompanyErrorInput(
            company_id=c.company_id,
            crawl_meta=getattr(c, "crawl_meta", None),
            url_index=getattr(c, "url_index", None),
        )
        by_company[c.company_id] = extract_company_error_metrics(inp)

    return aggregate_error_sections_for_view(
        company_ids_in_view=[c.company_id for c in view.companies],
        by_company=by_company,
        k_top=int(k_top),
    )
