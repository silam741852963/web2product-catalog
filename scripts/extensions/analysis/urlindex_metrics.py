from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .industry_views import IndustryView
from .utils_stats import summarize_distribution


def _pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round((num / den) * 100.0, 2)


def _iter_url_entries(url_index: Optional[dict]) -> List[Tuple[str, dict]]:
    if not isinstance(url_index, dict):
        return []
    out: List[Tuple[str, dict]] = []
    for k, v in url_index.items():
        if k == "__meta__":
            continue
        if isinstance(v, dict):
            out.append((str(k), v))
    return out


def _infer_is_suppressed(entry: dict) -> bool:
    status = entry.get("status")
    if isinstance(status, str) and status == "markdown_suppressed":
        return True

    gating_action = entry.get("gating_action")
    if isinstance(gating_action, str) and gating_action != "save":
        return True

    gating_accept = entry.get("gating_accept")
    if gating_accept is False:
        return True

    return False


def _suppressed_reason(entry: dict) -> str:
    for k in ("gating_reason", "gating_action_reason", "gating_action"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def _intish(v: object) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None


def _build_histogram_bins(
    values: List[int], *, target_bins: int = 20
) -> Dict[str, Any]:
    """
    Returns:
      histogram_bins: [{lo, hi, count}, ...] (count>0 only)
      bin_labels: ["lo–hi", ...]
      bin_counts: [count, ...]
    """
    vals: List[int] = []
    for v in values:
        try:
            iv = int(v)
            if iv >= 0:
                vals.append(iv)
        except Exception:
            continue

    if not vals:
        return {"histogram_bins": [], "bin_labels": [], "bin_counts": [], "bin_size": 1}

    vmax = max(vals)
    if vmax <= 0:
        return {
            "histogram_bins": [{"lo": 0, "hi": 0, "count": len(vals)}],
            "bin_labels": ["0"],
            "bin_counts": [len(vals)],
            "bin_size": 1,
        }

    # choose bin size to get ~target_bins
    bin_size = max(1, int(round(vmax / max(1, int(target_bins)))))
    if vmax > 500 and bin_size < 10:
        bin_size = 10
    if vmax > 5000 and bin_size < 50:
        bin_size = 50

    counts: Counter[int] = Counter()
    for v in vals:
        b = int(v // bin_size)
        counts[b] += 1

    bins: List[dict] = []
    labels: List[str] = []
    ys: List[int] = []

    for b in sorted(counts.keys()):
        c = int(counts[b])
        if c <= 0:
            continue
        lo = b * bin_size
        hi = (b + 1) * bin_size - 1
        if b == int(vmax // bin_size):
            hi = max(hi, vmax)

        lab = f"{lo}" if lo == hi else f"{lo}–{hi}"
        bins.append({"lo": int(lo), "hi": int(hi), "count": int(c)})
        labels.append(lab)
        ys.append(int(c))

    return {
        "histogram_bins": bins,
        "bin_labels": labels,
        "bin_counts": ys,
        "bin_size": bin_size,
    }


@dataclass(frozen=True, slots=True)
class CompanyUrlIndexInput:
    company_id: str
    crawl_meta: dict | None
    url_index: dict | None


@dataclass(frozen=True, slots=True)
class CompanyUrlIndexOutput:
    company_id: str

    # NEW canonical "total url count" field you can sum for global totals.
    total_urls: int

    # Completion totals (prefer crawl_meta if present, else infer)
    urls_markdown_done: int
    urls_llm_done: int

    slice_counts: Counter[str]

    # Page-ish distributions used elsewhere
    total_pages: int
    markdown_saved: int


def extract_company_urlindex_metrics(
    inp: CompanyUrlIndexInput,
) -> CompanyUrlIndexOutput:
    meta = inp.crawl_meta if isinstance(inp.crawl_meta, dict) else {}
    ui = inp.url_index if isinstance(inp.url_index, dict) else {}

    # Prefer crawl_meta numbers if present; else infer from url_index.
    total_urls = int(_intish(meta.get("urls_total")) or 0)
    urls_md = int(_intish(meta.get("urls_markdown_done")) or 0)
    urls_llm = int(_intish(meta.get("urls_llm_done")) or 0)

    if total_urls <= 0 and isinstance(ui, dict):
        total_urls = sum(1 for k in ui.keys() if k != "__meta__")

    if urls_md <= 0 and isinstance(ui, dict):
        m = ui.get("__meta__")
        if isinstance(m, dict):
            urls_md = int(_intish(m.get("markdown_saved")) or 0)

    if urls_llm <= 0:
        llm_cnt = 0
        for _, entry in _iter_url_entries(ui if isinstance(ui, dict) else None):
            status = entry.get("status")
            if isinstance(status, str) and status in ("llm_full_extracted", "llm_done"):
                llm_cnt += 1
        urls_llm = llm_cnt

    slice_counts: Counter[str] = Counter()
    if isinstance(ui, dict):
        for _, entry in _iter_url_entries(ui):
            status = entry.get("status")
            if isinstance(status, str) and status.strip():
                if status == "markdown_suppressed":
                    reason = _suppressed_reason(entry)
                    slice_counts[f"suppressed: {reason}"] += 1
                else:
                    slice_counts[status] += 1
            else:
                if _infer_is_suppressed(entry):
                    reason = _suppressed_reason(entry)
                    slice_counts[f"suppressed: {reason}"] += 1
                else:
                    slice_counts["unknown"] += 1

    total_pages: Optional[int] = None
    markdown_saved: Optional[int] = None

    m2 = ui.get("__meta__") if isinstance(ui, dict) else None
    if isinstance(m2, dict):
        total_pages = _intish(m2.get("total_pages"))
        markdown_saved = _intish(m2.get("markdown_saved"))

    if total_pages is None and isinstance(ui, dict):
        total_pages = sum(1 for k in ui.keys() if k != "__meta__")

    if markdown_saved is None:
        ms_cnt = 0
        for _, entry in _iter_url_entries(ui if isinstance(ui, dict) else None):
            status = entry.get("status")
            if isinstance(status, str) and status in (
                "markdown_saved",
                "llm_full_extracted",
                "llm_done",
            ):
                ms_cnt += 1
            else:
                ga = entry.get("gating_action")
                if isinstance(ga, str) and ga == "save":
                    ms_cnt += 1
        markdown_saved = ms_cnt

    return CompanyUrlIndexOutput(
        company_id=str(inp.company_id),
        total_urls=max(0, int(total_urls)),
        urls_markdown_done=max(0, int(urls_md)),
        urls_llm_done=max(0, int(urls_llm)),
        slice_counts=slice_counts,
        total_pages=max(0, int(total_pages or 0)),
        markdown_saved=max(0, int(markdown_saved or 0)),
    )


def aggregate_url_completion_section_for_view(
    *,
    company_ids_in_view: Iterable[str],
    by_company: Dict[str, CompanyUrlIndexOutput],
    global_state: Optional[dict] = None,
    is_global_view: bool = False,
) -> Dict[str, Any]:
    ids = list(company_ids_in_view)

    # Keep existing behavior: prefer global_state sums for global view if present.
    if global_state is not None and bool(is_global_view):
        urls_total_sum = int(global_state.get("urls_total_sum", 0) or 0)
        urls_markdown_done_sum = int(global_state.get("urls_markdown_done_sum", 0) or 0)
        urls_llm_done_sum = int(global_state.get("urls_llm_done_sum", 0) or 0)
    else:
        urls_total_sum = 0
        urls_markdown_done_sum = 0
        urls_llm_done_sum = 0
        for cid in ids:
            r = by_company.get(cid)
            if r is None:
                continue
            urls_total_sum += int(r.total_urls)
            urls_markdown_done_sum += int(r.urls_markdown_done)
            urls_llm_done_sum += int(r.urls_llm_done)

    slice_counts: Counter[str] = Counter()
    for cid in ids:
        r = by_company.get(cid)
        if r is None:
            continue
        slice_counts.update(r.slice_counts)

    slices = [{"label": k, "count": int(v)} for k, v in slice_counts.most_common()]
    md_saved = int(urls_markdown_done_sum)
    llm_done = int(urls_llm_done_sum)

    return {
        "urls": {
            "totals": {
                "urls_total_sum": int(urls_total_sum),
                "urls_markdown_done_sum": int(urls_markdown_done_sum),
                "urls_llm_done_sum": int(urls_llm_done_sum),
            },
            "pie": {"chart_id": "urls_completion_pie", "slices": slices},
            "bars": [
                {
                    "id": "urls_llm_done_ratio",
                    "label": "LLM done / Markdown saved",
                    "num": int(llm_done),
                    "den": max(1, int(md_saved)),
                    "pct": _pct(int(llm_done), max(1, int(md_saved))),
                }
            ],
        }
    }


def aggregate_page_distributions_section_for_view(
    *,
    view_key: str,
    company_ids_in_view: Iterable[str],
    by_company: Dict[str, CompanyUrlIndexOutput],
) -> Dict[str, Any]:
    ids = list(company_ids_in_view)

    total_pages_vals: List[int] = []
    md_saved_vals: List[int] = []

    for cid in ids:
        r = by_company.get(cid)
        if r is None:
            continue
        total_pages_vals.append(int(r.total_pages))
        md_saved_vals.append(int(r.markdown_saved))

    pages_hist = _build_histogram_bins(total_pages_vals, target_bins=20)
    md_hist = _build_histogram_bins(md_saved_vals, target_bins=20)

    vslug = str(view_key)
    return {
        "distributions": {
            "total_pages": {
                "histogram_bins": pages_hist["histogram_bins"],
                "bin_labels": pages_hist["bin_labels"],
                "bin_counts": pages_hist["bin_counts"],
                "summary": summarize_distribution(total_pages_vals),
                "charts": {"hist": f"{vslug}_total_pages_hist"},
            },
            "markdown_saved": {
                "histogram_bins": md_hist["histogram_bins"],
                "bin_labels": md_hist["bin_labels"],
                "bin_counts": md_hist["bin_counts"],
                "summary": summarize_distribution(md_saved_vals),
                "charts": {"hist": f"{vslug}_markdown_saved_hist"},
            },
        }
    }


# Backwards-compatible wrappers (kept as-is)
def compute_url_completion_section(
    view: IndustryView, *, global_state: Optional[dict] = None
) -> Dict[str, Any]:
    by_company: Dict[str, CompanyUrlIndexOutput] = {}
    for c in view.companies:
        inp = CompanyUrlIndexInput(
            company_id=c.company_id,
            crawl_meta=getattr(c, "crawl_meta", None),
            url_index=getattr(c, "url_index", None),
        )
        by_company[c.company_id] = extract_company_urlindex_metrics(inp)

    return aggregate_url_completion_section_for_view(
        company_ids_in_view=[c.company_id for c in view.companies],
        by_company=by_company,
        global_state=global_state,
        is_global_view=(view.key == "__global__"),
    )


def compute_page_distributions_section(view: IndustryView) -> Dict[str, Any]:
    by_company: Dict[str, CompanyUrlIndexOutput] = {}
    for c in view.companies:
        inp = CompanyUrlIndexInput(
            company_id=c.company_id,
            crawl_meta=getattr(c, "crawl_meta", None),
            url_index=getattr(c, "url_index", None),
        )
        by_company[c.company_id] = extract_company_urlindex_metrics(inp)

    return aggregate_page_distributions_section_for_view(
        view_key=view.key,
        company_ids_in_view=[c.company_id for c in view.companies],
        by_company=by_company,
    )
