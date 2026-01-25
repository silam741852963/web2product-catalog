from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from .artifact_layout import (
    AnalysisPaths,
    build_analysis_paths,
    write_bytes,
    write_json,
)
from .charts_bar import (
    make_bar,
    make_barh_stacked,
    make_histogram,
    make_histogram_binned,
)
from .charts_pie import make_pie
from .charts_scatter import make_scatter
from .data_loader import AnalysisDataset, CompanyRecord, load_analysis_dataset
from .domain_metrics import (
    CompanyDomainInput,
    CompanyDomainOutput,
    aggregate_domain_sections_for_view,
    extract_company_domain_metrics,
)
from .error_metrics import (
    CompanyErrorInput,
    CompanyErrorOutput,
    aggregate_error_sections_for_view,
    extract_company_error_metrics,
)
from .html_report import build_html_report
from .industry_views import IndustryView, build_industry_views
from .offering_metrics import (
    CompanyOfferingInput,
    CompanyOfferingOutput,
    aggregate_offering_sections_for_view,
    extract_company_offering_metrics,
)
from .state_metrics import (
    CompanyStateInput,
    CompanyStateOutput,
    aggregate_company_status_section_for_view,
    extract_company_state_metrics,
)
from .token_metrics import (
    CompanyTokenInput,
    CompanyTokenOutput,
    aggregate_token_sections_for_view,
    extract_company_token_metrics,
)
from .urlindex_metrics import (
    CompanyUrlIndexInput,
    CompanyUrlIndexOutput,
    aggregate_page_distributions_section_for_view,
    aggregate_url_completion_section_for_view,
    extract_company_urlindex_metrics,
)

logger = logging.getLogger(__name__)

_UNCLASSIFIED_GROUP_KEY = "industry:unknown"
_UNCLASSIFIED_LABEL = "Unclassified"


def _utc_iso_z() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(s: str) -> str:
    t = (s or "").strip().lower()
    out = []
    for ch in t:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", ".", "—"):
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "view"


def _write_chart_artifact(
    paths: AnalysisPaths, chart: Dict[str, Any]
) -> Dict[str, Any]:
    chart_id = str(chart["chart_id"])
    title = str(chart.get("title") or chart_id)

    spec_path = paths.charts_spec_dir / f"{chart_id}.json"
    png_path = paths.charts_png_dir / f"{chart_id}.png"

    write_json(spec_path, chart["plotly_spec"])
    write_bytes(png_path, chart["png_bytes"])

    logger.debug(
        "[analysis][persist] chart_id=%s spec=%s png=%s",
        chart_id,
        spec_path,
        png_path,
    )

    return {
        "chart_id": chart_id,
        "title": title,
        "spec": str(Path("charts/spec") / f"{chart_id}.json"),
        "png": str(Path("charts/png") / f"{chart_id}.png"),
    }


def _auto_workers(workers: int) -> int:
    if int(workers) > 0:
        return int(workers)
    cpu = os.cpu_count() or 4
    return min(32, int(cpu))


def _auto_chart_workers(company_workers: int) -> int:
    cpu = os.cpu_count() or 4
    base = min(32, max(4, int(cpu)))
    return min(
        base, max(4, min(32, int(company_workers) if company_workers > 0 else base))
    )


def _auto_view_workers(company_workers: int) -> int:
    cpu = os.cpu_count() or 4
    base = min(32, max(4, int(cpu)))
    return min(base, max(4, int(company_workers) if company_workers > 0 else base))


def _maybe_log_progress(
    prefix: str, done: int, total: int, step: int, started_at: float
) -> None:
    if total <= 0:
        return
    if done == total or (step > 0 and done % step == 0):
        elapsed = max(0.0001, time.perf_counter() - started_at)
        rate = done / elapsed
        pct = round((done / total) * 100.0, 2)
        logger.info(
            "%s %d/%d (%.2f%%) elapsed=%.1fs rate=%.1f/s",
            prefix,
            done,
            total,
            pct,
            elapsed,
            rate,
        )


def _extract_markdown_saved_from_url_index(url_index: object) -> int:
    md_saved = 0
    if isinstance(url_index, dict):
        meta = url_index.get("__meta__")
        if isinstance(meta, dict):
            v = meta.get("markdown_saved")
            if isinstance(v, int):
                md_saved = v
            elif isinstance(v, str) and v.isdigit():
                md_saved = int(v)
    return int(md_saved)


def _select_companies_for_views(
    views: Dict[str, IndustryView],
) -> Dict[str, CompanyRecord]:
    uniq: Dict[str, CompanyRecord] = {}
    for v in views.values():
        for c in v.companies:
            uniq.setdefault(c.company_id, c)
    return uniq


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str) and x.strip().lstrip("-").isdigit():
            return int(x.strip())
    except Exception:
        pass
    return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
    except Exception:
        pass
    return float(default)


def _compute_global_totals(
    *,
    companies_total: int,
    include_llm_metrics: bool,
    token_by_company: Dict[str, CompanyTokenOutput],
    offering_by_company: Dict[str, CompanyOfferingOutput],
    urlindex_by_company: Dict[str, CompanyUrlIndexOutput],
    global_state: Optional[dict] = None,
) -> Dict[str, Any]:
    # Prefer deterministic per-company sums.
    urls_sum = sum(int(r.total_urls) for r in urlindex_by_company.values())

    # If global_state has authoritative sums and per-company sum is missing/zero, fall back.
    if urls_sum <= 0 and isinstance(global_state, dict):
        urls_sum = int(global_state.get("urls_total_sum", 0) or 0)

    md_saved_sum = sum(int(r.markdown_saved) for r in token_by_company.values())
    md_tokens_sum = sum(int(r.md_tokens) for r in token_by_company.values())

    llm_out_tokens_sum = (
        sum(int(r.llm_output_tokens) for r in token_by_company.values())
        if bool(include_llm_metrics)
        else 0
    )

    offerings_sum = (
        sum(int(r.total) for r in offering_by_company.values())
        if bool(include_llm_metrics)
        else 0
    )

    return {
        "companies_total": int(companies_total),
        "urls_total": int(urls_sum),
        "markdown_saved_total": int(md_saved_sum),
        "md_tokens_total": int(md_tokens_sum),
        "llm_output_tokens_total": int(llm_out_tokens_sum),
        "offerings_total": int(offerings_sum),
        "llm_metrics_enabled": bool(include_llm_metrics),
    }


def _prefixed_chart_id(vslug: str, raw_id: str) -> str:
    raw = (raw_id or "").strip()
    if not raw:
        return f"{vslug}_chart"
    return f"{vslug}_{raw}"


def _plot_summary_bar(
    *, chart_id: str, title: str, summary: dict
) -> Optional[Dict[str, Any]]:
    if not isinstance(summary, dict) or not summary:
        return None

    keys_in_order = [
        ("p10", "p10"),
        ("p25", "p25"),
        ("median", "median"),
        ("mean", "mean"),
        ("p75", "p75"),
        ("p90", "p90"),
        ("p95", "p95"),
        ("p99", "p99"),
    ]
    labels: list[str] = []
    values: list[float] = []
    for k, lab in keys_in_order:
        if k in summary:
            labels.append(lab)
            values.append(_safe_float(summary.get(k), 0.0))

    if not labels:
        return None

    return make_bar(
        chart_id=chart_id,
        title=title,
        x=labels,
        y=values,
        x_title="stat",
        y_title="value",
        y_log=False,
    )


def _safe_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def _aligned_int_vector(ids: list[str], mapping: Dict[str, int]) -> list[int]:
    return [int(mapping.get(cid, 0) or 0) for cid in ids]


def _build_industry_group_key_label_map(ds: AnalysisDataset) -> Dict[str, str]:
    out: Dict[str, str] = {_UNCLASSIFIED_GROUP_KEY: _UNCLASSIFIED_LABEL}

    for c in ds.companies:
        try:
            gk = str(getattr(c, "industry_group_key", "") or "")
            if not gk:
                continue
            if gk == _UNCLASSIFIED_GROUP_KEY:
                out[gk] = _UNCLASSIFIED_LABEL
                continue

            lbl = getattr(c, "industry_label_lv1", None)
            if isinstance(lbl, str) and lbl.strip():
                out.setdefault(gk, lbl.strip())
                continue

            ind = getattr(c, "industry", None)
            if isinstance(ind, int) and gk.startswith("industry:"):
                out.setdefault(gk, f"Industry {ind}")
        except Exception:
            continue

    return out


def _industry_display_label(
    industry_key: str, industry_key_to_label: Dict[str, str]
) -> str:
    key = (industry_key or "").strip()
    if not key:
        return _UNCLASSIFIED_LABEL
    if key == _UNCLASSIFIED_GROUP_KEY:
        return _UNCLASSIFIED_LABEL
    lbl = industry_key_to_label.get(key)
    if isinstance(lbl, str) and lbl.strip():
        return lbl.strip()
    if key.startswith("industry:"):
        tail = key.split("industry:", 1)[1]
        if tail.isdigit():
            return f"Industry {int(tail)}"
    return key


def _normalize_offering_per_industry_rows_inplace(
    offerings_section: Dict[str, Any],
    industry_key_to_label: Dict[str, str],
) -> None:
    per_ind = offerings_section.get("per_industry")
    if not isinstance(per_ind, dict):
        return
    rows = per_ind.get("rows")
    if not isinstance(rows, list):
        return

    for r in rows:
        if not isinstance(r, dict):
            continue
        raw_key = r.get("industry")
        if not isinstance(raw_key, str):
            continue
        r.setdefault("industry_key", raw_key)
        r["industry"] = _industry_display_label(raw_key, industry_key_to_label)


def _parse_industry_key(view_key: str) -> Optional[int]:
    k = (view_key or "").strip()
    if not k.startswith("industry:"):
        return None
    tail = k.split("industry:", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def _sort_view_key(view_key: str) -> tuple[int, int, str]:
    k = (view_key or "").strip()

    if k == "__global__":
        return (0, 0, "")

    if k == _UNCLASSIFIED_GROUP_KEY:
        return (9, 0, k)

    n = _parse_industry_key(k)
    if n is not None:
        return (1, n, "")

    return (5, 0, k)


def _sorted_view_items(
    views: Dict[str, IndustryView],
) -> list[tuple[str, IndustryView]]:
    items = list(views.items())
    items.sort(key=lambda kv: _sort_view_key(kv[0]))
    return items


def run_analysis(
    *,
    out_dir: str | Path,
    run_tag: str = "analysis-latest",
    view_mode: str = "global",  # global | lv1 | lv1-all | lv2-all
    industry_id: Optional[int] = None,  # used when view_mode == lv1
    k_top_domains: int = 20,
    k_top_errors: int = 20,
    include_companies_sample: bool = False,
    max_companies: Optional[int] = None,
    workers: int = 0,
    per_doc_overhead: int = 200,
    prompt_overhead_by_industry: Optional[Dict[str, int]] = None,
    include_llm_metrics: bool = True,
) -> AnalysisPaths:
    logger.info("[analysis] __file__=%s", __file__)

    out_dir_p = Path(out_dir)
    logger.info(
        "[analysis] start out_dir=%s run_tag=%s view_mode=%s industry_id=%s max_companies=%s workers=%s include_llm_metrics=%s",
        out_dir_p,
        run_tag,
        view_mode,
        industry_id,
        max_companies,
        workers,
        bool(include_llm_metrics),
    )

    paths = build_analysis_paths(out_dir=out_dir_p, run_tag=run_tag)
    logger.info(
        "[analysis] paths report_json=%s report_html=%s charts_png_dir=%s charts_spec_dir=%s",
        paths.report_json,
        paths.report_html,
        paths.charts_png_dir,
        paths.charts_spec_dir,
    )

    ds: AnalysisDataset = load_analysis_dataset(
        out_dir=out_dir_p, max_companies=max_companies
    )
    logger.info(
        "[analysis] dataset loaded companies=%d output_root=%s has_global_state=%s industry_tables=%s",
        len(ds.companies),
        getattr(ds, "output_root", None),
        bool(ds.crawl_global_state),
        (type(ds.industry_tables).__name__ if ds.industry_tables is not None else None),
    )

    views = build_industry_views(ds, mode=view_mode, industry_id=industry_id)
    logger.info(
        "[analysis] views built count=%d keys_sample=%s",
        len(views),
        list(views.keys())[:10],
    )

    industry_key_to_label = _build_industry_group_key_label_map(ds)
    global_state = ds.crawl_global_state
    prompt_overhead_by_industry = prompt_overhead_by_industry or {}

    report: Dict[str, Any] = {
        "meta": {
            "out_dir": str(out_dir_p),
            "analysis_run_tag": run_tag,
            "generated_at": _utc_iso_z(),
            "crawl_global_state_updated_at": (
                (global_state or {}).get("updated_at")
                or (global_state or {}).get("crawl_global_state_updated_at")
                or None
            ),
            "total_companies_loaded": len(ds.companies),
            "total_views": len(views),
            "view_mode": view_mode,
            "industry_id": industry_id,
            "llm_metrics_enabled": bool(include_llm_metrics),
        },
        "views": {},
    }

    # --- Parallel chart writing infrastructure ---
    chart_specs: Dict[str, dict] = {}
    chart_specs_lock = Lock()

    chart_futures: Dict[Future[Dict[str, Any]], str] = {}
    chart_futures_lock = Lock()

    w_company = _auto_workers(workers)
    w_chart = _auto_chart_workers(w_company)
    w_view = _auto_view_workers(w_company)

    def persist_async(ex: ThreadPoolExecutor, chart: Dict[str, Any]) -> Dict[str, Any]:
        chart_id = str(chart["chart_id"])
        title = str(chart.get("title") or chart_id)

        with chart_specs_lock:
            chart_specs[chart_id] = chart["plotly_spec"]

        fut = ex.submit(_write_chart_artifact, paths, chart)
        with chart_futures_lock:
            chart_futures[fut] = chart_id

        return {
            "chart_id": chart_id,
            "title": title,
            "spec": str(Path("charts/spec") / f"{chart_id}.json"),
            "png": str(Path("charts/png") / f"{chart_id}.png"),
        }

    companies_to_process = _select_companies_for_views(views)
    company_list = sorted(companies_to_process.values(), key=lambda c: c.company_id)

    n_companies = len(company_list)
    logger.info(
        "analysis: view_mode=%s views=%d companies_to_process=%d workers=%d chart_workers=%d llm_metrics=%s",
        view_mode,
        len(views),
        n_companies,
        w_company,
        w_chart,
        bool(include_llm_metrics),
    )

    step = max(1, n_companies // 20)
    started_at = time.perf_counter()

    token_by_company: Dict[str, CompanyTokenOutput] = {}
    offering_by_company: Dict[str, CompanyOfferingOutput] = {}
    domain_by_company: Dict[str, CompanyDomainOutput] = {}
    error_by_company: Dict[str, CompanyErrorOutput] = {}
    urlindex_by_company: Dict[str, CompanyUrlIndexOutput] = {}
    state_by_company: Dict[str, CompanyStateOutput] = {}

    def _process_company(
        c: CompanyRecord,
    ) -> Tuple[
        CompanyTokenOutput,
        Optional[CompanyOfferingOutput],
        CompanyDomainOutput,
        CompanyErrorOutput,
        CompanyUrlIndexOutput,
        CompanyStateOutput,
    ]:
        md_saved = _extract_markdown_saved_from_url_index(c.url_index)

        tok_out = extract_company_token_metrics(
            CompanyTokenInput(
                company_id=c.company_id,
                industry_group_key=c.industry_group_key,
                html_dir=c.paths.html_dir,
                markdown_dir=c.paths.markdown_dir,
                product_dir=c.paths.product_dir,
                markdown_saved=int(md_saved),
                include_llm_tokens=bool(
                    include_llm_metrics
                ),  # CRITICAL: do not compute when disabled
            )
        )

        off_out: Optional[CompanyOfferingOutput] = None
        if bool(include_llm_metrics):
            off_out = extract_company_offering_metrics(
                CompanyOfferingInput(
                    company_id=c.company_id,
                    industry_group_key=c.industry_group_key,
                    company_profile=c.company_profile,
                )
            )

        dom_out = extract_company_domain_metrics(
            CompanyDomainInput(
                company_id=c.company_id,
                root_url=c.root_url or "",
                crawl_meta=c.crawl_meta,
                url_index=c.url_index,
            )
        )

        err_out = extract_company_error_metrics(
            CompanyErrorInput(
                company_id=c.company_id,
                crawl_meta=c.crawl_meta,
                url_index=c.url_index,
            )
        )

        ui_out = extract_company_urlindex_metrics(
            CompanyUrlIndexInput(
                company_id=c.company_id,
                crawl_meta=c.crawl_meta,
                url_index=c.url_index,
            )
        )

        st_out = extract_company_state_metrics(
            CompanyStateInput(
                company_id=c.company_id,
                crawl_meta=c.crawl_meta,
                url_index=c.url_index,
            )
        )

        return tok_out, off_out, dom_out, err_out, ui_out, st_out

    processed = 0
    logger.info("[analysis] per-company processing start n=%d", n_companies)
    with ThreadPoolExecutor(max_workers=w_company) as ex:
        futs = {ex.submit(_process_company, c): c.company_id for c in company_list}
        for fut in as_completed(futs):
            cid = futs.get(fut, "<?>")
            try:
                tok_out, off_out, dom_out, err_out, ui_out, st_out = fut.result()
                token_by_company[tok_out.company_id] = tok_out
                if bool(include_llm_metrics) and off_out is not None:
                    offering_by_company[off_out.company_id] = off_out
                domain_by_company[dom_out.company_id] = dom_out
                error_by_company[err_out.company_id] = err_out
                urlindex_by_company[ui_out.company_id] = ui_out
                state_by_company[st_out.company_id] = st_out
            except Exception:
                logger.exception("per-company analysis failed cid=%s", cid)
            finally:
                processed += 1
                _maybe_log_progress(
                    "processed:", processed, n_companies, step, started_at
                )

    logger.info(
        "[analysis] per-company processing done processed=%d token=%d offering=%d domain=%d error=%d urlindex=%d state=%d",
        processed,
        len(token_by_company),
        len(offering_by_company),
        len(domain_by_company),
        len(error_by_company),
        len(urlindex_by_company),
        len(state_by_company),
    )

    # --- Global totals (for HTML header) ---
    totals = _compute_global_totals(
        companies_total=int(n_companies),  # processed set (companies_to_process)
        include_llm_metrics=bool(include_llm_metrics),
        token_by_company=token_by_company,
        offering_by_company=offering_by_company,
        urlindex_by_company=urlindex_by_company,
        global_state=global_state,
    )

    meta = report.get("meta")
    if isinstance(meta, dict):
        meta["totals"] = totals
    md_tokens_all: Dict[str, int] = {
        cid: int(r.md_tokens) for cid, r in token_by_company.items()
    }
    llm_tokens_all: Dict[str, int] = (
        {cid: int(r.llm_output_tokens) for cid, r in token_by_company.items()}
        if bool(include_llm_metrics)
        else {}
    )
    total_pages_all: Dict[str, int] = {
        cid: int(r.total_pages) for cid, r in domain_by_company.items()
    }

    view_items = _sorted_view_items(views)
    total_views = len(view_items)

    logger.info("[analysis] building views total=%d", total_views)

    def _build_one_view(
        view_key: str,
        view: IndustryView,
        *,
        chart_ex: ThreadPoolExecutor,
    ) -> Tuple[str, Dict[str, Any]]:
        vslug = _slug(view_key)
        is_global = view_key == "__global__"
        company_ids_in_view = [c.company_id for c in view.companies]

        sections: Dict[str, Any] = {}
        charts: list[Dict[str, Any]] = []

        # Company status
        company_status = aggregate_company_status_section_for_view(
            company_ids_in_view=company_ids_in_view,
            by_company=state_by_company,
            global_state=global_state if is_global else None,
            is_global_view=bool(is_global),
        )
        sections.update(company_status)

        cs = company_status.get("company_status") or {}
        pie_def = cs.get("pie") or {}
        if isinstance(pie_def, dict) and pie_def:
            chart_id = _prefixed_chart_id(
                vslug, str(pie_def.get("chart_id") or "company_status_pie")
            )
            charts.append(
                persist_async(
                    chart_ex,
                    make_pie(
                        chart_id=chart_id,
                        title=f"{view.label}: Company status",
                        labels=[str(x) for x in (pie_def.get("labels") or [])],
                        values=[_as_int(x, 0) for x in (pie_def.get("values") or [])],
                    ),
                )
            )

        # URL completion
        url_completion = aggregate_url_completion_section_for_view(
            company_ids_in_view=company_ids_in_view,
            by_company=urlindex_by_company,
            global_state=global_state if is_global else None,
            is_global_view=bool(is_global),
        )
        sections.update(url_completion)

        urls = url_completion.get("urls") or {}
        urls_pie = urls.get("pie") or {}
        if isinstance(urls_pie, dict) and urls_pie:
            chart_id = _prefixed_chart_id(
                vslug, str(urls_pie.get("chart_id") or "urls_completion_pie")
            )
            slices = list(urls_pie.get("slices") or [])
            charts.append(
                persist_async(
                    chart_ex,
                    make_pie(
                        chart_id=chart_id,
                        title=f"{view.label}: URL completion",
                        labels=[
                            str(x.get("label", ""))
                            for x in slices
                            if isinstance(x, dict)
                        ],
                        values=[
                            _as_int(x.get("count", 0), 0)
                            for x in slices
                            if isinstance(x, dict)
                        ],
                    ),
                )
            )

        # Page distributions
        dists = aggregate_page_distributions_section_for_view(
            view_key=view.key,
            company_ids_in_view=company_ids_in_view,
            by_company=urlindex_by_company,
        )
        sections.update(dists)

        dist_block = dists.get("distributions") or {}
        if isinstance(dist_block, dict):
            for dist_name, title_name, x_title in (
                ("total_pages", "Total pages per company", "total pages (binned)"),
                (
                    "markdown_saved",
                    "Markdown saved per company",
                    "markdown saved (binned)",
                ),
            ):
                dist = dist_block.get(dist_name) or {}
                if not isinstance(dist, dict):
                    continue
                bin_labels = dist.get("bin_labels")
                bin_counts = dist.get("bin_counts")
                if (
                    isinstance(bin_labels, list)
                    and isinstance(bin_counts, list)
                    and bin_labels
                    and bin_counts
                ):
                    raw_id = (dist.get("charts") or {}).get(
                        "hist"
                    ) or f"{dist_name}_hist"
                    chart_id = _prefixed_chart_id(vslug, str(raw_id))
                    charts.append(
                        persist_async(
                            chart_ex,
                            make_histogram_binned(
                                chart_id=chart_id,
                                title=f"{view.label}: {title_name}",
                                x_labels=[str(x) for x in bin_labels],
                                counts=[_as_int(c, 0) for c in bin_counts],
                                x_title=x_title,
                                y_title="companies",
                                y_log=True,
                            ),
                        )
                    )

        # TOKENS (LLM charts omitted when disabled)
        token_section = aggregate_token_sections_for_view(
            company_ids_in_view=company_ids_in_view,
            by_company=token_by_company,
            per_doc_overhead=int(per_doc_overhead),
            prompt_overhead_by_industry=prompt_overhead_by_industry,
            include_llm_metrics=bool(include_llm_metrics),
        )
        sections.update(token_section)

        tok = sections.get("tokens") if isinstance(sections, dict) else None
        if isinstance(tok, dict):
            totals = tok.get("totals")
            if isinstance(totals, dict):
                md_total = _as_int(totals.get("md_tokens_total"), 0)
                pruned_total = _as_int(totals.get("pruned_tokens_total"), 0)
                chart_id = _prefixed_chart_id(vslug, "tokens_totals_stacked")
                charts.append(
                    persist_async(
                        chart_ex,
                        make_barh_stacked(
                            chart_id=chart_id,
                            title=f"{view.label}: Token totals (md vs pruned)",
                            categories=["tokens"],
                            series={"md": [md_total], "pruned": [pruned_total]},
                            x_title="tokens",
                            y_title="",
                            x_log=False,
                        ),
                    )
                )

            d = tok.get("distributions")
            if isinstance(d, dict):
                dist_keys = ["md_tokens"]
                if bool(include_llm_metrics):
                    dist_keys.append("llm_output_tokens")

                for dist_key in dist_keys:
                    dist_obj = d.get(dist_key)
                    if not isinstance(dist_obj, dict):
                        continue
                    summary = dist_obj.get("summary")
                    if not isinstance(summary, dict) or not summary:
                        continue
                    charts_def = dist_obj.get("charts") or {}
                    raw_bar_id = (
                        charts_def.get("bar") if isinstance(charts_def, dict) else None
                    )
                    chart_id = _prefixed_chart_id(
                        vslug, str(raw_bar_id or f"tokens_{dist_key}_summary_bar")
                    )
                    ch = _plot_summary_bar(
                        chart_id=chart_id,
                        title=f"{view.label}: {dist_key} (summary)",
                        summary=summary,
                    )
                    if ch is not None:
                        charts.append(persist_async(chart_ex, ch))

        # DOMAINS
        sections.update(
            aggregate_domain_sections_for_view(
                company_ids_in_view=company_ids_in_view,
                by_company=domain_by_company,
                k_top=int(k_top_domains),
            )
        )

        dom = sections.get("domains")
        if isinstance(dom, dict):
            offsite_pct = _safe_list(dom.get("offsite_pct_by_company"))
            if offsite_pct:
                raw_id = (
                    (dom.get("charts") or {})
                    if isinstance(dom.get("charts"), dict)
                    else {}
                ).get("offsite_pct_hist") or "offsite_pct_hist"
                chart_id = _prefixed_chart_id(vslug, str(raw_id))
                charts.append(
                    persist_async(
                        chart_ex,
                        make_histogram(
                            chart_id=chart_id,
                            title=f"{view.label}: Offsite % per company",
                            values=[_safe_float(x, 0.0) for x in offsite_pct],
                            x_title="offsite (%)",
                            y_title="companies",
                            y_log=True,
                        ),
                    )
                )

            top_dom = dom.get("top_offsite_domains") or {}
            rows = list(top_dom.get("rows") or [])
            if rows:
                labels = [
                    str(r[0])
                    for r in rows
                    if isinstance(r, (list, tuple)) and len(r) >= 2
                ]
                counts = [
                    _as_int(r[1], 0)
                    for r in rows
                    if isinstance(r, (list, tuple)) and len(r) >= 2
                ]
                raw_id = (
                    (dom.get("charts") or {})
                    if isinstance(dom.get("charts"), dict)
                    else {}
                ).get("top_offsite_domains_bar") or "top_offsite_domains_bar"
                chart_id = _prefixed_chart_id(vslug, str(raw_id))
                if labels and counts:
                    charts.append(
                        persist_async(
                            chart_ex,
                            make_bar(
                                chart_id=chart_id,
                                title=f"{view.label}: Top offsite domains",
                                x=labels,
                                y=counts,
                                x_title="domain",
                                y_title="pages",
                                y_log=True,
                            ),
                        )
                    )

        # ERRORS
        sections.update(
            aggregate_error_sections_for_view(
                company_ids_in_view=company_ids_in_view,
                by_company=error_by_company,
                k_top=int(k_top_errors),
            )
        )

        err = sections.get("errors")
        if isinstance(err, dict):
            errors_per_company = _safe_list(err.get("errors_per_company"))
            if errors_per_company:
                raw_id = (
                    (err.get("charts") or {})
                    if isinstance(err.get("charts"), dict)
                    else {}
                ).get("errors_total_hist") or "errors_total_hist"
                chart_id = _prefixed_chart_id(vslug, str(raw_id))
                charts.append(
                    persist_async(
                        chart_ex,
                        make_histogram(
                            chart_id=chart_id,
                            title=f"{view.label}: Error count per company",
                            values=[_safe_float(x, 0.0) for x in errors_per_company],
                            x_title="errors (count)",
                            y_title="companies",
                            y_log=True,
                        ),
                    )
                )

            top_sig = err.get("top_signatures") or {}
            sig_rows = list(top_sig.get("rows") or [])
            if sig_rows:
                labels = [
                    str(r.get("label", "")) for r in sig_rows if isinstance(r, dict)
                ]
                counts = [
                    _as_int(r.get("count", 0), 0)
                    for r in sig_rows
                    if isinstance(r, dict)
                ]
                raw_id = (
                    (err.get("charts") or {})
                    if isinstance(err.get("charts"), dict)
                    else {}
                ).get("top_signatures_bar") or "errors_top_signatures_bar"
                chart_id = _prefixed_chart_id(vslug, str(raw_id))
                if labels and counts:
                    charts.append(
                        persist_async(
                            chart_ex,
                            make_bar(
                                chart_id=chart_id,
                                title=f"{view.label}: Top error signatures",
                                x=labels,
                                y=counts,
                                x_title="signature",
                                y_title="count",
                                y_log=True,
                            ),
                        )
                    )

        # OFFERINGS (entirely omitted when LLM metrics disabled)
        if bool(include_llm_metrics):
            sections.update(
                aggregate_offering_sections_for_view(
                    company_ids_in_view=company_ids_in_view,
                    by_company=offering_by_company,
                    md_tokens_by_company=md_tokens_all,
                    llm_tokens_by_company=llm_tokens_all,
                    total_pages_by_company=total_pages_all,
                )
            )

            off = sections.get("offerings")
            if isinstance(off, dict):
                _normalize_offering_per_industry_rows_inplace(
                    off, industry_key_to_label
                )

                per_ind = off.get("per_industry") or {}
                ind_rows = list(per_ind.get("rows") or [])
                if ind_rows:
                    cats = [
                        str(r.get("industry", ""))
                        for r in ind_rows
                        if isinstance(r, dict)
                    ]
                    prod = [
                        _as_int(r.get("product", 0), 0)
                        for r in ind_rows
                        if isinstance(r, dict)
                    ]
                    serv = [
                        _as_int(r.get("service", 0), 0)
                        for r in ind_rows
                        if isinstance(r, dict)
                    ]

                    raw_id = (
                        (per_ind.get("charts") or {})
                        if isinstance(per_ind.get("charts"), dict)
                        else {}
                    ).get("stacked_bar") or "offering_industry_stacked_bar"
                    chart_id = _prefixed_chart_id(vslug, str(raw_id))

                    if cats:
                        charts.append(
                            persist_async(
                                chart_ex,
                                make_barh_stacked(
                                    chart_id=chart_id,
                                    title=f"{view.label}: Offerings by industry (product vs service)",
                                    categories=cats,
                                    series={"product": prod, "service": serv},
                                    x_title="offerings",
                                    y_title="industry",
                                    x_log=False,
                                ),
                            )
                        )

                feat = off.get("features") or {}
                offering_count = _safe_list(feat.get("offering_count"))
                if offering_count:
                    raw_id = (
                        ((off.get("per_company") or {}).get("charts") or {})
                        if isinstance(
                            ((off.get("per_company") or {}).get("charts")), dict
                        )
                        else {}
                    ).get("hist_total") or "offering_total_hist"
                    chart_id = _prefixed_chart_id(vslug, str(raw_id))
                    charts.append(
                        persist_async(
                            chart_ex,
                            make_histogram(
                                chart_id=chart_id,
                                title=f"{view.label}: Total offerings per company",
                                values=[_safe_float(x, 0.0) for x in offering_count],
                                x_title="offerings (count)",
                                y_title="companies",
                                y_log=True,
                            ),
                        )
                    )

        # SCATTERS (no offering/llm scatters when disabled)
        md_vec = _aligned_int_vector(company_ids_in_view, md_tokens_all)
        pages_vec = _aligned_int_vector(company_ids_in_view, total_pages_all)
        llm_vec = (
            _aligned_int_vector(company_ids_in_view, llm_tokens_all)
            if bool(include_llm_metrics)
            else []
        )

        if bool(include_llm_metrics):
            off = sections.get("offerings")
            if isinstance(off, dict):
                feat = off.get("features") or {}
                if isinstance(feat, dict):
                    x_map: Dict[str, list[float]] = {
                        "md_tokens": [float(x) for x in md_vec],
                        "total_pages": [float(x) for x in pages_vec],
                        "llm_tokens": [float(x) for x in llm_vec],
                    }

                    base_keys = [
                        "offering_count",
                        "offering_density",
                        "offering_per_md_token",
                        "offering_per_llm_token",
                        "unique_offering_ratio",
                    ]
                    for k in base_keys:
                        v = feat.get(k)
                        if isinstance(v, list) and len(v) == len(company_ids_in_view):
                            x_map[k] = [float(_safe_float(z, 0.0)) for z in v]

                    scatters = list(off.get("scatter_candidates") or [])
                    for sc in scatters:
                        if not isinstance(sc, dict):
                            continue
                        x_key = str(sc.get("x") or "")
                        y_key = str(sc.get("y") or "")
                        raw_id = str(sc.get("chart_id") or f"scatter_{x_key}_{y_key}")
                        if x_key not in x_map or y_key not in x_map:
                            continue
                        x_vals = x_map[x_key]
                        y_vals = x_map[y_key]
                        if not x_vals or not y_vals or len(x_vals) != len(y_vals):
                            continue
                        chart_id = _prefixed_chart_id(vslug, raw_id)
                        hover = [str(cid) for cid in company_ids_in_view]
                        charts.append(
                            persist_async(
                                chart_ex,
                                make_scatter(
                                    chart_id=chart_id,
                                    title=f"{view.label}: {x_key} vs {y_key}",
                                    x=x_vals,
                                    y=y_vals,
                                    x_title=x_key,
                                    y_title=y_key,
                                    text=hover,
                                ),
                            )
                        )

        # Domain scatter (always)
        if isinstance(dom, dict):
            off_pct = dom.get("offsite_pct_by_company")
            if isinstance(off_pct, list) and len(off_pct) == len(company_ids_in_view):
                off_pct_vec = [float(_safe_float(x, 0.0)) for x in off_pct]
                hover = [str(cid) for cid in company_ids_in_view]
                charts.append(
                    persist_async(
                        chart_ex,
                        make_scatter(
                            chart_id=_prefixed_chart_id(
                                vslug, "scatter_pages_offsite_pct"
                            ),
                            title=f"{view.label}: total_pages vs offsite_pct",
                            x=[float(x) for x in pages_vec],
                            y=off_pct_vec,
                            x_title="total_pages",
                            y_title="offsite_pct",
                            text=hover,
                        ),
                    )
                )

        # Errors scatter (always)
        if isinstance(err, dict):
            epc = err.get("errors_per_company")
            if isinstance(epc, list) and len(epc) == len(company_ids_in_view):
                err_vec = [float(_safe_float(x, 0.0)) for x in epc]
                hover = [str(cid) for cid in company_ids_in_view]
                charts.append(
                    persist_async(
                        chart_ex,
                        make_scatter(
                            chart_id=_prefixed_chart_id(vslug, "scatter_pages_errors"),
                            title=f"{view.label}: total_pages vs errors",
                            x=[float(x) for x in pages_vec],
                            y=err_vec,
                            x_title="total_pages",
                            y_title="errors",
                            text=hover,
                        ),
                    )
                )

        payload: Dict[str, Any] = {
            "view_key": view_key,
            "label": view.label,
            "company_count": len(view.companies),
            "sections": sections,
            "charts": charts,
        }

        if include_companies_sample:
            payload["companies_sample"] = [
                c.company_id for c in view.companies[: min(25, len(view.companies))]
            ]

        return view_key, payload

    chart_started_at = time.perf_counter()
    with (
        ThreadPoolExecutor(max_workers=w_chart) as chart_ex,
        ThreadPoolExecutor(max_workers=w_view) as view_ex,
    ):
        logger.info(
            "[analysis] building views parallel total=%d view_workers=%d chart_workers=%d",
            total_views,
            w_view,
            w_chart,
        )

        view_futs = {
            view_ex.submit(_build_one_view, view_key, view, chart_ex=chart_ex): view_key
            for (view_key, view) in view_items
        }

        results: Dict[str, Dict[str, Any]] = {}
        done_views = 0
        step_views = max(1, total_views // 20) if total_views > 0 else 1
        views_started_at = time.perf_counter()

        for fut in as_completed(view_futs):
            vk = view_futs.get(fut, "<?>")
            try:
                view_key, payload = fut.result()
                results[view_key] = payload
            except Exception:
                logger.exception("[analysis][view] aggregation failed key=%s", vk)
            finally:
                done_views += 1
                if total_views > 0:
                    _maybe_log_progress(
                        "views_built:",
                        done_views,
                        total_views,
                        step_views,
                        views_started_at,
                    )

        for view_key, _ in view_items:
            payload = results.get(view_key)
            if isinstance(payload, dict):
                report["views"][view_key] = payload

        logger.info(
            "[analysis] write report_json=%s (while chart writes running=%d)",
            paths.report_json,
            len(chart_futures),
        )
        write_json(paths.report_json, report)

        logger.info("[analysis] waiting chart writes=%d", len(chart_futures))
        done = 0
        total = len(chart_futures)
        step2 = max(1, total // 20) if total > 0 else 1

        with chart_futures_lock:
            pending = list(chart_futures.keys())

        for fut in as_completed(pending):
            with chart_futures_lock:
                chart_id = chart_futures.get(fut, "<?>")
            try:
                _ = fut.result()
            except Exception:
                logger.exception("[analysis] chart write failed chart_id=%s", chart_id)
            finally:
                done += 1
                if total > 0:
                    _maybe_log_progress(
                        "charts_written:", done, total, step2, chart_started_at
                    )

    logger.info(
        "[analysis] build_html_report out_path=%s chart_specs=%d",
        paths.report_html,
        len(chart_specs),
    )
    build_html_report(
        report=report, chart_specs=chart_specs, out_path=paths.report_html
    )

    logger.warning("[analysis] done")
    return paths
