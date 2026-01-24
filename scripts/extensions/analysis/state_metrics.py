from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .industry_views import IndustryView


_TERMINAL_STATUS_RE = re.compile(
    r"(terminal|doomed|dead|permanent|gave[_\s-]?up|abandon|skip)",
    re.IGNORECASE,
)
_CRAWLED_STATUS_RE = re.compile(
    r"(crawled|md_done|llm_done|done|finished|completed)",
    re.IGNORECASE,
)


def _pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round((num / den) * 100.0, 2)


@dataclass(frozen=True, slots=True)
class CompanyStateInput:
    company_id: str
    crawl_meta: dict | None
    url_index: dict | None


@dataclass(frozen=True, slots=True)
class CompanyStateOutput:
    company_id: str
    crawled: bool
    terminal_done: bool
    md_done: bool
    llm_done: bool


def extract_company_state_metrics(inp: CompanyStateInput) -> CompanyStateOutput:
    meta = inp.crawl_meta if isinstance(inp.crawl_meta, dict) else {}
    ui = inp.url_index if isinstance(inp.url_index, dict) else {}

    status = str(meta.get("status") or "")
    crawl_finished = bool(meta.get("crawl_finished") or False)

    terminal_done = bool(_TERMINAL_STATUS_RE.search(status))

    # "crawled" heuristic
    crawled = False
    if crawl_finished or _CRAWLED_STATUS_RE.search(status):
        crawled = True
    else:
        if isinstance(ui, dict):
            n_pages = sum(1 for k in ui.keys() if k != "__meta__")
            if n_pages > 0:
                crawled = True

    # md_done/llm_done: based on urls_total + done counts
    urls_total = int(meta.get("urls_total") or 0)
    urls_md = int(meta.get("urls_markdown_done") or 0)
    urls_llm = int(meta.get("urls_llm_done") or 0)

    if urls_total <= 0 and isinstance(ui, dict):
        urls_total = sum(1 for k in ui.keys() if k != "__meta__")

    md_done = bool(urls_total > 0 and urls_md >= urls_total)
    llm_done = bool(urls_total > 0 and urls_llm >= urls_total)

    return CompanyStateOutput(
        company_id=str(inp.company_id),
        crawled=bool(crawled),
        terminal_done=bool(terminal_done),
        md_done=bool(md_done),
        llm_done=bool(llm_done),
    )


def aggregate_company_status_section_for_view(
    *,
    company_ids_in_view: Iterable[str],
    by_company: Dict[str, CompanyStateOutput],
    global_state: Optional[dict] = None,
    is_global_view: bool = False,
) -> Dict[str, Any]:
    ids = list(company_ids_in_view)
    total_companies = len(ids)

    # Global preferred source
    if global_state is not None and bool(is_global_view):
        total_companies = int(
            global_state.get("total_companies", total_companies) or total_companies
        )
        crawled = int(global_state.get("crawled_companies", 0) or 0)
        terminal_done = int(global_state.get("terminal_done_companies", 0) or 0)

        md_done_pct = global_state.get("md_done_pct")
        llm_done_pct = global_state.get("llm_done_pct")

        md_done_pct_f = (
            float(md_done_pct) if isinstance(md_done_pct, (int, float)) else None
        )
        llm_done_pct_f = (
            float(llm_done_pct) if isinstance(llm_done_pct, (int, float)) else None
        )

        not_crawled = max(0, total_companies - crawled)

        # Fallback compute pct if absent
        if md_done_pct_f is None or llm_done_pct_f is None:
            md_done_cnt = 0
            llm_done_cnt = 0
            for cid in ids:
                r = by_company.get(cid)
                if r is None:
                    continue
                if r.md_done:
                    md_done_cnt += 1
                if r.llm_done:
                    llm_done_cnt += 1
            md_done_pct_f = _pct(md_done_cnt, total_companies)
            llm_done_pct_f = _pct(llm_done_cnt, total_companies)

        return {
            "company_status": {
                "total_companies": int(total_companies),
                "counts": {
                    "crawled": int(crawled),
                    "not_crawled": int(not_crawled),
                    "terminal_done": int(terminal_done),
                },
                "pcts": {
                    "crawled_pct": _pct(int(crawled), int(total_companies)),
                    "md_done_pct": float(md_done_pct_f),
                    "llm_done_pct": float(llm_done_pct_f),
                },
                "bars": [
                    {
                        "id": "crawled_ratio",
                        "label": "Crawled",
                        "num": int(crawled),
                        "den": int(total_companies),
                        "pct": _pct(int(crawled), int(total_companies)),
                    },
                    {
                        "id": "md_done_pct",
                        "label": "MD done",
                        "pct": float(md_done_pct_f),
                    },
                    {
                        "id": "llm_done_pct",
                        "label": "LLM done",
                        "pct": float(llm_done_pct_f),
                    },
                ],
                "pie": {
                    "chart_id": "company_status_pie",
                    "labels": ["Crawled", "Not crawled", "Terminal done"],
                    "values": [int(crawled), int(not_crawled), int(terminal_done)],
                },
            }
        }

    # Per-view derived
    crawled_cnt = 0
    terminal_done_cnt = 0
    md_done_cnt = 0
    llm_done_cnt = 0

    for cid in ids:
        r = by_company.get(cid)
        if r is None:
            continue
        if r.crawled:
            crawled_cnt += 1
        if r.terminal_done:
            terminal_done_cnt += 1
        if r.md_done:
            md_done_cnt += 1
        if r.llm_done:
            llm_done_cnt += 1

    not_crawled = max(0, total_companies - crawled_cnt)

    return {
        "company_status": {
            "total_companies": int(total_companies),
            "counts": {
                "crawled": int(crawled_cnt),
                "not_crawled": int(not_crawled),
                "terminal_done": int(terminal_done_cnt),
            },
            "pcts": {
                "crawled_pct": _pct(int(crawled_cnt), int(total_companies)),
                "md_done_pct": _pct(int(md_done_cnt), int(total_companies)),
                "llm_done_pct": _pct(int(llm_done_cnt), int(total_companies)),
            },
            "bars": [
                {
                    "id": "crawled_ratio",
                    "label": "Crawled",
                    "num": int(crawled_cnt),
                    "den": int(total_companies),
                    "pct": _pct(int(crawled_cnt), int(total_companies)),
                },
                {
                    "id": "md_done_pct",
                    "label": "MD done",
                    "pct": _pct(int(md_done_cnt), int(total_companies)),
                },
                {
                    "id": "llm_done_pct",
                    "label": "LLM done",
                    "pct": _pct(int(llm_done_cnt), int(total_companies)),
                },
            ],
            "pie": {
                "chart_id": "company_status_pie",
                "labels": ["Crawled", "Not crawled", "Terminal done"],
                "values": [int(crawled_cnt), int(not_crawled), int(terminal_done_cnt)],
            },
        }
    }


# -------------------------------------------------------------------
# Backwards-compatible wrapper (until __init__.py is wired to Option B)
# -------------------------------------------------------------------
def compute_company_status_section(
    view: IndustryView,
    *,
    global_state: Optional[dict] = None,
) -> Dict[str, Any]:
    by_company: Dict[str, CompanyStateOutput] = {}
    for c in view.companies:
        inp = CompanyStateInput(
            company_id=c.company_id,
            crawl_meta=getattr(c, "crawl_meta", None),
            url_index=getattr(c, "url_index", None),
        )
        by_company[c.company_id] = extract_company_state_metrics(inp)

    return aggregate_company_status_section_for_view(
        company_ids_in_view=[c.company_id for c in view.companies],
        by_company=by_company,
        global_state=global_state,
        is_global_view=(view.key == "__global__"),
    )
