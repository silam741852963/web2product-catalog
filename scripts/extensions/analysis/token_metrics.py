from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .utils_stats import summarize_distribution

_WORDLIKE_TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _count_tokens_deterministic(text: str) -> int:
    return len(_WORDLIKE_TOK_RE.findall(text or ""))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_files(root: Path, *, suffix: str) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.rglob(f"*{suffix}") if p.is_file()])


@dataclass(frozen=True, slots=True)
class CompanyTokenInput:
    company_id: str
    industry_group_key: str
    html_dir: Path
    markdown_dir: Path
    product_dir: Path
    markdown_saved: int
    include_llm_tokens: bool  # when False: do not read product_dir at all


@dataclass(frozen=True, slots=True)
class CompanyTokenOutput:
    company_id: str
    industry_group_key: str
    html_tokens: int
    md_tokens: int
    md_file_count: int
    llm_output_tokens: int  # RENAMED: actual_output_tokens -> llm_output_tokens
    markdown_saved: int


def extract_company_token_metrics(inp: CompanyTokenInput) -> CompanyTokenOutput:
    html_tok = 0
    for p in _iter_files(inp.html_dir, suffix=".html"):
        html_tok += _count_tokens_deterministic(_read_text(p))

    md_files = _iter_files(inp.markdown_dir, suffix=".md")
    md_tok = 0
    for p in md_files:
        md_tok += _count_tokens_deterministic(_read_text(p))

    out_tok = 0
    if bool(inp.include_llm_tokens):
        for p in _iter_files(inp.product_dir, suffix=".json"):
            try:
                obj = json.loads(_read_text(p))
                s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
            except Exception:
                s = _read_text(p)
            out_tok += _count_tokens_deterministic(s)

    return CompanyTokenOutput(
        company_id=inp.company_id,
        industry_group_key=inp.industry_group_key,
        html_tokens=int(html_tok),
        md_tokens=int(md_tok),
        md_file_count=int(len(md_files)),
        llm_output_tokens=int(out_tok),
        markdown_saved=int(inp.markdown_saved),
    )


def aggregate_token_sections_for_view(
    *,
    company_ids_in_view: Iterable[str],
    by_company: Dict[str, CompanyTokenOutput],
    per_doc_overhead: int,
    prompt_overhead_by_industry: Dict[str, int],
    include_llm_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Token section is always present (md + pruning totals).
    When include_llm_metrics=False:
      - do not include llm_output_tokens distributions
      - do not include llm_output_tokens by_company mapping
    """
    ids = list(company_ids_in_view)

    rows: List[CompanyTokenOutput] = []
    for cid in ids:
        r = by_company.get(cid)
        if r is not None:
            rows.append(r)

    html_tokens_by_company: Dict[str, int] = {}
    md_tokens_by_company: Dict[str, int] = {}
    md_file_count_by_company: Dict[str, int] = {}
    markdown_saved_by_company: Dict[str, int] = {}
    llm_out_tokens_by_company: Dict[str, int] = {}

    for r in rows:
        html_tokens_by_company[r.company_id] = int(r.html_tokens)
        md_tokens_by_company[r.company_id] = int(r.md_tokens)
        md_file_count_by_company[r.company_id] = int(r.md_file_count)
        markdown_saved_by_company[r.company_id] = int(r.markdown_saved)
        if bool(include_llm_metrics):
            llm_out_tokens_by_company[r.company_id] = int(r.llm_output_tokens)

    html_total = sum(html_tokens_by_company.values())
    md_total = sum(md_tokens_by_company.values())
    pruned_total = max(0, int(html_total) - int(md_total))

    md_vals = [md_tokens_by_company.get(cid, 0) for cid in ids]

    distributions: Dict[str, Any] = {
        "md_tokens": {
            "summary": summarize_distribution(md_vals),
            "charts": {"bar": "md_tokens_bar", "hist": "md_tokens_hist"},
        }
    }

    if bool(include_llm_metrics):
        llm_vals = [llm_out_tokens_by_company.get(cid, 0) for cid in ids]
        distributions["llm_output_tokens"] = {
            "summary": summarize_distribution(llm_vals),
            "charts": {
                "bar": "llm_output_tokens_bar",
                "hist": "llm_output_tokens_hist",
            },
        }

    by_company_block: Dict[str, Any] = {
        "md_tokens": md_tokens_by_company,
        "html_tokens": html_tokens_by_company,
        "md_file_count": md_file_count_by_company,
        "markdown_saved": markdown_saved_by_company,
    }
    if bool(include_llm_metrics):
        by_company_block["llm_output_tokens"] = llm_out_tokens_by_company

    return {
        "tokens": {
            "tokenizer": {
                "name": "regex_wordlike_v1",
                "note": "Deterministic approximate token counter; stable for comparing pruning & distributions.",
            },
            "totals": {
                "md_tokens_total": int(md_total),
                "pruned_tokens_total": int(pruned_total),
            },
            "distributions": distributions,
            "by_company": by_company_block,
        }
    }
