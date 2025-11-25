from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # allow "Z"
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d not in (0, 0.0, None) else 0.0


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _categorize_status_code(code: Optional[int]) -> Tuple[str, str]:
    """
    Returns (bucket, label) where bucket is one of:
      "ok", "redirect", "client_error", "server_error", "other"
    """
    if code is None:
        return "other", "None"
    try:
        c = int(code)
    except Exception:
        return "other", str(code)
    if 200 <= c <= 299:
        return "ok", str(c)
    if 300 <= c <= 399:
        return "redirect", str(c)
    if 400 <= c <= 499:
        return "client_error", str(c)
    if 500 <= c <= 599:
        return "server_error", str(c)
    return "other", str(c)


# --------------------------------------------------------------------------- #
# Per-company models (old analyze.py)
# --------------------------------------------------------------------------- #


@dataclass
class CrawlMeta:
    company_id: str
    root_url: str
    status: str
    urls_total: int
    urls_markdown_done: int
    urls_llm_done: int
    last_error: Optional[str]
    last_crawled_at: Optional[str]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CrawlMeta":
        return cls(
            company_id=data.get("company_id", ""),
            root_url=data.get("root_url", ""),
            status=data.get("status", ""),
            urls_total=int(data.get("urls_total", 0) or 0),
            urls_markdown_done=int(data.get("urls_markdown_done", 0) or 0),
            urls_llm_done=int(data.get("urls_llm_done", 0) or 0),
            last_error=data.get("last_error"),
            last_crawled_at=data.get("last_crawled_at"),
        )


@dataclass
class URLStats:
    total_urls: int = 0
    status_code_ok: int = 0
    status_code_redirect: int = 0
    status_code_client_error: int = 0
    status_code_server_error: int = 0
    status_code_other: int = 0

    error_count: int = 0

    gating_accept_true: int = 0
    gating_accept_false: int = 0

    presence_positive: int = 0  # presence == 1
    presence_zero: int = 0  # presence == 0 / missing

    extracted_positive: int = 0  # extracted > 0 (or == 1 if binary)
    extracted_zero: int = 0

    markdown_saved: int = 0  # status == "markdown_saved"
    markdown_other_status: int = 0

    md_total_words_sum: float = 0.0
    md_total_words_min: float = 0.0
    md_total_words_max: float = 0.0


@dataclass
class CompanyStats:
    company_id: str
    root_url: str

    # From crawl_meta.json
    crawl_status: str
    urls_total_meta: int
    urls_markdown_done_meta: int
    urls_llm_done_meta: int
    last_error: Optional[str]
    last_crawled_at: Optional[str]

    # From url_index.json aggregate
    url_stats: URLStats

    # Simple consistency flags
    urls_count_mismatch: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten URLStats for convenience in CSV/JSON
        url_stats = d.pop("url_stats", {})
        for k, v in url_stats.items():
            d[f"url_{k}"] = v
        return d


# --------------------------------------------------------------------------- #
# Global-aggregate models (adapted stat_analyzer.py)
# --------------------------------------------------------------------------- #


@dataclass
class CompanyRow:
    company_id: str
    root_url: str
    status: str

    urls_total: int
    urls_markdown_done: int
    urls_llm_done: int

    # URL-index level stats (per company, aggregated)
    url_count: int
    url_status_ok: int
    url_status_redirect: int
    url_status_client_error: int
    url_status_server_error: int
    url_status_other: int

    url_error_count: int

    gating_accept_true: int
    gating_accept_false: int

    presence_positive: int
    presence_zero: int

    extracted_positive: int
    extracted_zero: int

    markdown_saved: int
    markdown_suppressed: int
    markdown_other_status: int

    md_words_files: int  # how many url entries have md_total_words
    md_words_total: int
    md_words_mean_per_file: float
    md_words_median_per_file: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# File discovery for new structure
# --------------------------------------------------------------------------- #


def discover_companies(outputs_root: Path) -> List[Path]:
    """
    Discover company folders under outputs_root that contain metadata/crawl_meta.json.
    """
    if not outputs_root.exists():
        return []
    companies: List[Path] = []
    for child in outputs_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "metadata" / "crawl_meta.json").exists():
            companies.append(child)
    return sorted(companies)


def find_crawl_meta_files(outputs_root: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    Find all crawl_meta.json + url_index.json pairs under outputs_root/*/metadata/.
    Returns list of tuples (crawl_meta_path, url_index_path_or_None).
    """
    out: List[Tuple[Path, Optional[Path]]] = []
    for company_dir in discover_companies(outputs_root):
        meta_dir = company_dir / "metadata"
        meta_path = meta_dir / "crawl_meta.json"
        if not meta_path.exists():
            continue
        url_index = meta_dir / "url_index.json"
        out.append((meta_path, url_index if url_index.exists() else None))
    return out


def load_crawl_meta(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_url_index(fp: Optional[Path]) -> Dict[str, Any]:
    if not fp:
        return {}
    try:
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Per-company URL index aggregation (old analyze.py)
# --------------------------------------------------------------------------- #


def _aggregate_url_index_for_company(data: Dict[str, Any]) -> URLStats:
    stats = URLStats()
    first_word_value: Optional[float] = None

    for _url, rec in data.items():
        stats.total_urls += 1

        status_code = rec.get("status_code")
        bucket, _label = _categorize_status_code(status_code)

        if bucket == "ok":
            stats.status_code_ok += 1
        elif bucket == "redirect":
            stats.status_code_redirect += 1
        elif bucket == "client_error":
            stats.status_code_client_error += 1
        elif bucket == "server_error":
            stats.status_code_server_error += 1
        else:
            stats.status_code_other += 1

        # Error field (string) – treat non-empty as error
        err = rec.get("error") or ""
        if isinstance(err, str) and err.strip():
            stats.error_count += 1

        # Gating fields
        gating_accept = rec.get("gating_accept")
        if gating_accept is True:
            stats.gating_accept_true += 1
        elif gating_accept is False:
            stats.gating_accept_false += 1

        # Presence (0/1 binary)
        presence = rec.get("presence")
        if presence == 1:
            stats.presence_positive += 1
        else:
            stats.presence_zero += 1

        # Extracted (0/1 binary or count)
        extracted = rec.get("extracted")
        if isinstance(extracted, (int, float)) and extracted > 0:
            stats.extracted_positive += 1
        else:
            stats.extracted_zero += 1

        # Markdown status
        status = rec.get("status", "")
        if status == "markdown_saved":
            stats.markdown_saved += 1
        else:
            stats.markdown_other_status += 1

        # Word counts: use md_total_words from new url_index.json
        md_words = rec.get("md_total_words")
        if md_words is not None:
            try:
                w = float(md_words)
            except Exception:
                w = 0.0
            stats.md_total_words_sum += w
            if first_word_value is None:
                first_word_value = w
                stats.md_total_words_min = w
                stats.md_total_words_max = w
            else:
                if w < stats.md_total_words_min:
                    stats.md_total_words_min = w
                if w > stats.md_total_words_max:
                    stats.md_total_words_max = w

    return stats


def _load_company_stats(company_dir: Path) -> Optional[CompanyStats]:
    """
    Load crawl_meta.json and url_index.json from:
        {company_dir}/metadata/
    """
    meta_dir = company_dir / "metadata"
    crawl_meta_path = meta_dir / "crawl_meta.json"
    url_index_path = meta_dir / "url_index.json"

    crawl_data = _load_json(crawl_meta_path)
    url_index_data = _load_json(url_index_path)

    if not crawl_data and not url_index_data:
        # Nothing to analyze for this company
        return None

    crawl_meta = CrawlMeta.from_json(crawl_data or {})

    if url_index_data:
        url_stats = _aggregate_url_index_for_company(url_index_data)
    else:
        url_stats = URLStats()

    urls_count_mismatch = False
    if crawl_meta.urls_total and url_stats.total_urls:
        urls_count_mismatch = crawl_meta.urls_total != url_stats.total_urls

    return CompanyStats(
        company_id=crawl_meta.company_id or company_dir.name,
        root_url=crawl_meta.root_url,
        crawl_status=crawl_meta.status,
        urls_total_meta=crawl_meta.urls_total,
        urls_markdown_done_meta=crawl_meta.urls_markdown_done,
        urls_llm_done_meta=crawl_meta.urls_llm_done,
        last_error=crawl_meta.last_error,
        last_crawled_at=crawl_meta.last_crawled_at,
        url_stats=url_stats,
        urls_count_mismatch=urls_count_mismatch,
    )


def analyze_companies(
    outputs_root: Path,
    company_ids: Optional[List[str]] = None,
) -> List[CompanyStats]:
    """
    Analyze either specific company IDs (by folder name) or all discovered.
    """
    if company_ids:
        company_dirs = [outputs_root / cid for cid in company_ids]
    else:
        company_dirs = discover_companies(outputs_root)

    results: List[CompanyStats] = []
    for cdir in company_dirs:
        stats = _load_company_stats(cdir)
        if stats:
            results.append(stats)
    return results


# --------------------------------------------------------------------------- #
# Global aggregation (adapted stat_analyzer.py to new structure)
# --------------------------------------------------------------------------- #


def normalize_crawl_meta(doc: Dict[str, Any], url_index: Dict[str, Any]) -> CompanyRow:
    """
    Normalizes the *new* crawl_meta.json + url_index.json format into CompanyRow.
    - New crawl_meta.json is very simple (no seeding / saves).
    - url_index.json carries per-URL status, gating, md_total_words etc.
    """
    company_id = str(doc.get("company_id") or "")
    root_url = str(doc.get("root_url") or "")
    status = str(doc.get("status") or "")

    urls_total = int(doc.get("urls_total") or 0)
    urls_markdown_done = int(doc.get("urls_markdown_done") or 0)
    urls_llm_done = int(doc.get("urls_llm_done") or 0)

    # Aggregate url_index.json
    url_count = 0
    url_status_ok = 0
    url_status_redirect = 0
    url_status_client_error = 0
    url_status_server_error = 0
    url_status_other = 0

    url_error_count = 0

    gating_accept_true = 0
    gating_accept_false = 0

    presence_positive = 0
    presence_zero = 0

    extracted_positive = 0
    extracted_zero = 0

    markdown_saved = 0
    markdown_suppressed = 0
    markdown_other_status = 0

    md_word_vals: List[float] = []

    if url_index and isinstance(url_index, dict):
        for _url, rec in url_index.items():
            url_count += 1

            status_code = rec.get("status_code")
            bucket, _label = _categorize_status_code(status_code)
            if bucket == "ok":
                url_status_ok += 1
            elif bucket == "redirect":
                url_status_redirect += 1
            elif bucket == "client_error":
                url_status_client_error += 1
            elif bucket == "server_error":
                url_status_server_error += 1
            else:
                url_status_other += 1

            err = rec.get("error") or ""
            if isinstance(err, str) and err.strip():
                url_error_count += 1

            gating_accept = rec.get("gating_accept")
            if gating_accept is True:
                gating_accept_true += 1
            elif gating_accept is False:
                gating_accept_false += 1

            presence = rec.get("presence")
            if presence == 1:
                presence_positive += 1
            else:
                presence_zero += 1

            extracted = rec.get("extracted")
            if isinstance(extracted, (int, float)) and extracted > 0:
                extracted_positive += 1
            else:
                extracted_zero += 1

            status_field = rec.get("status", "")
            if status_field == "markdown_saved":
                markdown_saved += 1
            elif status_field == "markdown_suppressed":
                markdown_suppressed += 1
            else:
                markdown_other_status += 1

            md_words = rec.get("md_total_words")
            if md_words is not None:
                try:
                    w = float(md_words)
                except Exception:
                    w = 0.0
                if w >= 0:
                    md_word_vals.append(w)

    md_words_files = len(md_word_vals)
    md_words_total = int(sum(md_word_vals)) if md_word_vals else 0
    md_words_mean_per_file = float(md_words_total / md_words_files) if md_word_vals else 0.0
    md_words_median_per_file = float(np.median(md_word_vals)) if md_word_vals else 0.0

    return CompanyRow(
        company_id=company_id,
        root_url=root_url,
        status=status,
        urls_total=urls_total,
        urls_markdown_done=urls_markdown_done,
        urls_llm_done=urls_llm_done,
        url_count=url_count,
        url_status_ok=url_status_ok,
        url_status_redirect=url_status_redirect,
        url_status_client_error=url_status_client_error,
        url_status_server_error=url_status_server_error,
        url_status_other=url_status_other,
        url_error_count=url_error_count,
        gating_accept_true=gating_accept_true,
        gating_accept_false=gating_accept_false,
        presence_positive=presence_positive,
        presence_zero=presence_zero,
        extracted_positive=extracted_positive,
        extracted_zero=extracted_zero,
        markdown_saved=markdown_saved,
        markdown_suppressed=markdown_suppressed,
        markdown_other_status=markdown_other_status,
        md_words_files=md_words_files,
        md_words_total=md_words_total,
        md_words_mean_per_file=md_words_mean_per_file,
        md_words_median_per_file=md_words_median_per_file,
    )


def collect_dataframe(crawl_meta_files: Iterable[Tuple[Path, Optional[Path]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for meta_path, url_index_path in crawl_meta_files:
        try:
            doc = load_crawl_meta(meta_path)
            url_index = load_url_index(url_index_path) if url_index_path else {}
            row = normalize_crawl_meta(doc, url_index)
            d = row.as_dict()
            d["_meta_path"] = str(meta_path)
            d["_url_index_path"] = str(url_index_path) if url_index_path else None
            rows.append(d)
        except Exception as e:
            rows.append(
                {
                    "company_id": f"__PARSE_ERROR__:{meta_path.name}",
                    "root_url": "",
                    "status": "unknown",
                    "urls_total": 0,
                    "urls_markdown_done": 0,
                    "urls_llm_done": 0,
                    "url_count": 0,
                    "url_status_ok": 0,
                    "url_status_redirect": 0,
                    "url_status_client_error": 0,
                    "url_status_server_error": 0,
                    "url_status_other": 0,
                    "url_error_count": 0,
                    "gating_accept_true": 0,
                    "gating_accept_false": 0,
                    "presence_positive": 0,
                    "presence_zero": 0,
                    "extracted_positive": 0,
                    "extracted_zero": 0,
                    "markdown_saved": 0,
                    "markdown_suppressed": 0,
                    "markdown_other_status": 0,
                    "md_words_files": 0,
                    "md_words_total": 0,
                    "md_words_mean_per_file": 0.0,
                    "md_words_median_per_file": 0.0,
                    "_meta_path": str(meta_path),
                    "_url_index_path": str(url_index_path) if url_index_path else None,
                    "_error": str(e),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    numeric_cols = [
        "urls_total",
        "urls_markdown_done",
        "urls_llm_done",
        "url_count",
        "url_status_ok",
        "url_status_redirect",
        "url_status_client_error",
        "url_status_server_error",
        "url_status_other",
        "url_error_count",
        "gating_accept_true",
        "gating_accept_false",
        "presence_positive",
        "presence_zero",
        "extracted_positive",
        "extracted_zero",
        "markdown_saved",
        "markdown_suppressed",
        "markdown_other_status",
        "md_words_files",
        "md_words_total",
        "md_words_mean_per_file",
        "md_words_median_per_file",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _percentiles(series: pd.Series, cuts=(50, 75, 80, 90, 95, 97, 99)) -> Dict[str, float]:
    out = {}
    clean = series.dropna()
    for p in cuts:
        out[f"p{p}"] = float(np.percentile(clean, p)) if len(clean) else 0.0
    return out


def compute_summary(
    df: pd.DataFrame,
    cost_per_1k_tokens_usd: float = 0.03,
) -> Dict[str, Any]:
    """
    Create a JSON-serializable summary of the dataset, focused on:
    - page counts (urls_total)
    - markdown word volume & LLM cost estimates
    - basic gating / extraction coverage
    """
    n = len(df)

    urls_total_series = df["urls_total"] if "urls_total" in df else pd.Series(dtype=float)
    md_words_total_series = df["md_words_total"] if "md_words_total" in df else pd.Series(dtype=float)

    summary: Dict[str, Any] = {
        "companies_count": n,
        "urls_total_stats": {
            "count": int(urls_total_series.count()) if not urls_total_series.empty else 0,
            "mean": float(urls_total_series.mean()) if not urls_total_series.empty else 0.0,
            "median": float(urls_total_series.median()) if not urls_total_series.empty else 0.0,
            **(_percentiles(urls_total_series) if not urls_total_series.empty else {}),
            "sum": float(urls_total_series.sum()) if not urls_total_series.empty else 0.0,
        },
        "md_words_total_stats": {
            "count": int(md_words_total_series.count()) if not md_words_total_series.empty else 0,
            "mean": float(md_words_total_series.mean()) if not md_words_total_series.empty else 0.0,
            "median": float(md_words_total_series.median()) if not md_words_total_series.empty else 0.0,
            **(_percentiles(md_words_total_series) if not md_words_total_series.empty else {}),
            "sum": float(md_words_total_series.sum()) if not md_words_total_series.empty else 0.0,
        },
    }

    # Markdown / LLM cost estimates
    try:
        md_words_sum = float(md_words_total_series.sum()) if not md_words_total_series.empty else 0.0
        token_per_word = 1.3333333  # ~ 1 token ≈ 0.75 words → inverse ~1.3333
        estimated_total_tokens = md_words_sum * token_per_word
        estimated_cost_usd_total = (estimated_total_tokens / 1000.0) * float(cost_per_1k_tokens_usd)
        estimated_cost_usd_mean_per_company = (
            ((md_words_sum / max(1, len(df))) * token_per_word / 1000.0) * float(cost_per_1k_tokens_usd)
        )

        summary["markdown"] = {
            "md_words_total_sum": float(md_words_sum),
            "md_words_mean_per_company": float(md_words_total_series.mean()) if not md_words_total_series.empty else 0.0,
            "md_words_median_per_company": float(md_words_total_series.median()) if not md_words_total_series.empty else 0.0,
            "estimated_total_tokens": float(estimated_total_tokens),
            "estimated_cost_usd_total_at_specified_rate": float(estimated_cost_usd_total),
            "estimated_cost_usd_mean_per_company_at_specified_rate": float(estimated_cost_usd_mean_per_company),
            "token_per_word_assumption": float(token_per_word),
            "cost_per_1k_tokens_usd_used": float(cost_per_1k_tokens_usd),
        }
    except Exception:
        summary["markdown"] = {
            "md_words_total_sum": 0.0,
            "estimated_cost_usd_total_at_specified_rate": 0.0,
        }

    # Basic coverage stats (gating, presence, extraction)
    if "gating_accept_true" in df and "gating_accept_false" in df:
        total_urls = (df["gating_accept_true"] + df["gating_accept_false"]).sum()
        if total_urls > 0:
            gating_accept_rate = float(df["gating_accept_true"].sum() / total_urls)
        else:
            gating_accept_rate = 0.0
    else:
        gating_accept_rate = 0.0

    presence_pos = df["presence_positive"].sum() if "presence_positive" in df else 0
    presence_zero = df["presence_zero"].sum() if "presence_zero" in df else 0
    extracted_pos = df["extracted_positive"].sum() if "extracted_positive" in df else 0
    extracted_zero = df["extracted_zero"].sum() if "extracted_zero" in df else 0

    summary["coverage"] = {
        "gating_accept_rate": gating_accept_rate,
        "presence_positive_total": int(presence_pos),
        "presence_zero_total": int(presence_zero),
        "extracted_positive_total": int(extracted_pos),
        "extracted_zero_total": int(extracted_zero),
    }

    return summary


# --------------------------------------------------------------------------- #
# Plotting (matplotlib, simple distributions)
# --------------------------------------------------------------------------- #


def _savefig(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_hist_urls_total(df: pd.DataFrame, out: Path, bins: int = 40):
    if "urls_total" not in df:
        return
    series = df["urls_total"].dropna()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(series, bins=bins)
    ax.set_title("Distribution of urls_total per company")
    ax.set_xlabel("urls_total")
    ax.set_ylabel("companies")
    for p in (50, 75, 90, 95, 97, 99):
        v = np.percentile(series, p) if len(series) else 0
        ax.axvline(v, linestyle="--")
        ax.text(v, ax.get_ylim()[1] * 0.9, f"p{p}={int(v)}", rotation=90, va="top")
    _savefig(fig, out)


def plot_hist_md_words_total(df: pd.DataFrame, out: Path, bins: int = 40):
    if "md_words_total" not in df:
        return
    series = df["md_words_total"].dropna()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(series, bins=bins)
    ax.set_title("Distribution of total markdown words per company")
    ax.set_xlabel("total markdown words (company)")
    ax.set_ylabel("companies")
    for p in (50, 75, 90, 95):
        v = np.percentile(series, p) if len(series) else 0
        ax.axvline(v, linestyle="--")
        ax.text(v, ax.get_ylim()[1] * 0.9, f"p{p}={int(v)}", rotation=90, va="top")
    _savefig(fig, out)


# --------------------------------------------------------------------------- #
# Plotly interactive (optional)
# --------------------------------------------------------------------------- #


def _write_html(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


def iplot_hist_urls_total(df: pd.DataFrame, out: Path):
    if "urls_total" not in df or df["urls_total"].dropna().empty:
        return
    s = df[["company_id", "urls_total"]].dropna()
    fig = px.histogram(s, x="urls_total", nbins=40, title="Distribution of urls_total per company")
    for p in (80, 90, 95, 97, 99):
        v = float(np.percentile(s["urls_total"], p))
        fig.add_vline(x=v, line_dash="dash", annotation_text=f"p{p}={int(v)}", annotation_position="top")
    _write_html(fig, out)


def iplot_hist_md_words_total(df: pd.DataFrame, out: Path):
    if "md_words_total" not in df or df["md_words_total"].dropna().empty:
        return
    s = df[["company_id", "md_words_total"]].dropna()
    fig = px.histogram(
        s,
        x="md_words_total",
        nbins=40,
        title="Distribution of total markdown words per company",
    )
    _write_html(fig, out)


# --------------------------------------------------------------------------- #
# Global orchestrator (aggregate) - replaces old stat_analyzer CLI
# --------------------------------------------------------------------------- #


def analyze_and_plot(
    outputs_dir: Path,
    out_dir: Path,
    save_csv: bool = True,
    save_json_summary: bool = True,
    interactive: bool = False,
    cost_per_1k_tokens_usd: float = 0.03,
) -> Dict[str, Any]:
    """
    Scan outputs_dir for all metadata/{crawl_meta.json,url_index.json} files,
    aggregate, plot, and write artifacts.

    Returns a dict summary (also optionally saved to JSON).
    """
    crawl_meta_files = find_crawl_meta_files(outputs_dir)
    df = collect_dataframe(crawl_meta_files)

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        empty_summary = {"companies_count": 0, "message": "No crawl_meta.json found"}
        if save_json_summary:
            (out_dir / "summary.json").write_text(
                json.dumps(empty_summary, indent=2),
                encoding="utf-8",
            )
        return empty_summary

    # Save detailed CSV
    if save_csv:
        df.to_csv(out_dir / "crawl_meta_aggregate.csv", index=False, encoding="utf-8")

    # Compute summary + cost estimates
    summary = compute_summary(df, cost_per_1k_tokens_usd=cost_per_1k_tokens_usd)
    if save_json_summary:
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    # Charts
    plot_hist_urls_total(df, out_dir / "hist_urls_total.png")
    plot_hist_md_words_total(df, out_dir / "hist_md_words_total.png")

    # Interactive charts (Plotly HTML)
    if interactive:
        iout = out_dir / "interactive"
        iplot_hist_urls_total(df, iout / "hist_urls_total.html")
        iplot_hist_md_words_total(df, iout / "hist_md_words_total.html")
        summary["interactive_dir"] = str(iout.resolve())

    # add artifact locations
    summary["artifacts_dir"] = str(out_dir.resolve())
    summary["crawl_meta_aggregate"] = (
        str((out_dir / "crawl_meta_aggregate.csv").resolve()) if save_csv else None
    )
    summary["summary_json"] = (
        str((out_dir / "summary.json").resolve()) if save_json_summary else None
    )

    return summary


# --------------------------------------------------------------------------- #
# Global state helper (crawl_global_state.json)
# --------------------------------------------------------------------------- #


def _load_global_state(outputs_root: Path) -> Optional[Dict[str, Any]]:
    path = outputs_root / "crawl_global_state.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _print_global_state(outputs_root: Path) -> None:
    gs = _load_global_state(outputs_root)
    if not gs:
        return
    print("=" * 80)
    print("Global crawl state (from crawl_global_state.json):")
    print(f"  generated_at:        {gs.get('generated_at')}")
    print(f"  total_companies:     {gs.get('total_companies')}")
    print(f"  crawled_companies:   {gs.get('crawled_companies')}")
    print(f"  completed_companies: {gs.get('completed_companies')}")
    print(f"  percentage_completed:{gs.get('percentage_completed')}")
    by_status = gs.get("by_status") or {}
    if by_status:
        print("  by_status:")
        for k, v in by_status.items():
            print(f"    {k}: {v}")
    latest = gs.get("latest_run") or {}
    if latest:
        print("  latest_run:")
        for k, v in latest.items():
            print(f"    {k}: {v}")
    print()


# --------------------------------------------------------------------------- #
# Per-company reporting helpers
# --------------------------------------------------------------------------- #


def _print_company_stats(stats: CompanyStats) -> None:
    s = stats
    u = s.url_stats

    print("=" * 80)
    print(f"Company: {s.company_id}")
    print(f"Root URL: {s.root_url}")
    print(f"Crawl status: {s.crawl_status}")
    print(f"Last crawled at: {s.last_crawled_at}")
    print(f"Last error: {s.last_error!r}")
    print()

    print("--- Meta counts ---")
    print(f"  urls_total (meta):          {s.urls_total_meta}")
    print(f"  urls_markdown_done (meta):  {s.urls_markdown_done_meta}")
    print(f"  urls_llm_done (meta):       {s.urls_llm_done_meta}")
    print()

    print("--- URL index aggregate ---")
    print(f"  total_urls (index):         {u.total_urls}")
    if s.urls_count_mismatch:
        print("  WARNING: urls_total (meta) != total_urls (index)")

    print()
    print("  Status codes:")
    print(f"    2xx OK:                   {u.status_code_ok}")
    print(f"    3xx Redirect:             {u.status_code_redirect}")
    print(f"    4xx Client error:         {u.status_code_client_error}")
    print(f"    5xx Server error:         {u.status_code_server_error}")
    print(f"    Other / missing:          {u.status_code_other}")

    print()
    print("  Errors:")
    print(f"    Non-empty error field:    {u.error_count}")

    print()
    print("  Gating:")
    print(f"    gating_accept=True:       {u.gating_accept_true}")
    print(f"    gating_accept=False:      {u.gating_accept_false}")

    print()
    print("  Presence (LLM classifier):")
    print(f"    presence == 1:            {u.presence_positive}")
    print(f"    presence == 0/None:       {u.presence_zero}")

    print()
    print("  Extraction:")
    print(f"    extracted > 0:            {u.extracted_positive}")
    print(f"    extracted == 0/None:      {u.extracted_zero}")

    print()
    print("  Markdown status:")
    print(f"    status == 'markdown_saved': {u.markdown_saved}")
    print(f"    other statuses:             {u.markdown_other_status}")

    print()
    print("  Markdown word counts (md_total_words):")
    print(f"    sum:                      {u.md_total_words_sum:.1f}")
    if u.total_urls > 0:
        avg = u.md_total_words_sum / max(u.total_urls, 1)
        print(f"    avg per URL:              {avg:.1f}")
    else:
        print("    avg per URL:              n/a")
    print(f"    min:                      {u.md_total_words_min:.1f}")
    print(f"    max:                      {u.md_total_words_max:.1f}")
    print()


def _write_json(out_path: Path, stats_list: List[CompanyStats]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.to_dict() for s in stats_list]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(out_path: Path, stats_list: List[CompanyStats]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not stats_list:
        out_path.write_text("", encoding="utf-8")
        return
    rows = [s.to_dict() for s in stats_list]
    fieldnames = sorted(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze crawl metadata for one or more companies, and optionally "
            "produce global aggregate statistics and plots.\n"
            "Assumes structure: outputs/{company_id}/metadata/{crawl_meta.json,url_index.json} "
            "and optional outputs/crawl_global_state.json"
        )
    )
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing per-company folders (default: ./outputs).",
    )
    p.add_argument(
        "--company-id",
        action="append",
        default=[],
        help=(
            "Specific company ID(s) to analyze (folder names under outputs_root). "
            "Can be repeated. If omitted, all discovered companies are analyzed."
        ),
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write per-company JSON summary (list of companies).",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write per-company CSV summary (list of companies).",
    )
    p.add_argument(
        "--no-print",
        dest="do_print",
        action="store_false",
        help="Do not print per-company human-readable report to stdout.",
    )
    p.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help=(
            "If set, run global aggregate analysis over all companies and write "
            "summary + plots into this directory."
        ),
    )
    p.add_argument(
        "--aggregate-cost-per-1k-tokens-usd",
        type=float,
        default=0.03,
        help="Token cost used for LLM cost estimate in aggregate analysis (default: 0.03).",
    )
    p.add_argument(
        "--aggregate-interactive",
        action="store_true",
        help="If set, generate Plotly interactive HTML charts (requires plotly).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    outputs_root = args.outputs_root

    # Global crawl state overview (if available)
    _print_global_state(outputs_root)

    # Per-company stats
    stats_list = analyze_companies(
        outputs_root=outputs_root,
        company_ids=args.company_id or None,
    )

    if args.do_print:
        print(f"Found {len(stats_list)} compan(ies) to analyze.\n")
        for s in stats_list:
            _print_company_stats(s)

    if args.json_out:
        _write_json(args.json_out, stats_list)
        print(f"Wrote per-company JSON summary: {args.json_out}")

    if args.csv_out:
        _write_csv(args.csv_out, stats_list)
        print(f"Wrote per-company CSV summary: {args.csv_out}")

    # Global aggregate analysis (merged from old stat_analyzer)
    if args.aggregate_dir is not None:
        summary = analyze_and_plot(
            outputs_dir=outputs_root,
            out_dir=args.aggregate_dir,
            save_csv=True,
            save_json_summary=True,
            interactive=args.aggregate_interactive,
            cost_per_1k_tokens_usd=args.aggregate_cost_per_1k_tokens_usd,
        )
        print(f"\nAggregate analysis written to: {args.aggregate_dir}")
        print("Aggregate summary (key fields):")
        print(json.dumps(summary.get("urls_total_stats", {}), indent=2))


if __name__ == "__main__":
    main()