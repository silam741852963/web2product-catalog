from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

# ---------------------------
# Parsing helpers
# ---------------------------

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


def _get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_crawl_meta_files(outputs_root: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    Recursively find all crawl_meta.json files under outputs_root/**/checkpoints/.
    Returns list of tuples (crawl_meta_path, url_index_path_if_exists_or_None)
    """
    out: List[Tuple[Path, Optional[Path]]] = []
    for meta_path in sorted(outputs_root.rglob("checkpoints/crawl_meta.json")):
        url_index = meta_path.with_name("url_index.json")
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

# ---------------------------
# Normalization
# ---------------------------

@dataclass
class CompanyRow:
    bvdid: str
    company_name: Optional[str]
    stage: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    duration_sec: float
    urls_total: int
    urls_done: int
    urls_failed: int

    # Seeding stats
    seeding_source: Optional[str]
    discovered_total: int
    live_check_enabled: bool
    live_checked_total: int
    prefilter_kept: int
    prefilter_dropped_status: int
    prefilter_dropped_external: int
    prefilter_dropped_score: int
    filtered_total: int
    filtered_dropped_patterns_lang: int
    seed_roots_count: int
    seed_brand_count: int
    company_cap: Optional[int]
    company_cap_applied: bool

    # Per-root counts (top-line shape)
    per_root_max_discovered: int
    per_root_mean_discovered: float

    # Save stats (markdown gate etc.)
    saved_html_total: int
    saved_md_total: int
    saved_json_total: int
    md_suppressed_total: int
    resumed_skips: int

    # Markdown words summary (from url_index.json entries for this company)
    md_words_count: int            # number of markdown files with a words value
    md_words_total: int
    md_words_mean: float
    md_words_median: float

    # Derived ratios
    drop_ratio_prefilter: float           # 1 - prefilter_kept / discovered_total
    drop_ratio_patterns_lang: float       # filtered_dropped_patterns_lang / discovered_total
    drop_ratio_total: float               # 1 - filtered_total / discovered_total
    md_suppression_rate: float            # md_suppressed_total / (saved_md_total + md_suppressed_total)
    throughput_urls_per_min: float        # urls_total / minutes

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # make datetimes ISO when serializing out to CSV/JSON
        if self.started_at:
            d["started_at"] = self.started_at.isoformat()
        if self.finished_at:
            d["finished_at"] = self.finished_at.isoformat()
        return d


def normalize_crawl_meta(doc: Dict[str, Any], url_index: Dict[str, Any]) -> CompanyRow:
    """
    Normalizes the new crawl_meta.json format into the CompanyRow.
    url_index is optional and used to enrich some counts if available.
    """
    bvdid = str(doc.get("bvdid") or "")
    company_name = doc.get("company_name")
    stage = str(doc.get("stage") or "")

    started_at = _parse_dt(doc.get("started_at"))
    finished_at = _parse_dt(doc.get("finished_at"))
    duration_sec = (
        (finished_at - started_at).total_seconds()
        if (started_at and finished_at and finished_at >= started_at)
        else 0.0
    )

    urls_total = int(doc.get("urls_total") or 0)
    urls_done = int(doc.get("urls_done") or 0)
    urls_failed = int(doc.get("urls_failed") or 0)

    # Seeding block (all optional / default to 0)
    seeding = doc.get("seeding") or {}
    seeding_source = seeding.get("seeding_source") or doc.get("seed_source")

    discovered_total = int(seeding.get("discovered_total") or 0)
    live_check_enabled = bool(seeding.get("live_check_enabled") or seeding.get("live_check", False) or False)
    live_checked_total = int(seeding.get("live_checked_total") or 0)
    prefilter_kept = int(seeding.get("prefilter_kept") or 0)
    prefilter_dropped_status = int(seeding.get("prefilter_dropped_status") or 0)
    prefilter_dropped_external = int(seeding.get("prefilter_dropped_external") or 0)
    prefilter_dropped_score = int(seeding.get("prefilter_dropped_score") or 0)

    filtered_total = int(seeding.get("filtered_total") or urls_total or 0)
    filtered_dropped_patterns_lang = int(seeding.get("filtered_dropped_patterns_lang") or 0)

    seed_roots = seeding.get("seed_roots") or doc.get("seed_roots") or []
    seed_brand_count = int(seeding.get("seed_brand_count") or doc.get("seed_brand_count") or 0)

    per_root_discovered = seeding.get("per_root_discovered") or {}
    if isinstance(per_root_discovered, dict) and per_root_discovered:
        vals = list(int(v or 0) for v in per_root_discovered.values())
        per_root_max_discovered = int(max(vals))
        per_root_mean_discovered = float(sum(vals) / len(vals))
    else:
        per_root_max_discovered = 0
        per_root_mean_discovered = 0.0

    company_cap = seeding.get("company_cap")
    if company_cap is not None:
        try:
            company_cap = int(company_cap)
        except Exception:
            company_cap = None
    company_cap_applied = bool(seeding.get("company_cap_applied") or False)

    # Save stats (optional)
    saves = doc.get("saves") or {}
    saved_html_total = int(saves.get("saved_html_total") or 0)
    saved_md_total = int(saves.get("saved_md_total") or 0)
    saved_json_total = int(saves.get("saved_json_total") or 0)
    md_suppressed_total = int(saves.get("md_suppressed_total") or 0)
    resumed_skips = int(doc.get("resumed_skips") or 0)

    # If url_index provided, we can override some derived counts if that's more authoritative
    if url_index and isinstance(url_index, dict):
        # count url_index entries
        try:
            url_index_count = len(url_index)
            # If filtered_total is zero but url_index exists, set filtered_total to url_index_count
            if filtered_total == 0:
                filtered_total = url_index_count
        except Exception:
            pass

    # Gather markdown word counts from url_index (if provided)
    md_word_vals: List[int] = []
    if url_index and isinstance(url_index, dict):
        try:
            for u_ent in url_index.values():
                mw = None
                # Prefer explicit recorded markdown_words
                try:
                    mw = u_ent.get("markdown_words")
                except Exception:
                    mw = None

                if mw is None:
                    # if missing, try to estimate by reading markdown_path (best-effort)
                    mdp = u_ent.get("markdown_path")
                    if mdp:
                        try:
                            text = Path(str(mdp)).read_text(encoding="utf-8", errors="ignore")
                            # simple whitespace tokenization to count words
                            words = [w for w in re.split(r"\s+", text.strip()) if w]
                            mw = len(words)
                        except Exception:
                            mw = None
                if mw is not None:
                    try:
                        mw_i = int(mw)
                        if mw_i >= 0:
                            md_word_vals.append(mw_i)
                    except Exception:
                        # ignore malformed
                        pass
        except Exception:
            pass

    md_words_count = int(len(md_word_vals))
    md_words_total = int(sum(md_word_vals)) if md_word_vals else 0
    md_words_mean = float(sum(md_word_vals) / len(md_word_vals)) if md_word_vals else 0.0
    md_words_median = float(np.median(md_word_vals)) if md_word_vals else 0.0

    # Derived
    drop_ratio_prefilter = 1.0 - _safe_div(prefilter_kept, max(1, discovered_total))
    drop_ratio_patterns_lang = _safe_div(filtered_dropped_patterns_lang, max(1, discovered_total))
    drop_ratio_total = 1.0 - _safe_div(filtered_total, max(1, discovered_total))
    md_den = saved_md_total + md_suppressed_total
    md_suppression_rate = _safe_div(md_suppressed_total, md_den)

    throughput_urls_per_min = _safe_div(urls_total, (duration_sec / 60.0)) if duration_sec > 0 else 0.0

    return CompanyRow(
        bvdid=bvdid,
        company_name=company_name,
        stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        urls_total=urls_total,
        urls_done=urls_done,
        urls_failed=urls_failed,
        seeding_source=seeding_source,
        discovered_total=discovered_total,
        live_check_enabled=live_check_enabled,
        live_checked_total=live_checked_total,
        prefilter_kept=prefilter_kept,
        prefilter_dropped_status=prefilter_dropped_status,
        prefilter_dropped_external=prefilter_dropped_external,
        prefilter_dropped_score=prefilter_dropped_score,
        filtered_total=filtered_total,
        filtered_dropped_patterns_lang=filtered_dropped_patterns_lang,
        seed_roots_count=len(seed_roots) if isinstance(seed_roots, list) else 0,
        seed_brand_count=seed_brand_count,
        company_cap=company_cap,
        company_cap_applied=company_cap_applied,
        per_root_max_discovered=per_root_max_discovered,
        per_root_mean_discovered=per_root_mean_discovered,
        saved_html_total=saved_html_total,
        saved_md_total=saved_md_total,
        saved_json_total=saved_json_total,
        md_suppressed_total=md_suppressed_total,
        resumed_skips=resumed_skips,
        md_words_count=md_words_count,
        md_words_total=md_words_total,
        md_words_mean=md_words_mean,
        md_words_median=md_words_median,
        drop_ratio_prefilter=drop_ratio_prefilter,
        drop_ratio_patterns_lang=drop_ratio_patterns_lang,
        drop_ratio_total=drop_ratio_total,
        md_suppression_rate=md_suppression_rate,
        throughput_urls_per_min=throughput_urls_per_min,
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
            # attach company_name for interactive traceability
            d["company_name"] = doc.get("company_name")
            rows.append(d)
        except Exception as e:
            # keep going, but record a stub for debugging
            rows.append({
                "bvdid": f"__PARSE_ERROR__:{meta_path.name}",
                "company_name": None,
                "stage": "unknown",
                "started_at": None,
                "finished_at": None,
                "duration_sec": 0,
                "urls_total": 0,
                "urls_done": 0,
                "urls_failed": 0,
                "seeding_source": None,
                "discovered_total": 0,
                "live_check_enabled": False,
                "live_checked_total": 0,
                "prefilter_kept": 0,
                "prefilter_dropped_status": 0,
                "prefilter_dropped_external": 0,
                "prefilter_dropped_score": 0,
                "filtered_total": 0,
                "filtered_dropped_patterns_lang": 0,
                "seed_roots_count": 0,
                "seed_brand_count": 0,
                "company_cap": None,
                "company_cap_applied": False,
                "per_root_max_discovered": 0,
                "per_root_mean_discovered": 0.0,
                "saved_html_total": 0,
                "saved_md_total": 0,
                "saved_json_total": 0,
                "md_suppressed_total": 0,
                "resumed_skips": 0,
                "drop_ratio_prefilter": 0.0,
                "drop_ratio_patterns_lang": 0.0,
                "drop_ratio_total": 0.0,
                "md_suppression_rate": 0.0,
                "throughput_urls_per_min": 0.0,
                "md_words_count": 0,
                "md_words_total": 0,
                "md_words_mean": 0.0,
                "md_words_median": 0.0,
                "_meta_path": str(meta_path),
                "_url_index_path": str(url_index_path) if url_index_path else None,
                "_error": str(e),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Ensure numeric types where possible
    numeric_cols = [
        "duration_sec","urls_total","urls_done","urls_failed",
        "discovered_total","live_checked_total","prefilter_kept",
        "prefilter_dropped_status","prefilter_dropped_external","prefilter_dropped_score",
        "filtered_total","filtered_dropped_patterns_lang","seed_roots_count","seed_brand_count",
        "per_root_max_discovered","per_root_mean_discovered","saved_html_total","saved_md_total",
        "saved_json_total","md_suppressed_total","resumed_skips",
        "drop_ratio_prefilter","drop_ratio_patterns_lang","drop_ratio_total",
        "md_suppression_rate","throughput_urls_per_min",
        # markdown words fields
        "md_words_count","md_words_total","md_words_mean","md_words_median",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


# ---------------------------
# Global metrics & suggestions
# ---------------------------

def _percentiles(series: pd.Series, cuts=(50, 75, 80, 90, 95, 97, 99)) -> Dict[str, float]:
    out = {}
    clean = series.dropna()
    for p in cuts:
        out[f"p{p}"] = float(np.percentile(clean, p)) if len(clean) else 0.0
    return out


def compute_summary(df: pd.DataFrame, cap_percentile: float = 0.90, cost_per_1k_tokens_usd: float = 0.03) -> Dict[str, Any]:
    """
    Create a JSON-serializable summary of the dataset, with a recommended company cap and
    additional markdown-word / token cost estimates.
    """
    n = len(df)
    filtered = df["filtered_total"] if "filtered_total" in df else pd.Series(dtype=float)
    discovered = df["discovered_total"] if "discovered_total" in df else pd.Series(dtype=float)

    recommended_cap = int(np.percentile(filtered.dropna(), cap_percentile * 100)) if len(filtered.dropna()) else 0

    summary = {
        "companies_count": n,
        "has_live_check_any": bool(df["live_check_enabled"].any()) if "live_check_enabled" in df else False,
        "filtered_total_stats": {
            "count": int(filtered.count()),
            "mean": float(filtered.mean() if len(filtered) else 0.0),
            "median": float(filtered.median() if len(filtered) else 0.0),
            **_percentiles(filtered),
            "sum": float(filtered.sum() if len(filtered) else 0.0),
        },
        "discovered_total_stats": {
            "count": int(discovered.count()),
            "mean": float(discovered.mean() if len(discovered) else 0.0),
            "median": float(discovered.median() if len(discovered) else 0.0),
            **_percentiles(discovered),
            "sum": float(discovered.sum() if len(discovered) else 0.0),
        },
        "drop_ratios": {
            "prefilter_mean": float(df["drop_ratio_prefilter"].mean() if "drop_ratio_prefilter" in df else 0.0),
            "patterns_lang_mean": float(df["drop_ratio_patterns_lang"].mean() if "drop_ratio_patterns_lang" in df else 0.0),
            "total_mean": float(df["drop_ratio_total"].mean() if "drop_ratio_total" in df else 0.0),
        },
        "brands": {
            "mean_brand_count": float(df["seed_brand_count"].mean() if "seed_brand_count" in df else 0.0),
            "mean_roots_count": float(df["seed_roots_count"].mean() if "seed_roots_count" in df else 0.0),
        },
        "markdown": {
            "md_suppression_rate_mean": float(df["md_suppression_rate"].mean() if "md_suppression_rate" in df else 0.0),
            "saved_md_total_sum": int(df["saved_md_total"].sum() if "saved_md_total" in df else 0),
            "md_suppressed_total_sum": int(df["md_suppressed_total"].sum() if "md_suppressed_total" in df else 0),
        },
        "throughput": {
            "urls_per_min_mean": float(df["throughput_urls_per_min"].mean() if "throughput_urls_per_min" in df else 0.0),
            "urls_per_min_median": float(df["throughput_urls_per_min"].median() if "throughput_urls_per_min" in df else 0.0),
        },
        "recommendations": {
            "company_max_pages_cap_percentile": cap_percentile,
            "company_max_pages_recommended": recommended_cap,
        }
    }

    # -- Markdown-word global metrics & LLM cost estimates --
    try:
        md_words_total_series = pd.Series(dtype=float)
        if "md_words_total" in df.columns:
            md_words_total_series = df["md_words_total"].fillna(0).astype(float)

        md_count_companies_with_words = int((df["md_words_count"] > 0).sum()) if "md_words_count" in df.columns else 0
        md_files_estimated = int(df["md_words_count"].sum()) if "md_words_count" in df.columns else 0
        md_words_sum = float(md_words_total_series.sum()) if not md_words_total_series.empty else 0.0
        md_words_mean_per_company = float(md_words_total_series.mean()) if not md_words_total_series.empty else 0.0
        md_words_median_per_company = float(df["md_words_median"].median()) if "md_words_median" in df.columns else 0.0

        # token estimate and cost estimate
        # Assumption: tokens ≈ words * token_per_word
        token_per_word = 1.3333333  # ~ 1 token ≈ 0.75 words → inverse ~1.3333
        estimated_total_tokens = md_words_sum * token_per_word
        estimated_cost_usd_total = (estimated_total_tokens / 1000.0) * float(cost_per_1k_tokens_usd)
        estimated_cost_usd_mean_per_company = ((md_words_sum / max(1, len(df))) * token_per_word / 1000.0) * float(cost_per_1k_tokens_usd)

        summary["markdown"].update({
            "md_companies_with_word_counts": md_count_companies_with_words,
            "md_files_estimated_total": md_files_estimated,
            "md_words_total_sum": float(md_words_sum),
            "md_words_mean_per_company": float(md_words_mean_per_company),
            "md_words_median_per_company": float(md_words_median_per_company),
            "estimated_total_tokens": float(estimated_total_tokens),
            "estimated_cost_usd_total_at_specified_rate": float(estimated_cost_usd_total),
            "estimated_cost_usd_mean_per_company_at_specified_rate": float(estimated_cost_usd_mean_per_company),
            "token_per_word_assumption": float(token_per_word),
            "cost_per_1k_tokens_usd_used": float(cost_per_1k_tokens_usd),
        })
    except Exception:
        # don't fail if markdown aggregations blow up
        summary["markdown"].setdefault("md_words_total_sum", 0.0)
        summary["markdown"].setdefault("estimated_cost_usd_total_at_specified_rate", 0.0)

    return summary


# ---------------------------
# Plotting (matplotlib only)
# ---------------------------

def _savefig(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_hist_filtered(df: pd.DataFrame, out: Path, bins: int = 40):
    if "filtered_total" not in df:
        return
    series = df["filtered_total"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(series, bins=bins)
    ax.set_title("Distribution of filtered_total per company")
    ax.set_xlabel("filtered_total")
    ax.set_ylabel("companies")
    # percentiles to guide cap choice
    for p in (80, 90, 95, 97, 99):
        v = np.percentile(series, p) if len(series) else 0
        ax.axvline(v, linestyle="--")
        ax.text(v, ax.get_ylim()[1]*0.9, f"p{p}={int(v)}", rotation=90, va="top")
    _savefig(fig, out)


def plot_ecdf_filtered(df: pd.DataFrame, out: Path):
    if "filtered_total" not in df:
        return
    x = np.sort(df["filtered_total"].dropna().values)
    if len(x) == 0:
        return
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, y, drawstyle="steps-post")
    ax.set_title("ECDF of filtered_total")
    ax.set_xlabel("filtered_total")
    ax.set_ylabel("proportion of companies ≤ x")
    _savefig(fig, out)


def plot_funnel_means(df: pd.DataFrame, out: Path):
    cols = []
    vals = []
    for k in ("discovered_total", "prefilter_kept", "filtered_total"):
        if k in df:
            cols.append(k)
            vals.append(float(df[k].mean()))
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(cols, vals)
    ax.set_title("Mean funnel (discover → prefilter_keep → filtered)")
    ax.set_ylabel("mean pages")
    _savefig(fig, out)


def plot_scatter_discovered_vs_filtered(df: pd.DataFrame, out: Path):
    if not {"discovered_total", "filtered_total"} <= set(df.columns):
        return
    x = df["discovered_total"].values
    y = df["filtered_total"].values
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    sc = ax.scatter(x, y, s=12, alpha=0.7)
    lim = max(1, int(max(np.nanmax(x), np.nanmax(y))))
    ax.plot([0, lim], [0, lim])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Discovered vs Filtered")
    ax.set_xlabel("discovered_total")
    ax.set_ylabel("filtered_total")
    _savefig(fig, out)


def plot_hist_md_suppression(df: pd.DataFrame, out: Path, bins: int = 30):
    if "md_suppression_rate" not in df:
        return
    series = df["md_suppression_rate"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(series, bins=bins)
    ax.set_title("Markdown suppression rate distribution")
    ax.set_xlabel("suppression_rate = suppressed / (saved + suppressed)")
    ax.set_ylabel("companies")
    _savefig(fig, out)


def plot_throughput_vs_filtered(df: pd.DataFrame, out: Path):
    need = {"throughput_urls_per_min", "filtered_total"}
    if not need <= set(df.columns):
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(df["filtered_total"], df["throughput_urls_per_min"], s=12, alpha=0.7)
    ax.set_title("Throughput vs filtered_total")
    ax.set_xlabel("filtered_total")
    ax.set_ylabel("urls_per_min")
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
        ax.text(v, ax.get_ylim()[1]*0.9, f"p{p}={int(v)}", rotation=90, va="top")
    _savefig(fig, out)


def _write_html(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs='cdn' keeps files smaller but offline; switch to 'directory' for totally offline
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)

def iplot_hist_filtered(df: pd.DataFrame, out: Path):
    if "filtered_total" not in df or df["filtered_total"].dropna().empty or not _PLOTLY_OK:
        return
    s = df[["bvdid","company_name","filtered_total"]].dropna()
    fig = px.histogram(s, x="filtered_total", nbins=40, title="Distribution of filtered_total per company")
    for p in (80, 90, 95, 97, 99):
        v = float(np.percentile(s["filtered_total"], p))
        fig.add_vline(x=v, line_dash="dash", annotation_text=f"p{p}={int(v)}", annotation_position="top")
    _write_html(fig, out)

def iplot_ecdf_filtered(df: pd.DataFrame, out: Path):
    if "filtered_total" not in df or df["filtered_total"].dropna().empty or not _PLOTLY_OK:
        return
    fig = px.ecdf(df, x="filtered_total", title="ECDF of filtered_total")
    fig.update_traces(customdata=df[["bvdid","company_name"]].values, hovertemplate="filtered_total=%{x}<br>bvdid=%{customdata[0]}<br>company=%{customdata[1]}")
    _write_html(fig, out)

def iplot_funnel_means(df: pd.DataFrame, out: Path):
    if not _PLOTLY_OK: return
    cols = [c for c in ("discovered_total","prefilter_kept","filtered_total") if c in df]
    if not cols: return
    means = [float(df[c].mean()) for c in cols]
    fig = px.bar(x=cols, y=means, title="Mean funnel (discover → prefilter_keep → filtered)", labels={"x":"stage","y":"mean pages"})
    _write_html(fig, out)

def iplot_scatter_discovered_vs_filtered(df: pd.DataFrame, out: Path):
    if not _PLOTLY_OK or not {"discovered_total","filtered_total"} <= set(df.columns): return
    # Ensure interactive points are traceable: include bvdid and company_name in hover
    fig = px.scatter(
        df,
        x="discovered_total",
        y="filtered_total",
        hover_data=["bvdid", "company_name", "filtered_total", "discovered_total"],
        title="Discovered vs Filtered",
        opacity=0.8
    )
    # Tweak hover template for clarity
    fig.update_traces(marker=dict(size=8), hovertemplate="discovered=%{x}<br>filtered=%{y}<br>bvdid=%{customdata[0]}<br>company=%{customdata[1]}")
    # attach customdata explicitly so hovertemplate fields map:
    if "bvdid" in df.columns and "company_name" in df.columns:
        fig.data[0].customdata = df[["bvdid","company_name"]].values
    _write_html(fig, out)

def iplot_hist_md_suppression(df: pd.DataFrame, out: Path):
    if not _PLOTLY_OK or "md_suppression_rate" not in df or df["md_suppression_rate"].dropna().empty: return
    fig = px.histogram(df, x="md_suppression_rate", nbins=30, title="Markdown suppression rate distribution")
    _write_html(fig, out)

def iplot_throughput_vs_filtered(df: pd.DataFrame, out: Path):
    if not _PLOTLY_OK or not {"throughput_urls_per_min","filtered_total"} <= set(df.columns): return
    fig = px.scatter(
        df,
        x="filtered_total",
        y="throughput_urls_per_min",
        hover_data=["bvdid","company_name","filtered_total","throughput_urls_per_min"],
        title="Throughput vs filtered_total",
        opacity=0.8
    )
    if "bvdid" in df.columns and "company_name" in df.columns:
        fig.data[0].customdata = df[["bvdid","company_name"]].values
        fig.update_traces(hovertemplate="filtered=%{x}<br>urls_per_min=%{y}<br>bvdid=%{customdata[0]}<br>company=%{customdata[1]}")
    _write_html(fig, out)

# ---------------------------
# Orchestrator
# ---------------------------

def analyze_and_plot(
    outputs_dir: Path,
    out_dir: Path,
    cap_percentile: float = 0.90,
    save_csv: bool = True,
    save_json_summary: bool = True,
    interactive: bool = False,
    cost_per_1k_tokens_usd: float = 0.03,
) -> Dict[str, Any]:
    """
    Scan outputs_dir for all crawl_meta.json files, aggregate, plot, and write artifacts.

    Returns a dict summary (also optionally saved to JSON).
    """
    crawl_meta_files = find_crawl_meta_files(outputs_dir)
    df = collect_dataframe(crawl_meta_files)

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        empty_summary = {"companies_count": 0, "message": "No crawl_meta.json found"}
        if save_json_summary:
            (out_dir / "summary.json").write_text(json.dumps(empty_summary, indent=2), encoding="utf-8")
        return empty_summary

    # Save detailed CSV
    if save_csv:
        df.to_csv(out_dir / "crawl_meta_aggregate.csv", index=False, encoding="utf-8")

    # Compute summary + recommendations
    summary = compute_summary(df, cap_percentile=cap_percentile, cost_per_1k_tokens_usd=cost_per_1k_tokens_usd)
    if save_json_summary:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Charts
    plot_hist_filtered(df, out_dir / "hist_filtered_total.png")
    plot_ecdf_filtered(df, out_dir / "ecdf_filtered_total.png")
    plot_funnel_means(df, out_dir / "funnel_means.png")
    plot_scatter_discovered_vs_filtered(df, out_dir / "scatter_discovered_vs_filtered.png")
    plot_hist_md_suppression(df, out_dir / "hist_md_suppression.png")
    plot_throughput_vs_filtered(df, out_dir / "scatter_throughput_vs_filtered.png")
    plot_hist_md_words_total(df, out_dir / "hist_md_words_total.png")

    # Interactive charts (Plotly HTML)
    if interactive and _PLOTLY_OK:
        iout = out_dir / "interactive"
        iplot_hist_filtered(df, iout / "hist_filtered_total.html")
        iplot_ecdf_filtered(df, iout / "ecdf_filtered_total.html")
        iplot_funnel_means(df, iout / "funnel_means.html")
        iplot_scatter_discovered_vs_filtered(df, iout / "scatter_discovered_vs_filtered.html")
        iplot_hist_md_suppression(df, iout / "hist_md_suppression.html")
        iplot_throughput_vs_filtered(df, iout / "scatter_throughput_vs_filtered.html")
        # annotate summary with where the files are
        summary["interactive_dir"] = str(iout.resolve())
    elif interactive and not _PLOTLY_OK:
        summary["interactive_dir"] = None
        summary["interactive_error"] = "plotly not installed"

    # add artifact locations
    summary["artifacts_dir"] = str(out_dir.resolve())
    summary["crawl_meta_aggregate"] = str((out_dir / "crawl_meta_aggregate.csv").resolve()) if save_csv else None
    summary["summary_json"] = str((out_dir / "summary.json").resolve()) if save_json_summary else None

    return summary