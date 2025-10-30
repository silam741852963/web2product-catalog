from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def find_progress_files(outputs_root: Path) -> List[Path]:
    """
    Recursively find all progress.json files under outputs_root/**/checkpoints/.
    """
    return sorted(outputs_root.rglob("checkpoints/progress.json"))


def load_progress(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Normalization
# ---------------------------

@dataclass
class CompanyRow:
    hojin_id: str
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


def normalize_progress(doc: Dict[str, Any]) -> CompanyRow:
    hojin_id = str(doc.get("hojin_id") or "")
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
    live_check_enabled = bool(seeding.get("live_check_enabled") or False)
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
    resumed_skips = int(saves.get("resumed_skips") or doc.get("resumed_skips") or 0)

    # Derived
    drop_ratio_prefilter = 1.0 - _safe_div(prefilter_kept, max(1, discovered_total))
    drop_ratio_patterns_lang = _safe_div(filtered_dropped_patterns_lang, max(1, discovered_total))
    drop_ratio_total = 1.0 - _safe_div(filtered_total, max(1, discovered_total))
    md_den = saved_md_total + md_suppressed_total
    md_suppression_rate = _safe_div(md_suppressed_total, md_den)

    throughput_urls_per_min = _safe_div(urls_total, (duration_sec / 60.0)) if duration_sec > 0 else 0.0

    return CompanyRow(
        hojin_id=hojin_id,
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
        drop_ratio_prefilter=drop_ratio_prefilter,
        drop_ratio_patterns_lang=drop_ratio_patterns_lang,
        drop_ratio_total=drop_ratio_total,
        md_suppression_rate=md_suppression_rate,
        throughput_urls_per_min=throughput_urls_per_min,
    )


def collect_dataframe(progress_files: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fp in progress_files:
        try:
            doc = load_progress(fp)
            row = normalize_progress(doc)
            d = row.as_dict()
            d["_progress_path"] = str(fp)
            rows.append(d)
        except Exception as e:
            # keep going, but record a stub for debugging
            rows.append({
                "hojin_id": f"__PARSE_ERROR__:{fp.name}",
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
                "_progress_path": str(fp),
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
        "md_suppression_rate","throughput_urls_per_min"
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


def compute_summary(df: pd.DataFrame, cap_percentile: float = 0.90) -> Dict[str, Any]:
    """
    Create a JSON-serializable summary of the dataset, with a recommended company cap.
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
    ax.scatter(x, y, s=12, alpha=0.7)
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

def _write_html(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs='cdn' keeps files smaller but offline; switch to 'directory' for totally offline
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)

def iplot_hist_filtered(df: pd.DataFrame, out: Path):
    if "filtered_total" not in df or df["filtered_total"].dropna().empty or not _PLOTLY_OK:
        return
    s = df["filtered_total"].dropna()
    fig = px.histogram(s, x="filtered_total", nbins=40, title="Distribution of filtered_total per company")
    for p in (80, 90, 95, 97, 99):
        v = float(np.percentile(s, p))
        fig.add_vline(x=v, line_dash="dash", annotation_text=f"p{p}={int(v)}", annotation_position="top")
    _write_html(fig, out)

def iplot_ecdf_filtered(df: pd.DataFrame, out: Path):
    if "filtered_total" not in df or df["filtered_total"].dropna().empty or not _PLOTLY_OK:
        return
    fig = px.ecdf(df, x="filtered_total", title="ECDF of filtered_total")
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
    fig = px.scatter(df, x="discovered_total", y="filtered_total", title="Discovered vs Filtered", opacity=0.7)
    _write_html(fig, out)

def iplot_hist_md_suppression(df: pd.DataFrame, out: Path):
    if not _PLOTLY_OK or "md_suppression_rate" not in df or df["md_suppression_rate"].dropna().empty: return
    fig = px.histogram(df, x="md_suppression_rate", nbins=30, title="Markdown suppression rate distribution")
    _write_html(fig, out)

def iplot_throughput_vs_filtered(df: pd.DataFrame, out: Path):
    if not _PLOTLY_OK or not {"throughput_urls_per_min","filtered_total"} <= set(df.columns): return
    fig = px.scatter(df, x="filtered_total", y="throughput_urls_per_min", title="Throughput vs filtered_total", opacity=0.7)
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
) -> Dict[str, Any]:
    """
    Scan outputs_dir for all progress.json, aggregate, plot, and write artifacts.

    Returns a dict summary (also optionally saved to JSON).
    """
    progress_files = find_progress_files(outputs_dir)
    df = collect_dataframe(progress_files)

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        empty_summary = {"companies_count": 0, "message": "No progress.json found"}
        if save_json_summary:
            (out_dir / "summary.json").write_text(json.dumps(empty_summary, indent=2), encoding="utf-8")
        return empty_summary

    # Save detailed CSV
    if save_csv:
        df.to_csv(out_dir / "progress_aggregate.csv", index=False, encoding="utf-8")

    # Compute summary + recommendations
    summary = compute_summary(df, cap_percentile=cap_percentile)
    if save_json_summary:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Charts
    plot_hist_filtered(df, out_dir / "hist_filtered_total.png")
    plot_ecdf_filtered(df, out_dir / "ecdf_filtered_total.png")
    plot_funnel_means(df, out_dir / "funnel_means.png")
    plot_scatter_discovered_vs_filtered(df, out_dir / "scatter_discovered_vs_filtered.png")
    plot_hist_md_suppression(df, out_dir / "hist_md_suppression.png")
    plot_throughput_vs_filtered(df, out_dir / "scatter_throughput_vs_filtered.png")

    # Interactive charts (Plotly HTML)
    if interactive and _PLOTLY_OK:
        iout = out_dir / "interactive"
        iplot_hist_filtered(df, iout / "hist_filtered_total.html")
        iplot_ecdf_filtered(df, iout / "ecdf_filtered_total.html")
        iplot_funnel_means(df, iout / "funnel_means.html")
        iplot_scatter_discovered_vs_filtered(df, iout / "scatter_discovered_vs_filtered.html")
        iplot_hist_md_suppression(df, iout / "hist_md_suppression.html")
        iplot_throughput_vs_filtered(df, iout / "scatter_throughput_vs_filtered.html")
        # optionally annotate summary with where the files are
        summary["interactive_dir"] = str(iout.resolve())
    elif interactive and not _PLOTLY_OK:
        summary["interactive_dir"] = None
        summary["interactive_error"] = "plotly not installed"

    return summary