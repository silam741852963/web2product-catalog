from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Sequence


def _pct(values_sorted: Sequence[float], p: float) -> float:
    """
    Deterministic percentile with linear interpolation (like numpy default).
    """
    n = len(values_sorted)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values_sorted[0])

    if p <= 0:
        return float(values_sorted[0])
    if p >= 100:
        return float(values_sorted[-1])

    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    d0 = float(values_sorted[f]) * (c - k)
    d1 = float(values_sorted[c]) * (k - f)
    return d0 + d1


def summarize_distribution(
    values: List[float | int], *, include_minmax: bool = True
) -> Dict[str, float]:
    """
    Summary stats with mean/median adjacent.
    include_minmax lets callers drop min/max when they want “pct view” summaries only.
    """
    if not values:
        out: Dict[str, float] = {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p80": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p97": 0.0,
            "p99": 0.0,
        }
        if include_minmax:
            out["min"] = 0.0
            out["max"] = 0.0
        return out

    xs = [float(x) for x in values]
    xs_sorted = sorted(xs)

    mean = statistics.fmean(xs_sorted)
    median = statistics.median(xs_sorted)

    out2: Dict[str, float] = {
        "count": float(len(xs_sorted)),
        "mean": float(mean),
        "median": float(median),
        "p75": float(_pct(xs_sorted, 75.0)),
        "p80": float(_pct(xs_sorted, 80.0)),
        "p90": float(_pct(xs_sorted, 90.0)),
        "p95": float(_pct(xs_sorted, 95.0)),
        "p97": float(_pct(xs_sorted, 97.0)),
        "p99": float(_pct(xs_sorted, 99.0)),
    }
    if include_minmax:
        out2["min"] = float(xs_sorted[0])
        out2["max"] = float(xs_sorted[-1])
    return out2


def make_histogram_bins_edges(values: List[int], *, bins: int = 30) -> Dict[str, Any]:
    """
    Back-compat helper: returns edges+counts.
    """
    if bins <= 0:
        raise ValueError("bins must be > 0")

    if not values:
        return {"bins": int(bins), "edges": [0, 1], "counts": [0]}

    xs = [int(v) for v in values]
    lo = min(xs)
    hi = max(xs)

    if lo == hi:
        edges = [lo, lo + 1]
        counts = [len(xs)]
        return {"bins": 1, "edges": edges, "counts": counts}

    span = hi - lo
    width = max(1, math.ceil(span / bins))

    edges: List[int] = [lo]
    while edges[-1] <= hi:
        edges.append(edges[-1] + width)

    counts = [0 for _ in range(len(edges) - 1)]
    for v in xs:
        idx = (v - lo) // width
        if idx < 0:
            idx = 0
        if idx >= len(counts):
            idx = len(counts) - 1
        counts[int(idx)] += 1

    return {"bins": int(len(counts)), "edges": edges, "counts": counts}


def make_histogram_bins(values: List[int], *, bins: int = 30) -> Dict[str, Any]:
    """
    New contract helper (what urlindex_metrics wants):
      {
        "histogram_bins": [{"lo": int, "hi": int, "count": int}, ...],
        "bin_labels": ["lo–hi", ..., "≥N" optional],
        "bin_counts": [count, ...]
      }
    """
    raw = make_histogram_bins_edges(values, bins=int(bins))
    edges = list(raw.get("edges") or [])
    counts = list(raw.get("counts") or [])

    histogram_bins: List[Dict[str, int]] = []
    bin_labels: List[str] = []
    bin_counts: List[int] = []

    if len(edges) < 2 or len(counts) != (len(edges) - 1):
        return {"histogram_bins": [], "bin_labels": [], "bin_counts": []}

    for i in range(len(counts)):
        lo = int(edges[i])
        hi = int(edges[i + 1] - 1)  # inclusive label (matches “lo–hi”)
        c = int(counts[i])
        histogram_bins.append({"lo": lo, "hi": hi, "count": c})
        bin_labels.append(f"{lo}–{hi}")
        bin_counts.append(c)

    return {
        "histogram_bins": histogram_bins,
        "bin_labels": bin_labels,
        "bin_counts": bin_counts,
    }
