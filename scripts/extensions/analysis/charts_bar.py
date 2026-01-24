from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import plotly.graph_objects as go
import plotly.io as pio


def _require_png_bytes(fig: go.Figure) -> bytes:
    try:
        return pio.to_image(fig, format="png")
    except Exception as e:
        raise RuntimeError(
            "PNG export failed. Install kaleido (required): `pip install -U kaleido` "
            "and ensure plotly can access it."
        ) from e


def _apply_y_log(fig: go.Figure, y_log: bool) -> None:
    if not y_log:
        return
    # For log axes, remove/avoid 0 values. Best practice: caller should omit 0-count bins.
    fig.update_yaxes(type="log")


def make_bar(
    *,
    chart_id: str,
    title: str,
    x: List[Any],
    y: List[float | int],
    x_title: str = "",
    y_title: str = "",
    y_log: bool = False,
) -> Dict[str, Any]:
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    _apply_y_log(fig, y_log=y_log)
    png_bytes = _require_png_bytes(fig)
    return {
        "chart_id": chart_id,
        "title": title,
        "plotly_spec": fig.to_dict(),
        "png_bytes": png_bytes,
    }


def make_histogram(
    *,
    chart_id: str,
    title: str,
    values: List[int],
    x_title: str = "",
    y_title: str = "Count",
    bins: int = 30,
    y_log: bool = False,
) -> Dict[str, Any]:
    fig = go.Figure(data=[go.Histogram(x=[int(v) for v in values], nbinsx=int(bins))])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    _apply_y_log(fig, y_log=y_log)
    png_bytes = _require_png_bytes(fig)
    return {
        "chart_id": chart_id,
        "title": title,
        "plotly_spec": fig.to_dict(),
        "png_bytes": png_bytes,
    }


def make_histogram_binned(
    *,
    chart_id: str,
    title: str,
    x_labels: List[str],
    counts: List[int],
    x_title: str = "",
    y_title: str = "Count",
    y_log: bool = True,
) -> Dict[str, Any]:
    """
    Pre-binned histogram: x is bucket labels, y is count.

    NOTE: For y_log=True, you should avoid zeros in `counts` (drop empty bins).
    """
    if len(x_labels) != len(counts):
        raise ValueError("x_labels and counts must have the same length")

    fig = go.Figure(
        data=[go.Bar(x=[str(x) for x in x_labels], y=[int(c) for c in counts])]
    )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    _apply_y_log(fig, y_log=y_log)
    png_bytes = _require_png_bytes(fig)
    return {
        "chart_id": chart_id,
        "title": title,
        "plotly_spec": fig.to_dict(),
        "png_bytes": png_bytes,
    }


def make_barh_stacked(
    *,
    chart_id: str,
    title: str,
    categories: List[str],
    series: Mapping[str, List[int]],
    x_title: str = "",
    y_title: str = "",
    x_log: bool = False,
) -> Dict[str, Any]:
    traces: List[go.Bar] = []
    for name, vals in series.items():
        if len(vals) != len(categories):
            raise ValueError(f"Series '{name}' length must match categories length")
        traces.append(
            go.Bar(
                name=str(name), y=categories, x=[int(v) for v in vals], orientation="h"
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
    )
    if x_log:
        fig.update_xaxes(type="log")

    png_bytes = _require_png_bytes(fig)
    return {
        "chart_id": chart_id,
        "title": title,
        "plotly_spec": fig.to_dict(),
        "png_bytes": png_bytes,
    }
