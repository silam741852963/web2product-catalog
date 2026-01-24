from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def make_scatter(
    *,
    chart_id: str,
    title: str,
    x: List[float | int],
    y: List[float | int],
    x_title: str = "",
    y_title: str = "",
    text: Optional[List[str]] = None,
    x_log: bool = False,
    y_log: bool = False,
) -> Dict[str, Any]:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if text is not None and len(text) != len(x):
        raise ValueError("text must match x/y length when provided")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                text=text,
                hovertemplate=("%{text}<br>" if text else "")
                + f"{x_title}=%{{x}}<br>{y_title}=%{{y}}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")

    png_bytes = _require_png_bytes(fig)
    return {
        "chart_id": chart_id,
        "title": title,
        "plotly_spec": fig.to_dict(),
        "png_bytes": png_bytes,
    }
