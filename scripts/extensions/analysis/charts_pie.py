from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
import plotly.io as pio


def _require_png_bytes(fig: go.Figure) -> bytes:
    try:
        # This requires kaleido at runtime.
        return pio.to_image(fig, format="png")
    except Exception as e:
        raise RuntimeError(
            "PNG export failed. Install kaleido (required): `pip install -U kaleido` "
            "and ensure plotly can access it."
        ) from e


def make_pie(
    *, chart_id: str, title: str, labels: List[str], values: List[int]
) -> Dict[str, Any]:
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")

    total = sum(int(v) for v in values) or 1
    legend_labels = []
    for l, v in zip(labels, values):
        v_int = int(v)
        pct = round((v_int / total) * 100.0, 2)
        legend_labels.append(f"{l} — {v_int} ({pct}%)")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=legend_labels,  # legend shows these
                values=[int(v) for v in values],
                textinfo="percent",
                hovertemplate="%{label}<br>count=%{value}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=title, legend_title_text="")

    png_bytes = _require_png_bytes(fig)

    # Plotly JSON spec (dict)
    spec = fig.to_dict()

    return {
        "chart_id": chart_id,
        "title": title,
        "plotly_spec": spec,
        "png_bytes": png_bytes,
    }
