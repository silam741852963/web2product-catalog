from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


def build_html_report(
    *,
    report: Dict[str, Any],
    chart_specs: Dict[str, dict],
    out_path: Path,
) -> None:
    """
    Single-file HTML report.
    - JSON is removed entirely.
    - Per view: render ALL charts stacked vertically, full width.
    - On <details> toggle open: resize Plotly charts.
    """
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    title = f"Analysis Report — {meta.get('out_dir', '')}"
    generated_at = meta.get("generated_at", "")
    cgs_updated = meta.get("crawl_global_state_updated_at", "")

    views = report.get("views", {}) if isinstance(report, dict) else {}
    view_items = list(views.items())

    css = """
    :root { --bg:#0b0e14; --fg:#e6e6e6; --muted:#aab; --card:#121826; --border:#253045; --accent:#6ea8fe; --chip:#0f1522; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:var(--bg); color:var(--fg); }
    header { padding:18px 22px; border-bottom:1px solid var(--border); background:linear-gradient(180deg, #0b0e14, #0b0e14 60%, #0a0c12); position:sticky; top:0; z-index:10; }
    h1 { margin:0; font-size:18px; }
    .meta { margin-top:6px; font-size:12px; color:var(--muted); display:flex; gap:16px; flex-wrap:wrap; }
    .container { padding:18px 22px; }
    details { border:1px solid var(--border); background:var(--card); border-radius:12px; margin-bottom:14px; overflow:hidden; }
    summary { cursor:pointer; padding:12px 14px; font-weight:600; }
    .section { padding:12px 14px 16px; border-top:1px solid var(--border); }

    .note { font-size:12px; color:var(--muted); }
    .stack { display:flex; flex-direction:column; gap:12px; }

    .card { border:1px solid var(--border); border-radius:12px; padding:12px; background:rgba(255,255,255,0.02); }
    .card h3 { margin:0 0 8px; font-size:13px; color:#dfe7ff; }
    .chart { width:100%; height:480px; }
    """

    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        k, _ = item
        return (0, "") if k == "__global__" else (1, k)

    def _esc(s: Any) -> str:
        t = str(s)
        return (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _safe_dom_id(prefix: str, raw: str) -> str:
        base = f"{prefix}_{raw}".strip()
        base = re.sub(r"[^A-Za-z0-9_:\-\.]", "_", base)
        if base and base[0].isdigit():
            base = f"v_{base}"
        return base

    def render_view(view_key: str, view_obj: dict) -> str:
        charts = view_obj.get("charts", []) if isinstance(view_obj, dict) else []
        view_dom_id = _safe_dom_id("view", str(view_key))

        out: list[str] = []
        out.append("<div class='stack'>")

        if not charts:
            out.append("<div class='note'>No charts for this view.</div>")
            out.append("</div>")
            return "".join(out)

        for ch in charts:
            if not isinstance(ch, dict):
                continue
            cid = ch.get("chart_id")
            if not isinstance(cid, str) or not cid:
                continue

            plot_id = _safe_dom_id(view_dom_id + "__plot", cid)

            out.append("<div class='card'>")
            out.append(f"<h3>{_esc(ch.get('title', cid))}</h3>")
            out.append(f"<div id='{_esc(plot_id)}' class='chart'></div>")

            spec = chart_specs.get(cid)
            if spec is None:
                out.append(
                    "<div class='note'>Missing plotly spec for this chart id.</div>"
                )
                out.append("</div>")
                continue

            spec_json = json.dumps(spec, ensure_ascii=False)
            out.append(
                "<script>"
                f"(function(){{"
                f"var fig={spec_json};"
                f"var el=document.getElementById('{plot_id}');"
                f"if(el && window.Plotly){{"
                f"Plotly.react(el, fig.data||[], fig.layout||{{}}, fig.config||{{responsive:true}});"
                f"}}"
                f"}})();"
                "</script>"
            )

            out.append("</div>")  # card

        out.append("</div>")  # stack
        return "".join(out)

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'/>")
    parts.append(f"<title>{_esc(title)}</title>")
    parts.append(
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
    )
    parts.append(f"<style>{css}</style>")
    parts.append("<script src='https://cdn.plot.ly/plotly-2.30.0.min.js'></script>")

    # Resize on details open
    parts.append(
        "<script>"
        "(function(){"
        "  document.addEventListener('toggle', function(ev){"
        "    var d=ev.target;"
        "    if(!d || d.tagName!=='DETAILS') return;"
        "    if(!d.open) return;"
        "    if(!window.Plotly) return;"
        "    setTimeout(function(){"
        "      var charts=d.querySelectorAll('.chart');"
        "      for(var i=0;i<charts.length;i++){"
        "        try{ Plotly.Plots.resize(charts[i]); }catch(e){}"
        "      }"
        "    }, 0);"
        "  }, true);"
        "})();"
        "</script>"
    )

    parts.append("</head><body>")

    parts.append("<header>")
    parts.append(f"<h1>{_esc(title)}</h1>")
    parts.append("<div class='meta'>")
    parts.append(f"<div>generated_at: {_esc(generated_at)}</div>")
    parts.append(f"<div>crawl_global_state_updated_at: {_esc(cgs_updated)}</div>")
    parts.append(f"<div>views: {len(view_items)}</div>")
    parts.append("</div>")
    parts.append("</header>")

    parts.append("<div class='container'>")

    for key, v in sorted(view_items, key=sort_key):
        if not isinstance(v, dict):
            continue
        label = v.get("label") or key
        count = v.get("company_count") or v.get("companyCount") or ""
        open_attr = " open" if key == "__global__" else ""
        parts.append(f"<details{open_attr}>")
        parts.append(
            f"<summary>{_esc(label)} <span class='note'>(companies: {_esc(count)})</span></summary>"
        )
        parts.append("<div class='section'>")
        parts.append(render_view(str(key), v))
        parts.append("</div></details>")

    parts.append("</div></body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
