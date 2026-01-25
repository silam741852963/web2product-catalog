from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_UNCLASSIFIED_GROUP_KEY = "industry:unknown"


def _parse_industry_key(view_key: str) -> Optional[int]:
    k = (view_key or "").strip()
    if not k.startswith("industry:"):
        return None
    tail = k.split("industry:", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def _sort_view_key(view_key: str) -> Tuple[int, int, str]:
    """
    Ordering rules:
      1) __global__ first
      2) industry:1..N ascending
      3) other non-industry keys next (lexicographic)
      4) industry:unknown last
    """
    k = (view_key or "").strip()

    if k == "__global__":
        return (0, 0, "")

    if k == _UNCLASSIFIED_GROUP_KEY:
        return (9, 0, k)

    n = _parse_industry_key(k)
    if n is not None:
        return (1, n, "")

    return (5, 0, k)


def build_html_report(
    *,
    report: Dict[str, Any],
    chart_specs: Dict[str, dict],
    out_path: Path,
) -> None:
    """
    Single-file HTML report.
    - Per view: render ALL charts stacked vertically, full width.
    - On <details> toggle open: resize Plotly charts.
    - Views are ordered numerically: Industry 1..N, Unclassified last.
    - Renders global totals (meta.totals) at the top if present.
    """
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    title = f"Analysis Report — {meta.get('out_dir', '')}"
    generated_at = meta.get("generated_at", "")
    cgs_updated = meta.get("crawl_global_state_updated_at", "")
    llm_enabled = bool(meta.get("llm_metrics_enabled", True))
    totals = meta.get("totals") if isinstance(meta, dict) else None
    totals = totals if isinstance(totals, dict) else {}

    views = report.get("views", {}) if isinstance(report, dict) else {}
    view_items = list(views.items())
    view_items.sort(key=lambda kv: _sort_view_key(str(kv[0])))

    css = """
    :root { --bg:#0b0e14; --fg:#e6e6e6; --muted:#aab; --card:#121826; --border:#253045; --accent:#6ea8fe; --chip:#0f1522; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:var(--bg); color:var(--fg); }
    header { padding:18px 22px; border-bottom:1px solid var(--border); background:linear-gradient(180deg, #0b0e14, #0b0e14 60%, #0a0c12); position:sticky; top:0; z-index:10; }
    h1 { margin:0; font-size:18px; }
    .meta { margin-top:6px; font-size:12px; color:var(--muted); display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
    .totals { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; }
    .chip { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; border:1px solid var(--border); background:var(--chip); font-size:12px; color:var(--fg); }
    .chip b { color:#dfe7ff; font-weight:700; }
    .chip .k { color:var(--muted); font-weight:600; }
    .container { padding:18px 22px; }
    details { border:1px solid var(--border); background:var(--card); border-radius:12px; margin-bottom:14px; overflow:hidden; }
    summary { cursor:pointer; padding:12px 14px; font-weight:600; }
    .section { padding:12px 14px 16px; border-top:1px solid var(--border); }

    .note { font-size:12px; color:var(--muted); }
    .banner { margin:14px 0 18px; padding:10px 12px; border:1px dashed var(--border); border-radius:12px; background:rgba(255,255,255,0.02); }
    .stack { display:flex; flex-direction:column; gap:12px; }

    .card { border:1px solid var(--border); border-radius:12px; padding:12px; background:rgba(255,255,255,0.02); }
    .card h3 { margin:0 0 8px; font-size:13px; color:#dfe7ff; }
    .chart { width:100%; height:480px; }
    """

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

    def _fmt_int(x: Any) -> str:
        try:
            if isinstance(x, bool):
                return str(int(x))
            if isinstance(x, int):
                return f"{x:,}"
            if isinstance(x, float):
                return f"{int(x):,}"
            if isinstance(x, str) and x.strip().lstrip("-").isdigit():
                return f"{int(x.strip()):,}"
        except Exception:
            pass
        return str(x)

    def _get_tot(k: str, default: Any = 0) -> Any:
        v = totals.get(k, default) if isinstance(totals, dict) else default
        return v

    def render_totals_row() -> str:
        if not isinstance(totals, dict) or not totals:
            return ""

        companies_total = _get_tot("companies_total", None)
        urls_total = _get_tot("urls_total", None)
        md_saved_total = _get_tot("markdown_saved_total", None)
        md_tokens_total = _get_tot("md_tokens_total", None)
        llm_out_total = _get_tot("llm_output_tokens_total", None)
        offerings_total = _get_tot("offerings_total", None)
        totals_llm_enabled = bool(_get_tot("llm_metrics_enabled", llm_enabled))

        chips: list[str] = []
        if companies_total is not None:
            chips.append(
                f"<span class='chip'><span class='k'>companies</span> <b>{_esc(_fmt_int(companies_total))}</b></span>"
            )
        if urls_total is not None:
            chips.append(
                f"<span class='chip'><span class='k'>urls</span> <b>{_esc(_fmt_int(urls_total))}</b></span>"
            )
        if md_saved_total is not None:
            chips.append(
                f"<span class='chip'><span class='k'>markdown_saved</span> <b>{_esc(_fmt_int(md_saved_total))}</b></span>"
            )
        if md_tokens_total is not None:
            chips.append(
                f"<span class='chip'><span class='k'>md_tokens</span> <b>{_esc(_fmt_int(md_tokens_total))}</b></span>"
            )

        if totals_llm_enabled:
            if llm_out_total is not None:
                chips.append(
                    f"<span class='chip'><span class='k'>llm_output_tokens</span> <b>{_esc(_fmt_int(llm_out_total))}</b></span>"
                )
            if offerings_total is not None:
                chips.append(
                    f"<span class='chip'><span class='k'>offerings</span> <b>{_esc(_fmt_int(offerings_total))}</b></span>"
                )
        else:
            chips.append(
                "<span class='chip'><span class='k'>llm_metrics</span> <b>disabled</b></span>"
            )

        if not chips:
            return ""
        return "<div class='totals'>" + "".join(chips) + "</div>"

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
    parts.append(
        f"<div>llm_metrics: {_esc('enabled' if llm_enabled else 'disabled')}</div>"
    )
    parts.append("</div>")

    # NEW: totals row (right at top, after meta line)
    tr = render_totals_row()
    if tr:
        parts.append(tr)

    parts.append("</header>")

    parts.append("<div class='container'>")
    if not llm_enabled:
        parts.append(
            "<div class='banner'>"
            "<div><strong>LLM metrics disabled</strong></div>"
            "<div class='note'>Offerings section and LLM token charts/scatters were omitted by request.</div>"
            "</div>"
        )

    for key, v in view_items:
        if not isinstance(v, dict):
            continue
        label = v.get("label") or key
        count = v.get("company_count") or v.get("companyCount") or ""
        open_attr = " open" if str(key) == "__global__" else ""
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
