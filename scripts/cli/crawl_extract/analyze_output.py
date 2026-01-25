from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from extensions.io.output_paths import ensure_output_root

# Public API entrypoint
from extensions.analysis import run_analysis

LOG = logging.getLogger("analyze_output")


def _utc_run_tag() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"analysis-{ts}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_output.py",
        description="Analyze crawl output folder and write deterministic analysis artifacts under <out_dir>/analysis/<run_tag>/",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Root output directory containing crawl_global_state.json and company folders.",
    )

    # Views / industry selection
    p.add_argument(
        "--view",
        type=str,
        default="global",
        choices=["global", "lv1", "lv1-all", "lv2-all"],
        help=(
            "Which view(s) to build. "
            "global=only global view (default). "
            "lv1=only a specific lv1 industry (requires --industry). "
            "lv1-all=all lv1 views. "
            "lv2-all=all lv2 (industry+nace) views."
        ),
    )
    p.add_argument(
        "--industry",
        type=int,
        default=None,
        help="For --view lv1: the industry id (lv1 key). Example: --industry 9",
    )

    p.add_argument(
        "--k-top-domains",
        type=int,
        default=20,
        help="Top-K offsite domains to include (default: 20).",
    )
    p.add_argument(
        "--k-top-errors",
        type=int,
        default=20,
        help="Top-K error signatures to include (default: 20).",
    )
    p.add_argument(
        "--include-companies-sample",
        action="store_true",
        help="If set, include a small per-company sample in the report JSON.",
    )
    p.add_argument(
        "--max-companies",
        type=int,
        default=None,
        help="Debug limit: only load/process up to N companies.",
    )

    # LLM-related metrics toggle
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--llm-metrics",
        action="store_true",
        help="Enable LLM-related analysis (default).",
    )
    g.add_argument(
        "--no-llm-metrics",
        action="store_true",
        help="Disable LLM-related analysis (do not compute/aggregate/plot LLM & offerings).",
    )

    # Parallelism
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker threads for per-company analysis. 0 means auto (min(32, cpu_count)).",
    )

    # Run tag controls
    p.add_argument(
        "--run-tag",
        type=str,
        default="analysis-latest",
        help='Analysis run tag folder name under <out_dir>/analysis/. Default: "analysis-latest" (overwrite).',
    )
    p.add_argument(
        "--timestamp-run-tag",
        action="store_true",
        help='If set, ignore --run-tag and use "analysis-YYYYMMDD-HHMMSS" (UTC).',
    )

    # Logging
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Terminal log level (default: INFO).",
    )
    return p


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(str(args.log_level))

    out_dir = Path(args.out_dir)
    ensure_output_root(out_dir)

    run_tag = _utc_run_tag() if bool(args.timestamp_run_tag) else str(args.run_tag)

    workers = int(args.workers)
    if workers <= 0:
        cpu = os.cpu_count() or 4
        workers = min(32, cpu)

    view = str(args.view)
    industry_id = args.industry if args.industry is not None else None
    if view == "lv1" and industry_id is None:
        LOG.error("--view lv1 requires --industry <id>")
        return 2

    # Default is enabled unless explicitly disabled
    include_llm_metrics = True
    if bool(args.no_llm_metrics):
        include_llm_metrics = False
    elif bool(args.llm_metrics):
        include_llm_metrics = True

    kwargs: Dict[str, Any] = dict(
        out_dir=out_dir,
        run_tag=run_tag,
        view_mode=view,
        industry_id=industry_id,
        k_top_domains=int(args.k_top_domains),
        k_top_errors=int(args.k_top_errors),
        include_companies_sample=bool(args.include_companies_sample),
        max_companies=args.max_companies,
        workers=workers,
        include_llm_metrics=bool(include_llm_metrics),
    )

    paths = run_analysis(**kwargs)

    print(str(paths.analysis_root))
    print(str(paths.report_json))
    print(str(paths.report_html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
