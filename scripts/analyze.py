from __future__ import annotations

import argparse
import json
from pathlib import Path

from extensions.stats_analyzer import analyze_and_plot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate crawl_meta.json, compute statistics, and generate charts."
    )
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing per-company outputs/*/checkpoints/crawl_meta.json (and optional url_index.json).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_out"),
        help="Directory to write charts, CSV, and summary JSON.",
    )
    p.add_argument(
        "--cap-percentile",
        type=float,
        default=0.90,
        help="Percentile (0.0–1.0) to suggest company_max_pages (default: 0.90 → 90th).",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save the per-company CSV.",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="Do not save the summary JSON.",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Also write interactive HTML charts (requires plotly). Interactive charts include hover data to trace each point to a company (bvdid and company_name).",
    )
    p.add_argument(
        "--cost-per-1k-tokens",
        type=float,
        default=0.03,
        help="USD cost per 1k tokens to use for LLM extraction cost estimates in the summary (default: 0.03).",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze_and_plot(
        outputs_dir=args.outputs_dir,
        out_dir=args.out_dir,
        cap_percentile=args.cap_percentile,
        save_csv=not args.no_csv,
        save_json_summary=not args.no_json,
        interactive=args.interactive,
        cost_per_1k_tokens_usd=args.cost_per_1k_tokens,
    )

    # Print a compact console summary and where artifacts went
    print("\n=== Analysis Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {args.out_dir.resolve()}")
    print(
        "Key files: "
        "crawl_meta_aggregate.csv (if enabled), summary.json (if enabled), and several *.png charts.\n"
        "Interactive charts (if requested) contain hover info showing bvdid and company_name so you can identify which point belongs to which company.\n"
        "Recommendation → company_max_pages ~= summary['recommendations']['company_max_pages_recommended']"
    )


if __name__ == "__main__":
    main()