from __future__ import annotations

import argparse
import json
from pathlib import Path

from extensions.stats_analyzer import analyze_and_plot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate crawl progress, compute statistics, and generate charts."
    )
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing per-company outputs/*/checkpoints/progress.json",
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
        help="Also write interactive HTML charts (requires plotly).",
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
    )

    # Print a compact console summary and where artifacts went
    print("\n=== Analysis Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {args.out_dir.resolve()}")
    print(
        "Key files: "
        "progress_aggregate.csv, summary.json, and several *.png charts.\n"
        "Recommendation → company_max_pages ~= summary['recommendations']['company_max_pages_recommended']"
    )


if __name__ == "__main__":
    main()