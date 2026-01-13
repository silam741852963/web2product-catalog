from __future__ import annotations

import argparse
import sys
from typing import Optional

from extensions.nice_match.pipeline import run_nice_sentence_matching


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nice_match_run")

    p.add_argument(
        "--out-dir", required=True, help="Output root (same meaning as crawl out-dir)."
    )
    p.add_argument(
        "--company-file", required=True, help="CSV containing firm ids (bvd_id)."
    )
    p.add_argument(
        "--company-id-col",
        required=True,
        help="Column name in company-file for bvd_id.",
    )
    p.add_argument(
        "--nice-xlsx",
        required=True,
        help="Path to data/industry/nice_classification.xlsx",
    )
    p.add_argument(
        "--nice-sheet", default="0", help="Excel sheet index (int) or sheet name."
    )
    p.add_argument("--output-csv", required=True, help="Destination matches CSV path.")
    p.add_argument(
        "--sentences-csv", required=True, help="Destination sentences CSV path."
    )
    p.add_argument(
        "--stats-json",
        required=True,
        help="Destination JSON stats path (scraped firms + sentences-per-firm mean/median).",
    )
    p.add_argument(
        "--minimal",
        action="store_true",
        help="If set, matches CSV contains only: bvd_id,sentence_id,token_id.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (>=1). If >1, writes part files and merges deterministically.",
    )
    p.add_argument(
        "--missing-policy",
        required=True,
        choices=["error", "skip"],
        help="What to do if a firm's markdown folder is missing.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of firms processed.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)

    if ns.workers < 1:
        raise ValueError("--workers must be >= 1")

    sheet_raw = ns.nice_sheet
    if isinstance(sheet_raw, str) and sheet_raw.isdigit():
        sheet = int(sheet_raw)
    else:
        sheet = sheet_raw

    run_nice_sentence_matching(
        out_dir=str(ns.out_dir),
        company_file=str(ns.company_file),
        company_id_col=str(ns.company_id_col),
        nice_xlsx=str(ns.nice_xlsx),
        nice_sheet=sheet,
        output_csv=str(ns.output_csv),
        sentences_csv=str(ns.sentences_csv),
        stats_json=str(ns.stats_json),
        minimal=bool(ns.minimal),
        missing_policy=str(ns.missing_policy),
        limit=ns.limit,
        workers=int(ns.workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
