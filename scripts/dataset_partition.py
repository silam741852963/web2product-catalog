from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions.dataset_partitioning import (  # noqa: E402
    merge_run_roots,
    split_run_root,
    read_table_preserve,
    merge_tables_dedupe,
    write_table_preserve,
)

logger = logging.getLogger("dataset_partition")


def _setup_logging(level: str) -> None:
    lv = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lv,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.setLevel(lv)


def cmd_split(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.root)
    dataset = Path(args.dataset)

    bundle_root = Path(args.bundle_root)
    remain_out = Path(args.remaining_dataset_out)
    moved_out = Path(args.moved_dataset_out)

    move_ids_file = Path(args.move_ids_file) if args.move_ids_file else None

    return split_run_root(
        root=root,
        dataset_path=dataset,
        bundle_root=bundle_root,
        remain_dataset_out=remain_out,
        moved_dataset_out=moved_out,
        move_count=args.move_count,
        move_ids_file=move_ids_file,
        seed=int(args.seed),
        only_not_done=bool(args.only_not_done),
        move_mode=str(args.move_mode),
        include_ledger=bool(args.include_ledger),
        dry_run=bool(args.dry_run),
    )


def cmd_merge(args: argparse.Namespace) -> Dict[str, Any]:
    target = Path(args.target_root)
    sources = [Path(x) for x in args.source_roots]
    return merge_run_roots(
        target_root=target,
        source_roots=sources,
        move_mode=str(args.move_mode),
        include_ledger=bool(args.include_ledger),
        dry_run=bool(args.dry_run),
    )


def cmd_merge_csv(args: argparse.Namespace) -> Dict[str, Any]:
    base_path = Path(args.base_dataset)
    add_path = Path(args.add_dataset)
    out_path = Path(args.output_dataset)

    base = read_table_preserve(base_path)
    add = read_table_preserve(add_path)

    header_union, merged_rows = merge_tables_dedupe(base, base.rows, add, add.rows)

    # write with base delimiter + base id_col
    merged_table = type(base)(
        header=header_union,
        rows=[],
        id_col=base.id_col,
        url_col=base.url_col,
        delimiter=base.delimiter,
    )
    write_table_preserve(out_path, merged_table, merged_rows)
    return {
        "base_dataset": str(base_path),
        "add_dataset": str(add_path),
        "output_dataset": str(out_path),
        "merged_rows": len(merged_rows),
        "header_cols": len(header_union),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split/merge web2product-catalog run roots (outputs + crawl_state + retry) without leaving special traces."
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # split
    ps = sub.add_parser(
        "split",
        help="Split one run root into remaining + bundle (to move/copy to a free server).",
    )
    ps.add_argument(
        "--root",
        required=True,
        help="Run root (same folder as crawl_state.sqlite3, company dirs, _retry).",
    )
    ps.add_argument(
        "--dataset", required=True, help="Dataset CSV/TSV used by this run root."
    )
    ps.add_argument(
        "--bundle-root",
        required=True,
        help="Output run root for the moved partition (bundle).",
    )
    ps.add_argument(
        "--remaining-dataset-out",
        required=True,
        help="Write remaining dataset CSV/TSV here.",
    )
    ps.add_argument(
        "--moved-dataset-out",
        required=True,
        help="Write moved dataset CSV/TSV here (also copied into bundle).",
    )

    g = ps.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--move-count",
        type=int,
        default=None,
        help="How many companies to move (picked from candidates).",
    )
    g.add_argument(
        "--move-ids-file",
        type=str,
        default=None,
        help="File with company IDs (one per line) to move.",
    )

    ps.add_argument(
        "--seed", type=int, default=7, help="Seed used when selecting by --move-count."
    )
    ps.add_argument(
        "--only-not-done",
        action="store_true",
        help="Only move companies that are not in a done status (default behavior).",
    )
    ps.add_argument(
        "--move-mode",
        choices=["move", "copy"],
        default="move",
        help="move: remove moved companies from source root; copy: duplicate them.",
    )
    ps.add_argument(
        "--include-ledger",
        action="store_true",
        help="Also filter+copy failure_ledger.jsonl rows for moved IDs into bundle.",
    )
    ps.add_argument(
        "--dry-run", action="store_true", help="Plan only; do not write anything."
    )
    ps.set_defaults(func=cmd_split)

    # merge
    pm = sub.add_parser(
        "merge", help="Merge multiple run roots/bundles into a single target root."
    )
    pm.add_argument(
        "--target-root", required=True, help="Target run root to create/update."
    )
    pm.add_argument(
        "--source-roots",
        nargs="+",
        required=True,
        help="One or more run roots (or bundles) to merge.",
    )
    pm.add_argument(
        "--move-mode",
        choices=["move", "copy"],
        default="copy",
        help="copy: keep sources; move: move company dirs into target.",
    )
    pm.add_argument(
        "--include-ledger",
        action="store_true",
        help="Also concatenate failure_ledger.jsonl from sources into target.",
    )
    pm.add_argument(
        "--dry-run", action="store_true", help="Plan only; do not write anything."
    )
    pm.set_defaults(func=cmd_merge)

    # merge-csv
    pc = sub.add_parser(
        "merge-csv", help="Merge two dataset CSV/TSV files and dedupe by company id."
    )
    pc.add_argument("--base-dataset", required=True, help="Existing dataset CSV/TSV.")
    pc.add_argument("--add-dataset", required=True, help="Dataset CSV/TSV to append.")
    pc.add_argument(
        "--output-dataset", required=True, help="Merged output dataset CSV/TSV."
    )
    pc.set_defaults(func=cmd_merge_csv)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    res = args.func(args)  # type: ignore[attr-defined]
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
