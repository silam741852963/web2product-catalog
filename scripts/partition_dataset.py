from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from extensions.dataset.partitioning import (
    merge_run_roots,
    merge_tables_dedupe,
    read_table_preserve,
    split_run_root,
    write_table_preserve,
)

logger = logging.getLogger("partition_dataset")


def _dump_json(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _add_log_level(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s: %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run-root partitioning & merging tool (split/merge/merge-csv)."
    )
    _add_log_level(parser)

    sub = parser.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------
    # split
    # ------------------------------------------------------------
    p_split = sub.add_parser(
        "split", help="Split a run root into a bundle root + remaining/moved datasets."
    )
    p_split.add_argument("--root", required=True, type=Path)
    p_split.add_argument("--dataset", required=True, type=Path)
    p_split.add_argument("--bundle-root", required=True, type=Path)
    p_split.add_argument("--remaining-dataset-out", required=True, type=Path)
    p_split.add_argument("--moved-dataset-out", required=True, type=Path)

    g = p_split.add_mutually_exclusive_group(required=True)
    g.add_argument("--move-count", type=int, default=None)
    g.add_argument("--move-ids-file", type=Path, default=None)

    p_split.add_argument("--seed", type=int, default=7)
    p_split.add_argument("--only-not-done", action="store_true")
    p_split.add_argument("--move-mode", choices=["move", "copy"], default="move")
    p_split.add_argument("--include-ledger", action="store_true")
    p_split.add_argument("--dry-run", action="store_true")

    # ------------------------------------------------------------
    # merge
    # ------------------------------------------------------------
    p_merge = sub.add_parser(
        "merge", help="Merge multiple run roots/bundles into one target root."
    )
    p_merge.add_argument("--target-root", required=True, type=Path)
    p_merge.add_argument("--source-roots", required=True, nargs="+", type=Path)
    p_merge.add_argument("--move-mode", choices=["copy", "move"], default="copy")
    p_merge.add_argument("--include-ledger", action="store_true")
    p_merge.add_argument("--dry-run", action="store_true")

    # ------------------------------------------------------------
    # merge-csv
    # ------------------------------------------------------------
    p_mcsv = sub.add_parser(
        "merge-csv", help="Merge two dataset CSV/TSV files and dedupe by company id."
    )
    p_mcsv.add_argument("--base-dataset", required=True, type=Path)
    p_mcsv.add_argument("--add-dataset", required=True, type=Path)
    p_mcsv.add_argument("--output-dataset", required=True, type=Path)

    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    if args.cmd == "split":
        res = split_run_root(
            root=args.root,
            dataset=args.dataset,
            bundle_root=args.bundle_root,
            remaining_dataset_out=args.remaining_dataset_out,
            moved_dataset_out=args.moved_dataset_out,
            move_count=args.move_count,
            move_ids_file=args.move_ids_file,
            seed=int(args.seed),
            only_not_done=bool(args.only_not_done),
            move_mode=str(args.move_mode),
            include_ledger=bool(args.include_ledger),
            dry_run=bool(args.dry_run),
        )
        _dump_json(res)
        return

    if args.cmd == "merge":
        res = merge_run_roots(
            target_root=args.target_root,
            source_roots=args.source_roots,
            move_mode=str(args.move_mode),
            include_ledger=bool(args.include_ledger),
            dry_run=bool(args.dry_run),
        )
        _dump_json(res)
        return

    if args.cmd == "merge-csv":
        base = read_table_preserve(args.base_dataset)
        add = read_table_preserve(args.add_dataset)
        merged_table, merged_rows = merge_tables_dedupe(base, base.rows, add, add.rows)
        write_table_preserve(args.output_dataset, merged_table, merged_rows)

        res = {
            "cmd": "merge-csv",
            "base_dataset": str(args.base_dataset),
            "add_dataset": str(args.add_dataset),
            "output_dataset": str(args.output_dataset),
            "merged_rows": int(len(merged_rows)),
            "header_cols": int(len(merged_table.header)),
            "id_col": merged_table.id_col,
            "delimiter": "\\t" if merged_table.delimiter == "\t" else ",",
        }
        _dump_json(res)
        return

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
