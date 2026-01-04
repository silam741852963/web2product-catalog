from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from extensions.dataset.partitioning import (
    Table,
    merge_run_roots,
    merge_tables_dedupe,
    read_table_preserve,
    split_run_root,
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
    return split_run_root(
        root=Path(args.root),
        dataset_path=Path(args.dataset),
        bundle_root=Path(args.bundle_root),
        remain_dataset_out=Path(args.remaining_dataset_out),
        moved_dataset_out=Path(args.moved_dataset_out),
        move_count=args.move_count,
        move_ids_file=Path(args.move_ids_file) if args.move_ids_file else None,
        seed=int(args.seed),
        only_not_done=bool(args.only_not_done),
        move_mode=str(args.move_mode),
        include_ledger=bool(args.include_ledger),
        dry_run=bool(args.dry_run),
    )


def cmd_merge(args: argparse.Namespace) -> Dict[str, Any]:
    return merge_run_roots(
        target_root=Path(args.target_root),
        source_roots=[Path(x) for x in args.source_roots],
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

    merged_table = Table(
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
        description="Split/merge web2product-catalog run roots (outputs + crawl_state + retry) with crawl.state.py semantics."
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser(
        "split",
        help="Split one run root into remaining + bundle (to move/copy to another machine).",
    )
    ps.add_argument(
        "--root",
        required=True,
        help="Run root (folder containing crawl_state.sqlite3, company dirs, _retry).",
    )
    ps.add_argument(
        "--dataset",
        required=True,
        help="Dataset CSV/TSV used by this run root.",
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
        help="Write moved dataset CSV/TSV here (also copied into bundle root).",
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

    ps.add_argument("--seed", type=int, default=7)
    ps.add_argument(
        "--only-not-done",
        action="store_true",
        help="Only move companies that are not in a done status.",
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
    ps.add_argument("--dry-run", action="store_true")
    ps.set_defaults(func=cmd_split)

    pm = sub.add_parser(
        "merge", help="Merge multiple run roots/bundles into a single target root."
    )
    pm.add_argument("--target-root", required=True)
    pm.add_argument("--source-roots", nargs="+", required=True)
    pm.add_argument(
        "--move-mode",
        choices=["move", "copy"],
        default="copy",
        help="copy: keep sources; move: move company dirs into target and scrub sources.",
    )
    pm.add_argument("--include-ledger", action="store_true")
    pm.add_argument("--dry-run", action="store_true")
    pm.set_defaults(func=cmd_merge)

    pc = sub.add_parser(
        "merge-csv", help="Merge two dataset CSV/TSV files and dedupe by company id."
    )
    pc.add_argument("--base-dataset", required=True)
    pc.add_argument("--add-dataset", required=True)
    pc.add_argument("--output-dataset", required=True)
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
