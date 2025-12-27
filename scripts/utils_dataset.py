"""
utils_dataset.py - Dataset utilities for web2product-catalog style pipelines

Features
- convert:   .dta <-> .csv (via pandas)
- split:     split csv into N parts or chunk size
- select:    balanced sampling by industry (or any group field)
- stats:     quick dataset stats (counts, nulls, group distribution)
- dedup:     deduplicate by key(s)
- filter:    filter rows by simple conditions
- sort:      sort by columns
- schema:    show column names + dtypes
- join:      merge two files by key(s)

Examples
  # Convert Orbis .dta to csv
  python scripts/utils_dataset.py convert input.dta -o output.csv

  # Split into 5 parts for 5 servers
  python scripts/utils_dataset.py split input.csv --parts 5 -o out_dir/

  # Balanced sample: 1000 companies total across industry codes
  python scripts/utils_dataset.py select input.csv --group industry --total 1000 -o sample.csv

  # Balanced sample: 50 per industry, only keep certain columns
  python scripts/utils_dataset.py select input.csv --group industry --per-group 50 \
      --cols industry,bvdid,name -o sample.csv

  # Stats (overall + per industry)
  python scripts/utils_dataset.py stats input.csv --group industry

Notes
- Default key columns often used in your project: bvdid, website, company_id.
- For huge files, pandas still loads into memory. If your Orbis export is massive,
  split first, then sample on smaller chunks.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


# ---------------------------
# Helpers
# ---------------------------


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_table(
    path: Path, *, sep: str = ",", dtype: Optional[str] = None
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=sep, dtype=dtype, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=dtype, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".dta":
        # Orbis exports are often Stata. pandas can read them.
        return pd.read_stata(path, convert_categoricals=False)
    raise SystemExit(f"Unsupported input format: {path}")


def _write_table(df: pd.DataFrame, path: Path, *, sep: str = ",") -> None:
    _ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".tsv":
        df.to_csv(path, index=False, sep="\t")
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".dta":
        # Stata has some type constraints; try to be safe.
        df.to_stata(path, write_index=False, version=118)
        return
    raise SystemExit(f"Unsupported output format: {path}")


def _parse_cols(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def _maybe_select_cols(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    if not cols:
        return df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    return df[cols].copy()


def _seed_everything(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)


def _normalize_group_value(x) -> str:
    if pd.isna(x):
        return "__NA__"
    return str(x).strip()


# ---------------------------
# Commands
# ---------------------------


def cmd_convert(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output) if args.output else None

    df = _read_table(inp, sep=args.sep, dtype=args.dtype)

    if args.cols:
        df = _maybe_select_cols(df, _parse_cols(args.cols))

    if out is None:
        # default: switch between dta <-> csv
        if inp.suffix.lower() == ".dta":
            out = inp.with_suffix(".csv")
        else:
            out = inp.with_suffix(".dta")

    _write_table(df, out, sep=args.sep)
    print(f"Wrote: {out}  rows={len(df):,} cols={len(df.columns):,}")


def cmd_split(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype)
    n = len(df)

    if args.parts is None and args.chunk_size is None:
        raise SystemExit("Provide --parts or --chunk-size")

    if args.parts is not None:
        parts = int(args.parts)
        if parts <= 0:
            raise SystemExit("--parts must be > 0")
        chunk = math.ceil(n / parts)
    else:
        chunk = int(args.chunk_size)
        if chunk <= 0:
            raise SystemExit("--chunk-size must be > 0")
        parts = math.ceil(n / chunk)

    stem = args.prefix or inp.stem
    for i in range(parts):
        start = i * chunk
        end = min((i + 1) * chunk, n)
        if start >= end:
            break
        out = out_dir / f"{stem}.part{i + 1:02d}-of{parts:02d}.csv"
        df.iloc[start:end].to_csv(out, index=False)
        print(f"Wrote: {out} rows={end - start:,}")
    print(f"Done split: total_rows={n:,} parts={parts}")


def _balanced_allocation(
    group_counts: pd.Series,
    *,
    total: Optional[int],
    per_group: Optional[int],
    min_per_group: int,
    max_per_group: Optional[int],
) -> pd.Series:
    """
    Decide how many samples per group.
    - If per_group is set: clamp by available and max_per_group
    - Else if total is set: allocate roughly proportional but ensure min_per_group if possible.
    """
    counts = group_counts.copy()

    if per_group is not None:
        alloc = counts.apply(lambda c: min(c, per_group))
        if max_per_group is not None:
            alloc = alloc.clip(upper=max_per_group)
        if min_per_group > 0:
            # If a group has less than min_per_group available, we just take what's available.
            alloc = alloc.apply(
                lambda a, c=None: a
            )  # noop, already bounded by availability
        return alloc.astype(int)

    if total is None:
        raise SystemExit("Provide either --total or --per-group")

    total = int(total)
    if total <= 0:
        raise SystemExit("--total must be > 0")

    # Start with min_per_group baseline if possible.
    alloc = pd.Series(0, index=counts.index, dtype=int)

    if min_per_group > 0:
        base = counts.apply(lambda c: min(c, min_per_group)).astype(int)
        alloc += base
        remaining = total - int(base.sum())
    else:
        remaining = total

    if remaining <= 0:
        # We already filled enough by min_per_group
        return alloc.clip(upper=counts).astype(int)

    # Allocate remaining proportional to availability left
    left = (counts - alloc).clip(lower=0)
    left_sum = int(left.sum())
    if left_sum == 0:
        return alloc.astype(int)

    # Proportional float allocation
    frac = (left / left_sum) * remaining
    extra = frac.floor().astype(int)
    alloc += extra

    # Distribute the remainder by largest fractional parts
    rem = remaining - int(extra.sum())
    if rem > 0:
        order = (frac - frac.floor()).sort_values(ascending=False).index.tolist()
        for g in order:
            if rem <= 0:
                break
            if alloc[g] < counts[g]:
                alloc[g] += 1
                rem -= 1

    # Apply max_per_group if provided
    if max_per_group is not None:
        alloc = alloc.clip(upper=max_per_group)

    # Never exceed available
    alloc = alloc.clip(upper=counts)
    return alloc.astype(int)


def cmd_select(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)

    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype)

    group_col = args.group
    if group_col not in df.columns:
        raise SystemExit(
            f"Group column not found: {group_col}. Available: {list(df.columns)}"
        )

    # Normalize group values to avoid weird NaN group behavior.
    df[group_col] = df[group_col].map(_normalize_group_value)

    # Optional dedup before sampling (common for Orbis exports)
    if args.dedup_key:
        keys = _parse_cols(args.dedup_key)
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise SystemExit(f"Missing dedup keys: {missing}")
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="first")
        print(f"Dedup: {before:,} -> {len(df):,} by {keys}")

    # Optional filter (very lightweight)
    # Example: --where "country=US" or --where "website!=__NA__"
    if args.where:
        # support "col=value" or "col!=value"
        cond = args.where.strip()
        if "!=" in cond:
            col, val = cond.split("!=", 1)
            col, val = col.strip(), val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            before = len(df)
            df = df[df[col].astype(str) != val]
            print(f"Filter where {col}!={val}: {before:,} -> {len(df):,}")
        elif "=" in cond:
            col, val = cond.split("=", 1)
            col, val = col.strip(), val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            before = len(df)
            df = df[df[col].astype(str) == val]
            print(f"Filter where {col}={val}: {before:,} -> {len(df):,}")
        else:
            raise SystemExit("--where must be like col=value or col!=value")

    # Compute allocation
    counts = df.groupby(group_col).size().sort_index()
    alloc = _balanced_allocation(
        counts,
        total=args.total,
        per_group=args.per_group,
        min_per_group=int(args.min_per_group),
        max_per_group=args.max_per_group,
    )

    # Sample per group
    samples = []
    for g, k in alloc.items():
        if k <= 0:
            continue
        sub = df[df[group_col] == g]
        # reproducible group sampling if seed is set
        if args.seed is not None:
            # derive per-group seed so group order doesn’t affect result
            group_seed = hash((args.seed, g)) & 0xFFFFFFFF
            part = sub.sample(n=int(k), random_state=group_seed)
        else:
            part = sub.sample(n=int(k))
        samples.append(part)

    if not samples:
        raise SystemExit("No rows selected (allocation ended up as all zeros).")

    out_df = pd.concat(samples, ignore_index=True)

    # Optional shuffle for better distribution in output file
    if args.shuffle:
        if args.seed is not None:
            out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(
                drop=True
            )
        else:
            out_df = out_df.sample(frac=1.0).reset_index(drop=True)

    # Optional select columns
    if args.cols:
        out_df = _maybe_select_cols(out_df, _parse_cols(args.cols))

    _write_table(out_df, out, sep=args.sep)

    print(f"Wrote: {out} rows={len(out_df):,} cols={len(out_df.columns):,}")
    print("Selected per group:")
    # Print a compact report
    report = out_df.groupby(group_col).size().sort_values(ascending=False)
    for g, n in report.items():
        print(f"  {g}: {n:,}")


def cmd_stats(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    df = _read_table(inp, sep=args.sep, dtype=args.dtype)

    print(f"File: {inp}")
    print(f"Rows: {len(df):,}")
    print(f"Cols: {len(df.columns):,}")
    print("Columns:")
    for c in df.columns:
        nulls = int(df[c].isna().sum())
        print(f"  - {c}  dtype={df[c].dtype}  nulls={nulls:,}")

    if args.group:
        g = args.group
        if g not in df.columns:
            raise SystemExit(f"Group column not found: {g}")
        tmp = df[g].map(_normalize_group_value)
        dist = tmp.value_counts(dropna=False)
        print(f"\nGroup distribution: {g}")
        print(dist.to_string())


def cmd_dedup(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype)
    keys = _parse_cols(args.keys)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise SystemExit(f"Missing keys: {missing}")

    before = len(df)
    df = df.drop_duplicates(
        subset=keys, keep=("first" if args.keep == "first" else "last")
    )
    _write_table(df, out, sep=args.sep)
    print(f"Wrote: {out} rows={len(df):,} (dedup {before:,} -> {len(df):,})")


def cmd_filter(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype)

    # multiple --where supported
    for cond in args.where or []:
        cond = cond.strip()
        if "!=" in cond:
            col, val = cond.split("!=", 1)
            col, val = col.strip(), val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            df = df[df[col].astype(str) != val]
        elif "=" in cond:
            col, val = cond.split("=", 1)
            col, val = col.strip(), val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            df = df[df[col].astype(str) == val]
        else:
            raise SystemExit("--where must be like col=value or col!=value")

    if args.cols:
        df = _maybe_select_cols(df, _parse_cols(args.cols))

    _write_table(df, out, sep=args.sep)
    print(f"Wrote: {out} rows={len(df):,} cols={len(df.columns):,}")


def cmd_sort(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype)
    cols = _parse_cols(args.by)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing sort columns: {missing}")

    df = df.sort_values(by=cols, ascending=not args.desc, kind="mergesort")
    if args.cols:
        df = _maybe_select_cols(df, _parse_cols(args.cols))

    _write_table(df, out, sep=args.sep)
    print(f"Wrote: {out} rows={len(df):,}")


def cmd_schema(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    df = _read_table(inp, sep=args.sep, dtype=args.dtype)
    for c in df.columns:
        print(f"{c}\t{df[c].dtype}")


def cmd_join(args: argparse.Namespace) -> None:
    left = Path(args.left)
    right = Path(args.right)
    out = Path(args.output)

    df_l = _read_table(left, sep=args.sep, dtype=args.dtype)
    df_r = _read_table(right, sep=args.sep, dtype=args.dtype)

    keys = _parse_cols(args.on)
    for k in keys:
        if k not in df_l.columns:
            raise SystemExit(f"Key {k} missing in left")
        if k not in df_r.columns:
            raise SystemExit(f"Key {k} missing in right")

    how = args.how
    df = df_l.merge(df_r, on=keys, how=how, suffixes=("_l", "_r"))
    _write_table(df, out, sep=args.sep)
    print(f"Wrote: {out} rows={len(df):,} cols={len(df.columns):,}")


# ---------------------------
# CLI
# ---------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="utils_dataset.py",
        description="Dataset utilities: convert/split/select/stats/dedup/filter/sort/join",
    )
    p.add_argument(
        "--sep",
        default=",",
        help="CSV separator (default: ,). Ignored for .dta/.parquet.",
    )
    p.add_argument(
        "--dtype", default=None, help="Force dtype for CSV reading (e.g., 'string')"
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # convert
    s = sub.add_parser("convert", help="Convert between .dta/.csv/.tsv/.parquet")
    s.add_argument("input")
    s.add_argument("-o", "--output", default=None)
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.set_defaults(func=cmd_convert)

    # split
    s = sub.add_parser("split", help="Split a CSV into parts or chunks")
    s.add_argument("input")
    s.add_argument("-o", "--output-dir", required=True)
    s.add_argument("--parts", type=int, default=None, help="Number of parts")
    s.add_argument("--chunk-size", type=int, default=None, help="Rows per chunk")
    s.add_argument(
        "--prefix", default=None, help="Output file prefix (default: input stem)"
    )
    s.set_defaults(func=cmd_split)

    # select (balanced)
    s = sub.add_parser(
        "select", help="Balanced sampling by group field (e.g., industry)"
    )
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument(
        "--group", default="industry", help="Group field (default: industry)"
    )
    s.add_argument(
        "--total", type=int, default=None, help="Total rows to sample across all groups"
    )
    s.add_argument(
        "--per-group",
        type=int,
        default=None,
        help="Sample exactly this many per group (clamped by availability)",
    )
    s.add_argument(
        "--min-per-group",
        type=int,
        default=0,
        help="Ensure at least this many per group if possible (only when using --total)",
    )
    s.add_argument("--max-per-group", type=int, default=None, help="Cap per group")
    s.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling (default: 42). Use -1 to disable.",
    )
    s.add_argument("--shuffle", action="store_true", help="Shuffle final output rows")
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.add_argument(
        "--dedup-key",
        default=None,
        help="Comma-separated keys to deduplicate before sampling (e.g., bvdid)",
    )
    s.add_argument(
        "--where", default=None, help="Light filter like country=US or website!=__NA__"
    )
    s.set_defaults(func=cmd_select)

    # stats
    s = sub.add_parser("stats", help="Show dataset stats")
    s.add_argument("input")
    s.add_argument(
        "--group", default=None, help="Optional group field (e.g., industry)"
    )
    s.set_defaults(func=cmd_stats)

    # dedup
    s = sub.add_parser("dedup", help="Deduplicate rows by key(s)")
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument(
        "--keys",
        required=True,
        help="Comma-separated keys, e.g. bvdid or bvdid,website",
    )
    s.add_argument("--keep", choices=["first", "last"], default="first")
    s.set_defaults(func=cmd_dedup)

    # filter
    s = sub.add_parser("filter", help="Filter rows by conditions")
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument(
        "--where",
        action="append",
        help="Condition col=value or col!=value (repeatable)",
    )
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.set_defaults(func=cmd_filter)

    # sort
    s = sub.add_parser("sort", help="Sort by columns")
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--by", required=True, help="Comma-separated columns")
    s.add_argument("--desc", action="store_true")
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.set_defaults(func=cmd_sort)

    # schema
    s = sub.add_parser("schema", help="Print columns + dtypes")
    s.add_argument("input")
    s.set_defaults(func=cmd_schema)

    # join
    s = sub.add_parser("join", help="Join two datasets")
    s.add_argument("--left", required=True)
    s.add_argument("--right", required=True)
    s.add_argument("--on", required=True, help="Comma-separated join keys")
    s.add_argument("--how", choices=["inner", "left", "right", "outer"], default="left")
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(func=cmd_join)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Allow disabling seed via --seed -1
    if hasattr(args, "seed") and args.seed == -1:
        args.seed = None

    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
