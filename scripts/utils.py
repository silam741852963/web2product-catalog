"""
utils.py - Composable dataset utilities for web2product-catalog style pipelines

This module is a *toolkit* + *CLI* for deterministic dataset manipulation.

Design principles
- Single-responsibility commands: each subcommand does exactly one thing.
- Explicit behavior: required inputs must be provided; invalid inputs fail loudly.
- Composable workflows: use the `run` command to execute an explicit pipeline of steps.
- Multi-format I/O: supports csv/tsv/xlsx/parquet/dta (Stata).

Supported formats
- .csv, .tsv
- .xlsx (single sheet; default sheet=0)
- .parquet
- .dta (Stata)

===============================================================================
CLI Overview
===============================================================================

Core commands
- convert        Convert between supported formats
- split          Split a dataset into parts or chunk sizes
- copy           Copy with column selection/reorder/rename (multi-format)
- idcopy         ID-based cross-file copy (IDs from one file, rows from another)
- select         Group-based selection (first/last/random) with optional append
- stats          Print basic stats and optional group distribution
- dedup          Deduplicate by key(s)
- filter         Filter rows by col=value / col!=value (repeatable)
- sort           Sort by columns (stable)
- schema         Print columns + dtypes
- join           Join two datasets on key(s)
- run            Run an explicit pipeline (JSON) to chain commands

===============================================================================
Examples
===============================================================================

1) Convert
  python scripts/utils.py convert data/us/orbis.dta -o data/us/orbis.csv

2) Copy (select/reorder/rename)
  python scripts/utils.py copy data/in.xlsx -o data/out.csv \
    --cols bvdid,name,industry \
    --rename "bvdid:company_id,name:company_name" \
    --order company_id,company_name,industry

3) ID-based copy (IDs from file A, rows from file B)
  # Take IDs from ids.csv (col=bvdid), fetch matching rows from companies.csv (col=bvdid)
  python scripts/utils.py idcopy \
    --ids-file data/ids.csv --ids-col bvdid \
    --src-file data/companies.csv --src-id-col bvdid \
    -o data/matched.csv

  # With column mapping (flexible output schema):
  python scripts/utils.py idcopy \
    --ids-file data/ids.csv --ids-col bvdid \
    --src-file data/companies.csv --src-id-col bvdid \
    -o data/matched.csv \
    --map "bvdid:bvdid,name:name,website:url"

4) Group-based selection (beyond "first N")
  # 50 per industry, random
  python scripts/utils.py select data/us.csv -o data/sample.csv \
    --group industry --per-group 50 --strategy random --seed 42

  # first 10 per group (preserves input order)
  python scripts/utils.py select data/us.csv -o data/sample.csv \
    --group industry --per-group 10 --strategy first

  # last 10 per group (preserves input order)
  python scripts/utils.py select data/us.csv -o data/sample.csv \
    --group industry --per-group 10 --strategy last

5) Append selection into an existing file (augment)
  # Append new sampled rows into existing target, validating schema compatibility
  python scripts/utils.py select data/us.csv -o data/sample.csv \
    --group industry --per-group 10 --strategy random --seed 1 \
    --append

  # Append + dedup by key after append (optional)
  python scripts/utils.py select data/us.csv -o data/sample.csv \
    --group industry --per-group 10 --strategy random --seed 1 \
    --append --dedup-key bvdid

6) Run a pipeline (explicit chaining)
  Create pipeline.json:
  {
    "steps": [
      {
        "cmd": "dedup",
        "args": { "input": "data/us.csv", "output": "data/us.dedup.csv", "keys": "bvdid", "keep": "first" }
      },
      {
        "cmd": "select",
        "args": {
          "input": "data/us.dedup.csv",
          "output": "data/us.sample.csv",
          "group": "industry",
          "per_group": 50,
          "strategy": "random",
          "seed": 42,
          "cols": "industry,bvdid,name"
        }
      }
    ]
  }

  Run it:
    python scripts/utils.py run pipeline.json

===============================================================================
Typical workflows
===============================================================================

- Split → run crawler shards:
  split large.csv into 20 parts; distribute parts across servers.

- Dedup → select balanced evaluation set:
  dedup by bvdid; select 5 per industry, random seed 42; keep only columns needed.

- ID-driven incremental dataset building:
  maintain an "ids.csv" list of targets; idcopy from a master company table into a curated subset.

- Augment over time:
  repeatedly `select --append` into an evaluation file, using deterministic seeds per run and
  optional `--dedup-key` to avoid duplicates.

===============================================================================
Notes on determinism
===============================================================================
- `--strategy random` is deterministic if `--seed` is provided.
- `--strategy first/last` never uses randomness and preserves source row order.

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# =============================================================================
# I/O helpers
# =============================================================================


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_table(
    path: Path,
    *,
    sep: str,
    dtype: Optional[str],
    sheet: Optional[str],
) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, sep=sep, dtype=dtype, low_memory=False)

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=dtype, low_memory=False)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)

    if suffix == ".xlsx":
        # sheet may be None -> pandas default (0)
        return pd.read_excel(path, sheet_name=(sheet if sheet is not None else 0))

    raise SystemExit(f"Unsupported input format: {path}")


def _write_table(df: pd.DataFrame, path: Path, *, sep: str, sheet: str) -> None:
    _ensure_parent(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False, sep=sep)
        return

    if suffix == ".tsv":
        df.to_csv(path, index=False, sep="\t")
        return

    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return

    if suffix == ".dta":
        df.to_stata(path, write_index=False, version=118)
        return

    if suffix == ".xlsx":
        # write single sheet
        df.to_excel(path, index=False, sheet_name=sheet)
        return

    raise SystemExit(f"Unsupported output format: {path}")


# =============================================================================
# Column helpers
# =============================================================================


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    t = s.strip()
    if t == "":
        return None
    return [c.strip() for c in t.split(",") if c.strip()]


def _parse_mapping(s: Optional[str]) -> Dict[str, str]:
    """
    Parse mapping string: "old:new,old2:new2"
    """
    if s is None:
        return {}
    t = s.strip()
    if t == "":
        return {}
    out: Dict[str, str] = {}
    for part in t.split(","):
        p = part.strip()
        if p == "":
            continue
        if ":" not in p:
            raise SystemExit(f"Invalid mapping item (missing ':'): {p}")
        src, dst = p.split(":", 1)
        src = src.strip()
        dst = dst.strip()
        if src == "" or dst == "":
            raise SystemExit(f"Invalid mapping item (empty side): {p}")
        if src in out:
            raise SystemExit(f"Duplicate mapping source key: {src}")
        out[src] = dst
    return out


def _require_columns(df: pd.DataFrame, cols: Iterable[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {where}: {missing}")


def _apply_col_ops(
    df: pd.DataFrame,
    *,
    cols_keep: Optional[List[str]],
    rename_map: Dict[str, str],
    order: Optional[List[str]],
) -> pd.DataFrame:
    # 1) keep
    if cols_keep is not None:
        _require_columns(df, cols_keep, where="input")
        df = df[cols_keep].copy()

    # 2) rename (only allowed if source cols exist in current df)
    if rename_map:
        _require_columns(df, rename_map.keys(), where="input (rename)")
        df = df.rename(columns=rename_map)

    # 3) reorder
    if order is not None:
        _require_columns(df, order, where="input (order)")
        df = df[order].copy()

    return df


# =============================================================================
# Selection allocation
# =============================================================================


def _normalize_group_value(x: Any) -> str:
    if pd.isna(x):
        return "__NA__"
    return str(x).strip()


def _balanced_allocation(
    group_counts: pd.Series,
    *,
    total: Optional[int],
    per_group: Optional[int],
    min_per_group: int,
    max_per_group: Optional[int],
) -> pd.Series:
    counts = group_counts.copy()

    if per_group is not None:
        alloc = counts.apply(lambda c: min(int(c), int(per_group))).astype(int)
        if max_per_group is not None:
            alloc = alloc.clip(upper=int(max_per_group))
        alloc = alloc.clip(upper=counts.astype(int))
        return alloc.astype(int)

    if total is None:
        raise SystemExit("Provide either --total or --per-group")

    total_i = int(total)
    if total_i <= 0:
        raise SystemExit("--total must be > 0")

    alloc = pd.Series(0, index=counts.index, dtype=int)

    if min_per_group < 0:
        raise SystemExit("--min-per-group must be >= 0")

    if min_per_group > 0:
        base = counts.apply(lambda c: min(int(c), int(min_per_group))).astype(int)
        alloc += base
        remaining = total_i - int(base.sum())
    else:
        remaining = total_i

    if remaining <= 0:
        return alloc.clip(upper=counts.astype(int)).astype(int)

    left = (counts.astype(int) - alloc).clip(lower=0)
    left_sum = int(left.sum())
    if left_sum == 0:
        return alloc.astype(int)

    frac = (left / left_sum) * remaining
    extra = frac.floor().astype(int)
    alloc += extra

    rem = remaining - int(extra.sum())
    if rem > 0:
        order = (frac - frac.floor()).sort_values(ascending=False).index.tolist()
        for g in order:
            if rem == 0:
                break
            if alloc[g] < int(counts[g]):
                alloc[g] += 1
                rem -= 1

    if max_per_group is not None:
        alloc = alloc.clip(upper=int(max_per_group))

    alloc = alloc.clip(upper=counts.astype(int))
    return alloc.astype(int)


# =============================================================================
# Command implementations (single-responsibility)
# =============================================================================


def cmd_convert(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)
    df = _apply_col_ops(
        df,
        cols_keep=_parse_csv_list(args.cols),
        rename_map=_parse_mapping(args.rename),
        order=_parse_csv_list(args.order),
    )
    _write_table(df, out, sep=args.sep, sheet=args.sheet_out)
    print(f"Wrote: {out}  rows={len(df):,} cols={len(df.columns):,}")


def cmd_split(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)
    n = len(df)

    if args.parts is None and args.chunk_size is None:
        raise SystemExit("Provide --parts or --chunk-size")

    if args.parts is not None and args.chunk_size is not None:
        raise SystemExit("Provide only one of --parts or --chunk-size")

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

    stem = args.prefix if args.prefix is not None else inp.stem
    suffix = args.out_suffix.lower()

    if suffix not in {".csv", ".tsv", ".parquet", ".xlsx", ".dta"}:
        raise SystemExit(f"Unsupported --out-suffix: {suffix}")

    for i in range(parts):
        start = i * chunk
        end = min((i + 1) * chunk, n)
        if start >= end:
            break
        out = out_dir / f"{stem}.part{i + 1:02d}-of{parts:02d}{suffix}"
        _write_table(df.iloc[start:end].copy(), out, sep=args.sep, sheet=args.sheet_out)
        print(f"Wrote: {out} rows={end - start:,}")

    print(f"Done split: total_rows={n:,} parts={parts}")


def cmd_copy(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)
    df = _apply_col_ops(
        df,
        cols_keep=_parse_csv_list(args.cols),
        rename_map=_parse_mapping(args.rename),
        order=_parse_csv_list(args.order),
    )
    _write_table(df, out, sep=args.sep, sheet=args.sheet_out)
    print(f"Wrote: {out} rows={len(df):,} cols={len(df.columns):,}")


def cmd_idcopy(args: argparse.Namespace) -> None:
    ids_file = Path(args.ids_file)
    src_file = Path(args.src_file)
    out = Path(args.output)

    ids_df = _read_table(ids_file, sep=args.sep, dtype=args.dtype, sheet=args.sheet_ids)
    if args.ids_col not in ids_df.columns:
        raise SystemExit(f"--ids-col not found in ids-file: {args.ids_col}")
    ids_series = ids_df[args.ids_col].dropna().astype(str)
    ids: set[str] = set(ids_series.tolist())
    if len(ids) == 0:
        raise SystemExit("No IDs found in ids-file after dropping NA")

    src_df = _read_table(src_file, sep=args.sep, dtype=args.dtype, sheet=args.sheet_src)
    if args.src_id_col not in src_df.columns:
        raise SystemExit(f"--src-id-col not found in src-file: {args.src_id_col}")

    # Match IDs; skip rows where src_id_col is NA
    src_id = src_df[args.src_id_col]
    mask = src_id.notna() & src_id.astype(str).isin(ids)
    matched = src_df.loc[mask].copy()

    # Flexible column mapping:
    # - if --map provided, output columns are the mapped "dst" names, sourced from mapped "src" columns
    # - else copy all columns (optionally then apply --cols/--rename/--order)
    map_spec = _parse_mapping(args.map)
    if map_spec:
        _require_columns(matched, map_spec.keys(), where="src-file (map)")
        out_df = matched[list(map_spec.keys())].copy()
        out_df = out_df.rename(columns=map_spec)
    else:
        out_df = matched

    out_df = _apply_col_ops(
        out_df,
        cols_keep=_parse_csv_list(args.cols),
        rename_map=_parse_mapping(args.rename),
        order=_parse_csv_list(args.order),
    )

    _write_table(out_df, out, sep=args.sep, sheet=args.sheet_out)
    print(
        f"Wrote: {out} rows={len(out_df):,} cols={len(out_df.columns):,} (matched_ids={len(ids):,})"
    )


def _select_per_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    alloc: pd.Series,
    strategy: str,
    seed: Optional[int],
) -> pd.DataFrame:
    if strategy not in {"first", "last", "random"}:
        raise SystemExit(f"Unsupported --strategy: {strategy}")

    samples: List[pd.DataFrame] = []

    # Preserve original row order for first/last by using index order
    for g, k in alloc.items():
        kk = int(k)
        if kk <= 0:
            continue

        sub = df[df[group_col] == g]
        if len(sub) == 0:
            continue

        if strategy == "first":
            part = sub.iloc[:kk].copy()
        elif strategy == "last":
            part = sub.iloc[-kk:].copy()
        else:
            if seed is None:
                part = sub.sample(n=kk)
            else:
                group_seed = hash((seed, str(g))) & 0xFFFFFFFF
                part = sub.sample(n=kk, random_state=group_seed)

        samples.append(part)

    if not samples:
        raise SystemExit("No rows selected (allocation ended up as all zeros).")

    return pd.concat(samples, ignore_index=True)


def _validate_append_compat(existing: pd.DataFrame, new_df: pd.DataFrame) -> None:
    # Column names and order must match exactly
    if list(existing.columns) != list(new_df.columns):
        raise SystemExit(
            "Append validation failed: output columns do not match existing file columns exactly.\n"
            f"Existing: {list(existing.columns)}\n"
            f"New:      {list(new_df.columns)}"
        )


def cmd_select(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    if args.seed == -1:
        seed: Optional[int] = None
    else:
        seed = int(args.seed)

    if seed is not None:
        random.seed(seed)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)

    group_col = args.group
    if group_col not in df.columns:
        raise SystemExit(
            f"Group column not found: {group_col}. Available: {list(df.columns)}"
        )

    # Optional dedup before selection
    if args.pre_dedup_key is not None:
        keys = _parse_csv_list(args.pre_dedup_key)
        if keys is None:
            raise SystemExit("--pre-dedup-key is empty")
        _require_columns(df, keys, where="input (pre-dedup-key)")
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="first")
        print(f"Pre-dedup: {before:,} -> {len(df):,} by {keys}")

    # Optional lightweight where conditions (repeatable)
    for cond in args.where or []:
        cond = cond.strip()
        if "!=" in cond:
            col, val = cond.split("!=", 1)
            col = col.strip()
            val = val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            df = df[df[col].astype(str) != val]
        elif "=" in cond:
            col, val = cond.split("=", 1)
            col = col.strip()
            val = val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            df = df[df[col].astype(str) == val]
        else:
            raise SystemExit("--where must be like col=value or col!=value")

    # Normalize group values
    df[group_col] = df[group_col].map(_normalize_group_value)

    counts = df.groupby(group_col).size().sort_index()
    alloc = _balanced_allocation(
        counts,
        total=args.total,
        per_group=args.per_group,
        min_per_group=int(args.min_per_group),
        max_per_group=args.max_per_group,
    )

    out_df = _select_per_group(
        df,
        group_col=group_col,
        alloc=alloc,
        strategy=args.strategy,
        seed=seed,
    )

    if args.shuffle:
        if seed is None:
            out_df = out_df.sample(frac=1.0).reset_index(drop=True)
        else:
            out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Column operations (for the produced output schema)
    out_df = _apply_col_ops(
        out_df,
        cols_keep=_parse_csv_list(args.cols),
        rename_map=_parse_mapping(args.rename),
        order=_parse_csv_list(args.order),
    )

    # Append behavior
    if out.exists():
        if not args.append:
            raise SystemExit(
                f"Output already exists: {out}. Use --append to augment it."
            )

        existing = _read_table(
            out, sep=args.sep, dtype=args.dtype, sheet=args.sheet_existing
        )
        _validate_append_compat(existing, out_df)

        combined = pd.concat([existing, out_df], ignore_index=True)

        # Optional post-append dedup
        if args.dedup_key is not None:
            keys = _parse_csv_list(args.dedup_key)
            if keys is None:
                raise SystemExit("--dedup-key is empty")
            _require_columns(combined, keys, where="output (dedup-key)")
            before = len(combined)
            combined = combined.drop_duplicates(subset=keys, keep="first")
            print(f"Post-append dedup: {before:,} -> {len(combined):,} by {keys}")

        _write_table(combined, out, sep=args.sep, sheet=args.sheet_out)
        print(
            f"Appended: {out} rows_added={len(out_df):,} total_rows={len(combined):,} cols={len(combined.columns):,}"
        )
        return

    _write_table(out_df, out, sep=args.sep, sheet=args.sheet_out)
    print(f"Wrote: {out} rows={len(out_df):,} cols={len(out_df.columns):,}")

    report = out_df.groupby(group_col).size().sort_values(ascending=False)
    print("Selected per group:")
    for g, n in report.items():
        print(f"  {g}: {int(n):,}")


def cmd_stats(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)

    print(f"File: {inp}")
    print(f"Rows: {len(df):,}")
    print(f"Cols: {len(df.columns):,}")
    print("Columns:")
    for c in df.columns:
        nulls = int(df[c].isna().sum())
        print(f"  - {c}  dtype={df[c].dtype}  nulls={nulls:,}")

    if args.group is not None:
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

    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)
    keys = _parse_csv_list(args.keys)
    if keys is None:
        raise SystemExit("--keys is empty")
    _require_columns(df, keys, where="input (dedup)")

    before = len(df)
    keep = args.keep
    df = df.drop_duplicates(subset=keys, keep=keep)
    _write_table(df, out, sep=args.sep, sheet=args.sheet_out)
    print(f"Wrote: {out} rows={len(df):,} (dedup {before:,} -> {len(df):,})")


def cmd_filter(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)

    for cond in args.where:
        cond = cond.strip()
        if "!=" in cond:
            col, val = cond.split("!=", 1)
            col = col.strip()
            val = val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            df = df[df[col].astype(str) != val]
        elif "=" in cond:
            col, val = cond.split("=", 1)
            col = col.strip()
            val = val.strip()
            if col not in df.columns:
                raise SystemExit(f"--where column not found: {col}")
            df = df[df[col].astype(str) == val]
        else:
            raise SystemExit("--where must be like col=value or col!=value")

    df = _apply_col_ops(
        df,
        cols_keep=_parse_csv_list(args.cols),
        rename_map=_parse_mapping(args.rename),
        order=_parse_csv_list(args.order),
    )

    _write_table(df, out, sep=args.sep, sheet=args.sheet_out)
    print(f"Wrote: {out} rows={len(df):,} cols={len(df.columns):,}")


def cmd_sort(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)

    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)

    by = _parse_csv_list(args.by)
    if by is None:
        raise SystemExit("--by is empty")
    _require_columns(df, by, where="input (sort)")

    df = df.sort_values(by=by, ascending=(not args.desc), kind="mergesort")

    df = _apply_col_ops(
        df,
        cols_keep=_parse_csv_list(args.cols),
        rename_map=_parse_mapping(args.rename),
        order=_parse_csv_list(args.order),
    )

    _write_table(df, out, sep=args.sep, sheet=args.sheet_out)
    print(f"Wrote: {out} rows={len(df):,} cols={len(df.columns):,}")


def cmd_schema(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    df = _read_table(inp, sep=args.sep, dtype=args.dtype, sheet=args.sheet_in)
    for c in df.columns:
        print(f"{c}\t{df[c].dtype}")


def cmd_join(args: argparse.Namespace) -> None:
    left = Path(args.left)
    right = Path(args.right)
    out = Path(args.output)

    df_l = _read_table(left, sep=args.sep, dtype=args.dtype, sheet=args.sheet_left)
    df_r = _read_table(right, sep=args.sep, dtype=args.dtype, sheet=args.sheet_right)

    keys = _parse_csv_list(args.on)
    if keys is None:
        raise SystemExit("--on is empty")

    _require_columns(df_l, keys, where="left (join)")
    _require_columns(df_r, keys, where="right (join)")

    df = df_l.merge(df_r, on=keys, how=args.how, suffixes=("_l", "_r"))
    _write_table(df, out, sep=args.sep, sheet=args.sheet_out)
    print(f"Wrote: {out} rows={len(df):,} cols={len(df.columns):,}")


# =============================================================================
# Pipeline runner (explicit chaining)
# =============================================================================


@dataclass(frozen=True)
class PipelineStep:
    cmd: str
    args: Dict[str, Any]


def _load_pipeline(path: Path) -> List[PipelineStep]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Pipeline JSON must be an object with key 'steps'")
    if "steps" not in data:
        raise SystemExit("Pipeline JSON missing required key: steps")
    steps = data["steps"]
    if not isinstance(steps, list):
        raise SystemExit("'steps' must be a list")
    out: List[PipelineStep] = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            raise SystemExit(f"Step {i} must be an object")
        if "cmd" not in s or "args" not in s:
            raise SystemExit(f"Step {i} must contain keys: cmd, args")
        cmd = s["cmd"]
        args = s["args"]
        if not isinstance(cmd, str):
            raise SystemExit(f"Step {i}.cmd must be a string")
        if not isinstance(args, dict):
            raise SystemExit(f"Step {i}.args must be an object")
        out.append(PipelineStep(cmd=cmd, args=args))
    return out


def cmd_run(args: argparse.Namespace) -> None:
    pipeline_path = Path(args.pipeline)
    steps = _load_pipeline(pipeline_path)

    # Map pipeline commands to CLI subcommands: we reuse the same parsers to validate.
    parser = build_parser()

    for i, step in enumerate(steps):
        # Convert dict args -> argv list for that subcommand
        # This is explicit: pipeline must provide arguments exactly as CLI expects.
        argv: List[str] = [step.cmd]
        for k, v in step.args.items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
                else:
                    # false means omit flag; pipeline must not rely on implicit defaults
                    continue
            elif isinstance(v, list):
                for item in v:
                    argv.append(flag)
                    argv.append(str(item))
            else:
                argv.append(flag)
                argv.append(str(v))

        # Parse and execute
        ns = parser.parse_args(argv)
        ns.func(ns)
        print(f"[pipeline] step={i + 1}/{len(steps)} cmd={step.cmd} OK")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="utils.py",
        description="Dataset utilities toolkit (convert/split/copy/idcopy/select/...)",
    )
    p.add_argument(
        "--sep",
        default=",",
        help="CSV separator for .csv I/O (default: ,). Ignored for .parquet/.dta/.xlsx.",
    )
    p.add_argument(
        "--dtype",
        default=None,
        help="Force dtype for CSV reading (e.g., 'string'). Ignored for .xlsx/.parquet/.dta.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # convert
    s = sub.add_parser("convert", help="Convert between supported formats")
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--sheet-in", default=None, help="Input sheet name/index for .xlsx")
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.add_argument("--rename", default=None, help="Rename mapping old:new,old2:new2")
    s.add_argument("--order", default=None, help="Comma-separated column order")
    s.set_defaults(func=cmd_convert)

    # split
    s = sub.add_parser("split", help="Split dataset into parts or chunks")
    s.add_argument("input")
    s.add_argument("-o", "--output-dir", required=True)
    s.add_argument("--parts", type=int, default=None, help="Number of parts")
    s.add_argument("--chunk-size", type=int, default=None, help="Rows per chunk")
    s.add_argument("--prefix", default=None, help="Output file prefix")
    s.add_argument(
        "--out-suffix",
        default=".csv",
        help="Output suffix: .csv/.tsv/.parquet/.xlsx/.dta (default: .csv)",
    )
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.set_defaults(func=cmd_split)

    # copy
    s = sub.add_parser("copy", help="Copy with column selection/reorder/rename")
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.add_argument("--rename", default=None, help="Rename mapping old:new,old2:new2")
    s.add_argument("--order", default=None, help="Comma-separated column order")
    s.set_defaults(func=cmd_copy)

    # idcopy
    s = sub.add_parser(
        "idcopy", help="Copy rows from src-file whose IDs appear in ids-file"
    )
    s.add_argument("--ids-file", required=True)
    s.add_argument("--ids-col", required=True)
    s.add_argument("--src-file", required=True)
    s.add_argument("--src-id-col", required=True)
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--sheet-ids", default=None, help="Sheet for ids-file if .xlsx")
    s.add_argument("--sheet-src", default=None, help="Sheet for src-file if .xlsx")
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.add_argument(
        "--map",
        default=None,
        help="Optional mapping src_col:out_col,... (defines output schema)",
    )
    s.add_argument(
        "--cols", default=None, help="Comma-separated columns to keep (post-map)"
    )
    s.add_argument("--rename", default=None, help="Rename mapping old:new (post-map)")
    s.add_argument(
        "--order", default=None, help="Comma-separated column order (post-map)"
    )
    s.set_defaults(func=cmd_idcopy)

    # select
    s = sub.add_parser(
        "select", help="Group-based selection (first/last/random) with optional append"
    )
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.add_argument(
        "--sheet-existing",
        default=None,
        help="Existing output sheet for .xlsx when --append",
    )
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.add_argument(
        "--group", default="industry", help="Group column (default: industry)"
    )
    s.add_argument("--total", type=int, default=None, help="Total rows across groups")
    s.add_argument("--per-group", type=int, default=None, help="Rows per group")
    s.add_argument(
        "--min-per-group", type=int, default=0, help="Min per group when using --total"
    )
    s.add_argument("--max-per-group", type=int, default=None, help="Cap per group")
    s.add_argument(
        "--strategy",
        choices=["first", "last", "random"],
        default="random",
        help="Selection strategy per group",
    )
    s.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random strategy; use -1 to disable",
    )
    s.add_argument("--shuffle", action="store_true", help="Shuffle final output rows")
    s.add_argument(
        "--append", action="store_true", help="Append into existing output if it exists"
    )
    s.add_argument(
        "--dedup-key",
        default=None,
        help="Optional post-append dedup keys (comma-separated)",
    )
    s.add_argument(
        "--pre-dedup-key",
        default=None,
        help="Optional pre-selection dedup keys (comma-separated)",
    )
    s.add_argument(
        "--where",
        action="append",
        default=[],
        help="Condition col=value or col!=value (repeatable)",
    )
    s.add_argument(
        "--cols", default=None, help="Comma-separated columns to keep (output)"
    )
    s.add_argument("--rename", default=None, help="Rename mapping old:new (output)")
    s.add_argument(
        "--order", default=None, help="Comma-separated column order (output)"
    )
    s.set_defaults(func=cmd_select)

    # stats
    s = sub.add_parser("stats", help="Show dataset stats")
    s.add_argument("input")
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.add_argument(
        "--group", default=None, help="Optional group column for distribution"
    )
    s.set_defaults(func=cmd_stats)

    # dedup
    s = sub.add_parser("dedup", help="Deduplicate rows by key(s)")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.add_argument("--keys", required=True, help="Comma-separated keys")
    s.add_argument("--keep", choices=["first", "last"], default="first")
    s.set_defaults(func=cmd_dedup)

    # filter
    s = sub.add_parser("filter", help="Filter rows by conditions")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.add_argument(
        "--where",
        action="append",
        required=True,
        help="Condition col=value or col!=value (repeatable)",
    )
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.add_argument("--rename", default=None, help="Rename mapping old:new")
    s.add_argument("--order", default=None, help="Comma-separated column order")
    s.set_defaults(func=cmd_filter)

    # sort
    s = sub.add_parser("sort", help="Sort by columns (stable)")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.add_argument("--by", required=True, help="Comma-separated columns")
    s.add_argument("--desc", action="store_true")
    s.add_argument("--cols", default=None, help="Comma-separated columns to keep")
    s.add_argument("--rename", default=None, help="Rename mapping old:new")
    s.add_argument("--order", default=None, help="Comma-separated column order")
    s.set_defaults(func=cmd_sort)

    # schema
    s = sub.add_parser("schema", help="Print columns + dtypes")
    s.add_argument("input")
    s.add_argument("--sheet-in", default=None, help="Input sheet for .xlsx")
    s.set_defaults(func=cmd_schema)

    # join
    s = sub.add_parser("join", help="Join two datasets")
    s.add_argument("--left", required=True)
    s.add_argument("--right", required=True)
    s.add_argument("--sheet-left", default=None, help="Left sheet for .xlsx")
    s.add_argument("--sheet-right", default=None, help="Right sheet for .xlsx")
    s.add_argument("--on", required=True, help="Comma-separated join keys")
    s.add_argument("--how", choices=["inner", "left", "right", "outer"], default="left")
    s.add_argument("--output", required=True)
    s.add_argument("--sheet-out", default="Sheet1", help="Output sheet for .xlsx")
    s.set_defaults(func=cmd_join)

    # run pipeline
    s = sub.add_parser("run", help="Run an explicit JSON pipeline to chain commands")
    s.add_argument("pipeline", help="Path to pipeline JSON (see module docstring)")
    s.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
