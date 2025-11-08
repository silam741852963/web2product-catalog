from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from extensions.log_scan import (
    ascan_logs,
    create_test_csv,
    dump_hits_json,
    dump_hits_csv,
)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async log scanner: scan outputs/{bvdid}/logs/*.log for ERROR/WARNING blocks; print and optionally export."
    )
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"),
                   help="Root directory containing per-company folders (default: ./outputs).")
    p.add_argument("--logs-glob", type=str, default="*.log",
                   help="Glob pattern for log files relative to each logs/ dir (default: *.log). Use *.log* to include .log.gz.")
    p.add_argument("--encoding", type=str, default="utf-8",
                   help="File encoding for reading log files (default utf-8).")
    p.add_argument("--no-prefilter", dest="use_prefilter", action="store_false",
                   help="Disable prefilter token scan (slower, but may find more).")

    p.add_argument("--level", choices=["error", "warning", "all"], default="error",
                   help="Which severity to match (default: error).")
    p.add_argument("--pattern", action="append", default=[],
                   help="Extra regex pattern(s) to match (can be repeated).")

    p.add_argument("--context-before", type=int, default=0,
                   help="Lines before a hit to include (default: 0).")
    p.add_argument("--context-after", type=int, default=20,
                   help="Lines after a hit to include (default: 20).")

    p.add_argument("--print", dest="do_print", action="store_true",
                   help="Print hits to stdout.")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of printed hits (0 = no limit).")

    p.add_argument("--json-out", type=Path, default=None,
                   help="Write all hits as a JSON file (optional).")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="Write all hits as a CSV file (optional).")

    p.add_argument("--write-test-csv", type=Path, default=None,
                   help="Write a CSV of matched companies (BVDIDs) to this path.")
    p.add_argument("--source-csv", type=Path, default=None,
                   help="Original company CSV to enrich/normalize BVDIDs in the test CSV.")

    p.add_argument("--workers", type=int, default=64,
                   help="Max concurrent file scans (default: 64).")
    return p.parse_args()

def _print_hits(all_hits: List, limit: int = 0) -> None:
    count = 0
    for h in all_hits:
        if limit and count >= limit:
            break
        print("=" * 80)
        label = h.bvdid if h.bvdid != "UNKNOWN" else h.sanitized_bvdid
        print(f"{h.level} | {label} | {h.log_path}:{h.line_no}")
        print(h.message)
        if h.context_before:
            print("-- before --")
            for ln in h.context_before:
                print(ln)
        if h.context_after:
            print("-- after --")
            for ln in h.context_after:
                print(ln)
        count += 1
    if limit and len(all_hits) > limit:
        print(f"... ({len(all_hits) - limit} more not shown)")

async def _run_async(args: argparse.Namespace) -> None:
    hits_by_company, all_hits = await ascan_logs(
        outputs_root=args.outputs_root,
        logs_glob=args.logs_glob,
        level=args.level,
        patterns=args.pattern,
        context_before=args.context_before,
        context_after=args.context_after,
        encoding=args.encoding,
        workers=args.workers,
        use_prefilter=args.use_prefilter,
    )

    # Build company list, prefer original id if known
    companies = []
    seen = set()
    for lst in hits_by_company.values():
        for h in lst:
            key = h.bvdid if h.bvdid != "UNKNOWN" else h.sanitized_bvdid
            if key and key not in seen:
                companies.append(key)
                seen.add(key)

    print(f"Found {len(all_hits)} hit(s) across {len(companies)} compan(ies).")

    if args.do_print:
        _print_hits(all_hits, limit=args.limit)

    if args.json_out:
        dump_hits_json(all_hits, args.json_out)
        print(f"Wrote JSON: {args.json_out}")

    if args.csv_out:
        dump_hits_csv(all_hits, args.csv_out)
        print(f"Wrote CSV: {args.csv_out}")

    if args.write_test_csv:
        n = create_test_csv(
            out_csv=args.write_test_csv,
            companies=companies,
            source_csv=args.source_csv,
        )
        print(f"Wrote test CSV with {n} compan(ies): {args.write_test_csv}")

def main() -> None:
    args = _parse_args()
    asyncio.run(_run_async(args))

if __name__ == "__main__":
    main()