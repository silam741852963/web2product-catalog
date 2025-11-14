# generate_url_index.py
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Set

from extensions.url_index import discover_and_write_url_index
from extensions.global_state import GlobalState
from components.source_loader import (
    CompanyInput,
    load_companies_from_source,
    load_companies_from_dir,
)

# ---------- CLI parsing ----------

def _parse_accept_regions(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return set(parts) if parts else None

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_url_index",
        description="Discover links (link-only) per company and write outputs/*/checkpoints/url_index.json (seeded entries)."
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=Path, help="Input file (.csv, .tsv, .xlsx, .xls, .json, .jsonl, .ndjson, .parquet, .feather, .dta, .sas7bdat, .sav)")
    src.add_argument("--source-dir", type=Path, help="Directory of supported input files (scanned recursively)")
    p.add_argument("--source-pattern", default="*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav")

    p.add_argument("--limit", type=int, default=None, help="Max companies to process")

    p.add_argument("--max-pages", type=int, default=8000)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--per-host-cap", type=int, default=4000)
    p.add_argument("--dual-alpha", type=float, default=0.5)
    p.add_argument("--pos-query", default=None)
    p.add_argument("--neg-query", default=None)

    # NEW: scoring thresholds
    p.add_argument("--score-threshold", type=float, default=0.25,
                   help="Drop discovered URLs whose dual-BM25 score is below this value. Set to a negative value to disable.")
    seeds_meg = p.add_mutually_exclusive_group()
    seeds_meg.add_argument("--score-threshold-on-seeds", dest="score_threshold_on_seeds", action="store_true", default=True,
                           help="Apply score threshold to seed roots (default).")
    seeds_meg.add_argument("--no-score-threshold-on-seeds", dest="score_threshold_on_seeds", action="store_false",
                           help="Do NOT apply score threshold to seed roots.")

    p.add_argument("--lang-primary", default="en")
    p.add_argument("--accept-en-regions", default=None, help="Comma list like 'us,gb,uk,ca,au'")
    p.add_argument("--strict-cctld", action="store_true")
    # Keep legacy switch; name kept for backward compat
    p.add_argument("--no-drop-universal", action="store_true", help="Do NOT drop universal externals")
    p.add_argument("--include", nargs="*", default=None, help="Override include patterns (space-separated)")
    p.add_argument("--exclude", nargs="*", default=None, help="Override exclude patterns (space-separated)")
    p.add_argument("--dynamic-counts-file", type=Path, default=None)

    # Logging
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p

# ---------- Runner ----------

async def _run_for_company(ci: CompanyInput, args: argparse.Namespace, state: GlobalState) -> None:
    logging.getLogger("generate_url_index").info("→ [%s] %s (%s)", ci.bvdid, ci.name, ci.url)
    try:
        await discover_and_write_url_index(
            company_id=ci.bvdid,
            company_name=ci.name,
            base_url=ci.url,
            include=args.include if args.include else None,
            exclude=args.exclude if args.exclude else None,
            lang_primary=args.lang_primary or "en",
            accept_en_regions=_parse_accept_regions(args.accept_en_regions),
            strict_cctld=bool(args.strict_cctld),
            drop_universal_externals=not bool(args.no_drop_universal),
            max_pages=int(args.max_pages),
            max_depth=int(args.max_depth),
            per_host_page_cap=int(args.per_host_cap),
            dual_alpha=float(args.dual_alpha),
            pos_query=args.pos_query,
            neg_query=args.neg_query,
            score_threshold=args.score_threshold,
            score_threshold_on_seeds=args.score_threshold_on_seeds,
            dynamic_counts_file=args.dynamic_counts_file,
            state=state,
        )
    except Exception as e:
        logging.getLogger("generate_url_index").exception("[%s] failed: %s", ci.bvdid, e)

async def main_async() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, (args.log_level or "INFO").upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    log = logging.getLogger("generate_url_index")

    # Load companies list via source_loader (I/O aligned with run_crawl)
    if getattr(args, "source_dir", None):
        patterns = [p.strip() for p in (args.source_pattern or "").split(",") if p.strip()]
        companies: List[CompanyInput] = load_companies_from_dir(args.source_dir, patterns=patterns, recursive=True)
    else:
        companies = load_companies_from_source(args.source)

    if args.limit is not None and args.limit > 0:
        companies = companies[: args.limit]

    if not companies:
        log.error("No companies to process.")
        return

    log.info("Loaded %d companies. Begin discovery…", len(companies))

    state = GlobalState()  # records has_url_index + resume_mode for reuse_index

    # Sequential is OK (per-company discovery is light); can be parallelized if needed
    for ci in companies:
        await _run_for_company(ci, args, state)

    log.info("All done.")

if __name__ == "__main__":
    asyncio.run(main_async())