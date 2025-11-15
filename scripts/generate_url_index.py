from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

from extensions.url_index import discover_and_write_url_index
from extensions.global_state import GlobalState
from extensions.connectivity_guard import ConnectivityGuard
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

    # Per-company crawler knobs
    p.add_argument("--max-pages", type=int, default=8000)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--per-host-cap", type=int, default=4000)
    p.add_argument("--dual-alpha", type=float, default=0.5)
    p.add_argument("--pos-query", default=None)
    p.add_argument("--neg-query", default=None)

    # Scoring thresholds
    p.add_argument("--score-threshold", type=float, default=0.25,
                   help="Drop discovered URLs whose dual-BM25 score is below this value. Set to a negative value to disable.")
    seeds_meg = p.add_mutually_exclusive_group()
    seeds_meg.add_argument("--score-threshold-on-seeds", dest="score_threshold_on_seeds", action="store_true", default=True,
                           help="Apply score threshold to seed roots (default).")
    seeds_meg.add_argument("--no-score-threshold-on-seeds", dest="score_threshold_on_seeds", action="store_false",
                           help="Do NOT apply score threshold to seed roots.")

    # Language / externals
    p.add_argument("--lang-primary", default="en")
    p.add_argument("--accept-en-regions", default=None, help="Comma list like 'us,gb,uk,ca,au'")
    p.add_argument("--strict-cctld", action="store_true")
    p.add_argument("--no-drop-universal", action="store_true", help="Do NOT drop universal externals")
    p.add_argument("--include", nargs="*", default=None, help="Override include patterns (space-separated)")
    p.add_argument("--exclude", nargs="*", default=None, help="Override exclude patterns (space-separated)")
    p.add_argument("--dynamic-counts-file", type=Path, default=None)

    # Concurrency controls
    p.add_argument("--company-concurrency", type=int, default=6,
                   help="Max number of companies processed in parallel.")
    p.add_argument("--crawl-concurrency", type=int, default=8,
                   help="Worker tasks per company (parallel page expansion).")

    # Progress tracking
    p.add_argument("--progress-file", type=Path, default=Path("outputs") / "url_index_progress.json",
                   help="Where to write rolling progress (JSON).")

    # Connectivity guard (shared, process-wide)
    p.add_argument("--probe-host", default="1.1.1.1", help="Guard probe host (HTTPS recommended).")
    p.add_argument("--probe-port", type=int, default=443, help="Guard probe port (use 443 to mirror crawler traffic).")
    p.add_argument("--probe-interval", type=float, default=15.0, help="Seconds between probe cycles.")
    p.add_argument("--probe-trip", type=int, default=3, help="Consecutive failed probes to open circuit.")
    p.add_argument("--probe-timeout", type=float, default=2.5, help="TCP connect timeout for probe.")
    p.add_argument("--probe-cooloff-base", type=float, default=5.0, help="Base cooloff seconds before half-open retry.")
    p.add_argument("--probe-max-cooloff", type=float, default=300.0, help="Max cooloff seconds.")
    p.add_argument("--probe-backoff", type=float, default=2.0, help="Exponential backoff factor.")
    p.add_argument("--probe-jitter", type=float, default=0.25, help="Random jitter added to cooloff.")

    # Logging
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p

# ---------- Small helper for safe writes ----------

def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

# ---------- Runner ----------

async def _run_for_company(ci: CompanyInput, args: argparse.Namespace, state: GlobalState, *, guard: ConnectivityGuard) -> Tuple[str, int]:
    """
    Returns: (bvdid, seeded_count)
    """
    logging.getLogger("generate_url_index").info("→ [%s] %s (%s)", ci.bvdid, ci.name, ci.url)
    try:
        res = await discover_and_write_url_index(
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
            concurrency=int(args.crawl_concurrency),
            guard=guard,  # shared single guard
        )
        return (ci.bvdid, int(res.get("seeded", 0)))
    except Exception as e:
        logging.getLogger("generate_url_index").exception("[%s] failed: %s", ci.bvdid, e)
        return (ci.bvdid, 0)

class _Progress:
    def __init__(self, total: int, progress_path: Path, logger: logging.Logger) -> None:
        self.total = max(0, int(total))
        self.done = 0
        self.seeded_total = 0
        self.completed_ids: List[str] = []
        self.path = progress_path
        self._lock = asyncio.Lock()
        self._log = logger

    async def mark_done(self, bvdid: str, seeded_count: int) -> None:
        async with self._lock:
            self.done += 1
            self.seeded_total += max(0, int(seeded_count))
            self.completed_ids.append(bvdid)
            payload = {
                "total_companies": self.total,
                "completed_companies": self.done,
                "completed_ratio": round((self.done / self.total) if self.total else 1.0, 4),
                "seeded_urls_total": self.seeded_total,
                # keep recent tail to avoid unbounded growth
                "recent_completed_ids": self.completed_ids[-500:],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write_json(self.path, payload)
            pct = (100.0 * self.done / self.total) if self.total else 100.0
            self._log.info("Progress: %d/%d (%.1f%%) | seeded_urls_total=%d",
                           self.done, self.total, pct, self.seeded_total)

async def main_async() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, (args.log_level or "INFO").upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    log = logging.getLogger("generate_url_index")

    # Load companies list
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

    log.info("Loaded %d companies. Begin discovery… (company_concurrency=%d, crawl_concurrency=%d)",
             len(companies), int(args.company_concurrency), int(args.crawl_concurrency))

    # Shared, process-wide connectivity guard
    guard = ConnectivityGuard(
        probe_host=args.probe_host,
        probe_port=int(args.probe_port),
        interval_s=float(args.probe_interval),
        trip_heartbeats=int(args.probe_trip),
        connect_timeout_s=float(args.probe_timeout),
        base_cooloff_s=float(args.probe_cooloff_base),
        max_cooloff_s=float(args.probe_max_cooloff),
        backoff_factor=float(args.probe_backoff),
        jitter_s=float(args.probe_jitter),
        logger=log,
    )
    await guard.start()
    # If starting while offline, pause once up-front to avoid a burst of failing tasks.
    await guard.wait_until_healthy()
    log.info("ConnectivityGuard started (probe=%s:%s, interval=%.2fs, trip=%d).",
             args.probe_host, int(args.probe_port), float(args.probe_interval), int(args.probe_trip))

    state = GlobalState()
    progress = _Progress(total=len(companies), progress_path=Path(args.progress_file), logger=log)

    sem = asyncio.Semaphore(max(1, int(args.company_concurrency)))

    async def _guarded(ci: CompanyInput):
        async with sem:
            bvdid, seeded = await _run_for_company(ci, args, state, guard=guard)
            await progress.mark_done(bvdid, seeded)

    try:
        tasks = [asyncio.create_task(_guarded(ci)) for ci in companies]
        await asyncio.gather(*tasks)
    finally:
        # Stop the shared guard once all companies are done
        await guard.stop()

    log.info("All done. Final progress written to: %s", args.progress_file)

if __name__ == "__main__":
    asyncio.run(main_async())