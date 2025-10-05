#!/usr/bin/env python3
"""
Run the US batch scraper with true intra-company resume.

Inputs: data/input/us/*.csv (rows: hojin_id,company_name,url,hightechflag,us_flag)

For each row (company):
  - Load (or create) a per-company state file in data/checkpoints/companies/
  - If state.done is True, skip
  - Else crawl in PASSES using current state.pending as seeds
  - After each seed:
        * save HTML → Markdown (idempotent)
        * add newly discovered same-origin links to next frontier if not visited
        * persist state
  - When pending is empty (no new URLs), mark state.done = True, and also mark row done in the per-CSV checkpoint.

Resilience:
  - Per-CSV checkpoint (skips fully-done companies)
  - Per-company state (visited/pending frontier) saved after every seed
  - Bounded concurrency (companies & seeds)
  - Retries for transient failures remain in SiteCrawler + company attempts
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scraper.browser import init_browser, shutdown_browser
from scraper.crawler import SiteCrawler
from scraper.config import load_config
from scraper.utils import prune_html_for_markdown, html_to_markdown, clean_markdown, save_markdown, TransientHTTPError


# ---------------------------- Data Models ----------------------------

@dataclass(frozen=True)
class CompanyRow:
    hojin_id: str
    company_name: str
    url: str
    hightechflag: str
    us_flag: str

    @property
    def key(self) -> str:
        return f"{self.hojin_id}:{self.url.strip()}"


# ----------------------------- IO helpers ----------------------------

def _iter_input_files(base: Path, pattern: str = "*.csv") -> list[Path]:
    return sorted((base / "input" / "us").glob(pattern))


def _read_rows(csv_path: Path) -> Iterable[CompanyRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield CompanyRow(
                hojin_id=str(row.get("hojin_id", "")).strip(),
                company_name=str(row.get("company_name", "")).strip(),
                url=str(row.get("url", "")).strip(),
                hightechflag=str(row.get("hightechflag", "")).strip(),
                us_flag=str(row.get("us_flag", "")).strip(),
            )


def _csv_checkpoint_path(checkpoints_dir: Path, csv_input: Path) -> Path:
    return checkpoints_dir / f"{csv_input.stem}.json"


def _load_csv_checkpoint(checkpoints_dir: Path, csv_input: Path) -> set[str]:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _csv_checkpoint_path(checkpoints_dir, csv_input)
    if ckpt.exists():
        try:
            return set(json.loads(ckpt.read_text(encoding="utf-8")).get("done", []))
        except Exception:
            return set()
    return set()


def _save_csv_checkpoint(checkpoints_dir: Path, csv_input: Path, done_keys: set[str]) -> None:
    ckpt = _csv_checkpoint_path(checkpoints_dir, csv_input)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    body = {"done": sorted(done_keys)}
    tmp = ckpt.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ckpt)


def _company_state_path(checkpoints_dir: Path, row: CompanyRow) -> Path:
    # Stable filename per (hojin_id, host)
    host = (urlparse(row.url).hostname or "unknown-host").lower()
    if host.startswith("www."):
        host = host[4:]
    h = hashlib.sha1(row.url.strip().encode("utf-8")).hexdigest()[:10]
    return checkpoints_dir / "companies" / f"{row.hojin_id}_{host}_{h}.json"


def _load_company_state(path: Path, row: CompanyRow) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "hojin_id": row.hojin_id,
        "company": row.company_name,
        "homepage": row.url,
        "visited": [],
        "pending": [row.url] if row.url else [],
        "done": False,
    }


def _save_company_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------- Core routines ----------------------------

async def _crawl_seed_pass(
    cfg,
    crawler: SiteCrawler,
    seed_url: str,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    max_pages_per_seed: Optional[int],
) -> list:
    """
    Crawl starting from a single seed URL. Returns PageSnapshot list.
    We rely on SiteCrawler's internal queue; setting a cap helps avoid
    runaway expansion per seed.
    """
    snaps = await crawler.crawl_site(
        homepage=seed_url,
        max_pages=max_pages_per_seed,
        url_allow_regex=allow_regex,
        url_deny_regex=deny_regex,
    )
    return snaps


def _save_markdown_for_snaps(cfg, snaps: list) -> None:
    for snap in snaps:
        if not snap.html_path:
            continue
        p = Path(snap.html_path)
        if not p.exists():
            continue
        html = p.read_text(encoding="utf-8", errors="ignore")
        cleaned_html = prune_html_for_markdown(html)
        md = clean_markdown(html_to_markdown(cleaned_html))
        parsed = urlparse(snap.url)
        host = parsed.hostname or "unknown-host"
        url_path = parsed.path or "/"
        save_markdown(cfg.markdown_dir, host, url_path, snap.url, md)


def _update_state_from_snaps(state: dict, snaps: list, homepage_host: str) -> None:
    visited = set(state.get("visited", []))
    pending = set(state.get("pending", []))

    # Mark pages we just fetched as visited and remove from pending
    for s in snaps:
        if s.url not in visited:
            visited.add(s.url)
        if s.url in pending:
            pending.discard(s.url)

    # Add new same-origin links to pending (exclude ones we've visited)
    for s in snaps:
        for link in s.out_links:
            if urlparse(link).hostname == homepage_host and link not in visited:
                pending.add(link)

    state["visited"] = sorted(visited)
    state["pending"] = sorted(pending)


async def _drain_company(
    cfg,
    crawler: SiteCrawler,
    state_path: Path,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    max_pages_per_seed: Optional[int],
    seed_batch: int,
) -> bool:
    """
    Run iterative passes over the company's frontier until empty or no progress.
    Returns True if finished (pending empty), else False.
    """
    state = _load_company_state(state_path, row=None)  # row not needed on reload
    homepage_host = (urlparse(state["homepage"]).hostname or "").lower()
    if homepage_host.startswith("www."):
        homepage_host = homepage_host[4:]

    progress = False

    while state["pending"]:
        # Take a batch of seeds to parallelize a bit
        seeds = state["pending"][:seed_batch]

        # Schedule seeds with a small concurrency (not to overload same domain)
        # We reuse crawler.sem for per-domain limits; just await all here.
        tasks = [asyncio.create_task(_crawl_seed_pass(
            cfg, crawler, u, allow_regex=allow_regex, deny_regex=deny_regex,
            max_pages_per_seed=max_pages_per_seed
        )) for u in seeds]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process each seed's results and persist after each to be robust
        for seed, res in zip(seeds, results):
            # Remove seed from pending up front (will be re-added if discovered again)
            try:
                state["pending"].remove(seed)
            except ValueError:
                pass

            if isinstance(res, Exception):
                # transient errors were retried inside; on hard fail we can re-queue seed
                # for next attempt by leaving it out (or add back). We add back once.
                state["pending"].append(seed)
                _save_company_state(state_path, state)
                continue

            snaps = res or []
            prev_visited_len = len(state.get("visited", []))
            _save_markdown_for_snaps(cfg, snaps)
            _update_state_from_snaps(state, snaps, homepage_host)
            _save_company_state(state_path, state)

            if len(state["visited"]) > prev_visited_len:
                progress = True

        # If no progress in this loop, break to avoid spinning
        if not progress:
            break

        progress = False  # reset for next loop

    finished = len(state["pending"]) == 0
    state["done"] = bool(finished)
    _save_company_state(state_path, state)
    return finished


async def _process_company(
    cfg,
    context,
    row: CompanyRow,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    # true resume controls
    max_pages_per_seed: Optional[int],
    seed_batch: int,
    company_attempts: int,
) -> bool:
    """
    True intra-company resume:
      - Load company state (visited/pending)
      - Iterate seeds in batches; persist after each seed
      - Return True if finished (no pending), False otherwise
    """
    state_path = _company_state_path(cfg.checkpoints_dir, row)
    state = _load_company_state(state_path, row)
    if state.get("done"):
        return True

    crawler = SiteCrawler(cfg, context)

    for attempt in range(1, max(1, company_attempts) + 1):
        try:
            finished = await _drain_company(
                cfg, crawler, state_path,
                allow_regex=allow_regex,
                deny_regex=deny_regex,
                max_pages_per_seed=max_pages_per_seed,
                seed_batch=seed_batch,
            )
            return finished
        except TransientHTTPError:
            # let outer attempt retry
            await asyncio.sleep(0.8 * attempt)
        except Exception:
            # hard failure – stop; not marking done
            break

    return False


async def _process_csv(
    cfg,
    csv_path: Path,
    *,
    context,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    resume: bool,
    limit: Optional[int],
    max_pages_per_seed: Optional[int],
    seed_batch: int,
    company_attempts: int,
) -> None:
    done = _load_csv_checkpoint(cfg.checkpoints_dir, csv_path)
    rows = list(_read_rows(csv_path))
    if limit:
        rows = rows[:limit]

    sem = asyncio.Semaphore(cfg.max_companies_parallel)
    tasks: list[asyncio.Task] = []

    async def worker(row: CompanyRow):
        # If per-CSV checkpoint says it's done, skip entirely
        if resume and row.key in done:
            return

        async with sem:
            finished = await _process_company(
                cfg,
                context,
                row,
                allow_regex=allow_regex,
                deny_regex=deny_regex,
                max_pages_per_seed=max_pages_per_seed,
                seed_batch=seed_batch,
                company_attempts=max(2, company_attempts),
            )
            # Only mark done in CSV checkpoint when company state says done
            if finished:
                done.add(row.key)
                _save_csv_checkpoint(cfg.checkpoints_dir, csv_path, done)

    for row in rows:
        if not row.url:
            continue
        if resume and row.key in done:
            continue
        tasks.append(asyncio.create_task(worker(row)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _parse_args(argv: list[str]):
    import argparse

    p = argparse.ArgumentParser(description="Run the US batch scraper with true intra-company resume.")
    p.add_argument("--pattern", default="*.csv", help="Glob for input files under data/input/us/ (default: *.csv)")
    p.add_argument("--limit", type=int, default=None, help="Limit companies per CSV (debugging)")
    # Resume controls
    p.add_argument("--max-pages-per-seed", type=int, default=40, help="Cap pages fetched per seed pass (default: 40)")
    p.add_argument("--seed-batch", type=int, default=6, help="How many seeds to crawl concurrently per company (default: 6)")
    p.add_argument("--allow", type=str, default=None, help="Allow regex for URLs (e.g. /products|/solutions)")
    p.add_argument("--deny", type=str, default=None, help="Deny regex for URLs (e.g. /blog|/careers)")
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoints and start fresh")
    p.add_argument("--company-attempts", type=int, default=3, help="Attempts to recover a company on transient errors (default: 3)")
    return p.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    cfg = load_config()

    files = _iter_input_files(cfg.data_dir, pattern=args.pattern)
    if not files:
        print(f"No input files matched in {cfg.data_dir / 'input' / 'us'} with pattern {args.pattern}")
        return

    pw, browser, context = await init_browser(cfg)
    try:
        for csv_path in files:
            await _process_csv(
                cfg,
                csv_path,
                context=context,
                allow_regex=args.allow,
                deny_regex=args.deny,
                resume=not args.no_resume,
                limit=args.limit,
                max_pages_per_seed=args.max_pages_per_seed,
                seed_batch=args.seed_batch,
                company_attempts=args.company_attempts,
            )
    finally:
        await shutdown_browser(pw, browser, context)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()