#!/usr/bin/env python3
"""
Run the batch scraper with true intra-company resume.

Inputs:
  - Preferred: CSV shards via --input-glob (default from cfg.input_glob, e.g. "data/input/us/*.csv")
  - Fallback: single file cfg.input_urls_csv

CSV schema:
  hojin_id,company_name,url,hightechflag,us_flag

Resume:
  - Per-CSV checkpoint: data/checkpoints/<csv-name>.json {"done": [ "<hojin_id>:<url>", ... ]}
  - Per-company state:  data/checkpoints/companies/<hojin_id>_<host>_<hash>.json
      { visited: [...], pending: [...], done: bool, homepage: url, ... }

Robustness:
  - Seed/frontier persists after every processed seed
  - Retries within SiteCrawler; company-level attempts as a second guard
  - Bounded concurrency (companies)
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scraper.browser import init_browser, shutdown_browser
from scraper.crawler import SiteCrawler, NonRetryableHTTPError
from scraper.config import load_config
from scraper.utils import (
    prune_html_for_markdown, html_to_markdown, clean_markdown, save_markdown,
    TransientHTTPError, is_http_url, normalize_url, same_site, get_base_domain,
    looks_non_product_url, is_meaningful_markdown
)

log = logging.getLogger("scripts.run_scraper")


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

def _discover_input_files(*, arg_glob: Optional[str], cfg) -> list[Path]:
    """Resolve input CSVs via precedence: CLI glob -> cfg.input_glob -> cfg.input_urls_csv (single file)."""
    candidates: list[str] = []
    if arg_glob:
        candidates = glob(arg_glob)
        log.info("Using --input-glob=%s → %d file(s) found", arg_glob, len(candidates))
    else:
        if getattr(cfg, "input_glob", None):
            candidates = glob(cfg.input_glob)
            log.info("Using cfg.input_glob=%s → %d file(s) found", cfg.input_glob, len(candidates))

    files: list[Path] = [Path(p) for p in candidates if Path(p).exists()]
    if not files and getattr(cfg, "input_urls_csv", None) and Path(cfg.input_urls_csv).exists():
        files = [Path(cfg.input_urls_csv)]
        log.info("Falling back to single file: %s", cfg.input_urls_csv)

    return sorted(files)


def _read_rows(csv_path: Path) -> Iterable[CompanyRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_url = (row.get("url") or "").strip()
            url = normalize_url(raw_url) if raw_url else ""
            yield CompanyRow(
                hojin_id=str(row.get("hojin_id", "")).strip(),
                company_name=str(row.get("company_name", "")).strip(),
                url=url,
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
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            return set(data.get("done", []))
        except Exception:
            log.warning("CSV checkpoint %s was unreadable; starting fresh.", ckpt)
    return set()


def _save_csv_checkpoint(checkpoints_dir: Path, csv_input: Path, done_keys: set[str]) -> None:
    ckpt = _csv_checkpoint_path(checkpoints_dir, csv_input)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    body = {"done": sorted(done_keys)}
    tmp = ckpt.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ckpt)


def _company_state_path(checkpoints_dir: Path, row: CompanyRow) -> Path:
    host = (urlparse(row.url).hostname or "unknown-host").lower()
    if host.startswith("www."):
        host = host[4:]
    h = hashlib.sha1(row.url.strip().encode("utf-8")).hexdigest()[:10]
    return checkpoints_dir / "companies" / f"{row.hojin_id}_{host}_{h}.json"


def _load_company_state(path: Path, row: CompanyRow | None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Company state %s unreadable; recreating.", path)
    if row is None:
        return {"hojin_id": "", "company": "", "homepage": "", "visited": [], "pending": [], "done": False}
    homepage = normalize_url(row.url) if row.url else ""
    pending = [homepage] if homepage else []
    return {
        "hojin_id": row.hojin_id,
        "company": row.company_name,
        "homepage": homepage,
        "visited": [],
        "pending": pending,
        "done": False,
    }


def _save_company_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------- URL helpers ----------------------------

def _norm(u: str) -> str:
    try:
        return normalize_url(u)
    except Exception:
        return u or ""


def _pop_pending_equiv(state: dict, u: str) -> None:
    """Remove any pending entry equivalent to u once normalized."""
    nu = _norm(u)
    pending = state.get("pending", [])
    keep = []
    removed = False
    for p in pending:
        if _norm(p) == nu:
            removed = True
            continue
        keep.append(p)
    if removed:
        state["pending"] = keep


# ---------------------------- Core routines ----------------------------

async def _crawl_seed_pass(
    cfg,
    crawler: SiteCrawler,
    seed_url: str,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    max_pages_for_this_seed: Optional[int],
) -> list:
    snaps = await crawler.crawl_site(
        homepage=seed_url,
        max_pages=max_pages_for_this_seed if max_pages_for_this_seed is not None else getattr(cfg, "max_pages_per_company", None),
        url_allow_regex=allow_regex,
        url_deny_regex=deny_regex,
    )
    return snaps


def _render_markdown_from_html(cfg, url: str, html: str) -> str:
    # Small helper to keep logic in one place
    try:
        cleaned_html = prune_html_for_markdown(html)
    except Exception as e:
        log.warning("prune_html_for_markdown failed for %s: %s", url, e)
        cleaned_html = html
    try:
        md = clean_markdown(html_to_markdown(cleaned_html))
    except Exception as e:
        log.warning("HTML→Markdown failed for %s: %s", url, e)
        md = cleaned_html
    return md


async def _save_markdown_for_snaps(cfg, crawler: SiteCrawler, snaps: list, *, retried_dynamic: set[str]) -> None:
    """
    Save Markdown for each snapshot; if the markdown looks meaningless,
    try a one-time dynamic re-fetch and overwrite the artifacts.
    """
    for snap in snaps:
        if not getattr(snap, "html_path", None):
            continue
        p = Path(snap.html_path)
        if not p.exists():
            continue

        # 1) initial render from cached HTML
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        md = _render_markdown_from_html(cfg, snap.url, html)

        # 2) quality gate: if low quality and static-first is enabled, try once with Playwright
        want_dynamic_retry = (
            bool(getattr(cfg, "enable_static_first", False)) and
            snap.url not in retried_dynamic and
            not is_meaningful_markdown(md,
                                       min_chars=int(getattr(cfg, "md_min_chars", 400)),
                                       min_words=int(getattr(cfg, "md_min_words", 80)),
                                       min_uniq_words=int(getattr(cfg, "md_min_uniq", 50)),
                                       min_lines=int(getattr(cfg, "md_min_lines", 8)))
        )

        if want_dynamic_retry:
            try:
                dyn_snap = await crawler.fetch_dynamic_only(snap.url)
                # refresh local variables based on dynamic fetch
                if dyn_snap.html_path and Path(dyn_snap.html_path).exists():
                    html = Path(dyn_snap.html_path).read_text(encoding="utf-8", errors="ignore")
                    md = _render_markdown_from_html(cfg, snap.url, html)
                    retried_dynamic.add(snap.url)
                    # also, replace the outgoing links snapshot so the frontier can grow on JS pages
                    snap.out_links = dyn_snap.out_links
                    snap.html_path = dyn_snap.html_path
                    snap.title = dyn_snap.title
                    log.debug("Dynamic retry improved markdown for %s (size=%d chars).", snap.url, len(md))
            except NonRetryableHTTPError:
                log.debug("Dynamic retry 404 for %s, skipping.", snap.url)
            except TransientHTTPError as e:
                log.debug("Dynamic retry transient for %s: %s", snap.url, e)
            except Exception as e:
                log.debug("Dynamic retry failed for %s: %s", snap.url, e)

        # 3) save markdown
        parsed = urlparse(snap.url)
        host = (parsed.hostname or "unknown-host")
        url_path = parsed.path or "/"
        try:
            save_markdown(cfg.markdown_dir, host, url_path, snap.url, md)
        except Exception as e:
            log.warning("save_markdown failed for %s: %s", snap.url, e)


def _update_state_from_snaps(state: dict, snaps: list, homepage_url: str, allow_subdomains: bool) -> None:
    visited = set(_norm(u) for u in state.get("visited", []) if u)
    pending = set(_norm(u) for u in state.get("pending", []) if u)
    home_norm = _norm(homepage_url)

    for s in snaps:
        su = _norm(s.url)
        if su:
            visited.add(su)
            if su in pending:
                pending.discard(su)
            else:
                _pop_pending_equiv(state, s.url)

    for s in snaps:
        for link in getattr(s, "out_links", []) or []:
            lu = _norm(link)
            if not lu:
                continue
            if lu in visited or lu in pending:
                continue
            if looks_non_product_url(lu):  # extra belt-and-suspenders
                continue
            if same_site(home_norm, lu, allow_subdomains):
                pending.add(lu)

    state["visited"] = sorted(visited)
    state["pending"] = sorted(pending)


async def _drain_company(
    cfg,
    crawler: SiteCrawler,
    state_path: Path,
    state: dict,
    *,
    allow_regex: Optional[str],
    deny_regex: Optional[str],
    max_pages_per_seed: Optional[int],
    seed_batch: int,
) -> bool:
    homepage = _norm(state.get("homepage") or "")
    allow_subdomains = bool(getattr(cfg, "allow_subdomains", False))

    cap_total = int(getattr(cfg, "max_pages_per_company", 300) or 300)
    already = len({ _norm(u) for u in state.get("visited", []) if u })
    remaining = max(0, cap_total - already)

    if remaining == 0:
        state["done"] = True
        _save_company_state(state_path, state)
        log.debug("Company cap already reached (%d). Marking done.", cap_total)
        return True

    progress = False
    # share this across the whole company so we don’t repeatedly dynamic-retry the same URL
    retried_dynamic: set[str] = set()

    while state["pending"] and remaining > 0:
        pending_list = sorted({ _norm(u) for u in state["pending"] if u })
        state["pending"] = pending_list
        seeds = pending_list[:seed_batch]
        if not seeds:
            break

        # fair-share budget per seed this pass
        denom = max(1, len(seeds))
        share = max(1, remaining // denom)
        if max_pages_per_seed is not None:
            share = min(share, max_pages_per_seed)

        log.debug(
            "Draining %s: remaining=%d cap=%d batch=%d share/seed=%d (visited=%d)",
            state.get("company", ""), remaining, cap_total, len(seeds), share, already
        )

        tasks = [
            asyncio.create_task(
                _crawl_seed_pass(
                    cfg, crawler, u,
                    allow_regex=allow_regex,
                    deny_regex=deny_regex,
                    max_pages_for_this_seed=share,
                )
            )
            for u in seeds
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # remove attempted seeds from pending (normalized)
        current_pending = { _norm(u) for u in state.get("pending", []) if u }
        for s_url in seeds:
            current_pending.discard(s_url)
        state["pending"] = sorted(current_pending)

        for seed, res in zip(seeds, results):
            if isinstance(res, Exception):
                # keep failed seed for another pass
                if seed not in state["pending"]:
                    state["pending"].append(seed)
                _save_company_state(state_path, state)
                log.debug("Seed failed, re-queued: %s", seed)
                continue

            snaps = res or []
            prev_visited = len(state.get("visited", []))

            # IMPORTANT: _save_markdown_for_snaps is async and may dynamic-refetch via Playwright
            await _save_markdown_for_snaps(cfg, crawler, snaps, retried_dynamic=retried_dynamic)

            # update state & persist
            _update_state_from_snaps(
                state, snaps,
                homepage_url=homepage,
                allow_subdomains=allow_subdomains
            )
            _save_company_state(state_path, state)

            now_visited = len(state.get("visited", []))
            gained = max(0, now_visited - prev_visited)
            remaining = max(0, remaining - gained)
            already += gained

            if gained > 0:
                progress = True
                log.debug(
                    "Progress: +%d pages; visited=%d pending=%d remaining=%d",
                    gained, now_visited, len(state["pending"]), remaining
                )

        if remaining == 0:
            log.info("Company cap reached (%d). Stopping company.", cap_total)
            break

        if not progress:
            log.debug("No progress this pass for %s, stopping.", state.get("company", ""))
            break
        progress = False

    finished = remaining == 0 or len(state["pending"]) == 0
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
    max_pages_per_seed: Optional[int],
    seed_batch: int,
    company_attempts: int,
) -> bool:
    state_path = _company_state_path(cfg.checkpoints_dir, row)
    state = _load_company_state(state_path, row)
    if state.get("done"):
        return True

    _save_company_state(state_path, state)
    log.debug("Company %s: starting with pending=%d visited=%d",
              row.company_name, len(state["pending"]), len(state["visited"]))

    crawler = SiteCrawler(cfg, context)

    for attempt in range(1, max(1, company_attempts) + 1):
        try:
            finished = await _drain_company(
                cfg, crawler, state_path, state,
                allow_regex=allow_regex,
                deny_regex=deny_regex,
                max_pages_per_seed=max_pages_per_seed,
                seed_batch=seed_batch,
            )
            return finished
        except TransientHTTPError as e:
            log.warning("Transient company-level error (%s) on %s attempt %d; retrying...",
                        e, row.url, attempt)
            await asyncio.sleep(0.8 * attempt)
        except Exception:
            log.exception("Hard failure for company %s (%s); will not mark done.", row.company_name, row.url)
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
    rows_all = list(_read_rows(csv_path))

    rows_all = [
        r for r in rows_all
        if r.url and is_http_url(r.url) and (urlparse(r.url).hostname or "").strip()
    ]
    if limit:
        rows_all = rows_all[:limit]

    log.info("CSV %s: loaded %d rows", csv_path.name, len(rows_all))
    if resume:
        rows = [r for r in rows_all if r.key not in done]
        log.info("CSV %s: %d already complete (skipped), %d to process",
                 csv_path.name, len(rows_all) - len(rows), len(rows))
    else:
        rows = rows_all
        log.info("CSV %s: resume disabled; processing %d rows", csv_path.name, len(rows))

    if not rows:
        return

    sem = asyncio.Semaphore(cfg.max_companies_parallel)
    tasks: list[asyncio.Task] = []

    async def worker(row: CompanyRow):
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
            if finished:
                done.add(row.key)
                _save_csv_checkpoint(cfg.checkpoints_dir, csv_path, done)

    for row in rows:
        tasks.append(asyncio.create_task(worker(row)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _parse_args(argv: list[str]):
    import argparse

    p = argparse.ArgumentParser(description="Run the batch scraper with true intra-company resume.")
    p.add_argument("--input-glob", default=None,
                   help='Glob for input CSVs (default: cfg.input_glob, e.g. "data/input/us/*.csv")')
    p.add_argument("--pattern", default=None,
                   help="Deprecated: use --input-glob instead.")
    p.add_argument("--limit", type=int, default=None, help="Limit companies per CSV (debugging)")
    # Resume controls
    p.add_argument("--max-pages-per-seed", type=int, default=40,
                   help="Cap pages fetched per seed pass (default: 40); company-wide cap enforced too")
    p.add_argument("--seed-batch", type=int, default=6,
                   help="How many seeds to crawl concurrently per company (default: 6)")
    p.add_argument("--allow", type=str, default=None, help="Allow regex for URLs (e.g. /products|/solutions)")
    p.add_argument("--deny", type=str, default=None, help="Deny regex for URLs (e.g. /blog|/careers)")
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoints and start fresh")
    p.add_argument("--company-attempts", type=int, default=3, help="Attempts to recover a company on transient errors (default: 3)")
    p.add_argument("--debug", action="store_true", help="Set root logging level to DEBUG for this run")
    return p.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    cfg = load_config()

    if args.debug:
        logging.getLogger().setLevel("DEBUG")
        log.debug("Debug logging enabled")

    input_glob = args.input_glob or args.pattern or getattr(cfg, "input_glob", None)

    files = _discover_input_files(arg_glob=input_glob, cfg=cfg)
    if not files:
        log.error(
            "No input CSVs found. Checked --input-glob (%s), cfg.input_glob (%s), "
            "and fallback single file %s",
            args.input_glob or args.pattern, getattr(cfg, "input_glob", None), getattr(cfg, "input_urls_csv", None),
        )
        print('No input files found. Use --input-glob "data/input/us/*.csv" or set INPUT_GLOB env var.')
        return

    log.info("Discovered %d input file(s). Example: %s", len(files), files[0])

    total_rows = 0
    for pth in files[:5]:
        try:
            total_rows += sum(1 for _ in _read_rows(pth))
        except Exception:
            pass
    log.info("Sampling shows at least ~%d rows in first %d file(s).", total_rows, min(5, len(files)))

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