from __future__ import annotations

import argparse
from typing import Iterable, Optional

from configs.language import validate_lang_code
from extensions.crawl.runner_constants import RETRY_EXIT_CODE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deep crawl corporate websites (per company pipeline)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", type=str)
    g.add_argument("--company-file", type=str)

    p.add_argument("--company-id", type=str, default=None)
    p.add_argument("--lang", type=str, default="en")
    p.add_argument(
        "--out-dir", "--output-dir", dest="out_dir", type=str, default="outputs"
    )

    p.add_argument(
        "--strategy", choices=["bestfirst", "bfs_internal", "dfs"], default="bestfirst"
    )

    p.add_argument("--llm-mode", choices=["none", "presence", "full"], default="none")
    p.add_argument("--llm-model", type=str, default=None)

    p.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root path (used for git metadata in LLM patching).",
    )

    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    p.add_argument("--dataset-file", type=str, default=None)

    p.add_argument("--company-concurrency", type=int, default=18)
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--page-timeout-ms", type=int, default=30000)

    p.add_argument("--crawl4ai-cache-dir", type=str, default=None)
    p.add_argument(
        "--crawl4ai-cache-mode",
        choices=["enabled", "disabled", "read_only", "write_only", "bypass"],
        default="bypass",
    )

    p.add_argument(
        "--force-recrawl",
        action="store_true",
        help="Force full recrawl of all non-terminal companies (keeps Crawl4AI cache; skips pages==max_pages check).",
    )

    p.add_argument(
        "--retry-mode", choices=["all", "skip-retry", "only-retry"], default="all"
    )
    p.add_argument(
        "--retry-exit-code",
        type=int,
        default=RETRY_EXIT_CODE,
        help="Exit code used when retry is recommended/required.",
    )
    p.add_argument("--enable-session-log", action="store_true")
    p.add_argument("--enable-resource-monitor", action="store_true")
    p.add_argument("--finalize-in-progress-md", action="store_true")

    p.add_argument(
        "--industry-enrichment",
        action="store_true",
        help="Enable industry label enrichment.",
    )
    p.add_argument(
        "--no-industry-enrichment",
        action="store_true",
        help="Disable industry label enrichment even if LLM is enabled (overrides auto-enable).",
    )
    p.add_argument(
        "--industry-nace-path", type=str, default=None, help="Override nace.ods path."
    )
    p.add_argument(
        "--industry-fallback-path",
        type=str,
        default=None,
        help="Override industry.ods path.",
    )
    p.add_argument("--source-encoding", type=str, default="utf-8")
    p.add_argument("--source-limit", type=int, default=None)
    p.add_argument("--source-no-aggregate-same-url", action="store_true")
    p.add_argument("--source-no-interleave-domains", action="store_true")

    p.add_argument("--page-result-concurrency", type=int, default=8)
    p.add_argument("--page-queue-maxsize", type=int, default=32)
    p.add_argument("--url-index-queue-maxsize", type=int, default=1024)

    p.add_argument("--crawler-pool-size", type=int, default=9)
    p.add_argument("--crawler-recycle-after", type=int, default=12)

    p.add_argument("--max-start-per-tick", type=int, default=3)
    p.add_argument("--crawler-capacity-multiplier", type=int, default=3)
    p.add_argument("--idle-recycle-interval-sec", type=float, default=25.0)
    p.add_argument("--idle-recycle-raw-frac", type=float, default=0.88)
    p.add_argument("--idle-recycle-eff-frac", type=float, default=0.83)

    p.add_argument("--crawler-lease-timeout-sec", type=float, default=240.0)
    p.add_argument("--arun-init-timeout-sec", type=float, default=180.0)
    p.add_argument("--stream-no-yield-timeout-sec", type=float, default=600.0)
    p.add_argument("--submit-timeout-sec", type=float, default=60.0)
    p.add_argument("--resume-md-mode", choices=["direct", "deep"], default="direct")
    p.add_argument("--direct-fetch-url-timeout-sec", type=float, default=180.0)
    p.add_argument("--processor-finish-timeout-sec", type=float, default=360.0)
    p.add_argument("--generator-close-timeout-sec", type=float, default=60.0)
    p.add_argument("--company-crawl-timeout-sec", type=float, default=3600.0)

    p.add_argument("--company-progress-heartbeat-sec", type=float, default=30.0)
    p.add_argument("--company-progress-throttle-sec", type=float, default=12.0)

    p.add_argument("--global-state-write-interval-sec", type=float, default=1.5)

    return p


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = build_parser()
    ns = p.parse_args(list(argv) if argv is not None else None)

    try:
        ns.lang = validate_lang_code(getattr(ns, "lang", None))
    except ValueError as e:
        p.error(str(e))

    return ns
