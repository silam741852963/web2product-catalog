from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from extensions.crawl.state_calibration import calibrate, check
from extensions.io.load_source import (
    DEFAULT_INDUSTRY_FALLBACK_PATH,
    DEFAULT_NACE_INDUSTRY_PATH,
)
from extensions.io.output_paths import ensure_output_root


def _print_sample(title: str, s) -> None:
    print(f"\n== {title} ==")
    print(f"company_id: {s.company_id}")

    print("\n[DB snapshot excerpt]")
    snap = s.db_snapshot
    excerpt = {
        "company_id": snap.company_id,
        "root_url": snap.root_url,
        "name": snap.name,
        "industry": snap.industry,
        "nace": snap.nace,
        "industry_label": snap.industry_label,
        "industry_label_source": snap.industry_label_source,
        "status": snap.status,
        "crawl_finished": snap.crawl_finished,
        "urls_total": snap.urls_total,
        "urls_markdown_done": snap.urls_markdown_done,
        "urls_llm_done": snap.urls_llm_done,
        "done_reason": snap.done_reason,
        "done_at": snap.done_at,
        "last_error": snap.last_error,
        "created_at": snap.created_at,
        "updated_at": snap.updated_at,
        "last_crawled_at": snap.last_crawled_at,
        "max_pages": snap.max_pages,
    }
    print(json.dumps(excerpt, ensure_ascii=False, indent=2))

    print("\n[crawl_meta.json keys]")
    print(sorted(list(s.crawl_meta.keys())))

    print("\n[crawl_meta.json excerpt]")
    cm = s.crawl_meta
    cm_excerpt = {
        k: cm.get(k)
        for k in (
            "version_metadata",
            "company_id",
            "root_url",
            "name",
            "industry",
            "nace",
            "industry_label",
            "industry_label_source",
            "status",
            "crawl_finished",
            "urls_total",
            "urls_markdown_done",
            "urls_llm_done",
            "created_at",
            "updated_at",
            "last_crawled_at",
            "max_pages",
            "calibrated_at",
        )
    }
    print(json.dumps(cm_excerpt, ensure_ascii=False, indent=2))

    print("\n[url_index.__meta__ excerpt]")
    im = s.url_index_meta
    im_excerpt = {
        k: im.get(k)
        for k in (
            "company_id",
            "crawl_finished",
            "crawl_finished_at",
            "crawl_reason",
            "total_pages",
            "markdown_saved",
            "markdown_suppressed",
            "timeout_pages",
            "memory_pressure_pages",
            "pages_seen",
            "hard_max_pages",
            "hard_max_pages_hit",
            "created_at",
            "updated_at",
            "version_metadata",
        )
    }
    print(json.dumps(im_excerpt, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="crawl_state_calibrate")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--out-dir", type=str, required=True)
    p_check.add_argument("--db-path", type=str, default=None)  # optional override
    p_check.add_argument("--sample-company-id", type=str, default=None)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--out-dir", type=str, required=True)
    p_cal.add_argument("--db-path", type=str, default=None)  # optional override
    p_cal.add_argument("--sample-company-id", type=str, default=None)
    p_cal.add_argument("--no-global-state", action="store_true")
    p_cal.add_argument("--concurrency", type=int, default=32)

    # Source enrichment
    p_cal.add_argument("--dataset-file", type=str, default=None)
    p_cal.add_argument("--company-file", type=str, default=None)

    # Same defaults as extensions.io.load_source
    p_cal.add_argument(
        "--industry-nace-path", type=str, default=str(DEFAULT_NACE_INDUSTRY_PATH)
    )
    p_cal.add_argument(
        "--industry-fallback-path",
        type=str,
        default=str(DEFAULT_INDUSTRY_FALLBACK_PATH),
    )

    args = p.parse_args()

    out_dir = ensure_output_root(args.out_dir)

    db_path: Optional[Path]
    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (out_dir / "crawl_state.sqlite3").resolve()

    sample_company_id = args.sample_company_id

    if args.cmd == "check":
        s = check(out_dir=out_dir, db_path=db_path, sample_company_id=sample_company_id)
        _print_sample("CHECK", s)
        return

    if args.cmd == "calibrate":
        before = check(
            out_dir=out_dir, db_path=db_path, sample_company_id=sample_company_id
        )
        _print_sample("BEFORE", before)

        rep = calibrate(
            out_dir=out_dir,
            db_path=db_path,
            sample_company_id=sample_company_id,
            write_global_state=(not args.no_global_state),
            concurrency=int(args.concurrency),
            dataset_file=(
                Path(args.dataset_file).expanduser().resolve()
                if args.dataset_file
                else None
            ),
            company_file=(
                Path(args.company_file).expanduser().resolve()
                if args.company_file
                else None
            ),
            industry_nace_path=(
                Path(args.industry_nace_path).expanduser().resolve()
                if args.industry_nace_path
                else None
            ),
            industry_fallback_path=(
                Path(args.industry_fallback_path).expanduser().resolve()
                if args.industry_fallback_path
                else None
            ),
        )

        print("\n== CALIBRATION REPORT ==")
        print(
            json.dumps(
                {
                    "out_dir": rep.out_dir,
                    "db_path": rep.db_path,
                    "touched_companies": rep.touched_companies,
                    "wrote_global_state": rep.wrote_global_state,
                    "source_companies_loaded": rep.source_companies_loaded,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        _print_sample("AFTER", rep.sample_after)
        return

    raise RuntimeError(f"Unhandled cmd: {args.cmd!r}")


if __name__ == "__main__":
    main()
