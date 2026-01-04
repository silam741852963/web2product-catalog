from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from extensions.crawl.state_calibration import calibrate, check


def _print_sample(title: str, s) -> None:
    print(f"\n== {title} ==")
    print(f"bvdid: {s.bvdid}")

    print("\n[DB snapshot]")
    print(s.db_snapshot)

    print("\n[crawl_meta.json keys]")
    print(sorted(list(s.crawl_meta.keys())))

    print("\n[crawl_meta.json excerpt]")
    excerpt = {
        k: s.crawl_meta.get(k)
        for k in (
            "version",
            "company_name",
            "root_url",
            "industry",
            "nace",
            "industry_label",
            "industry_source",
        )
    }
    print(json.dumps(excerpt, ensure_ascii=False, indent=2))

    print("\n[url_index.__meta__ excerpt]")
    idx_excerpt = {
        k: s.url_index_meta.get(k)
        for k in (
            "version",
            "crawl_finished",
            "max_pages",
            "recrawl_requested",
            "recrawl_reason",
        )
    }
    print(json.dumps(idx_excerpt, ensure_ascii=False, indent=2))

    # Useful quick sanity:
    snap = s.db_snapshot
    print("\n[derived quick stats]")
    print(
        json.dumps(
            {
                "status": snap.status,
                "urls_total": snap.urls_total,
                "urls_markdown_done": snap.urls_markdown_done,
                "urls_llm_done": snap.urls_llm_done,
                "done_reason": snap.done_reason,
                "done_at": snap.done_at,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="crawl_state_calibrate")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--db-path", type=str, default=None)
    p_check.add_argument("--sample-bvdid", type=str, default=None)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--db-path", type=str, default=None)
    p_cal.add_argument("--sample-bvdid", type=str, default=None)
    p_cal.add_argument("--no-global-state", action="store_true")

    args = p.parse_args()

    db_path: Optional[Path] = Path(args.db_path).resolve() if args.db_path else None
    sample_bvdid = args.sample_bvdid

    if args.cmd == "check":
        s = check(db_path=db_path, sample_bvdid=sample_bvdid)
        _print_sample("CHECK", s)
        return

    if args.cmd == "calibrate":
        before = check(db_path=db_path, sample_bvdid=sample_bvdid)
        _print_sample("BEFORE", before)

        rep = calibrate(
            db_path=db_path,
            sample_bvdid=sample_bvdid,
            write_global_state=(not args.no_global_state),
        )

        print("\n== CALIBRATION REPORT ==")
        print(
            json.dumps(
                {
                    "db_path": rep.db_path,
                    "touched_companies": rep.touched_companies,
                    "wrote_global_state": rep.wrote_global_state,
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
