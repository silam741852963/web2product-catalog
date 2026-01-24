from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from configs.language import default_language_factory
from extensions.crawl.calibration import (
    calibrate,
    check,
    enforce_max_pages,
    fix_corrupt_url_indexes,
    reconcile,
    reset,
    scan_corrupt_url_indexes,
)
from extensions.io.load_source import (
    DEFAULT_INDUSTRY_FALLBACK_PATH,
    DEFAULT_NACE_INDUSTRY_PATH,
)
from extensions.io.output_paths import ensure_output_root

logger = logging.getLogger(__name__)


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


def _print_corruption_report(title: str, rep) -> None:
    print(f"\n== {title} ==")
    print(
        json.dumps(
            {
                "out_dir": rep.out_dir,
                "db_path": rep.db_path,
                "scanned_companies": rep.scanned_companies,
                "affected_companies": rep.affected_companies,
                "affected_files": getattr(rep, "affected_files", None),
                "quarantined_files": getattr(rep, "quarantined_files", None),
                "marked_pending": getattr(rep, "marked_pending", None),
                "run_done_unmarked": getattr(rep, "run_done_unmarked", None),
                "dry_run": getattr(rep, "dry_run", None),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if rep.examples:
        print("\n[Example corrupted file(s)]")
        for ex in rep.examples:
            print("-" * 80)
            print(f"company_id: {ex.company_id}")
            print(f"path:      {ex.path}")
            print(f"size:      {ex.size_bytes}")
            print(f"reason:    {ex.reason}")
            print(f"head_hex:  {ex.head_bytes_hex}")
            print(f"preview:   {ex.head_text_preview!r}")


def _print_reconcile_report(rep) -> None:
    print("\n== RECONCILE REPORT ==")
    print(
        json.dumps(
            {
                "out_dir": rep.out_dir,
                "db_path": rep.db_path,
                "dry_run": rep.dry_run,
                "scanned_companies": rep.scanned_companies,
                "upgraded_to_terminal_done": rep.upgraded_to_terminal_done,
                "wrote_global_state": rep.wrote_global_state,
                "examples": rep.example_company_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _print_reset_report(rep, *, delete_db_rows: bool) -> None:
    print("\n== RESET REPORT ==")
    print(
        json.dumps(
            {
                "out_dir": rep.out_dir,
                "db_path": rep.db_path,
                "dry_run": rep.dry_run,
                "targets": rep.targets,
                "delete_db_rows": bool(delete_db_rows),
                "scanned_companies": rep.scanned_companies,
                "selected_companies": rep.selected_companies,
                "deleted_dirs": rep.deleted_dirs,
                "missing_dirs": rep.missing_dirs,
                "db_rows_reset": rep.db_rows_reset,
                "run_done_rows_deleted": rep.run_done_rows_deleted,
                "retry_quarantine_rows_deleted": getattr(
                    rep, "retry_quarantine_rows_deleted", None
                ),
                "retry_state_rows_deleted": getattr(
                    rep, "retry_state_rows_deleted", None
                ),
                "wrote_global_state": rep.wrote_global_state,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if rep.candidates:
        print("\n[Candidates (with reasons)]")
        for c in rep.candidates:
            print("-" * 80)
            print(f"company_id: {c.company_id}")
            print(f"reasons:    {c.reasons}")


def _print_enforce_max_pages_report(rep) -> None:
    print("\n== ENFORCE MAX PAGES REPORT ==")
    print(
        json.dumps(
            {
                "out_dir": rep.out_dir,
                "db_path": rep.db_path,
                "dry_run": rep.dry_run,
                "selector": rep.selector,
                "scanned_companies": rep.scanned_companies,
                "applied_companies": rep.applied_companies,
                "skipped_companies": rep.skipped_companies,
                "total_candidates": rep.total_candidates,
                "total_kept": rep.total_kept,
                "total_overflow": rep.total_overflow,
                "wrote_global_state": rep.wrote_global_state,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if rep.companies:
        print("\n[Companies]")
        for c in rep.companies:
            print("-" * 80)
            print(
                json.dumps(
                    {
                        "company_id": c.company_id,
                        "max_pages": c.max_pages,
                        "candidates": c.candidates,
                        "kept": c.kept,
                        "overflow": c.overflow,
                        "moved_md": c.moved_md,
                        "moved_html": c.moved_html,
                        "skipped_reason": c.skipped_reason,
                        "example_overflow_urls": c.example_overflow_urls,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


def main() -> None:
    p = argparse.ArgumentParser(prog="crawl_state_calibrate")
    sub = p.add_subparsers(dest="cmd", required=True)

    p.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Global concurrency for subcommands that support it.",
    )

    p_check = sub.add_parser("check")
    p_check.add_argument("--out-dir", type=str, required=True)
    p_check.add_argument("--db-path", type=str, default=None)
    p_check.add_argument("--sample-company-id", type=str, default=None)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--out-dir", type=str, required=True)
    p_cal.add_argument("--db-path", type=str, default=None)
    p_cal.add_argument("--sample-company-id", type=str, default=None)
    p_cal.add_argument("--no-global-state", action="store_true")

    p_cal.add_argument("--dataset-file", type=str, default=None)
    p_cal.add_argument("--company-file", type=str, default=None)

    p_cal.add_argument(
        "--industry-nace-path", type=str, default=str(DEFAULT_NACE_INDUSTRY_PATH)
    )
    p_cal.add_argument(
        "--industry-fallback-path",
        type=str,
        default=str(DEFAULT_INDUSTRY_FALLBACK_PATH),
    )

    p_cor = sub.add_parser(
        "corrupt", help="scan/fix corrupted url_index.json (e.g., NUL-byte files)"
    )
    p_cor.add_argument("--out-dir", type=str, required=True)
    p_cor.add_argument("--db-path", type=str, default=None)
    p_cor.add_argument("--max-examples", type=int, default=1)

    p_cor.add_argument(
        "--mark-pending",
        action="store_true",
        help="If set: quarantine corrupted url_index.json and mark the company pending/not-done in DB.",
    )
    p_cor.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change but do not modify files/DB.",
    )
    p_cor.add_argument(
        "--no-quarantine",
        action="store_true",
        help="Do not rename corrupted files aside.",
    )
    p_cor.add_argument(
        "--no-unmark-run-done",
        action="store_true",
        help="Do not delete (latest_run, company_id) from run_company_done when marking pending.",
    )

    p_rec = sub.add_parser(
        "reconcile", help="reconcile terminal/quarantined rows (no deletion)"
    )
    p_rec.add_argument("--out-dir", type=str, required=True)
    p_rec.add_argument("--db-path", type=str, default=None)
    p_rec.add_argument("--dry-run", action="store_true")
    p_rec.add_argument("--no-global-state", action="store_true")
    p_rec.add_argument("--max-examples", type=int, default=10)

    p_res = sub.add_parser(
        "reset", help="delete output dirs + reset DB + unskip run_company_done"
    )
    p_res.add_argument("--out-dir", type=str, required=True)
    p_res.add_argument("--db-path", type=str, default=None)
    p_res.add_argument(
        "--target",
        action="append",
        default=[],
        choices=[
            "crawl_not_finished",
            "pending",
            "markdown_not_done",
            "quarantined",
            "url_index_corrupt",
            "url_index_empty",
            "missing_output_dir",
        ],
        help="OR-selection across targets; can be repeated",
    )
    p_res.add_argument("--include-company-id", action="append", default=[])
    p_res.add_argument("--exclude-company-id", action="append", default=[])
    p_res.add_argument("--dry-run", action="store_true")
    p_res.add_argument("--no-global-state", action="store_true")
    p_res.add_argument("--max-examples", type=int, default=200)
    p_res.add_argument(
        "--clear-retry-store",
        action="store_true",
        help="If set: also remove selected company_ids from out_dir/_retry/quarantine.json and out_dir/_retry/retry_state.json.",
    )
    p_res.add_argument(
        "--delete-db-rows",
        action="store_true",
        help="If set: delete selected company rows from DB (companies + run_company_done + any other table with company_id) instead of resetting them to pending.",
    )

    p_emp = sub.add_parser(
        "enforce-max-pages",
        help="rank URLs with DualBM25 (language-aware) and suppress overflow beyond max_pages",
    )
    p_emp.add_argument("--out-dir", type=str, required=True)
    p_emp.add_argument("--db-path", type=str, default=None)
    p_emp.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language key for default_language_factory (required only for enforce-max-pages).",
    )
    p_emp.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Hard cap; if omitted, uses companies.max_pages per company",
    )
    p_emp.add_argument(
        "--apply-to",
        type=str,
        default="crawl_finished_true",
        choices=["crawl_finished_true", "all"],
        help="Selector: default only applies to crawl_finished=1 companies",
    )
    p_emp.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change but do not move files or modify DB",
    )
    p_emp.add_argument("--no-global-state", action="store_true")
    p_emp.add_argument("--max-examples", type=int, default=50)

    args = p.parse_args()

    conc = int(args.concurrency)
    if conc <= 0:
        raise RuntimeError("--concurrency must be >= 1")

    out_dir = Path(ensure_output_root(args.out_dir)).expanduser().resolve()

    db_path: Optional[Path]
    if getattr(args, "db_path", None):
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (out_dir / "crawl_state.sqlite3").resolve()

    if args.cmd == "check":
        s = check(
            out_dir=out_dir, db_path=db_path, sample_company_id=args.sample_company_id
        )
        _print_sample("CHECK", s)
        return

    if args.cmd == "calibrate":
        before = check(
            out_dir=out_dir, db_path=db_path, sample_company_id=args.sample_company_id
        )
        _print_sample("BEFORE", before)

        rep = calibrate(
            out_dir=out_dir,
            db_path=db_path,
            sample_company_id=args.sample_company_id,
            write_global_state=(not args.no_global_state),
            concurrency=conc,
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
                    "source_companies_used": rep.source_companies_used,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        _print_sample("AFTER", rep.sample_after)
        return

    if args.cmd == "corrupt":
        quarantine = not bool(args.no_quarantine)
        unmark_run_done = not bool(args.no_unmark_run_done)

        if not args.mark_pending:
            rep = scan_corrupt_url_indexes(
                out_dir=out_dir,
                db_path=db_path,
                max_examples=int(args.max_examples),
                concurrency=conc,
            )
            _print_corruption_report("CORRUPTION SCAN", rep)
            return

        rep2 = fix_corrupt_url_indexes(
            out_dir=out_dir,
            db_path=db_path,
            max_examples=int(args.max_examples),
            mark_pending=True,
            quarantine=quarantine,
            unmark_run_done=unmark_run_done,
            dry_run=bool(args.dry_run),
            concurrency=conc,
        )
        _print_corruption_report("CORRUPTION FIX", rep2)
        return

    if args.cmd == "reconcile":
        rep = reconcile(
            out_dir=out_dir,
            db_path=db_path,
            write_global_state=(not args.no_global_state),
            dry_run=bool(args.dry_run),
            max_examples=int(args.max_examples),
            concurrency=conc,
        )
        _print_reconcile_report(rep)
        return

    if args.cmd == "reset":
        targets = set(getattr(args, "target", []) or [])
        if not targets:
            raise RuntimeError("reset requires at least one --target")

        delete_db_rows = bool(getattr(args, "delete_db_rows", False))

        rep = reset(
            out_dir=out_dir,
            db_path=db_path,
            targets=targets,
            include_company_ids=getattr(args, "include_company_id", []) or [],
            exclude_company_ids=getattr(args, "exclude_company_id", []) or [],
            write_global_state=(not args.no_global_state),
            dry_run=bool(args.dry_run),
            max_examples=int(args.max_examples),
            scan_concurrency=conc,
            clear_retry_store=bool(getattr(args, "clear_retry_store", False)),
            delete_db_rows=delete_db_rows,
        )
        _print_reset_report(rep, delete_db_rows=delete_db_rows)
        return

    if args.cmd == "enforce-max-pages":
        default_language_factory.set_language(str(args.lang))
        effective_langs = default_language_factory.effective_langs()
        logger.info(
            "language_ready lang_target=%s lang_effective=%s",
            str(args.lang),
            ",".join(effective_langs),
        )

        rep = enforce_max_pages(
            out_dir=out_dir,
            db_path=db_path,
            max_pages=(int(args.max_pages) if args.max_pages is not None else None),
            selector=str(args.apply_to),
            apply_only_if_crawl_finished=(str(args.apply_to) == "crawl_finished_true"),
            write_global_state=(not args.no_global_state),
            dry_run=bool(args.dry_run),
            max_examples=int(args.max_examples),
            concurrency=conc,
        )
        _print_enforce_max_pages_report(rep)
        return

    raise RuntimeError(f"Unhandled cmd: {args.cmd!r}")


if __name__ == "__main__":
    main()
