from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from extensions.io.output_paths import sanitize_bvdid

# =====================================================================================
# Models
# =====================================================================================


@dataclass
class LogHit:
    bvdid: str  # Original BVDID iff present in log lines
    sanitized_bvdid: str  # Folder id under outputs/
    level: str  # "ERROR" | "WARNING"
    log_path: str
    line_no: int  # 1-based
    message: str
    context_before: List[str]
    context_after: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


# =====================================================================================
# Defaults & Regex
# =====================================================================================

DEFAULT_ERROR_PATTERNS: List[re.Pattern] = [
    re.compile(r"\[(ERROR)\]", re.IGNORECASE),
    re.compile(r"\bError:\b", re.IGNORECASE),
    re.compile(r"\bRuntimeError\b", re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r"Page\.goto:\s*(Timeout|net::ERR_[A-Z_]+)", re.IGNORECASE),
    re.compile(r"Failed on navigating ACS-GOTO", re.IGNORECASE),
]

DEFAULT_WARNING_PATTERNS: List[re.Pattern] = [
    re.compile(r"\[(WARN|WARNING)\]", re.IGNORECASE),
    re.compile(r"\bWarning:\b", re.IGNORECASE),
]

BVDID_LINE_RE = re.compile(r"\[([A-Z]{2}\*[0-9A-Za-z]+)\]")

DEFAULT_PREFILTER_TOKENS = {
    "error",
    "warning",
    "runtimeerror",
    "traceback",
    "page.goto",
    "acs-goto",
    "failed on navigating",
    "net::err_",
    "timeout",
}


# =====================================================================================
# Helpers
# =====================================================================================


def _iter_company_log_files(outputs_root: Path, glob_pattern: str) -> List[Path]:
    """
    Iterate company log files under:
        outputs_root/**/log/<glob_pattern>

    Note: previously it was "logs/"; now logs are under companyid/log/.
    """
    root = outputs_root
    if not root.exists():
        return []
    return list(root.rglob(f"log/{glob_pattern}"))


def _compile_user_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    for p in patterns:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out


def _pick_matchers(
    level: str, extra_patterns: Optional[List[re.Pattern]] = None
) -> Tuple[List[re.Pattern], str]:
    extra_patterns = extra_patterns or []
    if level == "error":
        return (DEFAULT_ERROR_PATTERNS + extra_patterns, "ERROR")
    if level == "warning":
        return (DEFAULT_WARNING_PATTERNS + extra_patterns, "WARNING")
    return (DEFAULT_ERROR_PATTERNS + DEFAULT_WARNING_PATTERNS + extra_patterns, "ALL")


def _infer_level_from_line(line: str, default_level: str) -> str:
    if re.search(r"\[(ERROR)\]", line, re.IGNORECASE) or re.search(
        r"\bError:\b", line, re.IGNORECASE
    ):
        return "ERROR"
    if re.search(r"\[(WARN|WARNING)\]", line, re.IGNORECASE) or re.search(
        r"\bWarning:\b", line, re.IGNORECASE
    ):
        return "WARNING"
    # fallback to default preference
    if default_level == "error":
        return "ERROR"
    if default_level == "warning":
        return "WARNING"
    return "INFO"


def _extract_sanitized_id_from_path(path: Path) -> str:
    """
    Try to extract the sanitized company id from a full path like:
      /.../outputs/<sanitized_bvdid>/log/...
    """
    try:
        parts = list(path.resolve().parts)
    except Exception:
        parts = list(path.parts)
    try:
        idx = parts.index("outputs")
    except ValueError:
        return ""
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return ""


def _derive_prefilter_tokens(user_patterns: List[str]) -> List[str]:
    tokens = set(DEFAULT_PREFILTER_TOKENS)
    for p in user_patterns:
        s = p.lower()
        for t in re.split(r"[^a-z0-9:_/.\-]+", s):
            if len(t) >= 4:
                tokens.add(t)
    return sorted(tokens)


def _open_possibly_compressed(path: Path, encoding: str):
    """Return a file-like object for normal or .gz files."""
    if path.suffix.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding=encoding, errors="ignore")
    return path.open("r", encoding=encoding, errors="ignore")


# =====================================================================================
# Single-file scan (sync, used inside thread pool)
# =====================================================================================


def _scan_one_file_sync(
    log_path: Path,
    *,
    matchers: List[re.Pattern],
    default_level: str,
    prefilter_tokens: List[str],
    context_before: int,
    context_after: int,
    encoding: str,
    use_prefilter: bool = True,
) -> List[LogHit]:
    hits: List[LogHit] = []

    try:
        with _open_possibly_compressed(log_path, encoding=encoding) as fh:
            lines = fh.readlines()
    except Exception:
        # If we can't read file, just return no hits for that file
        return hits

    folder_id = _extract_sanitized_id_from_path(log_path) or "UNKNOWN"
    n = len(lines)
    i = 0
    current_bvdid: Optional[str] = None

    has_prefilter = bool(prefilter_tokens) and use_prefilter
    tokens = [t for t in prefilter_tokens] if has_prefilter else []

    def looks_interesting(s: str) -> bool:
        if not has_prefilter:
            return True
        low = s.lower()
        for t in tokens:
            if t in low:
                return True
        return False

    while i < n:
        raw_line = lines[i].rstrip("\n")
        if not raw_line:
            i += 1
            continue

        m_bv = BVDID_LINE_RE.search(raw_line)
        if m_bv:
            current_bvdid = m_bv.group(1)

        if looks_interesting(raw_line) and any(rx.search(raw_line) for rx in matchers):
            before_start = max(0, i - context_before)
            after_end = min(n, i + 1 + context_after)
            ctx_before = (
                [l.rstrip("\n") for l in lines[before_start:i]]
                if context_before
                else []
            )
            ctx_after = (
                [l.rstrip("\n") for l in lines[i + 1 : after_end]]
                if context_after
                else []
            )

            sev = _infer_level_from_line(raw_line, default_level)
            hit = LogHit(
                bvdid=current_bvdid or "UNKNOWN",
                sanitized_bvdid=folder_id,
                level=sev,
                log_path=str(log_path),
                line_no=i + 1,
                message=raw_line,
                context_before=ctx_before,
                context_after=ctx_after,
            )
            hits.append(hit)
            # skip past the context_after to avoid overlapping hits
            i = after_end
            continue

        i += 1

    return hits


# =====================================================================================
# Async scanner (public API)
# =====================================================================================


async def ascan_logs(
    *,
    outputs_root: Path = Path("outputs"),
    logs_glob: str = "*.log",
    level: str = "error",  # 'error' | 'warning' | 'all'
    patterns: Optional[List[str]] = None,  # extra user regex pattern strings
    context_before: int = 0,
    context_after: int = 20,
    encoding: str = "utf-8",
    workers: int = 64,  # max concurrent file scans
    use_prefilter: bool = True,
) -> Tuple[Dict[str, List[LogHit]], List[LogHit]]:
    """
    Async scan logs under outputs/{sanitized_bvdid}/log/*.log (and .log.gz).

    Returns (hits_by_company, all_hits).
    """
    log_files = _iter_company_log_files(outputs_root, logs_glob)
    if not log_files:
        return {}, []

    extra_patterns = _compile_user_patterns(patterns or [])
    matchers, default_level = _pick_matchers(level, extra_patterns)
    prefilter_tokens = _derive_prefilter_tokens(patterns or [])

    sem = asyncio.Semaphore(max(1, workers))

    async def _task(path: Path) -> List[LogHit]:
        async with sem:
            return await asyncio.to_thread(
                _scan_one_file_sync,
                path,
                matchers=matchers,
                default_level=default_level,
                prefilter_tokens=prefilter_tokens,
                context_before=context_before,
                context_after=context_after,
                encoding=encoding,
                use_prefilter=use_prefilter,
            )

    # chunk to avoid scheduling millions at once
    CHUNK = 1024
    all_hits: List[LogHit] = []
    for i in range(0, len(log_files), CHUNK):
        chunk = log_files[i : i + CHUNK]
        results = await asyncio.gather(
            *[_task(p) for p in chunk], return_exceptions=False
        )
        for hits in results:
            all_hits.extend(hits)

    # Group hits preferring original BVDID when present
    hits_by_company: Dict[str, List[LogHit]] = {}
    for h in all_hits:
        key = h.bvdid if h.bvdid != "UNKNOWN" else (h.sanitized_bvdid or "UNKNOWN")
        hits_by_company.setdefault(key, []).append(h)

    return hits_by_company, all_hits


# =====================================================================================
# Synchronous wrapper (compatibility)
# =====================================================================================


def scan_logs(
    *,
    outputs_root: Path = Path("outputs"),
    logs_glob: str = "*.log",
    level: str = "error",
    patterns: Optional[List[str]] = None,
    context_before: int = 0,
    context_after: int = 20,
    encoding: str = "utf-8",
    workers: int = 64,
    use_prefilter: bool = True,
) -> Tuple[Dict[str, List[LogHit]], List[LogHit]]:
    """
    Synchronous wrapper around ascan_logs().
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = ascan_logs(
        outputs_root=outputs_root,
        logs_glob=logs_glob,
        level=level,
        patterns=patterns,
        context_before=context_before,
        context_after=context_after,
        encoding=encoding,
        workers=workers,
        use_prefilter=use_prefilter,
    )

    if loop and loop.is_running():
        # running inside event loop: run in new thread loop
        return asyncio.run(coro)
    else:
        return asyncio.run(coro)


# =====================================================================================
# CSV enrichment helpers
# =====================================================================================


def _read_source_map_with_sanitized(
    source_csv: Optional[Path],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Returns:
      src_map: original_bvdid -> {name, url}
      rev_map: sanitized_bvdid -> original_bvdid
    """
    src_map: Dict[str, Dict[str, str]] = {}
    rev_map: Dict[str, str] = {}
    if not source_csv:
        return src_map, rev_map
    p = Path(source_csv)
    if not p.exists():
        return src_map, rev_map

    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = {h.lower(): h for h in (reader.fieldnames or [])}

        def pick(row: Dict[str, str], options: List[str], default: str = "") -> str:
            for opt in options:
                h = headers.get(opt)
                if h and row.get(h):
                    return row[h].strip()
            return default

        for row in reader:
            bvdid = pick(row, ["bvdid", "id"])
            if not bvdid:
                continue
            name = pick(row, ["name", "company", "company_name"])
            url = pick(row, ["url", "website", "root", "domain", "homepage"])
            src_map[bvdid] = {"name": name, "url": url}
            try:
                rev_map[sanitize_bvdid(bvdid)] = bvdid
            except Exception:
                rev_map[bvdid] = bvdid

    return src_map, rev_map


def create_test_csv(
    out_csv: Path,
    companies: Iterable[str],
    *,
    source_csv: Optional[Path] = None,
    include_header: bool = True,
) -> int:
    """
    Write a test CSV for the subset of companies.
    `companies` may contain original or sanitized BVDIDs.
    If `source_csv` is provided, enrich with name/url and normalize to original BVDID.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    companies = list(dict.fromkeys(companies))  # dedupe keep order
    src_map, rev_map = _read_source_map_with_sanitized(source_csv)

    rows: List[Dict[str, str]] = []
    for key in companies:
        original = key
        if key not in src_map and key in rev_map:
            original = rev_map[key]

        if original in src_map:
            rows.append(
                {
                    "bvdid": original,
                    "name": src_map[original].get("name", ""),
                    "url": src_map[original].get("url", ""),
                }
            )
        else:
            rows.append({"bvdid": original, "name": "", "url": ""})

    fieldnames = ["bvdid", "name", "url"]

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return len(rows)


# =====================================================================================
# Optional export to JSON / CSV (for CLI convenience)
# =====================================================================================


def dump_hits_json(hits: List[LogHit], out_json: Path) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [h.to_dict() for h in hits]
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def dump_hits_csv(hits: List[LogHit], out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bvdid", "sanitized_bvdid", "level", "log_path", "line_no", "message"]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for h in hits:
            writer.writerow(
                {
                    "bvdid": h.bvdid,
                    "sanitized_bvdid": h.sanitized_bvdid,
                    "level": h.level,
                    "log_path": h.log_path,
                    "line_no": h.line_no,
                    "message": h.message,
                }
            )


# =====================================================================================
# CLI (merged from scan_logs.py) + extraction feature
# =====================================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Async log scanner: scan outputs/{companyid}/log/*.log for ERROR/WARNING blocks; "
            "print and optionally export / extract pattern-matching lines."
        )
    )
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing per-company folders (default: ./outputs).",
    )
    p.add_argument(
        "--logs-glob",
        type=str,
        default="*.log",
        help="Glob pattern for log files relative to each log/ dir (default: *.log). Use *.log* to include .log.gz.",
    )
    p.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding for reading log files (default utf-8).",
    )
    p.add_argument(
        "--no-prefilter",
        dest="use_prefilter",
        action="store_false",
        help="Disable prefilter token scan (slower, but may find more).",
    )

    p.add_argument(
        "--level",
        choices=["error", "warning", "all"],
        default="error",
        help="Which severity to match (default: error).",
    )
    p.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Extra regex pattern(s) to match (can be repeated).",
    )

    p.add_argument(
        "--context-before",
        type=int,
        default=0,
        help="Lines before a hit to include (default: 0).",
    )
    p.add_argument(
        "--context-after",
        type=int,
        default=20,
        help="Lines after a hit to include (default: 20).",
    )

    p.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print hits to stdout.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of printed hits (0 = no limit).",
    )

    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write all hits as a JSON file (optional).",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Write all hits as a CSV file (optional).",
    )

    p.add_argument(
        "--write-test-csv",
        type=Path,
        default=None,
        help="Write a CSV of matched companies (BVDIDs) to this path.",
    )
    p.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="Original company CSV to enrich/normalize BVDIDs in the test CSV.",
    )

    p.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Max concurrent file scans (default: 64).",
    )

    p.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help=(
            "If set, for each company, write all hit lines whose message matches --pattern "
            "into a separate text file in this directory (one file per company)."
        ),
    )

    return p.parse_args()


def _print_hits(all_hits: List[LogHit], limit: int = 0) -> None:
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


def _extract_lines_to_files(
    all_hits: List[LogHit],
    *,
    patterns: List[str],
    extract_dir: Path,
) -> None:
    """
    Take all hits and, for lines whose `message` matches any of `patterns`,
    write them to per-company text files in `extract_dir`.

    Company key = original BVDID if known, otherwise sanitized_bvdid.
    Each line format:
        <LEVEL> <log_path>:<line_no> <message>
    """
    if not patterns:
        return

    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    regexes = _compile_user_patterns(patterns)

    by_company: Dict[str, List[str]] = {}

    for h in all_hits:
        msg = h.message or ""
        if not msg:
            continue
        if not any(rx.search(msg) for rx in regexes):
            continue

        key = h.bvdid if h.bvdid != "UNKNOWN" else (h.sanitized_bvdid or "UNKNOWN")
        line = f"{h.level} {h.log_path}:{h.line_no} {msg}"
        by_company.setdefault(key, []).append(line)

    if not by_company:
        return

    for key, lines in by_company.items():
        fname = f"{key}.logextract.txt"
        out_path = extract_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    companies: List[str] = []
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

    # New: extraction of pattern-matching lines into files
    if args.extract_dir is not None and args.pattern:
        _extract_lines_to_files(
            all_hits,
            patterns=args.pattern,
            extract_dir=args.extract_dir,
        )
        print(f"Wrote extracted lines to: {args.extract_dir}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()

