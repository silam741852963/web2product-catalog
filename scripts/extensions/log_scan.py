from __future__ import annotations

import asyncio
import csv
import gzip
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from extensions.output_paths import sanitize_bvdid

# -------- Models --------

@dataclass
class LogHit:
    bvdid: str                       # Original BVDID iff present in log lines
    sanitized_bvdid: str             # Folder id under outputs/
    level: str                       # "ERROR" | "WARNING"
    log_path: str
    line_no: int                     # 1-based
    message: str
    context_before: List[str]
    context_after: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

# -------- Defaults & Regex --------

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
    "error", "warning", "runtimeerror", "traceback",
    "page.goto", "acs-goto", "failed on navigating",
    "net::err_", "timeout",
}

# -------- Helpers --------

def _iter_company_log_files(outputs_root: Path, glob_pattern: str) -> List[Path]:
    root = outputs_root
    if not root.exists():
        return []
    return list(root.rglob(f"logs/{glob_pattern}"))

def _compile_user_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    for p in patterns:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out

def _pick_matchers(level: str, extra_patterns: Optional[List[re.Pattern]] = None) -> Tuple[List[re.Pattern], str]:
    extra_patterns = extra_patterns or []
    if level == "error":
        return (DEFAULT_ERROR_PATTERNS + extra_patterns, "ERROR")
    if level == "warning":
        return (DEFAULT_WARNING_PATTERNS + extra_patterns, "WARNING")
    return (DEFAULT_ERROR_PATTERNS + DEFAULT_WARNING_PATTERNS + extra_patterns, "ALL")

def _infer_level_from_line(line: str, default_level: str) -> str:
    if re.search(r"\[(ERROR)\]", line, re.IGNORECASE) or re.search(r"\bError:\b", line, re.IGNORECASE):
        return "ERROR"
    if re.search(r"\[(WARN|WARNING)\]", line, re.IGNORECASE) or re.search(r"\bWarning:\b", line, re.IGNORECASE):
        return "WARNING"
    # fallback to default preference
    if default_level == "error":
        return "ERROR"
    if default_level == "warning":
        return "WARNING"
    return "ERROR"

def _extract_sanitized_id_from_path(path: Path) -> str:
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
    """Return a file-like object for normal or .gz files"""
    if path.suffix.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding=encoding, errors="ignore")
    return path.open("r", encoding=encoding, errors="ignore")

# -------- Single-file scan (sync, used inside thread pool) --------

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
            ctx_before = [l.rstrip("\n") for l in lines[before_start:i]] if context_before else []
            ctx_after = [l.rstrip("\n") for l in lines[i+1:after_end]] if context_after else []

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

# -------- Async scanner (public API) --------

async def ascan_logs(
    *,
    outputs_root: Path = Path("outputs"),
    logs_glob: str = "*.log",
    level: str = "error",                   # 'error' | 'warning' | 'all'
    patterns: Optional[List[str]] = None,   # extra user regex pattern strings
    context_before: int = 0,
    context_after: int = 20,
    encoding: str = "utf-8",
    workers: int = 64,                      # max concurrent file scans
    use_prefilter: bool = True,
) -> Tuple[Dict[str, List[LogHit]], List[LogHit]]:
    """
    Async scan logs under outputs/{sanitized_bvdid}/logs/*.log (and .log.gz).
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
        chunk = log_files[i:i+CHUNK]
        results = await asyncio.gather(*[_task(p) for p in chunk], return_exceptions=False)
        for hits in results:
            all_hits.extend(hits)

    # Group hits preferring original BVDID when present
    hits_by_company: Dict[str, List[LogHit]] = {}
    for h in all_hits:
        key = h.bvdid if h.bvdid != "UNKNOWN" else (h.sanitized_bvdid or "UNKNOWN")
        hits_by_company.setdefault(key, []).append(h)

    return hits_by_company, all_hits

# -------- Synchronous wrapper (compatibility) --------

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

# -------- CSV enrichment helpers (unchanged, but robustified) --------

def _read_source_map_with_sanitized(source_csv: Optional[Path]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
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
            rows.append({"bvdid": original, "name": src_map[original].get("name", ""), "url": src_map[original].get("url", "")})
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

# -------- Optional export to JSON / CSV (for CLI convenience) --------

def dump_hits_json(hits: List[LogHit], out_json: Path) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [h.to_dict() for h in hits]
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def dump_hits_csv(hits: List[LogHit], out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bvdid", "sanitized_bvdid", "level", "log_path", "line_no", "message"]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for h in hits:
            writer.writerow({
                "bvdid": h.bvdid,
                "sanitized_bvdid": h.sanitized_bvdid,
                "level": h.level,
                "log_path": h.log_path,
                "line_no": h.line_no,
                "message": h.message,
            })