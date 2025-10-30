from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple, Set
from dataclasses import dataclass


@dataclass
class CompanyInput:
    hojin_id: str
    company_name: str
    url: str
    metadata: Dict[str, str]  # Other fields retained here


def _iter_csv_rows(
    path: Path,
    *,
    required_fields: List[str],
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> Iterable[CompanyInput]:
    """
    Internal helper: yields CompanyInput from a single CSV file.
    """
    count = 0
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and count >= limit:
                break
            if all((row.get(field, "") or "").strip() for field in required_fields):
                metadata = {k: v for k, v in row.items() if k not in required_fields}
                yield CompanyInput(
                    hojin_id=row["hojin_id"].strip(),
                    company_name=row["company_name"].strip(),
                    url=row["url"].strip(),
                    metadata=metadata,
                )
                count += 1


def _gather_csv_files(
    root: Path,
    *,
    pattern: str = "*.csv",
    recursive: bool = True,
) -> List[Path]:
    """
    Collect CSV files under a directory (or just return [root] if root is a file).
    """
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    if recursive:
        return sorted(p for p in root.rglob(pattern) if p.is_file())
    else:
        return sorted(p for p in root.glob(pattern) if p.is_file())


def _dedupe_records(
    records: Iterable[CompanyInput],
    *,
    keys: Tuple[str, str] = ("hojin_id", "url"),
) -> List[CompanyInput]:
    """
    Deduplicate CompanyInput list by composite (hojin_id, url) or other pair.
    """
    seen: Set[Tuple[str, str]] = set()
    out: List[CompanyInput] = []
    for r in records:
        k1 = getattr(r, keys[0], "").strip().lower()
        k2 = getattr(r, keys[1], "").strip().lower()
        key = (k1, k2)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_companies(
    path_or_dir: Path,
    *,
    required_fields: List[str] = ("hojin_id", "company_name", "url"),
    encoding: str = "utf-8",
    limit_per_file: Optional[int] = None,
    recursive: bool = True,
    pattern: str = "*.csv",
    dedupe: bool = True,
    dedupe_on: Tuple[str, str] = ("hojin_id", "url"),
) -> List[CompanyInput]:
    """
    Load companies from either a single CSV file or a directory of CSV files.

    Args:
        path_or_dir: file or directory path.
        required_fields: required columns in each CSV row.
        encoding: CSV encoding.
        limit_per_file: if set, limit rows per file (useful for debugging).
        recursive: if True and path_or_dir is a directory, search recursively.
        pattern: glob pattern for matching CSV filenames.
        dedupe: if True, dedupe records across all files by (hojin_id, url).
        dedupe_on: the pair of fields to dedupe on.

    Returns:
        List[CompanyInput]
    """
    root = Path(path_or_dir)
    files = _gather_csv_files(root, pattern=pattern, recursive=recursive)
    records: List[CompanyInput] = []
    for f in files:
        records.extend(
            _iter_csv_rows(
                f,
                required_fields=list(required_fields),
                encoding=encoding,
                limit=limit_per_file,
            )
        )
    if dedupe:
        records = _dedupe_records(records, keys=dedupe_on)
    return records


# Backwards-compatible name if other code imports it:
def load_companies_from_csv(
    path: Path,
    *,
    required_fields: List[str] = ["hojin_id", "company_name", "url"],
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> List[CompanyInput]:
    """
    Legacy single-file loader (kept for compatibility with existing imports).
    """
    return list(
        _iter_csv_rows(
            path,
            required_fields=required_fields,
            encoding=encoding,
            limit=limit,
        )
    )