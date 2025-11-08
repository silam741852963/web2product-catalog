from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple, Set
from dataclasses import dataclass


@dataclass
class CompanyInput:
    bvdid: str
    name: str
    url: str
    metadata: Dict[str, str]

    # Backward-compat properties
    @property
    def hojin_id(self) -> str:
        return self.bvdid

    @property
    def company_name(self) -> str:
        return self.name


def _resolve_keys(fieldnames: Optional[List[str]]) -> Tuple[str, str, str]:
    """
    Pick header names (case-insensitive) for (id, name, url).
    Prefers new dataset headers, but falls back to legacy ones.
    Returns original-cased header names found in the CSV (or '' if missing).
    """
    fset = { (fn or "").strip().lower(): fn for fn in (fieldnames or []) }

    def pick(*candidates: str) -> str:
        for c in candidates:
            if c in fset:
                return fset[c]
        return ""

    id_key   = pick("bvdid", "hojin_id", "id")
    name_key = pick("name", "company_name", "company")
    url_key  = pick("url", "website", "homepage", "home_page")
    return id_key, name_key, url_key


def _iter_csv_rows(
    path: Path,
    *,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> Iterable[CompanyInput]:
    """
    Yield CompanyInput from a single CSV file.

    Rules:
    - Mandatory: bvdid (can fall back to hojin_id/id), name (fallback company_name/company), url
    - Skip rows with empty/missing URL.
    - Extra columns go into `metadata` (excluding the 3 primary fields).
    """
    count = 0
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        id_key, name_key, url_key = _resolve_keys(reader.fieldnames)

        for row in reader:
            if limit is not None and count >= limit:
                break

            # Resolve values using resolved keys
            id_val = (row.get(id_key, "") or "").strip() if id_key else ""
            nm_val = (row.get(name_key, "") or "").strip() if name_key else ""
            url_val = (row.get(url_key, "") or "").strip() if url_key else ""

            # Inspect a lowercased view for legacy fallback
            row_l = { (k or "").strip().lower(): (v or "") for k, v in row.items() }

            if not id_val:
                id_val = (row_l.get("bvdid") or row_l.get("hojin_id") or row_l.get("id") or "").strip()
            if not nm_val:
                nm_val = (row_l.get("name") or row_l.get("company_name") or row_l.get("company") or "").strip()
            if not url_val:
                url_val = (row_l.get("url") or row_l.get("website") or row_l.get("homepage") or row_l.get("home_page") or "").strip()

            # Mandatory checks
            if not url_val or not id_val or not nm_val:
                continue

            # Build metadata excluding the chosen primary keys (case-insensitive)
            exclude = { (id_key or "").lower(), (name_key or "").lower(), (url_key or "").lower() }
            metadata: Dict[str, str] = {
                k: (v or "")
                for k, v in row.items()
                if (k or "").lower() not in exclude
            }

            yield CompanyInput(
                bvdid=id_val,
                name=nm_val,
                url=url_val,
                metadata=metadata,
            )
            count += 1


def _gather_csv_files(root: Path, *, pattern: str = "*.csv", recursive: bool = True) -> List[Path]:
    """
    Return CSV file list. If root is a file, return [root].
    If root is a directory, recurse for pattern.
    """
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in (root.rglob(pattern) if recursive else root.glob(pattern)) if p.is_file())


def _dedupe_records(
    records: Iterable[CompanyInput],
    *,
    keys: Tuple[str, str] = ("bvdid", "url"),
) -> List[CompanyInput]:
    """
    Deduplicate by (bvdid, url) or other pair of attributes.
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
    encoding: str = "utf-8",
    limit_per_file: Optional[int] = None,
    recursive: bool = True,
    pattern: str = "*.csv",
    dedupe: bool = True,
    dedupe_on: Tuple[str, str] = ("bvdid", "url"),
) -> List[CompanyInput]:
    """
    Load companies from a CSV file or a directory of CSV files.

    Canonical fields:
      - bvdid (required; legacy fallback: hojin_id/id)
      - name  (required; legacy fallback: company_name/company)
      - url   (required; legacy fallback: website/homepage/home_page)
    Rows with empty URL are skipped.
    """
    root = Path(path_or_dir)
    files = _gather_csv_files(root, pattern=pattern, recursive=recursive)
    records: List[CompanyInput] = []
    for f in files:
        records.extend(_iter_csv_rows(f, encoding=encoding, limit=limit_per_file))
    if dedupe:
        records = _dedupe_records(records, keys=dedupe_on)
    return records


def load_companies_from_csv(
    path_or_dir: Path,
    *,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> List[CompanyInput]:
    """
    Backward-friendly loader:
    - If `path_or_dir` is a CSV file, load it.
    - If it's a directory, recurse for *.csv (fixes cases where --csv points to a dir).
    """
    p = Path(path_or_dir)
    if p.is_dir():
        return load_companies(p, encoding=encoding, limit_per_file=limit, recursive=True, pattern="*.csv")
    # Single-file mode
    return list(_iter_csv_rows(p, encoding=encoding, limit=limit))
