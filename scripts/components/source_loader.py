from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


# -----------------------------
# Model
# -----------------------------
@dataclass
class CompanyInput:
    bvdid: str
    name: str
    url: str
    metadata: Dict[str, str]

    # Backward-compat props
    @property
    def hojin_id(self) -> str:
        return self.bvdid

    @property
    def company_name(self) -> str:
        return self.name


# -----------------------------
# Key resolution / normalization
# -----------------------------
_PRIMARY_ID_KEYS = ("bvdid", "hojin_id", "id")
_PRIMARY_NAME_KEYS = ("name", "company_name", "company")
_PRIMARY_URL_KEYS = ("url", "website", "homepage", "home_page")


def _resolve_keys(fieldnames: Optional[Sequence[str]]) -> Tuple[str, str, str]:
    """
    Pick header names (case-insensitive) for (id, name, url).
    Returns **original-cased** names from the file ('' if not found).
    """
    fset = { (fn or "").strip().lower(): fn for fn in (fieldnames or []) }

    def pick(candidates: Sequence[str]) -> str:
        for c in candidates:
            if c in fset:
                return fset[c]
        return ""

    return (
        pick(_PRIMARY_ID_KEYS),
        pick(_PRIMARY_NAME_KEYS),
        pick(_PRIMARY_URL_KEYS),
    )


def _row_to_company(row: Dict[str, str], id_key: str, name_key: str, url_key: str) -> Optional[CompanyInput]:
    # Canonical, original-cased view
    id_val = (row.get(id_key, "") or "").strip() if id_key else ""
    nm_val = (row.get(name_key, "") or "").strip() if name_key else ""
    url_val = (row.get(url_key, "") or "").strip() if url_key else ""

    # Lowercased fallback view
    row_l = { (k or "").strip().lower(): (v or "") for k, v in row.items() }

    if not id_val:
        for k in _PRIMARY_ID_KEYS:
            if row_l.get(k):
                id_val = row_l[k].strip()
                break
    if not nm_val:
        for k in _PRIMARY_NAME_KEYS:
            if row_l.get(k):
                nm_val = row_l[k].strip()
                break
    if not url_val:
        for k in _PRIMARY_URL_KEYS:
            if row_l.get(k):
                url_val = row_l[k].strip()
                break

    if not id_val or not nm_val or not url_val:
        return None

    exclude = { (id_key or "").lower(), (name_key or "").lower(), (url_key or "").lower() }
    metadata: Dict[str, str] = { k: (v or "") for k, v in row.items() if (k or "").lower() not in exclude }
    return CompanyInput(bvdid=id_val, name=nm_val, url=url_val, metadata=metadata)


# -----------------------------
# CSV / TSV (no pandas needed)
# -----------------------------
def _iter_csv(path: Path, *, encoding: str, delimiter: str, limit: Optional[int]) -> Iterator[CompanyInput]:
    count = 0
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        id_key, name_key, url_key = _resolve_keys(reader.fieldnames)
        for row in reader:
            if limit is not None and count >= limit:
                break
            ci = _row_to_company(row, id_key, name_key, url_key)
            if ci:
                yield ci
                count += 1


# -----------------------------
# JSON / NDJSON (no pandas)
# -----------------------------
def _iter_json_like(path: Path, *, encoding: str, limit: Optional[int]) -> Iterator[CompanyInput]:
    text = path.read_text(encoding=encoding, errors="ignore").strip()
    rows: List[Dict] = []

    def as_dicts(obj) -> List[Dict]:
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # common wrappers: {"data":[...]} / {"records":[...]}
            for k in ("data", "records", "items", "rows"):
                v = obj.get(k)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
            return [obj]
        return []

    # Try JSON first
    try:
        rows = as_dicts(json.loads(text))
    except Exception:
        # Try NDJSON / JSONL
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue

    # Stream
    if not rows:
        return iter(())
    # Determine keys once from the first row
    id_key, name_key, url_key = _resolve_keys(list(rows[0].keys()))
    count = 0
    for row in rows:
        if limit is not None and count >= limit:
            break
        if not isinstance(row, dict):
            continue
        ci = _row_to_company({str(k): str(v) if v is not None else "" for k, v in row.items()}, id_key, name_key, url_key)
        if ci:
            yield ci
            count += 1


# -----------------------------
# Pandas-backed formats (optional)
# -----------------------------
def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "This file type requires pandas (and possibly pyarrow/fastparquet/openpyxl). "
            "Install extras, e.g.: pip install pandas pyarrow openpyxl"
        ) from e
    return pd


def _iter_pandas_df(df, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    if df is None or df.empty:
        return iter(())  # type: ignore
    # Normalize to dict rows
    for i, row in enumerate(df.to_dict(orient="records")):
        if limit is not None and i >= limit:
            break
        # Resolve keys per row using all columns
        id_key, name_key, url_key = _resolve_keys(list(row.keys()))
        ci = _row_to_company(
            {str(k): ("" if row[k] is None else str(row[k])) for k in row.keys()},
            id_key, name_key, url_key
        )
        if ci:
            yield ci


def _iter_excel(path: Path, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    pd = _require_pandas()
    # Read first sheet only (multi-sheet: user can split or extend here)
    df = pd.read_excel(path, dtype=str)  # requires openpyxl/xlrd depending on file
    yield from _iter_pandas_df(df, limit=limit)


def _iter_stata(path: Path, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    pd = _require_pandas()
    df = pd.read_stata(path)  # dtype inference; coercion to str in iterator
    yield from _iter_pandas_df(df, limit=limit)


def _iter_parquet(path: Path, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    pd = _require_pandas()
    df = pd.read_parquet(path)  # needs pyarrow/fastparquet
    yield from _iter_pandas_df(df, limit=limit)


def _iter_feather(path: Path, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    pd = _require_pandas()
    df = pd.read_feather(path)  # needs pyarrow
    yield from _iter_pandas_df(df, limit=limit)


def _iter_sas7bdat(path: Path, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    pd = _require_pandas()
    df = pd.read_sas(path, format="sas7bdat")  # needs sas7bdat library via pandas
    yield from _iter_pandas_df(df, limit=limit)


def _iter_spss(path: Path, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    pd = _require_pandas()
    df = pd.read_spss(path)  # requires pyreadstat
    yield from _iter_pandas_df(df, limit=limit)


# -----------------------------
# Dispatcher
# -----------------------------
_SUPPORTED_EXT_HANDLERS = {
    ".csv":   lambda p, enc, lim: _iter_csv(p, encoding=enc, delimiter=",", limit=lim),
    ".tsv":   lambda p, enc, lim: _iter_csv(p, encoding=enc, delimiter="\t", limit=lim),
    ".txt":   lambda p, enc, lim: _iter_csv(p, encoding=enc, delimiter=",", limit=lim),  # permissive
    ".json":  lambda p, enc, lim: _iter_json_like(p, encoding=enc, limit=lim),
    ".jsonl": lambda p, enc, lim: _iter_json_like(p, encoding=enc, limit=lim),
    ".ndjson":lambda p, enc, lim: _iter_json_like(p, encoding=enc, limit=lim),
    ".xlsx":  lambda p, enc, lim: _iter_excel(p, limit=lim),
    ".xls":   lambda p, enc, lim: _iter_excel(p, limit=lim),
    ".dta":   lambda p, enc, lim: _iter_stata(p, limit=lim),
    ".parquet": lambda p, enc, lim: _iter_parquet(p, limit=lim),
    ".feather": lambda p, enc, lim: _iter_feather(p, limit=lim),
    ".sas7bdat": lambda p, enc, lim: _iter_sas7bdat(p, limit=lim),
    ".sav":   lambda p, enc, lim: _iter_spss(p, limit=lim),
}


def _ext(path: Path) -> str:
    return path.suffix.lower()


def _gather_files(root: Path, patterns: Sequence[str], recursive: bool = True) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files: List[Path] = []
    it = root.rglob if recursive else root.glob
    for pat in patterns:
        files.extend([p for p in it(pat) if p.is_file()])
    # Stable order
    return sorted(set(files))


def _dedupe_records(records: Iterable[CompanyInput], *, keys: Tuple[str, str] = ("bvdid", "url")) -> List[CompanyInput]:
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


# -----------------------------
# Public API
# -----------------------------
DEFAULT_SOURCE_PATTERNS = (
    "*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav"
)


def load_companies_from_source_file(
    path: Path,
    *,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> List[CompanyInput]:
    path = Path(path)
    ext = _ext(path)
    handler = _SUPPORTED_EXT_HANDLERS.get(ext)
    if not handler:
        raise ValueError(f"Unsupported source file extension: {ext} ({path.name})")
    return list(handler(path, encoding, limit))


def load_companies_from_dir(
    root: Path,
    *,
    patterns: Sequence[str] | None = None,
    recursive: bool = True,
    encoding: str = "utf-8",
    limit_per_file: Optional[int] = None,
    dedupe: bool = True,
    dedupe_on: Tuple[str, str] = ("bvdid", "url"),
) -> List[CompanyInput]:
    pats = patterns or [p.strip() for p in DEFAULT_SOURCE_PATTERNS.split(",")]
    files = _gather_files(root, pats, recursive=recursive)
    recs: List[CompanyInput] = []
    for f in files:
        try:
            recs.extend(load_companies_from_source_file(f, encoding=encoding, limit=limit_per_file))
        except ValueError:
            # Unsupported file type that matched the glob; skip
            continue
    if dedupe:
        recs = _dedupe_records(recs, keys=dedupe_on)
    return recs


def load_companies_from_source(
    path_or_dir: Path,
    *,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> List[CompanyInput]:
    """
    Backward-friendly single entry point:
      - If a directory: loads from all supported files (recursive), no per-file limit.
      - If a file: loads from that file.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        return load_companies_from_dir(p, encoding=encoding, limit_per_file=limit, recursive=True)
    return load_companies_from_source_file(p, encoding=encoding, limit=limit)