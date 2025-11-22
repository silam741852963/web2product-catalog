from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd


# ---------------------------------------------------------------------------
# Lightweight public suffix handling
# ---------------------------------------------------------------------------

# Two-level public suffixes that require one extra label to form the registrable domain
_TWO_LEVEL_SUFFIXES: Set[str] = {
    "co.uk", "org.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.jp", "ne.jp", "or.jp",
    "com.cn", "com.sg", "com.br",
}


def _registrable_domain(host: str) -> str:
    """
    Compute a lightweight registrable domain:

      - "example.com"           -> "example.com"
      - "foo.example.co.uk"     -> "example.co.uk"
      - "localhost" or single   -> "localhost"
    """
    h = (host or "").lower().strip(".")
    if not h:
        return ""
    labels = h.split(".")
    if len(labels) < 2:
        return h

    last2 = ".".join(labels[-2:])  # e.g. "example.com" or "co.uk"
    if last2 in _TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])  # e.g. "example.co.uk"
    return last2


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _host_from_url(u: str) -> str:
    """
    Robustly extract a host from:

      - Full URLs (with scheme)
      - Scheme-less: "example.com/path"
      - Protocol-relative: "//example.com"
    """
    s = (u or "").strip()
    if not s:
        return ""

    # Add a scheme if missing (handle //example.com and bare domains)
    if not _SCHEME_RE.match(s):
        if s.startswith("//"):
            s = "http:" + s
        else:
            s = "http://" + s

    try:
        host = (urlparse(s).hostname or "").lower().strip(".")
        if host.startswith("www.") and len(host) > 4:
            host = host[4:]
        return host
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# DataFrame / IO helpers (speed-optimized)
# ---------------------------------------------------------------------------

def _iter_urls_from_df(df: pd.DataFrame, column_name: str = "url") -> Iterable[str]:
    if df is None or df.empty:
        return []
    # Case-insensitive lookup of the 'url' column
    colmap = {c.lower(): c for c in df.columns}
    if column_name.lower() not in colmap:
        return []
    col = colmap[column_name.lower()]

    # Cast to string, strip, drop empties
    series = df[col].astype(str).map(lambda x: x.strip())
    series = series.replace("", pd.NA).dropna()
    return series.tolist()


def _read_table(
    path: Path,
    *,
    column_name: str = "url",
    usecols_hint: bool = True,
) -> pd.DataFrame:
    """
    Best-effort reader for many common formats.

    *Speed optimisation*: when possible, only load the `column_name` column.
    """
    ext = path.suffix.lower()
    usecols: Optional[List[str]] = [column_name] if usecols_hint else None

    try:
        if ext in {".csv", ".txt"}:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
                usecols=usecols,
            )

        if ext == ".tsv":
            return pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                na_values=[""],
                usecols=usecols,
            )

        if ext in {".xlsx", ".xls"}:
            # Excel reader ignores unknown usecols silently
            return pd.read_excel(path, dtype=str, usecols=usecols)

        if ext in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True, dtype=str)

        if ext == ".json":
            # Try NDJSON first, then array/object JSON
            try:
                return pd.read_json(path, lines=True, dtype=str)
            except Exception:
                return pd.read_json(path, dtype=str)

        if ext == ".parquet":
            return pd.read_parquet(path, columns=usecols)

        if ext == ".feather":
            return pd.read_feather(path, columns=usecols)

        if ext == ".dta":
            return pd.read_stata(path, columns=usecols)

        if ext == ".sas7bdat":
            # pandas.read_sas has limited column select for some engines
            return pd.read_sas(path)

        if ext == ".sav":
            return pd.read_spss(path)

        # Unknown: try CSV as a fallback
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            usecols=usecols,
        )
    except Exception:
        # Return empty DataFrame on any read error
        return pd.DataFrame()


def _gather_urls_from_file(
    path: Path,
    *,
    column_name: str = "url",
    limit: Optional[int] = None,
) -> List[str]:
    df = _read_table(path, column_name=column_name)
    urls = list(_iter_urls_from_df(df, column_name=column_name))
    if isinstance(limit, int) and limit > 0:
        return urls[:limit]
    return urls


def _gather_urls_from_dir(
    directory: Path,
    patterns: List[str],
    *,
    column_name: str = "url",
    recursive: bool = True,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Gather URLs from many files within a directory.

    *Speed optimisation*: respects a global `limit` across all files so we do
    not read more rows than needed.
    """
    urls: List[str] = []
    remaining: Optional[int] = limit

    for pat in patterns or ["*.csv"]:
        globber = directory.rglob if recursive else directory.glob
        for file in globber(pat):
            if not file.is_file():
                continue

            per_file_limit: Optional[int] = None
            if isinstance(remaining, int) and remaining > 0:
                per_file_limit = remaining

            batch = _gather_urls_from_file(
                file,
                column_name=column_name,
                limit=per_file_limit,
            )
            if not batch:
                continue

            urls.extend(batch)
            if isinstance(remaining, int):
                remaining -= len(batch)
                if remaining <= 0:
                    return urls

    return urls


# ---------------------------------------------------------------------------
# DatasetExternals: plugin-style representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetExternals:
    """
    Dataset-level view of all company URLs in the input corpus.

    - hosts: every normalized host seen in the dataset
    - registrable_domains: normalized registrable domains for those hosts
    """
    hosts: Set[str]
    registrable_domains: Set[str]

    # ------------ construction helpers ------------

    @classmethod
    def from_urls(cls, urls: Iterable[str]) -> "DatasetExternals":
        hosts: Set[str] = set()
        domains: Set[str] = set()

        for raw in urls or []:
            h = _host_from_url(raw)
            if not h:
                continue
            hosts.add(h)
            rd = _registrable_domain(h)
            if rd:
                domains.add(rd)

        return cls(hosts=hosts, registrable_domains=domains)

    @classmethod
    def from_companies(cls, companies: Iterable) -> "DatasetExternals":
        """
        Kept for compatibility when callers already have CompanyInput objects
        with a `.url` attribute.
        """
        hosts: Set[str] = set()
        domains: Set[str] = set()

        for c in companies or []:
            u = getattr(c, "url", "") or ""
            h = _host_from_url(u)
            if not h:
                continue
            hosts.add(h)
            rd = _registrable_domain(h)
            if rd:
                domains.add(rd)

        return cls(hosts=hosts, registrable_domains=domains)

    @classmethod
    def from_sources(
        cls,
        *,
        source: Optional[Path] = None,
        source_dir: Optional[Path] = None,
        pattern: str = "*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav",
        limit: Optional[int] = None,
        column_name: str = "url",
    ) -> "DatasetExternals":
        """
        Aggregate and deduplicate the `column_name` (default "url") directly from
        the provided source(s).

        - If `source_dir` is provided, search files by pattern(s).
        - Else if `source` is provided, read that file.

        The `limit` applies across all rows considered (pre-dedup) to bound IO.
        """
        urls: List[str] = []

        if source_dir:
            pats = [p.strip() for p in (pattern or "").split(",") if p.strip()]
            urls = _gather_urls_from_dir(
                Path(source_dir),
                pats,
                column_name=column_name,
                recursive=True,
                limit=limit,
            )
        elif source:
            urls = _gather_urls_from_file(
                Path(source),
                column_name=column_name,
                limit=limit,
            )
        else:
            # Nothing provided — return empty externals
            return cls(hosts=set(), registrable_domains=set())

        # Build sets from URLs
        return cls.from_urls(urls)


# ---------------------------------------------------------------------------
# Small helper used by filters / plugins
# ---------------------------------------------------------------------------

def registrable_domain_from_url(url: str) -> str:
    """
    Convenience helper for other extensions (e.g. UniversalExternalFilter)
    that want to derive a registrable domain directly from a URL string.
    """
    h = _host_from_url(url)
    return _registrable_domain(h) if h else ""