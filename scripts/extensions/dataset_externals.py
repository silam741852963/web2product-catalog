from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set
from urllib.parse import urlparse
import re
import pandas as pd


# Two-level public suffixes that require one extra label to form the registrable domain
_TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.jp", "ne.jp", "or.jp",
    "com.cn", "com.sg", "com.br",
}


def _registrable_domain(host: str) -> str:
    """
    Compute a lightweight registrable domain:
      - For normal TLDs: example.com  -> example.com
      - For 2LD suffixes: foo.example.co.uk -> example.co.uk
    """
    h = (host or "").lower().strip(".")
    if not h:
        return ""
    labels = h.split(".")
    if len(labels) < 2:
        return h

    last2 = ".".join(labels[-2:])           # e.g. "example.com" or "co.uk"
    if last2 in _TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])        # e.g. "example.co.uk"
    return last2                             # e.g. "example.com"


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _host_from_url(u: str) -> str:
    """
    Robustly extract a host from:
      - Full URLs (with scheme)
      - Scheme-less URLs like "example.com/path"
      - Protocol-relative URLs like "//example.com"
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
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _iter_urls_from_df(df: pd.DataFrame, column_name: str = "url") -> Iterable[str]:
    if df is None or df.empty:
        return []
    # Case-insensitive lookup of the 'url' column
    colmap = {c.lower(): c for c in df.columns}
    if column_name.lower() not in colmap:
        return []
    col = colmap[column_name.lower()]
    # Cast to string, strip, drop empties
    series = df[col].astype(str).map(lambda x: x.strip()).replace("", pd.NA).dropna()
    return series.tolist()


def _read_table(path: Path) -> pd.DataFrame:
    """
    Best-effort reader for many common formats. Returns a DataFrame or empty DF on failure.
    """
    try:
        ext = path.suffix.lower()
        if ext in {".csv", ".txt"}:
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path, dtype=str)
        if ext in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True, dtype=str)
        if ext == ".json":
            # Try NDJSON first, then array/object JSON
            try:
                return pd.read_json(path, lines=True, dtype=str)
            except Exception:
                return pd.read_json(path, dtype=str)
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".feather":
            return pd.read_feather(path)
        if ext == ".dta":
            return pd.read_stata(path)
        if ext == ".sas7bdat":
            return pd.read_sas(path)
        if ext == ".sav":
            return pd.read_spss(path)

        # Unknown: try CSV as a fallback
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception:
        # Return empty DataFrame on any read error
        return pd.DataFrame()


def _gather_urls_from_file(path: Path, *, column_name: str = "url") -> List[str]:
    df = _read_table(path)
    return list(_iter_urls_from_df(df, column_name=column_name))


def _gather_urls_from_dir(
    directory: Path,
    patterns: List[str],
    *,
    column_name: str = "url",
    recursive: bool = True,
) -> List[str]:
    urls: List[str] = []
    for pat in patterns or ["*.csv"]:
        globber = directory.rglob if recursive else directory.glob
        for file in globber(pat):
            if not file.is_file():
                continue
            urls.extend(_gather_urls_from_file(file, column_name=column_name))
    return urls


@dataclass
class DatasetExternals:
    hosts: Set[str]
    registrable_domains: Set[str]

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
        Kept for compatibility when callers already have CompanyInput objects.
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
        source: Optional[Path] = None,
        source_dir: Optional[Path] = None,
        pattern: str = "*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav",
        limit: Optional[int] = None,
        column_name: str = "url",
    ) -> "DatasetExternals":
        """
        Aggregate and deduplicate the 'url' column directly from the provided source(s).
        - If source_dir is provided, search files by pattern(s).
        - Else if source is provided, read that file.
        """
        urls: List[str] = []

        if source_dir:
            pats = [p.strip() for p in (pattern or "").split(",") if p.strip()]
            urls = _gather_urls_from_dir(Path(source_dir), pats, column_name=column_name, recursive=True)
        elif source:
            urls = _gather_urls_from_file(Path(source), column_name=column_name)
        else:
            # Nothing provided — return empty externals
            return cls(hosts=set(), registrable_domains=set())

        # Apply optional global limit to number of URL rows considered (pre-dedup)
        if isinstance(limit, int) and limit > 0:
            urls = urls[:limit]

        # Build sets
        return cls.from_urls(urls)