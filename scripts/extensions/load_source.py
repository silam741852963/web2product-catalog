from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =========================================================================== #
# Model
# =========================================================================== #


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


# =========================================================================== #
# URL normalization helpers
# =========================================================================== #

# Two-label public suffixes that require one extra label to form the registrable domain.
# e.g. foo.example.co.uk -> example.co.uk
_TWO_LEVEL_SUFFIXES: Set[str] = {
    "co.uk",
    "org.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.jp",
    "ne.jp",
    "or.jp",
    "com.cn",
    "com.sg",
    "com.br",
}

# Tracking params to drop outright; any key that matches these
_DROP_PARAMS: Set[str] = {
    "gclid",
    "fbclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "smid",
    "oly_anon_id",
    "oly_enc_id",
    "vero_conv",
    "vero_id",
    "ref",
    "referrer",
    "utm_id",
}
# Prefix-based (e.g., utm_source, utm_medium, utm_campaign, utm_term, utm_content)
_DROP_PARAM_PREFIXES: Tuple[str, ...] = ("utm_", "mtm_", "pk_")


def _registrable_domain(host: str) -> str:
    """
    Lightweight eTLD+1: return the registrable apex (example.co.uk, example.com, etc.).

    For a host like:
      - "example.com"        -> "example.com"
      - "foo.example.com"    -> "example.com"
      - "foo.example.co.uk"  -> "example.co.uk"
    """
    h = (host or "").lower().strip(".")
    if not h:
        return ""
    labels = h.split(".")
    if len(labels) < 2:
        return h

    last2 = ".".join(labels[-2:])
    if last2 in _TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        # If hostname ends with a known 2-label public suffix, registrable domain is last 3 labels.
        return ".".join(labels[-3:])
    return last2


def _canonical_host(raw_host: str, *, collapse_www: bool = True) -> str:
    h = (raw_host or "").strip().lower().strip(".")
    if not h:
        return ""
    # IDNA punycode normalize
    try:
        h = h.encode("idna").decode("ascii")
    except Exception:
        pass
    if collapse_www and h.startswith("www.") and len(h) > 4:
        h = h[4:]
    return h


def _normalize_home_path(path: str) -> str:
    p = (path or "").strip()
    if not p or p == "/":
        return "/"
    p_low = p.lower()
    # normalize common "home" equivalents to root
    if p_low in ("/index.html", "/index.htm", "/home", "/home/"):
        return "/"
    # remove trailing slash except root
    if p.endswith("/") and p != "/":
        p = p[:-1]
    return p or "/"


def _strip_tracking_params(query: str) -> str:
    if not query:
        return ""
    q = []
    for k, v in parse_qsl(query, keep_blank_values=True):
        kl = (k or "").lower()
        if kl in _DROP_PARAMS:
            continue
        if any(kl.startswith(pfx) for pfx in _DROP_PARAM_PREFIXES):
            continue
        q.append((k, v))
    if not q:
        return ""
    return urlencode(q, doseq=True)


def canonical_url(u: str, *, default_scheme: str = "https", collapse_www: bool = True) -> str:
    """
    Canonicalize a company homepage URL so we can aggregate duplicates:
      - ensure scheme (default https)
      - lowercase host, strip www.
      - drop default ports (80 on http, 443 on https)
      - normalize path (/, drop /index.html, trim trailing slash)
      - drop fragment
      - remove common tracking params (utm_*, gclid, fbclid, etc.)
    """
    s = (u or "").strip()
    if not s:
        return ""
    # If missing scheme, prefix default_scheme://
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", s):
        s = f"{default_scheme}://{s.lstrip('/')}"
    try:
        pu = urlparse(s)
    except Exception:
        return ""
    scheme = (pu.scheme or default_scheme).lower()
    host = _canonical_host(pu.hostname or "", collapse_www=collapse_www)
    if not host:
        return ""
    # drop default ports
    port = pu.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = host
    else:
        netloc = host if port is None else f"{host}:{port}"

    path = _normalize_home_path(pu.path or "/")
    query = _strip_tracking_params(pu.query or "")
    fragment = ""  # drop

    parts = (scheme, netloc, path, "", query, fragment)
    return urlunparse(parts)


# =========================================================================== #
# Key resolution / normalization
# =========================================================================== #

_PRIMARY_ID_KEYS = ("bvdid", "hojin_id", "id")
_PRIMARY_NAME_KEYS = ("name", "company_name", "company")
_PRIMARY_URL_KEYS = ("url", "website", "homepage", "home_page")


def _resolve_keys(fieldnames: Optional[Sequence[str]]) -> Tuple[str, str, str]:
    """
    Pick header names (case-insensitive) for (id, name, url).
    Returns **original-cased** names from the file ('' if not found).
    """
    fset = {(fn or "").strip().lower(): fn for fn in (fieldnames or [])}

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


def _row_to_company(
    row: Mapping[str, str],
    id_key: str,
    name_key: str,
    url_key: str,
) -> Optional[CompanyInput]:
    # Canonical, original-cased view
    id_val = (row.get(id_key, "") or "").strip() if id_key else ""
    nm_val = (row.get(name_key, "") or "").strip() if name_key else ""
    url_val = (row.get(url_key, "") or "").strip() if url_key else ""

    # Lowercased fallback view
    row_l = {(k or "").strip().lower(): (v or "") for k, v in row.items()}

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

    exclude = {(id_key or "").lower(), (name_key or "").lower(), (url_key or "").lower()}
    metadata: Dict[str, str] = {
        str(k): (v or "")
        for k, v in row.items()
        if (k or "").lower() not in exclude
    }
    return CompanyInput(bvdid=id_val, name=nm_val, url=url_val, metadata=metadata)


# =========================================================================== #
# Pandas helper (lazy import)
# =========================================================================== #


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "This file type requires pandas (and possibly pyarrow/fastparquet/openpyxl). "
            "Install extras, e.g.: pip install pandas pyarrow openpyxl"
        ) from e
    return pd


def _iter_pandas_df(df, *, limit: Optional[int]) -> Iterator[CompanyInput]:
    """
    Efficient DataFrame -> CompanyInput iterator.

    Speed-optimized vs. the naive .to_dict(orient='records') approach:
      - Resolve (id, name, url) keys once using df.columns.
      - Iterate with itertuples() to avoid per-row dict creation overhead.
    """
    if df is None or df.empty:
        return iter(())  # type: ignore

    cols: List[str] = [str(c) for c in df.columns]
    id_key, name_key, url_key = _resolve_keys(cols)

    for i, row in enumerate(df.itertuples(index=False, name=None)):
        if limit is not None and i >= limit:
            break
        values = list(row)
        row_dict: Dict[str, str] = {}
        for col, val in zip(cols, values):
            row_dict[col] = "" if val is None else str(val)
        ci = _row_to_company(row_dict, id_key, name_key, url_key)
        if ci:
            yield ci


# =========================================================================== #
# Plugin interface
# =========================================================================== #


class SourceFormatPlugin(Protocol):
    """
    Plugin interface for loading CompanyInput rows from a file.

    Each plugin:
      - declares the file extensions it supports
      - implements iter_companies(path, encoding, limit)
    """

    @property
    def name(self) -> str:  # pragma: no cover - interface
        ...

    @property
    def extensions(self) -> Tuple[str, ...]:  # pragma: no cover - interface
        ...

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:  # pragma: no cover - interface
        ...


# =========================================================================== #
# Concrete plugins
# =========================================================================== #

@dataclass(frozen=True)
class CsvLikePlugin(SourceFormatPlugin):
    """
    CSV / TSV / TXT plugin (no pandas).

    Uses csv.DictReader and resolves (id, name, url) once per file.
    """

    _name: str
    _extensions: Tuple[str, ...]
    _delimiter_map: Mapping[str, str]

    @property
    def name(self) -> str:
        return self._name

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:
        count = 0
        with path.open("r", encoding=encoding, newline="") as f:
            # Pick delimiter from map; default to comma.
            delim = self._delimiter_map.get(path.suffix.lower(), ",")
            reader = csv.DictReader(f, delimiter=delim)
            id_key, name_key, url_key = _resolve_keys(reader.fieldnames)
            for row in reader:
                if limit is not None and count >= limit:
                    break
                ci = _row_to_company(row, id_key, name_key, url_key)
                if ci:
                    yield ci
                    count += 1


@dataclass(frozen=True)
class JsonLikePlugin(SourceFormatPlugin):
    """
    JSON / NDJSON / JSONL plugin (no pandas).

    Speed considerations:
      - Attempt to parse as a single JSON blob first.
      - If that fails (or isn't list/dict-ish), fall back to line-by-line NDJSON.
      - Resolve keys once from the first row.
    """

    _name: str
    _extensions: Tuple[str, ...]

    @property
    def name(self) -> str:
        return self._name

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:
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
            rows = []

        # Fallback: NDJSON / JSONL
        if not rows:
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

        if not rows:
            return iter(())  # type: ignore

        # Determine keys once from the first row
        id_key, name_key, url_key = _resolve_keys(list(rows[0].keys()))
        count = 0
        for row in rows:
            if limit is not None and count >= limit:
                break
            if not isinstance(row, dict):
                continue
            ci = _row_to_company(
                {str(k): "" if v is None else str(v) for k, v in row.items()},
                id_key,
                name_key,
                url_key,
            )
            if ci:
                yield ci
                count += 1


@dataclass(frozen=True)
class ExcelPlugin(SourceFormatPlugin):
    """
    Excel plugin (.xls, .xlsx) backed by pandas.
    """

    _name: str = "excel"
    _extensions: Tuple[str, ...] = (".xls", ".xlsx")

    @property
    def name(self) -> str:
        return self._name

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,  # unused, kept for interface compat
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:
        pd = _require_pandas()
        df = pd.read_excel(path, dtype=str)
        yield from _iter_pandas_df(df, limit=limit)


@dataclass(frozen=True)
class StataPlugin(SourceFormatPlugin):
    _name: str = "stata"
    _extensions: Tuple[str, ...] = (".dta",)

    @property
    def name(self) -> str:
        return self._name

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,  # unused
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:
        pd = _require_pandas()
        df = pd.read_stata(path)
        yield from _iter_pandas_df(df, limit=limit)


@dataclass(frozen=True)
class ParquetPlugin(SourceFormatPlugin):
    _name: str = "parquet"
    _extensions: Tuple[str, ...] = (".parquet", ".feather")

    @property
    def name(self) -> str:
        return self._name

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,  # unused
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:
        pd = _require_pandas()
        if path.suffix.lower() == ".feather":
            df = pd.read_feather(path)
        else:
            df = pd.read_parquet(path)
        yield from _iter_pandas_df(df, limit=limit)


@dataclass(frozen=True)
class SasPlugin(SourceFormatPlugin):
    _name: str = "sas"
    _extensions: Tuple[str, ...] = (".sas7bdat",)

    @property
    def name(self) -> str:
        return self._name

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,  # unused
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:
        pd = _require_pandas()
        df = pd.read_sas(path, format="sas7bdat")
        yield from _iter_pandas_df(df, limit=limit)


@dataclass(frozen=True)
class SpssPlugin(SourceFormatPlugin):
    _name: str = "spss"
    _extensions: Tuple[str, ...] = (".sav",)

    @property
    def name(self) -> str:
        return self._name

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,  # unused
        limit: Optional[int],
    ) -> Iterator[CompanyInput]:
        pd = _require_pandas()
        df = pd.read_spss(path)
        yield from _iter_pandas_df(df, limit=limit)


# =========================================================================== #
# Plugin registry
# =========================================================================== #

def _build_plugin_registry() -> Dict[str, SourceFormatPlugin]:
    """
    Build extension -> plugin map.

    This is the main "plugin hub" for this module. To support a new format:
      1) Implement a SourceFormatPlugin.
      2) Add it here with its supported extensions.
    """
    csv_like = CsvLikePlugin(
        _name="csv_like",
        _extensions=(".csv", ".tsv", ".txt"),
        _delimiter_map={
            ".csv": ",",
            ".tsv": "\t",
            ".txt": ",",
        },
    )
    json_like = JsonLikePlugin(
        _name="json_like",
        _extensions=(".json", ".jsonl", ".ndjson"),
    )
    excel = ExcelPlugin()
    stata = StataPlugin()
    parquet = ParquetPlugin()
    sas = SasPlugin()
    spss = SpssPlugin()

    plugins: List[SourceFormatPlugin] = [
        csv_like,
        json_like,
        excel,
        stata,
        parquet,
        sas,
        spss,
    ]

    registry: Dict[str, SourceFormatPlugin] = {}
    for plugin in plugins:
        for ext in plugin.extensions:
            e = ext.lower()
            if e in registry:
                logger.warning(
                    "[load_source] extension %s already registered to %s; "
                    "overriding with %s",
                    e,
                    registry[e].name,
                    plugin.name,
                )
            registry[e] = plugin
    return registry


_PLUGIN_BY_EXT: Dict[str, SourceFormatPlugin] = _build_plugin_registry()


# =========================================================================== #
# File discovery & basic dedupe
# =========================================================================== #

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


def _dedupe_records(
    records: Iterable[CompanyInput],
    *,
    keys: Tuple[str, str] = ("bvdid", "url"),
) -> List[CompanyInput]:
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


# =========================================================================== #
# Preprocess: aggregate & schedule-aware ordering
# =========================================================================== #

def _aggregate_by_canonical_url(records: Iterable[CompanyInput]) -> List[CompanyInput]:
    """
    Aggregate rows that resolve to the same canonical URL.
    - New behavior:
        * bvdid is the '|' joined list of all unique ids for that URL.
        * name  is the ' | ' joined list of all unique names for that URL.
      This way, the "main" record explicitly reflects all merged companies.
    - For compatibility / traceability, we also store in metadata:
        * agg_ids:    'id1|id2|id3'
        * agg_names:  'n1|n2|n3'
        * agg_count:  N
        * primary_id: original first id
        * primary_name: original first name
    - Rewrites each row URL to its canonical form beforehand.
    """
    buckets: Dict[str, List[CompanyInput]] = {}
    for r in records:
        cu = canonical_url(r.url)
        if not cu:
            # Skip obviously broken URLs
            continue
        # rewrite in-place canonical URL for downstream
        rr = CompanyInput(
            bvdid=r.bvdid,
            name=r.name,
            url=cu,
            metadata=dict(r.metadata),
        )
        rr.metadata.setdefault("normalized_url", cu)
        buckets.setdefault(cu, []).append(rr)

    merged: List[CompanyInput] = []
    for cu, items in buckets.items():
        if not items:
            continue

        primary = items[0]
        ids: List[str] = []
        names: List[str] = []
        meta_merged: Dict[str, str] = dict(primary.metadata)

        for it in items:
            if it.bvdid and it.bvdid not in ids:
                ids.append(it.bvdid)
            if it.name and it.name not in names:
                names.append(it.name)

        # Fallback to primary if no ids/names collected for some reason
        if not ids and primary.bvdid:
            ids.append(primary.bvdid)
        if not names and primary.name:
            names.append(primary.name)

        agg_ids = "|".join(ids)
        agg_names = " | ".join(names)

        meta_merged["agg_ids"] = agg_ids
        meta_merged["agg_names"] = agg_names
        meta_merged["agg_count"] = str(len(items))
        meta_merged["canonicalized"] = "1"
        meta_merged["primary_id"] = primary.bvdid
        meta_merged["primary_name"] = primary.name

        # New "main" identity reflects all merged companies
        new_bvdid = agg_ids
        new_name = agg_names

        if len(items) > 1:
            # Explicitly log merges so you can audit them.
            logger.info(
                "[load_source] merged %d rows into canonical URL %s -> bvdid=%s name=%s",
                len(items),
                cu,
                new_bvdid,
                new_name,
            )

        merged.append(
            CompanyInput(
                bvdid=new_bvdid,
                name=new_name,
                url=cu,
                metadata=meta_merged,
            )
        )
    return merged


def _interleave_by_domain(records: List[CompanyInput]) -> List[CompanyInput]:
    """
    Round-robin by registrable domain to spread load across hosts/CDNs.
    This reduces bursty rate limits when running high concurrency.
    """
    domain_buckets: Dict[str, List[CompanyInput]] = {}
    for r in records:
        host = urlparse(r.url).hostname or ""
        dom = _registrable_domain(host)
        domain_buckets.setdefault(dom, []).append(r)

    # Preserve relative order within each domain bucket
    queue_order = sorted(domain_buckets.keys())
    out: List[CompanyInput] = []
    while True:
        progressed = False
        for d in queue_order:
            lst = domain_buckets.get(d, [])
            if lst:
                out.append(lst.pop(0))
                progressed = True
        if not progressed:
            break
    return out


# =========================================================================== #
# Public API
# =========================================================================== #

DEFAULT_SOURCE_PATTERNS = (
    "*.csv,*.tsv,*.xlsx,*.xls,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav"
)


def load_companies_from_source_file(
    path: Path,
    *,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> List[CompanyInput]:
    """
    Load companies from a single file using the registered format plugins.

    This is the plugin-aware replacement for the old extension table:
      - Detects the plugin based on file suffix.
      - Streams CompanyInput instances from the plugin.
    """
    path = Path(path)
    ext = _ext(path)
    plugin = _PLUGIN_BY_EXT.get(ext)
    if not plugin:
        raise ValueError(f"Unsupported source file extension: {ext} ({path.name})")
    return list(plugin.iter_companies(path, encoding=encoding, limit=limit))


def _postprocess_companies(
    recs: List[CompanyInput],
    *,
    aggregate_same_url: bool = True,
    interleave_domains: bool = True,
    basic_dedupe: bool = True,
) -> List[CompanyInput]:
    # First, dedupe (id,url) exact duplicates coming from multiple files
    if basic_dedupe:
        recs = _dedupe_records(recs, keys=("bvdid", "url"))

    # Aggregate/merge companies that share the same canonical URL
    if aggregate_same_url:
        recs = _aggregate_by_canonical_url(recs)

    # Interleave by registrable domain to spread concurrent hits
    if interleave_domains:
        recs = _interleave_by_domain(recs)

    return recs


def load_companies_from_dir(
    root: Path,
    *,
    patterns: Sequence[str] | None = None,
    recursive: bool = True,
    encoding: str = "utf-8",
    limit_per_file: Optional[int] = None,
    dedupe: bool = True,  # kept for backward compat; now superseded by postprocess
    aggregate_same_url: bool = True,
    interleave_domains: bool = True,
) -> List[CompanyInput]:
    """
    Load companies from all matching files under a directory, using plugins.

    - patterns: glob patterns for filenames (comma-separated DEFAULT_SOURCE_PATTERNS by default).
    - recursive: whether to search recursively.
    - limit_per_file: cap number of rows per file for large sources.
    """
    pats = patterns or [p.strip() for p in DEFAULT_SOURCE_PATTERNS.split(",")]
    files = _gather_files(root, pats, recursive=recursive)
    recs: List[CompanyInput] = []
    for f in files:
        try:
            recs.extend(
                load_companies_from_source_file(
                    f,
                    encoding=encoding,
                    limit=limit_per_file,
                )
            )
        except ValueError:
            # Unsupported file type that matched the glob; skip
            continue

    # Backward compat path (dedupe kept as alias for basic (id,url) dedupe)
    return _postprocess_companies(
        recs,
        aggregate_same_url=aggregate_same_url,
        interleave_domains=interleave_domains,
        basic_dedupe=dedupe,
    )


def load_companies_from_source(
    path_or_dir: Path,
    *,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
    aggregate_same_url: bool = True,
    interleave_domains: bool = True,
) -> List[CompanyInput]:
    """
    Backward-friendly single entry point:
      - If a directory: loads from all supported files (recursive), no per-file limit.
      - If a file: loads from that file.
      - Then performs preprocessing to:
          * canonicalize URLs
          * aggregate duplicates by URL (merge ids/names)
          * interleave by registrable domain
    """
    p = Path(path_or_dir)
    if p.is_dir():
        recs = load_companies_from_dir(
            p,
            encoding=encoding,
            limit_per_file=limit,
            recursive=True,
            dedupe=True,
            aggregate_same_url=aggregate_same_url,
            interleave_domains=interleave_domains,
        )
        return recs

    recs = load_companies_from_source_file(p, encoding=encoding, limit=limit)
    return _postprocess_companies(
        recs,
        aggregate_same_url=aggregate_same_url,
        interleave_domains=interleave_domains,
        basic_dedupe=True,
    )