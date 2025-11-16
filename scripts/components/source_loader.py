from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

logger = logging.getLogger(__name__)

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
# URL normalization helpers
# -----------------------------
# Two-label public suffixes that require one extra label to form the registrable domain.
# e.g. foo.example.co.uk -> example.co.uk
_TWO_LEVEL_SUFFIXES: Set[str] = {
    "co.uk", "org.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.jp", "ne.jp", "or.jp",
    "com.cn", "com.sg", "com.br",
}

# Tracking params to drop outright; any key that matches these
_DROP_PARAMS: Set[str] = {
    "gclid", "fbclid", "igshid", "mc_cid", "mc_eid", "msclkid",
    "smid", "oly_anon_id", "oly_enc_id", "vero_conv", "vero_id",
    "ref", "referrer", "utm_id",
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

    last2 = ".".join(labels[-2:])           # e.g. "example.com" or "co.uk"
    if last2 in _TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        # If hostname ends with a known 2-label public suffix, registrable domain is last 3 labels.
        return ".".join(labels[-3:])        # e.g. "example.co.uk"
    return last2                             # e.g. "example.com"


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


def _row_to_company(row: Dict[str, str], id_key: str, name_key: str, url_key: str) -> Optional[CompanyInput]:
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
    metadata: Dict[str, str] = {k: (v or "") for k, v in row.items() if (k or "").lower() not in exclude}
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
            {str(k): str(v) if v is not None else "" for k, v in row.items()},
            id_key,
            name_key,
            url_key,
        )
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
            id_key,
            name_key,
            url_key,
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
# File discovery & basic dedupe
# -----------------------------
_SUPPORTED_EXT_HANDLERS = {
    ".csv": lambda p, enc, lim: _iter_csv(p, encoding=enc, delimiter=",", limit=lim),
    ".tsv": lambda p, enc, lim: _iter_csv(p, encoding=enc, delimiter="\t", limit=lim),
    ".txt": lambda p, enc, lim: _iter_csv(p, encoding=enc, delimiter=",", limit=lim),  # permissive
    ".json": lambda p, enc, lim: _iter_json_like(p, encoding=enc, limit=lim),
    ".jsonl": lambda p, enc, lim: _iter_json_like(p, encoding=enc, limit=lim),
    ".ndjson": lambda p, enc, lim: _iter_json_like(p, encoding=enc, limit=lim),
    ".xlsx": lambda p, enc, lim: _iter_excel(p, limit=lim),
    ".xls": lambda p, enc, lim: _iter_excel(p, limit=lim),
    ".dta": lambda p, enc, lim: _iter_stata(p, limit=lim),
    ".parquet": lambda p, enc, lim: _iter_parquet(p, limit=lim),
    ".feather": lambda p, enc, lim: _iter_feather(p, limit=lim),
    ".sas7bdat": lambda p, enc, lim: _iter_sas7bdat(p, limit=lim),
    ".sav": lambda p, enc, lim: _iter_spss(p, limit=lim),
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
# Preprocess: aggregate & schedule-aware ordering
# -----------------------------
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
        r = CompanyInput(bvdid=r.bvdid, name=r.name, url=cu, metadata=dict(r.metadata))
        r.metadata.setdefault("normalized_url", cu)
        buckets.setdefault(cu, []).append(r)

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
                "[source_loader] merged %d rows into canonical URL %s -> bvdid=%s name=%s",
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
    pats = patterns or [p.strip() for p in DEFAULT_SOURCE_PATTERNS.split(",")]
    files = _gather_files(root, pats, recursive=recursive)
    recs: List[CompanyInput] = []
    for f in files:
        try:
            recs.extend(load_companies_from_source_file(f, encoding=encoding, limit=limit_per_file))
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