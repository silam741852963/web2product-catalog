from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
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

from configs.models import Company

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =========================================================================== #
# Model
# =========================================================================== #
#
# Canonical object produced by this module: configs.models.Company
#
# - Required:
#     company_id (bvdid / hojin_id / id)
#     root_url   (url / website / homepage / home_page)
# - Optional:
#     name
# - Everything else goes into Company.metadata (I/O stringly typed),
#   with deterministic normalization for industry/nace to satisfy Company.from_input.
# =========================================================================== #


# =========================================================================== #
# Industry enrichment (optional)
# =========================================================================== #

DEFAULT_NACE_INDUSTRY_PATH = Path("data/industry/nace.ods")
DEFAULT_INDUSTRY_FALLBACK_PATH = Path("data/industry/industry.ods")


@dataclass(frozen=True, slots=True)
class IndustryEnrichmentConfig:
    """
    Configuration for industry label enrichment.

    Contract (deterministic):
      - If enabled is True, the configured files must exist and be readable.
      - Any violation raises RuntimeError with clear instructions.
    """

    nace_path: Optional[Path] = None
    fallback_path: Optional[Path] = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class IndustryLookupTables:
    """
    Resolved mappings used to attach labels onto company records.

    - nace_key_to_label: (industry:int, nace:int) -> label(str)
    - nace_to_label:     nace(int) -> label(str)         (first-seen per nace)
    - industry_to_label: industry(int) -> label(str)
    """

    nace_key_to_label: Dict[Tuple[int, int], str]
    nace_to_label: Dict[int, str]
    industry_to_label: Dict[int, str]


# --------------------------------------------------------------------------- #
# Robust int parsing (supports "10.0", " 2711 ", etc.)
# --------------------------------------------------------------------------- #


def _is_nan_str(s: str) -> bool:
    sl = s.strip().lower()
    return sl in ("nan", "na", "n/a", "none", "null", "")


def _to_int(v: object) -> Optional[int]:
    """
    Accepts:
      - int
      - float (including nan)
      - "10", "10.0", " 10.000 ", "-3.0"
    Returns:
      int or None
    """
    if v is None:
        return None

    if isinstance(v, bool):
        return int(v)

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return int(v)

    s = str(v).strip()
    if _is_nan_str(s):
        return None

    if re.match(r"^-?\d+$", s):
        return int(s)

    if re.match(r"^-?\d+(\.\d+)?$", s):
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)

    return None


def _norm_col(c: str) -> str:
    return (c or "").strip().lower()


def _log_table_preview(
    *,
    tag: str,
    path: Path,
    columns: Sequence[str],
    rows_preview: Sequence[Mapping[str, object]],
    max_rows: int = 3,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    cols = [str(c) for c in columns]
    logger.debug("[%s] table=%s columns=%s", tag, str(path), cols)
    for i, r in enumerate(rows_preview[:max_rows]):
        try:
            logger.debug("[%s] table=%s row[%d]=%s", tag, str(path), i, dict(r))
        except Exception:
            logger.debug("[%s] table=%s row[%d]=<unprintable>", tag, str(path), i)


# --------------------------------------------------------------------------- #
# Strict table reading
# --------------------------------------------------------------------------- #


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pandas is required to read non-ODS industry files. "
            "Install: pip install pandas openpyxl pyarrow"
        ) from e
    return pd


def _require_odfpy():
    try:
        from odf.opendocument import load as odf_load  # type: ignore
        from odf.table import Table, TableCell, TableRow  # type: ignore
        from odf.text import P  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "odfpy is required to read .ods files deterministically. "
            "Install: pip install odfpy"
        ) from e
    return odf_load, Table, TableRow, TableCell, P


def _cell_text(cell, P) -> str:
    parts: List[str] = []
    for p in cell.getElementsByType(P):
        for n in getattr(p, "childNodes", []):
            if getattr(n, "nodeType", None) == 3:  # TEXT_NODE
                parts.append(str(getattr(n, "data", "")))
            else:
                d = getattr(n, "data", None)
                if d is not None:
                    parts.append(str(d))
    return " ".join(" ".join(parts).split()).strip()


def _read_ods_as_rows(path: Path) -> List[List[str]]:
    odf_load, Table, TableRow, TableCell, P = _require_odfpy()

    doc = odf_load(str(path))
    sheets = doc.spreadsheet.getElementsByType(Table)
    if not sheets:
        raise RuntimeError(f"ODS has no sheets: {path}")

    sheet = sheets[0]
    sheet_name = getattr(sheet, "getAttribute", lambda *_: None)("name")
    logger.debug(
        "[load_source] ods_open path=%s sheets=%d using_sheet=%s",
        str(path),
        len(sheets),
        sheet_name,
    )

    grid: List[List[str]] = []
    for row in sheet.getElementsByType(TableRow):
        row_repeat = int(row.getAttribute("numberrowsrepeated") or 1)
        row_cells: List[str] = []

        for cell in row.getElementsByType(TableCell):
            cell_repeat = int(cell.getAttribute("numbercolumnsrepeated") or 1)
            txt = _cell_text(cell, P)
            for _ in range(cell_repeat):
                row_cells.append(txt)

        while row_cells and (row_cells[-1] == "" or _is_nan_str(row_cells[-1])):
            row_cells.pop()

        for _ in range(row_repeat):
            grid.append(list(row_cells))

    while grid and (not grid[-1] or all((c == "" or _is_nan_str(c)) for c in grid[-1])):
        grid.pop()

    return grid


def _ods_grid_to_dicts(path: Path) -> List[Dict[str, str]]:
    grid = _read_ods_as_rows(path)
    if not grid:
        raise RuntimeError(f"ODS is empty: {path}")

    header_idx = None
    for i, r in enumerate(grid):
        if r and any((c.strip() != "" and not _is_nan_str(c)) for c in r):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"ODS has no header row: {path}")

    header_raw = [str(c).strip() for c in grid[header_idx]]
    header: List[str] = []
    keep_indices: List[int] = []
    for j, h in enumerate(header_raw):
        if h and not _is_nan_str(h):
            header.append(h)
            keep_indices.append(j)

    if not header:
        raise RuntimeError(f"ODS header row has no usable columns: {path}")

    rows: List[Dict[str, str]] = []
    for r in grid[header_idx + 1 :]:
        if not r:
            continue
        row_dict: Dict[str, str] = {}
        for h, j in zip(header, keep_indices):
            v = r[j] if j < len(r) else ""
            row_dict[h] = "" if v is None else str(v).strip()
        if all((vv == "" or _is_nan_str(vv)) for vv in row_dict.values()):
            continue
        rows.append(row_dict)

    if logger.isEnabledFor(logging.DEBUG):
        _log_table_preview(
            tag="load_source.ods",
            path=path,
            columns=header,
            rows_preview=rows[:3],
        )

    return rows


def _read_table_strict(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    p = Path(path)

    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Industry table missing or not a file: {p}")

    suf = p.suffix.lower()
    stat = p.stat()
    logger.debug(
        "[load_source] read_table path=%s suffix=%s size=%d",
        str(p),
        suf,
        stat.st_size,
    )

    if suf == ".ods":
        rows = _ods_grid_to_dicts(p)
        cols = list(rows[0].keys()) if rows else []
        return cols, rows

    pd = _require_pandas()

    if suf in (".xls", ".xlsx"):
        df = pd.read_excel(p, dtype=str)
    elif suf in (".csv", ".tsv", ".txt"):
        delim = "," if suf != ".tsv" else "\t"
        df = pd.read_csv(p, dtype=str, delimiter=delim)
    elif suf in (".json", ".jsonl", ".ndjson"):
        try:
            df = pd.read_json(p, dtype=str)
        except Exception:
            df = pd.read_json(p, dtype=str, lines=True)
    else:
        df = pd.read_csv(p, dtype=str)

    if df is None or df.empty:
        raise RuntimeError(f"Industry table is empty: {p}")

    cols = [str(c) for c in df.columns]
    rows_out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        d = {str(k): ("" if v is None else str(v)).strip() for k, v in row.items()}
        if all((vv == "" or _is_nan_str(vv)) for vv in d.values()):
            continue
        rows_out.append(d)

    if logger.isEnabledFor(logging.DEBUG):
        _log_table_preview(
            tag="load_source.pandas",
            path=p,
            columns=cols,
            rows_preview=rows_out[:3],
        )

    return cols, rows_out


# --------------------------------------------------------------------------- #
# Interfaces (signatures) for run.py and prompting/flags generation
# --------------------------------------------------------------------------- #


def load_industry_lookup_tables(
    config: IndustryEnrichmentConfig,
) -> Optional[IndustryLookupTables]:
    if not config.enabled:
        logger.info("[load_source] industry enrichment disabled")
        return None

    nace_path = (
        config.nace_path if config.nace_path is not None else DEFAULT_NACE_INDUSTRY_PATH
    )
    fallback_path = (
        config.fallback_path
        if config.fallback_path is not None
        else DEFAULT_INDUSTRY_FALLBACK_PATH
    )

    logger.info(
        "[load_source] industry enrichment enabled: nace_path=%s fallback_path=%s",
        str(nace_path),
        str(fallback_path),
    )

    nace_key_to_label: Dict[Tuple[int, int], str] = {}
    nace_to_label: Dict[int, str] = {}
    industry_to_label: Dict[int, str] = {}

    # ---- Primary: nace.ods (industry:int, nace:int, label:str)
    nace_cols, nace_rows = _read_table_strict(nace_path)
    nace_colmap = {_norm_col(c): c for c in nace_cols}

    if not (
        "industry" in nace_colmap and "nace" in nace_colmap and "label" in nace_colmap
    ):
        raise RuntimeError(
            f"Misformatted NACE table {nace_path}. Expected columns: industry,nace,label "
            f"(case-insensitive). Got: {nace_cols}"
        )

    c_ind = nace_colmap["industry"]
    c_nace = nace_colmap["nace"]
    c_label = nace_colmap["label"]

    read_rows = 0
    used_rows = 0
    skipped_rows = 0

    for row in nace_rows:
        read_rows += 1
        ind = _to_int(row.get(c_ind))
        nace = _to_int(row.get(c_nace))
        label = (row.get(c_label) or "").strip()

        if nace is None or not label:
            skipped_rows += 1
            continue

        if ind is not None:
            nace_key_to_label.setdefault((ind, nace), label)
        nace_to_label.setdefault(nace, label)
        used_rows += 1

    logger.info(
        "[load_source] nace_table_loaded rows_read=%d rows_used=%d rows_skipped=%d unique_keys=%d unique_nace=%d",
        read_rows,
        used_rows,
        skipped_rows,
        len(nace_key_to_label),
        len(nace_to_label),
    )

    # ---- Fallback: industry.ods (industry:int, label:str)
    fb_cols, fb_rows = _read_table_strict(fallback_path)
    fb_colmap = {_norm_col(c): c for c in fb_cols}

    if not ("industry" in fb_colmap and "label" in fb_colmap):
        raise RuntimeError(
            f"Misformatted fallback industry table {fallback_path}. Expected columns: industry,label "
            f"(case-insensitive). Got: {fb_cols}"
        )

    c_fbind = fb_colmap["industry"]
    c_fblabel = fb_colmap["label"]

    fb_read = 0
    fb_used = 0
    fb_skipped = 0

    for row in fb_rows:
        fb_read += 1
        ind = _to_int(row.get(c_fbind))
        label = (row.get(c_fblabel) or "").strip()

        if ind is None or not label:
            fb_skipped += 1
            continue

        industry_to_label.setdefault(ind, label)
        fb_used += 1

    logger.info(
        "[load_source] fallback_table_loaded rows_read=%d rows_used=%d rows_skipped=%d unique_industry=%d",
        fb_read,
        fb_used,
        fb_skipped,
        len(industry_to_label),
    )

    if not nace_key_to_label and not nace_to_label and not industry_to_label:
        raise RuntimeError(
            "Industry enrichment enabled, but no usable mappings were loaded from the provided tables."
        )

    return IndustryLookupTables(
        nace_key_to_label=nace_key_to_label,
        nace_to_label=nace_to_label,
        industry_to_label=industry_to_label,
    )


def resolve_industry_label(
    *,
    industry: Optional[int],
    nace: Optional[int],
    tables: IndustryLookupTables,
) -> Tuple[Optional[str], Optional[str]]:
    if industry is not None and nace is not None:
        lbl = tables.nace_key_to_label.get((industry, nace))
        if lbl:
            return lbl, "industry+nace"

    if nace is not None:
        lbl = tables.nace_to_label.get(nace)
        if lbl:
            return lbl, "nace_only"

    if industry is not None:
        lbl = tables.industry_to_label.get(industry)
        if lbl:
            return lbl, "industry_only"

    return None, None


def enrich_companies_with_industry_labels(
    companies: Iterable[Company],
    tables: Optional[IndustryLookupTables],
) -> List[Company]:
    """
    Adds deterministic industry label to Company.metadata + canonical fields
    via Company.from_input() (so downstream uses Company.industry_label*).
    """
    if tables is None:
        return list(companies)

    out: List[Company] = []
    resolved = 0
    unresolved = 0

    for c in companies:
        md: Dict[str, Any] = dict(c.metadata)

        industry_val = _to_int(md.get("industry"))
        nace_val = _to_int(md.get("nace"))

        label, source = resolve_industry_label(
            industry=industry_val, nace=nace_val, tables=tables
        )

        # Normalize numeric strings to satisfy configs.models._to_int_or_none (strict int()).
        if industry_val is not None:
            md["industry"] = str(industry_val)
        if nace_val is not None:
            md["nace"] = str(nace_val)

        if label and source:
            md["industry_label"] = label
            md["industry_label_source"] = source
            resolved += 1
        else:
            unresolved += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[load_source] enrich company_id=%s industry=%s nace=%s -> label=%s source=%s",
                c.company_id,
                industry_val,
                nace_val,
                label,
                source,
            )

        out.append(
            Company.from_input(
                company_id=c.company_id,
                root_url=c.root_url,
                name=c.name,
                metadata=md,
            )
        )

    logger.info(
        "[load_source] enrichment_summary companies=%d resolved=%d unresolved=%d",
        resolved + unresolved,
        resolved,
        unresolved,
    )
    return out


# =========================================================================== #
# URL normalization helpers
# =========================================================================== #

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
_DROP_PARAM_PREFIXES: Tuple[str, ...] = ("utm_", "mtm_", "pk_")


def _registrable_domain(host: str) -> str:
    h = (host or "").lower().strip(".")
    if not h:
        return ""
    labels = h.split(".")
    if len(labels) < 2:
        return h

    last2 = ".".join(labels[-2:])
    if last2 in _TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last2


def _canonical_host(raw_host: str, *, collapse_www: bool = True) -> str:
    h = (raw_host or "").strip().lower().strip(".")
    if not h:
        return ""
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
    if p_low in ("/index.html", "/index.htm", "/home", "/home/"):
        return "/"
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


def canonical_url(
    u: str, *, default_scheme: str = "https", collapse_www: bool = True
) -> str:
    s = (u or "").strip()
    if not s:
        return ""

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

    port = pu.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = host
    else:
        netloc = host if port is None else f"{host}:{port}"

    path = _normalize_home_path(pu.path or "/")
    query = _strip_tracking_params(pu.query or "")
    fragment = ""
    parts = (scheme, netloc, path, "", query, fragment)
    return urlunparse(parts)


# =========================================================================== #
# Key resolution / normalization
# =========================================================================== #

_PRIMARY_ID_KEYS = ("bvdid", "hojin_id", "id")
_PRIMARY_NAME_KEYS = ("name", "company_name", "company")
_PRIMARY_URL_KEYS = ("url", "website", "homepage", "home_page")


def _resolve_keys(fieldnames: Optional[Sequence[str]]) -> Tuple[str, str, str]:
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


def _normalize_company_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic normalization for metadata keys that must be strict-int
    for configs.models.Company.from_input().

    - Converts industry/nace to int-string if parseable (supports "10.0").
    - Leaves other keys untouched (stringly typed I/O by default).
    """
    out = dict(md)

    ind = _to_int(out.get("industry"))
    if ind is not None:
        out["industry"] = str(ind)

    nace = _to_int(out.get("nace"))
    if nace is not None:
        out["nace"] = str(nace)

    return out


def _row_to_company(
    row: Mapping[str, str],
    id_key: str,
    name_key: str,
    url_key: str,
) -> Optional[Company]:
    id_val = (row.get(id_key, "") or "").strip() if id_key else ""
    nm_val = (row.get(name_key, "") or "").strip() if name_key else ""
    url_val_raw = (row.get(url_key, "") or "").strip() if url_key else ""

    row_l = {
        (k or "").strip().lower(): ("" if v is None else str(v)) for k, v in row.items()
    }

    if not id_val:
        for k in _PRIMARY_ID_KEYS:
            vv = row_l.get(k, "")
            if vv.strip():
                id_val = vv.strip()
                break
    if not nm_val:
        for k in _PRIMARY_NAME_KEYS:
            vv = row_l.get(k, "")
            if vv.strip():
                nm_val = vv.strip()
                break
    if not url_val_raw:
        for k in _PRIMARY_URL_KEYS:
            vv = row_l.get(k, "")
            if vv.strip():
                url_val_raw = vv.strip()
                break

    if not id_val or not nm_val or not url_val_raw:
        return None

    url_val = canonical_url(url_val_raw)
    if not url_val:
        url_val = canonical_url(f"https://{url_val_raw.lstrip('/')}")
    if not url_val:
        return None

    exclude = {
        (id_key or "").lower(),
        (name_key or "").lower(),
        (url_key or "").lower(),
    }

    metadata: Dict[str, Any] = {
        str(k): ("" if v is None else str(v)).strip()
        for k, v in row.items()
        if (k or "").lower() not in exclude
    }

    metadata.setdefault("source_url_raw", url_val_raw)
    metadata.setdefault("normalized_url", url_val)

    metadata = _normalize_company_metadata(metadata)

    return Company.from_input(
        company_id=id_val,
        root_url=url_val,
        name=nm_val,
        metadata=metadata,
    )


def _iter_pandas_df(df, *, limit: Optional[int]) -> Iterator[Company]:
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
    @property
    def name(self) -> str:  # pragma: no cover
        ...

    @property
    def extensions(self) -> Tuple[str, ...]:  # pragma: no cover
        ...

    def iter_companies(
        self,
        path: Path,
        *,
        encoding: str,
        limit: Optional[int],
    ) -> Iterator[Company]:  # pragma: no cover
        ...


# =========================================================================== #
# Concrete plugins
# =========================================================================== #


@dataclass(frozen=True)
class CsvLikePlugin(SourceFormatPlugin):
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
    ) -> Iterator[Company]:
        count = 0
        with path.open("r", encoding=encoding, newline="") as f:
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
    ) -> Iterator[Company]:
        text = path.read_text(encoding=encoding, errors="ignore").strip()
        rows: List[Dict[str, Any]] = []

        def as_dicts(obj) -> List[Dict[str, Any]]:
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                for k in ("data", "records", "items", "rows"):
                    v = obj.get(k)
                    if isinstance(v, list):
                        return [x for x in v if isinstance(x, dict)]
                return [obj]
            return []

        try:
            rows = as_dicts(json.loads(text))
        except Exception:
            rows = []

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
        encoding: str,  # unused
        limit: Optional[int],
    ) -> Iterator[Company]:
        pd = _require_pandas()
        df = pd.read_excel(path, dtype=str)
        yield from _iter_pandas_df(df, limit=limit)


@dataclass(frozen=True)
class OdsPlugin(SourceFormatPlugin):
    """
    ODS plugin (.ods) backed by pandas for company sources.
    For industry enrichment, ODS is read using odfpy via _read_table_strict.
    """

    _name: str = "ods"
    _extensions: Tuple[str, ...] = (".ods",)

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
    ) -> Iterator[Company]:
        pd = _require_pandas()
        try:
            df = pd.read_excel(path, dtype=str, engine="odf")
        except Exception as e:
            raise RuntimeError(
                f"Failed to read ODS company source via pandas+odf engine: {path}. "
                f"Install: pip install odfpy. Underlying error: {e}"
            ) from e
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
    ) -> Iterator[Company]:
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
    ) -> Iterator[Company]:
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
    ) -> Iterator[Company]:
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
    ) -> Iterator[Company]:
        pd = _require_pandas()
        df = pd.read_spss(path)
        yield from _iter_pandas_df(df, limit=limit)


# =========================================================================== #
# Plugin registry
# =========================================================================== #


def _build_plugin_registry() -> Dict[str, SourceFormatPlugin]:
    csv_like = CsvLikePlugin(
        _name="csv_like",
        _extensions=(".csv", ".tsv", ".txt"),
        _delimiter_map={".csv": ",", ".tsv": "\t", ".txt": ","},
    )
    json_like = JsonLikePlugin(
        _name="json_like", _extensions=(".json", ".jsonl", ".ndjson")
    )
    excel = ExcelPlugin()
    ods = OdsPlugin()
    stata = StataPlugin()
    parquet = ParquetPlugin()
    sas = SasPlugin()
    spss = SpssPlugin()

    plugins: List[SourceFormatPlugin] = [
        csv_like,
        json_like,
        excel,
        ods,
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
                    "[load_source] extension %s already registered to %s; overriding with %s",
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


def _gather_files(
    root: Path, patterns: Sequence[str], recursive: bool = True
) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files: List[Path] = []
    it = root.rglob if recursive else root.glob
    for pat in patterns:
        files.extend([p for p in it(pat) if p.is_file()])
    return sorted(set(files))


def _dedupe_records(
    records: Iterable[Company],
    *,
    keys: Tuple[str, str] = ("company_id", "root_url"),
) -> List[Company]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Company] = []
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


def _aggregate_by_canonical_url(records: Iterable[Company]) -> List[Company]:
    buckets: Dict[str, List[Company]] = {}
    for r in records:
        cu = canonical_url(r.root_url)
        if not cu:
            continue
        rr = Company.from_input(
            company_id=r.company_id,
            root_url=cu,
            name=r.name,
            metadata=dict(r.metadata),
        )
        rr.metadata.setdefault("normalized_url", cu)
        buckets.setdefault(cu, []).append(rr)

    merged: List[Company] = []
    for cu, items in buckets.items():
        if not items:
            continue

        primary = items[0]

        ids: List[str] = []
        names: List[str] = []

        meta_merged: Dict[str, Any] = dict(primary.metadata)

        chosen_industry: Optional[int] = primary.industry
        chosen_nace: Optional[int] = primary.nace
        chosen_label: Optional[str] = primary.industry_label
        chosen_label_source: Optional[str] = primary.industry_label_source

        for it in items:
            if it.company_id and it.company_id not in ids:
                ids.append(it.company_id)
            if it.name and it.name not in names:
                names.append(it.name)

            if chosen_industry is None and it.industry is not None:
                chosen_industry = it.industry
            if chosen_nace is None and it.nace is not None:
                chosen_nace = it.nace
            if chosen_label is None and it.industry_label:
                chosen_label = it.industry_label
                chosen_label_source = it.industry_label_source

        if not ids and primary.company_id:
            ids.append(primary.company_id)
        if not names and primary.name:
            names.append(primary.name)

        agg_ids = "|".join(ids)
        agg_names = " | ".join(names)

        meta_merged["agg_ids"] = agg_ids
        meta_merged["agg_names"] = agg_names
        meta_merged["agg_count"] = str(len(items))
        meta_merged["canonicalized"] = "1"
        meta_merged["primary_id"] = primary.company_id
        meta_merged["primary_name"] = primary.name or ""

        if chosen_industry is not None:
            meta_merged["industry"] = str(int(chosen_industry))
        if chosen_nace is not None:
            meta_merged["nace"] = str(int(chosen_nace))
        if chosen_label:
            meta_merged["industry_label"] = chosen_label
        if chosen_label_source:
            meta_merged["industry_label_source"] = chosen_label_source

        if len(items) > 1:
            logger.debug(
                "[load_source] merged %d rows into canonical URL %s -> company_id=%s name=%s",
                len(items),
                cu,
                agg_ids,
                agg_names,
            )

        merged.append(
            Company.from_input(
                company_id=agg_ids,
                root_url=cu,
                name=agg_names,
                metadata=_normalize_company_metadata(meta_merged),
            )
        )

    return merged


def _interleave_by_domain(records: List[Company]) -> List[Company]:
    domain_buckets: Dict[str, List[Company]] = {}
    for r in records:
        host = urlparse(r.root_url).hostname or ""
        dom = _registrable_domain(host)
        domain_buckets.setdefault(dom, []).append(r)

    queue_order = sorted(domain_buckets.keys())
    out: List[Company] = []
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

DEFAULT_SOURCE_PATTERNS = "*.csv,*.tsv,*.xlsx,*.xls,*.ods,*.json,*.jsonl,*.ndjson,*.parquet,*.feather,*.dta,*.sas7bdat,*.sav"


def load_companies_from_source_file(
    path: Path,
    *,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> List[Company]:
    path = Path(path)
    ext = _ext(path)
    plugin = _PLUGIN_BY_EXT.get(ext)
    if not plugin:
        raise ValueError(f"Unsupported source file extension: {ext} ({path.name})")
    return list(plugin.iter_companies(path, encoding=encoding, limit=limit))


def _postprocess_companies(
    recs: List[Company],
    *,
    aggregate_same_url: bool = True,
    interleave_domains: bool = True,
    basic_dedupe: bool = True,
) -> List[Company]:
    if basic_dedupe:
        recs = _dedupe_records(recs, keys=("company_id", "root_url"))
    if aggregate_same_url:
        recs = _aggregate_by_canonical_url(recs)
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
    dedupe: bool = True,
    aggregate_same_url: bool = True,
    interleave_domains: bool = True,
) -> List[Company]:
    pats = patterns or [p.strip() for p in DEFAULT_SOURCE_PATTERNS.split(",")]
    files = _gather_files(root, pats, recursive=recursive)

    logger.info(
        "[load_source] discover root=%s files=%d recursive=%s patterns=%s",
        str(root),
        len(files),
        str(recursive),
        ",".join(pats),
    )

    recs: List[Company] = []
    for f in files:
        try:
            before = len(recs)
            recs.extend(
                load_companies_from_source_file(
                    f, encoding=encoding, limit=limit_per_file
                )
            )
            logger.debug(
                "[load_source] loaded file=%s added=%d total=%d",
                str(f),
                len(recs) - before,
                len(recs),
            )
        except ValueError:
            logger.debug("[load_source] skip_unsupported file=%s", str(f))
            continue

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
) -> List[Company]:
    p = Path(path_or_dir)
    if p.is_dir():
        return load_companies_from_dir(
            p,
            encoding=encoding,
            limit_per_file=limit,
            recursive=True,
            dedupe=True,
            aggregate_same_url=aggregate_same_url,
            interleave_domains=interleave_domains,
        )

    recs = load_companies_from_source_file(p, encoding=encoding, limit=limit)
    return _postprocess_companies(
        recs,
        aggregate_same_url=aggregate_same_url,
        interleave_domains=interleave_domains,
        basic_dedupe=True,
    )


def load_companies_from_source_with_industry(
    path_or_dir: Path,
    *,
    industry_config: Optional[IndustryEnrichmentConfig] = None,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
    aggregate_same_url: bool = True,
    interleave_domains: bool = True,
) -> List[Company]:
    """
    Single entry-point for run.py:

      1) Load companies (all formats supported).
      2) Load industry tables.
      3) Enrich each company with industry_label (deterministic).

    Deterministic behavior:
      - If industry_config.enabled is True (default), missing/misformatted/unreadable
        industry tables raise RuntimeError.
      - To skip enrichment, pass IndustryEnrichmentConfig(enabled=False).
    """
    companies = load_companies_from_source(
        path_or_dir,
        encoding=encoding,
        limit=limit,
        aggregate_same_url=aggregate_same_url,
        interleave_domains=interleave_domains,
    )

    cfg = industry_config if industry_config is not None else IndustryEnrichmentConfig()
    tables = load_industry_lookup_tables(cfg)
    if tables is None:
        return companies
    return enrich_companies_with_industry_labels(companies, tables)
