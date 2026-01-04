from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional deps
try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None  # type: ignore

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # type: ignore


# --------------------------------------------------------------------------- #
# UTC timestamp helper
# --------------------------------------------------------------------------- #


def _utc_iso_z() -> str:
    dt = datetime.now(timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Pricing defaults (USD per 1M tokens)
DEFAULT_INPUT_COST_PER_1M_CACHE_HIT = 0.028
DEFAULT_INPUT_COST_PER_1M_CACHE_MISS = 0.28
DEFAULT_OUTPUT_COST_PER_1M = 0.42

DEFAULT_CACHE_HIT_RATE = 0.30
APPROX_CHARS_PER_TOKEN = 4.0

STATUS_MISSING = "missing_status"
DB_MISSING = "missing_db"
META_MISSING = "missing_crawl_meta"

URL_INDEX_META_KEY = "__meta__"
URL_INDEX_RESERVED_PREFIX = "__"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None and raw.strip() else default


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_json_load(s: Any) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_div(n: float, d: float) -> float:
    try:
        if d in (0, 0.0, None):
            return 0.0
        return float(n) / float(d)
    except Exception:
        return 0.0


def _is_nat_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        if isinstance(x, bool):
            return False
        if isinstance(x, int):
            return True
        if isinstance(x, float):
            return float(x).is_integer()
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return False
            f = float(s)
            return float(f).is_integer()
        return False
    except Exception:
        return False


def _to_int_if_nat(x: Any) -> Any:
    return int(float(x)) if _is_nat_number(x) else x


# --------------------------------------------------------------------------- #
# Token counter (tiktoken if available; otherwise approx)
# --------------------------------------------------------------------------- #


class TokenCounter:
    def __init__(self) -> None:
        self.mode = _env_str("ANALYZE_TOKENIZER", "tiktoken").strip().lower()
        self._enc = None
        self._cache: Dict[str, Tuple[int, int]] = {}  # path -> (mtime_ns, tokens)

        if self.mode == "tiktoken" and tiktoken is not None:
            try:
                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None
                self.mode = "approx"
        else:
            self.mode = "approx"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        return int(len(text) / APPROX_CHARS_PER_TOKEN)

    def count_file(self, path: Path) -> int:
        try:
            st = path.stat()
        except Exception:
            return 0
        key = path.resolve().as_posix()
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))

        cached = self._cache.get(key)
        if cached and cached[0] == mtime_ns:
            return cached[1]

        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""

        tokens = self.count_text(txt)
        self._cache[key] = (mtime_ns, tokens)
        return tokens


# --------------------------------------------------------------------------- #
# URL status heuristics (match extensions.crawl.state.py)
# --------------------------------------------------------------------------- #

_MARKDOWN_COMPLETE_STATUSES = {
    "markdown_saved",
    "markdown_suppressed",
    "markdown_done",
    "md_done",
    "md_saved",
    "saved_markdown",
}

_LLM_COMPLETE_STATUSES = {
    "llm_extracted",
    "llm_extracted_empty",
    "llm_full_extracted",
    "llm_done",
    "extracted",
    "product_saved",
    "products_saved",
    "json_saved",
    "presence_done",
}


def _status_has_any(s: str, needles: Tuple[str, ...]) -> bool:
    return any(n in s for n in needles)


def _classify_url_entry(ent: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Return (markdown_done, llm_done) with the same tolerant heuristics as crawl.state.py.
    """
    status = str(ent.get("status") or "").strip().lower()

    has_md_path = bool(ent.get("markdown_path") or ent.get("md_path"))
    has_llm_artifact = bool(
        ent.get("json_path")
        or ent.get("product_path")
        or ent.get("products_path")
        or ent.get("llm_json_path")
        or ent.get("extraction_path")
    )

    presence_checked = bool(ent.get("presence_checked") or ent.get("presence_done"))
    extracted_flag = bool(ent.get("extracted") or ent.get("llm_extracted"))

    status_md_done = (
        status in _MARKDOWN_COMPLETE_STATUSES
        or _status_has_any(
            status, ("markdown_done", "markdown_saved", "md_done", "md_saved")
        )
        or (
            ("markdown" in status or status == "md")
            and _status_has_any(status, ("done", "saved", "complete", "suppressed"))
        )
    )
    status_llm_done = (
        status in _LLM_COMPLETE_STATUSES
        or _status_has_any(
            status,
            (
                "llm_done",
                "llm_extracted",
                "full_extracted",
                "product_saved",
                "products_saved",
                "json_saved",
            ),
        )
        or (
            ("llm" in status or "extract" in status)
            and _status_has_any(status, ("done", "saved", "complete", "extracted"))
        )
    )

    markdown_done = bool(has_md_path or status_md_done)
    if has_llm_artifact or extracted_flag or status_llm_done or presence_checked:
        markdown_done = True

    llm_done = bool(
        extracted_flag or has_llm_artifact or status_llm_done or presence_checked
    )
    return markdown_done, llm_done


def _is_reserved_url_index_key(k: Any) -> bool:
    try:
        return str(k).startswith(URL_INDEX_RESERVED_PREFIX)
    except Exception:
        return False


def _resolve_maybe(company_dir: Path, p: Optional[str]) -> Optional[Path]:
    if not p or not isinstance(p, str):
        return None
    try:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (company_dir / pp).resolve()
        else:
            pp = pp.resolve()
        return pp
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# HTTP status bucketing
# --------------------------------------------------------------------------- #


def _categorize_status_code(code: Optional[int]) -> Tuple[str, str]:
    if code is None:
        return "other", "None"
    try:
        c = int(code)
    except Exception:
        return "other", str(code)
    if 200 <= c <= 299:
        return "ok", str(c)
    if 300 <= c <= 399:
        return "redirect", str(c)
    if 400 <= c <= 499:
        return "client_error", str(c)
    if 500 <= c <= 599:
        return "server_error", str(c)
    return "other", str(c)


# --------------------------------------------------------------------------- #
# DB helpers (crawl_state.sqlite3)
# --------------------------------------------------------------------------- #


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _db_latest_run(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    if row is None:
        return None
    return dict(row)


def _db_companies(
    conn: sqlite3.Connection, company_ids: Optional[List[str]]
) -> List[Dict[str, Any]]:
    if company_ids:
        placeholders = ",".join("?" for _ in company_ids)
        rows = conn.execute(
            f"SELECT * FROM companies WHERE bvdid IN ({placeholders})",
            tuple(company_ids),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM companies").fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Company profile parsing (outputs/<company_id>/company_profile.json + metadata/company_profile.md)
# --------------------------------------------------------------------------- #


@dataclass
class CompanyProfileStats:
    present: bool = False
    pipeline_version: str = ""
    offerings_total: int = 0
    offerings_products: int = 0
    offerings_services: int = 0
    sources_total: int = 0
    desc_sentences_total: int = 0
    alias_total: int = 0

    embedding_model: str = ""
    embedding_dim: int = 0
    embedding_offerings_vecs: int = 0

    profile_json_tokens: int = 0
    profile_json_bytes: int = 0
    profile_md_tokens: int = 0
    profile_md_bytes: int = 0


def _analyze_company_profile(
    company_dir: Path, tc: TokenCounter
) -> CompanyProfileStats:
    prof = CompanyProfileStats()

    json_path = company_dir / "company_profile.json"
    md_path = company_dir / "metadata" / "company_profile.md"

    if not json_path.exists() and not md_path.exists():
        return prof

    prof.present = True

    if json_path.exists():
        try:
            prof.profile_json_bytes = int(json_path.stat().st_size)
        except Exception:
            prof.profile_json_bytes = 0

        prof.profile_json_tokens = int(tc.count_file(json_path))

        obj = _load_json(json_path)
        if isinstance(obj, dict):
            prof.pipeline_version = str(obj.get("pipeline_version") or "")
            offerings = obj.get("offerings")
            if isinstance(offerings, list):
                prof.offerings_total = int(len(offerings))
                for o in offerings:
                    if not isinstance(o, dict):
                        continue
                    typ = str(o.get("type") or "").strip().lower()
                    if typ == "product":
                        prof.offerings_products += 1
                    elif typ == "service":
                        prof.offerings_services += 1

                    srcs = o.get("sources")
                    if isinstance(srcs, list):
                        prof.sources_total += int(len(srcs))

                    descs = o.get("description")
                    if isinstance(descs, list):
                        prof.desc_sentences_total += int(
                            len([x for x in descs if isinstance(x, str) and x.strip()])
                        )

                    name_field = o.get("name")
                    if isinstance(name_field, list) and len(name_field) >= 2:
                        aliases = name_field[1]
                        if isinstance(aliases, list):
                            prof.alias_total += int(
                                len(
                                    [
                                        x
                                        for x in aliases
                                        if isinstance(x, str) and x.strip()
                                    ]
                                )
                            )

            emb = obj.get("embeddings")
            if isinstance(emb, dict):
                prof.embedding_model = str(emb.get("model") or "")
                try:
                    prof.embedding_dim = int(emb.get("dim") or 0)
                except Exception:
                    prof.embedding_dim = 0
                off_vecs = emb.get("embedding_offerings")
                if isinstance(off_vecs, dict):
                    prof.embedding_offerings_vecs = int(len(off_vecs))

    if md_path.exists():
        try:
            prof.profile_md_bytes = int(md_path.stat().st_size)
        except Exception:
            prof.profile_md_bytes = 0
        prof.profile_md_tokens = int(tc.count_file(md_path))

    return prof


# --------------------------------------------------------------------------- #
# Output model row
# --------------------------------------------------------------------------- #


@dataclass
class CompanyRow:
    company_id: str
    name: str
    root_url: str
    status: str

    # terminal info
    last_error: str
    done_reason: str
    done_at: str

    # industry
    industry: int
    nace: int
    industry_label: str
    industry_source: str

    # db url stats
    urls_total: int
    urls_markdown_done: int
    urls_llm_done: int

    # URL index aggregates
    url_count: int
    url_status_ok: int
    url_status_redirect: int
    url_status_client_error: int
    url_status_server_error: int
    url_status_other: int
    url_error_count: int
    gating_accept_true: int
    gating_accept_false: int
    presence_positive: int
    presence_zero: int
    extracted_positive: int
    extracted_zero: int
    markdown_saved: int
    markdown_suppressed: int
    markdown_other_status: int
    md_words_files: int
    md_words_total: int
    md_words_mean_per_file: float
    md_words_median_per_file: float

    # Token accounting
    md_tokens_all: int
    llm_input_tokens_done: int
    llm_output_tokens_done: int
    llm_done_pages: int
    llm_pending_pages: int
    product_files_total: int
    product_files_used_done: int

    # Cost
    cost_input_usd_expected: float
    cost_input_usd_all_hit: float
    cost_input_usd_all_miss: float
    cost_output_usd: float
    cost_total_usd_expected: float

    # Company profile
    profile_present: bool
    profile_pipeline_version: str
    profile_offerings_total: int
    profile_offerings_products: int
    profile_offerings_services: int
    profile_sources_total: int
    profile_desc_sentences_total: int
    profile_alias_total: int
    profile_embedding_model: str
    profile_embedding_dim: int
    profile_embedding_offerings_vecs: int
    profile_json_tokens: int
    profile_json_bytes: int
    profile_md_tokens: int
    profile_md_bytes: int

    # Debug paths
    _company_dir: str
    _meta_path: str
    _url_index_path: str
    _db_path: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Filesystem discovery / loading
# --------------------------------------------------------------------------- #


def _fallback_sanitize_bvdid(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (s or "").strip())


def _company_dir_for(outputs_root: Path, bvdid: str) -> Path:
    direct = (outputs_root / bvdid).resolve()
    if direct.exists():
        return direct
    try:
        from extensions.io.output_paths import sanitize_bvdid  # type: ignore

        safe = sanitize_bvdid(bvdid)
    except Exception:
        safe = _fallback_sanitize_bvdid(bvdid)
    alt = (outputs_root / safe).resolve()
    return alt


def discover_company_dirs(outputs_root: Path) -> List[Path]:
    if not outputs_root.exists():
        return []
    out: List[Path] = []
    for child in outputs_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "metadata" / "url_index.json").exists():
            out.append(child)
            continue
        if (child / "company_profile.json").exists() or (
            child / "metadata" / "company_profile.md"
        ).exists():
            out.append(child)
            continue
    return sorted(out)


def _load_url_index(company_dir: Path) -> Dict[str, Any]:
    p = company_dir / "metadata" / "url_index.json"
    obj = _load_json(p)
    return obj if isinstance(obj, dict) else {}


def _load_crawl_meta(company_dir: Path) -> Dict[str, Any]:
    p = company_dir / "metadata" / "crawl_meta.json"
    obj = _load_json(p)
    return obj if isinstance(obj, dict) else {}


# --------------------------------------------------------------------------- #
# Row computation
# --------------------------------------------------------------------------- #


def _row_from_db_company(
    outputs_root: Path,
    db_path: Path,
    company: Dict[str, Any],
    tc: TokenCounter,
    *,
    cache_hit_rate: float,
    input_cost_hit_per_1m: float,
    input_cost_miss_per_1m: float,
    output_cost_per_1m: float,
) -> CompanyRow:
    bvdid = str(company.get("bvdid") or "")
    company_dir = _company_dir_for(outputs_root, bvdid)

    url_index = _load_url_index(company_dir)
    crawl_meta = _load_crawl_meta(company_dir)

    name = str(company.get("name") or crawl_meta.get("company_name") or "")
    root_url = str(company.get("root_url") or crawl_meta.get("root_url") or "")

    status = str(company.get("status") or "").strip() or STATUS_MISSING

    last_error = str(company.get("last_error") or "")
    done_reason = str(company.get("done_reason") or "")
    done_at = str(company.get("done_at") or "")

    # Industry from DB (preferred)
    industry = int(company.get("industry") or 0)
    nace = int(company.get("nace") or 0)
    industry_label = (
        str(company.get("industry_label") or "Unclassified").strip() or "Unclassified"
    )
    industry_source = str(company.get("industry_source") or "").strip()

    urls_total = int(company.get("urls_total") or 0)
    urls_markdown_done = int(company.get("urls_markdown_done") or 0)
    urls_llm_done = int(company.get("urls_llm_done") or 0)

    # URL index aggregates
    url_count = 0
    url_status_ok = 0
    url_status_redirect = 0
    url_status_client_error = 0
    url_status_server_error = 0
    url_status_other = 0

    url_error_count = 0
    gating_accept_true = 0
    gating_accept_false = 0
    presence_positive = 0
    presence_zero = 0
    extracted_positive = 0
    extracted_zero = 0

    markdown_saved = 0
    markdown_suppressed = 0
    markdown_other_status = 0

    md_word_vals: List[float] = []

    llm_done_pages = 0
    llm_pending_pages = 0
    md_paths_for_llm_done: List[Path] = []
    product_paths_for_llm_done: List[Path] = []

    if isinstance(url_index, dict) and url_index:
        for url_key, rec0 in url_index.items():
            if _is_reserved_url_index_key(url_key):
                continue

            rec = rec0 if isinstance(rec0, dict) else {}
            url_count += 1

            status_code = rec.get("status_code")
            bucket, _ = _categorize_status_code(status_code)
            if bucket == "ok":
                url_status_ok += 1
            elif bucket == "redirect":
                url_status_redirect += 1
            elif bucket == "client_error":
                url_status_client_error += 1
            elif bucket == "server_error":
                url_status_server_error += 1
            else:
                url_status_other += 1

            err = rec.get("last_error") or rec.get("error") or ""
            if isinstance(err, str) and err.strip():
                url_error_count += 1

            gating_accept = rec.get("gating_accept")
            if gating_accept is True:
                gating_accept_true += 1
            elif gating_accept is False:
                gating_accept_false += 1

            presence_checked = bool(
                rec.get("presence_checked") or rec.get("presence_done")
            )
            presence_val = rec.get("presence")
            is_pos = False
            if presence_val is True:
                is_pos = True
            elif isinstance(presence_val, (int, float)) and float(presence_val) > 0:
                is_pos = True
            elif isinstance(presence_val, str) and presence_val.strip().lower() in {
                "1",
                "true",
                "yes",
                "positive",
            }:
                is_pos = True

            if is_pos:
                presence_positive += 1
            else:
                presence_zero += (
                    1 if presence_checked or True else 1
                )  # keep bucket stable

            exv = rec.get("extracted") or rec.get("llm_extracted")
            if exv is True:
                extracted_positive += 1
            elif isinstance(exv, (int, float)) and float(exv) > 0:
                extracted_positive += 1
            elif isinstance(exv, str) and exv.strip().lower() in {"1", "true", "yes"}:
                extracted_positive += 1
            else:
                extracted_zero += 1

            status_field = str(rec.get("status") or "")
            if status_field == "markdown_saved":
                markdown_saved += 1
            elif status_field == "markdown_suppressed":
                markdown_suppressed += 1
            else:
                markdown_other_status += 1

            md_words = rec.get("md_total_words")
            if md_words is not None:
                try:
                    w = float(md_words)
                except Exception:
                    w = 0.0
                if w >= 0:
                    md_word_vals.append(w)

            md_done, llm_done = _classify_url_entry(rec)
            if llm_done:
                llm_done_pages += 1

                mp = rec.get("markdown_path") or rec.get("md_path")
                mdp = _resolve_maybe(company_dir, mp) if isinstance(mp, str) else None
                if mdp is None:
                    pp0 = (
                        rec.get("product_path")
                        or rec.get("json_path")
                        or rec.get("llm_json_path")
                    )
                    ppp = (
                        _resolve_maybe(company_dir, pp0)
                        if isinstance(pp0, str)
                        else None
                    )
                    if ppp is not None:
                        guess = company_dir / "markdown" / f"{ppp.stem}.md"
                        if guess.exists():
                            mdp = guess.resolve()
                if mdp is not None and mdp.exists():
                    md_paths_for_llm_done.append(mdp)

                pp = (
                    rec.get("product_path")
                    or rec.get("json_path")
                    or rec.get("llm_json_path")
                    or rec.get("products_path")
                    or rec.get("extraction_path")
                )
                pth = _resolve_maybe(company_dir, pp) if isinstance(pp, str) else None
                if pth is not None and pth.exists():
                    product_paths_for_llm_done.append(pth)
            else:
                if md_done:
                    llm_pending_pages += 1

    md_words_files = int(len(md_word_vals))
    md_words_total = int(sum(md_word_vals)) if md_word_vals else 0
    md_words_mean_per_file = (
        float(md_words_total / md_words_files) if md_words_files else 0.0
    )
    md_words_median_per_file = float(np.median(md_word_vals)) if md_word_vals else 0.0

    md_tokens_all = 0
    md_dir = company_dir / "markdown"
    if md_dir.exists() and md_dir.is_dir():
        for p in md_dir.glob("*.md"):
            md_tokens_all += int(tc.count_file(p))

    md_paths_unique = list(
        {p.resolve().as_posix(): p for p in md_paths_for_llm_done}.values()
    )
    prod_paths_unique = list(
        {p.resolve().as_posix(): p for p in product_paths_for_llm_done}.values()
    )

    llm_input_tokens_done = int(sum(tc.count_file(p) for p in md_paths_unique))
    llm_output_tokens_done = int(sum(tc.count_file(p) for p in prod_paths_unique))

    product_dir = company_dir / "product"
    product_files_total = (
        int(len(list(product_dir.glob("*.json")))) if product_dir.exists() else 0
    )
    product_files_used_done = int(len(prod_paths_unique))

    cache_hit_rate = float(min(max(cache_hit_rate, 0.0), 1.0))
    expected_input_cost_per_1m = cache_hit_rate * float(input_cost_hit_per_1m) + (
        1.0 - cache_hit_rate
    ) * float(input_cost_miss_per_1m)

    cost_input_usd_expected = (
        llm_input_tokens_done / 1_000_000.0
    ) * expected_input_cost_per_1m
    cost_input_usd_all_hit = (llm_input_tokens_done / 1_000_000.0) * float(
        input_cost_hit_per_1m
    )
    cost_input_usd_all_miss = (llm_input_tokens_done / 1_000_000.0) * float(
        input_cost_miss_per_1m
    )
    cost_output_usd = (llm_output_tokens_done / 1_000_000.0) * float(output_cost_per_1m)
    cost_total_usd_expected = cost_input_usd_expected + cost_output_usd

    prof = _analyze_company_profile(company_dir, tc)

    return CompanyRow(
        company_id=bvdid,
        name=name,
        root_url=root_url,
        status=status,
        last_error=last_error,
        done_reason=done_reason,
        done_at=done_at,
        industry=industry,
        nace=nace,
        industry_label=industry_label,
        industry_source=industry_source,
        urls_total=urls_total,
        urls_markdown_done=urls_markdown_done,
        urls_llm_done=urls_llm_done,
        url_count=url_count,
        url_status_ok=url_status_ok,
        url_status_redirect=url_status_redirect,
        url_status_client_error=url_status_client_error,
        url_status_server_error=url_status_server_error,
        url_status_other=url_status_other,
        url_error_count=url_error_count,
        gating_accept_true=gating_accept_true,
        gating_accept_false=gating_accept_false,
        presence_positive=presence_positive,
        presence_zero=presence_zero,
        extracted_positive=extracted_positive,
        extracted_zero=extracted_zero,
        markdown_saved=markdown_saved,
        markdown_suppressed=markdown_suppressed,
        markdown_other_status=markdown_other_status,
        md_words_files=md_words_files,
        md_words_total=md_words_total,
        md_words_mean_per_file=md_words_mean_per_file,
        md_words_median_per_file=md_words_median_per_file,
        md_tokens_all=md_tokens_all,
        llm_input_tokens_done=llm_input_tokens_done,
        llm_output_tokens_done=llm_output_tokens_done,
        llm_done_pages=llm_done_pages,
        llm_pending_pages=llm_pending_pages,
        product_files_total=product_files_total,
        product_files_used_done=product_files_used_done,
        cost_input_usd_expected=cost_input_usd_expected,
        cost_input_usd_all_hit=cost_input_usd_all_hit,
        cost_input_usd_all_miss=cost_input_usd_all_miss,
        cost_output_usd=cost_output_usd,
        cost_total_usd_expected=cost_total_usd_expected,
        profile_present=bool(prof.present),
        profile_pipeline_version=prof.pipeline_version,
        profile_offerings_total=int(prof.offerings_total),
        profile_offerings_products=int(prof.offerings_products),
        profile_offerings_services=int(prof.offerings_services),
        profile_sources_total=int(prof.sources_total),
        profile_desc_sentences_total=int(prof.desc_sentences_total),
        profile_alias_total=int(prof.alias_total),
        profile_embedding_model=prof.embedding_model,
        profile_embedding_dim=int(prof.embedding_dim),
        profile_embedding_offerings_vecs=int(prof.embedding_offerings_vecs),
        profile_json_tokens=int(prof.profile_json_tokens),
        profile_json_bytes=int(prof.profile_json_bytes),
        profile_md_tokens=int(prof.profile_md_tokens),
        profile_md_bytes=int(prof.profile_md_bytes),
        _company_dir=str(company_dir.resolve()),
        _meta_path=str((company_dir / "metadata" / "crawl_meta.json").resolve()),
        _url_index_path=str((company_dir / "metadata" / "url_index.json").resolve()),
        _db_path=str(db_path.resolve()),
    )


def collect_dataframe(
    outputs_root: Path, company_ids: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tc = TokenCounter()

    cache_hit_rate = _env_float("ANALYZE_CACHE_HIT_RATE", DEFAULT_CACHE_HIT_RATE)
    input_cost_hit_per_1m = _env_float(
        "ANALYZE_INPUT_COST_PER_1M_CACHE_HIT", DEFAULT_INPUT_COST_PER_1M_CACHE_HIT
    )
    input_cost_miss_per_1m = _env_float(
        "ANALYZE_INPUT_COST_PER_1M_CACHE_MISS", DEFAULT_INPUT_COST_PER_1M_CACHE_MISS
    )
    output_cost_per_1m = _env_float(
        "ANALYZE_OUTPUT_COST_PER_1M", DEFAULT_OUTPUT_COST_PER_1M
    )

    # Prefer global state's db_path if present (new format)
    gs = _load_global_state(outputs_root)
    db_path = None
    if isinstance(gs, dict):
        dbp = gs.get("db_path")
        if isinstance(dbp, str) and dbp.strip():
            db_path = Path(dbp).expanduser().resolve()
    if db_path is None:
        db_path = (outputs_root / "crawl_state.sqlite3").resolve()

    run_cfg: Dict[str, Any] = {
        "generated_at": _utc_iso_z(),
        "outputs_root": str(outputs_root.resolve()),
        "db_path": str(db_path),
        "tokenizer_mode": tc.mode,
        "cache_hit_rate": float(min(max(cache_hit_rate, 0.0), 1.0)),
        "pricing_usd_per_1m": {
            "input_cache_hit": float(input_cost_hit_per_1m),
            "input_cache_miss": float(input_cost_miss_per_1m),
            "output": float(output_cost_per_1m),
        },
        "env_overrides": {
            "ANALYZE_TOKENIZER": os.getenv("ANALYZE_TOKENIZER"),
            "ANALYZE_CACHE_HIT_RATE": os.getenv("ANALYZE_CACHE_HIT_RATE"),
            "ANALYZE_INPUT_COST_PER_1M_CACHE_HIT": os.getenv(
                "ANALYZE_INPUT_COST_PER_1M_CACHE_HIT"
            ),
            "ANALYZE_INPUT_COST_PER_1M_CACHE_MISS": os.getenv(
                "ANALYZE_INPUT_COST_PER_1M_CACHE_MISS"
            ),
            "ANALYZE_OUTPUT_COST_PER_1M": os.getenv("ANALYZE_OUTPUT_COST_PER_1M"),
        },
        "global_state": gs,
    }

    rows: List[Dict[str, Any]] = []

    if db_path.exists():
        conn = _open_db(db_path)
        try:
            run_cfg["latest_run"] = _db_latest_run(conn)
            companies = _db_companies(conn, company_ids)
        finally:
            conn.close()

        for c in companies:
            try:
                row = _row_from_db_company(
                    outputs_root,
                    db_path,
                    c,
                    tc,
                    cache_hit_rate=cache_hit_rate,
                    input_cost_hit_per_1m=input_cost_hit_per_1m,
                    input_cost_miss_per_1m=input_cost_miss_per_1m,
                    output_cost_per_1m=output_cost_per_1m,
                )
                rows.append(row.as_dict())
            except Exception as e:
                bvdid = str(c.get("bvdid") or "")
                cdir = _company_dir_for(outputs_root, bvdid)
                rows.append(
                    {
                        "company_id": bvdid or cdir.name,
                        "name": str(c.get("name") or ""),
                        "root_url": str(c.get("root_url") or ""),
                        "status": "analyze_error",
                        "industry": int(c.get("industry") or 0),
                        "nace": int(c.get("nace") or 0),
                        "industry_label": str(
                            c.get("industry_label") or "Unclassified"
                        ),
                        "industry_source": str(c.get("industry_source") or ""),
                        "_company_dir": str(cdir.resolve()),
                        "_db_path": str(db_path),
                        "_error": str(e),
                    }
                )
    else:
        run_cfg["latest_run"] = None
        run_cfg["db_missing"] = True

        company_dirs = (
            [outputs_root / cid for cid in company_ids]
            if company_ids
            else discover_company_dirs(outputs_root)
        )
        company_dirs = [d for d in company_dirs if d.exists() and d.is_dir()]

        for cdir in company_dirs:
            crawl = _load_crawl_meta(cdir) or {}
            bvdid = str(crawl.get("company_id") or cdir.name)
            rows.append(
                {
                    "company_id": bvdid,
                    "name": str(crawl.get("company_name") or ""),
                    "root_url": str(crawl.get("root_url") or ""),
                    "status": DB_MISSING,
                    "industry": int(_to_int_if_nat(crawl.get("industry") or 0)),
                    "nace": int(_to_int_if_nat(crawl.get("nace") or 0)),
                    "industry_label": str(
                        crawl.get("industry_label") or "Unclassified"
                    ),
                    "industry_source": str(crawl.get("industry_source") or ""),
                    "_company_dir": str(cdir.resolve()),
                    "_db_path": str(db_path),
                }
            )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Robust numeric conversion
    numeric_candidates = [
        c
        for c in df.columns
        if c.startswith(
            ("url_", "md_", "llm_", "product_", "cost_", "profile_", "urls_")
        )
        or c in ("industry", "nace", "url_count")
    ]
    for c in numeric_candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass

    return df, run_cfg


# --------------------------------------------------------------------------- #
# Summary + percentiles
# --------------------------------------------------------------------------- #


def _percentiles(
    series: pd.Series, cuts=(50, 75, 80, 90, 95, 97, 99)
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    clean = series.dropna()
    if clean.empty:
        return {f"p{p}": 0 for p in cuts}
    for p in cuts:
        v = np.percentile(clean, p)
        out[f"p{p}"] = int(v) if float(v).is_integer() else float(v)
    return out


def compute_summary(df: pd.DataFrame, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = int(len(df)) if df is not None else 0

    def stat(col: str) -> Dict[str, Any]:
        if col not in df or df[col].dropna().empty:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "sum": 0,
                **{f"p{p}": 0 for p in (50, 75, 80, 90, 95, 97, 99)},
            }
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)

        sm = float(s.sum())
        mn = float(s.mean())
        md = float(s.median())

        sm_out: Any = int(sm) if sm.is_integer() else sm
        mn_out: Any = int(mn) if mn.is_integer() else mn
        md_out: Any = int(md) if md.is_integer() else md

        return {
            "count": int(s.count()),
            "mean": mn_out,
            "median": md_out,
            "sum": sm_out,
            **_percentiles(s),
        }

    input_tokens_sum = int(
        pd.to_numeric(df.get("llm_input_tokens_done", 0), errors="coerce")
        .fillna(0)
        .sum()
    )
    output_tokens_sum = int(
        pd.to_numeric(df.get("llm_output_tokens_done", 0), errors="coerce")
        .fillna(0)
        .sum()
    )
    cost_sum = float(
        pd.to_numeric(df.get("cost_total_usd_expected", 0), errors="coerce")
        .fillna(0)
        .sum()
    )

    cache_hit_rate = float(run_cfg.get("cache_hit_rate") or DEFAULT_CACHE_HIT_RATE)
    pricing = run_cfg.get("pricing_usd_per_1m") or {}
    in_hit = float(
        pricing.get("input_cache_hit") or DEFAULT_INPUT_COST_PER_1M_CACHE_HIT
    )
    in_miss = float(
        pricing.get("input_cache_miss") or DEFAULT_INPUT_COST_PER_1M_CACHE_MISS
    )
    out_cost = float(pricing.get("output") or DEFAULT_OUTPUT_COST_PER_1M)

    expected_input_cost_per_1m = (
        cache_hit_rate * in_hit + (1.0 - cache_hit_rate) * in_miss
    )

    total_input_cost_expected = (
        input_tokens_sum / 1_000_000.0
    ) * expected_input_cost_per_1m
    total_input_cost_all_hit = (input_tokens_sum / 1_000_000.0) * in_hit
    total_input_cost_all_miss = (input_tokens_sum / 1_000_000.0) * in_miss
    total_output_cost = (output_tokens_sum / 1_000_000.0) * out_cost
    total_cost_expected = total_input_cost_expected + total_output_cost

    status_dist: Dict[str, int] = {}
    if df is not None and not df.empty and "status" in df.columns:
        vc = (
            df["status"]
            .fillna(STATUS_MISSING)
            .astype(str)
            .replace("", STATUS_MISSING)
            .value_counts()
        )
        status_dist = {str(k): int(v) for k, v in vc.items()}

    terminal_reasons: Dict[str, int] = {}
    if (
        df is not None
        and not df.empty
        and "done_reason" in df.columns
        and "status" in df.columns
    ):
        tmp = df.copy()
        tmp["status"] = tmp["status"].fillna("").astype(str)
        tmp["done_reason"] = tmp["done_reason"].fillna("").astype(str)
        term = tmp[tmp["status"].astype(str).str.strip() == "terminal_done"]
        if not term.empty:
            vr = term["done_reason"].replace("", "unknown").value_counts()
            terminal_reasons = {str(k): int(v) for k, v in vr.items()}

    summary: Dict[str, Any] = {
        "generated_at": run_cfg.get("generated_at"),
        "outputs_root": run_cfg.get("outputs_root"),
        "db_path": run_cfg.get("db_path"),
        "latest_run": run_cfg.get("latest_run"),
        "tokenizer_mode": run_cfg.get("tokenizer_mode"),
        "companies_count": int(n),
        "pricing_usd_per_1m": {
            "input_cache_hit": in_hit,
            "input_cache_miss": in_miss,
            "output": out_cost,
        },
        "cache_hit_rate_assumed": cache_hit_rate,
        "status_distribution": status_dist,
        "terminal_done_reasons": terminal_reasons,
        "stats": {
            "urls_total": stat("urls_total"),
            "md_words_total": stat("md_words_total"),
            "md_tokens_all": stat("md_tokens_all"),
            "llm_input_tokens_done": stat("llm_input_tokens_done"),
            "llm_output_tokens_done": stat("llm_output_tokens_done"),
            "cost_total_usd_expected": stat("cost_total_usd_expected"),
            "profile_offerings_total": stat("profile_offerings_total"),
        },
        "totals": {
            "llm_input_tokens_done_sum": int(input_tokens_sum),
            "llm_output_tokens_done_sum": int(output_tokens_sum),
            "cost_total_usd_expected_sum": float(cost_sum),
        },
        "cost_breakdown_expected": {
            "input_cost_expected_usd": float(total_input_cost_expected),
            "input_cost_all_hit_usd": float(total_input_cost_all_hit),
            "input_cost_all_miss_usd": float(total_input_cost_all_miss),
            "output_cost_usd": float(total_output_cost),
            "total_cost_expected_usd": float(total_cost_expected),
        },
        "notes": {
            "pricing_units": "USD per 1M tokens",
            "llm_input_tokens_done_definition": "sum of tokens in markdown files referenced by url_index entries classified as llm_done",
            "llm_output_tokens_done_definition": "sum of tokens in product/json files referenced by url_index entries classified as llm_done",
        },
        "global_state": run_cfg.get("global_state"),
    }
    return summary


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _write_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Plots (matplotlib + plotly if available)
# --------------------------------------------------------------------------- #


def _savefig(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _plot_hist(
    df: pd.DataFrame, col: str, out: Path, title: str, xlabel: str, bins: int = 40
) -> None:
    if col not in df:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("companies")
    for p in (50, 75, 90, 95, 97, 99):
        v = float(np.percentile(s, p)) if len(s) else 0.0
        ax.axvline(v, linestyle="--")
        label = f"p{p}={int(v) if v.is_integer() else v:.0f}"
        ax.text(v, ax.get_ylim()[1] * 0.9, label, rotation=90, va="top")
    _savefig(fig, out)


def _write_plotly_hist(
    df: pd.DataFrame, col: str, out: Path, title: str, nbins: int = 40
) -> None:
    if px is None:
        return
    if col not in df:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return
    fig = px.histogram(df, x=col, nbins=nbins, title=title)  # type: ignore
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)  # type: ignore


# --------------------------------------------------------------------------- #
# Global state helper (crawl_global_state.json) - UPDATED to new schema
# --------------------------------------------------------------------------- #


def _load_global_state(outputs_root: Path) -> Optional[Dict[str, Any]]:
    path = outputs_root / "crawl_global_state.json"
    obj = _load_json(path)
    return obj if isinstance(obj, dict) else None


def _print_global_state(outputs_root: Path) -> None:
    gs = _load_global_state(outputs_root)
    if not gs:
        return

    print("=" * 80)
    print("Global crawl state (from crawl_global_state.json):")
    print(f"  updated_at:            {gs.get('updated_at')}")
    print(f"  output_root:           {gs.get('output_root')}")
    print(f"  db_path:               {gs.get('db_path')}")
    print(f"  total_companies:       {_to_int_if_nat(gs.get('total_companies'))}")
    print(f"  crawled_companies:     {_to_int_if_nat(gs.get('crawled_companies'))}")
    print(
        f"  md_done_companies:     {_to_int_if_nat(gs.get('md_done_companies'))}  ({gs.get('md_done_pct')})"
    )
    print(
        f"  llm_done_companies:    {_to_int_if_nat(gs.get('llm_done_companies'))}  ({gs.get('llm_done_pct')})"
    )
    print(
        f"  terminal_done_companies:{_to_int_if_nat(gs.get('terminal_done_companies'))}"
    )

    by_status = gs.get("by_status") or {}
    if isinstance(by_status, dict) and by_status:
        print("  by_status:")
        for k, v in by_status.items():
            print(f"    {k}: {_to_int_if_nat(v)}")

    ids = gs.get("company_ids_sample") or {}
    if isinstance(ids, dict) and ids:
        print("  company_ids_sample:")
        for k in ("pending", "in_progress", "done"):
            v = ids.get(k)
            if isinstance(v, list):
                print(f"    {k}: {len(v)} sample(s)")

    print(f"  urls_total_sum:        {_to_int_if_nat(gs.get('urls_total_sum'))}")
    print(
        f"  urls_markdown_done_sum:{_to_int_if_nat(gs.get('urls_markdown_done_sum'))}"
    )
    print(f"  urls_llm_done_sum:     {_to_int_if_nat(gs.get('urls_llm_done_sum'))}")

    tdr = gs.get("terminal_done_reasons") or {}
    if isinstance(tdr, dict) and tdr:
        print("  terminal_done_reasons:")
        for k, v in tdr.items():
            print(f"    {k}: {_to_int_if_nat(v)}")

    unknown = gs.get("unknown_statuses") or {}
    if isinstance(unknown, dict) and unknown:
        print("  unknown_statuses:")
        for k, v in unknown.items():
            print(f"    {k}: {_to_int_if_nat(v)}")

    lr = gs.get("latest_run") or {}
    if isinstance(lr, dict) and lr:
        print("  latest_run:")
        for k in (
            "run_id",
            "pipeline",
            "version",
            "args_hash",
            "crawl4ai_cache_base_dir",
            "crawl4ai_cache_mode",
            "started_at",
            "total_companies",
            "completed_companies",
            "last_company_bvdid",
            "last_updated",
        ):
            if k in lr:
                print(f"    {k}: {lr.get(k)}")
    print()


# --------------------------------------------------------------------------- #
# Terminal reporting helpers
# --------------------------------------------------------------------------- #


def _fmt_int(n: Any) -> str:
    try:
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _fmt_float(x: Any, digits: int = 6) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _print_basic_run_info(
    outputs_root: Path, out_dir: Path, run_cfg: Dict[str, Any], df: pd.DataFrame
) -> None:
    cache_hit_rate = float(run_cfg.get("cache_hit_rate") or 0.0)
    pricing = run_cfg.get("pricing_usd_per_1m") or {}
    in_hit = float(
        pricing.get("input_cache_hit") or DEFAULT_INPUT_COST_PER_1M_CACHE_HIT
    )
    in_miss = float(
        pricing.get("input_cache_miss") or DEFAULT_INPUT_COST_PER_1M_CACHE_MISS
    )
    out_cost = float(pricing.get("output") or DEFAULT_OUTPUT_COST_PER_1M)

    expected_input_cost_per_1m = (
        cache_hit_rate * in_hit + (1.0 - cache_hit_rate) * in_miss
    )

    print("=" * 80)
    print("Analyze configuration:")
    print(f"  outputs_root:                  {str(outputs_root.resolve())}")
    print(f"  out_dir:                       {str(out_dir.resolve())}")
    print(f"  db_path:                       {run_cfg.get('db_path')}")
    print(f"  tokenizer_mode:                {run_cfg.get('tokenizer_mode')}")
    print(f"  cache_hit_rate_assumed:        {cache_hit_rate:.2%}")
    print("  pricing (USD per 1M tokens):")
    print(f"    input_cache_hit_per_1m:      {in_hit}")
    print(f"    input_cache_miss_per_1m:     {in_miss}")
    print(f"    output_per_1m:               {out_cost}")
    print(f"  implied_input_expected_per_1m: {expected_input_cost_per_1m}")
    if px is None:
        print("  plotly:                        not installed (no interactive HTML)")
    if tiktoken is None:
        print(
            "  tiktoken:                      not installed (tokenizer_mode may be approx)"
        )
    env_overrides = run_cfg.get("env_overrides") or {}
    if isinstance(env_overrides, dict):
        active = {k: v for k, v in env_overrides.items() if v is not None}
        if active:
            print("  active env overrides:")
            for k, v in active.items():
                print(f"    {k}={v}")
    print()

    if df is None or df.empty:
        print("No companies discovered.")
        return

    n_companies = int(len(df))
    urls_total_sum = int(
        pd.to_numeric(df.get("urls_total", 0), errors="coerce").fillna(0).sum()
    )
    url_count_sum = int(
        pd.to_numeric(df.get("url_count", 0), errors="coerce").fillna(0).sum()
    )
    md_done_sum = int(
        pd.to_numeric(df.get("urls_markdown_done", 0), errors="coerce").fillna(0).sum()
    )
    llm_done_sum = int(
        pd.to_numeric(df.get("urls_llm_done", 0), errors="coerce").fillna(0).sum()
    )

    llm_input_tokens_sum = int(
        pd.to_numeric(df.get("llm_input_tokens_done", 0), errors="coerce")
        .fillna(0)
        .sum()
    )
    llm_output_tokens_sum = int(
        pd.to_numeric(df.get("llm_output_tokens_done", 0), errors="coerce")
        .fillna(0)
        .sum()
    )
    llm_done_pages_sum = int(
        pd.to_numeric(df.get("llm_done_pages", 0), errors="coerce").fillna(0).sum()
    )
    llm_pending_pages_sum = int(
        pd.to_numeric(df.get("llm_pending_pages", 0), errors="coerce").fillna(0).sum()
    )

    ok_sum = int(
        pd.to_numeric(df.get("url_status_ok", 0), errors="coerce").fillna(0).sum()
    )
    redir_sum = int(
        pd.to_numeric(df.get("url_status_redirect", 0), errors="coerce").fillna(0).sum()
    )
    c4_sum = int(
        pd.to_numeric(df.get("url_status_client_error", 0), errors="coerce")
        .fillna(0)
        .sum()
    )
    c5_sum = int(
        pd.to_numeric(df.get("url_status_server_error", 0), errors="coerce")
        .fillna(0)
        .sum()
    )
    other_sum = int(
        pd.to_numeric(df.get("url_status_other", 0), errors="coerce").fillna(0).sum()
    )
    err_sum = int(
        pd.to_numeric(df.get("url_error_count", 0), errors="coerce").fillna(0).sum()
    )

    md_rate = (
        _safe_div(md_done_sum, urls_total_sum)
        if urls_total_sum
        else _safe_div(md_done_sum, url_count_sum)
    )
    llm_rate = (
        _safe_div(llm_done_sum, urls_total_sum)
        if urls_total_sum
        else _safe_div(llm_done_sum, url_count_sum)
    )

    total_input_cost_expected = (
        llm_input_tokens_sum / 1_000_000.0
    ) * expected_input_cost_per_1m
    total_input_cost_all_hit = (llm_input_tokens_sum / 1_000_000.0) * in_hit
    total_input_cost_all_miss = (llm_input_tokens_sum / 1_000_000.0) * in_miss
    total_output_cost = (llm_output_tokens_sum / 1_000_000.0) * out_cost
    total_cost_expected = total_input_cost_expected + total_output_cost

    print("=" * 80)
    print("Dataset overview:")
    print(f"  companies:                     {_fmt_int(n_companies)}")
    if urls_total_sum:
        print(f"  urls_total (sum):              {_fmt_int(urls_total_sum)}")
    if url_count_sum:
        print(f"  url_index entries (sum):       {_fmt_int(url_count_sum)}")
    print(f"  markdown_done (sum):           {_fmt_int(md_done_sum)}  ({md_rate:.2%})")
    print(
        f"  llm_done (sum):                {_fmt_int(llm_done_sum)}  ({llm_rate:.2%})"
    )
    print(f"  llm_done_pages (sum):          {_fmt_int(llm_done_pages_sum)}")
    print(f"  llm_pending_pages (sum):       {_fmt_int(llm_pending_pages_sum)}")

    print("  HTTP status buckets (sum from url_index):")
    total_status = ok_sum + redir_sum + c4_sum + c5_sum + other_sum
    if total_status:
        print(
            f"    ok:                          {_fmt_int(ok_sum)}   ({_safe_div(ok_sum, total_status):.2%})"
        )
        print(
            f"    redirect:                    {_fmt_int(redir_sum)} ({_safe_div(redir_sum, total_status):.2%})"
        )
        print(
            f"    4xx:                         {_fmt_int(c4_sum)}   ({_safe_div(c4_sum, total_status):.2%})"
        )
        print(
            f"    5xx:                         {_fmt_int(c5_sum)}   ({_safe_div(c5_sum, total_status):.2%})"
        )
        print(
            f"    other:                       {_fmt_int(other_sum)} ({_safe_div(other_sum, total_status):.2%})"
        )
    print(f"  url_error_count (sum):         {_fmt_int(err_sum)}")

    print("=" * 80)
    print("Token + cost (based on llm_done pages only):")
    print(
        f"  llm_input_tokens_done (sum):   {_fmt_int(llm_input_tokens_sum)}  ({llm_input_tokens_sum / 1_000_000.0:.3f}M)"
    )
    print(
        f"  llm_output_tokens_done (sum):  {_fmt_int(llm_output_tokens_sum)}  ({llm_output_tokens_sum / 1_000_000.0:.3f}M)"
    )
    print("  cost bounds + expected (USD):")
    print(f"    input_all_hit:               {_fmt_float(total_input_cost_all_hit, 6)}")
    print(
        f"    input_all_miss:              {_fmt_float(total_input_cost_all_miss, 6)}"
    )
    print(
        f"    input_expected (@{cache_hit_rate:.0%} hit): {_fmt_float(total_input_cost_expected, 6)}"
    )
    print(f"    output:                      {_fmt_float(total_output_cost, 6)}")
    print(f"    total_expected:              {_fmt_float(total_cost_expected, 6)}")
    print()


def _print_status_distribution(df: pd.DataFrame, top_k: int = 12) -> None:
    if df is None or df.empty or "status" not in df:
        return
    vc = (
        df["status"]
        .fillna(STATUS_MISSING)
        .astype(str)
        .replace("", STATUS_MISSING)
        .value_counts()
    )
    if vc.empty:
        return
    print("=" * 80)
    print("Company status distribution (top):")
    for i, (k, v) in enumerate(vc.items()):
        if i >= top_k:
            break
        print(f"  {k}: {int(v)}")
    if len(vc) > top_k:
        print(f"  ... ({int(len(vc) - top_k)} more)")
    print()


def _print_top_companies(
    df: pd.DataFrame, *, col: str, title: str, top_k: int = 10
) -> None:
    if df is None or df.empty or col not in df:
        return
    s = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if s.empty:
        return
    tmp = df.copy()
    tmp[col] = s
    tmp = tmp.sort_values(col, ascending=False).head(top_k)
    print("=" * 80)
    print(title)
    for _, r in tmp.iterrows():
        cid = str(r.get("company_id") or "")
        root = str(r.get("root_url") or "")
        val = r.get(col)
        print(f"  {cid}: {col}={_fmt_int(val)}  root_url={root}")
    print()


def _print_analyze_errors(df: pd.DataFrame, top_k: int = 10) -> None:
    if df is None or df.empty or "_error" not in df:
        return
    err_df = df[df["_error"].notna() & (df["_error"].astype(str).str.strip() != "")]
    if err_df.empty:
        return
    print("=" * 80)
    print(f"Analyze errors (first {top_k}):")
    for _, r in err_df.head(top_k).iterrows():
        print(
            f"  company_id={r.get('company_id')}  status={r.get('status')}  error={r.get('_error')}"
        )
    print()


# --------------------------------------------------------------------------- #
# Industry reporting
# --------------------------------------------------------------------------- #


def _industry_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    tmp = df.copy()

    if "industry_label" not in tmp.columns:
        tmp["industry_label"] = "Unclassified"
    tmp["industry_label"] = (
        tmp["industry_label"]
        .fillna("Unclassified")
        .astype(str)
        .replace("", "Unclassified")
    )

    for c in (
        "urls_total",
        "urls_markdown_done",
        "urls_llm_done",
        "llm_input_tokens_done",
        "llm_output_tokens_done",
        "cost_total_usd_expected",
        "llm_done_pages",
        "llm_pending_pages",
        "url_error_count",
    ):
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

    g = tmp.groupby("industry_label", dropna=False)

    industry_summary = pd.DataFrame(
        {
            "industry_label": g.size().index,
            "companies": g.size().astype(int).values,
            "urls_total_sum": g["urls_total"].sum().astype(int).values
            if "urls_total" in tmp
            else 0,
            "urls_markdown_done_sum": g["urls_markdown_done"].sum().astype(int).values
            if "urls_markdown_done" in tmp
            else 0,
            "urls_llm_done_sum": g["urls_llm_done"].sum().astype(int).values
            if "urls_llm_done" in tmp
            else 0,
            "llm_done_pages_sum": g["llm_done_pages"].sum().astype(int).values
            if "llm_done_pages" in tmp
            else 0,
            "llm_pending_pages_sum": g["llm_pending_pages"].sum().astype(int).values
            if "llm_pending_pages" in tmp
            else 0,
            "llm_input_tokens_done_sum": g["llm_input_tokens_done"]
            .sum()
            .astype(int)
            .values
            if "llm_input_tokens_done" in tmp
            else 0,
            "llm_output_tokens_done_sum": g["llm_output_tokens_done"]
            .sum()
            .astype(int)
            .values
            if "llm_output_tokens_done" in tmp
            else 0,
            "url_error_count_sum": g["url_error_count"].sum().astype(int).values
            if "url_error_count" in tmp
            else 0,
            "cost_total_usd_expected_sum": g["cost_total_usd_expected"]
            .sum()
            .astype(float)
            .values
            if "cost_total_usd_expected" in tmp
            else 0.0,
        }
    )

    def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
        den2 = den.replace(0, np.nan)
        r = (num / den2).fillna(0.0)
        return r

    if "urls_total_sum" in industry_summary.columns:
        industry_summary["markdown_done_rate"] = _rate(
            industry_summary["urls_markdown_done_sum"],
            industry_summary["urls_total_sum"],
        )
        industry_summary["llm_done_rate"] = _rate(
            industry_summary["urls_llm_done_sum"], industry_summary["urls_total_sum"]
        )

    industry_summary = industry_summary.sort_values(
        ["companies", "cost_total_usd_expected_sum"], ascending=[False, False]
    )

    if "status" in tmp.columns:
        st = tmp.copy()
        st["status"] = (
            st["status"].fillna(STATUS_MISSING).astype(str).replace("", STATUS_MISSING)
        )
        piv = st.pivot_table(
            index="industry_label",
            columns="status",
            values="company_id",
            aggfunc="count",
            fill_value=0,
        ).reset_index()
        piv_cols = [c for c in piv.columns if c != "industry_label"]
        piv[piv_cols] = piv[piv_cols].astype(int)
        industry_status = piv
    else:
        industry_status = pd.DataFrame()

    return industry_summary, industry_status


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def run_analysis(
    outputs_root: Path,
    out_dir: Path,
    company_ids: Optional[List[str]] = None,
    *,
    write_plots: bool = True,
) -> Dict[str, Any]:
    df, run_cfg = collect_dataframe(outputs_root, company_ids=company_ids)

    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        summary = {**run_cfg, "companies_count": 0, "message": "No companies found."}
        _write_json(out_dir / "summary.json", summary)
        return summary

    _write_csv(out_dir / "companies.csv", df)
    _write_jsonl(out_dir / "companies.jsonl", df)

    summary = compute_summary(df, run_cfg)
    _write_json(out_dir / "summary.json", summary)

    industry_summary, industry_status = _industry_tables(df)
    if not industry_summary.empty:
        _write_csv(out_dir / "industry_summary.csv", industry_summary)
        _write_json(
            out_dir / "industry_summary.json",
            industry_summary.to_dict(orient="records"),
        )
    if not industry_status.empty:
        _write_csv(out_dir / "industry_status.csv", industry_status)
        _write_json(
            out_dir / "industry_status.json", industry_status.to_dict(orient="records")
        )

    if write_plots:
        _plot_hist(
            df,
            "urls_total",
            out_dir / "hist_urls_total.png",
            "Distribution of urls_total per company",
            "urls_total",
        )
        _plot_hist(
            df,
            "md_tokens_all",
            out_dir / "hist_md_tokens_all.png",
            "Distribution of markdown tokens (all md/*.md)",
            "md_tokens_all",
        )
        _plot_hist(
            df,
            "llm_input_tokens_done",
            out_dir / "hist_llm_input_tokens_done.png",
            "Distribution of LLM input tokens (done pages)",
            "llm_input_tokens_done",
        )
        _plot_hist(
            df,
            "llm_output_tokens_done",
            out_dir / "hist_llm_output_tokens_done.png",
            "Distribution of LLM output tokens (done pages)",
            "llm_output_tokens_done",
        )
        _plot_hist(
            df,
            "cost_total_usd_expected",
            out_dir / "hist_cost_total_usd_expected.png",
            "Distribution of expected total cost per company",
            "cost_total_usd_expected",
        )

        _write_plotly_hist(
            df,
            "urls_total",
            out_dir / "hist_urls_total.html",
            "Distribution of urls_total per company",
        )
        _write_plotly_hist(
            df,
            "llm_input_tokens_done",
            out_dir / "hist_llm_input_tokens_done.html",
            "Distribution of LLM input tokens (done pages)",
        )
        _write_plotly_hist(
            df,
            "llm_output_tokens_done",
            out_dir / "hist_llm_output_tokens_done.html",
            "Distribution of LLM output tokens (done pages)",
        )
        _write_plotly_hist(
            df,
            "cost_total_usd_expected",
            out_dir / "hist_cost_total_usd_expected.html",
            "Distribution of expected total cost per company",
        )

    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze web2product-catalog outputs (DB + url_index)."
    )
    p.add_argument(
        "outputs_root",
        type=str,
        help="Outputs root directory (contains crawl_state.sqlite3 and crawl_global_state.json).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for analysis artifacts (default: <outputs_root>/analysis_<timestamp>).",
    )
    p.add_argument(
        "--company-id",
        action="append",
        default=[],
        help="Analyze only this company id (repeatable). If omitted, analyze all companies in DB.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    p.add_argument(
        "--print-global-state",
        action="store_true",
        help="Print crawl_global_state.json summary before analysis.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print dataset overview tables to stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (
            outputs_root / f"analysis_{_utc_iso_z().replace(':', '').replace('-', '')}"
        )
    )
    company_ids = [str(x) for x in (args.company_id or []) if str(x).strip()] or None

    if args.print_global_state:
        _print_global_state(outputs_root)

    summary = run_analysis(
        outputs_root, out_dir, company_ids=company_ids, write_plots=(not args.no_plots)
    )

    if not args.quiet:
        df, run_cfg = collect_dataframe(outputs_root, company_ids=company_ids)
        _print_basic_run_info(outputs_root, out_dir, run_cfg, df)
        _print_status_distribution(df)
        _print_top_companies(
            df,
            col="llm_input_tokens_done",
            title="Top companies by llm_input_tokens_done",
        )
        _print_top_companies(
            df,
            col="llm_output_tokens_done",
            title="Top companies by llm_output_tokens_done",
        )
        _print_top_companies(
            df,
            col="cost_total_usd_expected",
            title="Top companies by expected cost (USD)",
        )
        _print_analyze_errors(df)

    print("=" * 80)
    print(f"Wrote analysis to: {str(out_dir)}")
    print(f"  - {str((out_dir / 'summary.json'))}")
    print(f"  - {str((out_dir / 'companies.csv'))}")
    print(f"  - {str((out_dir / 'companies.jsonl'))}")
    if (out_dir / "industry_summary.csv").exists():
        print(f"  - {str((out_dir / 'industry_summary.csv'))}")
    if (out_dir / "industry_status.csv").exists():
        print(f"  - {str((out_dir / 'industry_status.csv'))}")
    if (out_dir / "hist_urls_total.png").exists():
        print("  - hist_*.png (plots)")
    if (out_dir / "hist_urls_total.html").exists():
        print("  - hist_*.html (plotly)")
    _ = summary  # keep lint quiet


if __name__ == "__main__":
    main()
