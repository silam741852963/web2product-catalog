from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from configs.models import UrlIndexEntry, UrlIndexMeta, URL_INDEX_META_KEY

from .json_io import read_json_file, validate_json_bytes, write_json_file
from .paths import first_existing, url_index_candidates
from .types import VERSION_META_KEY, now_iso


def is_url_index_empty_semantic(idx: Dict[str, Any]) -> bool:
    """
    Deterministic definition of "empty index":
      - valid JSON dict, but:
        - missing __meta__, OR
        - __meta__ exists but total_pages/pages_seen/markdown_saved/markdown_suppressed all None/0
    """

    meta_raw = idx.get(URL_INDEX_META_KEY)
    if not isinstance(meta_raw, dict):
        return True

    def _to_i(v: Any) -> int:
        try:
            if v is None:
                return 0
            if isinstance(v, bool):
                return 0
            if isinstance(v, int):
                return max(0, v)
            if isinstance(v, float):
                return max(0, int(v))
            s = str(v).strip()
            if not s:
                return 0
            return max(0, int(float(s)))
        except Exception:
            return 0

    total_pages = _to_i(meta_raw.get("total_pages"))
    pages_seen = _to_i(meta_raw.get("pages_seen"))
    md_saved = _to_i(meta_raw.get("markdown_saved"))
    md_supp = _to_i(meta_raw.get("markdown_suppressed"))

    return total_pages == 0 and pages_seen == 0 and md_saved == 0 and md_supp == 0


def resolve_markdown_and_html_paths_from_url_index_entry(
    *,
    md_dir: Path,
    html_dir: Path,
    entry: Mapping[str, Any],
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Exact-field resolver for this pipeline's url_index schema.

    Expected keys:
      - entry["markdown_path"] : absolute path string from writer
      - entry["html_path"]     : absolute path string from writer

    Rule:
      - ignore any parent dirs from stored path; take basename only
      - treat basename as relative to current company markdown/html dirs
    """
    md_p: Optional[Path] = None
    html_p: Optional[Path] = None

    mp = entry.get("markdown_path")
    if mp:
        name = Path(str(mp)).name
        if name:
            md_p = (md_dir / name).resolve()

    hp = entry.get("html_path")
    if hp:
        name = Path(str(hp)).name
        if name:
            html_p = (html_dir / name).resolve()

    return md_p, html_p


def url_index_meta_patch_max_pages(
    *,
    meta: Dict[str, Any],
    company_id: str,
    kept: int,
    suppressed: int,
    hard_max_pages: int,
    hard_hit: bool,
    version_meta: Dict[str, Any],
    markdown_saved: int,
) -> Dict[str, Any]:
    out = dict(meta or {})
    out.setdefault("company_id", company_id)
    out["updated_at"] = now_iso()
    out[VERSION_META_KEY] = version_meta

    out["total_pages"] = int(kept)
    out["markdown_saved"] = int(markdown_saved)
    out["markdown_suppressed"] = int(suppressed)

    out["hard_max_pages"] = int(hard_max_pages)
    out["hard_max_pages_hit"] = bool(hard_hit)
    return out


def normalize_url_index_file(
    out_dir: Path,
    company_id: str,
    *,
    version_meta: Dict[str, Any],
) -> None:
    p = first_existing(url_index_candidates(out_dir, company_id))
    if p is None:
        return

    ok, _ = validate_json_bytes(p)
    if not ok:
        return

    idx = read_json_file(p)
    if not isinstance(idx, dict) or not idx:
        return

    out: Dict[str, Any] = {}
    for k, raw in idx.items():
        if str(k).startswith("__"):
            continue
        url = str(k)
        ent = raw if isinstance(raw, Mapping) else {}
        ent_dict = dict(ent)
        ent_dict.setdefault("company_id", company_id)
        ent_dict.setdefault("url", url)
        normalized = UrlIndexEntry.from_dict(ent_dict, company_id=company_id, url=url)
        out[url] = normalized.to_dict()

    meta_raw = idx.get(URL_INDEX_META_KEY)
    meta_dict = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
    meta_dict.setdefault("company_id", company_id)
    meta_dict.setdefault("created_at", now_iso())
    meta_dict["updated_at"] = now_iso()
    meta_dict[VERSION_META_KEY] = version_meta

    meta_norm = UrlIndexMeta.from_dict(meta_dict, company_id=company_id)
    out[URL_INDEX_META_KEY] = meta_norm.to_dict()

    write_json_file(p, out, pretty=False)
