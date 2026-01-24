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


# -------------------------
# Meta recomputation helpers
# -------------------------


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    if not s:
        return False
    return s not in ("0", "false", "no", "none", "null", "n/a")


def _status_norm(v: Any) -> str:
    try:
        return str(v or "").strip().lower()
    except Exception:
        return ""


def _has_any_path(entry: Mapping[str, Any], keys: Tuple[str, ...]) -> bool:
    for k in keys:
        v = entry.get(k)
        if v is None:
            continue
        try:
            s = str(v).strip()
        except Exception:
            continue
        if s:
            return True
    return False


def _entry_has_markdown_evidence(entry: Mapping[str, Any]) -> bool:
    # 1) explicit booleans
    if _truthy(entry.get("markdown_done")):
        return True
    if _truthy(entry.get("md_done")):
        return True

    # 2) known path keys
    if _has_any_path(entry, ("markdown_path", "md_path", "md_file", "markdown_file")):
        return True

    # 3) status / stage fields
    st = _status_norm(entry.get("status") or entry.get("state") or entry.get("stage"))
    if st in ("markdown_saved", "markdown_done", "llm_done"):
        return True

    # 4) any LLM artifact paths that imply markdown existed / was processed
    if _has_any_path(
        entry,
        (
            "llm_json_path",
            "extraction_json_path",
            "product_json_path",
            "products_json_path",
            "nice_json_path",
            "md_json_path",
        ),
    ):
        return True

    return False


def _entry_is_suppressed(entry: Mapping[str, Any]) -> bool:
    # Common boolean flags
    for k in (
        "suppressed",
        "markdown_suppressed",
        "md_suppressed",
        "is_suppressed",
        "suppressed_by_gating",
    ):
        if _truthy(entry.get(k)):
            return True

    # Gating decision sub-objects (best-effort)
    gating = entry.get("gating")
    if isinstance(gating, Mapping):
        decision = _status_norm(gating.get("decision") or gating.get("verdict"))
        if decision in (
            "suppress",
            "suppressed",
            "skip",
            "skipped",
            "blocked",
            "deny",
            "denied",
        ):
            return True

    # Status values that imply suppression
    st = _status_norm(entry.get("status") or entry.get("state") or entry.get("stage"))
    if st in ("suppressed", "markdown_suppressed", "skipped", "blocked", "gated"):
        return True

    return False


def _compute_meta_from_entries(
    *,
    entries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    # entries: url -> entrydict (no __meta__)
    pages_seen = 0
    markdown_saved = 0
    markdown_suppressed = 0
    timeout_pages = 0
    memory_pressure_pages = 0

    for _url, e in entries.items():
        pages_seen += 1

        if _entry_is_suppressed(e):
            markdown_suppressed += 1

        if _entry_has_markdown_evidence(e):
            markdown_saved += 1

        if _truthy(e.get("timeout_page_exceeded")):
            timeout_pages += 1

        if _truthy(e.get("memory_pressure")):
            memory_pressure_pages += 1

    return {
        "pages_seen": int(pages_seen),
        "total_pages": int(pages_seen),
        "markdown_saved": int(markdown_saved),
        "markdown_suppressed": int(markdown_suppressed),
        "timeout_pages": int(timeout_pages),
        "memory_pressure_pages": int(memory_pressure_pages),
    }


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

    # Normalize URL entries
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

    # Recompute meta from normalized entries (do NOT preserve stale counters)
    # Keep non-counter meta fields (created_at, crawl_finished, hard_max_pages, etc.) if present.
    meta_raw = idx.get(URL_INDEX_META_KEY)
    meta_dict: Dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}

    # Ensure required/expected bookkeeping
    meta_dict.setdefault("company_id", company_id)
    meta_dict.setdefault("created_at", now_iso())
    meta_dict["updated_at"] = now_iso()
    meta_dict[VERSION_META_KEY] = version_meta

    # Overwrite counter fields deterministically from body
    computed = _compute_meta_from_entries(entries={u: out[u] for u in out.keys()})
    meta_dict.update(computed)

    meta_norm = UrlIndexMeta.from_dict(meta_dict, company_id=company_id)
    out[URL_INDEX_META_KEY] = meta_norm.to_dict()

    write_json_file(p, out, pretty=False)
