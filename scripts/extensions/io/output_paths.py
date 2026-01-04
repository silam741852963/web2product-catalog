from __future__ import annotations

import json
import re
from hashlib import sha1
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------

_OUTPUT_ROOT: Path = Path("outputs").resolve()


def set_output_root(path: str, *, create: bool = True) -> None:
    """
    Set the global output root directory for the whole pipeline.

    This is the ONLY supported output-root setter.
    """
    global _OUTPUT_ROOT
    p = Path(path).expanduser().resolve()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    _OUTPUT_ROOT = p


def get_output_root() -> str:
    """Return current resolved output root as a string path."""
    return str(_OUTPUT_ROOT)


def ensure_output_root(out_dir: str | Path) -> Path:
    """
    Resolve + create output root dir, set it globally, and return Path.
    Canonical entry point for run.py (replaces run.py::_set_output_root).
    """
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    set_output_root(str(p), create=False)  # already created above
    return p


def global_path_obj(*parts: str, ensure_parent: bool = False) -> Path:
    """
    Path-typed convenience wrapper (replaces run.py::_global_path).
    """
    return Path(global_path(*parts, ensure_parent=ensure_parent))


def global_path(*parts: str, ensure_parent: bool = False) -> str:
    """
    Build a path under current output root.

    Examples:
      global_path("crawl_state.sqlite3")
      global_path("_retry", "retry_state.json", ensure_parent=True)
    """
    p = _OUTPUT_ROOT.joinpath(*[str(x) for x in parts if str(x)])
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def retry_base_dir() -> str:
    """Canonical retry base dir under output root."""
    p = _OUTPUT_ROOT / "_retry"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------

_INVALID_WIN_CHARS = r'<>:"/\\|?\*\x00-\x1F'  # invalid + control chars


def sanitize_bvdid(raw: str, *, replacement: str = "_", max_len: int = 100) -> str:
    """
    Make a filesystem-safe bvdid for directory/file names (Windows-safe).
    - Replaces invalid characters with `_`
    - Strips trailing spaces/dots
    - Collapses runs of `_`
    - Caps length to `max_len`
    """
    s = str(raw or "").strip()
    if not s:
        return "UNKNOWN"

    s = re.sub(f"[{_INVALID_WIN_CHARS}]", replacement, s)
    s = re.sub(r"\s+", replacement, s)
    s = re.sub(r"_+", "_", s)
    s = s.rstrip(" .")
    if not s:
        s = "UNKNOWN"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
        if not s:
            s = "UNKNOWN"
    return s


def _safe_slug_from_path(path: str, *, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", path or "/")
    slug = slug.strip("_")
    if not slug or slug == "/":
        slug = "index"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("._-") or "index"
    return slug


# ---------------------------------------------------------------------------
# Company layout
# ---------------------------------------------------------------------------


def ensure_company_dirs(bvdid: str) -> dict[str, str]:
    """
    Ensure canonical output folders exist for a given bvdid.

    Canonical layout:
      {OUTPUT_ROOT}/{safe_bvdid}/
        html/
        markdown/
        product/
        log/
        metadata/

    Returns dict[str, str] of absolute paths.
    """
    safe = sanitize_bvdid(bvdid)
    base = _OUTPUT_ROOT / safe
    html_dir = base / "html"
    md_dir = base / "markdown"
    product_dir = base / "product"
    log_dir = base / "log"
    metadata_dir = base / "metadata"

    for d in (html_dir, md_dir, product_dir, log_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base": str(base),
        "html": str(html_dir),
        "markdown": str(md_dir),
        "product": str(product_dir),
        "log": str(log_dir),
        "metadata": str(metadata_dir),
    }


def save_stage_output(
    bvdid: str,
    url: str,
    data: Union[str, dict[str, Any], list[Any]],
    stage: str,
    *,
    filename_hint: str | None = None,
    encoding: str = "utf-8",
) -> str:
    """
    Save pipeline output into the appropriate stage directory.

    Stages:
      - "html"     -> .../html/*.html
      - "markdown" -> .../markdown/*.md
      - "product"  -> .../product/*.json
      - "metadata" -> .../metadata/*.(json|txt) depending on content

    Filenames are based on netloc+path plus an 8-char hash of the URL to avoid collisions.
    """
    dirs = ensure_company_dirs(bvdid)

    stage = (stage or "").strip().lower()
    if stage not in ("html", "markdown", "product", "metadata"):
        raise ValueError(
            f"Invalid stage '{stage}' (expected html|markdown|product|metadata)"
        )

    subdir = Path(dirs[stage])

    parsed = urlparse(url or "")
    if parsed.netloc:
        base_slug = f"{parsed.netloc}_{_safe_slug_from_path(parsed.path or '/')}"
    else:
        base_slug = _safe_slug_from_path(parsed.path or "/")

    h = sha1((url or "").encode("utf-8")).hexdigest()[:8]
    fname = filename_hint or f"{base_slug}-{h}"

    if stage == "html":
        out_path = subdir / f"{fname}.html"
        out_path.write_text(str(data), encoding=encoding)
        return str(out_path)

    if stage == "markdown":
        out_path = subdir / f"{fname}.md"
        out_path.write_text(str(data), encoding=encoding)
        return str(out_path)

    if stage in ("product", "metadata"):
        out_path = subdir / f"{fname}.json"
        if isinstance(data, (dict, list)):
            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding=encoding
            )
        else:
            out_path.write_text(str(data), encoding=encoding)
        return str(out_path)
