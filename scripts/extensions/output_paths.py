from __future__ import annotations

import json
import re
from hashlib import sha1
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

# Base directory
OUTPUT_ROOT = Path("outputs")

# ---------------- Sanitization helpers ----------------

_INVALID_WIN_CHARS = r'<>:"/\\|?\*\x00-\x1F'  # also covers control chars


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
    s = s.rstrip(" .")  # Windows forbids trailing dot/space
    if not s:
        s = "UNKNOWN"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    if not s:
        s = "UNKNOWN"
    return s


def _safe_slug_from_path(path: str, *, max_len: int = 80) -> str:
    """
    Build a compact, safe slug from a URL path for file naming.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", path or "/")
    slug = slug.strip("_")
    if not slug or slug == "/":
        slug = "index"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("._-")
        if not slug:
            slug = "index"
    return slug


# ---------------- Public API ----------------

def ensure_company_dirs(bvdid: str) -> dict[str, Path]:
    """
    Ensure output folders exist for a given company ID.

    New canonical layout:
      outputs/{safe_bvdid}/
        html/        -> raw / cleaned HTML artifacts
        markdown/    -> generated Markdown
        product/     -> structured LLM output (product info, JSON)
        log/         -> per-company logs
        metadata/    -> crawl_meta.json, url_index.json, presence flags, etc.

    For backward compatibility, the returned mapping also exposes:
      - "json"        -> product/    (old name for LLM output)
      - "logs"        -> log/
      - "checkpoints" -> metadata/
    """
    safe = sanitize_bvdid(bvdid)
    base = OUTPUT_ROOT / safe

    html_dir = base / "html"
    md_dir = base / "markdown"
    product_dir = base / "product"
    log_dir = base / "log"
    metadata_dir = base / "metadata"

    dirs: dict[str, Path] = {
        "html": html_dir,
        "markdown": md_dir,
        "product": product_dir,
        "json": product_dir,        # legacy alias
        "log": log_dir,
        "logs": log_dir,            # legacy alias
        "metadata": metadata_dir,
        "checkpoints": metadata_dir,  # legacy alias
    }

    # Create each unique directory once
    for d in set(dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def save_stage_output(
    bvdid: str,
    url: str,
    data: Union[str, dict[str, Any]],
    stage: str,
    *,
    filename_hint: str | None = None,
    encoding: str = "utf-8",
) -> Path:
    """
    Save pipeline output into the appropriate stage directory.

    Canonical stages:
      - "html"     -> outputs/{safe}/html/*.html
      - "markdown" -> outputs/{safe}/markdown/*.md
      - "product"  -> outputs/{safe}/product/*.json

    For backward compatibility, "json" is treated as "product".

    Filenames are based on a sanitized combination of netloc + path and an
    8-char hash of the URL to avoid collisions.
    """
    dirs = ensure_company_dirs(bvdid)

    # Legacy alias
    if stage == "json":
        stage = "product"

    if stage not in ("html", "markdown", "product"):
        raise ValueError(f"Invalid stage '{stage}' (expected html|markdown|product)")

    subdir = dirs[stage if stage != "product" else "product"]

    parsed = urlparse(url)
    if parsed.netloc:
        base_slug = f"{parsed.netloc}_{_safe_slug_from_path(parsed.path or '/')}"
    else:
        base_slug = _safe_slug_from_path(parsed.path or "/")

    h = sha1(url.encode()).hexdigest()[:8]
    fname = filename_hint or f"{base_slug}-{h}"

    if stage == "product":
        out_path = subdir / f"{fname}.json"
        with open(out_path, "w", encoding=encoding) as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(data))
        return out_path

    # html / markdown
    ext = ".html" if stage == "html" else ".md"
    out_path = subdir / f"{fname}{ext}"
    with open(out_path, "w", encoding=encoding) as f:
        f.write(str(data))

    return out_path