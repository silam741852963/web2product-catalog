from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse
from hashlib import sha1

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
    Returns a mapping for html, markdown, json, logs, and checkpoints subfolders.
    Uses a Windows-safe sanitized id for the path, but you should continue
    to pass the original `bvdid` everywhere else in code.
    """
    safe = sanitize_bvdid(bvdid)
    base = OUTPUT_ROOT / safe
    dirs = {
        "html": base / "html",
        "markdown": base / "markdown",
        "json": base / "json",
        "logs": base / "logs",
        "checkpoints": base / "checkpoints",
    }
    for d in dirs.values():
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
    Save output into outputs/{safe_bvdid}/{stage}/ with sanitized filename.
    stage ∈ {"html","markdown","json"}.
    """
    dirs = ensure_company_dirs(bvdid)
    subdir = dirs.get(stage)
    if not subdir:
        raise ValueError(f"Invalid stage '{stage}' (expected html|markdown|json)")

    parsed = urlparse(url)
    # Include netloc in slug for nicer grouping across identical paths on different hosts
    base_slug = f"{parsed.netloc}_{_safe_slug_from_path(parsed.path or '/')}" if parsed.netloc else _safe_slug_from_path(parsed.path or "/")
    h = sha1(url.encode()).hexdigest()[:8]
    fname = filename_hint or f"{base_slug}-{h}"

    if stage == "json":
        out_path = subdir / f"{fname}.json"
        with open(out_path, "w", encoding=encoding) as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(data))
    else:
        ext = ".html" if stage == "html" else ".md"
        out_path = subdir / f"{fname}{ext}"
        with open(out_path, "w", encoding=encoding) as f:
            f.write(str(data))

    return out_path