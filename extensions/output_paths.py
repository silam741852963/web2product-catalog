from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Union

# Base directory
OUTPUT_ROOT = Path("outputs")

def ensure_company_dirs(hojin_id: str) -> dict[str, Path]:
    """
    Ensure output folders exist for a given company ID.
    Returns a mapping for html, markdown, and json subfolders.
    """
    base = OUTPUT_ROOT / hojin_id
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
    hojin_id: str,
    url: str,
    data: Union[str, dict[str, Any]],
    stage: str,
    *,
    filename_hint: str | None = None,
    encoding: str = "utf-8",
) -> Path:
    """
    Save output into outputs/{hojin_id}/{stage}/ with sanitized filename.
    stage ∈ {"html","markdown","json"}.
    """
    dirs = ensure_company_dirs(hojin_id)
    subdir = dirs.get(stage)
    if not subdir:
        raise ValueError(f"Invalid stage '{stage}' (expected html|markdown|json)")

    from urllib.parse import urlparse
    from hashlib import sha1
    from re import sub

    parsed = urlparse(url)
    slug = sub(r"[^A-Za-z0-9_-]+", "_", (parsed.path or "/")) or "index"
    h = sha1(url.encode()).hexdigest()[:8]
    fname = filename_hint or f"{slug}-{h}"

    if stage == "json":
        fname = f"{fname}.json"
        out_path = subdir / fname
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