from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from extensions.io.output_paths import sanitize_bvdid


def company_base(out_dir: Path, company_id: str) -> Path:
    return out_dir / sanitize_bvdid(company_id)


def company_output_dir(out_dir: Path, company_id: str) -> Path:
    return company_base(out_dir, company_id)


def delete_company_output_dir(out_dir: Path, company_id: str) -> Tuple[bool, bool]:
    """
    Returns (deleted, missing).
    Raises on deletion failure.
    """
    base = company_output_dir(out_dir, company_id)
    if not base.exists():
        return (False, True)
    if not base.is_dir():
        raise RuntimeError(f"output path exists but is not a directory: {base}")
    try:
        shutil.rmtree(base)
    except Exception as e:
        raise RuntimeError(f"failed to delete output dir: {base}") from e
    return (True, False)


def crawl_meta_candidates(out_dir: Path, company_id: str) -> List[Path]:
    base = company_base(out_dir, company_id)
    return [
        base / "meta" / "crawl_meta.json",
        base / "metadata" / "crawl_meta.json",
    ]


def url_index_candidates(out_dir: Path, company_id: str) -> List[Path]:
    base = company_base(out_dir, company_id)
    return [
        base / "url_index.json",
        base / "metadata" / "url_index.json",
    ]


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None
