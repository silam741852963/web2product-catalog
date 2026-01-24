from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from extensions.io.output_paths import ensure_output_root


@dataclass(frozen=True, slots=True)
class AnalysisPaths:
    analysis_root: Path
    report_json: Path
    report_html: Path
    charts_spec_dir: Path
    charts_png_dir: Path
    assets_dir: Path


def build_analysis_paths(*, out_dir: str | Path, run_tag: str) -> AnalysisPaths:
    root = ensure_output_root(Path(out_dir))
    analysis_root = root / "analysis" / run_tag
    charts_spec_dir = analysis_root / "charts" / "spec"
    charts_png_dir = analysis_root / "charts" / "png"
    assets_dir = analysis_root / "assets"

    return AnalysisPaths(
        analysis_root=analysis_root,
        report_json=analysis_root / "report.json",
        report_html=analysis_root / "report.html",
        charts_spec_dir=charts_spec_dir,
        charts_png_dir=charts_png_dir,
        assets_dir=assets_dir,
    )


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: AnalysisPaths) -> None:
    paths.analysis_root.mkdir(parents=True, exist_ok=True)
    paths.charts_spec_dir.mkdir(parents=True, exist_ok=True)
    paths.charts_png_dir.mkdir(parents=True, exist_ok=True)
    paths.assets_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    """
    Atomic, deterministic JSON write.
    - UTF-8
    - sort_keys=True
    - compact-ish but readable
    """
    _ensure_parent_dir(path)
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")

    # atomic replace on same filesystem
    d = path.parent
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(d))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def write_bytes(path: Path, data: bytes) -> None:
    """
    Atomic bytes write (used for PNG etc.)
    """
    _ensure_parent_dir(path)

    d = path.parent
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(d))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
