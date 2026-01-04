from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True, slots=True)
class GitVersionInfo:
    repo_root: str
    commit: str
    commit_short: str
    describe: str
    is_dirty: bool


def _run_git(repo_root: Path, args: list[str]) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return p.stdout.strip()


def get_git_version_info(repo_root: Path) -> GitVersionInfo:
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root does not exist: {repo_root}")

    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    commit_short = _run_git(repo_root, ["rev-parse", "--short", "HEAD"])
    describe = _run_git(repo_root, ["describe", "--tags", "--always", "--dirty"])
    dirty_out = _run_git(repo_root, ["status", "--porcelain"])
    is_dirty = bool(dirty_out)

    return GitVersionInfo(
        repo_root=str(repo_root),
        commit=commit,
        commit_short=commit_short,
        describe=describe,
        is_dirty=is_dirty,
    )


def version_metadata_dict(
    info: GitVersionInfo,
    *,
    component: str,
    component_version: Optional[str] = None,
) -> Dict[str, object]:
    if not component:
        raise ValueError("component must be non-empty.")
    d: Dict[str, object] = {
        "component": component,
        "git_repo_root": info.repo_root,
        "git_commit": info.commit,
        "git_commit_short": info.commit_short,
        "git_describe": info.describe,
        "git_is_dirty": info.is_dirty,
    }
    if component_version is not None:
        d["component_version"] = component_version
    return d


# -----------------------------------------------------------------------------
# New helpers: discover repo root + safe metadata (never throw from callers)
# -----------------------------------------------------------------------------

_DEFAULT_ENV_KEYS = ("CRAWL_REPO_ROOT", "REPO_ROOT")


def discover_repo_root(
    *,
    start_path: Optional[Path] = None,
    env_keys: tuple[str, ...] = _DEFAULT_ENV_KEYS,
) -> Optional[Path]:
    """
    Try to locate the git repository root.

    Order:
      1) env vars (CRAWL_REPO_ROOT / REPO_ROOT by default)
      2) walk parents from start_path (or cwd) looking for .git
    """
    # 1) env
    for k in env_keys:
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            p = Path(v.strip()).expanduser().resolve()
            if p.exists():
                return p

    # 2) walk parents from start_path or cwd
    base = start_path or Path.cwd()
    try:
        base = base.expanduser().resolve()
    except Exception:
        base = Path.cwd()

    if base.is_file():
        base = base.parent

    cur = base
    while True:
        git_dir = cur / ".git"
        if git_dir.exists():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    return None


def safe_get_git_version_info(repo_root: Path) -> Optional[GitVersionInfo]:
    """
    Like get_git_version_info, but returns None instead of raising
    when git is unavailable or repo_root isn't a git repo.
    """
    try:
        return get_git_version_info(repo_root)
    except Exception:
        return None


def safe_version_metadata(
    *,
    component: str,
    component_version: Optional[str] = None,
    start_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    env_keys: tuple[str, ...] = _DEFAULT_ENV_KEYS,
) -> Optional[Dict[str, object]]:
    """
    Best-effort metadata dict. Returns None if it can't produce metadata.
    Never raises.
    """
    try:
        rr = repo_root or discover_repo_root(start_path=start_path, env_keys=env_keys)
        if rr is None:
            return None
        info = safe_get_git_version_info(rr)
        if info is None:
            return None
        return version_metadata_dict(
            info, component=component, component_version=component_version
        )
    except Exception:
        return None
