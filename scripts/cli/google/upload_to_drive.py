from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests


"""
Google Drive uploader CLI.

Authentication and Drive routing are configured ONLY through environment variables.

Required environment variables:
  - GOOGLE_DRIVE_ACCESS_TOKEN
      OAuth2 access token used as Bearer token for Google Drive API calls.
      This tool does not attempt to refresh tokens. Provide a valid token at runtime.

Optional environment variables:
  - GOOGLE_DRIVE_API_BASE
      Base URL for Google APIs.
      Default: https://www.googleapis.com

Notes for Shared Drives / Drives:
  - If uploading into a Shared Drive, you normally need:
      * a destination folder id that exists inside that Shared Drive
      * and you must pass --drive-id <SHARED_DRIVE_ID> so listing/creation is scoped properly
    This tool will always set supportsAllDrives=true on API calls. It will fail loudly
    on permission issues.

Examples:
  Upload a file to My Drive root:
    python -m cli.google.upload_to_drive --src path/to/file.csv --parent-id root

  Upload a folder (replicate tree by creating folders in Drive):
    python -m cli.google.upload_to_drive --src outputs --parent-id <FOLDER_ID> --name outputs_backup

  Upload a folder as a single zip:
    python -m cli.google.upload_to_drive --src outputs --parent-id <FOLDER_ID> --name outputs_backup --zip-folder
"""


@dataclass(frozen=True, slots=True)
class DriveEnv:
    access_token: str
    api_base: str


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[gdrive-upload {ts}] {msg}", flush=True)


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _load_env() -> DriveEnv:
    access_token = _require_env("GOOGLE_DRIVE_ACCESS_TOKEN")
    api_base = os.environ.get(
        "GOOGLE_DRIVE_API_BASE", "https://www.googleapis.com"
    ).strip()
    if api_base == "":
        raise RuntimeError("GOOGLE_DRIVE_API_BASE is set but empty.")
    return DriveEnv(access_token=access_token, api_base=api_base)


def _auth_headers(env: DriveEnv) -> dict[str, str]:
    return {"Authorization": f"Bearer {env.access_token}"}


def _drive_files_create(
    env: DriveEnv,
    *,
    name: str,
    mime_type: str,
    parent_id: str,
    drive_id: Optional[str],
) -> str:
    url = f"{env.api_base}/drive/v3/files"
    params: dict[str, str] = {"supportsAllDrives": "true"}
    if drive_id is not None:
        params["supportsAllDrives"] = "true"

    payload = {"name": name, "mimeType": mime_type, "parents": [parent_id]}

    resp = requests.post(
        url,
        headers={
            **_auth_headers(env),
            "Content-Type": "application/json; charset=UTF-8",
        },
        params=params,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    file_id = data.get("id")
    if not isinstance(file_id, str) or file_id.strip() == "":
        raise RuntimeError(f"Drive create did not return a valid id. Response: {data}")
    return file_id


def _initiate_resumable_upload(
    env: DriveEnv,
    *,
    name: str,
    parent_id: str,
    drive_id: Optional[str],
    mime_type: str,
    content_length: int,
) -> str:
    url = f"{env.api_base}/upload/drive/v3/files"
    params: dict[str, str] = {"uploadType": "resumable", "supportsAllDrives": "true"}
    if drive_id is not None:
        params["supportsAllDrives"] = "true"

    metadata = {"name": name, "parents": [parent_id]}

    headers = {
        **_auth_headers(env),
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": mime_type,
        "X-Upload-Content-Length": str(content_length),
    }

    resp = requests.post(
        url,
        headers=headers,
        params=params,
        data=json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        timeout=120,
    )
    resp.raise_for_status()

    location = resp.headers.get("Location")
    if location is None or location.strip() == "":
        raise RuntimeError(
            "Resumable upload initiation did not return Location header."
        )
    return location


def _upload_resumable_bytes(
    *,
    upload_url: str,
    mime_type: str,
    content_length: int,
    file_path: Path,
) -> dict:
    with file_path.open("rb") as f:
        resp = requests.put(
            upload_url,
            headers={"Content-Type": mime_type, "Content-Length": str(content_length)},
            data=f,
            timeout=3600,
        )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected upload response JSON: {data}")
    return data


def _guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def _zip_folder(src_dir: Path, *, zip_path: Path, root_name: str) -> None:
    if zip_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(src_dir)
            arcname = str(Path(root_name) / rel)
            zf.write(p, arcname=arcname)


def _iter_local_files_in_tree(src_dir: Path) -> Iterable[Path]:
    for p in sorted(src_dir.rglob("*")):
        if p.is_file():
            yield p


def _ensure_drive_folder_tree(
    env: DriveEnv,
    *,
    local_root: Path,
    drive_parent_id: str,
    drive_id: Optional[str],
    root_name: str,
) -> str:
    _log(f"Creating Drive folder '{root_name}' under parent_id={drive_parent_id}")
    root_folder_id = _drive_files_create(
        env,
        name=root_name,
        mime_type="application/vnd.google-apps.folder",
        parent_id=drive_parent_id,
        drive_id=drive_id,
    )
    _log(f"Created Drive folder id={root_folder_id}")

    folder_cache: dict[Path, str] = {Path("."): root_folder_id}

    for file_path in _iter_local_files_in_tree(local_root):
        rel_parent = file_path.parent.relative_to(local_root)
        if rel_parent not in folder_cache:
            # Create intermediate folders deterministically in depth order.
            parts = list(rel_parent.parts)
            cur = Path(".")
            for part in parts:
                cur = cur / part
                if cur in folder_cache:
                    continue
                parent_drive_id = folder_cache[cur.parent]
                folder_id = _drive_files_create(
                    env,
                    name=part,
                    mime_type="application/vnd.google-apps.folder",
                    parent_id=parent_drive_id,
                    drive_id=drive_id,
                )
                folder_cache[cur] = folder_id

    return root_folder_id


def _upload_file(
    env: DriveEnv,
    *,
    file_path: Path,
    parent_id: str,
    drive_id: Optional[str],
    remote_name: str,
) -> str:
    if not file_path.exists():
        raise RuntimeError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise RuntimeError(f"Path is not a file: {file_path}")

    mime_type = _guess_mime(file_path)
    size = file_path.stat().st_size

    _log(
        f"Uploading file: local='{file_path}' size={size} mime='{mime_type}' as '{remote_name}'"
    )
    upload_url = _initiate_resumable_upload(
        env,
        name=remote_name,
        parent_id=parent_id,
        drive_id=drive_id,
        mime_type=mime_type,
        content_length=size,
    )
    res = _upload_resumable_bytes(
        upload_url=upload_url,
        mime_type=mime_type,
        content_length=size,
        file_path=file_path,
    )
    file_id = res.get("id")
    if not isinstance(file_id, str) or file_id.strip() == "":
        raise RuntimeError(f"Upload did not return a valid file id. Response: {res}")
    _log(f"Uploaded id={file_id}")
    return file_id


def _upload_folder_tree(
    env: DriveEnv,
    *,
    src_dir: Path,
    parent_id: str,
    drive_id: Optional[str],
    root_name: str,
) -> str:
    if not src_dir.exists():
        raise RuntimeError(f"Folder does not exist: {src_dir}")
    if not src_dir.is_dir():
        raise RuntimeError(f"Path is not a folder: {src_dir}")

    root_folder_id = _ensure_drive_folder_tree(
        env,
        local_root=src_dir,
        drive_parent_id=parent_id,
        drive_id=drive_id,
        root_name=root_name,
    )

    # Recompute a deterministic mapping local-folder -> drive-folder-id as we upload.
    folder_cache: dict[Path, str] = {Path("."): root_folder_id}

    def ensure_folder(rel_folder: Path) -> str:
        if rel_folder in folder_cache:
            return folder_cache[rel_folder]
        parts = list(rel_folder.parts)
        cur = Path(".")
        for part in parts:
            cur = cur / part
            if cur in folder_cache:
                continue
            parent_drive_id = folder_cache[cur.parent]
            folder_id = _drive_files_create(
                env,
                name=part,
                mime_type="application/vnd.google-apps.folder",
                parent_id=parent_drive_id,
                drive_id=drive_id,
            )
            folder_cache[cur] = folder_id
        return folder_cache[rel_folder]

    for file_path in _iter_local_files_in_tree(src_dir):
        rel = file_path.relative_to(src_dir)
        rel_parent = rel.parent
        dest_folder_id = ensure_folder(rel_parent)

        remote_name = file_path.name
        _upload_file(
            env,
            file_path=file_path,
            parent_id=dest_folder_id,
            drive_id=drive_id,
            remote_name=remote_name,
        )

    return root_folder_id


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="upload_to_drive")

    p.add_argument("--src", required=True, help="Local file or folder path to upload.")
    p.add_argument(
        "--parent-id",
        required=True,
        help="Destination Drive folder id (use 'root' for My Drive root).",
    )
    p.add_argument(
        "--drive-id",
        default=None,
        help="Shared Drive id, if uploading to a Shared Drive context.",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Remote name override. For folder upload, this is the created root folder name.",
    )
    p.add_argument(
        "--zip-folder",
        action="store_true",
        help="If src is a folder: zip it first then upload the zip file (single file upload).",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)

    env = _load_env()

    src = Path(ns.src).expanduser().resolve()
    parent_id: str = ns.parent_id
    drive_id: Optional[str] = ns.drive_id
    name: Optional[str] = ns.name
    zip_folder: bool = bool(ns.zip_folder)

    if not src.exists():
        raise RuntimeError(f"--src does not exist: {src}")

    if src.is_file():
        remote_name = name if name is not None else src.name
        _upload_file(
            env,
            file_path=src,
            parent_id=parent_id,
            drive_id=drive_id,
            remote_name=remote_name,
        )
        _log("Done.")
        return 0

    if not src.is_dir():
        raise RuntimeError(f"--src must be a file or directory: {src}")

    root_name = name if name is not None else src.name

    if zip_folder:
        tmp_dir = Path(".").resolve()
        zip_path = (tmp_dir / f"{root_name}.zip").resolve()
        _log(f"Compressing folder '{src}' into '{zip_path}' (zip root='{root_name}')")
        _zip_folder(src, zip_path=zip_path, root_name=root_name)
        try:
            _upload_file(
                env,
                file_path=zip_path,
                parent_id=parent_id,
                drive_id=drive_id,
                remote_name=zip_path.name,
            )
        finally:
            _log(f"Removing local zip '{zip_path}'")
            zip_path.unlink()
        _log("Done.")
        return 0

    _upload_folder_tree(
        env, src_dir=src, parent_id=parent_id, drive_id=drive_id, root_name=root_name
    )
    _log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
