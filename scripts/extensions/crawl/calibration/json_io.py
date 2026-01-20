from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except Exception:
        return None


def bytes_head_hex(b: bytes, n: int = 80) -> str:
    h = b[:n]
    return h.hex()


def bytes_text_preview(b: bytes, n: int = 200) -> str:
    try:
        s = b[:n].decode("utf-8", errors="replace")
    except Exception:
        return ""
    return s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _is_probably_binary_nul(b: bytes) -> bool:
    if not b:
        return False
    if b[:16] == b"\x00" * 16:
        return True
    early = b[:256]
    if b"\x00" in early:
        for ch in early:
            if ch in (9, 10, 13, 32):
                continue
            return ch == 0
        return True
    return False


def validate_json_bytes(path: Path) -> Tuple[bool, str]:
    """
    Returns (ok, reason_if_bad).
    """
    b = read_bytes(path)
    if b is None:
        return False, "unreadable"
    if len(b) == 0:
        return False, "empty"
    if _is_probably_binary_nul(b):
        return False, "nul_bytes_prefix_or_early"
    try:
        txt = b.decode("utf-8")
    except Exception:
        return False, "non_utf8_bytes"
    if txt.strip() == "":
        return False, "whitespace_only"
    try:
        obj = json.loads(txt)
    except Exception:
        return False, "json_parse_error"
    if not isinstance(obj, dict):
        return False, "json_not_dict"
    return True, ""


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        obj = json.loads(txt)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def write_json_file(path: Path, obj: Mapping[str, Any], *, pretty: bool) -> None:
    # Intentionally only touches existing files (calibration should not invent new files)
    if not path.exists():
        return
    if pretty:
        data = json.dumps(obj, ensure_ascii=False, indent=2)
    else:
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(data, encoding="utf-8")
