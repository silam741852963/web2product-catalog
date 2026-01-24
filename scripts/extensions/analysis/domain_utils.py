from __future__ import annotations

from collections import Counter
from typing import List
from urllib.parse import urlparse


def _best_effort_registrable(host: str) -> str:
    host = (host or "").lower().strip(".")
    if not host:
        return ""

    parts = host.split(".")
    if len(parts) <= 2:
        return host

    # minimal multi-part public suffix handling (no PSL dependency)
    multi = {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "com.au",
        "net.au",
        "org.au",
        "co.jp",
        "ne.jp",
        "or.jp",
        "co.kr",
        "com.br",
        "com.cn",
        "com.tw",
    }
    tld2 = ".".join(parts[-2:])
    tld3 = ".".join(parts[-3:])

    if tld2 in multi and len(parts) >= 3:
        return ".".join(parts[-3:])
    if tld3 in multi and len(parts) >= 4:
        return ".".join(parts[-4:])
    return ".".join(parts[-2:])


def extract_site_key(url: str) -> str:
    """
    Stable key for grouping domains in analysis.
    Returns values like:
      - example.com
      - (non-http)
      - (no-host)
      - (invalid-url)
    """
    try:
        p = urlparse(url)
    except Exception:
        return "(invalid-url)"

    if p.scheme not in ("http", "https"):
        return "(non-http)"

    host = (p.hostname or "").lower().strip(".")
    if not host:
        return "(no-host)"

    return _best_effort_registrable(host)


def normalized_site_key(url: str) -> str:
    """
    Normalization used for *onsite vs offsite* comparisons.
    Must be IDENTICAL for:
      - root URL key
      - per-url key
    """
    try:
        p = urlparse(url)
    except Exception:
        return ""

    if p.scheme not in ("http", "https"):
        return ""

    host = (p.hostname or "").lower().strip(".")
    if not host:
        return ""

    return _best_effort_registrable(host)


def root_site_key_from_url(root_url: str) -> str:
    """
    Root normalization (same method as per-url normalization).
    """
    return normalized_site_key(root_url)


def top_k_pairs(counter: Counter[str], *, k: int = 20) -> List[List[object]]:
    """
    Deterministic top-k: count desc, key asc.
    Returns list-of-[key, count] for JSON friendliness.
    """
    items = sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [[k_, int(v_)] for (k_, v_) in items[: int(k)]]
