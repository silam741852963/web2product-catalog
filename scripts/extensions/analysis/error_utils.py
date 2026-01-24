from __future__ import annotations

import re
from collections import Counter
from typing import List


_TIMEOUT_RE = re.compile(r"\b(timeout|timed out|read timeout|connect timeout)\b", re.I)
_DNS_RE = re.compile(
    r"\b(nxdomain|name or service not known|dns|nodename nor servname)\b", re.I
)
_SSL_RE = re.compile(r"\b(ssl|tls|certificate verify failed|handshake)\b", re.I)
_CONN_RE = re.compile(
    r"\b(connection (reset|refused)|econnreset|econnrefused|socket hang up)\b", re.I
)
_403_RE = re.compile(r"\b(403|forbidden)\b", re.I)
_404_RE = re.compile(r"\b(404|not found)\b", re.I)
_429_RE = re.compile(r"\b(429|too many requests|rate limit)\b", re.I)
_5XX_RE = re.compile(
    r"\b(5\d\d|server error|bad gateway|service unavailable|gateway timeout)\b", re.I
)


def truncate_label(s: str, max_len: int = 60) -> str:
    t = (s or "").strip()
    if len(t) <= max_len:
        return t
    if max_len <= 1:
        return "…"
    return t[: max_len - 1].rstrip() + "…"


def error_signature(error_text: str) -> str:
    """
    Best-effort signature mapping for analysis. Keep coarse buckets to stabilize top-k.
    """
    t = (error_text or "").strip()
    if not t:
        return ""

    if _TIMEOUT_RE.search(t):
        return "TIMEOUT"
    if _DNS_RE.search(t):
        return "DNS"
    if _SSL_RE.search(t):
        return "SSL"
    if _CONN_RE.search(t):
        return "CONNECTION"
    if _429_RE.search(t):
        return "HTTP_429"
    if _403_RE.search(t):
        return "HTTP_403"
    if _404_RE.search(t):
        return "HTTP_404"
    if _5XX_RE.search(t):
        return "HTTP_5XX"

    # Fallback: normalize to an uppercased short-ish label (avoid huge cardinality)
    up = re.sub(r"\s+", " ", t.upper())
    up = re.sub(r"[^A-Z0-9 _:/.-]", "", up)
    return truncate_label(up, max_len=15)


def top_k_pairs(counter: Counter[str], *, k: int = 20) -> List[List[object]]:
    """
    Deterministic top-k: count desc, key asc.
    Returns list-of-[key, count] for JSON friendliness.
    """
    items = sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [[k_, int(v_)] for (k_, v_) in items[: int(k)]]
