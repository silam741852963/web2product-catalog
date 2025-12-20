from __future__ import annotations

import re
from typing import List, Tuple

_WS = re.compile(r"\s+")
_TRAIL_PUNCT = re.compile(r"[\s\-\–\—:;,.|/]+$")

DENYLIST_SUBSTRINGS = [
    "learn more",
    "contact us",
    "privacy",
    "terms of use",
    "accessibility",
    "cookie",
    "reject all",
    "accept all",
]


def clean_ws(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def norm_name(s: str) -> str:
    s = clean_ws(s).lower()
    s = _TRAIL_PUNCT.sub("", s)
    return s


def norm_desc(s: str) -> str:
    return clean_ws(s)


def norm_type(s: str) -> str:
    t = clean_ws(s).lower()
    if t in {"product", "products", "brand", "line", "goods"}:
        return "product"
    if t in {
        "service",
        "services",
        "solution",
        "solutions",
        "capability",
        "capabilities",
    }:
        return "service"
    if "service" in t or "solution" in t or "capabilit" in t:
        return "service"
    return "product"


def is_denylisted(text: str) -> bool:
    s = (text or "").lower()
    return any(x in s for x in DENYLIST_SUBSTRINGS)


def split_sentences(text: str) -> List[str]:
    t = clean_ws(text)
    if not t:
        return []

    parts: List[str] = []
    for chunk in re.split(r"(?:\n|\r|•|\*|- )+", t):
        c = clean_ws(chunk)
        if c:
            parts.append(c)

    out: List[str] = []
    for p in parts:
        for s in re.split(r"(?<=[.!?])\s+", p):
            s2 = clean_ws(s)
            if s2:
                out.append(s2)

    seen = set()
    final: List[str] = []
    for s in out:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            final.append(s)
    return final


def evidence_probe(markdown_body: str, raw_name: str, raw_desc: str) -> bool:
    body = (markdown_body or "").lower()
    if not body:
        return False

    name = clean_ws(raw_name)
    if name:
        if name.lower() in body:
            return True

        toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) >= 3][:8]
        if len(toks) >= 2:
            hit = sum(1 for t in toks if t in body)
            if hit >= 2:
                return True

    desc = clean_ws(raw_desc)
    if desc:
        toks = [t for t in re.split(r"[^a-z0-9]+", desc.lower()) if len(t) >= 3][:8]
        if toks:
            phrase = " ".join(toks)
            if phrase and phrase in body:
                return True
            hit = sum(1 for t in toks if t in body)
            if hit >= 3:
                return True

    return False


def descriptiveness_score(name: str) -> Tuple[int, int, int]:
    n = clean_ws(name)
    tokens = n.split()
    has_upper = any(c.isupper() for c in n)
    has_digit = any(c.isdigit() for c in n)
    return (min(len(n), 80), len(tokens), int(has_upper) + int(has_digit))
