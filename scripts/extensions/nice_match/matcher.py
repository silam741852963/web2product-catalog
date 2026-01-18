from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from extensions.nice_match.nice_token_normalizer import NiceToken


_WORD_SPLIT_RE = re.compile(r"[A-Za-z0-9]+", re.ASCII)


def normalize_for_match(s: str) -> str:
    t = (s or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _alias_to_regex(alias: str) -> Optional[re.Pattern[str]]:
    """
    Build a deterministic case-insensitive regex over the ORIGINAL sentence text.
    Returns None if alias cannot be tokenized into any alnum words (skip deterministically).
    """
    words = _WORD_SPLIT_RE.findall((alias or "").lower())
    if len(words) == 0:
        return None

    inner = r"[^a-z0-9]+".join(re.escape(w) for w in words)
    pat = rf"\b{inner}\b"
    try:
        return re.compile(pat, flags=re.IGNORECASE)
    except re.error:
        return None


@dataclass(frozen=True, slots=True)
class PatternRef:
    token: NiceToken
    alias_raw: str
    alias_norm: str
    first_word: str
    regex: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class MatchHit:
    token: NiceToken
    matched_alias: str
    spans: list[tuple[int, int]]  # (start, end) offsets in ORIGINAL sentence text


def build_pattern_index(tokens: list[NiceToken]) -> dict[str, list[PatternRef]]:
    idx: dict[str, list[PatternRef]] = {}

    for t in tokens:
        for alias in t.aliases:
            a_norm = normalize_for_match(alias)
            if a_norm == "":
                continue

            rx = _alias_to_regex(alias)
            if rx is None:
                continue

            first = a_norm.split(" ", 1)[0]
            pr = PatternRef(
                token=t,
                alias_raw=alias,
                alias_norm=a_norm,
                first_word=first,
                regex=rx,
            )
            idx.setdefault(first, []).append(pr)

    for k in idx:
        idx[k] = sorted(idx[k], key=lambda p: (p.alias_norm, p.token.token_id))
    return idx


def match_sentence_hits(
    *, sentence_text: str, index: dict[str, list[PatternRef]]
) -> list[MatchHit]:
    s = sentence_text or ""
    s_norm = normalize_for_match(s)
    if s_norm == "":
        return []

    words = set(s_norm.split(" "))

    # token_id -> best PatternRef (longest alias_norm; tie: lexicographic)
    best: dict[str, PatternRef] = {}

    for w in sorted(words):
        cand = index.get(w)
        if cand is None:
            continue
        for pr in cand:
            spans = [(m.start(), m.end()) for m in pr.regex.finditer(s)]
            if len(spans) == 0:
                continue

            cur = best.get(pr.token.token_id)
            if cur is None:
                best[pr.token.token_id] = pr
            else:
                if len(pr.alias_norm) > len(cur.alias_norm):
                    best[pr.token.token_id] = pr
                elif (
                    len(pr.alias_norm) == len(cur.alias_norm)
                    and pr.alias_norm < cur.alias_norm
                ):
                    best[pr.token.token_id] = pr

    out: list[MatchHit] = []
    for tid in sorted(best.keys()):
        pr = best[tid]
        spans = [(m.start(), m.end()) for m in pr.regex.finditer(s)]
        if len(spans) == 0:
            # Deterministic safety: skip instead of raising.
            continue
        out.append(MatchHit(token=pr.token, matched_alias=pr.alias_raw, spans=spans))
    return out
