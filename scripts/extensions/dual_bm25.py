# dual_bm25.py
from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Callable, Optional

from extensions.filtering import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_QUERY_KEYS,
    UNIVERSAL_EXTERNALS,
)
from config import language_settings as langcfg

__all__ = [
    "build_default_negative_terms",
    "build_default_negative_query",
    "DualBM25Config",
    "DualBM25Combiner",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------
_FILE_EXT_STOP = { "pdf","ps","rtf","doc","docx","ppt","pptx","pps","xls","xlsx",
                  "csv","tsv","xml", "json","yml","yaml","md","rst","zip","rar",
                  "7z","gz","bz2","xz","tar","tgz","dmg", "exe","msi","apk","jpg",
                  "jpeg","png","gif","svg","webp","bmp","tif","tiff","ico", "mp3",
                  "wav","m4a","ogg","flac","aac","mp4","m4v","webm","mov","avi",
                  "wmv","mkv", "ics","eot","ttf","otf","woff","woff2","map", }

_STOPWORDS: Set[str] = set(langcfg.get_stopwords())

_SPLIT_RX = re.compile(r"[^a-z0-9]+")

def _basic_tokens(s: str) -> List[str]:
    s = (s or "").lower().strip()
    if not s:
        return []
    parts = [p for p in _SPLIT_RX.split(s) if p]
    return parts

def _pattern_to_tokens(p: str) -> List[str]:
    raw = (p or "").lower().strip()
    if not raw:
        return []

    # strip glob / regex syntactic noise to expose textual tokens
    raw = raw.replace("*", " ")
    raw = raw.replace("?", " ")
    raw = raw.replace("[", " ").replace("]", " ")
    raw = raw.replace("{", " ").replace("}", " ")
    raw = raw.replace("(", " ").replace(")", " ")
    raw = raw.replace("|", " ")
    raw = raw.replace("re:", " ").replace("regex:", " ")

    toks = _basic_tokens(raw)

    out: List[str] = []
    for t in toks:
        if not t or t in _STOPWORDS:
            continue
        if t.isdigit() or len(t) <= 2:
            continue
        if t in _FILE_EXT_STOP:
            continue
        out.append(t)
    return out

def _domain_to_token(host: str) -> str:
    h = (host or "").lower().strip().strip(".")
    if not h:
        return ""
    labels = [x for x in h.split(".") if x]
    if not labels:
        return ""
    base = labels[-2] if len(labels) >= 2 else labels[-1]
    base = base.strip("-_")
    return base if base and base not in _STOPWORDS else ""

# ---------------------------------------------------------------------------
# Default negative terms derived from filtering policy
# ---------------------------------------------------------------------------

def build_default_negative_terms(
    *,
    extra_patterns: Optional[Iterable[str]] = None,
    extra_query_keys: Optional[Iterable[str]] = None,
    include_universal_external_hosts: bool = True,
    min_len: int = 3,
    limit: Optional[int] = 128,
) -> List[str]:
    seen: Set[str] = set()
    collected: List[str] = []

    pattern_sources: List[str] = list(DEFAULT_EXCLUDE_PATTERNS)
    if extra_patterns:
        pattern_sources.extend(list(extra_patterns))
    for p in pattern_sources:
        for tok in _pattern_to_tokens(p):
            if len(tok) < min_len:
                continue
            if tok not in seen:
                seen.add(tok)
                collected.append(tok)

    qkeys = list(DEFAULT_EXCLUDE_QUERY_KEYS)
    if extra_query_keys:
        qkeys.extend(list(extra_query_keys))
    for k in qkeys:
        k = (k or "").lower().strip()
        if not k or k in _STOPWORDS:
            continue
        if len(k) < min_len:
            continue
        if k not in seen:
            seen.add(k)
            collected.append(k)

    if include_universal_external_hosts:
        for host in sorted(UNIVERSAL_EXTERNALS):
            tok = _domain_to_token(host)
            if tok and len(tok) >= min_len and tok not in seen:
                seen.add(tok)
                collected.append(tok)

    if limit is not None and limit > 0 and len(collected) > limit:
        collected = collected[:limit]

    if log.isEnabledFor(logging.DEBUG):
        log.debug("[dual_bm25] built %d negative tokens; sample=%s",
                  len(collected), collected[:12])
    return collected

def build_default_negative_query(**kwargs) -> str:
    q = " ".join(build_default_negative_terms(**kwargs))
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[dual_bm25] negative_query terms=%d len=%d", len(q.split()), len(q))
    return q

# ---------------------------------------------------------------------------
# Dual BM25 combiner (agnostic to underlying BM25 implementation)
# ---------------------------------------------------------------------------

@dataclass
class DualBM25Config:
    alpha: float = 0.6
    score_norm: Callable[[float], float] = staticmethod(lambda s: 1.0 - math.exp(-max(0.0, float(s))))
    clamp: bool = True

class DualBM25Combiner:
    """
    Usage:
        comb = DualBM25Combiner(bm25_fn)
        score = comb.score(doc_text, pos_query="product service solutions", neg_query=None)
    """
    def __init__(
        self,
        bm25_fn: Callable[[str, str], float],
        cfg: Optional[DualBM25Config] = None,
    ) -> None:
        self.bm25_fn = bm25_fn
        self.cfg = cfg or DualBM25Config()
        self._dbg_counter = 0  # limit per-call debug spam
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[dual_bm25] init alpha=%.2f clamp=%s", self.cfg.alpha, self.cfg.clamp)

    def score(
        self,
        text: str,
        *,
        pos_query: Optional[str],
        neg_query: Optional[str] = None,
    ) -> float:
        if not text:
            return 0.0

        raw_pos = self.bm25_fn(text, pos_query) if pos_query else 0.0
        pos = self.cfg.score_norm(raw_pos)

        neg_q = neg_query if (neg_query is not None and neg_query.strip()) else build_default_negative_query()
        raw_neg = self.bm25_fn(text, neg_q) if neg_q else 0.0
        neg = self.cfg.score_norm(raw_neg)

        alpha = float(self.cfg.alpha)
        combined = alpha * pos + (1.0 - alpha) * (1.0 - neg)
        if self.cfg.clamp:
            combined = max(0.0, min(1.0, combined))

        # Log first N calls at DEBUG to avoid flooding
        if log.isEnabledFor(logging.DEBUG) and self._dbg_counter < 50:
            self._dbg_counter += 1
            log.debug(
                "[dual_bm25] #%d pos_raw=%.4f pos=%.4f neg_raw=%.4f neg=%.4f alpha=%.2f => combined=%.4f",
                self._dbg_counter, raw_pos, pos, raw_neg, neg, alpha, combined
            )
        return combined