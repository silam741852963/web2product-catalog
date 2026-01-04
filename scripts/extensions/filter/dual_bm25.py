from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Mapping

from crawl4ai.deep_crawling.filters import URLFilter
from crawl4ai.deep_crawling.scorers import URLScorer
from crawl4ai.utils import HeadPeekr

__all__ = [
    "DualBM25Config",
    "DualBM25Filter",
    "DualBM25Scorer",
    "build_dual_bm25_components",
]

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass
class DualBM25Config:
    threshold: Optional[float] = 0.3
    alpha: float = 0.5
    k1: float = 1.2
    b: float = 0.75
    avgdl: int = 1000


class _DualBM25Core:
    __slots__ = ("pos_terms", "neg_terms", "pos_term_set", "neg_term_set", "cfg")

    def __init__(
        self,
        positive_query: str,
        negative_query: Optional[str] = None,
        cfg: Optional[DualBM25Config] = None,
    ) -> None:
        self.pos_terms: List[str] = _tokenize(positive_query)
        self.neg_terms: List[str] = _tokenize(negative_query) if negative_query else []
        self.pos_term_set = set(self.pos_terms)
        self.neg_term_set = set(self.neg_terms)
        self.cfg: DualBM25Config = cfg or DualBM25Config()

    def _build_tf(self, document: str) -> tuple[Dict[str, int], int]:
        doc_terms = _tokenize(document)
        doc_len = len(doc_terms)
        if doc_len == 0:
            return {}, 0
        tf: Dict[str, int] = {}
        for t in doc_terms:
            tf[t] = tf.get(t, 0) + 1
        return tf, doc_len

    def _bm25_from_tf(
        self, tf: Dict[str, int], doc_len: int, term_set: set[str]
    ) -> float:
        if not tf or not term_set or doc_len <= 0:
            return 0.0

        k1 = self.cfg.k1
        b = self.cfg.b
        avgdl = float(self.cfg.avgdl) if self.cfg.avgdl > 0 else float(doc_len)

        score = 0.0
        for term in term_set:
            freq = tf.get(term, 0)
            if freq == 0:
                continue

            # Within-doc IDF surrogate (kept as in your implementation)
            idf = math.log((1.0 + 1.0) / (freq + 0.5) + 1.0)

            numerator = freq * (k1 + 1.0)
            denominator = freq + k1 * (1.0 - b + b * (doc_len / avgdl))
            score += idf * (numerator / max(denominator, 1e-9))

        return score

    @staticmethod
    def _normalize_score(raw: float) -> float:
        return 1.0 - math.exp(-max(0.0, float(raw)))

    def score_document(self, document: str) -> Dict[str, float]:
        tf, doc_len = self._build_tf(document)

        raw_pos = self._bm25_from_tf(tf, doc_len, self.pos_term_set)
        raw_neg = (
            self._bm25_from_tf(tf, doc_len, self.neg_term_set)
            if self.neg_term_set
            else 0.0
        )

        pos_norm = self._normalize_score(raw_pos)
        neg_norm = self._normalize_score(raw_neg)

        alpha = float(self.cfg.alpha)
        combined = alpha * pos_norm + (1.0 - alpha) * (1.0 - neg_norm)

        return {
            "combined": combined,
            "raw_pos": raw_pos,
            "pos_norm": pos_norm,
            "raw_neg": raw_neg,
            "neg_norm": neg_norm,
        }

    def score_head_and_url(
        self, *, url: str, head_doc: Optional[str]
    ) -> Dict[str, float]:
        if head_doc and head_doc.strip():
            head_scores = self.score_document(head_doc)
        else:
            head_scores = {
                "combined": 0.0,
                "raw_pos": 0.0,
                "pos_norm": 0.0,
                "raw_neg": 0.0,
                "neg_norm": 0.0,
            }

        url_scores = self.score_document(url)

        head_combined = head_scores["combined"]
        url_combined = url_scores["combined"]

        head_empty = not head_doc or not head_doc.strip()
        url_neg_raw = url_scores["raw_neg"]

        head_has_pos = head_scores["raw_pos"] > 0.0
        head_has_neg = head_scores["raw_neg"] > 0.0
        head_neutral = not head_has_pos and not head_has_neg

        if head_empty and url_neg_raw > 0.0:
            fused_combined = 0.0
        elif head_neutral and url_neg_raw > 0.0:
            fused_combined = 0.0
        else:
            fused_combined = max(head_combined, url_combined)

        return {
            "head_combined": head_combined,
            "head_raw_pos": head_scores["raw_pos"],
            "head_pos_norm": head_scores["pos_norm"],
            "head_raw_neg": head_scores["raw_neg"],
            "head_neg_norm": head_scores["neg_norm"],
            "url_combined": url_combined,
            "url_raw_pos": url_scores["raw_pos"],
            "url_pos_norm": url_scores["pos_norm"],
            "url_raw_neg": url_scores["raw_neg"],
            "url_neg_norm": url_scores["neg_norm"],
            "fused_combined": fused_combined,
        }


class DualBM25Filter(URLFilter):
    __slots__ = ("_core",)

    def __init__(
        self,
        positive_query: str,
        negative_query: Optional[str] = None,
        *,
        cfg: Optional[DualBM25Config] = None,
        name: Optional[str] = None,
    ) -> None:
        try:
            super().__init__(name=name or "DualBM25Filter")
        except TypeError:  # pragma: no cover
            super().__init__()
            self.name = name or "DualBM25Filter"  # type: ignore[attr-defined]
        self._core = _DualBM25Core(positive_query, negative_query, cfg=cfg)

    def _build_document(self, html: str) -> str:
        title = HeadPeekr.get_title(html) or ""
        meta: Dict[str, str] = HeadPeekr.extract_meta_tags(html)

        parts: List[str] = []
        if title:
            parts.append((title + " ") * 3)
        desc = meta.get("description", "")
        if desc:
            parts.append((desc + " ") * 2)
        kw = meta.get("keywords", "")
        if kw:
            parts.append(kw)
        if meta:
            parts.append(" ".join(meta.values()))

        return " ".join(parts)

    async def apply(self, url: str) -> bool:
        head_html: Optional[str] = None
        try:
            head_html = await HeadPeekr.peek_html(url)
        except Exception:
            head_html = None

        head_doc: Optional[str] = self._build_document(head_html) if head_html else None

        scores = self._core.score_head_and_url(url=url, head_doc=head_doc)
        fused_combined = scores["fused_combined"]

        if self._core.cfg.threshold is not None:
            decision = fused_combined >= self._core.cfg.threshold
        else:
            decision = True

        self._update_stats(decision)
        return decision


class DualBM25Scorer(URLScorer):
    __slots__ = ("_core", "_doc_index")

    def __init__(
        self,
        positive_query: str,
        negative_query: Optional[str] = None,
        *,
        cfg: Optional[DualBM25Config] = None,
        doc_index: Optional[Mapping[str, str]] = None,
        weight: float = 1.0,
    ) -> None:
        try:
            super().__init__(weight=weight)
        except TypeError:  # pragma: no cover
            super().__init__()
            try:
                self.weight = weight  # type: ignore[attr-defined]
            except Exception:
                pass

        self._core = _DualBM25Core(positive_query, negative_query, cfg=cfg)
        self._doc_index = doc_index

    def _get_document_for_url(self, url: str) -> Optional[str]:
        if self._doc_index is not None:
            doc = self._doc_index.get(url)
            if doc:
                return doc
        return None

    def _calculate_score(self, url: str) -> float:
        head_doc = self._get_document_for_url(url)
        scores = self._core.score_head_and_url(url=url, head_doc=head_doc)
        return float(scores["fused_combined"])


def build_dual_bm25_components() -> Dict[str, Any]:
    from configs.language import default_language_factory

    product_tokens: List[str] = default_language_factory.get("PRODUCT_TOKENS", []) or []
    exclude_tokens: List[str] = default_language_factory.get("EXCLUDE_TOKENS", []) or []

    positive_query = " ".join(sorted(set(product_tokens)))
    negative_query = " ".join(sorted(set(exclude_tokens)))

    scorer_cfg = DualBM25Config(
        threshold=None,
        alpha=0.7,
        k1=1.2,
        b=0.75,
        avgdl=1000,
    )

    filter_cfg = DualBM25Config(
        threshold=0.5,
        alpha=0.5,
        k1=1.2,
        b=0.75,
        avgdl=1000,
    )

    url_scorer = DualBM25Scorer(
        positive_query=positive_query,
        negative_query=negative_query,
        cfg=scorer_cfg,
        doc_index=None,
        weight=1.0,
    )

    url_filter = DualBM25Filter(
        positive_query=positive_query,
        negative_query=negative_query,
        cfg=filter_cfg,
        name="DualBM25Filter",
    )

    return {
        "positive_query": positive_query,
        "negative_query": negative_query,
        "scorer_cfg": scorer_cfg,
        "filter_cfg": filter_cfg,
        "url_scorer": url_scorer,
        "url_filter": url_filter,
    }
