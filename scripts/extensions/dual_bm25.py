from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Mapping

from crawl4ai.deep_crawling.filters import URLFilter
from crawl4ai.deep_crawling.scorers import URLScorer
from crawl4ai.utils import HeadPeekr

__all__ = [
    "DualBM25Config",
    "DualBM25Filter",
    "DualBM25Scorer",
]

logger = logging.getLogger(__name__)

# Precompiled tokenization regex (lowercase alphanumerics)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """
    Simple, fast, case-insensitive tokenization for BM25 scoring.

    Split on non-alphanumeric boundaries so that a URL like:
        "https://mother-parkers.com/coffee-excellence/private-label-formats"
    becomes:
        ["https", "mother", "parkers", "com",
         "coffee", "excellence", "private", "label", "formats"]
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass
class DualBM25Config:
    """
    Configuration for DualBM25 scoring.

    - threshold: final combined score threshold in [0,1] to accept a URL
                 (used by DualBM25Filter; ignored by DualBM25Scorer).
    - alpha: weight for the positive query vs (1-alpha) for the negative.
    - k1, b, avgdl: standard BM25 parameters, tuned for head-only content.
    """
    threshold: Optional[float] = 0.3
    alpha: float = 0.5
    k1: float = 1.2
    b: float = 0.75
    avgdl: int = 1000


class _DualBM25Core:
    """
    Shared BM25 core logic between filter and scorer.

    Base combination for a single document:

        score = alpha * pos_norm + (1 - alpha) * (1 - neg_norm)

    with:
        pos_norm = 1 - exp(-BM25(doc, pos_query))
        neg_norm = 1 - exp(-BM25(doc, neg_query))

    For HEAD+URL fusion, see score_head_and_url().
    """

    __slots__ = (
        "pos_terms",
        "neg_terms",
        "pos_term_set",
        "neg_term_set",
        "cfg",
    )

    def __init__(
        self,
        positive_query: str,
        negative_query: Optional[str] = None,
        cfg: Optional[DualBM25Config] = None,
    ) -> None:
        # Tokenize once and keep both list + set:
        # - list preserved for debugging/inspection if needed
        # - set used for scoring to avoid per-call set() allocations
        self.pos_terms: List[str] = _tokenize(positive_query)
        self.neg_terms: List[str] = (
            _tokenize(negative_query) if negative_query else []
        )

        self.pos_term_set = set(self.pos_terms)
        self.neg_term_set = set(self.neg_terms)

        self.cfg: DualBM25Config = cfg or DualBM25Config()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DualBM25Core.__init__] pos_terms=%d neg_terms=%d alpha=%.3f "
                "k1=%.2f b=%.2f avgdl=%d",
                len(self.pos_terms),
                len(self.neg_terms),
                self.cfg.alpha,
                self.cfg.k1,
                self.cfg.b,
                self.cfg.avgdl,
            )

    # ------------------------------------------------------------------ #
    # Core BM25 helpers (single TF build shared for pos & neg)
    # ------------------------------------------------------------------ #

    def _build_tf(self, document: str) -> tuple[Dict[str, int], int]:
        """
        Tokenize document and build term-frequency map.

        Returns:
            (tf_dict, doc_len_tokens)
        """
        doc_terms = _tokenize(document)
        doc_len = len(doc_terms)
        if doc_len == 0:
            return {}, 0

        tf: Dict[str, int] = {}
        for t in doc_terms:
            tf[t] = tf.get(t, 0) + 1
        return tf, doc_len

    def _bm25_from_tf(
        self,
        tf: Dict[str, int],
        doc_len: int,
        term_set: set[str],
    ) -> float:
        """
        BM25 scoring using a pre-built term-frequency map and a set of query terms.
        """
        if not tf or not term_set or doc_len <= 0:
            return 0.0

        k1 = self.cfg.k1
        b = self.cfg.b
        avgdl = float(self.cfg.avgdl) if self.cfg.avgdl > 0 else float(doc_len)

        score = 0.0
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        matched_terms = 0

        for term in term_set:
            freq = tf.get(term, 0)
            if freq == 0:
                continue

            if debug_enabled:
                matched_terms += 1

            # Simple IDF surrogate using within-doc stats.
            idf = math.log((1.0 + 1.0) / (freq + 0.5) + 1.0)

            numerator = freq * (k1 + 1.0)
            denominator = freq + k1 * (1.0 - b + b * (doc_len / avgdl))
            contribution = idf * (numerator / max(denominator, 1e-9))
            score += contribution

        if debug_enabled:
            logger.debug(
                "[DualBM25Core._bm25] doc_len=%d matched_terms=%d raw_score=%.6f",
                doc_len,
                matched_terms,
                score,
            )

        return score

    @staticmethod
    def _normalize_score(raw: float) -> float:
        """
        Map unbounded BM25 score to [0, 1) via 1 - exp(-max(0, raw)).
        """
        return 1.0 - math.exp(-max(0.0, float(raw)))

    def score_document(self, document: str) -> Dict[str, float]:
        """
        Compute dual BM25 scores from a single pre-built document.

        Returns dict with:
            - combined: final combined score in [0,1]
            - raw_pos, pos_norm, raw_neg, neg_norm
        """
        # Single tokenization + TF map reused for both queries
        tf, doc_len = self._build_tf(document)

        # Positive
        raw_pos = self._bm25_from_tf(tf, doc_len, self.pos_term_set)

        # Negative
        if self.neg_term_set:
            raw_neg = self._bm25_from_tf(tf, doc_len, self.neg_term_set)
        else:
            raw_neg = 0.0

        pos_norm = self._normalize_score(raw_pos)
        neg_norm = self._normalize_score(raw_neg)

        alpha = float(self.cfg.alpha)
        combined = alpha * pos_norm + (1.0 - alpha) * (1.0 - neg_norm)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DualBM25Core.score_document] raw_pos=%.6f pos_norm=%.6f "
                "raw_neg=%.6f neg_norm=%.6f alpha=%.3f combined=%.6f",
                raw_pos,
                pos_norm,
                raw_neg,
                neg_norm,
                alpha,
                combined,
            )

        return {
            "combined": combined,
            "raw_pos": raw_pos,
            "pos_norm": pos_norm,
            "raw_neg": raw_neg,
            "neg_norm": neg_norm,
        }

    # ------------------------------------------------------------------ #
    # HEAD + URL fusion helper (shared by filter & scorer)
    # ------------------------------------------------------------------ #

    def score_head_and_url(
        self,
        *,
        url: str,
        head_doc: Optional[str],
    ) -> Dict[str, float]:
        # Head: allow None/empty ⇒ treated as no signal
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

        # URL: always scored
        url_scores = self.score_document(url)

        head_combined = head_scores["combined"]
        url_combined = url_scores["combined"]

        head_empty = not head_doc or not head_doc.strip()
        url_neg_raw = url_scores["raw_neg"]

        head_has_pos = head_scores["raw_pos"] > 0.0
        head_has_neg = head_scores["raw_neg"] > 0.0
        head_neutral = not head_has_pos and not head_has_neg

        # Option B: strong negative URL signal + no/neutral HEAD ⇒ force drop
        if head_empty and url_neg_raw > 0.0:
            fused_combined = 0.0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[DualBM25Core.score_head_and_url] url=%s head_empty=True "
                    "url_raw_neg=%.6f -> fused_combined=0.000000 (forced drop)",
                    url,
                    url_neg_raw,
                )

        elif head_neutral and url_neg_raw > 0.0:
            fused_combined = 0.0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[DualBM25Core.score_head_and_url] url=%s head_neutral=True "
                    "url_raw_neg=%.6f -> fused_combined=0.000000 (forced drop)",
                    url,
                    url_neg_raw,
                )

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
    """
    Asynchronous URL filter using a "dual" BM25 scheme over HEAD HTML + URL:

        doc_score     = alpha * pos_norm + (1 - alpha) * (1 - neg_norm)
        head_score    = doc_score(head_doc)
        url_score     = doc_score(url_string)
        fused_score   = max(head_score, url_score)

    Special rule (Option B):
        - If HEAD is effectively empty AND the URL has any negative hits
          (e.g. /terms-of-use, /contact-us) → fused_score is forced to 0.

    Decision:
        - If fused_score >= threshold  -> KEEP
        - Else                         -> DROP

    The filter fetches only the head HTML via HeadPeekr.peek_html(url), then
    builds a compact document from:
        - <title> (weight 3)
        - meta description (weight 2)
        - meta keywords + all meta content (weight 1)
    """

    __slots__ = ("_core",)

    def __init__(
        self,
        positive_query: str,
        negative_query: Optional[str] = None,
        *,
        cfg: Optional[DualBM25Config] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "DualBM25Filter")
        self._core = _DualBM25Core(positive_query, negative_query, cfg=cfg)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[DualBM25Filter.__init__] name=%s pos_terms=%d neg_terms=%d "
                "threshold=%s alpha=%.3f k1=%.2f b=%.2f avgdl=%d",
                self.name,
                len(self._core.pos_terms),
                len(self._core.neg_terms),
                (
                    f"{self._core.cfg.threshold:.3f}"
                    if self._core.cfg.threshold is not None
                    else "None"
                ),
                self._core.cfg.alpha,
                self._core.cfg.k1,
                self._core.cfg.b,
                self._core.cfg.avgdl,
            )

    # ------------------------------------------------------------------ #
    # Core doc builder for HEAD HTML
    # ------------------------------------------------------------------ #

    def _build_document(self, html: str) -> str:
        """
        Build a light-weight text representation from head HTML.
        """
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

        doc = " ".join(parts)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[DualBM25Filter._build_document] built doc_len_chars=%d "
                "title_len=%d meta_keys=%d",
                len(doc),
                len(title),
                len(meta),
            )
        return doc

    # ------------------------------------------------------------------ #
    # URLFilter API
    # ------------------------------------------------------------------ #

    async def apply(self, url: str) -> bool:
        """
        Asynchronously decide whether to KEEP the URL using dual BM25
        over both HEAD HTML and the URL string itself.

        Strategy:
          - Try to fetch HEAD HTML and build a head-document.
          - Call _DualBM25Core.score_head_and_url(url, head_doc).
          - fused_score = scores["fused_combined"].
          - If fused_score >= threshold -> KEEP, else DROP.

        If head HTML cannot be fetched or parsed, we pass head_doc=None and rely
        on URL-only scoring with the same fusion rules (including the negative-URL
        + empty-head drop behavior).
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("[DualBM25Filter.apply] url=%s START", url)

        head_html: Optional[str] = None
        try:
            head_html = await HeadPeekr.peek_html(url)
        except Exception as exc:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[DualBM25Filter.apply] url=%s head fetch error: %r "
                    "-> falling back to URL-only scoring",
                    url,
                    exc,
                )
            head_html = None

        head_doc: Optional[str] = None
        if head_html:
            head_doc = self._build_document(head_html)

        # Unified scoring: HEAD+URL fusion with Option B rule inside
        scores = self._core.score_head_and_url(url=url, head_doc=head_doc)
        fused_combined = scores["fused_combined"]

        # If threshold is None, treat as "always keep" (acts as pure scorer)
        if self._core.cfg.threshold is not None:
            decision = fused_combined >= self._core.cfg.threshold
        else:
            decision = True

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                (
                    "[DualBM25Filter.apply] url=%s "
                    "head_raw_pos=%.6f head_raw_neg=%.6f head_combined=%.6f "
                    "url_raw_pos=%.6f url_raw_neg=%.6f url_combined=%.6f "
                    "fused_combined=%.6f threshold=%s -> decision=%s"
                ),
                url,
                scores["head_raw_pos"],
                scores["head_raw_neg"],
                scores["head_combined"],
                scores["url_raw_pos"],
                scores["url_raw_neg"],
                scores["url_combined"],
                fused_combined,
                (
                    f"{self._core.cfg.threshold:.3f}"
                    if self._core.cfg.threshold is not None
                    else "None"
                ),
                decision,
            )

        self._update_stats(decision)
        return decision


class DualBM25Scorer(URLScorer):
    """
    URLScorer variant of Dual BM25.

    This scorer **does not** perform network I/O. Instead, it scores using a
    pre-built document for each URL (treated as the HEAD-like document), plus
    the URL string itself, with the exact same fusion logic as the filter.

    Usage patterns:

        # 1) Using a prebuilt doc index (recommended)
        doc_index = {url: "title description meta ...", ...}
        scorer = DualBM25Scorer(
            positive_query="product service solution platform",
            negative_query="blog news careers",
            doc_index=doc_index,
            cfg=DualBM25Config(threshold=None, alpha=0.7),
            weight=1.0,
        )

        score = scorer.score(url)  # returns fused_combined in [0,1] * weight

        # 2) Fallback: let it score based on the URL text only
        scorer = DualBM25Scorer(
            positive_query="product service solution platform",
            negative_query="blog news careers",
        )
        score = scorer.score(url)

    Note:
        - `cfg.threshold` is ignored here; this class only returns scores.
        - Output is the fused normalized dual-BM25 score in [0,1], multiplied by `weight`.
    """

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
        super().__init__(weight=weight)
        self._core = _DualBM25Core(positive_query, negative_query, cfg=cfg)
        self._doc_index = doc_index

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DualBM25Scorer.__init__] pos_terms=%d neg_terms=%d weight=%.3f "
                "avgdl=%d alpha=%.3f",
                len(self._core.pos_terms),
                len(self._core.neg_terms),
                self.weight,
                self._core.cfg.avgdl,
                self._core.cfg.alpha,
            )

    # ------------------------------------------------------------------ #
    # URLScorer core
    # ------------------------------------------------------------------ #

    def _get_document_for_url(self, url: str) -> Optional[str]:
        """
        Resolve a text document for a given URL.

        - If doc_index is provided, use doc_index[url] when available.
        - Otherwise, return None to indicate 'no head doc', and rely on URL-only.
        """
        if self._doc_index is not None:
            doc = self._doc_index.get(url)
            if doc:
                return doc
        return None

    def _calculate_score(self, url: str) -> float:
        """
        Calculate dual BM25 fused score for URL, using the same HEAD+URL
        fusion logic as DualBM25Filter.

        - head_doc = doc_index[url] if available, else None.
        - scores = core.score_head_and_url(url=url, head_doc=head_doc)
        - return scores["fused_combined"]
        """
        head_doc = self._get_document_for_url(url)

        # Unified scoring; same Option B rule as filter
        scores = self._core.score_head_and_url(url=url, head_doc=head_doc)
        fused = scores["fused_combined"]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DualBM25Scorer._calculate_score] url=%s fused_combined=%.6f "
                "head_raw_pos=%.6f head_raw_neg=%.6f url_raw_pos=%.6f url_raw_neg=%.6f",
                url,
                fused,
                scores["head_raw_pos"],
                scores["head_raw_neg"],
                scores["url_raw_pos"],
                scores["url_raw_neg"],
            )

        return fused