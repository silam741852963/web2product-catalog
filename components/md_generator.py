# components/md_generator.py
from __future__ import annotations

import logging
import re
from typing import Dict, Tuple, Optional

from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interstitial / gatekeeping patterns
#   We measure *dominance* (share of words matched by these patterns).
#   Only suppress saving if the share exceeds a threshold AND there are
#   at least `interstitial_min_hits` matches.
# ---------------------------------------------------------------------------

_INTERSTITIAL_PATTERNS = [
    # access blocks / WAF / bot checks
    r"\bforbidden\b",
    r"\baccess\s+denied\b",
    r"\byou\s+don'?t\s+have\s+permission\b",
    r"\bnot\s+authorized\b",
    r"\bunauthorized\b",
    r"\berror\s*(401|403)\b",
    r"\btemporarily\s+blocked\b",
    r"\bchecking\s+your\s+browser\b",
    r"\bverify\s+you\s+are\s+human\b",
    r"\bcaptcha\b",

    # cookie/consent interstitials
    r"\bcookies?\s+(policy|preferences|settings)\b",
    r"\bwe\s+use\s+cookies\b",
    r"\baccept\s+all\s+cookies\b",
    r"\bmanage\s+cookies\b",

    # paywalls / soft-walls
    r"\bsubscribe\s+to\s+continue\b",
    r"\bsign\s*in\s+to\s+continue\b",
    r"\blog\s*in\s+to\s+continue\b",

    # misc generic “wall” phrases often in thin pages
    r"\benable\s+javascript\b",
    r"\bprivacy\s+policy\b",
    r"\bterms\s+(and|&)\s+conditions\b",
]

_INTERSTITIAL_RE = re.compile("|".join(f"(?:{p})" for p in _INTERSTITIAL_PATTERNS), re.IGNORECASE)

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _word_count(s: str) -> int:
    return len(_WORD_RE.findall(s or ""))


def _interstitial_stats(md: str) -> Dict[str, float]:
    """
    Return {'hits': int, 'matched_words': int, 'share': float}
    """
    hits = list(_INTERSTITIAL_RE.finditer(md or ""))
    matched_words = 0
    for m in hits:
        matched_words += _word_count(m.group(0))
    total_words = max(_word_count(md), 1)  # avoid div by zero
    share = matched_words / total_words
    return {
        "hits": len(hits),
        "matched_words": matched_words,
        "share": share,
        "total_words": total_words,
    }


def should_save_markdown(
    markdown: str | None,
    *,
    # primary knob
    min_meaningful_words: int = 80,
    # be forgiving: only drop if interstitial dominates
    interstitial_max_share: float = 0.70,
    interstitial_min_hits: int = 2,
    url: str | None = None,
    **kwargs,
) -> tuple[bool, str]:
    """
    Return (keep: bool, reason: str)
    reasons: 'ok' | 'empty' | 'too_short' | 'interstitial_dominant'
    """

    # Back-compat alias (run_crawl.py used to pass min_words)
    if "min_words" in kwargs:
        try:
            min_meaningful_words = int(kwargs["min_words"])
        except Exception:
            pass

    if not markdown:
        logger.debug("[md.gate] drop empty url=%s", url)
        return False, "empty"

    text = str(markdown).strip()
    # quick word count
    words = re.findall(r"\b\w+\b", text)
    n_words = len(words)
    if n_words < int(min_meaningful_words):
        logger.debug("[md.gate] drop too_short url=%s words=%d min=%d",
                     url, n_words, min_meaningful_words)
        return False, "too_short"

    # interstitial dominance check (forgiving)
    matches = list(_INTERSTITIAL_RE.finditer(text))
    hits = len(matches)
    if hits:
        covered = sum(m.end() - m.start() for m in matches)
        share = covered / max(1, len(text))
        logger.debug("[md.gate] interstitial url=%s hits=%d share=%.2f max=%.2f",
                     url, hits, share, interstitial_max_share)
        if share >= interstitial_max_share and hits >= interstitial_min_hits:
            logger.debug("[md.gate] drop interstitial_dominant url=%s", url)
            return False, "interstitial_dominant"

    logger.debug("[md.gate] keep url=%s words=%d hits=%d", url, n_words, hits)
    return True, "ok"


# ---------------------------------------------------------------------------
# Markdown generator factory (respects --log-level via standard logging)
# ---------------------------------------------------------------------------

def build_default_md_generator(
    *,
    threshold: float = 0.48,
    threshold_type: str = "dynamic",  # or "fixed"
    min_word_threshold: int = 100,
    body_width: int = 0,
    ignore_links: bool = True,
    ignore_images: bool = True,
    content_source: str = "cleaned_html",  # or "fit_html" or "raw_html"

    # NEW (for callsite compatibility & visibility in logs):
    min_meaningful_words: Optional[int] = None,
    interstitial_max_share: float = 0.60,
    interstitial_min_hits: int = 2,
) -> DefaultMarkdownGenerator:
    """
    Constructs a Markdown generator using Crawl4AI's PruningContentFilter.
    The extra gating params are accepted for callsite compatibility and are
    attached to the generator for observability; gating itself is performed
    by should_save_markdown(...) at save time.
    """
    pruning_filter = PruningContentFilter(
        threshold=threshold,
        threshold_type=threshold_type,
        min_word_threshold=min_word_threshold,
    )

    options = {
        "ignore_links": ignore_links,
        "ignore_images": ignore_images,
        "body_width": body_width,
        # Expose gating config for debug/observability (no behavioral impact here):
        "_gating": {
            "min_meaningful_words": min_meaningful_words,
            "interstitial_max_share": interstitial_max_share,
            "interstitial_min_hits": interstitial_min_hits,
        }
    }

    generator = DefaultMarkdownGenerator(
        content_filter=pruning_filter,
        content_source=content_source,
        options=options,
    )

    logger.debug(
        "[md_gen] init content_source=%s prune(thr=%.2f type=%s min_words=%d) "
        "gating(min_meaningful_words=%s interstitial_max_share=%.2f interstitial_min_hits=%d) "
        "opts(ignore_links=%s ignore_images=%s body_width=%d)",
        content_source, threshold, threshold_type, min_word_threshold,
        str(min_meaningful_words), interstitial_max_share, interstitial_min_hits,
        ignore_links, ignore_images, body_width,
    )
    return generator