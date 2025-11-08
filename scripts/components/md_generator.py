from __future__ import annotations

import logging
import re
from typing import Dict, Tuple, Optional

from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from config import language_settings as langcfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language-provided compiled regexes
# ---------------------------------------------------------------------------

_INTERSTITIAL_RE = langcfg.get_interstitial_re()

_COOKIEISH_RE = langcfg.get_cookieish_re()

_PRODUCT_TOKENS_RE = langcfg.get_product_tokens_re()

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+\S")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")

def _word_count(s: str) -> int:
    return len(_WORD_RE.findall(s or ""))

def _fraction_words(pattern: re.Pattern, text: str) -> Tuple[int, float]:
    hits = 0
    matched = 0
    for m in pattern.finditer(text or ""):
        hits += 1
        matched += _word_count(m.group(0))
    total = max(_word_count(text), 1)
    return hits, matched / total

def _interstitial_stats(md: str) -> Dict[str, float]:
    """
    Return {'hits': int, 'matched_words': int, 'share': float, 'total_words': int}
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

def _structure_ok(md: str, *, require_structure: bool, ignore_links: bool) -> bool:
    if not require_structure:
        return True
    # Headings in MD are most reliable (DefaultMarkdownGenerator preserves them)
    if len(_HEADING_RE.findall(md)) >= 2:
        return True
    # Links signal some body depth, but may be disabled by generator options
    if not ignore_links and len(_MARKDOWN_LINK_RE.findall(md)) >= 5:
        return True
    # Product-ish vocabulary
    if _PRODUCT_TOKENS_RE.search(md or ""):
        return True
    return False

# ---------------------------------------------------------------------------
# Backward-compatible gating & quality gate 
# ---------------------------------------------------------------------------

def should_save_markdown(
    markdown: str | None,
    *,
    # primary knob
    min_meaningful_words: int = 5,
    # be forgiving: only drop if interstitial dominates
    interstitial_max_share: float = 0.70,
    interstitial_min_hits: int = 2,
    url: str | None = None,
    **kwargs,
) -> tuple[bool, str]:
    """
    Return (keep: bool, reason: str)
    reasons: 'ok' | 'empty' | 'too_short' | 'interstitial_dominant'
    NOTE: Kept for backward compatibility. Prefer `evaluate_markdown` below.
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

def evaluate_markdown(
    markdown: Optional[str],
    *,
    url: Optional[str] = None,
    allow_retry: bool = True,
    min_meaningful_words: int = 5,
    interstitial_max_share: float = 0.60,
    interstitial_min_hits: int = 2,
    cookie_max_fraction: float = 0.15,
    require_structure: bool = True,
    generator_ignores_links: bool = True,
) -> tuple[str, str, Dict[str, float]]:
    """
    Stronger quality gate with "save|suppress|retry" semantics.

    Returns: (action, reason, stats_dict)
      action ∈ {"save", "suppress", "retry"}
      reason is a stable code for logging/metrics
      stats include: total_words, interstitial_share, cookie_fraction, headings, links, product_token

    Retry is suggested when content looks cookie/overlay-gated rather than truly empty/short.
    """
    if not markdown:
        logger.debug("[md.qgate] suppress empty url=%s", url)
        return "suppress", "empty", {"total_words": 0}

    md = str(markdown).strip()
    total_words = _word_count(md)
    if total_words < min_meaningful_words:
        logger.debug("[md.qgate] suppress too_short url=%s words=%d min=%d",
                     url, total_words, min_meaningful_words)
        return "suppress", "too_short", {"total_words": total_words}

    # Interstitial dominance
    inter_stats = _interstitial_stats(md)
    # Cookie fraction (more targeted)
    cookie_hits, cookie_frac = _fraction_words(_COOKIEISH_RE, md)

    # Structure check
    headings = len(_HEADING_RE.findall(md))
    links = len(_MARKDOWN_LINK_RE.findall(md)) if not generator_ignores_links else 0
    has_product_tokens = bool(_PRODUCT_TOKENS_RE.search(md))

    structure_ok = _structure_ok(
        md,
        require_structure=require_structure,
        ignore_links=generator_ignores_links,
    )

    stats = {
        "total_words": float(total_words),
        "interstitial_share": float(inter_stats["share"]),
        "interstitial_hits": float(inter_stats["hits"]),
        "cookie_fraction": float(cookie_frac),
        "cookie_hits": float(cookie_hits),
        "headings": float(headings),
        "links": float(links),
        "product_token": 1.0 if has_product_tokens else 0.0,
    }

    # If cookie language dominates, try a retry (if allowed)
    if cookie_frac >= cookie_max_fraction and inter_stats["hits"] >= interstitial_min_hits:
        reason = "cookie_dominant"
        logger.debug("[md.qgate] %s url=%s cookie_frac=%.2f >= %.2f",
                     ("retry" if allow_retry else "suppress"), url, cookie_frac, cookie_max_fraction)
        return ("retry" if allow_retry else "suppress"), reason, stats

    # Generic interstitial dominance (WAF, JS required, etc.)
    if inter_stats["share"] >= interstitial_max_share and inter_stats["hits"] >= interstitial_min_hits:
        reason = "interstitial_dominant"
        logger.debug("[md.qgate] %s url=%s interstitial_share=%.2f >= %.2f",
                     ("retry" if allow_retry else "suppress"), url, inter_stats["share"], interstitial_max_share)
        return ("retry" if allow_retry else "suppress"), reason, stats

    if not structure_ok:
        reason = "structure_weak"
        logger.debug("[md.qgate] %s url=%s reason=%s", ("retry" if allow_retry else "suppress"), url, reason)
        return ("retry" if allow_retry else "suppress"), reason, stats

    logger.debug("[md.qgate] save url=%s words=%d inter=%.2f cookie=%.2f headings=%d product=%s",
                 url, total_words, inter_stats["share"], cookie_frac, headings, str(has_product_tokens))
    return "save", "ok", stats

# ---------------------------------------------------------------------------
# Markdown generator factory (respects --log-level via standard logging)
# ---------------------------------------------------------------------------

def build_default_md_generator(
    *,
    threshold: float = 0.48,
    threshold_type: str = "dynamic",  # or "fixed"
    min_word_threshold: int = 5,
    body_width: int = 0,
    ignore_links: bool = True,
    ignore_images: bool = True,
    content_source: str = "fit_html",  # or "cleaned_html" or "raw_html"
    min_meaningful_words: Optional[int] = None,
    interstitial_max_share: float = 0.60,
    interstitial_min_hits: int = 2,
    cookie_max_fraction: float = 0.15,
    require_structure: bool = True,
) -> DefaultMarkdownGenerator:
    """
    Constructs a Markdown generator using Crawl4AI's PruningContentFilter.
    The extra gating params are accepted for callsite compatibility and are
    attached to the generator for observability; gating itself is performed
    by evaluate_markdown(...) / should_save_markdown(...) at save time.
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
        "escape_html": True,
        # Expose gating config for debug/observability (no behavioral impact here):
        "_gating": {
            "min_meaningful_words": min_meaningful_words,
            "interstitial_max_share": interstitial_max_share,
            "interstitial_min_hits": interstitial_min_hits,
            "cookie_max_fraction": cookie_max_fraction,
            "require_structure": require_structure,
        }
    }

    generator = DefaultMarkdownGenerator(
        content_filter=pruning_filter,
        content_source=content_source,
        options=options,
    )

    logger.debug(
        "[md_gen] init content_source=%s prune(thr=%.2f type=%s min_words=%d) "
        "gating(min_meaningful_words=%s interstitial_max_share=%.2f interstitial_min_hits=%d cookie_max_fraction=%.2f require_structure=%s) "
        "opts(ignore_links=%s ignore_images=%s body_width=%d)",
        content_source, threshold, threshold_type, min_word_threshold,
        str(min_meaningful_words), interstitial_max_share, interstitial_min_hits, cookie_max_fraction, str(require_structure),
        ignore_links, ignore_images, body_width,
    )

    return generator