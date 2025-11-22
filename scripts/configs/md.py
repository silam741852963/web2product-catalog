from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Strategy interface
# --------------------------------------------------------------------------- #


class MarkdownGeneratorStrategy(Protocol):
    """
    Strategy interface: given high-level markdown parameters, produce
    a DefaultMarkdownGenerator instance.

    Implementations are responsible only for:
      - choosing the content filter (Pruning, BM25, LLM, etc.)
      - wiring crawl4ai's DefaultMarkdownGenerator
    """

    def build(
        self,
        *,
        # presentation / html2text options
        body_width: Optional[int] = None,
        ignore_links: Optional[bool] = None,
        ignore_images: Optional[bool] = None,
        content_source: Optional[str] = None,
        # gating-related knobs are attached for observability only;
        # gating logic itself lives in extensions.md_gating
        min_meaningful_words: Optional[int] = None,
        interstitial_max_share: Optional[float] = None,
        interstitial_min_hits: Optional[int] = None,
        cookie_max_fraction: Optional[float] = None,
        require_structure: Optional[bool] = None,
    ) -> DefaultMarkdownGenerator:  # pragma: no cover - interface
        ...


# --------------------------------------------------------------------------- #
# Default strategy (PruningContentFilter + fit_html)
# --------------------------------------------------------------------------- #


@dataclass
class PruningMarkdownGeneratorStrategy:
    """
    Default markdown strategy:

    - Uses PruningContentFilter to remove boilerplate.
    - Uses `fit_html` as content_source by default (optimized HTML).
    - Exposes gating-related config in `options["_gating"]` for
      downstream diagnostics; actual gating is done in extensions.md_gating.

    Threshold knobs are strategy-level (stable across the run),
    while presentation/gating knobs can be overridden per call via `.build(...)`.
    """

    # Pruning filter configuration (structural, usually stable)
    threshold: float = 0.12
    threshold_type: str = "fixed"   # or "dynamic"
    min_word_threshold: int = 30

    # Default markdown presentation options
    default_body_width: int = 0
    default_ignore_links: bool = True
    default_ignore_images: bool = True
    default_content_source: str = "fit_html"  # "fit_html" / "cleaned_html" / "raw_html"

    # Default gating-related values (for observability only)
    default_min_meaningful_words: Optional[int] = None
    default_interstitial_max_share: float = 0.60
    default_interstitial_min_hits: int = 2
    default_cookie_max_fraction: float = 0.15
    default_require_structure: bool = True

    def build(
        self,
        *,
        body_width: Optional[int] = None,
        ignore_links: Optional[bool] = None,
        ignore_images: Optional[bool] = None,
        content_source: Optional[str] = None,
        min_meaningful_words: Optional[int] = None,
        interstitial_max_share: Optional[float] = None,
        interstitial_min_hits: Optional[float] = None,
        cookie_max_fraction: Optional[float] = None,
        require_structure: Optional[bool] = None,
    ) -> DefaultMarkdownGenerator:
        # Resolve effective values, falling back to strategy defaults
        eff_body_width = self.default_body_width if body_width is None else int(body_width)
        eff_ignore_links = self.default_ignore_links if ignore_links is None else bool(ignore_links)
        eff_ignore_images = self.default_ignore_images if ignore_images is None else bool(ignore_images)
        eff_content_source = self.default_content_source if content_source is None else str(content_source)

        eff_min_meaningful_words = (
            self.default_min_meaningful_words if min_meaningful_words is None else int(min_meaningful_words)
        )
        eff_interstitial_max_share = (
            self.default_interstitial_max_share
            if interstitial_max_share is None
            else float(interstitial_max_share)
        )
        eff_interstitial_min_hits = (
            self.default_interstitial_min_hits
            if interstitial_min_hits is None
            else int(interstitial_min_hits)
        )
        eff_cookie_max_fraction = (
            self.default_cookie_max_fraction
            if cookie_max_fraction is None
            else float(cookie_max_fraction)
        )
        eff_require_structure = (
            self.default_require_structure
            if require_structure is None
            else bool(require_structure)
        )

        # 1) Content filter
        pruning_filter = PruningContentFilter(
            threshold=self.threshold,
            threshold_type=self.threshold_type,
            min_word_threshold=self.min_word_threshold,
        )

        # 2) Markdown generator options
        options = {
            "ignore_links": eff_ignore_links,
            "ignore_images": eff_ignore_images,
            "body_width": eff_body_width,
            "escape_html": True,
            # Expose gating config for debug/observability (no behavior here):
            "_gating": {
                "min_meaningful_words": eff_min_meaningful_words,
                "interstitial_max_share": eff_interstitial_max_share,
                "interstitial_min_hits": eff_interstitial_min_hits,
                "cookie_max_fraction": eff_cookie_max_fraction,
                "require_structure": eff_require_structure,
            },
        }

        generator = DefaultMarkdownGenerator(
            content_filter=pruning_filter,
            content_source=eff_content_source,
            options=options,
        )

        logger.debug(
            "[md_gen] init content_source=%s "
            "prune(thr=%.2f type=%s min_words=%d) "
            "gating(min_meaningful_words=%s interstitial_max_share=%.2f "
            "interstitial_min_hits=%d cookie_max_fraction=%.2f require_structure=%s) "
            "opts(ignore_links=%s ignore_images=%s body_width=%d)",
            eff_content_source,
            self.threshold,
            self.threshold_type,
            self.min_word_threshold,
            str(eff_min_meaningful_words),
            eff_interstitial_max_share,
            eff_interstitial_min_hits,
            eff_cookie_max_fraction,
            str(eff_require_structure),
            eff_ignore_links,
            eff_ignore_images,
            eff_body_width,
        )

        return generator


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


@dataclass
class MarkdownGeneratorFactory:
    """
    Factory that uses a MarkdownGeneratorStrategy.
    Higher-level code should depend on this abstraction, not directly
    on crawl4ai internals.
    """

    strategy: MarkdownGeneratorStrategy

    def create(
        self,
        *,
        body_width: Optional[int] = None,
        ignore_links: Optional[bool] = None,
        ignore_images: Optional[bool] = None,
        content_source: Optional[str] = None,
        min_meaningful_words: Optional[int] = None,
        interstitial_max_share: Optional[float] = None,
        interstitial_min_hits: Optional[float] = None,
        cookie_max_fraction: Optional[float] = None,
        require_structure: Optional[bool] = None,
    ) -> DefaultMarkdownGenerator:
        return self.strategy.build(
            body_width=body_width,
            ignore_links=ignore_links,
            ignore_images=ignore_images,
            content_source=content_source,
            min_meaningful_words=min_meaningful_words,
            interstitial_max_share=interstitial_max_share,
            interstitial_min_hits=interstitial_min_hits,
            cookie_max_fraction=cookie_max_fraction,
            require_structure=require_structure,
        )


# --------------------------------------------------------------------------- #
# Default, injectable instances + backward-compatible helper
# --------------------------------------------------------------------------- #

#: Default pruning markdown strategy used across the project.
default_md_strategy = PruningMarkdownGeneratorStrategy()

#: Default factory; import this and call `.create(...)` from run code.
default_md_factory = MarkdownGeneratorFactory(strategy=default_md_strategy)


def build_default_md_generator(
    *,
    threshold: float = 0.12,
    threshold_type: str = "fixed",
    min_word_threshold: int = 30,
    body_width: int = 0,
    ignore_links: bool = True,
    ignore_images: bool = True,
    content_source: str = "fit_html",
    min_meaningful_words: Optional[int] = None,
    interstitial_max_share: float = 0.60,
    interstitial_min_hits: int = 2,
    cookie_max_fraction: float = 0.15,
    require_structure: bool = True,
) -> DefaultMarkdownGenerator:
    """
    Backward-compatible convenience wrapper that mirrors the old function
    signature. Internally this configures the default strategy and uses
    the factory to build the generator.

    Prefer injecting `MarkdownGeneratorFactory` instead of calling this
    directly in new code.
    """
    # Reconfigure the strategy thresholds if the caller overrides them
    default_md_strategy.threshold = float(threshold)
    default_md_strategy.threshold_type = str(threshold_type)
    default_md_strategy.min_word_threshold = int(min_word_threshold)

    return default_md_factory.create(
        body_width=body_width,
        ignore_links=ignore_links,
        ignore_images=ignore_images,
        content_source=content_source,
        min_meaningful_words=min_meaningful_words,
        interstitial_max_share=interstitial_max_share,
        interstitial_min_hits=interstitial_min_hits,
        cookie_max_fraction=cookie_max_fraction,
        require_structure=require_structure,
    )


__all__ = [
    "MarkdownGeneratorStrategy",
    "PruningMarkdownGeneratorStrategy",
    "MarkdownGeneratorFactory",
    "default_md_strategy",
    "default_md_factory",
    "build_default_md_generator",
]