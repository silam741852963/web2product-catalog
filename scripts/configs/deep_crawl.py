from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
)
from crawl4ai.deep_crawling.filters import FilterChain

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Strategy interface
# --------------------------------------------------------------------------- #


class DeepCrawlStrategyProvider(Protocol):
    """
    Strategy interface for constructing a Crawl4AI deep crawl strategy.

    Implementations encapsulate:
      - BFS / DFS / BestFirst choice
      - Default max_depth / include_external / max_pages / score_threshold
      - Optional defaults for filter_chain / url_scorer

    Higher-level code should depend on this interface instead of concrete
    Crawl4AI classes, and wire instances via DI.
    """

    def build(
        self,
        *,
        max_depth: Optional[int] = None,
        include_external: Optional[bool] = None,
        max_pages: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_chain: Optional[FilterChain] = None,
        url_scorer: Optional[Any] = None,
    ) -> Any:  # return type is a Crawl4AI deep crawl strategy instance
        ...  # pragma: no cover - interface


# --------------------------------------------------------------------------- #
# Concrete providers
# --------------------------------------------------------------------------- #


@dataclass
class BFSDeepCrawlStrategyProvider(DeepCrawlStrategyProvider):
    """
    Breadth-first deep crawl configuration.

    Typical use:
      - Comprehensive but bounded exploration (site maps, product trees).
      - Good default for "full but controlled" coverage.

    Defaults are overridable on .build().
    """

    default_max_depth: int = 4
    default_include_external: bool = False
    default_max_pages: Optional[int] = None
    default_score_threshold: Optional[float] = None
    default_filter_chain: Optional[FilterChain] = None
    default_url_scorer: Optional[Any] = None

    def build(
        self,
        *,
        max_depth: Optional[int] = None,
        include_external: Optional[bool] = None,
        max_pages: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_chain: Optional[FilterChain] = None,
        url_scorer: Optional[Any] = None,
    ) -> BFSDeepCrawlStrategy:
        used_max_depth = self.default_max_depth if max_depth is None else int(max_depth)
        used_include_external = (
            self.default_include_external if include_external is None else bool(include_external)
        )
        raw_max = self.default_max_pages if max_pages is None else max_pages
        used_max_pages = int(raw_max) if raw_max is not None else 999999        
        used_score_threshold = (
            self.default_score_threshold if score_threshold is None else float(score_threshold)
        )
        used_filter_chain = filter_chain if filter_chain is not None else self.default_filter_chain
        used_url_scorer = url_scorer if url_scorer is not None else self.default_url_scorer

        logger.info(
            "[deep_crawl] BFS provider build max_depth=%s include_external=%s "
            "max_pages=%s score_threshold=%s has_filter_chain=%s has_url_scorer=%s",
            used_max_depth,
            used_include_external,
            used_max_pages,
            used_score_threshold,
            used_filter_chain is not None,
            used_url_scorer is not None,
        )

        return BFSDeepCrawlStrategy(
            max_depth=used_max_depth,
            include_external=used_include_external,
            max_pages=used_max_pages,
            score_threshold=used_score_threshold,
            filter_chain=used_filter_chain,
            url_scorer=used_url_scorer,
        )


@dataclass
class DFSDeepCrawlStrategyProvider(DeepCrawlStrategyProvider):
    """
    Depth-first deep crawl configuration.

    Typical use:
      - Deep exploration of a branch (e.g., documentation chains).
      - When you want to reach far from a starting point before spreading.
    """

    default_max_depth: int = 4
    default_include_external: bool = False
    default_max_pages: Optional[int] = None
    default_score_threshold: Optional[float] = None
    default_filter_chain: Optional[FilterChain] = None
    default_url_scorer: Optional[Any] = None

    def build(
        self,
        *,
        max_depth: Optional[int] = None,
        include_external: Optional[bool] = None,
        max_pages: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_chain: Optional[FilterChain] = None,
        url_scorer: Optional[Any] = None,
    ) -> DFSDeepCrawlStrategy:
        used_max_depth = self.default_max_depth if max_depth is None else int(max_depth)
        used_include_external = (
            self.default_include_external if include_external is None else bool(include_external)
        )
        raw_max = self.default_max_pages if max_pages is None else max_pages
        used_max_pages = int(raw_max) if raw_max is not None else 999999
        used_score_threshold = (
            self.default_score_threshold if score_threshold is None else float(score_threshold)
        )
        used_filter_chain = filter_chain if filter_chain is not None else self.default_filter_chain
        used_url_scorer = url_scorer if url_scorer is not None else self.default_url_scorer

        logger.info(
            "[deep_crawl] DFS provider build max_depth=%s include_external=%s "
            "max_pages=%s score_threshold=%s has_filter_chain=%s has_url_scorer=%s",
            used_max_depth,
            used_include_external,
            used_max_pages,
            used_score_threshold,
            used_filter_chain is not None,
            used_url_scorer is not None,
        )

        return DFSDeepCrawlStrategy(
            max_depth=used_max_depth,
            include_external=used_include_external,
            max_pages=used_max_pages,
            score_threshold=used_score_threshold,
            filter_chain=used_filter_chain,
            url_scorer=used_url_scorer,
        )


@dataclass
class BestFirstDeepCrawlStrategyProvider(DeepCrawlStrategyProvider):
    """
    Best-first deep crawl configuration (recommended).

    Typical use:
      - Relevance-prioritized crawling using scorers (e.g., KeywordRelevanceScorer).
      - Focusing compute on high-value pages first.

    Note:
      - `score_threshold` is generally not needed for BestFirstCrawlingStrategy
        but is accepted for convenience; pass-through if provided.
    """

    default_max_depth: int = 4
    default_include_external: bool = False
    default_max_pages: Optional[int] = None
    default_filter_chain: Optional[FilterChain] = None
    default_url_scorer: Optional[Any] = None
    default_score_threshold: Optional[float] = None  # optional, rarely used

    def build(
        self,
        *,
        max_depth: Optional[int] = None,
        include_external: Optional[bool] = None,
        max_pages: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_chain: Optional[FilterChain] = None,
        url_scorer: Optional[Any] = None,
    ) -> BestFirstCrawlingStrategy:
        used_max_depth = self.default_max_depth if max_depth is None else int(max_depth)
        used_include_external = (
            self.default_include_external if include_external is None else bool(include_external)
        )
        raw_max = self.default_max_pages if max_pages is None else max_pages
        used_max_pages = int(raw_max) if raw_max is not None else 999999        
        used_filter_chain = filter_chain if filter_chain is not None else self.default_filter_chain
        used_url_scorer = url_scorer if url_scorer is not None else self.default_url_scorer
        used_score_threshold = (
            self.default_score_threshold if score_threshold is None else float(score_threshold)
        )

        logger.info(
            "[deep_crawl] BestFirst provider build max_depth=%s include_external=%s "
            "max_pages=%s has_filter_chain=%s has_url_scorer=%s score_threshold=%s",
            used_max_depth,
            used_include_external,
            used_max_pages,
            used_filter_chain is not None,
            used_url_scorer is not None,
            used_score_threshold,
        )

        # BestFirstCrawlingStrategy does not *require* score_threshold;
        # if the underlying implementation does not accept it, you can
        # remove it from the call site and keep it only for logging.
        kwargs: dict = {
            "max_depth": used_max_depth,
            "include_external": used_include_external,
            "max_pages": used_max_pages,
            "filter_chain": used_filter_chain,
            "url_scorer": used_url_scorer,
        }
        # Only pass score_threshold if the project chooses to use it
        # and the Crawl4AI version supports it.
        if used_score_threshold is not None:
            kwargs["score_threshold"] = used_score_threshold  # type: ignore[assignment]

        return BestFirstCrawlingStrategy(**kwargs)


# --------------------------------------------------------------------------- #
# Factory (DI-friendly)
# --------------------------------------------------------------------------- #


@dataclass
class DeepCrawlStrategyFactory:
    """
    Factory that uses a DeepCrawlStrategyProvider.

    Higher-level orchestration (run scripts, services) should depend on this
    instead of directly instantiating Crawl4AI strategies.

    Example:

        from configs.deep_crawl_strategy import (
            default_bfs_internal_factory,
        )
        from crawl4ai import CrawlerRunConfig

        factory = default_bfs_internal_factory
        deep_strategy = factory.create(max_depth=2)

        config = CrawlerRunConfig(
            deep_crawl_strategy=deep_strategy,
            # ...
        )
    """

    provider: DeepCrawlStrategyProvider

    def create(
        self,
        *,
        max_depth: Optional[int] = None,
        include_external: Optional[bool] = None,
        max_pages: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_chain: Optional[FilterChain] = None,
        url_scorer: Optional[Any] = None,
    ) -> Any:
        return self.provider.build(
            max_depth=max_depth,
            include_external=include_external,
            max_pages=max_pages,
            score_threshold=score_threshold,
            filter_chain=filter_chain,
            url_scorer=url_scorer,
        )


# --------------------------------------------------------------------------- #
# Default, injectable instances
# --------------------------------------------------------------------------- #

#: Default BFS provider tuned for internal-domain product-ish crawling.
#: - No external domains
#: - Conservative depth
#: - max_pages can be set at call site; here we keep it None (unbounded).
default_bfs_internal_provider = BFSDeepCrawlStrategyProvider(
    default_max_depth=4,
    default_include_external=False,
    default_max_pages=None,
    default_score_threshold=None,
)

#: Default BestFirst provider (recommended) with no scorer pre-wired.
#: - Caller must inject a url_scorer (e.g., KeywordRelevanceScorer)
#:   when calling the factory, or set default_url_scorer on the provider.
default_bestfirst_provider = BestFirstDeepCrawlStrategyProvider(
    default_max_depth=4,
    default_include_external=True,
    default_max_pages=None,
    default_filter_chain=None,
    default_url_scorer=None,
    default_score_threshold=None,
)

#: Simple defaults: import and call `.create(...)` where needed.
default_bfs_internal_factory = DeepCrawlStrategyFactory(
    provider=default_bfs_internal_provider
)

default_bestfirst_factory = DeepCrawlStrategyFactory(
    provider=default_bestfirst_provider
)


__all__ = [
    "DeepCrawlStrategyProvider",
    "BFSDeepCrawlStrategyProvider",
    "DFSDeepCrawlStrategyProvider",
    "BestFirstDeepCrawlStrategyProvider",
    "DeepCrawlStrategyFactory",
    "default_bfs_internal_provider",
    "default_bestfirst_provider",
    "default_bfs_internal_factory",
    "default_bestfirst_factory",
]