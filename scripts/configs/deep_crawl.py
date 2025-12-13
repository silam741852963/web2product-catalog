from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from crawl4ai.deep_crawling.filters import FilterChain

from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy as _BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy as _DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy as _BestFirstCrawlingStrategy,
)


logger = logging.getLogger(__name__)

_UNBOUNDED_MAX_PAGES = 999_999


def _coerce_max_pages(requested: Optional[int], default: Optional[int]) -> int:
    raw = default if requested is None else requested
    if raw is None:
        return _UNBOUNDED_MAX_PAGES
    try:
        v = int(raw)
    except Exception:
        return _UNBOUNDED_MAX_PAGES
    if v <= 0:
        return _UNBOUNDED_MAX_PAGES
    return v


def _coerce_int(value: Optional[int], default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_bool(value: Optional[bool], default: bool) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _coerce_float(value: Optional[float], default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Strategy interface
# --------------------------------------------------------------------------- #


class DeepCrawlStrategyProvider(Protocol):
    def build(
        self,
        *,
        max_depth: Optional[int] = None,
        include_external: Optional[bool] = None,
        max_pages: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_chain: Optional[FilterChain] = None,
        url_scorer: Optional[Any] = None,
    ) -> Any:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# Concrete providers
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BFSDeepCrawlStrategyProvider(DeepCrawlStrategyProvider):
    default_max_depth: int = 3
    default_include_external: bool = False
    default_max_pages: Optional[int] = 200
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
    ) -> Any:
        if _BFSDeepCrawlStrategy is None:  # pragma: no cover
            raise ImportError("Crawl4AI BFSDeepCrawlStrategy is not available.")

        used_max_depth = _coerce_int(max_depth, self.default_max_depth)
        used_include_external = _coerce_bool(
            include_external, self.default_include_external
        )
        used_max_pages = _coerce_max_pages(max_pages, self.default_max_pages)
        used_score_threshold = _coerce_float(
            score_threshold, self.default_score_threshold
        )
        used_filter_chain = (
            filter_chain if filter_chain is not None else self.default_filter_chain
        )
        used_url_scorer = (
            url_scorer if url_scorer is not None else self.default_url_scorer
        )

        logger.info(
            "[deep_crawl] BFS build max_depth=%s include_external=%s max_pages=%s "
            "score_threshold=%s has_filter_chain=%s has_url_scorer=%s",
            used_max_depth,
            used_include_external,
            used_max_pages,
            used_score_threshold,
            used_filter_chain is not None,
            used_url_scorer is not None,
        )

        return _BFSDeepCrawlStrategy(
            max_depth=used_max_depth,
            include_external=used_include_external,
            max_pages=used_max_pages,
            score_threshold=used_score_threshold,
            filter_chain=used_filter_chain,
            url_scorer=used_url_scorer,
        )


@dataclass(slots=True)
class DFSDeepCrawlStrategyProvider(DeepCrawlStrategyProvider):
    default_max_depth: int = 3
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
    ) -> Any:
        if _DFSDeepCrawlStrategy is None:  # pragma: no cover
            raise ImportError("Crawl4AI DFSDeepCrawlStrategy is not available.")

        used_max_depth = _coerce_int(max_depth, self.default_max_depth)
        used_include_external = _coerce_bool(
            include_external, self.default_include_external
        )
        used_max_pages = _coerce_max_pages(max_pages, self.default_max_pages)
        used_score_threshold = _coerce_float(
            score_threshold, self.default_score_threshold
        )
        used_filter_chain = (
            filter_chain if filter_chain is not None else self.default_filter_chain
        )
        used_url_scorer = (
            url_scorer if url_scorer is not None else self.default_url_scorer
        )

        logger.info(
            "[deep_crawl] DFS build max_depth=%s include_external=%s max_pages=%s "
            "score_threshold=%s has_filter_chain=%s has_url_scorer=%s",
            used_max_depth,
            used_include_external,
            used_max_pages,
            used_score_threshold,
            used_filter_chain is not None,
            used_url_scorer is not None,
        )

        return _DFSDeepCrawlStrategy(
            max_depth=used_max_depth,
            include_external=used_include_external,
            max_pages=used_max_pages,
            score_threshold=used_score_threshold,
            filter_chain=used_filter_chain,
            url_scorer=used_url_scorer,
        )


@dataclass(slots=True)
class BestFirstDeepCrawlStrategyProvider(DeepCrawlStrategyProvider):
    default_max_depth: int = 3
    default_include_external: bool = False
    default_max_pages: Optional[int] = None
    default_filter_chain: Optional[FilterChain] = None
    default_url_scorer: Optional[Any] = None
    default_score_threshold: Optional[float] = None

    def build(
        self,
        *,
        max_depth: Optional[int] = None,
        include_external: Optional[bool] = None,
        max_pages: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_chain: Optional[FilterChain] = None,
        url_scorer: Optional[Any] = None,
    ) -> Any:
        if _BestFirstCrawlingStrategy is None:  # pragma: no cover
            raise ImportError("Crawl4AI BestFirstCrawlingStrategy is not available.")

        used_max_depth = _coerce_int(max_depth, self.default_max_depth)
        used_include_external = _coerce_bool(
            include_external, self.default_include_external
        )
        used_max_pages = _coerce_max_pages(max_pages, self.default_max_pages)
        used_filter_chain = (
            filter_chain if filter_chain is not None else self.default_filter_chain
        )
        used_url_scorer = (
            url_scorer if url_scorer is not None else self.default_url_scorer
        )
        used_score_threshold = _coerce_float(
            score_threshold, self.default_score_threshold
        )

        logger.info(
            "[deep_crawl] BestFirst build max_depth=%s include_external=%s max_pages=%s "
            "has_filter_chain=%s has_url_scorer=%s score_threshold=%s",
            used_max_depth,
            used_include_external,
            used_max_pages,
            used_filter_chain is not None,
            used_url_scorer is not None,
            used_score_threshold,
        )

        kwargs: dict[str, Any] = {
            "max_depth": used_max_depth,
            "include_external": used_include_external,
            "max_pages": used_max_pages,
            "filter_chain": used_filter_chain,
            "url_scorer": used_url_scorer,
        }
        if used_score_threshold is not None:
            kwargs["score_threshold"] = used_score_threshold

        try:
            return _BestFirstCrawlingStrategy(**kwargs)
        except TypeError as e:  # pragma: no cover
            if "score_threshold" in kwargs and "score_threshold" in str(e):
                kwargs.pop("score_threshold", None)
                return _BestFirstCrawlingStrategy(**kwargs)
            raise


# --------------------------------------------------------------------------- #
# Factory (DI-friendly)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DeepCrawlStrategyFactory:
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

default_bfs_internal_provider = BFSDeepCrawlStrategyProvider(
    default_max_depth=3,
    default_include_external=False,
    default_max_pages=None,
    default_score_threshold=None,
)

default_bestfirst_provider = BestFirstDeepCrawlStrategyProvider(
    default_max_depth=3,
    default_include_external=True,
    default_max_pages=None,
    default_filter_chain=None,
    default_url_scorer=None,
    default_score_threshold=None,
)

default_bfs_internal_factory = DeepCrawlStrategyFactory(
    provider=default_bfs_internal_provider
)
default_bestfirst_factory = DeepCrawlStrategyFactory(
    provider=default_bestfirst_provider
)


def build_deep_strategy(
    strategy: str,
    *,
    filter_chain: FilterChain,
    url_scorer: Optional[Any],
    dfs_factory: Optional["DeepCrawlStrategyFactory"],
    max_pages: Optional[int],
) -> Any:
    """
    Convenience helper used by run.py.
    strategy: "bestfirst" | "bfs_internal" | "dfs"
    """
    if strategy == "bestfirst":
        return default_bestfirst_factory.create(
            filter_chain=filter_chain,
            url_scorer=url_scorer,
            max_pages=max_pages,
        )

    if strategy == "bfs_internal":
        return default_bfs_internal_factory.create(
            filter_chain=filter_chain,
            url_scorer=None,
            max_pages=max_pages,
        )

    if strategy == "dfs":
        if dfs_factory is None:
            raise ValueError(
                "DFS strategy requested but `dfs_factory` is None. "
                "Construct a DeepCrawlStrategyFactory with DFSDeepCrawlStrategyProvider and pass it in."
            )
        return dfs_factory.create(
            filter_chain=filter_chain,
            url_scorer=None,
            max_pages=max_pages,
        )

    raise ValueError(f"Unknown deep crawl strategy: {strategy!r}")


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
    "build_deep_strategy",
]
