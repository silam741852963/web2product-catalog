from __future__ import annotations

"""
Centralized Crawl4AI CrawlerRunConfig strategies.

New architecture:
- Configuration-as-Strategy:
    * CrawlerRunStrategy defines how to build a CrawlerRunConfig.
    * DefaultCrawlerRunStrategy implements the common project baseline.
- Factory:
    * CrawlerRunConfigFactory builds configs from a given strategy.
- DI:
    * High-level orchestration code injects the strategy/factory it needs.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

from crawl4ai import CrawlerRunConfig


class CrawlerRunStrategy(Protocol):
    """
    Strategy interface: encapsulates how to build a CrawlerRunConfig
    for a given crawling profile (fast discovery, deep product, debug, etc.).
    """

    def build(
        self,
        *,
        # Common overrides
        page_timeout: Optional[int] = None,
        delay_before_return_html: Optional[float] = None,
        extraction_strategy=None,
        markdown_generator=None,
        cache_mode=None,
        js_code=None,
        wait_for: Optional[str] = None,
        screenshot: Optional[bool] = None,
        pdf: Optional[bool] = None,
        capture_mhtml: Optional[bool] = None,
        locale: Optional[str] = None,
        timezone_id: Optional[str] = None,
        verbose: Optional[bool] = None,
        stream: Optional[bool] = None,
    ) -> CrawlerRunConfig:  # pragma: no cover - interface
        ...


@dataclass
class DefaultCrawlerRunStrategy:
    """
    Project-wide baseline crawler strategy.

    Intent:
    - Keep it minimal and robust.
    - Focus on HTML/markdown extraction; let extraction_strategy be injected.
    - Make it easy to override a few key fields per run via the factory.
    """

    # Baseline defaults – tuned for generic product-site crawling
    base_word_count_threshold: int = 200
    base_page_timeout: int = 120_000  # ms
    base_delay_before_return_html: float = 2.0  # seconds

    # Resource / telemetry defaults
    base_verbose: bool = True
    base_stream: bool = True

    def build(
        self,
        *,
        page_timeout: Optional[int] = None,
        delay_before_return_html: Optional[float] = None,
        extraction_strategy=None,
        markdown_generator=None,
        cache_mode=None,
        js_code=None,
        wait_for: Optional[str] = None,
        screenshot: Optional[bool] = None,
        pdf: Optional[bool] = None,
        capture_mhtml: Optional[bool] = None,
        locale: Optional[str] = None,
        timezone_id: Optional[str] = None,
        verbose: Optional[bool] = None,
        stream: Optional[bool] = None,
    ) -> CrawlerRunConfig:
        cfg = CrawlerRunConfig(
            # ------------------------------------------------------------------ #
            # Core text / content handling
            # ------------------------------------------------------------------ #
            word_count_threshold=self.base_word_count_threshold,
            extraction_strategy=extraction_strategy,
            markdown_generator=markdown_generator,
            cache_mode=cache_mode,
            js_code=js_code,
            wait_for=wait_for,
            screenshot=bool(screenshot) if screenshot is not None else False,
            pdf=bool(pdf) if pdf is not None else False,
            capture_mhtml=bool(capture_mhtml) if capture_mhtml is not None else False,
            # ------------------------------------------------------------------ #
            # Navigation / timing / JS
            # ------------------------------------------------------------------ #
            delay_before_return_html=(
                delay_before_return_html
                if delay_before_return_html is not None
                else self.base_delay_before_return_html
            ),
            page_timeout=(
                page_timeout if page_timeout is not None else self.base_page_timeout
            ),
            # ------------------------------------------------------------------ #
            # Location / identity
            # ------------------------------------------------------------------ #
            locale=locale,
            timezone_id=timezone_id,
            # ------------------------------------------------------------------ #
            # Logging / streaming
            # ------------------------------------------------------------------ #
            verbose=verbose if verbose is not None else self.base_verbose,
            stream=stream if stream is not None else self.base_stream,
        )

        return cfg


@dataclass
class CrawlerRunConfigFactory:
    """
    Factory for creating CrawlerRunConfig instances from a given strategy.

    Usage (example):
        from configs.crawler import default_crawler_strategy, default_crawler_factory

        # Project-wide base
        base_cfg = default_crawler_factory.create()

        # Per-task variant
        deep_cfg = default_crawler_factory.create(
            page_timeout=60_000,
            delay_before_return_html=5,
            wait_for="css:main"
        )
    """

    strategy: CrawlerRunStrategy

    def create(
        self,
        *,
        page_timeout: Optional[int] = None,
        delay_before_return_html: Optional[float] = None,
        extraction_strategy=None,
        markdown_generator=None,
        cache_mode=None,
        js_code=None,
        wait_for: Optional[str] = None,
        screenshot: Optional[bool] = None,
        pdf: Optional[bool] = None,
        capture_mhtml: Optional[bool] = None,
        locale: Optional[str] = None,
        timezone_id: Optional[str] = None,
        verbose: Optional[bool] = None,
        stream: Optional[bool] = None,
    ) -> CrawlerRunConfig:
        return self.strategy.build(
            page_timeout=page_timeout,
            delay_before_return_html=delay_before_return_html,
            extraction_strategy=extraction_strategy,
            markdown_generator=markdown_generator,
            cache_mode=cache_mode,
            js_code=js_code,
            wait_for=wait_for,
            screenshot=screenshot,
            pdf=pdf,
            capture_mhtml=capture_mhtml,
            locale=locale,
            timezone_id=timezone_id,
            verbose=verbose,
            stream=stream,
        )


# -------------------------------------------------------------------------- #
# Default, injectable instances
# -------------------------------------------------------------------------- #

#: Default strategy used across the project.
default_crawler_strategy = DefaultCrawlerRunStrategy()

#: Default factory: import this and call `.create(...)` instead of using
#: CrawlerRunConfig directly in most places.
default_crawler_factory = CrawlerRunConfigFactory(strategy=default_crawler_strategy)

__all__ = [
    "CrawlerRunStrategy",
    "DefaultCrawlerRunStrategy",
    "CrawlerRunConfigFactory",
    "default_crawler_strategy",
    "default_crawler_factory",
]