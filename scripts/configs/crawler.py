from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union

from crawl4ai import CacheMode, CrawlerRunConfig


PathLike = Union[str, Path]


def _apply_crawl4ai_base_directory(
    cache_base_dir: Optional[PathLike],
) -> Optional[Path]:
    """
    Crawl4AI cache reuse requires future runs to point to the same base directory.
    Crawl4AI reads this from the CRAWL4_AI_BASE_DIRECTORY env var (if set).

    Behavior:
      - If cache_base_dir is None: do nothing, return None.
      - Else: expand/resolve, create directory, set env var, return resolved Path.
    """
    if cache_base_dir is None:
        return None

    p = Path(cache_base_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(p)
    return p


class CrawlerRunStrategy(Protocol):
    """
    Strategy interface: encapsulates how to build a CrawlerRunConfig.
    """

    def build(
        self,
        *,
        page_timeout: Optional[int] = None,
        delay_before_return_html: Optional[float] = None,
        extraction_strategy=None,
        markdown_generator=None,
        cache_mode: Optional[CacheMode] = None,
        cache_base_dir: Optional[PathLike] = None,
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
    """

    base_word_count_threshold: int = 200
    base_page_timeout: int = 60_000  # ms
    base_delay_before_return_html: float = 2.0  # seconds

    base_verbose: bool = True
    base_stream: bool = True

    base_cache_mode: CacheMode = CacheMode.ENABLED

    def build(
        self,
        *,
        page_timeout: Optional[int] = None,
        delay_before_return_html: Optional[float] = None,
        extraction_strategy=None,
        markdown_generator=None,
        cache_mode: Optional[CacheMode] = None,
        cache_base_dir: Optional[PathLike] = None,
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
        _apply_crawl4ai_base_directory(cache_base_dir)

        effective_cache_mode = (
            cache_mode if cache_mode is not None else self.base_cache_mode
        )

        cfg = CrawlerRunConfig(
            word_count_threshold=self.base_word_count_threshold,
            extraction_strategy=extraction_strategy,
            markdown_generator=markdown_generator,
            cache_mode=effective_cache_mode,
            js_code=js_code,
            wait_for=wait_for,
            screenshot=bool(screenshot) if screenshot is not None else False,
            pdf=bool(pdf) if pdf is not None else False,
            capture_mhtml=bool(capture_mhtml) if capture_mhtml is not None else False,
            delay_before_return_html=(
                delay_before_return_html
                if delay_before_return_html is not None
                else self.base_delay_before_return_html
            ),
            page_timeout=(
                page_timeout if page_timeout is not None else self.base_page_timeout
            ),
            locale=locale,
            timezone_id=timezone_id,
            verbose=verbose if verbose is not None else self.base_verbose,
            stream=stream if stream is not None else self.base_stream,
        )
        return cfg


@dataclass
class CrawlerRunConfigFactory:
    """
    Factory for creating CrawlerRunConfig instances from a given strategy.
    """

    strategy: CrawlerRunStrategy

    def create(
        self,
        *,
        page_timeout: Optional[int] = None,
        delay_before_return_html: Optional[float] = None,
        extraction_strategy=None,
        markdown_generator=None,
        cache_mode: Optional[CacheMode] = None,
        cache_base_dir: Optional[PathLike] = None,
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
            cache_base_dir=cache_base_dir,
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


default_crawler_strategy = DefaultCrawlerRunStrategy()
default_crawler_factory = CrawlerRunConfigFactory(strategy=default_crawler_strategy)

__all__ = [
    "CacheMode",
    "CrawlerRunStrategy",
    "DefaultCrawlerRunStrategy",
    "CrawlerRunConfigFactory",
    "default_crawler_strategy",
    "default_crawler_factory",
]
