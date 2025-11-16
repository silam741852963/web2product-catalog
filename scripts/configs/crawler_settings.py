from __future__ import annotations

"""
Centralized Crawl4AI CrawlerRunConfig defaults.

Usage
-----
    from configs.crawler_settings import crawler_base_cfg

    # Create a per-task variant without mutating the base
    run_config = crawler_base_cfg.clone(
        page_timeout=120_000,
        wait_for="css:main"
    )

All parameters that used to go directly into `AsyncWebCrawler.arun()` should now
be passed via `CrawlerRunConfig` (or a clone of `crawler_base_cfg`).
"""

from crawl4ai import CrawlerRunConfig

def make_default_crawler_run_config() -> CrawlerRunConfig:
    """
    Build the project-wide default CrawlerRunConfig.

    This is intended as the *base* configuration for most crawls in this repo.
    Other modules should obtain a copy via `crawler_base_cfg.clone(...)`
    and override only what they need (e.g. timeouts, wait_for, js_code, etc.).
    """

    return CrawlerRunConfig(
        # ------------------------------------------------------------------ #
        # Page navigation / timing / JS
        # ------------------------------------------------------------------ #
        wait_for="networkidle",               # e.g. "css:.loaded-block" – override per use-case
        delay_before_return_html=2,  # seconds; can be raised for very dynamic pages
        page_timeout=60_000,         # ms – per-page timeout; override for tricky sites

        # ------------------------------------------------------------------ #
        # Session & anti-bot
        # ------------------------------------------------------------------ #
        session_id=None,             # set per-company or per-domain if you need state
        magic=True,                  # enable Crawl4AI's stealth features
        simulate_user=True,          # mimic basic user interactions / delays
        override_navigator=True,     # patch navigator.* to avoid trivial bot checks

        # ------------------------------------------------------------------ #
        # Filter
        # ------------------------------------------------------------------ #
        exclude_all_images=True
    )


# -------------------------------------------------------------------------- #
# Global base config
# -------------------------------------------------------------------------- #

# Raw base config (not exported directly; use `crawler_base_cfg` instead)
_base_config: CrawlerRunConfig = make_default_crawler_run_config()

#: Public base configuration to be imported by other modules.
#  Always use `crawler_base_cfg.clone(...)` before mutating/overriding.
crawler_base_cfg: CrawlerRunConfig = _base_config.clone()

__all__ = [
    "crawler_base_cfg",
    "make_default_crawler_run_config",
]