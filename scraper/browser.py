from __future__ import annotations

import logging
import os
from typing import Tuple, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Playwright

from .config import Config
from .utils import TransientHTTPError

logger = logging.getLogger(__name__)


async def init_browser(cfg: Config) -> Tuple[Playwright, Browser, BrowserContext]:
    """
    Start Playwright, launch Chromium headless, and create a context with:
    - Custom User-Agent from config
    - Reasonable default timeouts
    - HTTP headers to look like a real browser
    """
    proxy = None
    proxy_url = os.getenv("SCRAPER_PROXY")
    if proxy_url:
        proxy = {"server": proxy_url}
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        # You can pass proxies or args here when needed
        args=[
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy=proxy
    )
    context = await browser.new_context(
        user_agent=cfg.user_agent,
        viewport={"width": 1366, "height": 900},
        java_script_enabled=True,
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-CH-UA": '"Chromium";v="120", "Not=A?Brand";v="99"',
        },
    )

    # Global timeouts
    context.set_default_timeout(cfg.page_load_timeout_ms)
    context.set_default_navigation_timeout(cfg.page_load_timeout_ms)

    if getattr(cfg, "block_heavy_resources", True):
        await _install_request_blocking(context)

    logger.info("Browser initialized (UA: %s)", cfg.user_agent)
    return pw, browser, context


async def _install_request_blocking(context: BrowserContext) -> None:
    """
    Abort images, fonts, media to reduce bandwidth for text scraping.
    """
    async def route_handler(route, request):
        resource_type = request.resource_type
        if resource_type in {"image", "media", "font"}:
            return await route.abort()
        return await route.continue_()

    await context.route("**/*", route_handler)


async def shutdown_browser(pw: Playwright, browser: Browser, context: Optional[BrowserContext] = None) -> None:
    """
    Gracefully close context and browser, then stop Playwright.
    """
    try:
        if context:
            await context.close()
    except Exception as e:
        logger.warning("Error while closing context: %s", e)

    try:
        await browser.close()
    except Exception as e:
        logger.warning("Error while closing browser: %s", e)

    try:
        await pw.stop()
    except Exception as e:
        logger.warning("Error while stopping Playwright: %s", e)


async def new_page(context: BrowserContext):
    """
    Helper to open a new page with robust error handling.
    """
    try:
        page = await context.new_page()
        return page
    except Exception as e:
        logger.error("Failed to open new page: %s", e)
        raise TransientHTTPError(str(e))
