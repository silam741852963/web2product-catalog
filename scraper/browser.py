from __future__ import annotations

import sys
import logging
from typing import Tuple, Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Browser, BrowserContext, Playwright

from .config import Config
from .utils import (
    TransientHTTPError,
    GLOBAL_THROTTLE,
    parse_retry_after_header,
)

logger = logging.getLogger(__name__)


def _browser_args(cfg: Config) -> list[str]:
    enable_gpu = bool(getattr(cfg, "browser_enable_gpu", True))
    args = [
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-features=IsolateOrigins,site-per-process",
        "--headless=new",
    ]
    if enable_gpu:
        args.extend([
            "--ignore-gpu-blocklist",
            "--enable-webgl",
            "--disable-software-rasterizer",
            "--enable-gpu-rasterization",
            "--enable-zero-copy",
            "--use-gl=angle" if sys.platform.startswith("win") else "--use-gl=egl",
        ])
    else:
        args.append("--disable-gpu")

    extra = getattr(cfg, "browser_args_extra", None)
    if extra:
        for a in extra:
            if isinstance(a, str) and a.strip():
                args.append(a.strip())
    return args

async def init_browser(cfg: Config) -> Tuple[Playwright, Browser, BrowserContext]:
    """
    Start Playwright, launch Chromium headless, and create a context with:
    - Custom User-Agent from config
    - Reasonable default timeouts
    - Optional proxy from cfg.proxy_server
    - Optional request blocking for heavy resources
    - 429-aware throttling via response event hook
    """
    proxy = {"server": cfg.proxy_server} if getattr(cfg, "proxy_server", None) else None

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=_browser_args(cfg),
        proxy=proxy,
        slow_mo=getattr(cfg, "browser_slow_mo_ms", 0) or 0,
    )

    context = await browser.new_context(
        user_agent=cfg.user_agent,
        viewport={"width": 1366, "height": 900},
        java_script_enabled=True,
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            # Mild client hints; keep generic to avoid fingerprinting sticks
            "Sec-CH-UA": '"Chromium";v="120", "Not=A?Brand";v="99"',
        },
        bypass_csp=getattr(cfg, "browser_bypass_csp", False),
    )

    # Global timeouts
    context.set_default_timeout(cfg.page_load_timeout_ms)
    context.set_default_navigation_timeout(cfg.page_load_timeout_ms)

    # Optional: install resource blocking (images/media/fonts) to save bandwidth
    if getattr(cfg, "block_heavy_resources", True):
        await _install_request_blocking(context)

    # Hook: throttle on 429 responses observed by the browser
    _install_429_response_hook(context)

    logger.info(
        "Browser initialized (UA: %s, proxy=%s, gpu=%s)",
        cfg.user_agent,
        bool(proxy),
        bool(getattr(cfg, "browser_enable_gpu", False)),
    )
    return pw, browser, context


async def _install_request_blocking(context: BrowserContext) -> None:
    """
    Abort images, fonts, media to reduce bandwidth for text scraping.
    """
    async def route_handler(route, request):
        rtype = request.resource_type
        if rtype in {"image", "media", "font"}:
            return await route.abort()
        return await route.continue_()

    await context.route("**/*", route_handler)


def _install_429_response_hook(context: BrowserContext) -> None:
    """
    When Playwright sees a 429, parse Retry-After and penalize the host
    in the global throttle so subsequent requests to that host back off.
    """

    async def _on_response(resp):
        try:
            status = resp.status
            if status == 429:
                url = resp.url
                host = urlparse(url).hostname or "unknown-host"
                # Headers is dict[str, str] in Playwright Python
                headers = getattr(resp, "headers", {}) or {}
                ra = parse_retry_after_header(headers)
                # If site doesn't send Retry-After, set a sane backoff (e.g., 20s default)
                delay_s = ra if ra is not None else 20.0
                await GLOBAL_THROTTLE.penalize(host, delay_s, reason="browser:429")
                logger.warning("Browser saw 429 for %s; backing off host=%s for %.1fs", url, host, delay_s)
        except Exception as e:
            logger.debug("429 hook error: %s", e)

    context.on("response", _on_response)


async def shutdown_browser(
    pw: Playwright, browser: Browser, context: Optional[BrowserContext] = None
) -> None:
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
        # Cooperate with the global throttle when opening a new page to a host
        # (No URL yet; callers should still call utils.play_goto() which waits for host)
        return await context.new_page()
    except Exception as e:
        logger.error("Failed to open new page: %s", e)
        raise TransientHTTPError(str(e))