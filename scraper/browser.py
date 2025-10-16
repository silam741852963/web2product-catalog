from __future__ import annotations

import asyncio
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Tuple, Optional, Deque, Any
from collections import deque
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Browser, BrowserContext, Playwright, Page

from .config import Config
from .utils import (
    TransientHTTPError,
    GLOBAL_THROTTLE,
    parse_retry_after_header,
    try_close_page,
    PageContext,
)

logger = logging.getLogger(__name__)

# ---------------------------
# Module-scoped active state
# ---------------------------
_ACTIVE: dict[str, Any] = {
    "cfg": None,
    "pw": None,
    "browser": None,
    "context": None,
    "sem": None,
    "open_pages": 0,
    "open_lock": None,
    "pages_fifo": None,
    "nav_count": 0,
    "started_at": 0.0,
    "recycle_lock": None,
    "recycling": False,
    "recycle_event": None,
    "watchdog_task": None,
    "loop_old_ex_handler": None,
}

# Benign/expected aborts + site TLS/H2/network quirks we don't want to spam logs for
_SILENCE_PATTERNS = (
    "net::ERR_ABORTED",
    "frame was detached",
    "Target closed",
    "Target page, context or browser has been closed",
    "Execution context was destroyed",
    "Navigation failed because page was closed",
    "Navigation failed because frame was detached",
    "TargetClosedError",
    # TLS / HTTP quirks
    "ERR_CERT_COMMON_NAME_INVALID",
    "ERR_SSL_VERSION_OR_CIPHER_MISMATCH",
    "ERR_CERT_AUTHORITY_INVALID",
    "ERR_CERT_INVALID",
    "ERR_HTTP2_PROTOCOL_ERROR",
    "ERR_NAME_NOT_RESOLVED",
    # Renderer/connection crash page
    "chrome-error://chromewebdata/",
    # Network timeouts
    "ERR_CONNECTION_TIMED_OUT",
)

# If these substrings appear in a loop exception *and* the exception class is TimeoutError,
# it's almost certainly a Playwright nav timeout we already handle upstream — silence it.
_SILENCE_IF_TIMEOUT_CONTAINS = (
    'navigating to "http',
    'waiting until "',
)


def _browser_args(cfg: Config) -> list[str]:
    enable_gpu = bool(getattr(cfg, "browser_enable_gpu", True))
    disable_h2 = bool(getattr(cfg, "browser_disable_http2", False))
    args: list[str] = [
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-features=IsolateOrigins,site-per-process",
        "--headless=new",
    ]
    if disable_h2:
        args.append("--disable-http2")

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
        try:
            for a in extra:
                if isinstance(a, str) and a.strip():
                    args.append(a.strip())
        except Exception:
            pass
    return args


def _install_loop_exception_silencer() -> None:
    """
    Suppress noisy loop-level 'Future exception was never retrieved' logs for
    common, expected Playwright failures we already classify & handle.
    """
    loop = asyncio.get_running_loop()
    prev = loop.get_exception_handler()
    _ACTIVE["loop_old_ex_handler"] = prev

    def _handler(_loop, context: dict):
        exc = context.get("exception")
        message = context.get("message", "")
        text = f"{exc!r}" if exc else message
        cls_name = ""
        try:
            cls_name = type(exc).__name__
        except Exception:
            pass

        # Class- or message-based silence
        if (text and any(p in text for p in _SILENCE_PATTERNS)) \
           or (cls_name in ("TargetClosedError",)) \
           or (cls_name in ("TimeoutError",) and text and any(p in text for p in _SILENCE_IF_TIMEOUT_CONTAINS)):
            logger.debug("Suppressed loop exception: %s", text or cls_name)
            return

        if prev:
            prev(_loop, context)
        else:
            _loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


def _restore_loop_exception_handler() -> None:
    try:
        loop = asyncio.get_running_loop()
        prev = _ACTIVE.get("loop_old_ex_handler")
        loop.set_exception_handler(prev)
    except Exception:
        pass


async def init_browser(cfg: Config) -> Tuple[Playwright, Browser, BrowserContext]:
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
            "Sec-CH-UA": '"Chromium";v="120", "Not=A?Brand";v="99"',
        },
        bypass_csp=getattr(cfg, "browser_bypass_csp", False),
        ignore_https_errors=getattr(cfg, "browser_ignore_https_errors", True),
    )

    context.set_default_timeout(cfg.page_load_timeout_ms)
    context.set_default_navigation_timeout(cfg.page_load_timeout_ms)

    if getattr(cfg, "block_heavy_resources", True):
        await _install_request_blocking(context)

    _install_429_response_hook(context)

    # init state
    _ACTIVE.update({
        "cfg": cfg,
        "pw": pw,
        "browser": browser,
        "context": context,
        "sem": asyncio.Semaphore(cfg.max_global_pages_open),
        "open_pages": 0,
        "open_lock": asyncio.Lock(),
        "pages_fifo": deque(),
        "nav_count": 0,
        "started_at": time.monotonic(),
        "recycle_lock": asyncio.Lock(),
        "recycling": False,
        "recycle_event": asyncio.Event(),
        "watchdog_task": None,
    })
    _ACTIVE["recycle_event"].set()

    _install_loop_exception_silencer()
    _ACTIVE["watchdog_task"] = asyncio.create_task(_watchdog_loop(), name="browser_watchdog")

    logger.info(
        "Browser initialized (UA=%s, proxy=%s, gpu=%s, max_global_pages=%d, ignore_https_errors=%s)",
        cfg.user_agent,
        bool(proxy),
        bool(getattr(cfg, "browser_enable_gpu", False)),
        cfg.max_global_pages_open,
        bool(getattr(cfg, "browser_ignore_https_errors", True)),
    )
    return pw, browser, context


async def shutdown_browser(
    pw: Playwright, browser: Browser, context: Optional[BrowserContext] = None
) -> None:
    t: Optional[asyncio.Task] = _ACTIVE.get("watchdog_task")
    if t and not t.done():
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Watchdog termination error: %s", e)
    _ACTIVE["watchdog_task"] = None

    try:
        await _close_all_pages()
    except Exception as e:
        logger.warning("Error closing pages: %s", e)

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

    _restore_loop_exception_handler()
    for k in list(_ACTIVE.keys()):
        _ACTIVE[k] = None


# ---------------------------
# Page acquisition utilities
# ---------------------------

async def _install_request_blocking(context: BrowserContext) -> None:
    async def route_handler(route, request):
        rtype = request.resource_type
        if rtype in {"image", "media", "font"}:
            return await route.abort()
        return await route.continue_()
    await context.route("**/*", route_handler)


def _install_429_response_hook(context: BrowserContext) -> None:
    async def _on_response(resp):
        try:
            if resp.status == 429:
                url = resp.url
                host = urlparse(url).hostname or "unknown-host"
                headers = getattr(resp, "headers", {}) or {}
                ra = parse_retry_after_header(headers)
                delay_s = ra if ra is not None else 20.0
                await GLOBAL_THROTTLE.penalize(host, delay_s, reason="browser:429")
                logger.warning("Browser saw 429 for %s; backing off host=%s for %.1fs", url, host, delay_s)
        except Exception as e:
            logger.debug("429 hook error: %s", e)
    context.on("response", _on_response)


def _active_context() -> BrowserContext:
    ctx = _ACTIVE.get("context")
    if ctx is None:
        raise RuntimeError("Browser context not initialized")
    return ctx


def _attach_release_on_close(page: Page) -> None:
    setattr(page, "_pool_release_done", False)
    setattr(page, "_in_use", False)

    async def _release():
        try:
            if getattr(page, "_pool_release_done", False):
                return
            setattr(page, "_pool_release_done", True)

            fifo: Deque = _ACTIVE["pages_fifo"]
            try:
                for idx, (_ts, p) in enumerate(list(fifo)):
                    if p is page:
                        del fifo[idx]
                        break
            except Exception:
                pass

            try:
                async with _ACTIVE["open_lock"]:
                    _ACTIVE["open_pages"] = max(0, int(_ACTIVE["open_pages"]) - 1)
            except Exception:
                pass
            try:
                _ACTIVE["sem"].release()
            except Exception:
                pass
        except Exception as e:
            logger.debug("Page release hook error: %s", e)

    def _on_close() -> None:
        asyncio.create_task(_release())

    page.on("close", _on_close)


async def _hard_recycle(reason: str) -> None:
    """
    Immediate recycle of browser/context; used when driver connection is lost.
    """
    cfg: Config = _ACTIVE["cfg"]
    if cfg is None:
        return

    lock: asyncio.Lock = _ACTIVE["recycle_lock"]
    if lock is None:
        return

    async with lock:
        if _ACTIVE["recycling"]:
            ev: asyncio.Event = _ACTIVE["recycle_event"]
            if ev:
                await ev.wait()
            return

        _ACTIVE["recycling"] = True
        _ACTIVE["recycle_event"].clear()
        logger.warning("Hard recycling Chromium due to: %s", reason)

        # best effort close current context/browser
        try:
            old_ctx = _ACTIVE["context"]
            if old_ctx:
                await old_ctx.close()
        except Exception:
            pass
        try:
            if _ACTIVE["browser"]:
                await _ACTIVE["browser"].close()
        except Exception:
            pass

        try:
            pw: Playwright = _ACTIVE["pw"]
            proxy = {"server": cfg.proxy_server} if getattr(cfg, "proxy_server", None) else None
            new_browser = await pw.chromium.launch(
                headless=True,
                args=_browser_args(cfg),
                proxy=proxy,
                slow_mo=getattr(cfg, "browser_slow_mo_ms", 0) or 0,
            )
            new_ctx = await new_browser.new_context(
                user_agent=cfg.user_agent,
                viewport={"width": 1366, "height": 900},
                java_script_enabled=True,
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-CH-UA": '"Chromium";v="120", "Not=A?Brand";v="99"',
                },
                bypass_csp=getattr(cfg, "browser_bypass_csp", False),
                ignore_https_errors=getattr(cfg, "browser_ignore_https_errors", True),
            )
            new_ctx.set_default_timeout(cfg.page_load_timeout_ms)
            new_ctx.set_default_navigation_timeout(cfg.page_load_timeout_ms)
            if getattr(cfg, "block_heavy_resources", True):
                await _install_request_blocking(new_ctx)
            _install_429_response_hook(new_ctx)

            _ACTIVE["browser"] = new_browser
            _ACTIVE["context"] = new_ctx
            _ACTIVE["nav_count"] = 0
            _ACTIVE["started_at"] = time.monotonic()
            logger.info("Hard recycle complete.")
        except Exception as e:
            logger.error("Hard recycle failed: %s", e)
        finally:
            _ACTIVE["recycling"] = False
            _ACTIVE["recycle_event"].set()


async def _maybe_recycle_if_due() -> None:
    cfg: Config = _ACTIVE["cfg"]
    if cfg is None:
        return
    nav_count = int(_ACTIVE["nav_count"])
    elapsed = time.monotonic() - float(_ACTIVE["started_at"])
    if nav_count < cfg.browser_recycle_after_pages and elapsed < cfg.browser_recycle_after_seconds:
        return

    lock: asyncio.Lock = _ACTIVE["recycle_lock"]
    if lock is None:
        return

    if _ACTIVE["recycling"]:
        ev: asyncio.Event = _ACTIVE["recycle_event"]
        if ev:
            await ev.wait()
        return

    async with lock:
        if _ACTIVE["recycling"]:
            ev: asyncio.Event = _ACTIVE["recycle_event"]
            if ev:
                await ev.wait()
            return

        _ACTIVE["recycling"] = True
        _ACTIVE["recycle_event"].clear()
        logger.warning(
            "Recycling Chromium: nav_count=%d elapsed=%.0fs (limits: pages=%d, seconds=%d)",
            nav_count, elapsed, cfg.browser_recycle_after_pages, cfg.browser_recycle_after_seconds
        )

        while int(_ACTIVE["open_pages"]) > 0:
            await asyncio.sleep(0.2)

        try:
            old_ctx = _ACTIVE["context"]
            if old_ctx:
                await old_ctx.close()
        except Exception:
            pass
        try:
            if _ACTIVE["browser"]:
                await _ACTIVE["browser"].close()
        except Exception:
            pass

        try:
            pw: Playwright = _ACTIVE["pw"]
            proxy = {"server": cfg.proxy_server} if getattr(cfg, "proxy_server", None) else None
            new_browser = await pw.chromium.launch(
                headless=True,
                args=_browser_args(cfg),
                proxy=proxy,
                slow_mo=getattr(cfg, "browser_slow_mo_ms", 0) or 0,
            )
            new_ctx = await new_browser.new_context(
                user_agent=cfg.user_agent,
                viewport={"width": 1366, "height": 900},
                java_script_enabled=True,
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-CH-UA": '"Chromium";v="120", "Not=A?Brand";v="99"',
                },
                bypass_csp=getattr(cfg, "browser_bypass_csp", False),
                ignore_https_errors=getattr(cfg, "browser_ignore_https_errors", True),
            )
            new_ctx.set_default_timeout(cfg.page_load_timeout_ms)
            new_ctx.set_default_navigation_timeout(cfg.page_load_timeout_ms)
            if getattr(cfg, "block_heavy_resources", True):
                await _install_request_blocking(new_ctx)
            _install_429_response_hook(new_ctx)

            _ACTIVE["browser"] = new_browser
            _ACTIVE["context"] = new_ctx
            _ACTIVE["nav_count"] = 0
            _ACTIVE["started_at"] = time.monotonic()
            logger.info("Chromium recycled successfully.")
        finally:
            _ACTIVE["recycling"] = False
            _ACTIVE["recycle_event"].set()


async def _close_all_pages() -> None:
    cfg: Config = _ACTIVE["cfg"]
    fifo: Deque = _ACTIVE.get("pages_fifo") or deque()
    while fifo:
        _ts, pg = fifo.pop()
        try:
            await try_close_page(pg, cfg.page_close_timeout_ms)
        except Exception:
            pass
    try:
        async with _ACTIVE["open_lock"]:
            _ACTIVE["open_pages"] = 0
    except Exception:
        pass
    sem: asyncio.Semaphore = _ACTIVE.get("sem")
    if sem and _ACTIVE.get("cfg"):
        _ACTIVE["sem"] = asyncio.Semaphore(_ACTIVE["cfg"].max_global_pages_open)


async def _watchdog_loop() -> None:
    cfg: Config = _ACTIVE["cfg"]
    interval = max(5, int(cfg.watchdog_interval_seconds))
    threshold = int(cfg.max_global_pages_open)
    idle_age_s = 15.0

    while True:
        try:
            await asyncio.sleep(interval)
            open_pages = int(_ACTIVE["open_pages"])
            sem: asyncio.Semaphore = _ACTIVE["sem"]
            ctx = _active_context()
            ctx_pages = 0
            try:
                ctx_pages = len(ctx.pages)
            except Exception:
                pass

            logger.info(
                "[watchdog] open_pages=%d ctx.pages=%d sem_value≈%s nav_count=%d",
                open_pages, ctx_pages, getattr(sem, "_value", "n/a"), int(_ACTIVE["nav_count"])
            )

            max_allowed = int(threshold * 1.05)
            if open_pages > max_allowed:
                to_close = open_pages - threshold
                fifo: Deque = _ACTIVE["pages_fifo"]
                logger.warning("[watchdog] above cap: attempting to close up to %d idle pages", to_close)
                closed = 0
                now = time.monotonic()
                for _ in range(len(fifo)):
                    if closed >= to_close:
                        break
                    ts, page = fifo[0]
                    in_use = bool(getattr(page, "_in_use", False))
                    is_closed = False
                    try:
                        is_closed = page.is_closed()
                    except Exception:
                        pass
                    if in_use or is_closed or (now - ts) < idle_age_s:
                        fifo.rotate(-1)
                        continue
                    fifo.popleft()
                    try:
                        await try_close_page(page, cfg.page_close_timeout_ms)
                        closed += 1
                    except Exception:
                        pass
                if closed:
                    logger.warning("[watchdog] closed %d idle pages to reduce pressure", closed)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug("Watchdog loop error: %s", e)


@asynccontextmanager
async def acquire_page(context: Optional[PageContext] = None):
    # wait out recycle
    ev: asyncio.Event = _ACTIVE["recycle_event"]
    if ev:
        await ev.wait()

    await _maybe_recycle_if_due()

    sem: asyncio.Semaphore = _ACTIVE["sem"]
    await sem.acquire()

    cfg: Config = _ACTIVE["cfg"]
    page: Optional[Page] = None
    yielded = False
    try:
        # try to open a page; if the driver is gone, hard recycle once and retry
        for attempt in (1, 2):
            try:
                ctx = _active_context()
                page = await ctx.new_page()
                break
            except Exception as e:
                msg = str(e)
                if "Connection closed while reading from the driver" in msg or "Browser has been closed" in msg or "Target closed" in msg:
                    logger.error("Context.new_page failed (attempt %d): %s", attempt, msg)
                    await _hard_recycle(msg)
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.2)
                    continue
                else:
                    raise

        _ACTIVE["nav_count"] = int(_ACTIVE["nav_count"]) + 1
        async with _ACTIVE["open_lock"]:
            _ACTIVE["open_pages"] = int(_ACTIVE["open_pages"]) + 1
        _ACTIVE["pages_fifo"].append((time.monotonic(), page))
        _attach_release_on_close(page)
        setattr(page, "_in_use", True)

        yielded = True
        try:
            yield page
        finally:
            try:
                setattr(page, "_in_use", False)
            except Exception:
                pass
            if page is not None and not getattr(page, "_pool_release_done", False):
                try:
                    await try_close_page(page, cfg.page_close_timeout_ms)
                except Exception:
                    pass

    except Exception as e:
        if not yielded:
            logger.error("Failed to open/use a new page: %s", e)
            try:
                sem.release()
            except Exception:
                pass
        else:
            logger.debug("Page context exited with exception: %s", e)
        raise


async def new_page(context: Optional[PageContext] = None) -> Page:
    # prefer acquire_page for RAII; this remains for compat
    ev: asyncio.Event = _ACTIVE["recycle_event"]
    if ev:
        await ev.wait()

    await _maybe_recycle_if_due()

    sem: asyncio.Semaphore = _ACTIVE["sem"]
    await sem.acquire()
    cfg: Config = _ACTIVE["cfg"]
    try:
        for attempt in (1, 2):
            try:
                ctx = _active_context()
                page = await ctx.new_page()
                break
            except Exception as e:
                msg = str(e)
                if "Connection closed while reading from the driver" in msg or "Browser has been closed" in msg or "Target closed" in msg:
                    logger.error("Context.new_page failed (attempt %d): %s", attempt, msg)
                    await _hard_recycle(msg)
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.2)
                    continue
                else:
                    raise

        _ACTIVE["nav_count"] = int(_ACTIVE["nav_count"]) + 1
        async with _ACTIVE["open_lock"]:
            _ACTIVE["open_pages"] = int(_ACTIVE["open_pages"]) + 1
        _ACTIVE["pages_fifo"].append((time.monotonic(), page))
        _attach_release_on_close(page)
        setattr(page, "_in_use", False)
        return page
    except Exception as e:
        try:
            sem.release()
        except Exception:
            pass
        logger.error("Failed to open new page: %s", e)
        raise TransientHTTPError(str(e))