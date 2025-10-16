import pytest

from scraper.browser import init_browser, shutdown_browser, new_page  # noqa: F401
from scraper.utils import TransientHTTPError


class FakeConfig:
    # Core fields used by init_browser/new_page
    user_agent = "pytest-UA"
    page_load_timeout_ms = 12345
    navigation_wait_until = "domcontentloaded"
    max_pages_per_domain_parallel = 2
    cache_html = False
    block_heavy_resources = True  # toggleable
    proxy_server = None           # set in test
    per_page_delay_ms = 0

    # Browser flags read by _browser_args / init
    browser_enable_gpu = True
    browser_args_extra = None
    browser_slow_mo_ms = 0
    browser_bypass_csp = False
    browser_ignore_https_errors = True

    # Global limits & recycle knobs expected by browser.init_browser
    max_global_pages_open = 128
    page_close_timeout_ms = 1500
    browser_recycle_after_pages = 10_000
    browser_recycle_after_seconds = 21_600  # 6h
    watchdog_interval_seconds = 30
    max_httpx_clients = 1


class StubPage:
    def __init__(self):
        self.closed = False
        self._events = {}

    def on(self, event: str, callback):
        self._events.setdefault(event, []).append(callback)

    def off(self, event: str, callback):
        if event in self._events:
            try:
                self._events[event].remove(callback)
            except ValueError:
                pass

    async def close(self):
        # mark closed
        self.closed = True
        # emit "close" to trigger the release hook in browser.py
        for cb in list(self._events.get("close", [])):
            try:
                cb()
            except Exception:
                pass


class StubContext:
    def __init__(self):
        self.closed = False
        self._routes = []
        self._default_timeout = None
        self._default_navigation_timeout = None
        self._pages = []
        self._new_page_raises = None
        self._events = {}  # needed for context.on(...)

    async def route(self, pattern, handler):
        self._routes.append((pattern, handler))

    def set_default_timeout(self, ms):
        self._default_timeout = ms

    def set_default_navigation_timeout(self, ms):
        self._default_navigation_timeout = ms

    async def new_page(self):
        if self._new_page_raises:
            raise self._new_page_raises
        p = StubPage()
        self._pages.append(p)
        return p

    @property
    def pages(self):
        # mimic Playwright's .pages property used by watchdog/metrics
        return list(self._pages)

    async def close(self):
        self.closed = True

    def on(self, event: str, callback):
        # minimal event registry used by the 429 response hook
        self._events.setdefault(event, []).append(callback)

    def off(self, event: str, callback):
        if event in self._events:
            try:
                self._events[event].remove(callback)
            except ValueError:
                pass


class StubBrowser:
    def __init__(self, context):
        self.context = context
        self.closed = False

    async def new_context(self, **kwargs):
        return self.context

    async def close(self):
        self.closed = True


class StubChromium:
    def __init__(self, browser):
        self.browser = browser
        self._launch_kwargs = None

    async def launch(self, **kwargs):
        self._launch_kwargs = kwargs
        return self.browser


class StubPlaywright:
    def __init__(self, chromium):
        self.chromium = chromium
        self.stopped = False

    async def stop(self):
        self.stopped = True


class AsyncPlaywrightFactory:
    def __init__(self, pw):
        self._pw = pw

    async def start(self):
        return self._pw


@pytest.mark.asyncio
async def test_init_browser_with_blocking_and_proxy_toggle(monkeypatch):
    cfg = FakeConfig()

    # build stubs
    context = StubContext()
    browser = StubBrowser(context=context)
    chromium = StubChromium(browser=browser)
    pw = StubPlaywright(chromium=chromium)
    async_playwright_factory = AsyncPlaywrightFactory(pw)

    import scraper.browser as browser_mod
    monkeypatch.setattr(browser_mod, "async_playwright", lambda: async_playwright_factory)

    # set proxy via config (no env usage)
    cfg.proxy_server = "http://localhost:8888"

    # call
    pw_ret, browser_ret, context_ret = await init_browser(cfg)

    assert pw_ret is pw
    assert browser_ret is browser
    assert context_ret is context

    # default timeouts applied
    assert context._default_timeout == cfg.page_load_timeout_ms
    assert context._default_navigation_timeout == cfg.page_load_timeout_ms

    # request blocking installed (toggle on)
    assert len(context._routes) >= 1
    pattern, handler = context._routes[0]
    assert pattern == "**/*"
    assert callable(handler)

    # proxy passed through to chromium.launch
    assert chromium._launch_kwargs is not None
    assert "proxy" in chromium._launch_kwargs
    assert chromium._launch_kwargs["proxy"] == {"server": cfg.proxy_server}

    # shutdown
    await shutdown_browser(pw_ret, browser_ret, context_ret)
    assert context_ret.closed is True
    assert browser_ret.closed is True
    assert pw.stopped is True


@pytest.mark.asyncio
async def test_init_browser_without_blocking(monkeypatch):
    cfg = FakeConfig()
    cfg.block_heavy_resources = False  # toggle OFF
    cfg.proxy_server = None            # no proxy

    context = StubContext()
    browser = StubBrowser(context=context)
    chromium = StubChromium(browser=browser)
    pw = StubPlaywright(chromium=chromium)
    async_playwright_factory = AsyncPlaywrightFactory(pw)

    import scraper.browser as browser_mod
    monkeypatch.setattr(browser_mod, "async_playwright", lambda: async_playwright_factory)

    pw_ret, browser_ret, ctx = await init_browser(cfg)

    # no routes installed when blocking disabled
    assert ctx._routes == []

    # and chromium.launch received no proxy
    assert chromium._launch_kwargs is not None
    assert "proxy" in chromium._launch_kwargs
    assert chromium._launch_kwargs["proxy"] is None

    # shutdown
    await shutdown_browser(pw_ret, browser_ret, ctx)
    assert ctx.closed is True
    assert browser_ret.closed is True
    assert pw.stopped is True


@pytest.mark.asyncio
async def test_new_page_ok_and_error_path(monkeypatch):
    cfg = FakeConfig()
    context = StubContext()
    browser = StubBrowser(context=context)
    chromium = StubChromium(browser=browser)
    pw = StubPlaywright(chromium=chromium)
    async_playwright_factory = AsyncPlaywrightFactory(pw)

    import scraper.browser as browser_mod
    monkeypatch.setattr(browser_mod, "async_playwright", lambda: async_playwright_factory)

    # init
    pw_ret, browser_ret, ctx = await init_browser(cfg)

    # success path
    page = await new_page(ctx)
    assert isinstance(page, StubPage)
    assert page.closed is False

    # close to trigger release hook (page.on("close", ...))
    await page.close()
    assert page.closed is True

    # error path -> ensure TransientHTTPError raised
    ctx._new_page_raises = RuntimeError("boom")
    with pytest.raises(TransientHTTPError):
        await new_page(ctx)

    # shutdown
    await shutdown_browser(pw_ret, browser_ret, ctx)
    assert ctx.closed is True
    assert browser_ret.closed is True
    assert pw.stopped is True