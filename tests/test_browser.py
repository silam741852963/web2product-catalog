import pytest

from scraper.browser import init_browser, shutdown_browser, new_page  # noqa: F401
from scraper.utils import TransientHTTPError


class FakeConfig:
    user_agent = "pytest-UA"
    page_load_timeout_ms = 12345
    navigation_wait_until = "domcontentloaded"
    max_pages_per_domain_parallel = 2
    cache_html = False
    block_heavy_resources = True  # toggleable
    proxy_server = None           # NEW: pass proxy via config (not env)
    per_page_delay_ms = 0         # optional


class StubPage:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class StubContext:
    def __init__(self):
        self.closed = False
        self._routes = []
        self._default_timeout = None
        self._default_navigation_timeout = None
        self._pages = []
        self._new_page_raises = None

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

    async def close(self):
        self.closed = True


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

    _, _, ctx = await init_browser(cfg)

    # no routes installed when blocking disabled
    assert ctx._routes == []

    # and chromium.launch received no proxy
    assert chromium._launch_kwargs is not None
    assert "proxy" in chromium._launch_kwargs
    assert chromium._launch_kwargs["proxy"] is None


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
    _, _, ctx = await init_browser(cfg)

    # success path
    page = await new_page(ctx)
    assert isinstance(page, StubPage)
    assert page.closed is False

    # error path -> ensure TransientHTTPError raised
    ctx._new_page_raises = RuntimeError("boom")
    with pytest.raises(TransientHTTPError):
        await new_page(ctx)