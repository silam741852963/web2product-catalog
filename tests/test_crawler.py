# tests/test_crawler.py
import asyncio
from pathlib import Path
import pytest

from scraper.crawler import SiteCrawler
from scraper.utils import TransientHTTPError

# ---------- Fakes / Stubs for Playwright ----------

class FakeResponse:
    def __init__(self, status=200):
        self._status = status

    @property
    def status(self):
        return self._status


class FakePage:
    """
    Minimal page stub:
      - holds a canonical page URL
      - returns raw HTML string
      - returns a title
      - returns raw hrefs from eval_on_selector_all("a[href]", ...)
      - supports wait_for_selector (no-op)
    """
    def __init__(self, url, html, title, hrefs=None, status=200):
        self._url = url
        self._html = html
        self._title = title
        self._hrefs = hrefs or []
        self._status = status
        self.closed = False

    async def goto(self, url, wait_until=None, timeout=None):
        # Simulate navigation result status only; content/title come from this stub
        return FakeResponse(self._status)

    async def content(self):
        return self._html

    async def title(self):
        return self._title

    async def eval_on_selector_all(self, selector, js):
        # Return raw hrefs (relative or absolute), not normalized URLs
        return list(self._hrefs)

    async def wait_for_selector(self, selector, timeout=None):
        # no-op; assume success
        return None

    async def close(self):
        self.closed = True


class FakeContext:
    def __init__(self, pages_by_url):
        self._pages_by_url = pages_by_url

    async def new_page(self):
        return FakeNewPageProxy(self._pages_by_url)

    async def close(self):
        pass


class FakeNewPageProxy:
    """
    Proxy that swaps in the right FakePage for each goto().
    Exposes .url and wait_for_selector to match the crawler's usage.
    """
    def __init__(self, pages_by_url):
        self._pages_by_url = pages_by_url
        self._impl = None

    async def goto(self, url, wait_until=None, timeout=None):
        page = self._pages_by_url[url]
        self._impl = page
        return await page.goto(url, wait_until=wait_until, timeout=timeout)

    @property
    def url(self):
        return "" if self._impl is None else self._impl._url

    async def wait_for_selector(self, selector, timeout=None):
        if self._impl is None:
            return None
        return await self._impl.wait_for_selector(selector, timeout=timeout)

    async def content(self):
        return await self._impl.content()

    async def title(self):
        return await self._impl.title()

    async def eval_on_selector_all(self, selector, js):
        return await self._impl.eval_on_selector_all(selector, js)

    async def close(self):
        if self._impl:
            await self._impl.close()


# ---------- Fake Config ----------

class FakeCfg:
    def __init__(self, tmpdir: Path):
        self.user_agent = "pytest-UA"
        self.page_load_timeout_ms = 5000
        self.navigation_wait_until = "domcontentloaded"
        self.max_pages_per_domain_parallel = 3
        self.cache_html = True
        self.scraped_html_dir = tmpdir
        # crawler knobs used by SiteCrawler
        self.per_page_delay_ms = 0
        self.crawler_max_retries = 2
        # NEW: language/subdomain + company cap
        self.allow_subdomains = False
        self.max_pages_per_company = 100
        self.primary_lang = "en"
        self.lang_path_deny = ("/fr", "/de", "/es")
        self.lang_query_keys = ("lang", "hl", "locale")
        self.lang_subdomain_deny = ("fr.", "de.", "es.")


# ---------- Tests ----------

@pytest.mark.asyncio
async def test_crawl_site_collects_internal_links_and_saves_html(tmp_path):
    home = "https://example.com/"
    a = "https://example.com/products/a"
    about = "https://example.com/about"
    b = "https://example.com/products/b"
    external = "http://external.com/x"

    # IMPORTANT: supply raw hrefs (relative and absolute), not normalized URLs
    pages = {
        home: FakePage(
            url=home,
            html=("<html><title>Home</title>"
                  "<a href='/products/a'>A</a><a href='/about'>About</a>"
                  "<a href='http://external.com/x'>E</a></html>"),
            title="Home",
            hrefs=["/products/a", "/about", external],
        ),
        a: FakePage(url=a, html="<html><title>Product A</title><h1>A</h1></html>", title="Product A"),
        about: FakePage(url=about, html="<html><title>About</title><a href='/products/b'>B</a></html>", title="About", hrefs=["/products/b"]),
        b: FakePage(url=b, html="<html><title>Product B</title><h1>B</h1></html>", title="Product B"),
    }

    context = FakeContext(pages_by_url=pages)
    cfg = FakeCfg(tmp_path)

    crawler = SiteCrawler(cfg, context)
    snapshots = await crawler.crawl_site(homepage=home)

    urls = sorted(s.url for s in snapshots)
    assert urls == sorted([home.rstrip("/"), a, about, b]) or urls == sorted([home, a, about, b])

    home_snap = next(s for s in snapshots if s.url.startswith(home.rstrip("/")))
    assert a in home_snap.out_links and about in home_snap.out_links
    assert all(not l.startswith("http://external.com") for l in home_snap.out_links)

    for s in snapshots:
        assert s.html_path is not None
        p = Path(s.html_path)
        assert p.exists()
        assert p.parent.name == "example.com"


@pytest.mark.asyncio
async def test_crawl_site_respects_allow_and_deny_filters(tmp_path):
    home = "https://acme.test/"
    prods = "https://acme.test/products"
    careers = "https://acme.test/careers"
    prod1 = "https://acme.test/products/p1"
    prod2 = "https://acme.test/products/p2"

    pages = {
        home: FakePage(url=home, html="<html>home</html>", title="home", hrefs=[prods, careers]),
        prods: FakePage(url=prods, html="<html>prods</html>", title="prods", hrefs=[prod1, prod2]),
        careers: FakePage(url=careers, html="<html>careers</html>", title="careers"),
        prod1: FakePage(url=prod1, html="<html>p1</html>", title="p1"),
        prod2: FakePage(url=prod2, html="<html>p2</html>", title="p2"),
    }

    context = FakeContext(pages_by_url=pages)
    cfg = FakeCfg(tmp_path)
    crawler = SiteCrawler(cfg, context)

    snaps = await crawler.crawl_site(
        homepage=home,
        url_allow_regex=r"/products",
        url_deny_regex=r"/p2$",
    )
    urls = sorted(s.url for s in snaps)
    assert urls == sorted([home, prods, prod1])


@pytest.mark.asyncio
async def test_crawl_site_max_pages_cap(tmp_path):
    home = "https://limit.test/"
    a = "https://limit.test/a"
    b = "https://limit.test/b"
    c = "https://limit.test/c"

    pages = {
        home: FakePage(url=home, html="home", title="home", hrefs=[a, b]),
        a: FakePage(url=a, html="a", title="a", hrefs=[c]),
        b: FakePage(url=b, html="b", title="b"),
        c: FakePage(url=c, html="c", title="c"),
    }
    context = FakeContext(pages_by_url=pages)
    cfg = FakeCfg(tmp_path)

    crawler = SiteCrawler(cfg, context)
    snaps = await crawler.crawl_site(homepage=home, max_pages=2)
    assert len(snaps) <= 2


@pytest.mark.asyncio
async def test_fetch_retry_on_transient_error(monkeypatch, tmp_path):
    home = "https://retry.test/"
    a = "https://retry.test/a"

    pages = {
        home: FakePage(url=home, html="home", title="home", hrefs=[a]),
        a: FakePage(url=a, html="a", title="a"),
    }
    context = FakeContext(pages_by_url=pages)
    cfg = FakeCfg(tmp_path)

    crawler = SiteCrawler(cfg, context)

    calls = {"n": 0}
    real = crawler._fetch_and_extract

    async def flaky(url: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientHTTPError("temporary")
        return await real(url)

    monkeypatch.setattr(crawler, "_fetch_and_extract", flaky)

    snaps = await crawler.crawl_site(homepage=home)
    urls = sorted(s.url for s in snaps)
    assert urls == sorted([home, a])
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_http_error_status_triggers_transient(tmp_path):
    home = "https://err.test/"
    bad = "https://err.test/404"

    pages = {
        home: FakePage(url=home, html="home", title="home", hrefs=[bad]),
        bad: FakePage(url=bad, html="not found", title="err", hrefs=[], status=404),
    }
    context = FakeContext(pages_by_url=pages)
    cfg = FakeCfg(tmp_path)

    crawler = SiteCrawler(cfg, context)
    snaps = await crawler.crawl_site(homepage=home)

    urls = sorted(s.url for s in snaps)
    assert urls == [home]


@pytest.mark.asyncio
async def test_language_filters_drop_non_primary(tmp_path):
    home = "https://example.com/"
    fr = "https://example.com/fr/produits"
    de_q = "https://example.com/products?lang=de"
    prod = "https://example.com/products"

    pages = {
        home: FakePage(url=home, html="<html>home</html>", title="home",
                       hrefs=["/fr/produits", "/products?lang=de", "/products"]),
        fr:   FakePage(url=fr,   html="<html>fr</html>",   title="fr"),
        de_q: FakePage(url=de_q, html="<html>de</html>",   title="de"),
        prod: FakePage(url=prod, html="<html>prod</html>", title="prod"),
    }

    context = FakeContext(pages_by_url=pages)
    cfg = FakeCfg(tmp_path)
    # language policy in FakeCfg denies /fr and ?lang=de

    crawler = SiteCrawler(cfg, context)
    snaps = await crawler.crawl_site(homepage=home)

    urls = sorted(s.url for s in snaps)
    # should only include home and /products
    assert urls == sorted([home, prod])


@pytest.mark.asyncio
async def test_subdomain_toggle(tmp_path):
    home = "https://example.com/"
    docs = "https://docs.example.com/guide"
    prod = "https://example.com/products"

    pages = {
        home: FakePage(url=home, html="<html>home</html>", title="home",
                       hrefs=[docs, "/products"]),
        docs: FakePage(url=docs, html="<html>docs</html>", title="docs"),
        prod: FakePage(url=prod, html="<html>prod</html>", title="prod"),
    }

    # 1) disallow subdomains: docs should be excluded
    context = FakeContext(pages_by_url=pages)
    cfg = FakeCfg(tmp_path)
    cfg.allow_subdomains = False

    crawler = SiteCrawler(cfg, context)
    snaps = await crawler.crawl_site(homepage=home)
    urls = sorted(s.url for s in snaps)
    assert urls == sorted([home, prod])

    # 2) allow subdomains: docs should be included
    context2 = FakeContext(pages_by_url=pages)
    cfg2 = FakeCfg(tmp_path)
    cfg2.allow_subdomains = True

    crawler2 = SiteCrawler(cfg2, context2)
    snaps2 = await crawler2.crawl_site(homepage=home)
    urls2 = sorted(s.url for s in snaps2)
    assert urls2 == sorted([home, docs, prod])