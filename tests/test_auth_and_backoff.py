import pytest
from pathlib import Path
from types import SimpleNamespace

from scraper.crawler import SiteCrawler, NonRetryableHTTPError, TransientHTTPError
from scripts.run_scraper import _note_backoff, _save_company_state

# ----- minimal fake Playwright context & page -----
class FakePage:
    def __init__(self, final_url):
        self.url = final_url
    async def close(self):
        return None

class FakeContext:
    async def new_page(self):
        return FakePage(final_url="https://example.com/")

def _mk_cfg(**over):
    base = dict(
        navigation_wait_until="domcontentloaded",
        page_load_timeout_ms=5000,
        cache_html=False,
        allow_subdomains=True,
        max_pages_per_domain_parallel=1,   # required by crawler
        per_page_delay_ms=0,               # avoid sleeps
        enable_static_first=False,         # go straight to dynamic
    )
    base.update(over)
    return SimpleNamespace(**base)

@pytest.mark.asyncio
async def test_401_homepage_transient(monkeypatch):
    from scraper import crawler as crawler_mod
    async def _fake_play_goto(page, url, waits, timeout):
        return (401, "")
    monkeypatch.setattr(crawler_mod, "play_goto", _fake_play_goto)

    cfg = _mk_cfg()
    ctx = FakeContext()
    crawler = SiteCrawler(cfg, ctx)
    crawler._homepage_url = "https://example.com/"

    with pytest.raises(TransientHTTPError):
        await crawler.fetch_dynamic_only("https://example.com/")

@pytest.mark.asyncio
async def test_403_non_homepage_nonretryable(monkeypatch):
    from scraper import crawler as crawler_mod
    async def _fake_play_goto(page, url, waits, timeout):
        return (403, "")
    monkeypatch.setattr(crawler_mod, "play_goto", _fake_play_goto)

    cfg = _mk_cfg()
    ctx = FakeContext()
    crawler = SiteCrawler(cfg, ctx)
    crawler._homepage_url = "https://example.com/"

    with pytest.raises(NonRetryableHTTPError):
        await crawler.fetch_dynamic_only("https://example.com/inner")

def test_429_backoff_increments(tmp_path: Path):
    cfg = _mk_cfg()
    state_path = tmp_path / "company.json"
    state = {"company": "ACME"}

    _note_backoff(cfg, state_path, state, "example.com")
    _note_backoff(cfg, state_path, state, "example.com")
    assert state["backoff_hits"] == 2