# tests/test_utils.py
from pathlib import Path
import json
import re
import pytest

from scraper import utils


def test_normalize_url():
    assert utils.normalize_url("HTTPS://Example.com:443/Path#frag") == "https://example.com/Path"
    assert utils.normalize_url("http://Example.com:80/") == "http://example.com/"
    assert utils.normalize_url("example.com") == "http://example.com/"


def test_is_same_domain_and_http_check():
    assert utils.is_same_domain("https://a.example.com/x", "https://a.example.com/y")
    assert not utils.is_same_domain("https://a.example.com", "https://b.example.com")
    assert utils.is_http_url("http://x")
    assert utils.is_http_url("https://x")
    assert not utils.is_http_url("ftp://x")


def test_slugify():
    assert utils.slugify(" Hello, World! ") == "hello-world"
    assert utils.slugify("A" * 500).startswith("a" * 80)


def test_html_to_markdown_and_cleaning(monkeypatch):
    html = "<h1>Title</h1><p>Hello <b>world</b>.</p><footer>Privacy Policy</footer>"
    md = utils.html_to_markdown(html)
    assert "Title" in md and "Hello" in md

    cleaned = utils.clean_markdown(md)
    assert "Privacy Policy" not in cleaned

    # simulate missing markdownify to hit fallback
    monkeypatch.setattr(utils, "_markdownify", None)
    md2 = utils.html_to_markdown(html)
    assert "Title" in md2 and "<" not in md2

def test_prune_html_for_markdown_removes_chrome():
    html = """
    <html>
      <head><title>SuperWidget</title></head>
      <body>
        <header id="site-header">Mega Menu</header>
        <div class="cookie-banner">We use cookies to improve...</div>
        <nav class="navbar">Top Nav</nav>
        <main>
          <h1>SuperWidget</h1>
          <section><h2>Features</h2><ul><li>Fast</li><li>Light</li></ul></section>
          <section><h2>Specifications</h2><p>Size: 10cm</p></section>
        </main>
        <footer>© 2025 ACME — Privacy Policy</footer>
      </body>
    </html>
    """
    cleaned_html = utils.prune_html_for_markdown(html)
    md = utils.html_to_markdown(cleaned_html)
    md = utils.clean_markdown(md)

    assert "SuperWidget" in md
    assert "Features" in md
    assert "cookie" not in md.lower()
    assert "Mega Menu" not in md
    assert "Privacy Policy" not in md


def test_chunk_text_respects_size():
    text = "para1\n\n" + "x" * 300 + "\n\npara3"
    parts = list(utils.chunk_text(text, max_chars=120))
    assert len(parts) >= 2
    assert all(len(p) <= 120 for p in parts)


def test_atomic_write_and_append_jsonl(tmp_path: Path):
    f = tmp_path / "out.jsonl"
    utils.append_jsonl(f, '{"a":1}')
    utils.append_jsonl(f, '{"a":2}')
    lines = f.read_text().splitlines()
    assert lines == ['{"a":1}', '{"a":2}']

    txt = tmp_path / "file.txt"
    utils.atomic_write_text(txt, "hello")
    assert txt.read_text() == "hello"
    utils.atomic_write_text(txt, "world")
    assert txt.read_text() == "world"


def test_retry_sync_succeeds_after_failures():
    attempts = {"n": 0}

    @utils.retry_sync(max_attempts=3, initial_delay_ms=1, max_delay_ms=2, jitter_ms=0)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise utils.TransientHTTPError("try again")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_retry_async_succeeds_after_failures():
    attempts = {"n": 0}

    @utils.retry_async(max_attempts=3, initial_delay_ms=1, max_delay_ms=2, jitter_ms=0)
    async def flaky_async():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise utils.TransientHTTPError("try again")
        return "ok"

    assert await flaky_async() == "ok"
    assert attempts["n"] == 3


def test_should_crawl_url_gate():
    assert utils.should_crawl_url(respect_robots=False, robots_txt_allowed=None) is True
    assert utils.should_crawl_url(respect_robots=True, robots_txt_allowed=True) is True
    assert utils.should_crawl_url(respect_robots=True, robots_txt_allowed=False) is False


def test_save_markdown_layout(tmp_path: Path):
    md = "# Title\n\nHello"
    host = "www.example.com"  # ensure 'www.' is dropped
    url = "https://www.example.com/products/alpha"
    url_path = "/products/alpha"
    out = utils.save_markdown(tmp_path, host, url_path, url, md)
    assert out.exists()
    # filename includes slug + short hash
    assert out.parent.name == "example.com"
    assert out.suffix == ".md"
    assert re.match(r".+-[0-9a-f]{10}\.md$", out.name)


def test_safe_json_loads_variants():
    good = {"a": 1, "b": {"c": 2}}
    s = json.dumps(good)
    assert utils.safe_json_loads(s) == good

    # with extra tokens around JSON
    noisy = "NOTE:\n" + s + "\nEND"
    parsed = utils.safe_json_loads(noisy)
    assert parsed == good

    # invalid entirely
    assert utils.safe_json_loads("not json") is None
