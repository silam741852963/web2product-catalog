from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from urllib.parse import urlparse

from playwright.async_api import BrowserContext

from .config import Config
from .utils import (
    normalize_url,
    is_http_url,
    is_same_domain,
    slugify,
    atomic_write_text,
    retry_async,
    TransientHTTPError,
)

logger = logging.getLogger(__name__)


@dataclass
class PageSnapshot:
    url: str
    title: Optional[str]
    html_path: Optional[str]  # path on disk if cached
    out_links: List[str] = field(default_factory=list)


class SiteCrawler:
    """
    Concurrent, retrying, per-domain crawler:
    - Starts from a homepage
    - Follows internal links (same host)
    - Saves raw HTML (optional, per config.cache_html)
    - Returns PageSnapshot list for downstream processing
    """

    def __init__(self, cfg: Config, context: BrowserContext):
        self.cfg = cfg
        self.context = context
        self.wait_until = cfg.navigation_wait_until  # "load" | "domcontentloaded" | "networkidle" | "commit"

        # Per-domain concurrency limiter
        self.sem = asyncio.Semaphore(cfg.max_pages_per_domain_parallel)

        # CHANGE: track per-URL attempts to allow queue-level retries when transient errors occur
        self._attempts: Dict[str, int] = {}
        self._max_retries: int = int(getattr(cfg, "crawler_max_retries", 2))

    async def crawl_site(
        self,
        homepage: str,
        max_pages: Optional[int] = None,
        url_allow_regex: Optional[str] = None,
        url_deny_regex: Optional[str] = None,
    ) -> List[PageSnapshot]:
        start_url = normalize_url(homepage)
        if not is_http_url(start_url):
            logger.warning("Skip non-http URL: %s", start_url)
            return []

        host = urlparse(start_url).hostname or ""
        logger.info("Crawling %s", host)

        visited: Set[str] = set()
        queue: asyncio.Queue[str] = asyncio.Queue()
        await queue.put(start_url)

        snapshots: Dict[str, PageSnapshot] = {}

        allow_re = re.compile(url_allow_regex) if url_allow_regex else None
        deny_re = re.compile(url_deny_regex) if url_deny_regex else None

        async def worker():
            while True:
                try:
                    url = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    return

                if url in visited:
                    queue.task_done()
                    continue

                if max_pages is not None and len(visited) >= max_pages:
                    queue.task_done()
                    continue

                visited.add(url)
                try:
                    async with self.sem:
                        snap = await self._fetch_and_extract(url)
                    snapshots[url] = snap

                    # Enqueue new links
                    for link in snap.out_links:
                        if link in visited:
                            continue
                        if not is_same_domain(url, link):
                            continue
                        if allow_re and not allow_re.search(link):
                            continue
                        if deny_re and deny_re.search(link):
                            continue
                        await queue.put(link)

                except TransientHTTPError as e:
                    # CHANGE: queue-level bounded retry if fetch failed transiently
                    attempts = self._attempts.get(url, 0) + 1
                    self._attempts[url] = attempts
                    if attempts <= self._max_retries:
                        logger.warning("Transient error on %s (attempt %d/%d): %s", url, attempts, self._max_retries, e)
                        # Remove from visited so we can attempt again
                        visited.discard(url)
                        await queue.put(url)
                    else:
                        logger.warning("Giving up on %s after %d attempts", url, attempts)
                except Exception as e:
                    logger.error("Uncaught error on %s: %s", url, e)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self.cfg.max_pages_per_domain_parallel)]
        await queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        logger.info("Crawl finished: %s pages from %s", len(snapshots), host)
        return list(snapshots.values())

    @retry_async(
        max_attempts=4,
        initial_delay_ms=500,
        max_delay_ms=8000,
        jitter_ms=300,
    )
    async def _fetch_and_extract(self, url: str) -> PageSnapshot:
        page = await self.context.new_page()
        try:
            wait_sequence = [self.wait_until, "domcontentloaded", "load"]
            last_err = None
            for wu in wait_sequence:
                try:
                    resp = await page.goto(url, wait_until=wu, timeout=self.cfg.page_load_timeout_ms)
                    break
                except Exception as e:
                    last_err = e
                    continue
            if last_err and resp is None:
                raise TransientHTTPError(f"Navigation failed for {url}: {last_err}")
            resp = await page.goto(url, wait_until=self.wait_until, timeout=self.cfg.page_load_timeout_ms)
            if resp is not None:
                status = resp.status
                if status >= 400:
                    raise TransientHTTPError(f"HTTP {status} for {url}")

            title = await _safe_title(page)
            html = await page.content()

            # hard cap enormous HTML blobs (avoid blowing downstream memory)
            if len(html) > 3_000_000:  # ~3 MB
                html = html[:3_000_000]


            html_path = None
            if self.cfg.cache_html:
                html_path = self._save_html(url, html)

            links = await _extract_links(page)
            links = [normalize_url(l) for l in links if is_http_url(l)]

            # CHANGE: only keep same-domain links in out_links so tests don't see externals
            same_domain_links = [l for l in links if is_same_domain(url, l)]

            return PageSnapshot(url=url, title=title, html_path=html_path, out_links=same_domain_links)

        except Exception as e:
            raise TransientHTTPError(str(e))
        finally:
            try:
                await page.close()
            except Exception:
                pass

    def _save_html(self, url: str, html: str) -> str:
        parsed = urlparse(url)
        host = parsed.hostname or "unknown-host"
        base = parsed.path or "/"
        if base.endswith("/"):
            base += "index"
        slug = slugify(base, max_len=80)
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        fname = f"{slug}-{h}.html"

        out_dir = self.cfg.scraped_html_dir / host
        out_path = out_dir / fname
        atomic_write_text(out_path, html)
        return str(out_path)


async def _safe_title(page) -> Optional[str]:
    try:
        return await page.title()
    except Exception:
        return None


async def _extract_links(page) -> List[str]:
    try:
        hrefs: List[str] = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href)",
        )
        cleaned: List[str] = []
        seen = set()
        for h in hrefs:
            if not h:
                continue
            if not h.startswith("http"):
                continue
            h_clean = re.sub(r"#.*$", "", h)
            if h_clean not in seen:
                seen.add(h_clean)
                cleaned.append(h_clean)
        return cleaned
    except Exception as e:
        logger.warning("Failed extracting links: %s", e)
        return []