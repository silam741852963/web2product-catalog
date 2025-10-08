from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from urllib.parse import urlparse

import httpx
from playwright.async_api import BrowserContext


from .config import Config
from .utils import (
    normalize_url,
    is_http_url,
    slugify,
    atomic_write_text,
    retry_async,
    TransientHTTPError,
    NonRetryableHTTPError,
    NeedsBrowser,
    get_base_domain,
    looks_like_js_app,
    extract_links_static,
    extract_title_static,
    same_site,
    http_status_to_exc,
    httpx_client,
    play_goto,
    play_title,
    play_links,
    filter_links,
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
    - Follows internal links (same host or eTLD+1 if allow_subdomains)
    - Saves raw HTML (optional, per config.cache_html)
    - Returns PageSnapshot list for downstream processing
    """

    def __init__(self, cfg: Config, context: BrowserContext):
        self.cfg = cfg
        self.context = context
        # "load" | "domcontentloaded" | "networkidle" | "commit"
        self.wait_until = cfg.navigation_wait_until

        # Per-domain concurrency limiter
        self.sem = asyncio.Semaphore(cfg.max_pages_per_domain_parallel)

        # track per-URL attempts (queue-level retries for transient errors)
        self._attempts: Dict[str, int] = {}
        self._max_retries: int = int(getattr(cfg, "crawler_max_retries", 3))

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

        # Default to config cap if not explicitly provided
        if max_pages is None:
            max_pages = getattr(self.cfg, "max_pages_per_company", None)

        host = urlparse(start_url).hostname or ""
        logger.info("Crawling %s", host)

        visited: Set[str] = set()
        queued: Set[str] = set()  # prevent duplicate enqueues
        queue: asyncio.Queue[str] = asyncio.Queue()
        await queue.put(start_url)
        queued.add(start_url)

        snapshots: Dict[str, PageSnapshot] = {}

        import re

        allow_re = re.compile(url_allow_regex) if url_allow_regex else None
        deny_re  = re.compile(url_deny_regex)  if url_deny_regex  else None

        async def worker():
            while True:
                try:
                    url = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    return

                # Cap: if we hit max_pages, drain fast
                if max_pages is not None and len(visited) >= max_pages:
                    queue.task_done()
                    continue

                if url in visited:
                    queue.task_done()
                    continue

                visited.add(url)
                try:
                    async with self.sem:
                        snap = await self._fetch_and_extract(url, allow_re=allow_re, deny_re=deny_re)

                    # optional polite delay to reduce bans
                    if self.cfg.per_page_delay_ms > 0:
                        await asyncio.sleep(self.cfg.per_page_delay_ms / 1000.0)

                    snapshots[url] = snap

                    # Enqueue new links (already filtered in snap.out_links)
                    for link in snap.out_links:
                        if max_pages is not None and len(visited) >= max_pages:
                            break
                        if link in visited or link in queued:
                            continue
                        await queue.put(link)
                        queued.add(link)

                except NonRetryableHTTPError as e:
                    # Do not retry 404s etc.
                    logger.warning("Non-retryable error on %s: %s", url, e)
                except TransientHTTPError as e:
                    attempts = self._attempts.get(url, 0) + 1
                    self._attempts[url] = attempts
                    if attempts <= self._max_retries:
                        logger.warning(
                            "Transient error on %s (attempt %d/%d): %s",
                            url, attempts, self._max_retries, e
                        )
                        visited.discard(url)  # let it retry
                        await queue.put(url)
                        queued.add(url)
                    else:
                        logger.warning("Giving up on %s after %d attempts", url, attempts)
                except Exception as e:
                    logger.error("Uncaught error on %s: %s", url, e)
                finally:
                    queue.task_done()

        # Compile regex here (import placed late to avoid shadowing)
        import re  # noqa: WPS433 (intentional local import for clarity)

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
    async def _fetch_and_extract(
        self,
        url: str,
        *,
        allow_re=None,
        deny_re=None,
    ) -> PageSnapshot:
        """
        Static-first: try httpx; on JS-app, auth/blocked statuses, or TLS/network problems,
        fall back to Playwright rendering once within the same attempt.
        """
        if bool(getattr(self.cfg, "enable_static_first", False)):
            try:
                return await self._fetch_static(url, allow_re=allow_re, deny_re=deny_re)
            except NeedsBrowser:
                # fall through to Playwright below
                pass
            except NonRetryableHTTPError:
                # 404 etc. — propagate so caller won't retry
                raise
            except TransientHTTPError:
                # let retry decorator handle, but first try a browser pass
                pass
            except Exception as e:
                # unknown issues → attempt browser before treating as transient
                logger.debug("Static fetch unknown error on %s: %s (will try browser)", url, e)

        # Playwright fallback / dynamic render
        page = await self.context.new_page()
        status = None
        try:
            status, html = await play_goto(
                page,
                url,
                [self.wait_until, "domcontentloaded", "load"],
                self.cfg.page_load_timeout_ms,
            )
            exc = http_status_to_exc(status)
            if exc:
                raise exc

            title = await play_title(page)
            if html and len(html) > 3_000_000:  # clamp giant pages
                html = html[:3_000_000]

            html_path = None
            if self.cfg.cache_html:
                html_path = self._save_html(url, html or "")

            links = await play_links(page)
            links = [normalize_url(l) for l in links if is_http_url(l)]
            filtered = filter_links(url, links, self.cfg, allow_re=allow_re, deny_re=deny_re, product_only=True)

            return PageSnapshot(url=url, title=title, html_path=html_path, out_links=filtered)

        except NonRetryableHTTPError:
            raise
        except Exception as e:
            if status is not None:
                logger.debug("Error on %s (status=%s): %s", url, status, e)
            raise TransientHTTPError(str(e))
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def fetch_dynamic_only(self, url: str, *, allow_re=None, deny_re=None) -> PageSnapshot:
        """
        Always render with Playwright once (no retry decorator here);
        let caller decide if/how to retry.
        """
        page = await self.context.new_page()
        status = None
        try:
            status, html = await play_goto(
                page,
                url,
                [self.wait_until, "domcontentloaded", "load"],
                self.cfg.page_load_timeout_ms,
            )
            exc = http_status_to_exc(status)
            if exc:
                raise exc

            title = await play_title(page)
            if html and len(html) > 3_000_000:
                html = html[:3_000_000]

            html_path = None
            if self.cfg.cache_html:
                html_path = self._save_html(url, html or "")

            links = await play_links(page)
            links = [normalize_url(l) for l in links if is_http_url(l)]
            filtered = filter_links(url, links, self.cfg, allow_re=allow_re, deny_re=deny_re, product_only=False)

            return PageSnapshot(url=url, title=title, html_path=html_path, out_links=filtered)

        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _fetch_static(self, url: str, *, allow_re=None, deny_re=None) -> PageSnapshot:
        """
        Fetch with httpx; save HTML immediately; if page looks client-rendered
        or blocked (auth/CAPTCHA), raise NeedsBrowser to re-fetch with Playwright.
        """
        client = httpx_client(self.cfg)
        try:
            try:
                r = await client.get(url)
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                # TLS/timeouts/network problems → try browser
                raise NeedsBrowser(str(e)) from e

            s = r.status_code
            if s == 404:
                raise NonRetryableHTTPError(f"HTTP 404 for {url}")
            if s in (401, 403, 429, 503):
                # likely bot protection or auth needed → browser
                raise NeedsBrowser(f"HTTP {s} for {url}")
            if s >= 400:
                # other 4xx/5xx → transient (let retry)
                raise TransientHTTPError(f"HTTP {s} for {url}")

            html = r.text or ""

            # Save raw HTML *before* heuristics so we keep a debug artifact
            html_path = None
            if self.cfg.cache_html:
                # clamp before saving to avoid giant files
                max_bytes = int(getattr(self.cfg, "static_max_bytes", 2_000_000))
                if len(html) > max_bytes:
                    html = html[:max_bytes]
                html_path = self._save_html(url, html)

            # Heuristic: if it looks like a JS app and visible text is tiny, ask for browser
            threshold = int(getattr(self.cfg, "static_js_app_text_threshold", 300))
            if looks_like_js_app(html, threshold):
                raise NeedsBrowser("Likely client-rendered app")

            # Extract links/title using static parser
            title = extract_title_static(html)
            links = extract_links_static(html, url)
            links = [normalize_url(l) for l in links if is_http_url(l)]

            filtered = filter_links(url, links, self.cfg, allow_re=allow_re, deny_re=deny_re, product_only=True)

            return PageSnapshot(url=url, title=title, html_path=html_path, out_links=filtered)

        except NeedsBrowser:
            # propagate to _fetch_and_extract to run Playwright
            raise
        except NonRetryableHTTPError:
            raise
        except Exception as e:
            # Unknown condition → treat transient so retry_async can re-attempt or fall back
            raise TransientHTTPError(str(e)) from e
        finally:
            await client.aclose()

    def _save_html(self, url: str, html: str) -> str:
        parsed = urlparse(url)
        host = (parsed.hostname or "unknown-host").lower()
        base_host = get_base_domain(host)  # collapse subdomains to eTLD+1

        base = parsed.path or "/"
        if base.endswith("/"):
            base += "index"
        slug = slugify(base, max_len=80)

        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        fname = f"{slug}-{h}.html"

        out_dir = self.cfg.scraped_html_dir / base_host
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        atomic_write_text(out_path, html)
        return str(out_path)