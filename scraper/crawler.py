from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Callable
from urllib.parse import urlparse, urljoin

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
    html_path: Optional[str]
    out_links: List[str] = field(default_factory=list)
    dropped: List[tuple[str, str]] = field(default_factory=list)
    content_sha1: Optional[str] = None
    not_modified: bool = False


class SiteCrawler:
    """
    Concurrent, retrying, per-domain crawler.
    Emits status-driven signals to the runner via callbacks (redirects, canonical, backoff, server errors).
    """

    def __init__(
        self,
        cfg: Config,
        context: BrowserContext,
        *,
        on_redirect: Optional[Callable[[str, str, int, bool], None]] = None,
        on_canonical: Optional[Callable[[str, str, bool], None]] = None,
        on_backoff: Optional[Callable[[str], None]] = None,
        on_server_error: Optional[Callable[[str, str, int], None]] = None,
    ):
        self.cfg = cfg
        self.context = context
        self.wait_until = cfg.navigation_wait_until

        self.on_redirect = on_redirect
        self.on_canonical = on_canonical
        self.on_backoff = on_backoff
        self.on_server_error = on_server_error

        self.sem = asyncio.Semaphore(cfg.max_pages_per_domain_parallel)
        self._attempts: Dict[str, int] = {}
        self._max_retries: int = int(getattr(cfg, "crawler_max_retries", 3))

        # set per-crawl homepage to detect “homepage special cases”
        self._homepage_url: Optional[str] = None

        self._ims: Dict[str, float] = {}
        self._prio_seq: int = 0 

    # ----------------- public API -----------------

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

        if max_pages is None:
            max_pages = getattr(self.cfg, "max_pages_per_company", None)

        host = urlparse(start_url).hostname or ""
        logger.info("Crawling %s", host)

        visited: Set[str] = set()
        queued: Set[str] = set()

        # --- priority queue: producty first ---
        from asyncio import PriorityQueue
        queue: PriorityQueue[tuple[int, int, str]] = PriorityQueue()
        prio_seq = 0

        import re
        allow_re = re.compile(url_allow_regex) if url_allow_regex else None
        deny_re  = re.compile(url_deny_regex)  if url_deny_regex  else None

        def _priority(u: str) -> int:
            # 0 for product-like URLs; 1 otherwise
            try:
                from .utils import is_producty_url
                return 0 if is_producty_url(u) else 1
            except Exception:
                return 1

        await queue.put((_priority(start_url), prio_seq, start_url))
        prio_seq += 1
        queued.add(start_url)

        snapshots: Dict[str, PageSnapshot] = {}

        async def worker():
            nonlocal prio_seq
            while True:
                try:
                    prio, _, url = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    return

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

                    if self.cfg.per_page_delay_ms > 0:
                        await asyncio.sleep(self.cfg.per_page_delay_ms / 1000.0)

                    snapshots[url] = snap

                    for link in snap.out_links:
                        if max_pages is not None and len(visited) >= max_pages:
                            break
                        if link in visited or link in queued:
                            continue
                        await queue.put((_priority(link), prio_seq, link))
                        prio_seq += 1
                        queued.add(link)

                except NonRetryableHTTPError as e:
                    logger.debug("Non-retryable on %s: %s", url, e)
                except TransientHTTPError as e:
                    attempts = getattr(self, "_attempts", {}).get(url, 0) + 1
                    self._attempts[url] = attempts
                    if attempts <= getattr(self, "_max_retries", 3):
                        logger.debug("Transient error on %s (attempt %d/%d): %s", url, attempts, self._max_retries, e)
                        visited.discard(url)
                        await queue.put((_priority(url), prio_seq, url))
                        prio_seq += 1
                        queued.add(url)
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

    
    def set_if_modified_since(self, last_seen: Optional[Dict[str, float]]):
        self._ims = {normalize_url(k): float(v) for k, v in (last_seen or {}).items() if v}


    # ----------------- internals -----------------

    def _is_homepage(self, src: str) -> bool:
        try:
            return normalize_url(src) == normalize_url(self._homepage_url or "")
        except Exception:
            return False

    @staticmethod
    def _extract_canonical_href(html: str) -> Optional[str]:
        """
        Minimal parse for <link rel="canonical" href="..."> without BeautifulSoup dep.
        """
        if not html:
            return None
        # allow rel="canonical" in any order (rel attr must contain 'canonical')
        m = re.search(
            r'<link\s+[^>]*rel=["\']?[^"\']*canonical[^"\']*["\']?[^>]*>',
            html,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        tag = m.group(0)
        href = re.search(r'href=["\']([^"\']+)["\']', tag, flags=re.IGNORECASE)
        return href.group(1) if href else None

    def _signal_redirect(self, src: str, dst: str, code: int):
        if self.on_redirect:
            try:
                self.on_redirect(src, dst, code, self._is_homepage(src))
            except Exception as e:
                logger.debug("on_redirect callback failed: %s", e)

    def _signal_canonical(self, src: str, dst: str):
        if self.on_canonical:
            try:
                self.on_canonical(src, dst, self._is_homepage(src))
            except Exception as e:
                logger.debug("on_canonical callback failed: %s", e)

    def _signal_backoff(self, host: str):
        if self.on_backoff:
            try:
                self.on_backoff(host)
            except Exception as e:
                logger.debug("on_backoff callback failed: %s", e)

    def _signal_server_error(self, host: str, path: str, status: int):
        if self.on_server_error:
            try:
                self.on_server_error(host, path, status)
            except Exception as e:
                logger.debug("on_server_error callback failed: %s", e)

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
        Static-first; fallback to Playwright; emit signals for redirects, canonicals, and backoff/server errors.
        """
        if bool(getattr(self.cfg, "enable_static_first", False)):
            try:
                return await self._fetch_static(url, allow_re=allow_re, deny_re=deny_re)
            except NeedsBrowser:
                pass
            except NonRetryableHTTPError:
                raise
            except TransientHTTPError:
                pass
            except Exception as e:
                logger.debug("Static fetch unknown error on %s: %s (will try browser)", url, e)

        # Dynamic render
        page = await self.context.new_page()
        status = None
        try:
            status, html = await play_goto(
                page,
                url,
                [self.wait_until, "domcontentloaded", "load"],
                self.cfg.page_load_timeout_ms,
            )

            # status-aware decisions
            if status == 429:
                host = urlparse(url).hostname or ""
                self._signal_backoff(host)

            exc = http_status_to_exc(status)
            if exc:
                # Special-case 401/403: allow extra grace on homepage only
                if status in (401, 403):
                    if self._is_homepage(url):
                        raise TransientHTTPError(f"HTTP {status} on homepage {url}")
                    raise NonRetryableHTTPError(f"HTTP {status} for {url}")
                if 500 <= (status or 0) <= 599:
                    up = urlparse(url)
                    self._signal_server_error(up.hostname or "", up.path or "", status or 0)
                raise exc

            # JS redirect off-site
            final_url = page.url
            if not same_site(url, final_url, self.cfg.allow_subdomains):
                logger.debug("Off-domain JS redirect: %s -> %s ; dropping.", url, final_url)
                # treat as redirect event (no HTTP code; use 307 as a neutral JS redirect code)
                self._signal_redirect(url, final_url, 307)
                raise NonRetryableHTTPError(f"Off-domain JS redirect to {final_url}")

            title = await play_title(page)
            if html and len(html) > 3_000_000:
                html = html[:3_000_000]

            # 200 + canonical pointing off-site → signal + possibly suppress frontier (runner decides)
            canon = self._extract_canonical_href(html or "")
            if canon:
                try:
                    if get_base_domain(urlparse(canon).hostname or "") != get_base_domain(urlparse(final_url).hostname or ""):
                        logger.debug("Off-site canonical: %s -> %s", final_url, canon)
                        self._signal_canonical(final_url, canon)
                except Exception:
                    pass

            html_path = None
            if self.cfg.cache_html:
                html_path = self._save_html(final_url, html or "")

            links = await play_links(page)
            links = [normalize_url(l) for l in links if is_http_url(l)]

            dropped: list[tuple[str, str]] = []
            filtered = filter_links(
                final_url,
                links,
                self.cfg,
                allow_re=allow_re,
                deny_re=deny_re,
                product_only=True,
                on_drop=lambda u, r: dropped.append((u, r)),
            )

            return PageSnapshot(url=final_url, title=title, html_path=html_path, out_links=filtered, dropped=dropped)

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
        Always render with Playwright once (no retry decorator here).
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

            if status == 429:
                host = urlparse(url).hostname or ""
                self._signal_backoff(host)

            exc = http_status_to_exc(status)
            if exc:
                if status in (401, 403):
                    if self._is_homepage(url):
                        raise TransientHTTPError(f"HTTP {status} on homepage {url}")
                    raise NonRetryableHTTPError(f"HTTP {status} for {url}")
                if 500 <= (status or 0) <= 599:
                    up = urlparse(url)
                    self._signal_server_error(up.hostname or "", up.path or "", status or 0)
                raise exc

            final_url = page.url
            if not same_site(url, final_url, self.cfg.allow_subdomains):
                logger.debug("Off-domain JS redirect: %s -> %s ; dropping.", url, final_url)
                self._signal_redirect(url, final_url, 307)
                raise NonRetryableHTTPError(f"Off-domain JS redirect to {final_url}")

            title = await play_title(page)
            if html and len(html) > 3_000_000:
                html = html[:3_000_000]

            canon = self._extract_canonical_href(html or "")
            if canon:
                try:
                    if get_base_domain(urlparse(canon).hostname or "") != get_base_domain(urlparse(final_url).hostname or ""):
                        logger.debug("Off-site canonical: %s -> %s", final_url, canon)
                        self._signal_canonical(final_url, canon)
                except Exception:
                    pass

            html_path = None
            if self.cfg.cache_html:
                html_path = self._save_html(final_url, html or "")

            links = await play_links(page)
            links = [normalize_url(l) for l in links if is_http_url(l)]

            dropped: list[tuple[str, str]] = []
            filtered = filter_links(
                final_url,
                links,
                self.cfg,
                allow_re=allow_re,
                deny_re=deny_re,
                product_only=True,
                on_drop=lambda u, r: dropped.append((u, r)),
            )

            return PageSnapshot(url=final_url, title=title, html_path=html_path, out_links=filtered, dropped=dropped)

        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _fetch_static(self, url: str, *, allow_re=None, deny_re=None) -> PageSnapshot:
        """
        Fetch with httpx; preflight 3xx; save HTML; JS-app heuristic → NeedsBrowser.
        Emits redirect/backoff/server-error signals for runner state.
        """
        client = httpx_client(self.cfg)
        try:
            # --- preflight: detect off-site 3xx early (don’t spend cycles following) ---
            try:
                r0 = await client.get(url, follow_redirects=False)
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                raise NeedsBrowser(str(e)) from e

            if 300 <= r0.status_code < 400:
                loc = r0.headers.get("location") or r0.headers.get("Location")
                if loc:
                    target = urljoin(url, loc)
                    # signal redirect (runner may count toward migration/probation)
                    try:
                        self._signal_redirect(url, target, r0.status_code)
                    except Exception:
                        pass
                    # off-site: drop fast (your runner handles migration candidates)
                    if not same_site(url, target, self.cfg.allow_subdomains):
                        logger.debug("Off-domain redirect: %s -> %s ; dropping.", url, target)
                        raise NonRetryableHTTPError(f"Off-domain redirect to {target}")

            # --- follow with conditional headers (If-Modified-Since) ---
            try:
                ims_epoch = getattr(self, "_ims", {}).get(normalize_url(url))
                headers = None
                if ims_epoch:
                    from .utils import httpdate_from_epoch
                    headers = {"If-Modified-Since": httpdate_from_epoch(float(ims_epoch))}
                r = await client.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                raise NeedsBrowser(str(e)) from e

            s = r.status_code
            if s == 304:
                # Not modified since last crawl — return empty frontier
                return PageSnapshot(url=url, title=None, html_path=None, out_links=[], dropped=dropped)

            if s == 429:
                host = urlparse(url).hostname or ""
                self._signal_backoff(host)

            if s == 404:
                raise NonRetryableHTTPError(f"HTTP 404 for {url}")
            if s in (401, 403):
                # try browser; dynamic path will special-case homepage/non-homepage
                raise NeedsBrowser(f"HTTP {s} for {url}")
            if 500 <= s <= 599:
                up = urlparse(url)
                self._signal_server_error(up.hostname or "", up.path or "", s)
                raise TransientHTTPError(f"HTTP {s} for {url}")
            if s >= 400:
                raise TransientHTTPError(f"HTTP {s} for {url}")

            html = r.text or ""

            # Save (truncate if needed)
            html_path = None
            if self.cfg.cache_html:
                max_bytes = int(getattr(self.cfg, "static_max_bytes", 2_000_000))
                if len(html) > max_bytes:
                    html = html[:max_bytes]
                html_path = self._save_html(url, html)

            # JS-app heuristic → punt to browser
            threshold = int(getattr(self.cfg, "static_js_app_text_threshold", 300))
            if looks_like_js_app(html, threshold):
                raise NeedsBrowser("Likely client-rendered app")

            # 200 with off-site rel=canonical → signal
            canon = self._extract_canonical_href(html or "")
            if canon:
                try:
                    if get_base_domain(urlparse(canon).hostname or "") != get_base_domain(urlparse(url).hostname or ""):
                        logger.debug("Off-site canonical: %s -> %s", url, canon)
                        self._signal_canonical(url, canon)
                except Exception:
                    pass

            # Extract title/links; filter downstream
            title = extract_title_static(html)
            links = extract_links_static(html, url)
            links = [normalize_url(l) for l in links if is_http_url(l)]

            dropped: list[tuple[str, str]] = []
            filtered = filter_links(
                url,
                links,
                self.cfg,
                allow_re=allow_re,
                deny_re=deny_re,
                product_only=True,
                on_drop=lambda u, r: dropped.append((u, r)),
            )

            try:
                from hashlib import sha1 as _sha1
                content_sha1 = _sha1(html.encode("utf-8", errors="ignore")).hexdigest()
            except Exception:
                content_sha1 = None

            snap = PageSnapshot(url=url, title=title, html_path=html_path, out_links=filtered, dropped=dropped)
            setattr(snap, "content_sha1", content_sha1)

            return snap

        except NeedsBrowser:
            raise
        except NonRetryableHTTPError:
            raise
        except Exception as e:
            raise TransientHTTPError(str(e)) from e
        finally:
            await client.aclose()


    def _save_html(self, url: str, html: str) -> str:
        parsed = urlparse(url)
        host = (parsed.hostname or "unknown-host").lower()
        base_host = get_base_domain(host)

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