from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Callable
from urllib.parse import urlparse, urljoin

import httpx
from playwright.async_api import BrowserContext, Error as PWError

from .config import Config
from .browser import acquire_page
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

# Patterns to classify PW errors more precisely
_PW_SSL_NONRETRY = (
    "ERR_CERT_COMMON_NAME_INVALID",
    "ERR_SSL_VERSION_OR_CIPHER_MISMATCH",
    "ERR_CERT_AUTHORITY_INVALID",
    "ERR_CERT_INVALID",
)
_PW_HTTP2_TRANSIENT = ("ERR_HTTP2_PROTOCOL_ERROR",)
_PW_RENDERER_CRASH = ("chrome-error://chromewebdata",)
_PW_NET_TIMEOUT = ("ERR_CONNECTION_TIMED_OUT",)
_PW_NAV_TIMEOUT_MARKER = ("Timeout", "navigating to", "waiting until")


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
    def __init__(
        self,
        cfg: Config,
        context: BrowserContext,
        *,
        on_redirect: Optional[Callable[[str, str, int, bool], None]] = None,
        on_canonical: Optional[Callable[[str, str, bool], None]] = None,
        on_backoff: Optional[Callable[[str], None]] = None,
        on_server_error: Optional[Callable[[str, str, int], None]] = None,
        on_http_status: Optional[Callable[[str, str, int], None]] = None,
    ):
        self.cfg = cfg
        self.context = context
        self.wait_until = cfg.navigation_wait_until

        self.on_redirect = on_redirect
        self.on_canonical = on_canonical
        self.on_backoff = on_backoff
        self.on_server_error = on_server_error
        self.on_http_status = on_http_status

        self.sem = asyncio.Semaphore(cfg.max_pages_per_domain_parallel)
        self._attempts: Dict[str, int] = {}
        self._max_retries: int = int(getattr(cfg, "crawler_max_retries", 3))

        self._homepage_url: Optional[str] = None
        self._ims: Dict[str, float] = {}

        self._throttle_lock = asyncio.Lock()
        self._last_request_t: float = 0.0
        self._penalty_ms: float = 0.0

        self._client: httpx.AsyncClient = httpx_client(self.cfg)
        self._closed: bool = False

    async def crawl_site(
        self,
        homepage: str,
        max_pages: Optional[int] = None,
        url_allow_regex: Optional[str] = None,
        url_deny_regex: Optional[str] = None,
    ) -> List[PageSnapshot]:
        start_url = normalize_url(homepage)
        self._homepage_url = start_url
        if not is_http_url(start_url):
            logger.warning("Skip non-http URL: %s", start_url)
            return []

        if max_pages is None:
            max_pages = getattr(self.cfg, "max_pages_per_company", None)

        host = urlparse(start_url).hostname or ""
        logger.info("Crawling %s", host)

        visited: Set[str] = set()
        queued: Set[str] = set()

        from asyncio import PriorityQueue
        queue: PriorityQueue[tuple[int, int, str]] = PriorityQueue()
        prio_seq = 0

        import re as _re
        allow_re = _re.compile(url_allow_regex) if url_allow_regex else None
        deny_re  = _re.compile(url_deny_regex)  if url_deny_regex  else None

        def _priority(u: str) -> int:
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
                    await self._throttle_if_needed()

                    async with self.sem:
                        snap = await self._fetch_and_extract(url, allow_re=allow_re, deny_re=deny_re)

                    if self.cfg.per_page_delay_ms > 0:
                        await asyncio.sleep(self.cfg.per_page_delay_ms / 1000.0)

                    snapshots[url] = snap
                    self._decay_penalty_soft()

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
                        await asyncio.sleep(min(1.0, self._penalty_ms / 1000.0))
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
        try:
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.aclose()

        logger.info("Crawl finished: %s pages from %s", len(snapshots), host)
        return list(snapshots.values())

    async def aclose(self):
        if not self._closed:
            self._closed = True
            try:
                await self._client.aclose()
            except Exception:
                pass

    def set_if_modified_since(self, last_seen: Optional[Dict[str, float]]):
        self._ims = {normalize_url(k): float(v) for k, v in (last_seen or {}).items() if v}

    def _is_homepage(self, src: str) -> bool:
        try:
            return normalize_url(src) == normalize_url(self._homepage_url or "")
        except Exception:
            return False

    # pacing / penalty
    async def _throttle_if_needed(self):
        min_interval_s = max(0.0, (getattr(self.cfg, "host_min_interval_ms", 0) or 0) / 1000.0)
        penalty_s = max(0.0, self._penalty_ms / 1000.0)
        async with self._throttle_lock:
            now = time.monotonic()
            delay = 0.0
            if self._last_request_t > 0 and min_interval_s > 0:
                elapsed = now - self._last_request_t
                if elapsed < min_interval_s:
                    delay += (min_interval_s - elapsed)
            delay += penalty_s
            self._last_request_t = now + delay
        if delay > 0:
            await asyncio.sleep(delay)

    def _bump_penalty_on_429(self):
        max_ms = int(getattr(self.cfg, "throttle_penalty_max_ms", 30000))
        init_ms = int(getattr(self.cfg, "throttle_penalty_initial_ms", 2000))
        mult = float(getattr(self.cfg, "backoff_on_429", 1.5) or 1.5)
        if self._penalty_ms <= 0:
            self._penalty_ms = float(init_ms)
        else:
            self._penalty_ms = min(float(max_ms), self._penalty_ms * mult)

    def _bump_penalty_on_timeout(self):
        """
        Smaller bump than 429: network slowness or renderer hiccup.
        """
        max_ms = int(getattr(self.cfg, "throttle_penalty_max_ms", 30000))
        base = max(500.0, self._penalty_ms)  # at least 0.5s
        self._penalty_ms = min(float(max_ms), base * 1.25)

    def _decay_penalty_on_success(self):
        decay = float(getattr(self.cfg, "throttle_penalty_decay_mult", 0.66) or 0.66)
        if self._penalty_ms > 0:
            self._penalty_ms *= decay
            if self._penalty_ms < 50:
                self._penalty_ms = 0.0

    def _decay_penalty_soft(self):
        if self._penalty_ms > 0:
            self._penalty_ms *= 0.98
            if self._penalty_ms < 50:
                self._penalty_ms = 0.0

    @staticmethod
    def _extract_canonical_href(html: str) -> Optional[str]:
        if not html:
            return None
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
            except Exception:
                pass

    def _signal_canonical(self, src: str, dst: str):
        if self.on_canonical:
            try:
                self.on_canonical(src, dst, self._is_homepage(src))
            except Exception:
                pass

    def _signal_backoff(self, host: str):
        if self.on_backoff:
            try:
                self.on_backoff(host)
            except Exception:
                pass

    def _signal_server_error(self, host: str, path: str, status: int):
        if self.on_server_error:
            try:
                self.on_server_error(host, path, status)
            except Exception:
                pass

    def _signal_http_status(self, host: str, path: str, status: int):
        if self.on_http_status:
            try:
                self.on_http_status(host, path, status)
            except Exception:
                pass

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

        async with acquire_page(self.context) as page:
            try:
                overall_nav_timeout_s = max(1.0, (self.cfg.page_load_timeout_ms * 1.5) / 1000.0)
                status, html = await asyncio.wait_for(
                    play_goto(
                        page,
                        url,
                        [self.wait_until, "domcontentloaded", "load"],
                        self.cfg.page_load_timeout_ms,
                    ),
                    timeout=overall_nav_timeout_s,
                )

                try:
                    if page.is_closed():
                        raise TransientHTTPError("Page closed during navigation/post-processing")
                except Exception:
                    pass

                if status == 429:
                    host = urlparse(url).hostname or ""
                    self._signal_backoff(host)
                    self._bump_penalty_on_429()

                if status is not None:
                    up = urlparse(url)
                    self._signal_http_status(up.hostname or "", up.path or "", int(status))

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

                try:
                    title = await play_title(page)
                except PWError as e:
                    raise TransientHTTPError(f"Playwright title read failed: {e}")

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

                try:
                    links = await play_links(page)
                except PWError as e:
                    raise TransientHTTPError(f"Playwright links read failed: {e}")

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

                self._decay_penalty_on_success()

                return PageSnapshot(url=final_url, title=title, html_path=html_path, out_links=filtered, dropped=dropped)

            except asyncio.TimeoutError:
                # Our outer watchdog fired — transient and apply smaller penalty
                self._bump_penalty_on_timeout()
                raise TransientHTTPError("Navigation watchdog timeout")
            except PWError as e:
                msg = str(e)
                # Non-retryable TLS
                if any(p in msg for p in _PW_SSL_NONRETRY):
                    raise NonRetryableHTTPError(f"TLS error: {msg}")
                # HTTP/2 flaps: transient
                if any(p in msg for p in _PW_HTTP2_TRANSIENT):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"HTTP/2 protocol error: {msg}")
                # Renderer crash/error page: transient
                if any(p in msg for p in _PW_RENDERER_CRASH):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"Renderer crash/error page: {msg}")
                # Network timeout: transient + mild penalty
                if any(p in msg for p in _PW_NET_TIMEOUT):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"Network timeout: {msg}")
                # Playwright navigation timeout (class TimeoutError shows up as PWError)
                if any(p in msg for p in _PW_NAV_TIMEOUT_MARKER):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"Playwright timeout: {msg}")
                # Generic PW error: treat as transient
                self._bump_penalty_on_timeout()
                raise TransientHTTPError(f"Playwright error: {msg}")
            except Exception as e:
                self._bump_penalty_on_timeout()
                raise TransientHTTPError(f"Dynamic fetch error: {e}")

    async def fetch_dynamic_only(self, url: str, *, allow_re=None, deny_re=None) -> PageSnapshot:
        async with acquire_page(self.context) as page:
            try:
                overall_nav_timeout_s = max(1.0, (self.cfg.page_load_timeout_ms * 1.5) / 1000.0)
                status, html = await asyncio.wait_for(
                    play_goto(
                        page,
                        url,
                        [self.wait_until, "domcontentloaded", "load"],
                        self.cfg.page_load_timeout_ms,
                    ),
                    timeout=overall_nav_timeout_s,
                )

                try:
                    if page.is_closed():
                        raise TransientHTTPError("Page closed during navigation/post-processing")
                except Exception:
                    pass

                if status == 429:
                    host = urlparse(url).hostname or ""
                    self._signal_backoff(host)
                    self._bump_penalty_on_429()

                if status is not None:
                    up = urlparse(url)
                    self._signal_http_status(up.hostname or "", up.path or "", int(status))

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

                try:
                    title = await play_title(page)
                except PWError as e:
                    raise TransientHTTPError(f"Playwright title read failed: {e}")

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

                try:
                    links = await play_links(page)
                except PWError as e:
                    raise TransientHTTPError(f"Playwright links read failed: {e}")

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

                self._decay_penalty_on_success()

                return PageSnapshot(url=final_url, title=title, html_path=html_path, out_links=filtered, dropped=dropped)

            except asyncio.TimeoutError:
                self._bump_penalty_on_timeout()
                raise TransientHTTPError("Navigation watchdog timeout")
            except PWError as e:
                msg = str(e)
                if any(p in msg for p in _PW_SSL_NONRETRY):
                    raise NonRetryableHTTPError(f"TLS error: {msg}")
                if any(p in msg for p in _PW_HTTP2_TRANSIENT):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"HTTP/2 protocol error: {msg}")
                if any(p in msg for p in _PW_RENDERER_CRASH):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"Renderer crash/error page: {msg}")
                if any(p in msg for p in _PW_NET_TIMEOUT):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"Network timeout: {msg}")
                if any(p in msg for p in _PW_NAV_TIMEOUT_MARKER):
                    self._bump_penalty_on_timeout()
                    raise TransientHTTPError(f"Playwright timeout: {msg}")
                self._bump_penalty_on_timeout()
                raise TransientHTTPError(f"Playwright error: {msg}")
            except Exception as e:
                self._bump_penalty_on_timeout()
                raise TransientHTTPError(f"Dynamic fetch error: {e}")

    async def _fetch_static(self, url: str, *, allow_re=None, deny_re=None) -> PageSnapshot:
        client = self._client
        try:
            r0 = await client.get(url, follow_redirects=False)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise NeedsBrowser(str(e)) from e

        if 300 <= r0.status_code < 400:
            loc = r0.headers.get("location") or r0.headers.get("Location")
            if loc:
                target = urljoin(url, loc)
                try:
                    self._signal_redirect(url, target, r0.status_code)
                except Exception:
                    pass
                if not same_site(url, target, self.cfg.allow_subdomains):
                    logger.debug("Off-domain redirect: %s -> %s ; dropping.", url, target)
                    raise NonRetryableHTTPError(f"Off-domain redirect to {target}")

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
            self._decay_penalty_on_success()
            return PageSnapshot(url=url, title=None, html_path=None, out_links=[], dropped=[])

        if s == 429:
            host = urlparse(url).hostname or ""
            self._signal_backoff(host)
            self._bump_penalty_on_429()

        up = urlparse(url)
        self._signal_http_status(up.hostname or "", up.path or "", int(s))

        if s == 404:
            raise NonRetryableHTTPError(f"HTTP 404 for {url}")
        if s in (401, 403):
            raise NeedsBrowser(f"HTTP {s} for {url}")
        if 500 <= s <= 599:
            self._signal_server_error(up.hostname or "", up.path or "", s)
            raise TransientHTTPError(f"HTTP {s} for {url}")
        if s >= 400:
            raise TransientHTTPError(f"HTTP {s} for {url}")

        html = r.text or ""
        html_path = None
        if self.cfg.cache_html:
            max_bytes = int(getattr(self.cfg, "static_max_bytes", 2_000_000))
            if len(html) > max_bytes:
                html = html[:max_bytes]
            html_path = self._save_html(url, html)

        threshold = int(getattr(self.cfg, "static_js_app_text_threshold", 300))
        if looks_like_js_app(html, threshold):
            raise NeedsBrowser("Likely client-rendered app")

        canon = self._extract_canonical_href(html or "")
        if canon:
            try:
                if get_base_domain(urlparse(canon).hostname or "") != get_base_domain(urlparse(url).hostname or ""):
                    logger.debug("Off-site canonical: %s -> %s", url, canon)
                    self._signal_canonical(url, canon)
            except Exception:
                pass

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
            content_sha1 = hashlib.sha1(html.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            content_sha1 = None

        self._decay_penalty_on_success()

        snap = PageSnapshot(url=url, title=title, html_path=html_path, out_links=filtered, dropped=dropped)
        setattr(snap, "content_sha1", content_sha1)
        return snap

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