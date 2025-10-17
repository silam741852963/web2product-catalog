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
import httpcore
from playwright.async_api import BrowserContext

from .config import Config
from .browser import maybe_render_html   # static-first, budgeted render
from .utils import (
    normalize_url,
    is_http_url,
    slugify,
    atomic_write_text,
    retry_async,
    TransientHTTPError,
    NonRetryableHTTPError,
    get_base_domain,
    looks_like_js_app,
    extract_links_static,
    extract_title_static,
    same_site,
    httpx_client,
    should_render,            # lightweight static gating
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

        # Dual httpx clients for HTTP/2 and HTTP/1.1 with per-host downgrade on flakiness
        self._client_h2: httpx.AsyncClient = self._new_client(http2=True)
        self._client_h1: httpx.AsyncClient = self._new_client(http2=False)
        self._h2_blacklist: set[str] = set()
        self._closed: bool = False

    # ---------- client construction / lifecycle & selection ----------

    def _new_client(self, *, http2: bool) -> httpx.AsyncClient:
        """
        Version-agnostic client factory:
        - Try utils.httpx_client(cfg, http2=...), if that kw exists.
        - Else build a minimal AsyncClient locally with widely-supported kwargs.
        - If proxies aren't supported by this httpx version, omit them.
        """
        # Path A: your utils has http2 kw
        try:
            return httpx_client(self.cfg, http2=http2)  # type: ignore[call-arg]
        except TypeError:
            pass
        except Exception:
            # If utils.httpx_client threw something else, fall back to local too
            pass

        # Path B: build locally with broadly-supported args
        timeout_cfg = getattr(self.cfg, "http_timeout", 20.0)
        timeout = timeout_cfg if isinstance(timeout_cfg, httpx.Timeout) else httpx.Timeout(float(timeout_cfg))

        limits = httpx.Limits(
            max_connections=int(getattr(self.cfg, "http_max_connections", 128)),
            max_keepalive_connections=int(getattr(self.cfg, "http_max_keepalive", 32)),
            keepalive_expiry=float(getattr(self.cfg, "http_keepalive_expiry", 30.0)),
        )

        headers = {"Accept": "*/*"}
        ua = getattr(self.cfg, "user_agent", None)
        if ua:
            headers["User-Agent"] = ua

        trust_env = bool(getattr(self.cfg, "http_trust_env", True))
        proxy = getattr(self.cfg, "http_proxy", None)

        base_kwargs = dict(http2=http2, timeout=timeout, limits=limits, headers=headers, trust_env=trust_env)

        # Try with proxies if provided; fall back if this httpx doesn't accept the kw
        if proxy:
            try:
                return httpx.AsyncClient(proxies=proxy, **base_kwargs)  # type: ignore[arg-type]
            except TypeError:
                pass  # proxies kw not accepted in this runtime

        return httpx.AsyncClient(**base_kwargs)

    def _ensure_client(self) -> None:
        """Recreate clients if missing/closed."""
        if getattr(self, "_client_h2", None) is None or getattr(self._client_h2, "is_closed", False):
            self._client_h2 = self._new_client(http2=True)
        if getattr(self, "_client_h1", None) is None or getattr(self._client_h1, "is_closed", False):
            self._client_h1 = self._new_client(http2=False)

    def _client_for_host(self, host: str, *, prefer_http1: bool = False):
        """
        Choose the httpx client for this host.
        If the host previously errored on HTTP/2, prefer HTTP/1.1 with Connection: close.
        """
        if prefer_http1 or host in self._h2_blacklist:
            return self._client_h1, {"Connection": "close"}
        return self._client_h2, {}

    async def crawl_site(
        self,
        homepage: str,
        max_pages: Optional[int] = None,
        url_allow_regex: Optional[str] = None,
        url_deny_regex: Optional[str] = None,
    ) -> List[PageSnapshot]:
        # ensure clients alive before first use
        self._ensure_client()

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
                        logger.debug("Transient error on %s attempt %d/%d: %s", url, attempts, self._max_retries, e)
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
            # If your runner closes the crawler, you can remove the next line.
            await self.aclose()

        logger.info("Crawl finished: %s pages from %s", len(snapshots), host)
        return list(snapshots.values())

    async def aclose(self):
        if not self._closed:
            self._closed = True
            for attr in ("_client_h2", "_client_h1"):
                cli = getattr(self, attr, None)
                if cli is not None:
                    try:
                        await cli.aclose()
                    except Exception:
                        pass
                    setattr(self, attr, None)

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
        """Smaller bump than 429: network slowness or renderer hiccup."""
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

    # -----------------------------
    # DRY extract + filter routine
    # -----------------------------
    def _extract_and_filter(self, base_url: str, html: str, *, cache_html: bool, allow_re=None, deny_re=None) -> PageSnapshot:
        """Common logic to build PageSnapshot from HTML (static or rendered)."""
        if not isinstance(html, str):
            html = html or ""
        if cache_html:
            # Cap size before write
            max_bytes = int(getattr(self.cfg, "static_max_bytes", 2_000_000))
            if len(html) > max_bytes:
                html = html[:max_bytes]

        canon = self._extract_canonical_href(html or "")
        if canon:
            try:
                if get_base_domain(urlparse(canon).hostname or "") != get_base_domain(urlparse(base_url).hostname or ""):
                    logger.debug("Off-site canonical: %s -> %s", base_url, canon)
                    self._signal_canonical(base_url, canon)
            except Exception:
                pass

        title = extract_title_static(html)
        links = extract_links_static(html, base_url)
        links = [normalize_url(l) for l in links if is_http_url(l)]

        dropped: list[tuple[str, str]] = []
        filtered = filter_links(
            base_url,
            links,
            self.cfg,
            allow_re=allow_re,
            deny_re=deny_re,
            product_only=True,
            on_drop=lambda u, r: dropped.append((u, r)),
        )

        html_path = None
        if cache_html:
            html_path = self._save_html(base_url, html)

        try:
            content_sha1 = hashlib.sha1(html.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            content_sha1 = None

        snap = PageSnapshot(url=base_url, title=title, html_path=html_path, out_links=filtered, dropped=dropped)
        setattr(snap, "content_sha1", content_sha1)
        return snap

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
        """Static-first: fetch with httpx; render only if heuristics+budget say so."""
        # ensure clients exist
        self._ensure_client()
        up = urlparse(url)
        host = (up.hostname or "").lower()
        client, extra_headers = self._client_for_host(host)

        # Step 1: cheap redirect probe
        try:
            r0 = await client.get(url, follow_redirects=False, headers=extra_headers)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            # transient network: retry allowed by decorator
            raise TransientHTTPError(str(e)) from e
        except (httpx.RemoteProtocolError, httpcore.RemoteProtocolError, httpcore.ProtocolError) as e:
            # auto-downgrade this host to HTTP/1.1 once
            if host not in self._h2_blacklist:
                self._h2_blacklist.add(host)
                client, extra_headers = self._client_for_host(host, prefer_http1=True)
                try:
                    r0 = await client.get(url, follow_redirects=False, headers=extra_headers)
                except Exception as ee:
                    raise TransientHTTPError(f"h2->h1 retry failed: {ee}") from ee
            else:
                raise TransientHTTPError(str(e)) from e

        if 300 <= r0.status_code < 400:
            loc = r0.headers.get("location") or r0.headers.get("Location")
            if loc:
                target = urljoin(url, loc)
                self._signal_redirect(url, target, r0.status_code)
                if not same_site(url, target, getattr(self.cfg, "allow_subdomains", True)):
                    logger.debug("Off-domain redirect: %s -> %s ; dropping.", url, target)
                    raise NonRetryableHTTPError(f"Off-domain redirect to {target}")

        # Step 2: Conditional GET (If-Modified-Since)
        try:
            ims_epoch = getattr(self, "_ims", {}).get(normalize_url(url))
            headers = dict(extra_headers)
            if ims_epoch:
                from .utils import httpdate_from_epoch
                headers["If-Modified-Since"] = httpdate_from_epoch(float(ims_epoch))
            r = await client.get(url, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise TransientHTTPError(str(e)) from e
        except (httpx.RemoteProtocolError, httpcore.RemoteProtocolError, httpcore.ProtocolError) as e:
            # downgrade to HTTP/1.1 once on protocol errors
            if host not in self._h2_blacklist:
                self._h2_blacklist.add(host)
                client, extra_headers = self._client_for_host(host, prefer_http1=True)
                headers = dict(extra_headers)
                if ims_epoch:
                    from .utils import httpdate_from_epoch
                    headers["If-Modified-Since"] = httpdate_from_epoch(float(ims_epoch))
                try:
                    r = await client.get(url, headers=headers)
                except Exception as ee:
                    raise TransientHTTPError(f"h2->h1 retry failed: {ee}") from ee
            else:
                raise TransientHTTPError(str(e)) from e

        s = r.status_code
        self._signal_http_status(up.hostname or "", up.path or "", int(s))

        if s == 304:
            self._decay_penalty_on_success()
            return PageSnapshot(url=url, title=None, html_path=None, out_links=[], dropped=[], not_modified=True)

        if s == 429:
            self._signal_backoff(host)
            self._bump_penalty_on_429()

        if s == 404:
            raise NonRetryableHTTPError(f"HTTP 404 for {url}")
        if s in (401, 403):
            # Policy: treat as non-retryable (no render) unless explicitly desired
            raise NonRetryableHTTPError(f"HTTP {s} for {url}")
        if 500 <= s <= 599:
            self._signal_server_error(host, up.path or "", s)
            raise TransientHTTPError(f"HTTP {s} for {url}")
        if s >= 400:
            raise TransientHTTPError(f"HTTP {s} for {url}")

        html = r.text or ""

        # Static-first gate — only render when clearly needed & budget allows
        if should_render(url, html) and await self._render_budget_allows(url):
            rendered = await maybe_render_html(url, html, context=self.context)
            if rendered:
                self._decay_penalty_on_success()
                return self._extract_and_filter(url, rendered, cache_html=self.cfg.cache_html, allow_re=allow_re, deny_re=deny_re)

        # If it *looks* like a JS app but budget denied, keep static result
        threshold = int(getattr(self.cfg, "static_js_app_text_threshold", 800))
        if looks_like_js_app(html, threshold):
            logger.debug("Static page looks like JS app but render budget denied — keeping static for %s", url)

        self._decay_penalty_on_success()
        return self._extract_and_filter(url, html, cache_html=self.cfg.cache_html, allow_re=allow_re, deny_re=deny_re)

    async def _render_budget_allows(self, url: str) -> bool:
        """Ask browser's budget (via maybe_render_html semantics). We simulate check by a dry-run:
        Return True if we *might* render (final decision is inside maybe_render_html)."""
        return True

    # -------------------
    # HTML file writer
    # -------------------
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