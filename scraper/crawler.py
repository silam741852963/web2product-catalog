from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from urllib.parse import urlparse, urljoin

from playwright.async_api import BrowserContext

from .config import Config
from .utils import (
    normalize_url,
    is_http_url,
    is_same_domain,  # still used when subdomains are disabled
    slugify,
    atomic_write_text,
    retry_async,
    TransientHTTPError,
)

logger = logging.getLogger(__name__)

# --- Non-retryable error for permanent failures like 404 ---
class NonRetryableHTTPError(Exception):
    pass


@dataclass
class PageSnapshot:
    url: str
    title: Optional[str]
    html_path: Optional[str]  # path on disk if cached
    out_links: List[str] = field(default_factory=list)


# --- URL hygiene gates ---
DENY_EXTS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".csv", ".mp4", ".mov", ".avi",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
)
TEL_PATH_RE = re.compile(r"(?:^|/)(?:tel|fax)\d{6,}(?:/|$)", flags=re.IGNORECASE)

def _should_visit(u: str) -> bool:
    """Skip obvious non-HTML or junk routes."""
    try:
        p = urlparse(u)
        if any(p.path.lower().endswith(ext) for ext in DENY_EXTS):
            return False
        if "download=YES" in (p.query or "").upper():
            return False
        if TEL_PATH_RE.search(p.path or ""):
            return False
        return True
    except Exception:
        return False


# --- Language / translation filtering ---
def _language_ok(u: str, cfg: Config) -> bool:
    """
    Heuristics to avoid non-primary language URLs:
      - deny language subdomains (e.g., fr., de., es.)
      - deny language path prefixes (e.g., /fr, /de)
      - deny language query keys when != en (lang, locale, hl)
    """
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        # subdomain deny prefixes
        if any(host.startswith(prefix.strip().lower()) for prefix in cfg.lang_subdomain_deny):
            return False

        # path deny tokens
        path = (p.path or "").lower()
        for tok in cfg.lang_path_deny:
            tok = tok.strip().lower()
            if not tok:
                continue
            # ensure leading slash semantics
            if not tok.startswith("/"):
                tok = "/" + tok
            if path == tok or path.startswith(tok + "/"):
                return False

        # query keys deny (if value is not en/en-us)
        q = (p.query or "").lower()
        if q:
            for key in cfg.lang_query_keys:
                key = key.strip().lower()
                if not key:
                    continue
                # crude check for presence
                if f"{key}=" in q:
                    # if not explicitly en or en-us, drop
                    if not re.search(fr"{key}=(en|en-us)\b", q):
                        return False

        return True
    except Exception:
        # On parse failures, be conservative and allow
        return True


# --- Subdomain support helpers ---
def _registrable(host: str) -> str:
    """
    Simple fallback: last two labels (ok for most .com/.net). If you need
    bullet-proof eTLD+1 worldwide, we can switch to `tldextract`.
    """
    host = (host or "").lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host

def _same_site(a: str, b: str, allow_subdomains: bool) -> bool:
    if not allow_subdomains:
        return is_same_domain(a, b)
    ha = _registrable(urlparse(a).hostname or "")
    hb = _registrable(urlparse(b).hostname or "")
    return ha == hb


class SiteCrawler:
    """
    Concurrent, retrying, per-domain crawler:
    - Starts from a homepage
    - Follows internal links (same host by default; optionally subdomains)
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

        allow_re = re.compile(url_allow_regex) if url_allow_regex else None
        deny_re = re.compile(url_deny_regex) if url_deny_regex else None

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
                        snap = await self._fetch_and_extract(url)

                    # optional polite delay to reduce bans
                    if self.cfg.per_page_delay_ms > 0:
                        await asyncio.sleep(self.cfg.per_page_delay_ms / 1000.0)

                    snapshots[url] = snap

                    # Enqueue new links
                    for link in snap.out_links:
                        if max_pages is not None and len(visited) >= max_pages:
                            break
                        if link in visited or link in queued:
                            continue
                        # site scope
                        if not _same_site(url, link, getattr(self.cfg, "allow_subdomains", False)):
                            continue
                        # language policy
                        if not _language_ok(link, self.cfg):
                            continue
                        # user allow/deny
                        if allow_re and not allow_re.search(link):
                            continue
                        if deny_re and deny_re.search(link):
                            continue
                        # file/junk filters
                        if not _should_visit(link):
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
        resp = None  # ensure defined for logging/guards
        try:
            # Try a small cascade of wait strategies (stop on first success)
            last_err = None
            for wu in [self.wait_until, "domcontentloaded", "load"]:
                try:
                    resp = await page.goto(url, wait_until=wu, timeout=self.cfg.page_load_timeout_ms)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    continue
            if last_err is not None and resp is None:
                raise TransientHTTPError(f"Navigation failed for {url}: {last_err}")

            # Status guard (if we received a response)
            status = getattr(resp, "status", None)
            if status is not None:
                s = int(status)
                if s == 404:
                    # permanent — do NOT retry
                    raise NonRetryableHTTPError(f"HTTP 404 for {url}")
                if s >= 400:
                    # transient — let retry_async/worker retry
                    raise TransientHTTPError(f"HTTP {s} for {url}")

            # Ensure content is present (best-effort)
            try:
                await page.wait_for_selector("body", timeout=3000)
            except Exception:
                pass

            title = await _safe_title(page)
            html = await page.content()

            # Clamp giant HTML to 3MB to avoid downstream issues
            if html and len(html) > 3_000_000:
                html = html[:3_000_000]

            html_path = None
            if self.cfg.cache_html:
                html_path = self._save_html(url, html)

            links = await _extract_links(page)
            # Absolute, normalized, HTTP(S) only
            links = [normalize_url(l) for l in links if is_http_url(l)]

            # Same-site + language filter early (reduces out_links payload)
            filtered = []
            for l in links:
                if not _same_site(url, l, getattr(self.cfg, "allow_subdomains", False)):
                    continue
                if not _language_ok(l, self.cfg):
                    continue
                if not _should_visit(l):
                    continue
                filtered.append(l)

            return PageSnapshot(url=url, title=title, html_path=html_path, out_links=filtered)

        except NonRetryableHTTPError:
            raise  # bubble up as non-retryable
        except Exception as e:
            # include status if available for better logs
            st = getattr(resp, "status", None)
            if st is not None:
                logger.debug("Error on %s (status=%s): %s", url, st, e)
            # treat the rest as transient so outer retry can apply
            raise TransientHTTPError(str(e))
        finally:
            try:
                await page.close()
            except Exception:
                pass

    def _save_html(self, url: str, html: str) -> str:
        parsed = urlparse(url)
        host = (parsed.hostname or "unknown-host").lower()
        if host.startswith("www."):
            host = host[4:]  # drop leading www.

        base = parsed.path or "/"
        if base.endswith("/"):
            base += "index"
        slug = slugify(base, max_len=80)

        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        fname = f"{slug}-{h}.html"

        out_dir = self.cfg.scraped_html_dir / host
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        atomic_write_text(out_path, html)
        return str(out_path)


async def _safe_title(page) -> Optional[str]:
    try:
        return await page.title()
    except Exception:
        return None


async def _extract_links(page) -> List[str]:
    """
    Extract links; resolve relative hrefs against the current page URL.
    We intentionally use getAttribute('href') and resolve with urljoin to
    ensure relative links like '/surecare-service' are included.
    """
    try:
        raw_hrefs: List[str] = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.getAttribute('href'))",
        )
        base = page.url  # property access (not awaitable)
        cleaned: List[str] = []
        seen = set()
        for h in raw_hrefs:
            if not h:
                continue
            abs_u = urljoin(base, h)
            # strip fragment
            abs_u = re.sub(r"#.*$", "", abs_u)
            if abs_u not in seen:
                seen.add(abs_u)
                cleaned.append(abs_u)
        return cleaned
    except Exception as e:
        logger.warning("Failed extracting links: %s", e)
        return []