from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Iterable, List, Set
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)

# --- HTTP client settings ---
USER_AGENT = "FastSitemapFetcher/0.1"
TIMEOUT_S = 8.0

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


# ----------------------------
# URL helpers (unchanged)
# ----------------------------
def _normalize_host(h: str) -> str:
    h = (h or "").lower().strip().rstrip(".")
    return h[4:] if h.startswith("www.") else h


def _same_site(u: str, root: str) -> bool:
    cu = urlparse(u)
    ru = urlparse(root)
    if not cu.netloc or not ru.netloc:
        return False
    return _normalize_host(cu.netloc) == _normalize_host(ru.netloc)


def _ensure_https_root(url_or_host: str) -> str:
    s = (url_or_host or "").strip()
    if not s:
        return "https://"
    p = urlparse(s if "://" in s else f"https://{s}")
    scheme = "https"
    netloc = p.netloc or p.path
    return urlunparse((scheme, netloc, "/", "", "", ""))


def _to_https(u: str) -> str:
    if not u:
        return u
    p = urlparse(u)
    if p.scheme == "http":
        return urlunparse(("https", p.netloc, p.path, p.params, p.query, p.fragment))
    return u


# ----------------------------
# Sitemap fetching/parsing
# ----------------------------
def _fetch(url: str) -> bytes | None:
    try:
        resp = _SESSION.get(url, timeout=TIMEOUT_S)
        if resp.status_code == 200:
            return resp.content
    except requests.RequestException:
        return None
    return None


def _get_robots_sitemaps(homepage: str) -> List[str]:
    """
    Fetch robots.txt and extract all Sitemap: URLs.
    """
    parsed = urlparse(homepage)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    data = _fetch(robots_url)
    if not data:
        return []

    sitemaps: List[str] = []
    for line in data.decode(errors="ignore").splitlines():
        line = line.strip()
        if not line.lower().startswith("sitemap:"):
            continue
        # Example: "Sitemap: https://example.com/sitemap.xml"
        _, url = line.split(":", 1)
        sitemap_url = url.strip()
        if sitemap_url:
            sitemaps.append(sitemap_url)
    return sitemaps


def _guess_sitemap_urls(homepage: str) -> List[str]:
    """
    If robots.txt gives nothing, try some common sitemap locations.
    """
    parsed = urlparse(homepage)
    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemap-index.xml",
        "/sitemap/sitemap.xml",
    ]
    found: List[str] = []
    for path in candidates:
        url = urljoin(base, path)
        data = _fetch(url)
        if data:
            found.append(url)
    return found


def _is_sitemap_index(root: ET.Element) -> bool:
    # <sitemapindex xmlns="...">
    return root.tag.endswith("sitemapindex")


def _iter_sitemap_locs(xml_bytes: bytes) -> Iterable[str]:
    """
    Stream through an XML sitemap or sitemap index and yield all <loc> values.
    Uses BytesIO so ET.iterparse sees a file-like object.
    """
    buffer = io.BytesIO(xml_bytes)
    context = ET.iterparse(buffer, events=("end",))
    for event, elem in context:
        if elem.tag.endswith("loc") and elem.text:
            yield elem.text.strip()
        # free memory
        elem.clear()


def _collect_all_urls_from_sitemaps(sitemap_urls: List[str]) -> Set[str]:
    """
    Given one or more sitemap or sitemap index URLs,
    follow sitemap indexes and collect all page URLs.
    """
    urls: Set[str] = set()
    to_visit: List[str] = list(sitemap_urls)
    visited_sitemaps: Set[str] = set()

    while to_visit:
        sm_url = to_visit.pop()
        if sm_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sm_url)

        data = _fetch(sm_url)
        if not data:
            continue

        # Peek at root tag to know if it's index or plain sitemap
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            # malformed or weird encoding; skip
            continue

        if _is_sitemap_index(root):
            # sitemap index: its <loc> entries are sub-sitemaps
            for loc in root.iter():
                if loc.tag.endswith("loc") and loc.text:
                    to_visit.append(loc.text.strip())
        else:
            # normal sitemap: <loc> entries are page URLs
            for loc_url in _iter_sitemap_locs(data):
                urls.add(loc_url)

    return urls


def _collect_with_own_sitemap_logic(homepage: str) -> List[str]:
    """
    Replacement for _collect_with_usp:
    - discover sitemap URLs via robots.txt or common patterns
    - parse them and return a sorted list of page URLs
    """
    # Discover sitemap URLs
    sm_urls = _get_robots_sitemaps(homepage)
    if not sm_urls:
        sm_urls = _guess_sitemap_urls(homepage)
    if not sm_urls:
        return []

    all_urls = _collect_all_urls_from_sitemaps(sm_urls)
    return sorted(all_urls)


# ----------------------------
# Public async API
# ----------------------------
async def discover_from_sitemaps(
    base_root: str,
    *,
    timeout_s: float = 30.0,  # kept for interface symmetry; not directly used
) -> List[str]:
    """
    Discover page URLs via sitemaps (HTTPS-only) using our own logic:
      - robots.txt Sitemap: lines
      - common sitemap paths if robots.txt is empty
      - streaming XML parsing for performance

    Semantics kept similar to the original USP-based function:
      - normalize to HTTPS root
      - same-site filtering
      - HTTPS dedupe
    """
    root = _ensure_https_root(base_root)
    log.info("[sitemap_hunter] discover via custom sitemaps: %s", root)
    t0 = time.perf_counter()

    try:
        # Run blocking HTTP/XML work in a thread
        urls = await asyncio.to_thread(_collect_with_own_sitemap_logic, root)
    except Exception as e:
        log.info("[sitemap_hunter] error %s: %s", root, e)
        return []

    # same-site filter
    urls = [u for u in urls if _same_site(u, root)]

    dt = time.perf_counter() - t0
    log.info("[sitemap_hunter] done %s → %d urls (%.2fs)", root, len(urls), dt)

    # final https dedupe
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls:
        v = _to_https(u)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out