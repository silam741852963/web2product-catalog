from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Set
from urllib.parse import urlparse, urlunparse

log = logging.getLogger(__name__)

# NOTE: We avoid passing version-specific kwargs; use the simplest, portable call.
from usp.tree import sitemap_tree_for_homepage


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


def _collect_with_usp(homepage: str) -> List[str]:
    """
    Portable call that works across usp versions. No custom client, no extra kwargs.
    """
    tree = sitemap_tree_for_homepage(homepage)  # positional-only for broad compatibility
    urls: Set[str] = set()
    for page in tree.all_pages():
        try:
            urls.add(_to_https(page.url))
        except Exception:
            # be permissive: skip any odd page objects
            pass
    return sorted(urls)


async def discover_from_sitemaps(
    base_root: str,
    *,
    timeout_s: float = 30.0,  # kept for interface symmetry; actual timeout is library-managed
) -> List[str]:
    """
    Discover page URLs via sitemaps (HTTPS-only), with simple, version-agnostic USP usage.
    """
    root = _ensure_https_root(base_root)
    log.info("[sitemap_hunter] build tree: %s", root)
    t0 = time.perf_counter()

    try:
        urls = await asyncio.to_thread(_collect_with_usp, root)
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