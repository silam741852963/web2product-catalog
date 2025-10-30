# extensions/brand_discovery.py
from __future__ import annotations

import os
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup

from .filtering import is_universal_external

logger = logging.getLogger("extensions.brand_discovery")

# Whether to validate discovered brand ROOTS using lightweight HTTP HEAD calls.
# Default OFF to avoid lots of network chatter.
BRAND_HEAD_VALIDATE: bool = False

# Cap the number of root candidates we will validate with HEAD (if enabled).
BRAND_MAX_CANDIDATES: int = 20

# Cap how many distinct anchors per external HOST we keep before scoring.
BRAND_MAX_PER_HOST: int = 3

# Cap anchors processed from the homepage to avoid pathological pages.
BRAND_MAX_ANCHORS_SCAN: int = 1000

# Per-request timeout (seconds)
BRAND_HTTP_TIMEOUT: float = 12

# Optional debug: log EACH <a> found (can be noisy)
BRAND_LOG_ANCHORS: bool = False

# -----------------------------
# Heuristics
# -----------------------------
NOISY_SUBDOMAIN_PREFIXES: Tuple[str, ...] = (
    "news", "press", "media", "about", "careers", 
    "jobs", "blog", "events", "support", "help", "status",
)

# Negative path tokens that are not brand homes
NEG_PATH_TOKENS: Tuple[str, ...] = (
    "/privacy", "/terms", "/legal", "/account", "/signin", "/login",
)

@dataclass
class Candidate:
    root: str
    score: float
    examples: List[str]
    anchors_count: int


# -----------------------------
# Small helpers
# -----------------------------
def _host(u: str) -> str:
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:
        return ""

def _root(u: str) -> str:
    pu = urlparse(u)
    scheme = pu.scheme or "https"
    host = (pu.hostname or "").lower()
    return f"{scheme}://{host}"

def _same_host(a: str, b: str) -> bool:
    return _host(a) == _host(b)

def _has_noisy_subdomain(u: str) -> Tuple[bool, str]:
    host = _host(u)
    if not host:
        return False, ""
    labels = host.split(".")
    if len(labels) < 3:
        return False, ""
    sub = labels[0]
    if sub in NOISY_SUBDOMAIN_PREFIXES:
        return True, sub
    return False, ""

def _has_neg_path_token(u: str) -> Tuple[bool, str]:
    path = (urlparse(u).path or "").lower()
    for t in NEG_PATH_TOKENS:
        if t in path:
            return True, t
    return False, ""

def _resolve_href(base_url: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if not href or href.startswith("#") or href.startswith("javascript:"):
        return None
    if not re.match(r"^https?://", href, flags=re.I):
        href = urljoin(base_url, href)
    if not re.match(r"^https?://", href, flags=re.I):
        return None
    return href

async def _head_ok(client: httpx.AsyncClient, root: str, timeout: float) -> bool:
    """
    Lightweight validation of a root. Treat 2xx/3xx as valid; accept 405 (HEAD not allowed).
    """
    try:
        r = await client.head(root, timeout=timeout, follow_redirects=True)
        ok = (r.status_code < 400) or (r.status_code == 405)
        logger.debug("[brand_discovery] HEAD %s -> %s (%s)", root, "OK" if ok else "BAD", r.status_code)
        return ok
    except Exception as e:
        logger.debug("[brand_discovery] HEAD %s -> EXC %s", root, repr(e))
        return False


# -----------------------------
# Main entry
# -----------------------------
async def discover_brand_sites(base_url: str, *, http_timeout: float = BRAND_HTTP_TIMEOUT) -> List[str]:
    """
    Fetch the company's homepage and extract EXTERNAL anchors that look like
    first-party brand sites owned by the company.

    Strongly filters:
      - Only external (different host) links
      - Drops universal externals / social hosts
      - Drops negative path tokens (/privacy, /terms, /login, ...)
      - Drops noisy subdomains on the TARGET (ir., investors., etc.)

    Optional HEAD validation for top candidates is gated by BRAND_HEAD_VALIDATE.
    Returns distinct brand ROOTS (scheme://host).
    """
    logger.info("[brand_discovery] start base_url=%s", base_url)

    # Conservative HTTP client shaping
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=8)
    async with httpx.AsyncClient(
        timeout=http_timeout,
        follow_redirects=True,
        http2=True,
        limits=limits,
        headers={"User-Agent": "brand-discovery/1.0 (+compatible)"}
    ) as client:
        # 1) Fetch homepage
        try:
            r = await client.get(base_url)
            final_url = str(r.url)
            html = r.text or ""
            base_host = _host(final_url) or _host(base_url)
            logger.debug(
                "[brand_discovery] fetched final_url=%s base_host=%s bytes=%d",
                final_url, base_host, len(html)
            )
        except Exception as e:
            logger.warning("[brand_discovery] fetch failed base_url=%s err=%s", base_url, e)
            return []

        # 2) Parse anchors (cap scan to avoid pathological pages)
        soup = BeautifulSoup(html, "html.parser")
        anchors_all = soup.find_all("a")
        anchors = anchors_all[:BRAND_MAX_ANCHORS_SCAN]
        if BRAND_LOG_ANCHORS:
            for idx, a in enumerate(anchors):
                href = (a.get("href") or "").strip()
                text = (a.get_text() or "").strip()
                logger.debug("[brand_discovery] anchor[%04d] href=%s text=%s", idx, href, text[:100])

        logger.debug("[brand_discovery] anchors_found=%d (scanned=%d)", len(anchors_all), len(anchors))

        # 3) First-pass filtering: keep EXTERNAL non-social/non-noisy/non-neg
        seen: set[str] = set()
        kept_urls: List[str] = []
        drop_same_host = drop_neg = drop_social = drop_noisy = non_http = 0

        for a in anchors:
            abs_url = _resolve_href(final_url, a.get("href") or "")
            if not abs_url:
                non_http += 1
                continue
            if abs_url in seen:
                continue
            seen.add(abs_url)

            if _same_host(abs_url, final_url):
                drop_same_host += 1
                continue

            neg_hit, _ = _has_neg_path_token(abs_url)
            if neg_hit:
                drop_neg += 1
                continue

            if is_universal_external(abs_url):
                drop_social += 1
                continue

            noisy, _ = _has_noisy_subdomain(abs_url)
            if noisy:
                drop_noisy += 1
                continue

            kept_urls.append(abs_url)

        logger.debug(
            "[brand_discovery] keep/drop summary: total=%d, non_http=%d, "
            "drop_same_host=%d, drop_neg=%d, drop_social=%d, drop_noisy_subdomain=%d, kept=%d",
            len(anchors), non_http, drop_same_host, drop_neg, drop_social, drop_noisy, len(kept_urls),
        )

        if not kept_urls:
            logger.info("[brand_discovery] discovered 0 candidate brand roots on %s", base_url)
            return []

        # 4) Bucket by ROOT, cap per host, and score
        buckets: Dict[str, List[str]] = {}
        for u in kept_urls:
            root = _root(u)
            lst = buckets.setdefault(root, [])
            if len(lst) < BRAND_MAX_PER_HOST:
                lst.append(u)

        candidates: List[Candidate] = []
        for root, urls in buckets.items():
            anchors_count = len(urls)
            # Simple monotone scoring; weight by the number of distinct anchors pointing to the same root
            score = min(1.0, 0.2 + 0.1 * anchors_count)
            candidates.append(Candidate(root=root, score=score, examples=urls[:5], anchors_count=anchors_count))

        candidates.sort(key=lambda c: (c.score, c.anchors_count), reverse=True)

        for i, c in enumerate(candidates[:10]):  # show a few for debug
            logger.debug(
                "[brand_discovery] candidate[%02d] root=%s score=%.3f anchors=%d examples=%d",
                i, c.root, c.score, c.anchors_count, len(c.examples)
            )
            for ex in c.examples:
                logger.debug("  └─ ex: %s", ex)

        # 5) (Optional) HEAD validation of top candidates
        results: List[str] = []
        if BRAND_HEAD_VALIDATE:
            to_validate = candidates[:BRAND_MAX_CANDIDATES]
            if to_validate:
                sem = asyncio.Semaphore(10)  # bound concurrent HEADs

                async def _guarded_head(root: str) -> Tuple[str, bool]:
                    async with sem:
                        ok = await _head_ok(client, root, timeout=min(6.0, http_timeout))
                        return root, ok

                validations = await asyncio.gather(*[_guarded_head(c.root) for c in to_validate], return_exceptions=False)
                for root, ok in validations:
                    if ok:
                        results.append(root)
        else:
            # If not validating, just return all candidate roots
            results = [c.root for c in candidates]

        # Deduplicate & finalize
        results = sorted(set(results))
        logger.info("[brand_discovery] discovered %d candidate brand roots on %s", len(results), base_url)
        return results