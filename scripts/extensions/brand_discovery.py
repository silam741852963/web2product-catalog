from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, urljoin

import httpx

from extensions.filtering import is_universal_external, should_block_external_for_dataset

logger = logging.getLogger("extensions.brand_discovery")

BRAND_MAX_ANCHORS_SCAN: int = 1000
BRAND_MAX_PER_HOST: int = 1
BRAND_LOG_ANCHORS: bool = False
BRAND_HTTP_TIMEOUT: float = 10.0

HTTPX_MAX_CONNECTIONS: int = 8
HTTPX_MAX_KEEPALIVE: int = 4

NOISY_SUBDOMAIN_PREFIXES: Tuple[str, ...] = (
    "news", "press", "media", "about", "careers",
    "jobs", "blog", "events", "support", "help", "status",
)

NEG_PATH_TOKENS: Tuple[str, ...] = (
    "/privacy", "/terms", "/legal", "/account", "/signin", "/login",
)


@dataclass
class Candidate:
    root: str
    score: float
    examples: List[str]
    anchors_count: int


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


_HREF_RE = re.compile(r'href=(?:"([^"]+)"|\'([^\']+)\')', flags=re.I)


async def discover_brand_sites(base_url: str, *, http_timeout: float = BRAND_HTTP_TIMEOUT) -> List[str]:
    """
    Lightweight brand discovery:
     - GET homepage (single GET)
     - Extract anchors via tiny regex
     - Resolve & filter external anchors
     - Bucket by root (scheme://host), cap per-host, and score
    """
    logger.info("[brand_discovery] start base_url=%s", base_url)

    anchors_list: List[str] = []
    final_url: str = base_url

    limits = httpx.Limits(max_connections=HTTPX_MAX_CONNECTIONS, max_keepalive_connections=HTTPX_MAX_KEEPALIVE)
    timeout = httpx.Timeout(http_timeout, connect=http_timeout)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, limits=limits) as client:
            try:
                r = await client.get(base_url)
            except Exception as e:
                logger.warning("[brand_discovery] GET failed base_url=%s err=%s", base_url, e)
                return []

            final_url = str(r.url or base_url)
            html = r.text or ""
            if not html:
                logger.info("[brand_discovery] empty html for %s", base_url)
                return []

            hrefs = _HREF_RE.findall(html)
            # flatten and cap once
            for a, b in hrefs:
                anchors_list.append(a or b)
                if len(anchors_list) >= BRAND_MAX_ANCHORS_SCAN:
                    break

    except Exception as e:
        logger.warning("[brand_discovery] httpx client error base_url=%s err=%s", base_url, e)
        return []

    base_host = _host(final_url) or _host(base_url)
    if BRAND_LOG_ANCHORS:
        for idx, h in enumerate(anchors_list):
            logger.debug("[brand_discovery] anchor[%04d] raw=%s", idx, h)

    if not anchors_list:
        logger.info("[brand_discovery] no anchors discovered on %s", base_url)
        return []

    seen: set[str] = set()
    kept_urls: List[str] = []
    drop_same_host = drop_neg = drop_social = drop_noisy = non_http = drop_dataset = 0

    for raw in anchors_list:
        abs_url = _resolve_href(final_url, raw)
        if not abs_url:
            non_http += 1
            logger.debug("[brand_discovery.skip] href=%s reason=non_http_or_fragment", raw)
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)

        if _same_host(abs_url, final_url):
            drop_same_host += 1
            logger.debug("[brand_discovery.skip] href=%s reason=same_host", abs_url)
            continue

        if should_block_external_for_dataset(final_url, abs_url):
            drop_dataset += 1
            logger.debug("[brand_discovery.skip] href=%s reason=dataset_external_block", abs_url)
            continue

        neg_hit, neg_tok = _has_neg_path_token(abs_url)
        if neg_hit:
            drop_neg += 1
            logger.debug("[brand_discovery.skip] href=%s reason=neg_path token=%s", abs_url, neg_tok)
            continue

        if is_universal_external(abs_url):
            drop_social += 1
            logger.debug("[brand_discovery.skip] href=%s reason=universal_external", abs_url)
            continue

        noisy, sub = _has_noisy_subdomain(abs_url)
        if noisy:
            drop_noisy += 1
            logger.debug("[brand_discovery.skip] href=%s reason=noisy_subdomain sub=%s", abs_url, sub)
            continue

        kept_urls.append(abs_url)

    logger.debug(
        "[brand_discovery] keep/drop summary: total=%d, non_http=%d, drop_same_host=%d, drop_neg=%d, drop_social=%d, drop_noisy_subdomain=%d, drop_dataset=%d, kept=%d",
        len(anchors_list), non_http, drop_same_host, drop_neg, drop_social, drop_noisy, drop_dataset, len(kept_urls),
    )

    if not kept_urls:
        logger.info("[brand_discovery] discovered 0 candidate brand roots on %s", base_url)
        return []

    buckets: Dict[str, List[str]] = {}
    for u in kept_urls:
        root = _root(u)
        lst = buckets.setdefault(root, [])
        if len(lst) < BRAND_MAX_PER_HOST:
            lst.append(u)
        else:
            logger.debug(
                "[brand_discovery.skip] href=%s reason=per_host_cap_exceeded host=%s cap=%d",
                u, root, BRAND_MAX_PER_HOST
            )

    candidates: List[Candidate] = []
    for root, urls in buckets.items():
        anchors_count = len(urls)
        score = min(1.0, 0.2 + 0.1 * anchors_count)
        candidates.append(Candidate(root=root, score=score, examples=urls[:5], anchors_count=anchors_count))

    candidates.sort(key=lambda c: (c.score, c.anchors_count), reverse=True)

    for i, c in enumerate(candidates[:10]):
        logger.debug(
            "[brand_discovery] candidate[%02d] root=%s score=%.3f anchors=%d examples=%d",
            i, c.root, c.score, c.anchors_count, len(c.examples),
        )
        if BRAND_LOG_ANCHORS:
            for ex in c.examples:
                logger.debug("  └─ ex: %s", ex)

    results = sorted({c.root for c in candidates})
    logger.info("[brand_discovery] discovered %d candidate brand roots on %s", len(results), base_url)
    return results