from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import string
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterable, Callable, Awaitable, Optional, Any, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urljoin
from email.utils import formatdate, parsedate_to_datetime

import httpx
import tldextract
from bs4 import BeautifulSoup, Comment
from bs4.element import Tag
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from collections import Counter

import asyncio
from contextlib import suppress

try:
    # Only if Playwright is available in this env
    from playwright._impl._errors import Error as PlaywrightError   # type: ignore
except Exception:
    PlaywrightError = Exception  # fallback typing



try:
    from markdownify import markdownify as _markdownify
except Exception as e:  # pragma: no cover
    _markdownify = None
    logging.getLogger(__name__).warning(
        "markdownify import failed (%s). Falling back to plain-text stripping. "
        "Ensure 'markdownify' and 'beautifulsoup4' are installed in this interpreter.",
        repr(e),
    )

logger = logging.getLogger(__name__)

# ========== Environment & Logging helpers (moved from config.py) ==========

def getenv_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() else default

def getenv_int(name: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if min_val is not None:
        val = max(min_val, val)
    if max_val is not None:
        val = min(max_val, val)
    return val

def getenv_float(name: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    if min_val is not None:
        val = max(min_val, val)
    if max_val is not None:
        val = min(max_val, val)
    return val

def getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# parse CSV-ish envs into tuples (trim blanks)
def getenv_csv(name: str, default_csv: str) -> Tuple[str, ...]:
    raw = getenv_str(name, default_csv)
    parts = [x.strip() for x in raw.split(",")]
    return tuple(p for p in parts if p)

def init_logging(log_path: Path, level: int = logging.INFO) -> None:
    """
    Simple file+console logger. Call once early (e.g., in run_scraper.py) with cfg.log_file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(levelname)s %(asctime)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

# ========== Exceptions & HTTP status mapping ==========

class TransientHTTPError(Exception):
    """Retryable transient HTTP/Net error (429/5xx/timeouts)."""

class NonRetryableHTTPError(Exception):
    """Non-retryable client error (e.g., 404) or policy block."""

class NeedsBrowser(Exception):
    """Signals that static fetch is insufficient and a browser is required."""

def http_status_to_exc(status: Optional[int]) -> Optional[Exception]:
    if status is None:
        return None
    if status == 404:
        return NonRetryableHTTPError("404 Not Found")
    if status >= 400:
        # Treat all 4xx as transient for now except 404 above. Many 403/429/401 are temporary/gate-keepers.
        return TransientHTTPError(f"HTTP {status}")
    return None

# ========== URL & domain helpers ==========

# ===== Super-aggressive file extensions to drop (treat as files, not pages) =====
_DENY_FILE_EXTS = (
    # docs / data
    ".pdf", ".ps", ".rtf", ".doc", ".docx", ".ppt", ".pptx", ".pps", ".xls", ".xlsx",
    ".csv", ".tsv", ".xml", ".json", ".yml", ".yaml", ".md", ".rst",
    # archives / binaries / installers
    ".zip", ".rar", ".7z", ".gz", ".bz2", ".xz", ".tar", ".tgz", ".dmg", ".exe", ".msi", ".apk",
    # images / icons
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".tif", ".tiff", ".ico",
    # audio
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac",
    # video
    ".mp4", ".m4v", ".webm", ".mov", ".avi", ".wmv", ".mkv",
    # misc
    ".ics", ".eot", ".ttf", ".otf", ".woff", ".woff2", ".map",
)

# ===== Endpoints that are almost never useful for product scraping =====
_BACKEND_ENDPOINT_HINTS = (
    # APIs / services
    "/api/", "/rest/", "/graphql", "/oembed", "/wp-json", "/feeds", "/feed",
    # admin/cms
    "/wp-admin", "/xmlrpc.php", "/umbraco", "/cms/", "/admin/", "/joomla", "/drupal",
    # machine listings
    "/sitemap", "/sitemap.xml", "/sitemaps", "/robots.txt", "/rss", "/atom",
    # classic server tech / handlers
    ".jsp", ".do", ".action", ".ashx", ".aspx", ".cfm", ".cgi",
    # infra/system paths
    "/cdn-cgi/",
)

# ===== Query keys that should NOT affect canonical equality (collapse variants) =====
_IGNORED_QUERY_KEYS_FOR_DEDUPE = {
    # tracking / campaign
    "gclid", "fbclid", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "mc_cid", "mc_eid", "ref", "trk", "cmpgn", "campaign", "source", "medium", "cid", "bid",
    # session-ish
    "session", "sessionid", "sid", "token", "auth", "hash",
    # sorting / filtering / searches
    "sort", "orderby", "order", "dir", "view", "layout", "display", "grid", "list",
    "filter", "filters", "facet", "facets", "category", "cat", "tag", "tags",
    "q", "query", "search", "s", "k", "keyword",
    "page", "paged", "pagenum", "start", "offset", "limit", "per_page", "size",
    # i18n toggles
    "lang", "hl", "locale", "region", "lc", "lr",
    # wordpress / cms cruft
    "page_id", "p", "post_type",
    # forums/boards noise
    "postid", "reply", "thread", "topic",
}

_JS_APP_SIGNATURES = (
    "__NEXT_DATA__", 'id="__next"', "data-reactroot", "reactdom",
    "window.__NUXT__", 'id="app"', "ng-app", "sapper", "astro-island",
    "vite", "webpackjsonp",
)

# Strong allow-hints: if the path has these, we DO NOT block
_PRODUCT_HINTS = (
    "product", "products", "solution", "solutions",
    "service", "services", "catalog", "portfolio",
    "platform", "features", "pricing", "specs", "datasheet", "datasheets",
    "store", "shop",
    # B2B/industrial nouns commonly used as product detail
    "grade", "grades", "sds", "msds", "safety-data-sheet", "safety-data-sheets",
)

# Obvious non-product path fragments
_NON_PRODUCT_PATH_PARTS = (
    # legal / compliance
    "privacy", "cookies", "cookie-policy", "cookie-preferences", "cookie-settings",
    "terms", "legal", "disclaimer", "compliance", "accessibility",
    "policy", "policies", "charter", "bylaws", "guidelines", "ethics",
    "anti-bribery", "sanctions", "human-rights", "code-of-conduct",
    "governance", "board", "leadership", "management", "members",
    "leader", "leaders", "leadership-team", "our-team", "team",

    # investor relations / finance
    "investor", "investors", "ir", "sec", "earnings", "transcript",
    "prospectus", "summaryprospectus", "factcard", "sai", "holdings",
    "annual-report", "semi-annual", "shareholder", "premiumdiscount", "market",
    "markets",

    # comms / newsroom
    "newsroom", "press", "press-release", "press-releases", "media", "news",

    # forms / lead-gen
    "information-request", "request-information", "request-info", "request",
    "contact", "contact-us", "form", "forms", "subscribe", "newsletter",

    # content marketing (usually non-product)
    "whitepaper", "whitepapers", "case-study", "case-studies", "ebook", "e-book",
    "resources", "resource-center", "brochure", "blog", "blogs", "stories",
    "giveaway", "community", "explained"

    # auth / account / ecommerce
    "login", "log-in", "signin", "sign-in", "signup", "sign-up", "register",
    "account", "my-account", "profile", "auth", "sso", "oauth",
    "cart", "checkout", "wishlist", "compare", "enrollment"

    # site chrome / discovery
    "sitemap", "search", "search-results", "results", "site-search",
    "preferences", "settings", "help", "support", "faq",

    # wp / feeds / apis
    "wp-admin", "wp-login", "wp-json", "xmlrpc.php", "feed", "rss", "atom",
    "wp-content/uploads", "media-library", "document-library", "asset-library",

    # infra/system
    "cdn-cgi",

    # i18n helpers
    "translate", "lang", "locale",

    # careers
    "careers", "jobs", "recruit", "hiring",

    # misc known time sinks
    "acquire/tel",
    # video & social slugs that often 302 off-site or thin content
    "video", "videos", "eb-video", "eb-video-en", "media-center",
    # event/webinar marketing
    "event", "events", "webinar", "webinars",
    # category/tag archives (WP/Drupal etc.)
    "category", "tag", "tags", "archive", "archives", "author",
    # training portals that rarely have product detail
    "training", "education", "ausbildung",
    # store/branch locators
    "locations",
)

_NON_PRODUCT_QUERY_KEYS = (
    # tracking / sessions
    "gclid", "fbclid", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "session", "sessionid", "token", "auth", "sso", "oauth", "redirect", "ref", "trk", "_gl"
    # downloads / attachments
    "download", "attachment", "attachment_id",
    # wordpress & CMS cruft
    "page_id", "p", "post_type", "orderby", "order", "cat", "tag",
    # i18n toggles
    "lang", "hl", "locale", "lc", "lr", "region",
    # forums/boards noise
    "postid", "reply", "thread", "topic",
)

_NON_PRODUCT_HOST_TOKENS = (
    "investor", "investors", "ir", "investorrelations",
    "press", "news", "events",
    "careers", "jobs",
    "support", "help",
    "auth", "login", "accounts",
    "blog", "community",
    "stories", "makers",
)

_TEL_PATH_RE = re.compile(r"(?:^|/)(?:tel|fax)\d{6,}(?:/|$)", re.IGNORECASE)

# Common ISO language tags (2-letter + popular regionized forms)
_LANG_CODES = {
    # European
    "fr", "fr-fr", "fr-ca", "de", "de-de", "es", "es-es", "it", "pt", "pt-pt", "pt-br",
    "nl", "sv", "no", "da", "fi", "pl", "cs", "sk", "sl", "hu", "el", "ro", "bg", "hr", "sr", "uk", "et", "lv", "lt",
    # Asian
    "zh", "zh-cn", "zh-tw", "zh-hk", "ja", "jp", "ko", "kr", "th", "vi", "id", "ms", "hi", "bn", "ta", "te", "ur", "fa",
    # Middle East
    "ar", "he", "tr", "ir",
    # Misc
    "ru",
}

# compile a path prefix regex like ^/(fr|fr-fr|de|zh-cn)(/|$)
_LANG_PATH_PREFIX_RE = re.compile(
    r"^/(?P<tag>[a-z]{2}(?:-[a-z]{2})?)(?:/|$)", re.IGNORECASE
)

# Boundary-aware core product tokens (safe against "production"/"productive").
# Matches hyphen/underscore/slash/dot/query boundaries.
_PRODUCT_CORE_RE = re.compile(
    r"(?i)"                                # case-insensitive
    r"(?:(?<=^)|(?<=[/._?#-]))"            # left boundary: start or / . _ ? # -
    r"(product|products|solution|solutions|service|services|catalog|catalogue|"
    r"store|shop|pricing|plans|features|specs|datasheet|data-sheet|price|buy|quote)"
    r"(?=$|[/._?#-])"                      # right boundary: end or / . _ ? # -
)

_COOKIE_QUERY_KEYS = {"optanonconsent", "consent", "gdpr", "euconsent", "cookie"}
_COOKIE_PATH_HINTS = ("cookie", "cookies", "cookie-policy", "consent", "privacy-center", "preferences")


# ========== Throttling / Backoff (429-aware) ==========

class TokenBucket:
    """
    Simple async token bucket for rate limiting.
    capacity: max tokens
    refill_rate: tokens per second
    """
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = max(0.1, capacity)
        self.refill_rate = max(0.1, refill_rate)
        self._tokens = self.capacity
        self._last = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0):
        async with self._lock:
            await self._drain_until(tokens)

    async def _drain_until(self, tokens: float):
        while True:
            now = time.perf_counter()
            elapsed = max(0.0, now - self._last)
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            # Not enough tokens—sleep until next refill tick
            deficit = tokens - self._tokens
            sleep_s = max(0.01, deficit / self.refill_rate)
            await asyncio.sleep(sleep_s)


class GlobalThrottle:
    """
    Global + per-host throttling with 'penalize' on 429 using Retry-After when available.
    - Defaults can be overridden by env:
        SCRAPER_GLOBAL_RPS (default 12.0)
        SCRAPER_PER_HOST_RPS (default 2.0)
        SCRAPER_RETRY_AFTER_MAX_S (cap for Retry-After, default 120)
    """
    def __init__(self):
        self.global_rps = getenv_float("SCRAPER_GLOBAL_RPS", 12.0, 0.5)
        self.per_host_rps = getenv_float("SCRAPER_PER_HOST_RPS", 2.0, 0.25)
        self.retry_after_cap_s = getenv_float("SCRAPER_RETRY_AFTER_MAX_S", 120.0, 1.0)

        self._global_bucket = TokenBucket(capacity=max(1.0, self.global_rps), refill_rate=self.global_rps)
        self._host_buckets: dict[str, TokenBucket] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _bucket_for_host(self, host: str) -> TokenBucket:
        host = host or "unknown-host"
        b = self._host_buckets.get(host)
        if b is None:
            # capacity ~= 2 seconds worth, so small bursts are ok
            cap = max(1.0, self.per_host_rps * 2.0)
            b = self._host_buckets[host] = TokenBucket(capacity=cap, refill_rate=self.per_host_rps)
        return b

    async def wait_for_host(self, host: str):
        host = host or "unknown-host"
        # If host is temporarily blocked due to previous 429s, wait it out
        async with self._lock:
            blocked_until = self._blocked_until.get(host, 0.0)
        now = time.time()
        if blocked_until > now:
            delay = max(0.0, blocked_until - now)
            if delay > 0:
                await asyncio.sleep(delay)

        # Global then per-host
        await self._global_bucket.acquire(1.0)
        await self._bucket_for_host(host).acquire(1.0)

    async def penalize(self, host: str, delay_s: float, *, reason: str = "429") -> None:
        """
        Block host until now + delay_s (capped), and also softly drop its rps for a while by
        starving the bucket (natural refill will recover).
        """
        if not delay_s or delay_s < 0:
            return
        delay_s = float(min(self.retry_after_cap_s, max(0.0, delay_s)))
        until = time.time() + delay_s
        async with self._lock:
            prev = self._blocked_until.get(host, 0.0)
            self._blocked_until[host] = max(prev, until)
        logger.info("Throttle: penalized host=%s for %.1fs (%s)", host, delay_s, reason)


GLOBAL_THROTTLE = GlobalThrottle()


def parse_retry_after_header(headers: dict[str, str] | Any) -> Optional[float]:
    """
    Parse Retry-After header. Supports:
      - integer seconds
      - HTTP-date
    Returns seconds (float) or None.
    """
    if not headers:
        return None
    try:
        # normalize lookup
        ra = None
        for k, v in headers.items():
            if k.lower() == "retry-after":
                ra = v
                break
        if not ra:
            return None
        ra = ra.strip()
        if not ra:
            return None
        # numeric seconds?
        if ra.isdigit():
            return float(int(ra))
        # HTTP-date
        dt = parsedate_to_datetime(ra)
        if not dt:
            return None
        delta = (dt.timestamp() - time.time())
        return float(max(0.0, delta))
    except Exception:
        return None


# ========== Domain helpers ==========

def get_base_domain(host: str) -> str:
    """
    Return registrable domain (eTLD+1); fall back to host if unknown.
    """
    if not host:
        return "unknown-host"
    host = host.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    try:
        ext = tldextract.extract(host)
        td = getattr(ext, "top_domain_under_public_suffix", None)
        if td:
            return td
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
    except Exception:
        pass
    return host

def normalize_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)

    if not parsed.scheme and not parsed.netloc and parsed.path:
        m = re.match(r"^(?P<host>[A-Za-z0-9.-]+)(?::(?P<port>\d+))?$", parsed.path)
        if m and "." in m.group("host"):
            host = m.group("host").lower()
            port = m.group("port")
            netloc = f"{host}:{port}" if port else host
            return urlunparse(("http", netloc, "/", "", "", ""))

    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname.lower() if parsed.hostname else ""
    netloc = host
    if parsed.port and not ((scheme == "http" and parsed.port == 80) or (scheme == "https" and parsed.port == 443)):
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"
    query = parsed.query
    return urlunparse((scheme, netloc, path, "", query, ""))

def is_same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.hostname or "").lower() == (pb.hostname or "").lower()

def is_http_url(url: str) -> bool:
    s = urlparse(url).scheme.lower()
    return s in {"http", "https"}


# ========== Same-site policy (incl. alias hosts) ==========

def _collect_extra_hosts(cfg) -> set[str]:
    """
    Build the extra eTLD+1 host set that should be considered same-site.
    Reads cfg.extra_same_site_hosts (tuple or list), and also optionally
    the env var EXTRA_SAME_SITE_HOSTS (comma separated).
    """
    extra: set[str] = set()
    try:
        seq = getattr(cfg, "extra_same_site_hosts", ()) or ()
        for h in seq:
            h = (h or "").strip().lower()
            if h:
                extra.add(h)
    except Exception:
        pass

    try:
        raw = os.getenv("EXTRA_SAME_SITE_HOSTS", "")
        if raw:
            for h in raw.split(","):
                h = (h or "").strip().lower()
                if h:
                    extra.add(h)
    except Exception:
        pass
    return extra


def expand_same_site(
    base_url: str,
    target_url: str,
    allow_subdomains: bool,
    extra_hosts: Optional[Iterable[str]] = None,
) -> bool:
    """
    Same-site policy:
      - If allow_subdomains=True → eTLD+1 match is enough.
      - If allow_subdomains=False → exact hostname match only.
      - Aliases/probation: if target eTLD+1 is in extra_hosts → treat as same-site;
        when allow_subdomains=False, accept only the bare base (and www.) host, not arbitrary subdomains.
    """
    try:
        bp = urlparse(base_url)
        tp = urlparse(target_url)
        if not (bp.hostname and tp.hostname):
            return False

        b_host = bp.hostname.lower()
        t_host = tp.hostname.lower()
        b_base = get_base_domain(b_host)
        t_base = get_base_domain(t_host)

        # Primary host comparison
        if allow_subdomains:
            if b_base == t_base:
                return True
        else:
            if b_host == t_host:
                return True

        # Aliases/probation hosts (by eTLD+1)
        extras = set(h.lower().strip() for h in (extra_hosts or []) if h)
        if t_base in extras:
            if allow_subdomains:
                return True
            # strict: allow only the bare base (and www.) as same-site when subdomains are disabled
            if t_host == t_base or t_host == f"www.{t_base}":
                return True

        return False
    except Exception:
        return False

def same_site(
    base_url: str,
    target_url: str,
    allow_subdomains: bool,
    extra_hosts: Optional[Iterable[str]] = None,
) -> bool:
    return expand_same_site(base_url, target_url, allow_subdomains, extra_hosts=extra_hosts)


# ========== Canonicalization & filtering helpers ==========

def _canonicalize_for_dedupe(u: str) -> str:
    """
    Normalize a URL for deduplication:
      - lower-case scheme/host
      - drop fragment
      - remove ignored query keys
      - sort remaining query keys
      - keep path as-is (so /a and /a/ differ)
    """
    pu = urlparse(u)
    scheme = (pu.scheme or "http").lower()
    host = (pu.hostname or "").lower()
    netloc = f"{host}:{pu.port}" if pu.port and not ((scheme == "http" and pu.port == 80) or (scheme == "https" and pu.port == 443)) else host
    path = pu.path or "/"

    # strip junk query keys; sort remaining for stable equality
    if pu.query:
        kept = [(k, v) for (k, v) in parse_qsl(pu.query, keep_blank_values=True) if k.lower() not in _IGNORED_QUERY_KEYS_FOR_DEDUPE]
        if kept:
            kept.sort(key=lambda kv: (kv[0].lower(), kv[1]))
            query = "&".join(f"{k}={v}" if v != "" else k for k, v in kept)
        else:
            query = ""
    else:
        query = ""

    return urlunparse((scheme, netloc, path, "", query, ""))

def _looks_backend_or_file(u: str) -> bool:
    pu = urlparse(u)
    path = (pu.path or "").lower()

    # media/docs: deny by extension
    for ext in _DENY_FILE_EXTS:
        if path.endswith(ext):
            return True

    # backend-y endpoints and feeds
    for hint in _BACKEND_ENDPOINT_HINTS:
        if hint in path:
            return True

    return False


# ========== Language & "productiness" heuristics ==========

_ENGLISH_COUNTRY_ALIASES = {
    # Common English-country ccTLDs we should accept as English when 'en' is primary
    "us", "gb", "uk", "ca", "au", "nz", "ie", "sg", "in", "za", "ph"
}

_LANG_TAG_RE = re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$", re.IGNORECASE)

def _normalize_primary_lang_tags(primary_lang: str | Iterable[str]) -> set[str]:
    """
    Normalize PRIMARY_LANG into a set of accepted tags.
    Accepts a string "en,us,en-gb" or an iterable of tags.
    If 'en' is present, include common English country aliases as accepted tags.
    All values are lowercased.
    """
    if isinstance(primary_lang, str):
        raw = [x.strip() for x in primary_lang.split(",") if x.strip()]
    else:
        raw = [str(x).strip() for x in (primary_lang or []) if str(x).strip()]

    tags = {t.lower() for t in raw if _LANG_TAG_RE.match(t.lower()) or len(t) == 2}
    if "en" in tags:
        tags |= _ENGLISH_COUNTRY_ALIASES  # treat these as English as well
        # also include a few common regioned forms
        tags |= {"en-us", "en-gb", "en-au", "en-ca", "en-nz", "en-ie", "en-sg", "en-in", "en-za", "en-ph"}
    return tags or {"en"}

def _accepts_primary(tag: str, accepted: set[str]) -> bool:
    """
    Does a detected lang/locale token count as primary?
    Accept if it equals any accepted tag, or startswith an accepted language prefix (e.g., 'en-us' startswith 'en').
    """
    t = tag.lower()
    if t in accepted:
        return True
    # If an accepted tag is a bare language like 'en', allow 'en-*' forms.
    for a in accepted:
        if len(a) == 2 and t.startswith(a + "-"):
            return True
    return False

from functools import lru_cache

@lru_cache(maxsize=64)
def _normalize_primary_lang_tags_cached(primary_lang_csv: str) -> set[str]:
    return _normalize_primary_lang_tags(primary_lang_csv)
def is_non_english_url(u: str, primary_lang: str = "en") -> bool:
    """
    Return True if the URL clearly targets a non-primary locale via subdomain, path prefix, or query values.
    PRIMARY_LANG can be a comma-separated list, e.g., "en,us" to allow 'us' as English.
    Examples:
      - PRIMARY_LANG="en" accepts en, en-US, en-GB, and common English ccTLDs like /us/ or subdomain us.example.com
      - PRIMARY_LANG="en,us" explicitly also accepts 'us'
    """
    try:
        accepted = _normalize_primary_lang_tags(primary_lang)

        pu = urlparse(u)
        host = (pu.hostname or "").lower()
        path = (pu.path or "/").lower()
        q = dict(parse_qsl(pu.query or "", keep_blank_values=True))

        # 1) subdomain like fr.example.com, de.example.com, zh-cn.example.com, us.example.com
        labels = host.split(".")
        if len(labels) >= 3:  # has a subdomain
            sub = labels[0].lower()
            # only consider tokens that look like lang/locale or 2-letter cc
            if sub != "www" and (_LANG_TAG_RE.match(sub) or len(sub) == 2):
                if not _accepts_primary(sub, accepted):
                    return True

        # 2) path prefix like /fr/, /de-de/, /zh-cn/, /us/
        m = _LANG_PATH_PREFIX_RE.match(path)
        if m:
            tag = m.group("tag").lower()
            if not _accepts_primary(tag, accepted):
                return True

        # 3) query keys lang/hl/locale/lc/lr/region set to a non-accepted value
        for k in ("lang", "hl", "locale", "lc", "lr", "region"):
            v = (q.get(k) or "").lower()
            if v and (_LANG_TAG_RE.match(v) or len(v) == 2):
                if not _accepts_primary(v, accepted):
                    return True

        return False
    except Exception:
        return False

def is_producty_url(u: str) -> bool:
    """
    Brand-agnostic heuristic for product/solution URLs:
      1) boundary-aware core tokens (handles 'cbd-products', 'investment-products', 'products-for-sale', etc.)
      2) plural last-segment heuristic for short category paths (/pet/dogs, /car/vehicles/, /banking/accounts)
    """
    if not u:
        return False

    p = urlparse(u)
    path = (p.path or "").lower()

    # 1) core tokens with safe boundaries (works for concatenations)
    if _PRODUCT_CORE_RE.search(path):
        return True

    # 2) simple plural/category leaf heuristic
    segs = [s for s in path.split("/") if s]
    if 1 < len(segs) <= 5:
        last = segs[-1]
        if len(last) >= 3 and last.endswith("s") and "-" not in last and not last.isdigit():
            return True

    return False


def looks_non_product_url(u: str) -> bool:
    """
    Returns True if the URL is clearly non-product.
    IMPORTANT: If the URL is 'producty' by structure (is_producty_url),
    we return False (i.e., do NOT classify as non-product), except for hard-stops
    like backend endpoints or obvious files—those are handled earlier in filtering.
    """
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        path = (p.path or "").lower()

        # home pages can be product portals
        if path in ("", "/"):
            return False

        # NEW: productiness overrides non-product tokens (except hard-stops handled elsewhere)
        if is_producty_url(u):
            return False

        # telephone-like slugs (rare company directory pages)
        if _TEL_PATH_RE.search(path or ""):
            return True

        # Host-level non-product tokens (press, careers, etc.)
        if any(tok in host.split(".") for tok in _NON_PRODUCT_HOST_TOKENS):
            return True

        # Path fragments that are commonly non-product (legal, press, etc.)
        for frag in _NON_PRODUCT_PATH_PARTS:
            if frag in path:
                return True

        # Query hints (attachments, wp cruft, forum noise, i18n toggles)
        if p.query:
            q = dict(parse_qsl(p.query, keep_blank_values=True))
            for k in _NON_PRODUCT_QUERY_KEYS:
                if k in q:
                    return True

        # Cookie/consent specific filters (path & query)
        if any(tok in path for tok in _COOKIE_PATH_HINTS):
            return True
        if p.query:
            q = dict(parse_qsl(p.query, keep_blank_values=True))
            if any(k.lower() in _COOKIE_QUERY_KEYS for k in q):
                return True

        return False
    except Exception:
        return False


# ========== Date helpers ==========

def httpdate_from_epoch(ts: float) -> str:
    """RFC 7231 IMF-fixdate."""
    try:
        return formatdate(ts, usegmt=True)
    except Exception:
        return formatdate(0, usegmt=True)

def epoch_from_httpdate(s: str) -> Optional[float]:
    try:
        return float(parsedate_to_datetime(s).timestamp())
    except Exception:
        return None


# ========== Retry decorators ==========

def retry_sync(max_attempts: int, initial_delay_ms: int, max_delay_ms: int, jitter_ms: int):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            initial=initial_delay_ms / 1000.0,
            max=max_delay_ms / 1000.0,
            jitter=jitter_ms / 1000.0,
        ),
        retry=retry_if_exception_type((TransientHTTPError, IOError, TimeoutError)),
    )

def retry_async(max_attempts: int, initial_delay_ms: int, max_delay_ms: int, jitter_ms: int):
    def _decorator(fn: Callable[..., Awaitable]):
        @retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(
                initial=initial_delay_ms / 1000.0,
                max=max_delay_ms / 1000.0,
                jitter=jitter_ms / 1000.0,
            ),
            retry=retry_if_exception_type((TransientHTTPError, IOError, TimeoutError)),
        )
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)
        return wrapper
    return _decorator


# ========== Static parsing / HTML utils ==========

def visible_text_len(html: str) -> int:
    if not html:
        return 0
    try:
        soup = BeautifulSoup(html, "lxml")
        return len((soup.get_text(" ", strip=True) or ""))
    except Exception:
        return 0

def looks_like_js_app(html: str, threshold: int) -> bool:
    if not html:
        return True
    lower = html.lower()
    # cheap substring checks first
    if "enable javascript" in lower or "requires javascript" in lower:
        return True
    if any(sig in lower for sig in _JS_APP_SIGNATURES):
        return visible_text_len(html) < threshold
    return False

def extract_links_static(html: str, base_url: str) -> list[str]:
    links: list[str] = []
    seen = set()
    try:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            if not href:
                continue
            abs_u = urljoin(base_url, href)
            abs_u = re.sub(r"#.*$", "", abs_u)
            if abs_u not in seen:
                seen.add(abs_u)
                links.append(abs_u)
        return links
    except Exception:
        return []

def extract_title_static(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, "lxml")
        t = soup.title.string if soup.title else None
        return (t or "").strip() or None
    except Exception:
        return None


# ========== HTML pruning (before Markdown) ==========

_CHROME_ID_CLASS = re.compile(
    r"(?:^|[-_])(header|footer|nav|navbar|menu|mega|breadcrumb|aside|sidebar|"
    r"social|share|subscribe|newsletter|signup|modal|popup|overlay|cookie|gdpr|consent|"
    r"banner|intercom|chat|widget|toolbar|sticky|floating|drawer|offcanvas|"
    r"comments|related|pagination|pager|sitemap|language|locale|switcher|"
    r"searchbox|search-bar|promo|advert|ad-)\b",
    re.I,
)
_COMPLIANCE_TEXT = re.compile(
    r"(cookie|gdpr|consent|we use cookies|privacy\s+policy|terms\s+of\s+use|"
    r"copyright|all rights reserved|trademark|forward-looking statements)",
    re.I,
)
_CTA_TEXT = re.compile(
    r"(request (a )?demo|contact sales|start (your )?trial|get started|"
    r"join (our )?newsletter|subscribe)", re.I
)

def _safe_classes(el) -> str:
    try:
        val = el.get("class")
        if isinstance(val, (list, tuple)):
            return " ".join(str(x) for x in val if x)
        if isinstance(val, str):
            return val
    except Exception:
        pass
    return ""

def _safe_id(el) -> str:
    try:
        v = el.get("id")
        return v if isinstance(v, str) else ""
    except Exception:
        return ""

def prune_html_for_markdown(html: str, *, keep_first_cta: bool = True) -> str:
    if not html:
        return html

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    try:
        for c in list(soup(string=lambda t: isinstance(t, Comment))):  # type: ignore
            try:
                c.extract()
            except Exception:
                pass
    except Exception:
        pass

    for tag in list(soup.find_all(["script", "style", "noscript", "svg", "canvas"])):  # noqa: B006
        if isinstance(tag, Tag):
            try:
                tag.decompose()
            except Exception:
                pass

    for tag in list(soup.select("[role='navigation'], [role='banner'], [role='contentinfo'], nav, aside, header, footer")):
        if isinstance(tag, Tag):
            try:
                tag.decompose()
            except Exception:
                pass

    for el in list(soup.find_all(attrs={"class": True})):
        if not isinstance(el, Tag):
            continue
        if _CHROME_ID_CLASS.search(_safe_classes(el)):
            try:
                el.decompose()
            except Exception:
                pass

    for el in list(soup.find_all(attrs={"id": True})):
        if not isinstance(el, Tag):
            continue
        if _CHROME_ID_CLASS.search(_safe_id(el)):
            try:
                el.decompose()
            except Exception:
                pass

    for el in list(soup.find_all(True)):
        if not isinstance(el, Tag):
            continue
        if el.name in ("main", "article", "section"):
            continue
        try:
            txt = (el.get_text(" ", strip=True) or "")[:500].lower()
        except Exception:
            continue
        if not txt:
            continue
        if _COMPLIANCE_TEXT.search(txt):
            try:
                el.decompose()
            except Exception:
                pass
            continue
        if _CTA_TEXT.search(txt):
            if keep_first_cta:
                keep_first_cta = False
            else:
                try:
                    el.decompose()
                except Exception:
                    pass

    for el in list(soup.find_all(True)):
        if not isinstance(el, Tag):
            continue
        if el.name in ("main", "article", "section"):
            continue
        try:
            txt = el.get_text("", strip=True)
        except Exception:
            txt = ""
        if not txt:
            try:
                el.decompose()
            except Exception:
                pass

    try:
        if not soup.find("main"):
            candidates = sorted(
                (el for el in list(soup.find_all(["article", "section", "div"])) if isinstance(el, Tag)),
                key=lambda e: len((e.get_text(" ", strip=True) or "")),
                reverse=True,
            )
            if candidates:
                keep = candidates[0]
                parent = keep.parent
                if parent:
                    for sib in list(getattr(parent, "children", []) or []):
                        if isinstance(sib, Tag) and sib is not keep:
                            try:
                                sib.decompose()
                            except Exception:
                                pass
    except Exception:
        pass

    for a in list(soup.find_all("a")):
        if not isinstance(a, Tag):
            continue
        try:
            href = (a.get("href") or "").strip()
        except Exception:
            href = ""
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            try:
                a.attrs.pop("href", None)
            except Exception:
                pass

    return str(soup)


# ========== HTML → Markdown ==========

@dataclass
class MarkdownOptions:
    strip: bool = True
    heading_style: str = "ATX"
    code_language: Optional[str] = None

def html_to_markdown(html: str, opts: MarkdownOptions = MarkdownOptions()) -> str:
    if not html:
        return ""
    if _markdownify is None:
        logger.warning("markdownify not installed; returning plain text fallback")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    md = _markdownify(html, heading_style=opts.heading_style.lower())
    if opts.strip:
        md = md.strip()
    return md

def clean_markdown(md: str, remove_boilerplate: bool = True) -> str:
    if not md:
        return md
    lines = [ln.rstrip() for ln in md.splitlines()]
    if remove_boilerplate:
        filtered = []
        for ln in lines:
            if len(ln) <= 2:
                continue
            if ln.lower().startswith(("cookie", "privacy policy", "terms of", "login", "subscribe")):
                continue
            filtered.append(ln)
        lines = filtered
    cleaned: list[str] = []
    blank = False
    for ln in lines:
        if ln.strip() == "":
            if not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(ln)
            blank = False
    return "\n".join(cleaned).strip()

INTERSTITIAL_PATTERNS = (
    "Just a moment", "Checking your browser", "cf-chl-", "Verifying you are human",
    "before proceeding", "5 seconds", "DDoS protection by"
)

def detect_interstitial(html: str) -> bool:
    low = (html or "").lower()
    return any(p.lower() in low for p in INTERSTITIAL_PATTERNS)

def is_thin_content(html: str, *, min_visible_chars: int = 1200) -> bool:
    return visible_text_len(html) < min_visible_chars

# ========== LLM chunking helpers ==========

def chunk_text(text: str, max_chars: int) -> Iterable[str]:
    if not text:
        return []
    paragraphs = text.split("\n\n")
    buf: list[str] = []
    length = 0
    for para in paragraphs:
        add_len = len(para) + (2 if buf else 0)
        if length + add_len <= max_chars:
            buf.append(para)
            length += add_len
        else:
            if buf:
                yield "\n\n".join(buf)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    yield para[i:i + max_chars]
                buf, length = [], 0
            else:
                buf, length = [para], len(para)
    if buf:
        yield "\n\n".join(buf)


# ========== File I/O ==========

def atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """
    Write text atomically using a NamedTemporaryFile and os.replace on the same filesystem.
    Safer than Path.write_text and reduces partial writes.
    """
    import os, tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def append_jsonl(path: Path, json_line: str, encoding: str = "utf-8") -> None:
    """
    Fast single-line append using O_APPEND. For bulk writes, prefer BufferedJsonlWriter.
    """
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    b = (json_line.rstrip() + "\n").encode(encoding, "utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, b)
    finally:
        os.close(fd)

class BufferedJsonlWriter:
    """Buffered JSONL writer for high-throughput appends.

    Usage:
        with BufferedJsonlWriter(Path("out.jsonl")) as w:
            w.write_obj({"a": 1})
            w.write_line('{"b":2}')
    """
    def __init__(self, path: Path, encoding: str = "utf-8", buffer_size: int = 1 << 16):
        import io
        self._path = path
        self._encoding = encoding
        self._buffer_size = buffer_size
        self._fd = None
        self._bio = None

    def __enter__(self):
        import io, os
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self._path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        raw = os.fdopen(self._fd, "ab", buffering=0)
        self._bio = io.BufferedWriter(raw, buffer_size=self._buffer_size)
        return self

    def __exit__(self, exc_type, exc, tb):
        import os
        try:
            if self._bio:
                self._bio.flush()
                os.fsync(self._fd)
        finally:
            try:
                if self._bio:
                    self._bio.close()
            finally:
                self._bio = None
                self._fd = None

    def write_line(self, line: str):
        self._bio.write((line.rstrip() + "\n").encode(self._encoding, "utf-8"))

    def write_obj(self, obj: dict):
        import json
        self.write_line(json.dumps(obj, ensure_ascii=False))

def save_markdown(base_dir: Path, host: str, url_path: str, url: str, md: str) -> Path:
    """
    Save markdown under the registrable domain folder (eTLD+1),
    collapsing subdomains (investors.example.com -> example.com).
    """
    base_host = get_base_domain(host or "unknown-host")
    path = url_path or "/"
    if path.endswith("/"):
        path += "index"
    slug = slugify(path, max_len=80)
    h = sha1(url.encode("utf-8")).hexdigest()[:10]
    fname = f"{slug}-{h}.md"
    out_path = base_dir / base_host / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, md)
    return out_path

def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_.]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-._")
    return text or "untitled"

def safe_json_loads(s: str) -> Optional[dict]:
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None

def markdown_quality(md: str) -> dict:
    if not isinstance(md, str):
        return {"chars": 0, "words": 0, "alpha": 0, "lines": 0, "uniq_words": 0}
    chars = len(md)
    words_list = [w.strip(string.punctuation) for w in md.split()]
    words = len(words_list)
    uniq_words = len(set(w.lower() for w in words_list if w))
    alpha = sum(ch.isalnum() for ch in md)
    lines = md.count("\n") + 1
    return {"chars": chars, "words": words, "alpha": alpha, "lines": lines, "uniq_words": uniq_words}

def is_meaningful_markdown(md: str,
                           *,
                           min_chars: int = 400,
                           min_words: int = 80,
                           min_uniq_words: int = 50,
                           min_lines: int = 8) -> bool:
    q = markdown_quality(md)
    if q["chars"] < min_chars:
        return False
    if q["words"] < min_words:
        return False
    if q["uniq_words"] < min_uniq_words:
        return False
    if q["lines"] < min_lines:
        return False
    return True


# ========== Robots gate ==========

def should_crawl_url(respect_robots: bool, robots_txt_allowed: Optional[bool]) -> bool:
    if not respect_robots:
        return True
    return bool(robots_txt_allowed)


# ========== HTTPX client (shared static fetch wiring) ==========

def httpx_client(cfg) -> httpx.AsyncClient:
    """
    Return a preconfigured AsyncClient honoring cfg static settings and cooperating with
    GLOBAL_THROTTLE. It will:
      - Wait on global + per-host token buckets before each request
      - Parse Retry-After on 429 responses and penalize the host accordingly
    """
    limits = httpx.Limits(max_keepalive_connections=16, max_connections=64)
    timeout = httpx.Timeout(cfg.static_timeout_ms / 1000.0)
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    async def _on_request(request: httpx.Request):
        try:
            host = (request.url.host or "unknown-host").lower()
            await GLOBAL_THROTTLE.wait_for_host(host)
        except Exception as e:
            logger.debug("throttle.request hook error: %s", e)

    async def _on_response(response: httpx.Response):
        try:
            status = response.status_code
            if status == 429:
                host = (response.request.url.host or "unknown-host").lower()
                delay = parse_retry_after_header(response.headers) or 20.0
                await GLOBAL_THROTTLE.penalize(host, delay, reason="httpx:429")
                logger.warning("HTTPX saw 429 for %s; backing off host=%s for %.1fs",
                               str(response.request.url), host, delay)
        except Exception as e:
            logger.debug("throttle.response hook error: %s", e)

    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        headers=headers,
        follow_redirects=True,
        http2=cfg.static_http2,
        max_redirects=cfg.static_max_redirects,
        event_hooks={
            "request": [_on_request],
            "response": [_on_response],
        },
    )


# ========== Playwright helpers (shared dynamic fetch wiring) ==========

async def play_goto(page, url: str, wait_chain: list[str], timeout_ms: int) -> tuple[Optional[int], str]:
    """
    Try a chain of wait_until strategies; return (status, html).

    Cooperative throttling:
      - Before navigating, wait on GLOBAL_THROTTLE for the target host.
      - A 429 encountered by the browser is handled by the browser response hook
        (see browser._install_429_response_hook), but we still return the status.
    """
    # Throttle before attempting navigation
    try:
        host = (urlparse(url).hostname or "unknown-host").lower()
        await GLOBAL_THROTTLE.wait_for_host(host)
    except Exception as e:
        logger.debug("throttle before goto failed: %s", e)

    last_status = None
    for wu in wait_chain:
        try:
            resp = await page.goto(url, wait_until=wu, timeout=timeout_ms)
            last_status = resp.status if resp else None
            break
        except Exception:
            continue
    # Ensure body exists, then pull html
    try:
        await page.wait_for_selector("body", timeout=timeout_ms)
    except Exception:
        pass
    try:
        html = await page.content()
    except Exception:
        html = ""
    return last_status, html

async def play_title(page) -> Optional[str]:
    try:
        t = await page.title()
        return (t or "").strip() or None
    except Exception:
        return None

async def play_links(page) -> list[str]:
    links: list[str] = []
    try:
        anchors = await page.locator("a[href]").element_handles()
        seen = set()
        for a in anchors:
            try:
                href = await a.get_attribute("href")
            except Exception:
                href = None
            if not href:
                continue
            try:
                abs_u = await a.evaluate("(el, base) => new URL(el.getAttribute('href'), base).toString()", page.url)
            except Exception:
                # fallback: best effort
                abs_u = urljoin(page.url, href)
            abs_u = re.sub(r"#.*$", "", abs_u)
            if abs_u not in seen:
                seen.add(abs_u)
                links.append(abs_u)
    except Exception:
        pass
    return links

async def try_close_page(page, timeout_ms: int = 1500) -> None:
    """
    Best-effort, bounded-time page close to avoid dangling Playwright objects
    when the event loop is under load or the browser is recycling.
    """
    if page is None:
        return
    try:
        await asyncio.wait_for(page.close(), timeout=max(0.1, (timeout_ms or 1) / 1000.0))
    except Exception:
        # swallow – we're shutting down / recycling; page might already be gone
        pass
    
class PageContext:
    """
    Small async context manager used by browser.acquire_page() to ensure
    a page is always closed within a bounded time and the global semaphore released.
    """
    def __init__(self, context, sem, close_timeout_ms: int):
        self._context = context
        self._sem = sem
        self._close_timeout_ms = close_timeout_ms
        self._page = None

    async def __aenter__(self):
        await self._sem.acquire()
        self._page = await self._context.new_page()
        return self._page

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await try_close_page(self._page, self._close_timeout_ms)
        finally:
            try:
                self._sem.release()
            except Exception:
                pass

async def await_cancelled(task: asyncio.Task, *, timeout: float = 1.0) -> None:
    """
    Cancel an asyncio task and await its completion to avoid the
    'Future exception was never retrieved' warning.
    """
    if task.done():
        with suppress(Exception):
            _ = task.result()
        return
    task.cancel()
    with suppress(asyncio.CancelledError, PlaywrightError, Exception):
        await asyncio.wait_for(task, timeout=timeout)



# ========== Filter config sourcing (env-first, cfg-optional) ==========
from dataclasses import dataclass

@dataclass(frozen=True)
class FilterParams:
    allow_subdomains: bool = True
    primary_lang: str = "en"
    lang_subdomain_deny: tuple[str, ...] = ()
    lang_path_deny: tuple[str, ...] = ()
    lang_query_keys: tuple[str, ...] = ()

def _as_tuple_csv(val) -> tuple[str, ...]:
    try:
        if isinstance(val, (list, tuple)):
            return tuple(str(x).strip().lower() for x in val if str(x).strip())
        if isinstance(val, str):
            from .utils import getenv_csv  # tolerate intra-module when used elsewhere
            parts = [x.strip().lower() for x in val.split(",")]
            return tuple(p for p in parts if p)
    except Exception:
        pass
    return ()

def _get_filter_params(cfg) -> FilterParams:
    """
    Source filter knobs from cfg when available, otherwise from env.
    This decouples filter logic from config.py (per refactor) while keeping BC if
    older code still sets these on cfg.
    Env variables honored:
      - ALLOW_SUBDOMAINS=true|false
      - PRIMARY_LANG= "en" or "en,us" etc.
      - LANG_SUBDOMAIN_DENY= "fr.,de." (prefix matches)
      - LANG_PATH_DENY= "/fr/,/de-de/"
      - LANG_QUERY_KEYS= "lang,hl,locale"
    """
    # Defaults
    allow_subdomains = True
    primary_lang = "en"
    lang_subdomain_deny: tuple[str, ...] = ()
    lang_path_deny: tuple[str, ...] = ()
    lang_query_keys: tuple[str, ...] = ()

    # Try cfg first (for backward compatibility)
    try:
        if hasattr(cfg, "allow_subdomains"):
            allow_subdomains = bool(getattr(cfg, "allow_subdomains"))
    except Exception:
        pass
    try:
        if getattr(cfg, "primary_lang", None):
            primary_lang = str(getattr(cfg, "primary_lang") or "en")
    except Exception:
        pass
    try:
        if getattr(cfg, "lang_subdomain_deny", None):
            lang_subdomain_deny = _as_tuple_csv(getattr(cfg, "lang_subdomain_deny"))
    except Exception:
        pass
    try:
        if getattr(cfg, "lang_path_deny", None):
            lang_path_deny = _as_tuple_csv(getattr(cfg, "lang_path_deny"))
    except Exception:
        pass
    try:
        if getattr(cfg, "lang_query_keys", None):
            lang_query_keys = _as_tuple_csv(getattr(cfg, "lang_query_keys"))
    except Exception:
        pass

    # Fallback to env if cfg not set
    try:
        from .utils import getenv_bool, getenv_str, getenv_csv
    except Exception:
        # when utils is used as a flat module, getenv_* already defined above
        getenv_bool = globals().get("getenv_bool")
        getenv_str = globals().get("getenv_str")
        getenv_csv = globals().get("getenv_csv")

    if allow_subdomains is True and getenv_bool:
        # Only override if env explicitly set (preserve cfg override otherwise)
        try:
            env_allow = getenv_str("ALLOW_SUBDOMAINS", "").strip().lower()
            if env_allow in {"true","1","yes","on"}:
                allow_subdomains = True
            elif env_allow in {"false","0","no","off"}:
                allow_subdomains = False
        except Exception:
            pass

    try:
        env_lang = getenv_str("PRIMARY_LANG", "") if getenv_str else ""
        if env_lang:
            primary_lang = env_lang
    except Exception:
        pass

    # env CSVs
    try:
        env_sub = getenv_csv("LANG_SUBDOMAIN_DENY", "") if getenv_csv else ()
        if env_sub:
            lang_subdomain_deny = tuple(s.lower() for s in env_sub)
    except Exception:
        pass
    try:
        env_path = getenv_csv("LANG_PATH_DENY", "") if getenv_csv else ()
        if env_path:
            lang_path_deny = tuple(s.lower() for s in env_path)
    except Exception:
        pass
    try:
        env_q = getenv_csv("LANG_QUERY_KEYS", "") if getenv_csv else ()
        if env_q:
            lang_query_keys = tuple(s.lower() for s in env_q)
    except Exception:
        pass

    return FilterParams(
        allow_subdomains=allow_subdomains,
        primary_lang=primary_lang,
        lang_subdomain_deny=lang_subdomain_deny,
        lang_path_deny=lang_path_deny,
        lang_query_keys=lang_query_keys,
    )


# ========== Static-first rendering decision (reduce Playwright usage) ==========
def should_render(url: str, html: str, *, js_text_threshold: int | None = None) -> bool:
    """Return True if a JS render is *likely necessary*.
    Heuristics:
      - empty or near-empty visible text while JS-app signatures are present
      - explicit 'enable javascript' messaging
      - extremely script-heavy pages
      - client-only hash routing (/#/)
    The threshold defaults to env STATIC_JS_APP_TEXT_THRESHOLD or 800.
    """
    try:
        if not html:
            return True
        lower = html.lower()
        if "enable javascript" in lower or "requires javascript" in lower:
            return True
        # pick threshold
        try:
            if js_text_threshold is None and 'getenv_int' in globals():
                js_text_threshold = getenv_int("STATIC_JS_APP_TEXT_THRESHOLD", 800, 200, 4000)  # type: ignore
        except Exception:
            pass
        js_text_threshold = js_text_threshold or 800
        if any(sig in lower for sig in _JS_APP_SIGNATURES):
            if visible_text_len(html) < js_text_threshold:
                return True
        # script density
        scripts = lower.count("<script")
        if scripts > 60 and visible_text_len(html) < js_text_threshold * 0.6:
            return True
        # client hash routing
        if "/#/" in url:
            return True
        return False
    except Exception:
        return False
# ========== Uniform link filtering ==========

def _trace_enabled(cfg) -> bool:
    try:
        if getattr(cfg, "debug_filter", False):
            return True
    except Exception:
        pass
    return os.getenv("SCRAPER_FILTER_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}

def _reason(counter: Counter, key: str):
    counter[key] += 1


# --- fast URL helpers for the filtering hot path ---
from typing import NamedTuple

class _ParsedURL(NamedTuple):
    host: str
    path: str
    query: str
    scheme: str

def _parse_url_fast(u: str) -> _ParsedURL:
    pu = urlparse(u)
    return _ParsedURL(
        host=(pu.hostname or "").lower(),
        path=(pu.path or "").lower(),
        query=(pu.query or ""),
        scheme=(pu.scheme or "").lower(),
    )

def _looks_backend_or_file_fast(parsed: _ParsedURL) -> bool:
    path = parsed.path
    for ext in _DENY_FILE_EXTS:
        if path.endswith(ext):
            return True
    for hint in _BACKEND_ENDPOINT_HINTS:
        if hint in path:
            return True
    return False

def _is_producty_path(path: str) -> bool:
    if _PRODUCT_CORE_RE.search(path):
        return True
    segs = [s for s in path.split("/") if s]
    if 1 < len(segs) <= 5:
        last = segs[-1]
        if len(last) >= 3 and last.endswith("s") and "-" not in last and not last.isdigit():
            return True
    return False

def _looks_non_product_fast(parsed: _ParsedURL) -> bool:
    path = parsed.path
    host = parsed.host

    if path in ("", "/"):
        return False
    if _is_producty_path(path):
        return False
    if _TEL_PATH_RE.search(path or ""):
        return True
    if any(tok in host.split(".") for tok in _NON_PRODUCT_HOST_TOKENS):
        return True
    for frag in _NON_PRODUCT_PATH_PARTS:
        if frag in path:
            return True
    if parsed.query:
        q = dict(parse_qsl(parsed.query, keep_blank_values=True))
        for k in _NON_PRODUCT_QUERY_KEYS:
            if k in q:
                return True

    if any(tok in path for tok in _COOKIE_PATH_HINTS):
        return True
    if parsed.query:
        q = q if 'q' in locals() else dict(parse_qsl(parsed.query, keep_blank_values=True))
        if any(k.lower() in _COOKIE_QUERY_KEYS for k in q):
            return True
    return False
def filter_links(
    source_url: str,
    links: list[str],
    cfg,
    *,
    allow_re: re.Pattern | None = None,
    deny_re: re.Pattern | None = None,
    product_only: bool = True,
    on_drop: Optional[Callable[[str, str], None]] = None,
    on_keep: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """
    Optimized filtering hot path:
      - minimize urlparse/parse_qsl, lowercase, and allocations
      - early hard-stops (non-http, off-site, backend/file, language gates)
      - canonical dedupe once
    Behavior is unchanged vs the previous implementation.
    """
    out: list[str] = []
    seen_canon: set[str] = set()
    source_canon = _canonicalize_for_dedupe(source_url)

    trace = _trace_enabled(cfg)
    reasons = Counter()
    kept = 0
    VERBOSE_CAP = 200
    verbose_count = 0

    def vlog(msg: str):
        nonlocal verbose_count
        if trace and verbose_count < VERBOSE_CAP:
            logger.info("[filter] %s", msg)
            verbose_count += 1

    if trace:
        logger.info(
            "[filter] source=%s allow=%s deny=%s product_only=%s links_in=%d",
            source_url,
            getattr(allow_re, "pattern", None),
            getattr(deny_re, "pattern", None),
            product_only,
            len(links),
        )

    params = _get_filter_params(cfg)
    allow_subdomains = params.allow_subdomains
    extra_hosts = _collect_extra_hosts(cfg)
    accepted_langs = _normalize_primary_lang_tags(params.primary_lang)
    lang_subdomain_deny = set(params.lang_subdomain_deny)
    lang_path_deny = tuple(params.lang_path_deny)
    lang_query_keys = tuple(params.lang_query_keys)

    for u in links:
        if not u or u[:4].lower() != "http":
            reasons["skip:not-http"] += 1
            if on_drop: on_drop(u, "not-http")
            vlog(f"DROP (not-http): {u}")
            continue

        if not same_site(source_url, u, allow_subdomains, extra_hosts=extra_hosts):
            reasons["drop:off-site"] += 1
            if on_drop: on_drop(u, "off-site")
            vlog(f"DROP (off-site): {u}")
            continue

        p = _parse_url_fast(u)

        if _looks_backend_or_file_fast(p):
            reasons["drop:backend-or-file"] += 1
            if on_drop: on_drop(u, "backend-or-file")
            vlog(f"DROP (backend/file): {u}")
            continue

        labels = p.host.split(".")
        if len(labels) >= 3:
            sub = labels[0]
            if sub != "www" and (_LANG_TAG_RE.match(sub) or len(sub) == 2):
                if not _accepts_primary(sub, accepted_langs):
                    reasons["drop:non-english"] += 1
                    if on_drop: on_drop(u, "non-english")
                    vlog(f"DROP (non-en:sub): {u}")
                    continue

        m = _LANG_PATH_PREFIX_RE.match(p.path)
        if m:
            tag = m.group("tag").lower()
            if not _accepts_primary(tag, accepted_langs):
                reasons["drop:non-english"] += 1
                if on_drop: on_drop(u, "non-english")
                vlog(f"DROP (non-en:path): {u}")
                continue

        if lang_subdomain_deny and any(sub and p.host.startswith(sub) for sub in lang_subdomain_deny):
            reasons["drop:lang-subdomain"] += 1
            if on_drop: on_drop(u, "lang-subdomain")
            vlog(f"DROP (lang-subdomain): {u}")
            continue
        if lang_path_deny and any(tok in p.path for tok in lang_path_deny):
            reasons["drop:lang-path"] += 1
            if on_drop: on_drop(u, "lang-path")
            vlog(f"DROP (lang-path): {u}")
            continue
        if lang_query_keys and p.query:
            q = dict(parse_qsl(p.query, keep_blank_values=True))
            if any(k in q for k in lang_query_keys):
                reasons["drop:lang-query-key"] += 1
                if on_drop: on_drop(u, "lang-query-key")
                vlog(f"DROP (lang-query-key): {u}")
                continue

        producty = _is_producty_path(p.path)

        if allow_re and not allow_re.search(u) and not producty:
            reasons["drop:not-match-allow"] += 1
            if on_drop: on_drop(u, "no-allow-match")
            vlog(f"DROP (no allow match): {u}")
            continue

        if deny_re and deny_re.search(u) and not producty:
            reasons["drop:match-deny"] += 1
            if on_drop: on_drop(u, "deny-match")
            vlog(f"DROP (deny match): {u}")
            continue

        if product_only and _looks_non_product_fast(p):
            reasons["drop:non-product"] += 1
            if on_drop: on_drop(u, "non-product")
            vlog(f"DROP (non-product): {u}")
            continue

        canon = _canonicalize_for_dedupe(u)
        if canon == source_canon:
            reasons["drop:self-canon"] += 1
            if on_drop: on_drop(u, "self-canonical")
            vlog(f"DROP (self-canon): {u} -> {canon}")
            continue
        if canon in seen_canon:
            reasons["drop:dup-canon"] += 1
            if on_drop: on_drop(u, "dup-canonical")
            vlog(f"DROP (dup-canon): {u} -> {canon}")
            continue

        seen_canon.add(canon)
        out.append(u)
        kept += 1
        if on_keep: on_keep(u)
        vlog(f"KEEP: {u} -> {canon}")

    if trace:
        logger.info(
            "[filter] kept=%d dropped=%d by_reason=%s",
            kept, len(links) - kept, dict(reasons.most_common())
        )
        if verbose_count >= VERBOSE_CAP:
            logger.info("[filter] verbose cap reached (%d); further per-URL logs suppressed.", VERBOSE_CAP)
    else:
        if links:
            logger.info(
                "[filter] %s -> kept=%d/%d (top drops: %s)",
                urlparse(source_url).hostname or "source",
                kept, len(links),
                ", ".join(f"{k}:{v}" for k, v in reasons.most_common()[:5]) or "none"
            )

    return out