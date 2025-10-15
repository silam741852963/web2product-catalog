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

def is_non_english_url(u: str, primary_lang: str = "en") -> bool:
    """
    Return True if the URL clearly targets a non-English locale via subdomain,
    path prefix, or query parameter values.
    """
    try:
        pu = urlparse(u)
        host = (pu.hostname or "").lower()
        path = (pu.path or "/").lower()
        q = dict(parse_qsl(pu.query or "", keep_blank_values=True))
        primary = (primary_lang or "en").lower()

        # 1) subdomain like fr.example.com, de.example.com, zh-cn.example.com
        #    (ignore common www)
        labels = host.split(".")
        if len(labels) >= 3:  # subdomain present
            sub = labels[0]
            if sub in _LANG_CODES and not sub.startswith(primary):
                return True
            # accept regioned english (en, en-us) only; anything else is non-en
            if sub != "www" and re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", sub):
                if not sub.startswith(primary):
                    return True

        # 2) path prefix like /fr/, /de-de/, /zh-cn/
        m = _LANG_PATH_PREFIX_RE.match(path)
        if m:
            tag = m.group("tag").lower()
            if tag in _LANG_CODES and not tag.startswith(primary):
                return True
            if re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", tag) and not tag.startswith(primary):
                return True

        # 3) query keys lang/hl/locale/lc/lr/region set to non-en
        for k in ("lang", "hl", "locale", "lc", "lr", "region"):
            v = (q.get(k) or "").lower()
            if v and not v.startswith(primary):
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
    if any(sig in lower for sig in _JS_APP_SIGNATURES) and visible_text_len(html) < threshold:
        return True
    if "enable javascript" in lower or "requires javascript" in lower:
        return True
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(data, encoding=encoding)
    tmp.replace(path)

def append_jsonl(path: Path, json_line: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=encoding) as f:
        f.write(json_line.rstrip() + "\n")

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
    Apply same-site, language, backend/file, regex, and product heuristics.
    Collapse "same page, different query" via canonicalization.
    When tracing is enabled (cfg.debug_filter or SCRAPER_FILTER_TRACE=1),
    log per-URL decisions (capped) and a final summary.
    """
    out: list[str] = []
    seen_canon: set[str] = set()
    source_canon = _canonicalize_for_dedupe(source_url)

    trace = _trace_enabled(cfg)
    reasons = Counter()
    kept = 0
    # cap verbose per-URL logs so we don't flood (per invocation)
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

    for u in links:
        # baseline normalization for logging
        canon = None

        if not is_http_url(u):
            _reason(reasons, "skip:not-http")
            if on_drop: on_drop(u, "not-http")
            vlog(f"DROP (not-http): {u}")
            continue

        # same-site (respect subdomain policy, incl. aliases/probation via cfg.extra_same_site_hosts)
        extra_hosts = _collect_extra_hosts(cfg)
        if not same_site(source_url, u, cfg.allow_subdomains, extra_hosts=extra_hosts):
            _reason(reasons, "drop:off-site")
            if on_drop: on_drop(u, "off-site")
            vlog(f"DROP (off-site): {u}")
            continue

        pu = urlparse(u)
        host = (pu.hostname or "").lower()
        path = (pu.path or "").lower()

        # --- HARD language filter (English-only) ---
        if is_non_english_url(u, getattr(cfg, "primary_lang", "en")):
            _reason(reasons, "drop:non-english")
            if on_drop: on_drop(u, "non-english")
            vlog(f"DROP (non-en): {u}")
            continue

        # complementary language gates (subdomain/path/query)
        if any(sub and host.startswith(sub.strip().lower()) for sub in cfg.lang_subdomain_deny):
            _reason(reasons, "drop:lang-subdomain")
            if on_drop: on_drop(u, "lang-subdomain")
            vlog(f"DROP (lang-subdomain): {u}")
            continue
        if any(p and p in path for p in cfg.lang_path_deny):
            _reason(reasons, "drop:lang-path")
            if on_drop: on_drop(u, "lang-path")
            vlog(f"DROP (lang-path): {u}")
            continue
        if pu.query:
            q = dict(parse_qsl(pu.query, keep_blank_values=True))
            if any(k in q for k in cfg.lang_query_keys):
                _reason(reasons, "drop:lang-query-key")
                if on_drop: on_drop(u, "lang-query-key")
                vlog(f"DROP (lang-query-key): {u}")
                continue

        # backend endpoints & files (hard stop)
        if _looks_backend_or_file(u):
            _reason(reasons, "drop:backend-or-file")
            if on_drop: on_drop(u, "backend-or-file")
            vlog(f"DROP (backend/file): {u}")
            continue

        # --- Structure-aware productiness (brand-agnostic) ---
        is_producty = is_producty_url(u)

        # allow/deny regex policy (SOFT allow for producty; deny overridden by producty)
        if allow_re and not allow_re.search(u):
            if not is_producty:
                _reason(reasons, "drop:not-match-allow")
                if on_drop: on_drop(u, "no-allow-match")
                vlog(f"DROP (no allow match): {u}")
                continue

        if deny_re and deny_re.search(u):
            if not is_producty:  # productiness overrides deny unless hard-stop (handled above)
                _reason(reasons, "drop:match-deny")
                if on_drop: on_drop(u, "deny-match")
                vlog(f"DROP (deny match): {u}")
                continue

        # product-like heuristic (after SOFT allow/deny): if product_only, drop obvious non-product
        if product_only and looks_non_product_url(u):
            _reason(reasons, "drop:non-product")
            if on_drop: on_drop(u, "non-product")
            vlog(f"DROP (non-product): {u}")
            continue

        # canonicalize & dedupe (collapse same page, different query)
        canon = _canonicalize_for_dedupe(u)
        if canon == source_canon:
            _reason(reasons, "drop:self-canon")
            if on_drop: on_drop(u, "self-canonical")
            vlog(f"DROP (self-canon): {u} -> {canon}")
            continue
        if canon in seen_canon:
            _reason(reasons, "drop:dup-canon")
            if on_drop: on_drop(u, "dup-canonical")
            vlog(f"DROP (dup-canon): {u} -> {canon}")
            continue

        seen_canon.add(canon)
        out.append(u)
        kept += 1
        if on_keep: on_keep(u)
        vlog(f"KEEP: {u} -> {canon}")

    # Summary line
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