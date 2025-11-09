from __future__ import annotations

import fnmatch
import logging
import re
from functools import lru_cache
from typing import Optional, List, Iterable, Dict, Any, Tuple
from urllib.parse import urlparse, parse_qsl

from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter,
    SEOFilter,
    ContentRelevanceFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from config import language_settings as langcfg

logger = logging.getLogger(__name__)

__all__ = [
    # policies
    "UNIVERSAL_EXTERNALS",
    "DEFAULT_INCLUDE_PATTERNS",
    "DEFAULT_EXCLUDE_PATTERNS",
    "DEFAULT_EXCLUDE_QUERY_KEYS",
    # externals / brand
    "is_universal_external",
    "same_registrable_domain",
    "classify_external",
    # language
    "is_non_english_url",
    "should_drop_for_language",
    "drop_by_response_language",
    "apply_live_language_screen",
    # keep/drop + scoring
    "url_priority",
    "url_should_keep",
    "apply_url_filters",
    "keep_patterns",
    "exclude_patterns",
    "filter_by_patterns",
    # deep-crawl helpers (optional)
    "make_basic_filter_chain",
    "make_relevance_filter",
    "make_seo_filter",
    "make_keyword_scorer",
    # seeding convenience
    "default_product_bm25_query",
]

# =============================================================================
# Centralized "universal external" policy (single source of truth)
# =============================================================================

UNIVERSAL_EXTERNALS: set[str] = {
    # --- Social / community / media mega-platforms ---
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "linkedin.com", "tiktok.com", "pinterest.com", "reddit.com", "slack.com",
    "medium.com", "substack.com", "discord.com", "quora.com", "trustpilot.com",
    "glassdoor.com", "crunchbase.com",

    # --- Link shorteners / trackers ---
    "t.co", "goo.gl", "bit.ly", "bitly.com", "tinyurl.com", "ow.ly", "buff.ly",
    "rebrand.ly", "lnkd.in", "mailchi.mp", "hubspotlinks.com", "adobe.ly",
    "shorturl.at", "urlr.me", "clickfunnels.com",

    # --- App stores ---
    "apps.apple.com", "itunes.apple.com", "play.google.com",
    "appgallery.huawei.com", "galaxy.store", "microsoft.com/store",
    "apps.microsoft.com",

    # --- SSO / identity providers ---
    "accounts.google.com", "login.microsoftonline.com", "auth0.com", "okta.com",
    "onelogin.com", "pingidentity.com", "appleid.apple.com", "login.salesforce.com",

    # --- Docs / KB SaaS & hosted help centers ---
    "intercom.help", "zendesk.com", "freshdesk.com", "helpscoutdocs.com",
    "readme.io", "gitbook.io", "document360.io", "service-now.com",
    "notion.site", "confluence.com", "knowledgeowl.com", "zoho.com",
    "helpjuice.com", "deskera.com",

    # --- Atlassian cloud ---
    "atlassian.net", "statuspage.io", "jira.com", "confluence.net",

    # --- Dev hosting / code forges ---
    "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    "stackblitz.com", "codesandbox.io", "replit.com",

    # --- Google properties (not product sites) ---
    "developers.google.com", "support.google.com", "docs.google.com",
    "drive.google.com", "forms.gle", "maps.google.com", "maps.app.goo.gl",
    "calendar.google.com", "photos.google.com", "accounts.google.com",

    # --- General news / media ---
    "newsweek.com", "cnn.com", "bbc.co.uk", "reuters.com", "bloomberg.com",
    "nytimes.com", "washingtonpost.com", "theguardian.com", "forbes.com",
    "yahoo.com", "nbcnews.com", "cnbc.com", "businessinsider.com", "techcrunch.com",

    # --- Status / uptime / monitoring ---
    "status.io", "statuspal.io", "statuspage.io", "uptime.com", "pingdom.com",
    "betteruptime.com", "freshstatus.io",

    # --- E-commerce / hosting / site builders ---
    "shopify.com", "myshopify.com", "bigcommerce.com", "magento.com",
    "woocommerce.com", "wix.com", "squarespace.com", "weebly.com",
    "webflow.io", "wordpress.com", "godaddy.com", "strikingly.com",
    "ecwid.com", "prestashop.com", "3dcart.com", "shopbase.com",
    "shift4shop.com", "lightspeedhq.com", "zohocommerce.com",

    # --- Marketing / CRM / analytics / ads ---
    "mailchimp.com", "campaignmonitor.com", "activecampaign.com", "hubspot.com",
    "salesforce.com", "pardot.com", "marketo.com", "keap.com", "sendgrid.com",
    "constantcontact.com", "convertkit.com", "drip.com", "klaviyo.com",
    "adobe.com", "doubleclick.net", "googletagmanager.com", "google-analytics.com",
    "analytics.google.com", "tagmanager.google.com", "mixpanel.com",
    "segment.com", "hotjar.com", "fullstory.com", "crazyegg.com",
    "facebook.net", "fbcdn.net", "twitteranalytics.com",

    # --- CDN / asset / security ---
    "cloudflare.com", "cdn.cloudflare.net", "akamaihd.net", "akamaized.net",
    "fastly.net", "vercel.app", "netlify.app", "edgekey.net", "edgesuite.net",
    "stackpathcdn.com", "azureedge.net", "firebaseapp.com", "cloudfront.net",

    # --- Payment / checkout platforms ---
    "stripe.com", "paypal.com", "braintreepayments.com", "squareup.com",
    "adyen.com", "authorize.net", "klarna.com", "afterpay.com", "affirm.com",

    # --- Booking / ticketing / delivery SaaS ---
    "eventbrite.com", "ticketmaster.com", "opentable.com", "resy.com",
    "doordash.com", "ubereats.com", "postmates.com", "grubhub.com",

    # --- Survey / forms / scheduling ---
    "surveymonkey.com", "typeform.com", "jotform.com", "googleforms.com",
    "calendly.com", "doodle.com", "wufoo.com",

    # --- Other frequent false positives ---
    "sproutsocialstatus.com", "newswhip.com", "mozilla.org", "google.com",
    "microsoft.com", "apple.com", "amazon.com", "aws.amazon.com", "adobe.io",
    "creativecommons.org", "ietf.org", "w3.org", "archive.org", "who.int",
    "un.org", "europa.eu", "g2.com", "capterra.com", "producthunt.com",
}

FUNCTIONAL_SUBDOMAIN_PREFIXES: tuple[str, ...] = (
    "community", "status", "learning", "learn", "docs", "developer", "developers",
    "help", "support", "kb", "academy", "login", "signin", "auth", "accounts",
    "news", "press", "media",
)

# =============================================================================
# Robust pattern matching (glob-to-regex + native regex)
# =============================================================================

# Prefix to declare an explicit regex pattern (case-insensitive).
REGEX_PREFIX = ("re:", "regex:")

@lru_cache(maxsize=4096)
def _compile_one(pattern: str) -> re.Pattern:
    """
    Compile a single pattern. Supports:
    - Glob patterns (default), translated via fnmatch.translate
    - Regex patterns if prefixed with 're:' or 'regex:' (case-insensitive)
    """
    p = (pattern or "").strip()
    if not p:
        return re.compile(r"^$")  # never matches

    # Explicit regex
    lower = p.lower()
    for prefix in REGEX_PREFIX:
        if lower.startswith(prefix):
            body = p[len(prefix):].strip()
            try:
                return re.compile(body, re.IGNORECASE)
            except re.error as e:
                logger.warning("Invalid regex pattern '%s': %s. Falling back to no-op.", p, e)
                return re.compile(r"^$")

    # Glob → regex
    try:
        # fnmatch.translate already anchors ^...$, we add re.IGNORECASE
        return re.compile(fnmatch.translate(p), re.IGNORECASE)
    except re.error as e:
        logger.warning("Failed to compile glob '%s': %s. Falling back to literal-safe regex.", p, e)
        return re.compile(re.escape(p), re.IGNORECASE)

def _compile_many(patterns: List[str]) -> List[re.Pattern]:
    return [_compile_one(p) for p in (patterns or []) if p and p.strip()]

def _url_for_match(url: str) -> str:
    # Case-insensitive match against full URL string
    return (url or "").strip()

def _match_any(url: str, patterns: List[str]) -> bool:
    if not patterns:
        return False
    s = _url_for_match(url)
    for rx in _compile_many(patterns):
        if rx.search(s):
            return True
    return False

def _count_hits(url: str, patterns: List[str]) -> int:
    if not patterns:
        return 0
    s = _url_for_match(url)
    c = 0
    for rx in _compile_many(patterns):
        if rx.search(s):
            c += 1
    return c

# --- SMART INCLUDE TOKENS ----------------------------------------------------
# Use language-configured tokens (keeps values identical to previous default)
SMART_INCLUDE_TOKENS: tuple[str, ...] = tuple(langcfg.get("SMART_INCLUDE_TOKENS"))

# precompile a word-ish boundary for tokens inside a path segment
#   matches token preceded/followed by start/end/[-_./]
#   keeps it lenient enough to catch hyphen/underscore compounds
SMART_TOKEN_RX_CACHE: Dict[str, re.Pattern] = {}

def _token_rx(tok: str) -> re.Pattern:
    rx = SMART_TOKEN_RX_CACHE.get(tok)
    if rx is None:
        # Allow forms like: foo-token, token-foo, /token/, /token-foo/, foo_token, token.html
        rx = re.compile(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", re.IGNORECASE)
        SMART_TOKEN_RX_CACHE[tok] = rx
    return rx

def _smart_include_hit(url: str) -> bool:
    try:
        p = urlparse(url).path or "/"
    except Exception:
        p = url or ""
    for tok in SMART_INCLUDE_TOKENS:
        if _token_rx(tok).search(p):
            return True
    return False

# =============================================================================
# Helpers
# =============================================================================

def _hostname(url_or_host: Any) -> str:
    try:
        if isinstance(url_or_host, str):
            s = url_or_host.strip()
            if not s:
                return ""
            if "://" in s or s.startswith(("file://", "raw:")):
                return (urlparse(s).hostname or "").lower()
            return s.lower()
        if isinstance(url_or_host, dict):
            for k in ("final_url", "url", "href", "link"):
                v = url_or_host.get(k)
                if isinstance(v, str):
                    return _hostname(v)
        s = str(url_or_host)
        if "://" in s:
            return (urlparse(s).hostname or "").lower()
        return s.lower()
    except Exception:
        return ""

def _label_boundary_match(host: str, suffix: str) -> bool:
    h = _hostname(host)
    s = suffix.lower()
    return h == s or h.endswith("." + s)

def _has_functional_prefix(host: str) -> bool:
    h = _hostname(host)
    if not h:
        return False
    labels = h.split(".")
    if len(labels) < 3:
        return False
    return labels[0] in FUNCTIONAL_SUBDOMAIN_PREFIXES

def is_universal_external(url_or_host: Any) -> bool:
    host = _hostname(url_or_host)
    hit = False
    if host:
        hit = any(_label_boundary_match(host, sfx) for sfx in UNIVERSAL_EXTERNALS)
        if not hit and _has_functional_prefix(host):
            hit = True
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering] is_universal_external(%s -> %s) -> %s", url_or_host, host, hit)
    return hit

# =============================================================================
# Default URL pattern heuristics (KEEP / DROP)
# =============================================================================

DEFAULT_INCLUDE_PATTERNS: List[str] = list(langcfg.get("DEFAULT_INCLUDE_PATTERNS"))

DEFAULT_EXCLUDE_PATTERNS: List[str] = list(langcfg.get("DEFAULT_EXCLUDE_PATTERNS"))

DEFAULT_EXCLUDE_QUERY_KEYS: List[str] = list(langcfg.get("DEFAULT_EXCLUDE_QUERY_KEYS"))

# =============================================================================
# Deep crawling filter helpers (optional)
# =============================================================================

def make_basic_filter_chain(
    *,
    allowed_domains: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
) -> FilterChain:
    filters = []
    if allowed_domains:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[filtering] DomainFilter allowed=%s", allowed_domains)
        filters.append(DomainFilter(allowed_domains=allowed_domains))  # type: ignore
    if patterns:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[filtering] URLPatternFilter patterns=%s", patterns)
        filters.append(URLPatternFilter(patterns=patterns))  # type: ignore
    if content_types:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[filtering] ContentTypeFilter allowed_types=%s", content_types)
        filters.append(ContentTypeFilter(allowed_types=content_types))  # type: ignore
    return FilterChain(filters)  # type: ignore

def make_relevance_filter(query: str, threshold: float = 0.7) -> FilterChain:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering] ContentRelevanceFilter query=%s threshold=%.2f", query, threshold)
    return FilterChain([ContentRelevanceFilter(query=query, threshold=threshold)])  # type: ignore

def make_seo_filter(keywords: List[str], threshold: float = 0.5) -> FilterChain:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering] SEOFilter keywords=%s threshold=%.2f", keywords, threshold)
    return FilterChain([SEOFilter(keywords=keywords, threshold=threshold)])  # type: ignore

def make_keyword_scorer(keywords: List[str], weight: float = 0.7) -> KeywordRelevanceScorer:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering] KeywordRelevanceScorer keywords=%s weight=%.2f", keywords, weight)
    return KeywordRelevanceScorer(keywords=keywords, weight=weight)  # type: ignore

# =============================================================================
# URL seeding helpers (status/score filtering)
# =============================================================================

def filter_seeded_urls(
    urls: Iterable[Dict[str, Any]],
    *,
    require_status: Optional[str] = "valid",
    min_score: Optional[float] = None,
    drop_universal_externals: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in urls:
        u_url = str(u.get("final_url") or u.get("url") or "")
        if not u_url:
            continue

        if drop_universal_externals and is_universal_external(u_url):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering] drop universal external: %s", u_url)
            continue

        if require_status and u.get("status") != require_status:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[filtering] drop status!=%s: %s (status=%s)",
                    require_status, u_url, u.get("status")
                )
            continue

        if min_score is not None:
            score = u.get("relevance_score")
            if score is None or float(score) < float(min_score):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[filtering] drop score<%.2f: %s (score=%s)",
                        float(min_score), u_url, score
                    )
                continue

        out.append(u)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering] filter_seeded_urls -> kept=%d", len(out))
    return out

# -----------------------------------------------------------------------------


def keep_patterns(urls: Iterable[Dict[str, Any]], patterns: List[str]) -> List[Dict[str, Any]]:
    """
    Keep only URLs matching any of the provided patterns (glob or regex with 're:' prefix).
    """
    out = []
    for u in urls:
        url = str(u.get("final_url") or u.get("url") or "")
        if _match_any(url, patterns) or _smart_include_hit(url):
            out.append(u)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering] keep_patterns -> kept=%d", len(out))
    return out


def exclude_patterns(
    urls: Iterable[Dict[str, Any]],
    patterns: List[str],
    *,
    include_override: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Exclude URLs matching any of the provided patterns — BUT if a URL also
    matches any pattern in `include_override` OR the smart include tokens,
    we KEEP it (include wins).
    """
    include_override = include_override or []
    out = []
    for u in urls:
        url = str(u.get("final_url") or u.get("url") or "")
        inc_hit = _match_any(url, include_override) or _smart_include_hit(url)
        exc_hit = _match_any(url, patterns)
        if exc_hit and not inc_hit:
            continue
        out.append(u)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering] exclude_patterns -> kept=%d", len(out))
    return out


def filter_by_patterns(
    urls: Iterable[Dict[str, Any]],
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Unified include/exclude pass with **include-over-exclude precedence**.
    - If a URL matches both include and exclude → KEEP.
    - If it matches exclude only → DROP.
    - If it matches include only → KEEP.
    - If neither matches → KEEP (patterns act as hints, not absolute unless exclude hits).

    Supports glob patterns and explicit regex ('re:' prefix) and also
    uses smart include tokens to rescue near-misses like '/power-solutions/'.
    """
    include = include or []
    exclude = exclude or []
    out: List[Dict[str, Any]] = []
    for u in urls:
        url = str(u.get("final_url") or u.get("url") or "")
        inc_hit = _match_any(url, include) or _smart_include_hit(url)
        exc_hit = _match_any(url, exclude)
        if exc_hit and not inc_hit:
            continue
        out.append(u)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[filtering] filter_by_patterns(include=%d, exclude=%d) -> kept=%d",
            len(include), len(exclude), len(out)
        )
    return out

# =============================================================================
# Language filtering (URL-level heuristics)
# =============================================================================

# language-sensitive constants imported from language_settings (keeps original values)
_LANG_CODES = set(langcfg.get("LANG_CODES"))

_REGIONAL_LANG_CODES = set(langcfg.get("REGIONAL_LANG_CODES"))

_ENGLISH_REGIONS = set(langcfg.get("ENGLISH_REGIONS"))

_CC_TO_LANG = dict(langcfg.get("CC_TO_LANG"))

def _first_path_segment(path: str) -> str:
    if not path:
        return ""
    for part in path.split("/"):
        if part:
            return part.lower()
    return ""

def _host_labels(host: str) -> list[str]:
    return (host or "").lower().strip(".").split(".") if host else []

def _registrable_domain(host: str) -> str:
    labels = _host_labels(host)
    if len(labels) < 2:
        return host or ""
    last2 = ".".join(labels[-2:])
    last3 = ".".join(labels[-3:]) if len(labels) >= 3 else ""
    if last3 in {
        "co.uk", "org.uk", "ac.uk",
        "com.au", "net.au", "org.au",
        "co.jp", "ne.jp", "or.jp",
        "com.cn", "com.sg", "com.br",
    } and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last2

def _normalize_token(tok: str) -> str:
    return (tok or "").strip().lower().replace("_", "-")

def _is_lang_token(tok: str) -> bool:
    t = _normalize_token(tok)
    return (t in _LANG_CODES) or (t in _REGIONAL_LANG_CODES)

def _cc_as_lang(tok: str) -> str | None:
    t = _normalize_token(tok)
    return _CC_TO_LANG.get(t)

def _is_englishish_token(tok: str, accept_en_regions: set[str], primary_lang: str) -> bool:
    t = _normalize_token(tok)
    if t == "www":
        return True
    if t == primary_lang or t.startswith(f"{primary_lang}-"):
        return True
    if t in accept_en_regions or t in {f"{primary_lang}-{r}" for r in accept_en_regions}:
        return True
    return False

def _token_is_non_english(tok: str, accept_en_regions: set[str], primary_lang: str) -> bool:
    t = _normalize_token(tok)
    if not t:
        return False
    if _is_lang_token(t):
        if _is_englishish_token(t, accept_en_regions, primary_lang):
            return False
        root = t.split("-")[0]
        return (root in _LANG_CODES) and (root != "en")
    mapped = _cc_as_lang(t)
    if mapped:
        return mapped != "en"
    return False

def same_registrable_domain(a: str, b: str) -> bool:
    ha = (urlparse(a).hostname or "").lower()
    hb = (urlparse(b).hostname or "").lower()
    if not ha or not hb:
        return False
    return _registrable_domain(ha) == _registrable_domain(hb)

def is_non_english_url(
    u: str,
    primary_lang: str = "en",
    accept_en_regions: set[str] | None = None,
    strict_cctld: bool = False,
) -> bool:
    acc = set(accept_en_regions or _ENGLISH_REGIONS)

    try:
        pu = urlparse(u)
    except Exception:
        return False

    host = (pu.hostname or "").lower()
    path = (pu.path or "/").lower()
    q = {k.lower(): _normalize_token(v) for k, v in parse_qsl(pu.query or "", keep_blank_values=True)}

    # 1) Subdomain markers (only DROP on explicit non-en; never early-keep)
    labels = _host_labels(host)
    if len(labels) >= 3:
        sub = labels[0]
        if _token_is_non_english(sub, acc, primary_lang):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering.lang] sub=%s -> non_en=True url=%s", sub, u)
            return True

    # 2) First path segment (only DROP on explicit non-en; never early-keep)
    first_seg = _first_path_segment(path)
    if first_seg and _token_is_non_english(first_seg, acc, primary_lang):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[filtering.lang] path-seg=%s -> non_en=True url=%s", first_seg, u)
        return True

    # 3) Query param language hints (DROP on explicit non-en)
    for k in {"lang", "hl", "locale", "lc", "lr", "region"}:
        v = q.get(k, "")
        if not v:
            continue
        if _token_is_non_english(v, acc, primary_lang):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering.lang] query %s=%s -> non_en=True url=%s", k, v, u)
            return True
        if len(v) == 2 and v in _LANG_CODES and v != primary_lang:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering.lang] query %s=%s (2-letter) non_en -> drop url=%s", k, v, u)
            return True

    # 4) ccTLD fallback (optional; conservative)
    if strict_cctld:
        labels = _host_labels(host)
        cc = labels[-1] if labels else ""
        likely_non_en = {
            "fr","de","es","it","pt","ru","jp","kr","cn","tw","hk","vn","th","id","my",
            "tr","pl","cz","sk","ro","hu","bg","gr","ua","nl","be","dk","no","se","fi",
            "il","ir","sa"
        }
        if cc and cc not in _ENGLISH_REGIONS and cc not in {"us","uk","gb"} and cc in likely_non_en:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering.lang] strict ccTLD=%s non_en -> drop url=%s", cc, u)
            return True

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering.lang] default keep url=%s", u)
    return False

def should_drop_for_language(
    u: str,
    primary_lang: str = "en",
    accept_en_regions: set[str] | None = None,
    strict_cctld: bool = False,
) -> bool:
    drop = is_non_english_url(
        u,
        primary_lang=primary_lang,
        accept_en_regions=accept_en_regions,
        strict_cctld=strict_cctld,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering.lang] should_drop_for_language(%s) -> %s", u, drop)
    return drop

# =============================================================================
# Live-check language hardening (headers & <html lang>)
# =============================================================================

def _normalize_lang_code(code: str) -> str:
    return (code or "").strip().lower().replace("_", "-")

def _is_non_english_lang_code(code: str, accept_en_regions: set[str]) -> bool:
    c = _normalize_lang_code(code)
    if not c:
        return False
    if c == "en" or c in {"en-us", "en-gb"}:
        return False
    if c in accept_en_regions or c.startswith("en-"):
        return False
    root = c.split("-")[0]
    return (root in _LANG_CODES) and (root != "en")

def drop_by_response_language(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    html_lang: Optional[str] = None,
    primary_lang: str = "en",
    accept_en_regions: Optional[set[str]] = None,
) -> bool:
    acc = set(accept_en_regions or _ENGLISH_REGIONS)

    if headers:
        lowered = {str(k).lower(): str(v) for k, v in headers.items()}
        h = lowered.get("content-language") or lowered.get("content_language")
        if h:
            tokens = [t.strip() for t in h.replace(";", ",").split(",") if t.strip()]
            for tok in tokens:
                if _is_non_english_lang_code(tok, acc):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("[filtering.live-lang] drop by Content-Language=%s url=%s", tok, url)
                    return True

    if html_lang and _is_non_english_lang_code(html_lang, acc):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[filtering.live-lang] drop by <html lang=%s> url=%s", html_lang, url)
        return True

    return False

def apply_live_language_screen(
    items: List[Dict[str, Any]],
    headers_key: str = "headers",
    html_lang_key: str = "html_lang",
    primary_lang: str = "en",
    accept_en_regions: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    acc = set(accept_en_regions or _ENGLISH_REGIONS)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for u in items:
        url = u.get("final_url") or u.get("url") or ""
        if not url:
            continue
        headers = u.get(headers_key)
        html_lang = u.get(html_lang_key)
        if drop_by_response_language(url, headers=headers, html_lang=html_lang,
                                     primary_lang=primary_lang, accept_en_regions=acc):
            dropped += 1
            continue
        kept.append(u)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering.live-lang] kept=%d dropped=%d (by response metadata)", len(kept), dropped)
    return kept

# =============================================================================
# First-party brand / cross-domain helpers
# =============================================================================

def classify_external(
    base_url: str,
    candidate_url: str,
    allowed_brand_hosts: Optional[set[str]] = None,
) -> str:
    """
    Classify a cross-domain candidate relative to base_url:
      - 'brand'     : host explicitly whitelisted
      - 'same_site' : same registrable domain
      - 'external'  : everything else
    """
    allowed = set(allowed_brand_hosts or ())
    base_host = (urlparse(base_url).hostname or "").lower()
    cand_host = (urlparse(candidate_url).hostname or "").lower()

    if not base_host or not cand_host:
        return "external"

    if cand_host in allowed:
        return "brand"
    if _registrable_domain(base_host) == _registrable_domain(cand_host):
        return "same_site"
    return "external"

# =============================================================================
# Priority scoring & composite filtering
# =============================================================================

def _pattern_hits(url: str, patterns: List[str]) -> int:
    # Use compiled regex counting
    return _count_hits(url, patterns)

def url_priority(
    url: str,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Tuple[int, bool, bool]:
    include = include or []
    exclude = exclude or []
    inc_hits = _count_hits(url, include) + (1 if _smart_include_hit(url) else 0)
    exc_hits = _count_hits(url, exclude)
    score = inc_hits * 10 - exc_hits * 5
    return score, inc_hits > 0, exc_hits > 0

def url_should_keep(
    url: str,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    drop_universal_externals: bool = True,
    lang_primary: str = "en",
    lang_accept_en_regions: Optional[set[str]] = None,
    lang_strict_cctld: bool = False,
    include_overrides_language: bool = False,
) -> bool:
    """
    Decide if a URL should be crawled:

    • Drop universal externals outright.
    • Language filter applies first (include does NOT override unless include_overrides_language=True).
    • If both include & exclude match → KEEP (include wins).
    • If exclude only matches → DROP.
    • If include only matches → KEEP.
    • If neither matches → KEEP (patterns act as hints, not absolute unless exclude-only hits).

    Matching supports glob patterns and explicit regex via 're:' prefix.
    Smart include tokens also raise include-likelihood for near misses (/power-solutions/).
    """
    include = include or []
    exclude = exclude or []

    if drop_universal_externals and is_universal_external(url):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[filtering.keep] universal external -> drop: %s", url)
        return False

    score, inc_hit, exc_hit = url_priority(url, include=include, exclude=exclude)

    # Language gate (unless explicitly overridden by include)
    if not (include_overrides_language and inc_hit):
        if should_drop_for_language(
            url,
            primary_lang=lang_primary,
            accept_en_regions=lang_accept_en_regions,
            strict_cctld=lang_strict_cctld,
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering.keep] language drop -> %s (score=%d inc=%s exc=%s)", url, score, inc_hit, exc_hit)
            return False

    # Exclude is hard drop unless include also hits
    if exc_hit and not inc_hit:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[filtering.keep] exclude hit without include -> drop: %s (score=%d)", url, score)
        return False

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[filtering.keep] keep: %s (score=%d inc=%s exc=%s)", url, score, inc_hit, exc_hit)
    return True

def apply_url_filters(
    urls: Iterable[str] | Iterable[Dict[str, Any]],
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    drop_universal_externals: bool = True,
    lang_primary: str = "en",
    lang_accept_en_regions: Optional[set[str]] = None,
    lang_strict_cctld: bool = False,
    include_overrides_language: bool = False,
    sort_by_priority: bool = True,
    base_url: Optional[str] = None,
    keep_brand_domains: bool = False,
    allowed_brand_hosts: Optional[set[str]] = None,
) -> List[str] | List[Dict[str, Any]]:
    """
    Apply language policy (first), then include/exclude, then optional brand allowance.

    Key behaviors:
      • Include-pattern hit overrides exclude-pattern hit (KEEP if both match).
      • Same-registrable-domain is considered first-party, but it does NOT bypass path excludes.
      • Explicitly allowed brand hosts (allowed_brand_hosts) may bypass path excludes.
      • Language drop has precedence unless include_overrides_language=True.

    Matching supports glob and 're:' regex patterns + smart include tokens.
    """
    include = include or []
    exclude = exclude or []
    allowed_brand_hosts = set(allowed_brand_hosts or ())

    items = list(urls)
    is_dict_items = bool(items and isinstance(items[0], dict))

    kept: List[Tuple[int, bool, bool, Any]] = []

    for item in items:
        u = item["url"] if is_dict_items else str(item)
        score, inc_hit, exc_hit = url_priority(u, include=include, exclude=exclude)

        # Universal externals are dropped regardless
        if drop_universal_externals and is_universal_external(u):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering.apply] DROP universal external: %s", u)
            continue

        # Language drop (unless include override)
        lang_drop = not (include_overrides_language and inc_hit) and should_drop_for_language(
            u,
            primary_lang=lang_primary,
            accept_en_regions=lang_accept_en_regions,
            strict_cctld=lang_strict_cctld,
        )
        if lang_drop:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[filtering.apply] DROP by language: %s", u)
            continue

        # Brand classification
        brand_allowed_host = False
        brand_first_party = False
        if keep_brand_domains:
            if base_url:
                brand_first_party = same_registrable_domain(base_url, u)
            host = (urlparse(u).hostname or "").lower()
            if host in allowed_brand_hosts:
                brand_allowed_host = True

        # Exclude without include is a hard drop (unless explicitly allowed host)
        if exc_hit and not inc_hit and not brand_allowed_host:
            if logger.isEnabledFor(logging.DEBUG):
                why = "exclude (same-reg first-party but no include)" if brand_first_party else "exclude"
                logger.debug("[filtering.apply] DROP by %s: %s", why, u)
            continue

        kept.append((score, inc_hit, exc_hit, item))

    if sort_by_priority and kept:
        # Sort: include-hit first, then by score descending
        kept.sort(key=lambda t: (not t[1], -t[0]))

    result = [t[3] for t in kept]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[filtering.apply] kept=%d / total=%d (sorted=%s, keep_brand=%s)",
            len(result), len(items), bool(sort_by_priority), bool(keep_brand_domains),
        )
    return result

# =============================================================================
# Seeding convenience: universal BM25 query for product/service pages
# =============================================================================

def default_product_bm25_query() -> str:
    """
    Compact BM25 query string tuned for seeding to favor product/service landing pages.
    Use with SeedingConfig(query=..., scoring_method="bm25").
    """
    # Use language settings provider so this is language-aware and identical to previous defaults
    return langcfg.default_product_bm25_query()