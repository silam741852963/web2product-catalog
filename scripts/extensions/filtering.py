from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Iterable, Optional, Set, Tuple
from urllib.parse import urlsplit

from crawl4ai.deep_crawling.filters import URLFilter

from configs import language as lang_cfg

__all__ = [
    "UniversalExternalFilter",
    "HTMLContentFilter",
    "LanguageAwareURLFilter",
    "FirstTimeURLFilter",
]

logger = logging.getLogger(__name__)


def _lstrip_lower(s: str) -> str:
    # Avoid repeated allocations in tight paths; callers should already have str.
    return (s or "").lstrip().lower()


def _strip_dot_lower(host: str) -> str:
    return (host or "").strip().strip(".").lower()


def _strip_www(host: str) -> str:
    h = _strip_dot_lower(host)
    if h.startswith("www.") and len(h) > 4:
        return h[4:]
    return h


def _fast_host_from_url(url: str) -> str:
    """
    Fast best-effort host extraction for typical crawler URLs.

    - Handles:
        https://a.b/c
        http://a.b
        //a.b/c
        a.b/c
        a.b
    - Avoids urlsplit for the common "bare host/path" case.
    """
    s = (url or "").strip()
    if not s:
        return ""

    # Protocol-relative
    if s.startswith("//"):
        try:
            return _strip_dot_lower(urlsplit("http:" + s).hostname or "")
        except Exception:
            return ""

    # No scheme: treat as "host[/...]" (fast path)
    if "://" not in s:
        host = s.split("/", 1)[0].strip()
        # Strip port if present (not IPv6-safe, but good enough for bare hosts)
        if ":" in host and not host.startswith("["):
            host = host.split(":", 1)[0]
        return _strip_dot_lower(host)

    try:
        return _strip_dot_lower(urlsplit(s).hostname or "")
    except Exception:
        return ""


# =============================================================================
# Registrable domain (lightweight eTLD+1-ish)
# =============================================================================

_TWO_LEVEL_SUFFIXES: frozenset[str] = frozenset(
    {
        "co.uk",
        "org.uk",
        "ac.uk",
        "com.au",
        "net.au",
        "org.au",
        "co.jp",
        "ne.jp",
        "or.jp",
        "com.cn",
        "com.sg",
        "com.br",
    }
)

_GOV_SECOND_LEVELS: frozenset[str] = frozenset({"gov", "gouv", "go", "govt"})


def _registrable_domain(host: str) -> str:
    """
    Lightweight registrable domain:
      - foo.example.com -> example.com
      - a.b.example.co.uk -> example.co.uk
      - localhost -> localhost
    """
    h = _strip_dot_lower(host)
    if not h:
        return ""
    # Fast split (no filtering)
    labels = h.split(".")
    if len(labels) < 2:
        return h

    last2 = labels[-2] + "." + labels[-1]
    if last2 in _TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        return labels[-3] + "." + last2
    return last2


def _is_government_domain(host: str) -> bool:
    """
    Heuristic government domain detection based on registrable domain:
      - foo.gov
      - foo.gov.<cc>
      - foo.gouv.<cc>
      - foo.go.<cc>
      - foo.govt.<cc>
    """
    rd = _registrable_domain(host)
    if not rd:
        return False
    parts = rd.split(".")
    if len(parts) < 2:
        return False
    if parts[-1] == "gov":
        return True
    return parts[-2] in _GOV_SECOND_LEVELS


def _top_level_label(host: str) -> str:
    h = _strip_dot_lower(host)
    if not h:
        return ""
    # rsplit is faster than split for just last label
    if "." not in h:
        return h
    return h.rsplit(".", 1)[-1]


def _suffix_match(host_norm: str, suffix_norm: str) -> bool:
    """
    Label boundary protected suffix match.
    Inputs must be already normalized (lower, stripped dots).
    """
    if not host_norm or not suffix_norm:
        return False
    return host_norm == suffix_norm or host_norm.endswith("." + suffix_norm)


# =============================================================================
# Shared normalization caches
# =============================================================================


@lru_cache(maxsize=32)
def _normalized_universal_domains_cache(key: Tuple[str, ...]) -> frozenset[str]:
    """
    Takes a stable (sorted, deduped) tuple key and returns registrable domains.
    """
    out: Set[str] = set()
    for raw in key:
        h = _strip_dot_lower(_fast_host_from_url(raw) or raw)
        if not h:
            continue
        rd = _registrable_domain(h)
        if rd:
            out.add(rd)
    return frozenset(out)


@lru_cache(maxsize=64)
def _normalized_dataset_domains_cache(key: Tuple[str, ...]) -> frozenset[str]:
    out: Set[str] = set()
    for raw in key:
        h = _strip_dot_lower(_fast_host_from_url(raw) or raw)
        if not h:
            continue
        rd = _registrable_domain(h)
        if rd:
            out.add(rd)
    return frozenset(out)


# =============================================================================
# UniversalExternalFilter
# =============================================================================

FUNCTIONAL_SUBDOMAIN_PREFIXES: tuple[str, ...] = (
    "community",
    "status",
    "learning",
    "learn",
    "docs",
    "developer",
    "developers",
    "help",
    "support",
    "kb",
    "academy",
    "login",
    "signin",
    "auth",
    "accounts",
    "news",
    "press",
    "media",
)

# Strong infra/analytics/CDN keywords (checked against host string only)
UNIVERSAL_HOST_KEYWORDS: tuple[str, ...] = (
    "cdn",
    "analytics",
    "telemetry",
    "metrics",
    "tracking",
    "tracker",
    "tagmanager",
    "tagsrv",
    "pixel",
    "adsrv",
    "adservice",
    "statcounter",
    "stats.",
    "trk.",
    ".trk",
    "redirect",
    "logx",
    "logs.",
    "click",
    "short",
    "lnk",
)


class UniversalExternalFilter(URLFilter):
    """
    Drop "universal external" URLs (social/CDN/analytics/docs SaaS/etc.)
    and enforce dataset-level isolation between companies.

    Policy:
      - company domain -> KEEP
      - other dataset company domains -> DROP
      - universal externals -> DROP
      - other externals -> KEEP
    """

    __slots__ = (
        "_universal_domains",
        "_dataset_domains",
        "_company_domain",
        "_debug",
    )

    _DEFAULT_UNIVERSAL_SUFFIXES: frozenset[str] = frozenset(
        {
            # Social / community / media
            "facebook.com",
            "instagram.com",
            "twitter.com",
            "x.com",
            "youtube.com",
            "linkedin.com",
            "tiktok.com",
            "pinterest.com",
            "reddit.com",
            "slack.com",
            "medium.com",
            "substack.com",
            "discord.com",
            "quora.com",
            "trustpilot.com",
            "glassdoor.com",
            "crunchbase.com",
            "spotify.com",
            # Shorteners / trackers
            "t.co",
            "goo.gl",
            "bit.ly",
            "bitly.com",
            "tinyurl.com",
            "ow.ly",
            "buff.ly",
            "rebrand.ly",
            "lnkd.in",
            "mailchi.mp",
            "hubspotlinks.com",
            "adobe.ly",
            "shorturl.at",
            "urlr.me",
            "clickfunnels.com",
            # App stores
            "apple.com",
            "play.google.com",
            "appgallery.huawei.com",
            "galaxy.store",
            "microsoft.com",
            "apps.microsoft.com",
            # Identity providers
            "accounts.google.com",
            "login.microsoftonline.com",
            "auth0.com",
            "okta.com",
            "onelogin.com",
            "pingidentity.com",
            "appleid.apple.com",
            "login.salesforce.com",
            # Docs / KB SaaS
            "intercom.help",
            "zendesk.com",
            "freshdesk.com",
            "helpscoutdocs.com",
            "readme.io",
            "gitbook.io",
            "document360.io",
            "service-now.com",
            "notion.site",
            "confluence.com",
            "knowledgeowl.com",
            "zoho.com",
            "helpjuice.com",
            "deskera.com",
            # Atlassian cloud
            "atlassian.net",
            "statuspage.io",
            "jira.com",
            "confluence.net",
            # Dev hosting / code forges
            "github.com",
            "gitlab.com",
            "bitbucket.org",
            "sourceforge.net",
            "stackblitz.com",
            "codesandbox.io",
            "replit.com",
            # Google properties (non-product)
            "developers.google.com",
            "support.google.com",
            "docs.google.com",
            "drive.google.com",
            "forms.gle",
            "maps.google.com",
            "maps.app.goo.gl",
            "calendar.google.com",
            "photos.google.com",
            "fonts.googleapis.com",
            "maps.googleapis.com",
            # General news / media
            "newsweek.com",
            "cnn.com",
            "bbc.co.uk",
            "reuters.com",
            "bloomberg.com",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
            "forbes.com",
            "yahoo.com",
            "nbcnews.com",
            "cnbc.com",
            "businessinsider.com",
            "techcrunch.com",
            # Status / uptime / monitoring
            "status.io",
            "statuspal.io",
            "uptime.com",
            "pingdom.com",
            "betteruptime.com",
            "freshstatus.io",
            # E-commerce / hosting / site builders
            "shopify.com",
            "myshopify.com",
            "bigcommerce.com",
            "magento.com",
            "woocommerce.com",
            "wix.com",
            "squarespace.com",
            "weebly.com",
            "webflow.io",
            "wordpress.com",
            "godaddy.com",
            "strikingly.com",
            "ecwid.com",
            "prestashop.com",
            "3dcart.com",
            "shopbase.com",
            "shift4shop.com",
            "lightspeedhq.com",
            "zohocommerce.com",
            "shop.app",
            "framer.com",
            "squarespace-cdn.com",
            # Marketing / CRM / analytics / ads
            "mailchimp.com",
            "campaignmonitor.com",
            "activecampaign.com",
            "hubspot.com",
            "salesforce.com",
            "pardot.com",
            "marketo.com",
            "keap.com",
            "sendgrid.com",
            "constantcontact.com",
            "convertkit.com",
            "drip.com",
            "klaviyo.com",
            "adobe.com",
            "doubleclick.net",
            "google-analytics.com",
            "analytics.google.com",
            "tagmanager.google.com",
            "mixpanel.com",
            "segment.com",
            "hotjar.com",
            "fullstory.com",
            "crazyegg.com",
            "facebook.net",
            "fbcdn.net",
            "twitteranalytics.com",
            # CDN / asset / security / storage / fonts / icons
            "cloudflare.com",
            "cdn.cloudflare.net",
            "akamaihd.net",
            "akamaized.net",
            "fastly.net",
            "vercel.app",
            "netlify.app",
            "edgekey.net",
            "edgesuite.net",
            "stackpathcdn.com",
            "azureedge.net",
            "firebaseapp.com",
            "cloudfront.net",
            "storage.googleapis.com",
            "ajax.googleapis.com",
            "s1.wp.com",
            "s2.wp.com",
            "gmpg.org",
            "kit.fontawesome.com",
            "fonts.gstatic.com",
            "fonts.bunny.net",
            "stats.wp.com",
            "wp.me",
            # Monitoring / Ads
            "criteo.com",
            "crwdcntrl.net",
            "adsrvr.org",
            "appsflyer.com",
            "useinsider.com",
            "newrelic.com",
            "dynatrace.com",
            # Plugin / SaaS widget hosts
            "pages.dev",
            "list-manage.com",
            "statcounter.com",
            "s.gravatar.com",
            "googleadservices.com",
            "hubspot.net",
            # Payment / checkout
            "stripe.com",
            "paypal.com",
            "braintreepayments.com",
            "squareup.com",
            "adyen.com",
            "authorize.net",
            "klarna.com",
            "afterpay.com",
            "affirm.com",
            # Booking / ticketing / delivery
            "eventbrite.com",
            "ticketmaster.com",
            "opentable.com",
            "resy.com",
            "doordash.com",
            "ubereats.com",
            "postmates.com",
            "grubhub.com",
            # Survey / forms / scheduling
            "surveymonkey.com",
            "typeform.com",
            "jotform.com",
            "googleforms.com",
            "calendly.com",
            "doodle.com",
            "wufoo.com",
            # Corporate/standards/directories
            "google.com",
            "amazon.com",
            "aws.amazon.com",
            "adobe.io",
            "creativecommons.org",
            "ietf.org",
            "w3.org",
            "archive.org",
            "who.int",
            "un.org",
            "europa.eu",
            "g2.com",
            "capterra.com",
            "producthunt.com",
            # Government
            "canada.ca",
            "priv.gc.ca",
        }
    )

    def __init__(
        self,
        *,
        universal_suffixes: Optional[Iterable[str]] = None,
        dataset_externals: Optional[Iterable[str]] = None,
        company_url: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        try:
            super().__init__(name=name or "UniversalExternalFilter")
        except TypeError:  # pragma: no cover
            super().__init__()
            try:
                self.name = name or "UniversalExternalFilter"  # type: ignore[attr-defined]
            except Exception:
                pass

        self._debug = self.logger.isEnabledFor(logging.DEBUG)

        self._universal_domains: frozenset[str] = frozenset()
        self._dataset_domains: frozenset[str] = frozenset()
        self._company_domain: str = ""

        if universal_suffixes is None:
            universal_suffixes = self._DEFAULT_UNIVERSAL_SUFFIXES
        self.update_universal_suffixes(universal_suffixes, replace=True)

        if dataset_externals:
            self.update_dataset_externals(dataset_externals, replace=True)

        if company_url:
            self.set_company_url(company_url)

        if self._debug:
            self.logger.debug(
                "[UniversalExternalFilter.__init__] universal=%d dataset=%d company_domain=%s",
                len(self._universal_domains),
                len(self._dataset_domains),
                self._company_domain,
            )

    # ---------------------------
    # Configuration helpers
    # ---------------------------

    def set_company_url(self, company_url: str) -> None:
        h = _strip_www(_fast_host_from_url(company_url))
        self._company_domain = _registrable_domain(h) or h
        self._classify_host.cache_clear()
        if self._debug:
            self.logger.debug(
                "[UniversalExternalFilter.set_company_url] company_url=%s company_domain=%s",
                company_url,
                self._company_domain,
            )

    def update_universal_suffixes(
        self, suffixes: Iterable[str], *, replace: bool = False
    ) -> None:
        # Stable cache key to maximize cache hits across runs/instances
        key = tuple(
            sorted({str(s).strip() for s in (suffixes or ()) if str(s).strip()})
        )
        normalized = _normalized_universal_domains_cache(key) if key else frozenset()

        if replace:
            self._universal_domains = normalized
        else:
            self._universal_domains = frozenset(
                set(self._universal_domains).union(normalized)
            )

        self._classify_host.cache_clear()
        if self._debug:
            self.logger.debug(
                "[UniversalExternalFilter.update_universal_suffixes] replace=%s now=%d",
                replace,
                len(self._universal_domains),
            )

    def update_dataset_externals(
        self, hosts_or_domains: Iterable[str], *, replace: bool = False
    ) -> None:
        key = tuple(
            sorted({str(s).strip() for s in (hosts_or_domains or ()) if str(s).strip()})
        )
        normalized = _normalized_dataset_domains_cache(key) if key else frozenset()

        if replace:
            self._dataset_domains = normalized
        else:
            self._dataset_domains = frozenset(
                set(self._dataset_domains).union(normalized)
            )

        self._classify_host.cache_clear()
        if self._debug:
            self.logger.debug(
                "[UniversalExternalFilter.update_dataset_externals] replace=%s now=%d",
                replace,
                len(self._dataset_domains),
            )

    # ---------------------------
    # Core logic
    # ---------------------------

    @lru_cache(maxsize=16384)
    def _classify_host(self, host: str) -> str:
        """
        Return one of: "company" | "dataset" | "universal" | "other"
        """
        h = _strip_www(_strip_dot_lower(host))
        if not h:
            return "other"

        rd = _registrable_domain(h)

        # 1) Current company always wins
        if rd and self._company_domain and rd == self._company_domain:
            return "company"

        # 2) Other dataset companies (prevent cross-company bleed)
        if rd and rd in self._dataset_domains:
            return "dataset"

        # 3) Government domains are universal externals
        if _is_government_domain(h):
            return "universal"

        # 4) Universal externals by registrable domain membership (O(1))
        if rd and rd in self._universal_domains:
            return "universal"

        # 5) Functional prefix: check first label only (no split allocation)
        dot = h.find(".")
        if dot > 0:
            first = h[:dot]
            # Only meaningful if there are at least 3 labels (a.b.c)
            if h.find(".", dot + 1) != -1 and first in FUNCTIONAL_SUBDOMAIN_PREFIXES:
                return "universal"

        # 6) Strong infra keywords in host string
        for kw in UNIVERSAL_HOST_KEYWORDS:
            if kw in h:
                return "universal"

        return "other"

    def apply(self, url: str) -> bool:
        # Fast-path: get host without heavy parsing.
        host = _fast_host_from_url(url)
        if not host:
            self._update_stats(False)
            if self._debug:
                self.logger.debug(
                    "[UniversalExternalFilter.apply] url=%s host='' -> DROP", url
                )
            return False

        cls = self._classify_host(host)
        if cls == "company":
            res = True
        elif cls in ("dataset", "universal"):
            res = False
        else:
            res = True

        self._update_stats(res)

        if self._debug:
            self.logger.debug(
                "[UniversalExternalFilter.apply] url=%s host=%s classification=%s -> %s",
                url,
                host,
                cls,
                "KEEP" if res else "DROP",
            )
        return res


# =============================================================================
# HTMLContentFilter
# =============================================================================


class HTMLContentFilter(URLFilter):
    """
    Keep URLs likely to be HTML over HTTP(S).

    Performance notes:
      - Early string-based drop for 'mailto:' and 'tel:' BEFORE any urlsplit().
      - Then parse scheme/path once (urlsplit).
      - Extension check uses last path segment only.
    """

    __slots__ = ("_reject_exts", "_debug")

    _DEFAULT_REJECT_EXTS: frozenset[str] = frozenset(
        {
            # Docs / office
            "pdf",
            "ps",
            "rtf",
            "doc",
            "docx",
            "ppt",
            "pptx",
            "pps",
            "xls",
            "xlsx",
            "csv",
            "tsv",
            "xml",
            "json",
            "yml",
            "yaml",
            "md",
            "rst",
            # Archives / binaries
            "zip",
            "rar",
            "7z",
            "gz",
            "bz2",
            "xz",
            "tar",
            "tgz",
            "dmg",
            "exe",
            "msi",
            "apk",
            "iso",
            # Images
            "jpg",
            "jpeg",
            "png",
            "gif",
            "svg",
            "webp",
            "bmp",
            "tif",
            "tiff",
            "ico",
            # Audio / video
            "mp3",
            "wav",
            "m4a",
            "ogg",
            "flac",
            "aac",
            "mp4",
            "m4v",
            "webm",
            "mov",
            "avi",
            "wmv",
            "mkv",
            # Fonts / maps
            "eot",
            "ttf",
            "otf",
            "woff",
            "woff2",
            "map",
        }
    )

    def __init__(
        self,
        *,
        reject_exts: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        try:
            super().__init__(name=name or "HTMLContentFilter")
        except TypeError:  # pragma: no cover
            super().__init__()
            try:
                self.name = name or "HTMLContentFilter"  # type: ignore[attr-defined]
            except Exception:
                pass

        self._debug = self.logger.isEnabledFor(logging.DEBUG)

        src = reject_exts if reject_exts is not None else self._DEFAULT_REJECT_EXTS
        self._reject_exts = frozenset(
            {str(e).lower().lstrip(".") for e in src if str(e).strip()}
        )

    @staticmethod
    def _extract_extension(path: str) -> str:
        # Only the final segment matters
        seg = (path or "").rsplit("/", 1)[-1]
        if "." not in seg:
            return ""
        return seg.rsplit(".", 1)[-1].lower()

    def apply(self, url: str) -> bool:
        try:
            p = urlsplit(url.strip())
            scheme = (p.scheme or "").lower()
            path = p.path or ""
        except Exception:
            # Parse failed: be conservative (DROP)
            self._update_stats(False)
            if self._debug:
                self.logger.debug(
                    "[HTMLContentFilter.apply] url=%s -> DROP (parse_error)", url
                )
            return False

        # Non-http(s) schemes are not HTML pages
        if scheme and scheme not in ("http", "https"):
            self._update_stats(False)
            if self._debug:
                self.logger.debug(
                    "[HTMLContentFilter.apply] url=%s scheme=%s -> DROP (non-http)",
                    url,
                    scheme,
                )
            return False

        ext = self._extract_extension(path)
        if not ext:
            self._update_stats(True)
            return True

        res = ext not in self._reject_exts
        self._update_stats(res)
        return res


# =============================================================================
# LanguageAwareURLFilter
# =============================================================================


class LanguageAwareURLFilter(URLFilter):
    """
    Filter URLs using ONLY language-related rules from configs.language.

    - TLD allow/deny per language
    - Host suffix allow/deny per language
    - Path language tokens (segment match) per language
    """

    __slots__ = (
        "_lang_code",
        "_allowed_tlds",
        "_blocked_tlds",
        "_allowed_host_suffixes",
        "_blocked_host_suffixes",
        "_token_to_lang",
        "_debug",
    )

    def __init__(
        self,
        *,
        lang_code: str = "en",
        allowed_tlds: Optional[Iterable[str]] = None,
        blocked_tlds: Optional[Iterable[str]] = None,
        allowed_host_suffixes: Optional[Iterable[str]] = None,
        blocked_host_suffixes: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        **_ignored: object,  # backward-compat for old include/exclude patterns
    ) -> None:
        lang = (lang_code or "en").lower()

        try:
            super().__init__(name=name or f"LanguageAwareURLFilter[{lang}]")
        except TypeError:  # pragma: no cover
            super().__init__()
            try:
                self.name = name or f"LanguageAwareURLFilter[{lang}]"  # type: ignore[attr-defined]
            except Exception:
                pass

        self._debug = self.logger.isEnabledFor(logging.DEBUG)
        self._lang_code = lang

        spec: Dict[str, object] = lang_cfg.get_lang_spec(lang)

        spec_allow_tld = spec.get("LANG_TLD_ALLOW", {}) or {}  # type: ignore[assignment]
        spec_deny_tld = spec.get("LANG_TLD_DENY", {}) or {}  # type: ignore[assignment]

        allow_src = (
            allowed_tlds
            if allowed_tlds is not None
            else (
                spec_allow_tld.get(lang, []) if isinstance(spec_allow_tld, dict) else []
            )
        )
        deny_src = (
            blocked_tlds
            if blocked_tlds is not None
            else (
                spec_deny_tld.get(lang, []) if isinstance(spec_deny_tld, dict) else []
            )
        )

        self._allowed_tlds = frozenset(
            {str(t).lower().lstrip(".") for t in (allow_src or []) if str(t).strip()}
        )
        self._blocked_tlds = frozenset(
            {str(t).lower().lstrip(".") for t in (deny_src or []) if str(t).strip()}
        )

        spec_allow_host = spec.get("LANG_HOST_ALLOW_SUFFIXES", {}) or {}  # type: ignore[assignment]
        spec_block_host = spec.get("LANG_HOST_BLOCK_SUFFIXES", {}) or {}  # type: ignore[assignment]

        allow_host_src = (
            allowed_host_suffixes
            if allowed_host_suffixes is not None
            else (
                spec_allow_host.get(lang, [])
                if isinstance(spec_allow_host, dict)
                else []
            )
        )
        block_host_src = (
            blocked_host_suffixes
            if blocked_host_suffixes is not None
            else (
                spec_block_host.get(lang, [])
                if isinstance(spec_block_host, dict)
                else []
            )
        )

        # Keep as tuples for fast iteration (small lists), normalized and stripped.
        self._allowed_host_suffixes = tuple(
            sorted(
                {_strip_dot_lower(h) for h in (allow_host_src or []) if str(h).strip()},
                key=len,
                reverse=True,
            )
        )
        self._blocked_host_suffixes = tuple(
            sorted(
                {_strip_dot_lower(h) for h in (block_host_src or []) if str(h).strip()},
                key=len,
                reverse=True,
            )
        )

        # Path language tokens: build token -> lang mapping for O(segments) scan
        raw_path_tokens = spec.get("PATH_LANG_TOKENS", {}) or {}  # type: ignore[assignment]
        token_to_lang: Dict[str, str] = {}
        if isinstance(raw_path_tokens, dict):
            for code, toks in raw_path_tokens.items():
                c = str(code).lower()
                for tok in toks or []:
                    t = str(tok).lower()
                    if t and t not in token_to_lang:
                        token_to_lang[t] = c
        self._token_to_lang = token_to_lang

    def _host_allowed_by_suffix(self, host_norm: str) -> bool:
        if not self._allowed_host_suffixes:
            return True
        for sfx in self._allowed_host_suffixes:
            if _suffix_match(host_norm, sfx):
                return True
        return False

    def _host_blocked_by_suffix(self, host_norm: str) -> bool:
        if not self._blocked_host_suffixes:
            return False
        for sfx in self._blocked_host_suffixes:
            if _suffix_match(host_norm, sfx):
                return True
        return False

    def _tld_allowed(self, host_norm: str) -> bool:
        tld = _top_level_label(host_norm)
        if self._allowed_tlds and tld not in self._allowed_tlds:
            return False
        if self._blocked_tlds and tld in self._blocked_tlds:
            return False
        return True

    def _detect_path_language(self, path: str) -> Optional[str]:
        if not self._token_to_lang:
            return None
        # Split only once; segment checks are dict lookups.
        for seg in (path or "").split("/"):
            if not seg:
                continue
            code = self._token_to_lang.get(seg.lower())
            if code:
                return code
        return None

    def apply(self, url: str) -> bool:
        # NOTE: we rely on urlsplit for correctness here; language rules need host+path.
        try:
            p = urlsplit(url.strip())
            host = _strip_dot_lower(p.hostname or "")
            path = p.path or "/"
        except Exception:
            self._update_stats(False)
            if self._debug:
                self.logger.debug(
                    "[LanguageAwareURLFilter.apply] url=%s -> DROP (parse_error)", url
                )
            return False

        if not host:
            self._update_stats(False)
            return False

        # Host suffix rules
        if self._host_blocked_by_suffix(host):
            self._update_stats(False)
            return False

        if not self._host_allowed_by_suffix(host):
            self._update_stats(False)
            return False

        # TLD rules
        if not self._tld_allowed(host):
            self._update_stats(False)
            return False

        # Path language mismatch
        detected = self._detect_path_language(path)
        if detected is not None and detected != self._lang_code:
            self._update_stats(False)
            return False

        self._update_stats(True)
        return True


# =============================================================================
# FirstTimeURLFilter
# =============================================================================


class FirstTimeURLFilter(URLFilter):
    """Accept a URL only the first time it is seen."""

    __slots__ = ("seen_urls", "_debug")

    def __init__(self) -> None:
        try:
            super().__init__(name="FirstTimeURLFilter")
        except TypeError:  # pragma: no cover
            super().__init__()
            try:
                self.name = "FirstTimeURLFilter"  # type: ignore[attr-defined]
            except Exception:
                pass

        self._debug = self.logger.isEnabledFor(logging.DEBUG)
        self.seen_urls: Set[str] = set()

    def apply(self, url: str) -> bool:
        if url in self.seen_urls:
            self._update_stats(False)
            if self._debug:
                self.logger.debug(
                    "[FirstTimeURLFilter.apply] url=%s -> DROP (seen)", url
                )
            return False

        self.seen_urls.add(url)
        self._update_stats(True)
        if self._debug:
            self.logger.debug("[FirstTimeURLFilter.apply] url=%s -> KEEP (first)", url)
        return True
