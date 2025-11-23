from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from crawl4ai.deep_crawling.filters import URLFilter

from configs import language as lang_cfg

__all__ = [
    "UniversalExternalFilter",
    "HTMLContentFilter",
    "LanguageAwareURLFilter",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Host / domain helpers
# --------------------------------------------------------------------------- #


def _normalize_host(value: str) -> str:
    """Best-effort host normalizer (module-level helper)."""
    try:
        if not value:
            return ""
        value = value.strip()
        if "://" in value:
            return (urlparse(value).hostname or "").lower()
        return value.lower()
    except Exception:
        return ""


def _host_labels(host: str) -> List[str]:
    host = (host or "").strip().strip(".").lower()
    return [p for p in host.split(".") if p]


# Multi-label public suffixes (for eTLD+1 registrable domain logic)
_MULTI_LABEL_PUBLIC_SUFFIXES: Set[str] = {
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


def _registrable_domain(host: str) -> str:
    """
    Lightweight eTLD+1-ish: return registrable apex (example.co.uk, example.com, etc.).
    Handles common multi-label public suffixes.
    """
    labels = _host_labels(host)
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]

    last2 = ".".join(labels[-2:])
    last3 = ".".join(labels[-3:]) if len(labels) >= 3 else ""

    if last2 in _MULTI_LABEL_PUBLIC_SUFFIXES and last3:
        return last3
    return last2


# Common government-style second-level labels (combined with ccTLDs)
_GOV_SECOND_LEVELS: Set[str] = {"gov", "gouv", "go", "govt"}


def _is_government_domain(host: str) -> bool:
    """
    Heuristic: treat any registrable domain like:
      - foo.gov
      - foo.gov.<cc>
      - foo.gouv.<cc>
      - foo.go.<cc>
      - foo.govt.<cc>
    as "government".
    """
    rd = _registrable_domain(host)
    if not rd:
        return False

    labels = rd.split(".")
    if len(labels) < 2:
        return False

    if labels[-1] == "gov":
        return True

    if len(labels) >= 2:
        sec = labels[-2]
        if sec in _GOV_SECOND_LEVELS:
            return True

    return False


def _label_boundary_match_normalized(host_norm: str, suffix: str) -> bool:
    """
    Host-level suffix match with label boundary protection.
    Assumes `host_norm` is already normalized and stripped.
    """
    h = (host_norm or "").strip().strip(".").lower()
    s = (suffix or "").strip().strip(".").lower()
    if not h or not s:
        return False
    return h == s or h.endswith("." + s)


def _label_boundary_match(host: str, suffix: str) -> bool:
    """
    Host-level suffix match with label boundary protection.
    This variant accepts arbitrary host strings and normalizes internally.
    """
    return _label_boundary_match_normalized(_normalize_host(host), suffix)


def _top_level_label(host: str) -> str:
    """
    Simple TLD extractor: last label of the host.
    """
    labels = _host_labels(host)
    return labels[-1] if labels else ""


# Functional subdomain prefixes (status, docs, support, etc.)
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


def _has_functional_prefix(host: str) -> bool:
    """
    Return True if the host clearly looks like a docs/status/help/infra subdomain.
    """
    h = _normalize_host(host)
    if not h:
        return False
    labels = _host_labels(h)
    if len(labels) < 3:
        return False
    return labels[0] in FUNCTIONAL_SUBDOMAIN_PREFIXES


# Additional heuristic host keywords that almost always mean "universal external"
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
    "logx",
    "logs.",
    "statcounter",
    "stats.",
    "short",
    "lnk",
    "click",
    "trk.",
    ".trk",
    "redirect",
)


# --------------------------------------------------------------------------- #
# Normalization caches for heavy, shared sets
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=32)
def _normalized_universal_domains_cache(suffixes: Tuple[str, ...]) -> frozenset[str]:
    """
    Normalize a collection of universal suffixes to registrable domains.

    Cached so that repeated UniversalExternalFilter instances sharing the same
    suffix set don't recompute normalization.
    """
    normalized: Set[str] = set()
    for s in suffixes:
        h = _normalize_host(s)
        if not h:
            continue
        rd = _registrable_domain(h)
        if rd:
            normalized.add(rd)
    return frozenset(normalized)


@lru_cache(maxsize=64)
def _normalized_dataset_domains_cache(hosts: Tuple[str, ...]) -> frozenset[str]:
    """
    Normalize a collection of dataset external hosts to registrable domains.

    Cached so that repeated UniversalExternalFilter instances sharing the same
    dataset list don't recompute normalization.
    """
    normalized: Set[str] = set()
    for s in hosts:
        h = _normalize_host(s)
        if not h:
            continue
        rd = _registrable_domain(h)
        if rd:
            normalized.add(rd)
    return frozenset(normalized)


# --------------------------------------------------------------------------- #
# UniversalExternalFilter
# --------------------------------------------------------------------------- #


class UniversalExternalFilter(URLFilter):
    """
    Drop "universal external" URLs (social, CDNs, analytics, docs SaaS, etc.)
    and enforce dataset-level isolation between companies.

    Semantics:

      - We maintain two disjoint notions of "externals":
        * universal externals: infra/social/tracking/etc. (hard-coded suffixes,
          govt domains, functional prefixes, infra keywords).
        * dataset externals: registrable domains that *belong* to the dataset
          of company URLs (from DatasetExternals).

      - For a given company, identified by its registrable domain:

        * That company's own domain       -> ALWAYS kept.
        * Other dataset company domains   -> ALWAYS dropped (prevent cross-company bleed).
        * Universal external domains      -> ALWAYS dropped.
        * Everything else (non-dataset, non-universal) -> kept, so that a
          separate policy can decide how to handle arbitrary externals.
    """

    __slots__ = (
        "_universal_domains",
        "_dataset_domains",
        "_company_domain",
    )

    _DEFAULT_UNIVERSAL_SUFFIXES: Set[str] = {
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
        "hubspot.com",
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

    def __init__(
        self,
        *,
        universal_suffixes: Optional[Iterable[str]] = None,
        dataset_externals: Optional[Iterable[str]] = None,
        company_url: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "UniversalExternalFilter")

        self._universal_domains: Set[str] = set()
        self._dataset_domains: Set[str] = set()
        self._company_domain: str = ""

        if universal_suffixes is None:
            universal_suffixes = self._DEFAULT_UNIVERSAL_SUFFIXES

        self.update_universal_suffixes(universal_suffixes, replace=True)

        if dataset_externals:
            self.update_dataset_externals(dataset_externals, replace=True)

        if company_url:
            self.set_company_url(company_url)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[UniversalExternalFilter.__init__] universal=%d dataset=%d company_domain=%s",
                len(self._universal_domains),
                len(self._dataset_domains),
                self._company_domain,
            )

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #

    def set_company_url(self, company_url: str) -> None:
        """
        Set the current company's canonical URL (or host) so we can distinguish:

          - that company's own registrable domain (keep),
          - other dataset company domains (drop),
          - pure universal externals (drop).
        """
        h = _normalize_host(company_url)
        if not h:
            self._company_domain = ""
        else:
            rd = _registrable_domain(h)
            self._company_domain = rd or h

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[UniversalExternalFilter.set_company_url] company_url=%s company_domain=%s",
                company_url,
                self._company_domain,
            )

        # Changing company domain affects classification cache
        self._classify_host.cache_clear()

    def update_universal_suffixes(
        self,
        suffixes: Iterable[str],
        *,
        replace: bool = False,
    ) -> None:
        """
        Update universal external suffixes.

        The expensive normalization step is cached across instances when the
        same suffix collection is provided (e.g., default set).
        """
        seq = tuple(suffixes or ())
        if not seq:
            normalized_set: Set[str] = set()
        else:
            normalized_set = set(_normalized_universal_domains_cache(seq))

        if replace:
            self._universal_domains = normalized_set
        else:
            self._universal_domains.update(normalized_set)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[UniversalExternalFilter.update_universal_suffixes] replace=%s now=%d",
                replace,
                len(self._universal_domains),
            )

        self._classify_host.cache_clear()

    def update_dataset_externals(
        self,
        hosts_or_domains: Iterable[str],
        *,
        replace: bool = False,
    ) -> None:
        """
        Update the set of registrable domains that *belong* to the dataset
        of company URLs. These represent other companies whose URLs we want
        to treat as cross-company externals.

        Heavy normalization work is cached across instances when the same
        dataset_externals collection is reused.
        """
        seq = tuple(hosts_or_domains or ())
        if not seq:
            normalized_set: Set[str] = set()
        else:
            normalized_set = set(_normalized_dataset_domains_cache(seq))

        if replace:
            self._dataset_domains = normalized_set
        else:
            self._dataset_domains.update(normalized_set)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[UniversalExternalFilter.update_dataset_externals] replace=%s now=%d",
                replace,
                len(self._dataset_domains),
            )

        self._classify_host.cache_clear()

    # ------------------------------------------------------------------ #
    # Core logic
    # ------------------------------------------------------------------ #

    @lru_cache(maxsize=8192)
    def _classify_host(self, host: str) -> str:
        """
        Return one of:
          - "company"   : host belongs to the current company domain
          - "dataset"   : host belongs to another dataset company domain
          - "universal" : social/infra/analytics/etc., or government
          - "other"     : anything else (non-dataset, non-universal)
        """
        h = _normalize_host(host)
        if not h:
            return "other"

        rd = _registrable_domain(h)

        # 1) Current company's own registrable domain
        if rd and self._company_domain and rd == self._company_domain:
            return "company"

        # 2) Known dataset company domains (belong to dataset but NOT this company)
        if rd and rd in self._dataset_domains:
            return "dataset"

        # 3) Government domains are always treated as universal externals
        if _is_government_domain(h):
            return "universal"

        # 4) Hard universal suffixes
        for sfx in self._universal_domains:
            if not sfx:
                continue
            if rd == sfx or _label_boundary_match_normalized(h, sfx):
                return "universal"

        # 5) Functional subdomain prefixes (status/docs/support/etc.)
        if _has_functional_prefix(h):
            return "universal"

        # 6) Strong infra/analytics/CDN keywords
        lowered = ".".join(_host_labels(h))
        for kw in UNIVERSAL_HOST_KEYWORDS:
            if kw in lowered:
                return "universal"

        return "other"

    def apply(self, url: str) -> bool:
        """
        Decide whether to KEEP (True) or DROP (False) a URL.

        Policy:

          - "company"   -> keep
          - "dataset"   -> drop  (other dataset companies)
          - "universal" -> drop  (social/CDN/analytics/etc.)
          - "other"     -> keep  (non-dataset externals; handled elsewhere)
        """
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
        except Exception:
            host = ""
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[UniversalExternalFilter.apply] url=%s host_parse_error -> DROP",
                    url,
                )
            self._update_stats(result)
            return result

        classification = self._classify_host(host)

        if classification == "company":
            result = True
            reason = "company_domain"
        elif classification in ("dataset", "universal"):
            result = False
            reason = classification
        else:
            # "other" -> non-dataset externals; allowed so a higher-level policy
            # can decide what to do with them.
            result = True
            reason = "other_external"

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[UniversalExternalFilter.apply] url=%s host=%s classification=%s reason=%s -> %s",
                url,
                host,
                classification,
                reason,
                "KEEP" if result else "DROP",
            )

        self._update_stats(result)
        return result


# --------------------------------------------------------------------------- #
# HTMLContentFilter
# --------------------------------------------------------------------------- #


class HTMLContentFilter(URLFilter):
    """
    Only keep URLs likely to be HTML over HTTP(S).

    Strategy:
      - If scheme is non-HTTP(S) (mailto, tel, javascript, etc.) → DROP.
      - If no extension → assume HTML → KEEP.
      - If extension in known non-HTML list → DROP.
      - Otherwise → KEEP (php/asp/jsp/etc).
    """

    __slots__ = ("_reject_exts",)

    _DEFAULT_REJECT_EXTS: Set[str] = {
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

    def __init__(
        self,
        *,
        reject_exts: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "HTMLContentFilter")
        reject_exts = reject_exts or self._DEFAULT_REJECT_EXTS
        self._reject_exts = {e.lower().lstrip(".") for e in reject_exts}

    @staticmethod
    def _extract_extension_from_path(path: str) -> str:
        segment = (path or "").rsplit("/", 1)[-1]
        if "." not in segment:
            return ""
        ext = segment.rsplit(".", 1)[-1].lower()
        return ext

    def apply(self, url: str) -> bool:
        # First, drop non-HTTP(S) schemes like mailto:, tel:, javascript:, data:, etc.
        try:
            parsed = urlparse(url)
            scheme = (parsed.scheme or "").lower()
            path = parsed.path or ""
        except Exception:
            scheme = ""
            path = url or ""

        if scheme and scheme not in ("http", "https"):
            # Explicitly DROP mailto:/tel:/javascript:/data:/...
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[HTMLContentFilter.apply] url=%s scheme=%s -> DROP (non-http scheme)",
                    url,
                    scheme,
                )
            self._update_stats(result)
            return result

        ext = self._extract_extension_from_path(path)

        if not ext:
            result = True
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[HTMLContentFilter.apply] url=%s ext='' -> KEEP (no extension)",
                    url,
                )
            self._update_stats(result)
            return result

        if ext in self._reject_exts:
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[HTMLContentFilter.apply] url=%s ext=%s -> DROP (non-HTML)",
                    url,
                    ext,
                )
            self._update_stats(result)
            return result

        result = True
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[HTMLContentFilter.apply] url=%s ext=%s -> KEEP",
                url,
                ext,
            )
        self._update_stats(result)
        return result


# --------------------------------------------------------------------------- #
# LanguageAwareURLFilter
# --------------------------------------------------------------------------- #


class LanguageAwareURLFilter(URLFilter):
    """
    Filter URLs using ONLY language-related rules from configs.language.

    Behavior:
      - Applies TLD allow/deny per language:
            LANG_TLD_ALLOW[lang] -> allowed TLDs (e.g., {"com","jp"})
            LANG_TLD_DENY[lang]  -> blocked TLDs
      - Applies host suffix allow/deny:
            LANG_HOST_ALLOW_SUFFIXES[lang] -> host suffixes to KEEP
            LANG_HOST_BLOCK_SUFFIXES[lang] -> host suffixes to DROP
      - Uses PATH_LANG_TOKENS to detect explicit language codes in URL paths
        (e.g. /ja/, /en-us/) and drops URLs whose path-language conflicts
        with the active lang_code.

    NOTE:
      - DEFAULT_INCLUDE_PATTERNS / DEFAULT_EXCLUDE_PATTERNS are NO LONGER USED.
      - include_patterns / exclude_patterns constructor arguments are accepted
        for backward compatibility but are completely ignored.
    """

    __slots__ = (
        "_lang_code",
        "_allowed_tlds",
        "_blocked_tlds",
        "_allowed_host_suffixes",
        "_blocked_host_suffixes",
        "_path_lang_tokens",  # global mapping of path language tokens
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
    ) -> None:
        super().__init__(name=name or f"LanguageAwareURLFilter[{lang_code}]")

        self._lang_code = (lang_code or "en").lower()
        # Load the full language spec (base + lang_<code>.py overlay)
        spec: Dict[str, any] = lang_cfg.get_lang_spec(self._lang_code)

        # TLD rules are stored as language → {tlds}
        spec_allow_tld: Dict[str, Iterable[str]] = spec.get("LANG_TLD_ALLOW", {}) or {}
        spec_deny_tld: Dict[str, Iterable[str]] = spec.get("LANG_TLD_DENY", {}) or {}

        allow_src = (
            allowed_tlds
            if allowed_tlds is not None
            else spec_allow_tld.get(self._lang_code, [])
        )
        deny_src = (
            blocked_tlds
            if blocked_tlds is not None
            else spec_deny_tld.get(self._lang_code, [])
        )

        self._allowed_tlds: Set[str] = {t.lower().lstrip(".") for t in (allow_src or [])}
        self._blocked_tlds: Set[str] = {t.lower().lstrip(".") for t in (deny_src or [])}

        # Host suffix rules are likewise language → {suffixes}
        spec_allow_host: Dict[str, Iterable[str]] = spec.get(
            "LANG_HOST_ALLOW_SUFFIXES", {}
        ) or {}
        spec_block_host: Dict[str, Iterable[str]] = spec.get(
            "LANG_HOST_BLOCK_SUFFIXES", {}
        ) or {}

        allow_host_src = (
            allowed_host_suffixes
            if allowed_host_suffixes is not None
            else spec_allow_host.get(self._lang_code, [])
        )
        block_host_src = (
            blocked_host_suffixes
            if blocked_host_suffixes is not None
            else spec_block_host.get(self._lang_code, [])
        )

        self._allowed_host_suffixes: Set[str] = {
            h.lower().strip(".") for h in (allow_host_src or [])
        }
        self._blocked_host_suffixes: Set[str] = {
            h.lower().strip(".") for h in (block_host_src or [])
        }

        # Path language tokens from language config (global mapping)
        raw_path_tokens: Dict[str, Iterable[str]] = spec.get("PATH_LANG_TOKENS", {}) or {}
        self._path_lang_tokens: Dict[str, Set[str]] = {
            code.lower(): {str(v).lower() for v in vals}
            for code, vals in raw_path_tokens.items()
        }

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[LanguageAwareURLFilter.__init__] lang=%s "
                "allowed_tlds=%s blocked_tlds=%s allowed_host_suffixes=%s "
                "blocked_host_suffixes=%s path_lang_tokens_keys=%s",
                self._lang_code,
                sorted(list(self._allowed_tlds))[:10],
                sorted(list(self._blocked_tlds))[:10],
                sorted(list(self._allowed_host_suffixes))[:10],
                sorted(list(self._blocked_host_suffixes))[:10],
                sorted(list(self._path_lang_tokens.keys()))[:10],
            )

    # ------------------------------------------------------------------ #
    # Core helpers (language-only)
    # ------------------------------------------------------------------ #

    def _host_allowed_by_suffix(self, host: str) -> bool:
        if not self._allowed_host_suffixes:
            return True  # no restriction
        h = _normalize_host(host)
        return any(
            _label_boundary_match_normalized(h, sfx)
            for sfx in self._allowed_host_suffixes
        )

    def _host_blocked_by_suffix(self, host: str) -> bool:
        if not self._blocked_host_suffixes:
            return False
        h = _normalize_host(host)
        return any(
            _label_boundary_match_normalized(h, sfx)
            for sfx in self._blocked_host_suffixes
        )

    def _tld_allowed(self, host: str) -> bool:
        tld = _top_level_label(host).lstrip(".").lower()
        if self._allowed_tlds and tld not in self._allowed_tlds:
            return False
        if self._blocked_tlds and tld in self._blocked_tlds:
            return False
        return True

    def _detect_path_language(self, path: str) -> Optional[str]:
        """
        Inspect URL path segments and try to infer an explicit language tag,
        using PATH_LANG_TOKENS from configs.language.

        Returns:
            canonical language code if found (e.g. "ja", "en"), else None.
        """
        if not self._path_lang_tokens:
            return None

        segments = [seg.lower() for seg in (path or "").split("/") if seg]
        if not segments:
            return None

        for code, tokens in self._path_lang_tokens.items():
            # Exact segment match (e.g., "ja", "ja-jp", "en-us")
            if any(seg in tokens for seg in segments):
                return code

        return None

    # ------------------------------------------------------------------ #
    # Core logic
    # ------------------------------------------------------------------ #

    def apply(self, url: str) -> bool:
        """
        Decide whether to KEEP the URL based purely on language-aware rules.

        Returns:
            True  -> keep / follow
            False -> drop
        """
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            path = parsed.path or "/"
        except Exception:
            host = ""
            path = "/"
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[LanguageAwareURLFilter.apply] url=%s parse_error -> DROP",
                    url,
                )
            self._update_stats(result)
            return result

        # Host-level rules (suffix + TLD)
        if self._host_blocked_by_suffix(host):
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[LanguageAwareURLFilter.apply] url=%s host=%s -> DROP (blocked_host_suffix)",
                    url,
                    host,
                )
            self._update_stats(result)
            return result

        if not self._host_allowed_by_suffix(host):
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[LanguageAwareURLFilter.apply] url=%s host=%s -> DROP (not_allowed_by_suffix)",
                    url,
                    host,
                )
            self._update_stats(result)
            return result

        if not self._tld_allowed(host):
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[LanguageAwareURLFilter.apply] url=%s host=%s -> DROP (tld_not_allowed)",
                    url,
                    host,
                )
            self._update_stats(result)
            return result

        # Path language mismatch (e.g. /ja/... on an "en" crawl)
        detected_lang = self._detect_path_language(path)
        if detected_lang is not None and detected_lang != self._lang_code:
            # Hard drop URLs whose path is clearly in a different language,
            # e.g. lang_code="en" but path looks Japanese (/ja/...).
            result = False
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[LanguageAwareURLFilter.apply] url=%s host=%s path=%s "
                    "-> DROP (path_lang_mismatch detected=%s active=%s)",
                    url,
                    host,
                    path,
                    detected_lang,
                    self._lang_code,
                )
            self._update_stats(result)
            return result

        # Passed all language checks → KEEP
        result = True
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "[LanguageAwareURLFilter.apply] url=%s host=%s path=%s -> KEEP (language_ok)",
                url,
                host,
                path,
            )
        self._update_stats(result)
        return result