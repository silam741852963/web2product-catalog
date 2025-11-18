from __future__ import annotations

from typing import Dict, List, Optional, Iterable
from datetime import datetime, timedelta
from urllib.parse import urlparse

from crawl4ai import BrowserConfig

# -----------------------------
# Header builders
# -----------------------------
def _accept_language_string(lang: str = "en-US") -> str:
    """
    Strong English-forward Accept-Language.
    Example: "en-US,en;q=0.9,en-GB;q=0.8"
    """
    lang = (lang or "en-US").strip()
    primary = lang.split("-")[0]
    # Bias to requested lang, then generic English, then another common English region
    return f"{lang},{primary};q=0.9,en;q=0.8,en-GB;q=0.7"


def _default_accept_header() -> str:
    # Typical HTML-forward accept header that still lets servers behave sanely.
    return "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.1"


def _merge_headers(base: Dict[str, str], extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    out = dict(base or {})
    if extra:
        out.update({k: v for k, v in extra.items() if v is not None})
    return out


# -----------------------------
# Cookie helpers
# -----------------------------
def _cookie(name: str, value: str, url: str, *, secure: bool = True) -> Dict:
    """
    Build a Playwright-compatible cookie dict scoped to a URL.
    """
    # Most banners accept Lax cookies; strict isn't needed.
    return {
        "name": name,
        "value": value,
        "url": url,
        "path": "/",
        "httpOnly": False,
        "secure": bool(secure),
        "sameSite": "Lax",
    }


def _language_cookies(url: str, lang: str) -> List[Dict]:
    """
    Common language keys various frameworks look for.
    Harmless if a site ignores unknown names.
    """
    keys = [
        "locale", "lang", "language", "siteLocale",
        "mk_locale", "MKLocale", "selectedLocale",
        "culture", "i18nLocale",
    ]
    return [_cookie(k, lang, url) for k in keys]


def _region_cookies(url: str, country: str, currency: str) -> List[Dict]:
    """
    Region/currency cookies used by many shops/CDNs.
    """
    country = (country or "").upper()
    currency = (currency or "").upper()
    names = {
        # country selectors
        "country": country,
        "countryCode": country,
        "preferred_country": country,
        "market": country,
        # currency selectors
        "currency": currency,
        "preferred_currency": currency,
        "selectedCurrency": currency,
    }
    return [_cookie(k, v, url) for k, v in names.items() if v]


def _consent_cookies(url: str) -> List[Dict]:
    """
    Conservative, widely-used consent cookies to bypass banners.
    Values are generic but accepted by many OneTrust/Cookiebot setups.
    (These are best-effort; they won't break sites that ignore them.)
    """
    now = datetime.utcnow()
    # OneTrust
    one_trust = [
        _cookie("OptanonConsent",
                f"isIABGlobal=false&datestamp={now:%Y-%m-%dT%H:%M:%S}Z"
                "&version=6.16.0&hosts=&consentId=cb-basic&interactionCount=1"
                "&landingPath=NotLandingPage&groups=C0001:1,C0002:1,C0003:1,C0004:1",
                url),
        _cookie("OptanonAlertBoxClosed", now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), url),
    ]
    # Cookiebot
    cookiebot = [
        _cookie("CookieConsent", '{"stamp":"cb-basic","necessary":true,"preferences":true,"statistics":true,"marketing":true}', url)
    ]
    # CCPA / U.S. privacy (basic allow)
    us_priv = [_cookie("usprivacy", "1---", url)]
    # Generic banner statuses
    generic = [
        _cookie("cookieconsent_status", "allow", url),
        _cookie("cookie_banner", "dismissed", url),
    ]
    return one_trust + cookiebot + us_priv + generic


def _normalize_url_scope(u: str) -> Optional[str]:
    try:
        p = urlparse(u)
        if not p.scheme or not p.netloc:
            return None
        # Scope cookies to the top-level HTTPS origin
        scheme = "https"
        return f"{scheme}://{p.netloc}/"
    except Exception:
        return None


def _build_common_cookies(
    base_urls: Iterable[str],
    *,
    lang: str,
    country: Optional[str],
    currency: Optional[str],
    include_consent: bool,
) -> List[Dict]:
    """
    For each URL, build a robust set of language/region/consent cookies.
    """
    cookies: List[Dict] = []
    for raw in (base_urls or []):
        scope = _normalize_url_scope(raw)
        if not scope:
            continue
        cookies.extend(_language_cookies(scope, lang))
        if country or currency:
            cookies.extend(_region_cookies(scope, country or "", currency or ""))
        if include_consent:
            cookies.extend(_consent_cookies(scope))
    return cookies


# -----------------------------
# Public factory
# -----------------------------
def make_browser_config(
    *,
    lang: str = "en-US",
    headless: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    user_agent: Optional[str] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    proxy_config: Optional[dict] = None,
    persistent: bool = False,
    user_data_dir: Optional[str] = None,
    # cookies
    add_common_cookies_for: Optional[List[str]] = None,
    include_consent_cookies: bool = True,
    region_country: Optional[str] = None,   # e.g., "US"
    region_currency: Optional[str] = None,  # e.g., "USD"
    # headers
    extra_headers: Optional[Dict[str, str]] = None,
    # performance toggles
    light_mode: bool = True,
    text_mode: bool = True,  # disables images in Crawl4AI
    enable_stealth: bool = True,
    ignore_https_errors: bool = True,
) -> BrowserConfig:
    """
    Build a BrowserConfig optimized for HTML/text extraction:
    - Strong English/region preferences
    - Save-Data & DNT hints
    - Optional persistent context (reuse sessions)
    - Optional common cookie set (language/region/consent) per domain
    - Conservative Chromium flags to reduce background work
    """
    # 1) HTTP headers
    headers_base = {
        "Accept": _default_accept_header(),
        "Accept-Language": _accept_language_string(lang),
        # Ask servers/CDNs to minimize payloads where honored
        "Save-Data": "on",
        "DNT": "1",
        # Some servers key on this for main doc navigations
        "Upgrade-Insecure-Requests": "1",
    }
    headers = _merge_headers(headers_base, extra_headers)

    # 2) Optional cookie pre-seeding (per provided base URLs)
    cookies: List[Dict] = []
    if add_common_cookies_for:
        cookies = _build_common_cookies(
            add_common_cookies_for,
            lang=lang,
            country=region_country,
            currency=region_currency,
            include_consent=include_consent_cookies,
        )

    # 3) Persistent profile to reuse sessions/cookies between runs
    use_persistent_context = bool(persistent)
    if use_persistent_context and not user_data_dir:
        raise ValueError("user_data_dir must be provided when persistent=True")

    # 4) Conservative Chromium flags to reduce background work/requests
    #    Keep these moderate; overly aggressive flags can break sites.
    #    NOTE: Crawl4AI/Playwright already supply many safe defaults.
    lang_flag = f"--lang={lang}"
    extra_args = [
        lang_flag,
        "--no-default-browser-check",
        "--no-first-run",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-client-side-phishing-detection",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-features=Translate,TranslateUI,PreloadMediaEngagementData,InterestFeedContentSuggestions,OptimizationHintsFetching,OptimizationTargetPrediction",
        "--disable-sync",
        "--metrics-recording-only",
        "--disable-renderer-backgrounding",
        "--disable-ipc-flooding-protection",
        "--disable-notifications",
        "--mute-audio",
        # Redundant with text_mode, but helps on some sites:
        "--blink-settings=imagesEnabled=false",
    ]

    cfg = BrowserConfig(
        browser_type="chromium",
        headless=headless,
        proxy_config=proxy_config,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        verbose=False,
        use_persistent_context=use_persistent_context,
        user_data_dir=user_data_dir,
        cookies=cookies,
        headers=headers,
        user_agent=user_agent,
        text_mode=text_mode,     # disable images
        light_mode=light_mode,   # lighter page features
        extra_args=extra_args,
        enable_stealth=enable_stealth,
        # If your environment occasionally sees sketchy TLS:
        # ignore_https_errors=ignore_https_errors,   # only if supported by your Crawl4AI version
    )
    return cfg


# -----------------------------
# Ready-made configs
# -----------------------------
# Default “fast text” config. You can pass domain list at runtime if you want cookies.
browser_cfg = make_browser_config(
    lang="en-US",
    headless=True,
    enable_stealth=True,
    viewport_width=1280,
    viewport_height=720,
    persistent=False,
    add_common_cookies_for=[],
    include_consent_cookies=True,
    region_country="US",
    region_currency="USD",
)