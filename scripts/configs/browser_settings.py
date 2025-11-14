from __future__ import annotations

from typing import Dict, List, Optional
from crawl4ai import BrowserConfig


# -----------------------------
# Helpers
# -----------------------------
def _accept_language_string(lang: str = "en-US") -> str:
    """
    Build a strong English-forward Accept-Language string.
    Example: "en-US,en;q=0.9"
    """
    lang = (lang or "en-US").strip()
    if "-" not in lang:
        lang = f"{lang}-US"
    base = lang
    primary = base.split("-")[0]
    return f"{base},{primary};q=0.9,en;q=0.8"


def _language_cookies_for(url: str, lang: str = "en-US") -> List[Dict]:
    """
    Prepare a set of *candidate* language cookies commonly used across sites.
    Harmless if the site ignores unknown names.
    """
    names = [
        "locale", "lang", "language", "siteLocale",
        "mk_locale", "MKLocale", "selectedLocale",
        "culture", "i18nLocale",
    ]
    cookies = []
    for name in names:
        cookies.append(
            {
                "name": name,
                "value": lang,
                "url": url,           # Playwright requires a URL scope
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            }
        )
    return cookies


def _merge_headers(base: Dict[str, str], extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    out = dict(base or {})
    if extra:
        out.update({k: v for k, v in extra.items() if v is not None})
    return out


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
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    proxy_config: Optional[dict] = None,
    persistent: bool = False,
    user_data_dir: Optional[str] = None,
    add_language_cookies_for: Optional[List[str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    light_mode: bool = False,
    text_mode: bool = False,
    ignore_https_errors: bool = True,
) -> BrowserConfig:
    """
    Build a BrowserConfig with strong English preference and optional language cookies.

    Args:
        lang:              Preferred locale (e.g., "en-US", "en-GB").
        headless:          Headless mode toggle.
        viewport_width:    Initial width.
        viewport_height:   Initial height.
        user_agent:        Custom UA string.
        proxy_config:      Dict for proxy (e.g., {"server": "...", "username": "..."}).
        persistent:        Use persistent browser context (keeps cookies, sessions).
        user_data_dir:     Required if persistent=True to store profile data.
        add_language_cookies_for:
                           List of base URLs to plant language cookies into
                           (e.g., ["https://www.marykay.com"]).
        extra_headers:     Additional headers to send on every request.
        light_mode:        Performance-friendly toggles in crawl4ai.
        text_mode:         If True, disable images/other heavy content.
        ignore_https_errors:
                           Skip TLS errors.
    """
    # 1) Accept-Language header
    accept_language = _accept_language_string(lang)

    # 2) Chromium language switch affects navigator.language etc.
    #    (This pairs with the HTTP header above.)
    lang_flag = f"--lang={lang}"

    # Disables Chromium translate UI so it doesn’t attempt auto-translation noise.
    disable_translate_flag = "--disable-features=Translate,TranslateUI,LanguageSettings"

    # 3) Headers: Accept-Language plus any extras
    base_headers = {"Accept-Language": accept_language}
    headers = _merge_headers(base_headers, extra_headers)

    # 4) Optional language cookies for specific domains
    cookies: List[Dict] = []
    for base_url in (add_language_cookies_for or []):
        cookies.extend(_language_cookies_for(base_url, lang=lang))

    # 5) Persistent context if desired
    use_persistent_context = bool(persistent)
    if use_persistent_context and not user_data_dir:
        # Playwright requires user_data_dir for persistent contexts
        raise ValueError("user_data_dir must be provided when persistent=True")

    # 6) Assemble BrowserConfig
    cfg = BrowserConfig(
        browser_type="chromium",
        headless=headless,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        proxy_config=proxy_config,
        user_agent=user_agent,
        use_persistent_context=use_persistent_context,
        user_data_dir=user_data_dir,
        ignore_https_errors=ignore_https_errors,
        java_script_enabled=True,
        cookies=cookies,
        headers=headers,
        light_mode=light_mode,
        text_mode=text_mode,
        extra_args=[lang_flag, disable_translate_flag],
    )
    return cfg


# -----------------------------
# Common ready-made configs
# -----------------------------
# Strong English preference, also plants language cookies for Mary Kay.
browser_cfg = make_browser_config(
    lang="en-US",
    headless=True,
    viewport_width=1280,
    viewport_height=720,
)
