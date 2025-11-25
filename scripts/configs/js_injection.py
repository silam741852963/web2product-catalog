from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Protocol, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _safe_js_string(s: str) -> str:
    return json.dumps(s)


# --------------------------------------------------------------------------- #
# Cookie & interstitial defaults
# --------------------------------------------------------------------------- #

DEFAULT_COOKIE_TEXT_PHRASES: Tuple[str, ...] = (
    "accept all",
    "accept cookies",
    "allow all",
    "i accept",
    "i agree",
    "save & close",
    "save & accept",
)

DEFAULT_COOKIE_CLICK_SELECTORS: Tuple[str, ...] = (
    # OneTrust / Optanon
    "#onetrust-accept-btn-handler",
    "#onetrrust-reject-all-handler",
    ".onetrust-accept-btn-handler",
    ".optanon-allow-all",
    ".optanon-accept-all",
    # CookieYes / CookieLaw
    ".cky-btn-accept",
    ".cky-consent-container [data-cky-tag='accept-button']",
    # Some legacy classes
    ".cc-allow",
    ".cc-allow-all",
    ".cc-btn.cc-allow",
    # TrustArc family
    "#truste-consent-button",
    ".truste_accept",
    ".trustarc-accept",
)


OVERLAY_CLEANUP_JS: str = r"""
(() => {
  try {
    document.body.style.overflow = 'auto';
    document.body.style.position = 'static';
  } catch(e) {}

  const fullscreens = Array.from(document.querySelectorAll('div,section'))
    .filter(el => {
      try {
        const st = window.getComputedStyle(el);
        if (!st) return false;
        const isFixed = st.position === 'fixed' || st.position === 'sticky';
        const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
        const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        const big = (el.offsetWidth >= vw * 0.7 && el.offsetHeight >= vh * 0.5) ||
                    (el.offsetWidth >= vw * 0.4 && el.offsetHeight >= vh * 0.7);
        const cls = (el.className || '').toLowerCase();
        const id  = (el.id || '').toLowerCase();
        const cookiey = /cookie|consent|privacy|gdpr|overlay|modal|dialog|consent-banner/.test(cls) ||
                        /cookie|consent|privacy|gdpr|overlay|modal|dialog|consent-banner/.test(id);
        return (isFixed && big && cookiey);
      } catch (e) {
        return false;
      }
    });
  fullscreens.forEach(el => {
    try { el.remove(); } catch(e) {}
  });

  const mainish = document.querySelectorAll('main,[role="main"],article,.content,.product-grid,.products');
  mainish.forEach(m => {
    try {
      if (m.getAttribute && m.getAttribute('aria-hidden') === 'true') m.removeAttribute('aria-hidden');
      m.style.display = m.style.display || 'block';
    } catch(e) {}
  });

  try { window.scrollTo(0, Math.min(200, document.body.scrollHeight || 0)); } catch(e) {}
})();
"""


# --------------------------------------------------------------------------- #
# JS Injection Strategy interface
# --------------------------------------------------------------------------- #


class JSInjectionStrategy(Protocol):
    """
    Strategy interface for building JS snippets used in page interaction.

    Implementations are pure configuration/templating layers:
      - no Crawl4AI imports
      - no side effects
    """

    def build_cookie_playbook(
        self,
        *,
        selectors: Iterable[str] | None = None,
        text_phrases: Iterable[str] | None = None,
        attempts: int | None = None,
        wait_ms: int | None = None,
    ) -> str:  # pragma: no cover
        ...

    def overlay_cleanup_js(self) -> str:  # pragma: no cover
        ...

    def build_anti_interstitial(
        self,
        *,
        aggressive: bool = False,
    ) -> str:  # pragma: no cover
        ...

    def build_content_ready_wait(
        self,
        *,
        min_chars: int | None = None,
        max_cookie_hits: int | None = None,
    ) -> str:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# Default JS injection strategy implementation
# --------------------------------------------------------------------------- #


@dataclass
class DefaultJSInjectionStrategy:
    """
    Default JS injection strategy:

    - Cookie / consent playbook (document + same-origin iframes)
    - Overlay cleanup helper
    - Anti-interstitial script (cookie, legal, age, etc.) [opt-in]
    - Content-ready `wait_for` condition
    """

    # Cookie playbook defaults
    default_cookie_selectors: Tuple[str, ...] = DEFAULT_COOKIE_CLICK_SELECTORS
    default_cookie_text_phrases: Tuple[str, ...] = DEFAULT_COOKIE_TEXT_PHRASES
    default_cookie_attempts: int = 4
    default_cookie_wait_ms: int = 400

    # Content-ready defaults
    default_min_content_chars: int = 800
    default_max_cookie_hits: int = 3

    def build_cookie_playbook(
        self,
        *,
        selectors: Iterable[str] | None = None,
        text_phrases: Iterable[str] | None = None,
        attempts: int | None = None,
        wait_ms: int | None = None,
    ) -> str:
        sel_list: List[str] = list(selectors or self.default_cookie_selectors)
        txt_list: List[str] = list(text_phrases or self.default_cookie_text_phrases)
        eff_attempts = (
            self.default_cookie_attempts if attempts is None else max(1, int(attempts))
        )
        eff_wait_ms = self.default_cookie_wait_ms if wait_ms is None else max(
            80, int(wait_ms)
        )

        js_selectors_array = "[" + ",".join(_safe_js_string(s) for s in sel_list) + "]"
        text_array = "[" + ",".join(_safe_js_string(p) for p in txt_list) + "]"

        # Safer cookie playbook:
        # - Only clicks inside cookie/consent/privacy contexts.
        # - Never clicks <a> (except the skip-to-main helper with hash links).
        # - Limited retries with modest delay.
        return f"""
(() => {{
  const selectors = {js_selectors_array};
  const text_phrases = {text_array};
  let tries = {eff_attempts};
  const wait = {eff_wait_ms};

  const is_visible = (el) => {{
    if (!el) return false;
    try {{
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return false;
      const style = window.getComputedStyle(el);
      if (style && (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0')) return false;
      return true;
    }} catch(e) {{ return false; }}
  }};

  const clickEl = (el) => {{
    try {{
      if (!el) return false;
      if (typeof el.click === 'function') {{
        el.click();
        return true;
      }}
      const ev = new MouseEvent('click', {{bubbles: true, cancelable: true}});
      return el.dispatchEvent(ev);
    }} catch (e) {{
      return false;
    }}
  }};

  const isCookieContext = (el) => {{
    const rx = /(cookie|cookies|consent|gdpr|privacy|tracking)/i;
    let cur = el;
    while (cur && cur !== document.body) {{
      try {{
        const cls = (cur.className || '').toString().toLowerCase();
        const id  = (cur.id || '').toString().toLowerCase();
        const aria = (cur.getAttribute && (cur.getAttribute('aria-label') || '')).toLowerCase();
        if (rx.test(cls) || rx.test(id) || rx.test(aria)) return true;
      }} catch(e) {{}}
      cur = cur.parentElement;
    }}
    return false;
  }};

  const clickBySelectors = (root) => {{
    for (const s of selectors) {{
      try {{
        const el = root.querySelector(s);
        if (el && is_visible(el) && isCookieContext(el)) {{
          clickEl(el);
        }}
      }} catch (e) {{}}
    }}
  }};

  const clickByText = (root) => {{
    const candidates = Array.from(root.querySelectorAll('button,[role="button"],input[type="button"],input[type="submit"]'));
    for (const el of candidates) {{
      try {{
        if (!is_visible(el)) continue;
        if (!isCookieContext(el)) continue;
        const txt = (el.innerText || el.value || '').trim().toLowerCase();
        if (!txt) continue;
        for (const p of text_phrases) {{
          if (p && txt.includes(p)) {{
            clickEl(el);
            break;
          }}
        }}
      }} catch(e) {{}}
    }}
  }};

  const clickSkipToMain = (root) => {{
    try {{
      const skip = root.querySelector('a[href^="#main"], a[href^="#content"], a.skip-to-content, a[rel="skip"]');
      if (skip && is_visible(skip)) {{
        clickEl(skip);
      }} else {{
        const links = Array.from(root.querySelectorAll('a[href^="#"]'));
        for (const a of links) {{
          const t = (a.innerText || '').trim().toLowerCase();
          if (!t) continue;
          if (t.includes('skip') && (t.includes('content') || t.includes('main'))) {{
            clickEl(a);
            break;
          }}
        }}
      }}
    }} catch(e) {{}}
  }};

  const tick = () => {{
    try {{
      clickBySelectors(document);
      clickByText(document);
      clickSkipToMain(document);

      const iframes = Array.from(document.querySelectorAll('iframe'));
      for (const fr of iframes) {{
        try {{
          if (!fr.contentWindow || !fr.contentDocument) continue;
          clickBySelectors(fr.contentDocument);
          clickByText(fr.contentDocument);
          clickSkipToMain(fr.contentDocument);
        }} catch (e) {{}}
      }}
    }} catch (e) {{}}
    tries--;
    if (tries > 0) setTimeout(tick, wait);
  }};
  setTimeout(tick, wait);
}})();
"""

    def overlay_cleanup_js(self) -> str:
        return OVERLAY_CLEANUP_JS

    def build_anti_interstitial(
        self,
        *,
        aggressive: bool = False,
    ) -> str:
        """
        Safer anti-interstitial:

        - Only interacts with overlays that look like cookie/consent/privacy banners.
        - Never clicks <a> elements (reduces risk of navigation / external apps).
        - Narrower text patterns (no generic OK/Continue/Yes/Confirm).
        - Shorter, less frequent interval.
        """
        accept_texts = [
            "accept",
            "accept all",
            "accept cookies",
            "allow all",
            "i accept",
            "i agree",
            "save & accept",
            "save & close",
            # common EU language variants
            "alle zulassen",
            "zustimmen",
            "einverstanden",
            "accepter",
            "tout accepter",
            "j'accepte",
            "aceptar",
            "aceptar todo",
            "permitir cookies",
        ]

        close_texts = [
            "reject all",
            "decline",
            "use necessary cookies only",
            "necessary cookies only",
            "only necessary",
            "close",
            "dismiss",
            "continue without accepting",
        ]

        vendor_buttons = [
            "#onetrust-accept-btn-handler",
            "button#onetrust-accept-btn-handler",
            "#onetrust-reject-all-handler",
            "#onetrust-pc-btn-handler",
            "#CybotCookiebotDialogBodyButtonAccept",
            "button#CybotCookiebotDialogBodyButtonAccept",
            "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
            "#truste-consent-button",
            "button.truste-button1",
            "button#acceptAllButton",
            ".qc-cmp2-summary-buttons .qc-cmp2-accept-all",
            ".qc-cmp2-footer .qc-cmp2-accept-all",
            "#didomi-notice-agree-button",
            "button[data-action='accept']",
            "button.cky-btn-accept",
            ".cky-btn-accept",
            "#cookie_action_close_header",
            "#accept-recommended-btn-handler",
            ".iubenda-cs-accept-btn",
            "[aria-label*='accept' i]",
            "[aria-label*='agree' i]",
            "[data-testid*='accept' i]",
            "[data-testid*='agree' i]",
        ]

        big_fraction = "0.25" if aggressive else "0.33"
        max_ticks = "30" if aggressive else "15"
        interval_ms = "250" if aggressive else "400"

        return f"""
    (() => {{
      const BTN_TEXTS_ACCEPT = {accept_texts!r};
      const BTN_TEXTS_CLOSE  = {close_texts!r};
      const VENDOR_SELECTORS = {vendor_buttons!r};
      const BIG_FRAC = {big_fraction};
      const MAX_TICKS = {max_ticks};
      const INTERVAL_MS = {interval_ms};
      const COOKIE_RX = /(cookie|cookies|consent|gdpr|privacy|tracking)/i;

      const isVisible = (el) => {{
        try {{
          const r = el.getBoundingClientRect();
          const cs = getComputedStyle(el);
          return r.width > 1 && r.height > 1 &&
                 cs.visibility !== 'hidden' &&
                 cs.display !== 'none' &&
                 cs.opacity !== '0';
        }} catch (_e) {{ return false; }}
      }};

      const textLike = (el) => {{
        try {{ return (el.innerText || el.textContent || "").trim().toLowerCase(); }}
        catch (_e) {{ return ""; }}
      }};

      const isCookieOverlay = (el) => {{
        if (!el) return false;
        try {{
          const cls = (el.className || '').toString().toLowerCase();
          const id  = (el.id || '').toString().toLowerCase();
          const aria = (el.getAttribute && (el.getAttribute('aria-label') || '')).toLowerCase();
          if (COOKIE_RX.test(cls) || COOKIE_RX.test(id) || COOKIE_RX.test(aria)) return true;
          const sample = (el.innerText || '').slice(0, 600).toLowerCase();
          return COOKIE_RX.test(sample);
        }} catch (_e) {{
          return false;
        }}
      }};

      const findCookieRoots = () => {{
        const roots = [];
        const candidates = document.querySelectorAll(
          '[role="dialog"],[aria-modal="true"],div,section,aside,footer,header'
        );
        candidates.forEach(el => {{
          if (isCookieOverlay(el)) roots.push(el);
        }});
        if (!roots.length) {{
          const banner = document.querySelector(
            '[id*="cookie" i],[class*="cookie" i],[id*="consent" i],[class*="consent" i]'
          );
          if (banner) roots.push(banner);
        }}
        return roots;
      }};

      const clickIf = (el) => {{
        try {{
          if (!isVisible(el)) return false;
          el.click();
          return true;
        }} catch (_e) {{ return false; }}
      }};

      const clickButtonsByText = (root, texts) => {{
        let clicked = false;
        const btns = Array.from(
          root.querySelectorAll("button, [role='button'], input[type='button'], input[type='submit']")
        );
        for (const b of btns) {{
          const t = textLike(b);
          if (!t) continue;
          for (const k of texts) {{
            if (!k) continue;
            if (t === k || t.includes(k)) {{
              if (clickIf(b)) clicked = true;
            }}
          }}
        }}
        return clicked;
      }};

      const tryVendorSelectors = () => {{
        let clicked = false;
        for (const sel of VENDOR_SELECTORS) {{
          try {{
            document.querySelectorAll(sel).forEach(el => {{
              if (!isCookieOverlay(el)) return;
              if (clickIf(el)) clicked = true;
            }});
          }} catch (_e) {{}}
        }}
        return clicked;
      }};

      const removeBigCookieOverlays = () => {{
        const vw = window.innerWidth || 1;
        const vh = window.innerHeight || 1;
        const nodes = Array.from(
          document.querySelectorAll('[role="dialog"],[aria-modal="true"],div,section,aside,footer,header')
        );
        for (const el of nodes) {{
          try {{
            if (!isCookieOverlay(el)) continue;
            const cs = getComputedStyle(el);
            const r = el.getBoundingClientRect();
            const big = r.width/vw > BIG_FRAC && r.height/vh > BIG_FRAC;
            const layered = (cs.position === 'fixed' || cs.position === 'sticky') &&
                            parseInt(cs.zIndex || '0', 10) >= 1000;
            if (big && layered) {{
              el.remove();
            }}
          }} catch (_e) {{}}
        }}
      }};

      let ticks = 0;
      const iv = setInterval(() => {{
        try {{
          const roots = findCookieRoots();
          if (roots.length) {{
            roots.forEach(root => {{
              clickButtonsByText(root, BTN_TEXTS_ACCEPT);
              clickButtonsByText(root, BTN_TEXTS_CLOSE);
            }});
          }}
          tryVendorSelectors();
          removeBigCookieOverlays();
          ticks++;
          if (ticks >= MAX_TICKS) clearInterval(iv);
        }} catch (_e) {{
          clearInterval(iv);
        }}
      }}, INTERVAL_MS);
    }})();
    """

    def build_content_ready_wait(
        self,
        *,
        min_chars: int | None = None,
        max_cookie_hits: int | None = None,
    ) -> str:
        eff_min_chars = (
            self.default_min_content_chars if min_chars is None else int(min_chars)
        )
        eff_max_cookie_hits = (
            self.default_max_cookie_hits if max_cookie_hits is None else int(max_cookie_hits)
        )
        return f"""js:() => {{
      const root = document.querySelector('main,[role="main"],article,.content,.product-grid,.products');
      if (!root) return false;
      const text = (root.innerText || '').trim();
      if (text.length < {eff_min_chars}) return false;
      const cookieish = /(cookie|your\\s+device|personalized\\s+web\\s+experience|consent|privacy\\s+preferences|consent|cookie\\s+banner)/ig;
      const hits = (text.match(cookieish) || []).length;
      return hits <= {eff_max_cookie_hits};
    }}"""


# --------------------------------------------------------------------------- #
# JS Injection Factory
# --------------------------------------------------------------------------- #


@dataclass
class JSInjectionFactory:
    """
    Factory that uses a JSInjectionStrategy.
    """

    strategy: JSInjectionStrategy

    def cookie_playbook_js(
        self,
        *,
        selectors: Iterable[str] | None = None,
        text_phrases: Iterable[str] | None = None,
        attempts: int | None = None,
        wait_ms: int | None = None,
    ) -> str:
        return self.strategy.build_cookie_playbook(
            selectors=selectors,
            text_phrases=text_phrases,
            attempts=attempts,
            wait_ms=wait_ms,
        )

    def overlay_cleanup_js(self) -> str:
        return self.strategy.overlay_cleanup_js()

    def anti_interstitial_js(self, *, aggressive: bool = False) -> str:
        return self.strategy.build_anti_interstitial(aggressive=aggressive)

    def content_ready_wait_js(
        self,
        *,
        min_chars: int | None = None,
        max_cookie_hits: int | None = None,
    ) -> str:
        return self.strategy.build_content_ready_wait(
            min_chars=min_chars,
            max_cookie_hits=max_cookie_hits,
        )


# --------------------------------------------------------------------------- #
# Page interaction policy & interaction config (no retry)
# --------------------------------------------------------------------------- #


@dataclass
class PageInteractionPolicy:
    """
    Policy object that describes how we interact with pages.
    """

    # Cookie / consent handling
    enable_cookie_playbook: bool = True
    cookie_retry_attempts: int = 6
    cookie_retry_interval_ms: int = 300
    cookie_selectors: Tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_COOKIE_CLICK_SELECTORS
    )

    # Anti-interstitial (opt-in, safer)
    enable_anti_interstitial: bool = False
    anti_interstitial_aggressive: bool = False

    # Wait / timing
    wait_timeout_ms: int = 60000
    delay_before_return_sec: float = 1.2
    min_content_chars: int = 800
    max_cookie_hits: int = 3

    # Virtual scrolling
    virtual_scroll: bool = False
    scroll_container_selectors: Tuple[str, ...] = (
        ".products",
        ".product-grid",
        ".category-grid",
    )
    scroll_count: int = 20
    scroll_by: str = "container_height"  # or "viewport"
    wait_after_scroll_sec: float = 0.8


@dataclass
class InteractionConfig:
    """
    Pure interaction description; run.py turns this into CrawlerRunConfig
    using configs.crawler + crawl4ai.
    """

    js_code: List[str]
    wait_for: str
    page_timeout_ms: int
    delay_before_return_sec: float
    js_only: bool
    virtual_scroll: bool
    scroll_container_selectors: Tuple[str, ...]
    scroll_count: int
    scroll_by: str
    wait_after_scroll_sec: float


class PageInteractionStrategy(Protocol):
    """
    Strategy interface for turning a PageInteractionPolicy into
    InteractionConfig objects.
    """

    def build_base_config(
        self,
        url: str,
        policy: PageInteractionPolicy,
        *,
        js_only: bool,
    ) -> InteractionConfig:  # pragma: no cover
        ...

    def first_pass_config(
        self,
        url: str,
        policy: PageInteractionPolicy,
    ) -> InteractionConfig:  # pragma: no cover
        ...


@dataclass
class DefaultPageInteractionStrategy:
    """
    Default implementation of PageInteractionStrategy.

    It composes JSInjectionFactory for JS snippets.
    """

    js_factory: JSInjectionFactory

    def build_base_config(
        self,
        url: str,
        policy: PageInteractionPolicy,
        *,
        js_only: bool,
    ) -> InteractionConfig:
        js_snippets: List[str] = []

        if policy.enable_cookie_playbook:
            js_snippets.append(
                self.js_factory.cookie_playbook_js(
                    selectors=policy.cookie_selectors,
                    attempts=policy.cookie_retry_attempts,
                    wait_ms=policy.cookie_retry_interval_ms,
                )
            )
            js_snippets.append(self.js_factory.overlay_cleanup_js())

        if policy.enable_anti_interstitial:
            js_snippets.append(
                self.js_factory.anti_interstitial_js(
                    aggressive=policy.anti_interstitial_aggressive
                )
            )

        wait_for = self.js_factory.content_ready_wait_js(
            min_chars=policy.min_content_chars,
            max_cookie_hits=policy.max_cookie_hits,
        )

        cfg = InteractionConfig(
            js_code=js_snippets,
            wait_for=wait_for,
            page_timeout_ms=policy.wait_timeout_ms,
            delay_before_return_sec=policy.delay_before_return_sec,
            js_only=js_only,
            virtual_scroll=policy.virtual_scroll,
            scroll_container_selectors=policy.scroll_container_selectors,
            scroll_count=policy.scroll_count,
            scroll_by=policy.scroll_by,
            wait_after_scroll_sec=policy.wait_after_scroll_sec,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[PageInteraction] build_base_config: url=%s js_only=%s "
                "cookie_playbook=%s anti_interstitial=%s virtual_scroll=%s",
                url,
                js_only,
                policy.enable_cookie_playbook,
                policy.enable_anti_interstitial,
                policy.virtual_scroll,
            )

        return cfg

    def first_pass_config(
        self,
        url: str,
        policy: PageInteractionPolicy,
    ) -> InteractionConfig:
        return self.build_base_config(url, policy, js_only=False)


@dataclass
class PageInteractionFactory:
    """
    Factory that uses a PageInteractionStrategy.
    """

    strategy: PageInteractionStrategy

    def base_config(
        self,
        url: str,
        policy: PageInteractionPolicy,
        *,
        js_only: bool,
    ) -> InteractionConfig:
        return self.strategy.build_base_config(url, policy, js_only=js_only)

    def first_pass_config(
        self,
        url: str,
        policy: PageInteractionPolicy,
    ) -> InteractionConfig:
        return self.strategy.first_pass_config(url, policy)


# --------------------------------------------------------------------------- #
# Default, injectable instances
# --------------------------------------------------------------------------- #

default_js_strategy = DefaultJSInjectionStrategy()
default_js_factory = JSInjectionFactory(strategy=default_js_strategy)

default_page_interaction_strategy = DefaultPageInteractionStrategy(
    js_factory=default_js_factory
)
default_page_interaction_factory = PageInteractionFactory(
    strategy=default_page_interaction_strategy
)

__all__ = [
    # Cookie defaults
    "DEFAULT_COOKIE_TEXT_PHRASES",
    "DEFAULT_COOKIE_CLICK_SELECTORS",
    "OVERLAY_CLEANUP_JS",
    # JS injection
    "JSInjectionStrategy",
    "DefaultJSInjectionStrategy",
    "JSInjectionFactory",
    "default_js_strategy",
    "default_js_factory",
    # Page interaction
    "PageInteractionPolicy",
    "InteractionConfig",
    "PageInteractionStrategy",
    "DefaultPageInteractionStrategy",
    "PageInteractionFactory",
    "default_page_interaction_strategy",
    "default_page_interaction_factory",
]