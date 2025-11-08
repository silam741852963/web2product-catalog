from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Optional

from crawl4ai import CrawlerRunConfig, CacheMode, VirtualScrollConfig

logger = logging.getLogger(__name__)

# Common cookie/consent/overlay selectors (tried in order)
# Keep this compact and general; sites vary wildly.
COOKIE_CLICK_SELECTORS: Tuple[str, ...] = (
    # OneTrust / Optanon
    "#onetrust-accept-btn-handler",
    "#onetrust-reject-all-handler",
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
    # In-house common phrases but left as text-match fallback (not Playwright pseudos)
    # NOTE: don't include Playwright pseudos like `:has-text()` here (they break document.querySelector)
)

# Text phrases to search for as a fallback (case-insensitive, substring match)
COOKIE_TEXT_PHRASES: Tuple[str, ...] = (
    "accept all",
    "agree",
    "allow all",
    "accept cookies",
    "i agree",
    "save & close",
    "allow",
)

# JS snippet to clear overflow and remove common full-screen overlays
OVERLAY_CLEANUP_JS = r"""
(() => {
  try {
    // unlock body scrolling if a modal locked it
    document.body.style.overflow = 'auto';
    document.body.style.position = 'static';
  } catch(e) {}

  // remove obvious fixed-position overlays if tagged as consent/overlay
  const fullscreens = Array.from(document.querySelectorAll('div,section'))
    .filter(el => {
      try {
        const st = window.getComputedStyle(el);
        if (!st) return false;
        const isFixed = st.position === 'fixed' || st.position === 'sticky';
        const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
        const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        const big = (el.offsetWidth >= vw * 0.7 && el.offsetHeight >= vh * 0.5) || (el.offsetWidth >= vw * 0.4 && el.offsetHeight >= vh * 0.7);
        const cls = (el.className || '').toLowerCase();
        const id  = (el.id || '').toLowerCase();
        const cookiey = /cookie|consent|privacy|gdpr|overlay|modal|dialog|consent-banner/.test(cls) || /cookie|consent|privacy|gdpr|overlay|modal|dialog|consent-banner/.test(id);
        return (isFixed && big && cookiey);
      } catch (e) {
        return false;
      }
    });
  fullscreens.forEach(el => {
    try { el.remove(); } catch(e) {}
  });

  // sometimes aria-hidden="true" sits on main wrappers after overlay
  const mainish = document.querySelectorAll('main,[role="main"],article,.content,.product-grid,.products');
  mainish.forEach(m => {
    try {
      if (m.getAttribute && m.getAttribute('aria-hidden') === 'true') m.removeAttribute('aria-hidden');
      // if element not visible but contains text, make it visible
      m.style.display = m.style.display || 'block';
    } catch(e) {}
  });

  // a gentle scroll can trigger lazy hydration
  try { window.scrollTo(0, Math.min(200, document.body.scrollHeight || 0)); } catch(e) {}
})();
"""

def _safe_js_string(s: str) -> str:
    # JSON encode to produce a safely quoted JS string literal
    return json.dumps(s)

def _robust_cookie_playbook_js(selectors: Iterable[str], attempts: int, wait_ms: int) -> str:
    """
    Repeatedly tries to click cookie buttons on the main document and any *same-origin* iframes.
    Uses both selector matching and visible text fallback. Safely JSON-escapes selector strings.
    Designed to survive late hydration. Fails silently on cross-origin frames.
    """
    sel_list = list(selectors or [])
    js_selectors_array = "[" + ",".join([_safe_js_string(s) for s in sel_list]) + "]"
    attempts = max(1, int(attempts))
    wait_ms = max(50, int(wait_ms))
    # Text phrases fallback
    text_phrases = "[" + ",".join([_safe_js_string(p) for p in COOKIE_TEXT_PHRASES]) + "]"

    return f"""
(() => {{
  const selectors = {js_selectors_array};
  const text_phrases = {text_phrases};
  let tries = {attempts};
  const wait = {wait_ms};

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

  const clickBySelectors = (root) => {{
    for (const s of selectors) {{
      try {{
        const el = root.querySelector(s);
        if (el && is_visible(el)) {{
          clickEl(el);
        }}
      }} catch (e) {{
        // invalid selector for document.querySelector -> ignore
      }}
    }}
  }};

  const clickByText = (root) => {{
    // scan candidate elements and match by innerText phrases
    const candidates = Array.from(root.querySelectorAll('button,input[type="button"],input[type="submit"],a'));
    for (const el of candidates) {{
      try {{
        if (!is_visible(el)) continue;
        const txt = (el.innerText || el.value || '').trim().toLowerCase();
        if (!txt) continue;
        for (const p of text_phrases) {{
          if (txt.indexOf(p) !== -1) {{
            clickEl(el);
            break;
          }}
        }}
      }} catch(e) {{}}
    }}
  }};

  const clickSkipToMain = (root) => {{
    try {{
      // common "skip to content" patterns
      const skip = root.querySelector('a[href^="#main"], a[href^="#content"], a.skip-to-content, a[rel="skip"]');
      if (skip && is_visible(skip)) {{
        clickEl(skip);
      }} else {{
        // try link text
        const links = Array.from(root.querySelectorAll('a'));
        for (const a of links) {{
          try {{
            const t = (a.innerText || '').trim().toLowerCase();
            if (!t) continue;
            if (t.indexOf('skip') !== -1 && (t.indexOf('content') !== -1 || t.indexOf('main') !== -1)) {{
              clickEl(a);
              break;
            }}
          }} catch (e) {{}}
        }}
      }}
    }} catch(e) {{}}
  }};

  const tick = () => {{
    try {{
      clickBySelectors(document);
      clickByText(document);
      clickSkipToMain(document);

      // attempt same-origin iframes
      const iframes = Array.from(document.querySelectorAll('iframe'));
      for (const fr of iframes) {{
        try {{
          if (!fr.contentWindow || !fr.contentDocument) continue;
          clickBySelectors(fr.contentDocument);
          clickByText(fr.contentDocument);
          clickSkipToMain(fr.contentDocument);
        }} catch (e) {{ /* cross-origin or blocked; ignore */ }}
      }}
    }} catch (e) {{}}
    tries--;
    if (tries > 0) setTimeout(tick, wait);
  }};
  setTimeout(tick, wait);
}})();
"""

def _content_ready_wait_js(min_chars: int = 800, max_cookie_hits: int = 3) -> str:
    """
    A JS wait condition that:
      - requires a main-like container present,
      - text length >= min_chars,
      - cookie-ish vocabulary does not dominate (<= max_cookie_hits).
    """
    # Use a plain JS function string (the crawler expects a "js:() => { ... }" style)
    return f"""js:() => {{
      const root = document.querySelector('main,[role="main"],article,.content,.product-grid,.products');
      if (!root) return false;
      const text = (root.innerText || '').trim();
      if (text.length < {min_chars}) return false;
      const cookieish = /(cookie|your\\s+device|personalized\\s+web\\s+experience|consent|privacy\\s+preferences|consent|cookie\\s+banner)/ig;
      const hits = (text.match(cookieish) || []).length;
      return hits <= {max_cookie_hits};
    }}"""

@dataclass
class PageInteractionPolicy:
    enable_cookie_playbook: bool = True
    max_in_session_retries: int = 1
    wait_timeout_ms: int = 60000
    delay_before_return_sec: float = 1.2

    # NEW: cookie retry window
    cookie_retry_attempts: int = 6          # 6 * 300ms ≈ 1.8s
    cookie_retry_interval_ms: int = 300

    # Virtual scrolling (off by default; enable per-URL heuristics in caller)
    virtual_scroll: bool = False
    scroll_container_selectors: Tuple[str, ...] = (".products", ".product-grid", ".category-grid")
    scroll_count: int = 20
    scroll_by: str = "container_height"  # or "viewport"
    wait_after_scroll_sec: float = 0.8

    # Session handling
    session_id_prefix: str = "w2p"

    # Wait tuning
    min_content_chars: int = 800
    max_cookie_hits: int = 3

    # Consent selectors
    cookie_selectors: Tuple[str, ...] = field(default_factory=lambda: COOKIE_CLICK_SELECTORS)

def stable_session_id(url: str, prefix: str = "w2p") -> str:
    h = hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"

def _base_config(policy: PageInteractionPolicy, *, js_only: bool, session_id: str) -> CrawlerRunConfig:
    """
    Build a CrawlerRunConfig tuned for interaction:
      - optionally injects cookie/overlay playbook JS
      - sets wait_for condition for content readiness
      - configures virtual scroll if enabled
      - uses cache bypass for retries to avoid stale DOM
    """
    js_snippets: List[str] = []
    if policy.enable_cookie_playbook:
        # fire a robust, multi-attempt playbook that also probes same-origin iframes
        js_snippets.append(_robust_cookie_playbook_js(
            policy.cookie_selectors,
            policy.cookie_retry_attempts,
            policy.cookie_retry_interval_ms
        ))
        js_snippets.append(OVERLAY_CLEANUP_JS)

    wait_for = _content_ready_wait_js(
        min_chars=policy.min_content_chars,
        max_cookie_hits=policy.max_cookie_hits
    )

    # Optional Virtual Scroll configuration
    vconf: Optional[VirtualScrollConfig] = None
    if policy.virtual_scroll:
        vconf = VirtualScrollConfig(
            container_selector=",".join(policy.scroll_container_selectors),
            scroll_count=policy.scroll_count,
            scroll_by=policy.scroll_by,
            wait_after_scroll=policy.wait_after_scroll_sec
        )

    cfg = CrawlerRunConfig(
        js_code=js_snippets if js_snippets else None,
        wait_for=wait_for,
        page_timeout=policy.wait_timeout_ms,
        delay_before_return_html=policy.delay_before_return_sec,
        session_id=session_id,
        js_only=js_only,
        cache_mode=CacheMode.BYPASS,             # avoid stale DOM during retries
        remove_overlay_elements=True,            # built-in best-effort removal
        virtual_scroll_config=vconf
    )
    return cfg

def first_pass_config(url: str, policy: PageInteractionPolicy) -> CrawlerRunConfig:
    sid = stable_session_id(url, policy.session_id_prefix)
    return _base_config(policy, js_only=False, session_id=sid)

def retry_pass_config(url: str, policy: PageInteractionPolicy) -> CrawlerRunConfig:
    sid = stable_session_id(url, policy.session_id_prefix)
    # JS-only retry: set js_only True
    return _base_config(policy, js_only=True, session_id=sid)

def make_interaction_plan(url: str, policy: PageInteractionPolicy):
    """
    Returns (initial_config, retry_configs_list). Caller can run initial first,
    then each retry config (usually length <= policy.max_in_session_retries).
    """
    init = first_pass_config(url, policy)
    retries = [retry_pass_config(url, policy) for _ in range(max(0, policy.max_in_session_retries))]
    return init, retries