from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional

InterstitialKind = Literal["none", "cookie", "legal", "age", "login", "paywall", "unknown"]


# ----------------------------
# Text heuristics (Markdown)
# ----------------------------

# Cookie / consent banners (multi-vendor, multi-locale keywords)
_COOKIE_RE = re.compile(
    r"""
    \b(
        cookie\s?(notice|policy|preferences|settings|consent|choices?)|
        we\suse\s(cookies|tracking)|
        accept\s(all|cookies?)|
        agree\s(to\s(cookies|terms))|
        your\sprivacy\s(choice|preferences?)|
        do\snot\ssell\smy\s(personal\s)?data|
        gdpr|ccpa|cmp|one(trust)?|quantcast|trustarc|cookiebot|didomi|cookieyes|iubenda
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Terms / UGC / legal takeovers
_LEGAL_RE = re.compile(
    r"""
    \b(
        terms(\s+and\s+conditions)?|
        terms\s+of\s+(use|service)|
        user(\s+generated)?\s+content|ugc|
        privacy\s+policy|
        license|licensed|sub-?licens(e|able)|
        royalty[-\s]?free|perpetual|irrevoc(able|ably)|
        indemnif(y|ication)|warrant(y|ies?)|
        governed\s+by|jurisdiction|
        publicity\s+and\s+privacy|
        right\s+to\s+use|grant\s+(us|you)\s+the\s+right
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Age gates (e.g., alcohol/cosmetics regions)
_AGE_RE = re.compile(
    r"""
    \b(
        age\s+(gate|verification|check)|are\syou\s(over|older\s+than)\s+(18|19|21)|
        enter\syour\s(date\s+of\s+birth|dob)|i\s(am|confirm)\s(over|older)|
        you\smust\sbe\s(at\s+least)?\s*(18|19|21)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Login / paywall (kept conservative)
_LOGIN_RE = re.compile(
    r"""
    \b(
        sign\s?in|log\s?in|account\s+required|please\s+sign\s+in|
        members?\sonly|subscription\s+required|subscribe\s+to\s+continue|
        paywall|premium\s+content
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORD_RE = re.compile(r"[a-z0-9’']+", re.IGNORECASE)


@dataclass(frozen=True)
class InterstitialThresholds:
    # Minimum total words before we even consider dominance
    min_words: int = 80

    # Cookie dominance: these are intentionally low; cookie text is short but decisive
    cookie_min_hits: int = 4
    cookie_share_threshold: float = 0.005  # 0.5%

    # Legal/UGC dominance: typically long-form legalese, so higher
    legal_min_hits: int = 12
    legal_share_threshold: float = 0.015  # 1.5%

    # Age / login / paywall dominance: short but decisive
    age_min_hits: int = 2
    age_share_threshold: float = 0.003

    login_min_hits: int = 3
    login_share_threshold: float = 0.004


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _score(text: str, rx: re.Pattern[str]) -> Tuple[int, float]:
    total = max(1, _count_words(text))
    hits = len(rx.findall(text))
    share = hits / float(total)
    return hits, share


def detect_interstitial(
    text: str,
    *,
    thresholds: Optional[InterstitialThresholds] = None,
) -> Tuple[InterstitialKind, Dict[str, float]]:
    """
    Inspect plain/markdown text and decide whether an interstitial is dominating.

    Returns:
        (kind, stats)
        kind ∈ {"cookie","legal","age","login","paywall","unknown","none"}
        stats contains total_words, *_hits, *_share
    """
    th = thresholds or InterstitialThresholds()
    t = (text or "").strip()
    if not t:
        return "none", {"total_words": 0.0}

    total_words = float(_count_words(t))
    stats: Dict[str, float] = {"total_words": total_words}

    if total_words < th.min_words:
        # Too small to conclude "dominance"
        return "none", stats

    # Evaluate families
    c_hits, c_share = _score(t, _COOKIE_RE)
    l_hits, l_share = _score(t, _LEGAL_RE)
    a_hits, a_share = _score(t, _AGE_RE)
    g_hits, g_share = _score(t, _LOGIN_RE)

    stats.update(
        {
            "cookie_hits": float(c_hits),
            "cookie_share": float(c_share),
            "legal_hits": float(l_hits),
            "legal_share": float(l_share),
            "age_hits": float(a_hits),
            "age_share": float(a_share),
            "login_hits": float(g_hits),
            "login_share": float(g_share),
        }
    )

    # Order matters: cookie walls are the most common; legal UGC takeovers next
    if (c_hits >= th.cookie_min_hits) or (c_share >= th.cookie_share_threshold):
        return "cookie", stats

    if (l_hits >= th.legal_min_hits) or (l_share >= th.legal_share_threshold):
        return "legal", stats

    if (a_hits >= th.age_min_hits) or (a_share >= th.age_share_threshold):
        return "age", stats

    if (g_hits >= th.login_min_hits) or (g_share >= th.login_share_threshold):
        # We don't try to bypass real auth/paywalls; classify for caller
        return "login", stats

    return "none", stats


def reason_from_kind(kind: InterstitialKind) -> str:
    if kind in ("cookie", "legal", "age"):
        return f"{kind}-dominant"
    if kind in ("login", "paywall"):
        return f"{kind}-required"
    if kind == "unknown":
        return "interstitial"
    return "ok"


# ----------------------------
# JS playbook (browser pass)
# ----------------------------

def anti_interstitial_js(aggressive: bool = False) -> str:
    """
    Best-effort client JS to dismiss common overlays:
    - CMP cookie banners (OneTrust, TrustArc, Quantcast, Didomi, Cookiebot, CookieYes, Iubenda, etc.)
    - Generic modals and full-screen fixed layers
    - Age gates (by pressing obvious confirm buttons)
    It only clicks visible buttons, and removes *large fixed/sticky* overlays.
    """
    # Words we search in visible buttons (English + common variants)
    # Keep short list for safety; we aren't auto-submitting forms.
    accept_texts = [
        "accept", "i accept", "accept all", "agree", "i agree", "got it", "allow all",
        "allow", "continue", "ok", "okay", "save & accept", "yes", "confirm",
        # EU/DE/FR mini variants
        "alle zulassen", "zustimmen", "einverstanden", "accepter", "tout accepter",
        "d'accord", "j'accepte", "aceptar", "aceptar todo", "permitir",
    ]

    # Obvious reject/close texts too (some CMPs only show settings + save)
    close_texts = ["close", "dismiss", "x"]

    # Known CMP/vendor selectors (kept conservative)
    vendor_buttons = [
        # OneTrust
        "#onetrust-accept-btn-handler", "button#onetrust-accept-btn-handler",
        "#onetrust-reject-all-handler", "#onetrust-pc-btn-handler",
        # Cookiebot
        "#CybotCookiebotDialogBodyButtonAccept", "button#CybotCookiebotDialogBodyButtonAccept",
        "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
        # TrustArc
        "#truste-consent-button", "button.truste-button1", "button#acceptAllButton",
        # Quantcast
        ".qc-cmp2-summary-buttons .qc-cmp2-accept-all", ".qc-cmp2-footer .qc-cmp2-accept-all",
        # Didomi
        "#didomi-notice-agree-button", "button[data-action='accept']",
        # CookieYes
        "button.cky-btn-accept", ".cky-btn-accept", "#cookie_action_close_header",
        # CookiePro
        "#accept-recommended-btn-handler",
        # Iubenda
        ".iubenda-cs-accept-btn",
        # Generic closers
        "[aria-label*='accept' i]", "[aria-label*='agree' i]",
        "[data-testid*='accept' i]", "[data-testid*='agree' i]",
    ]

    # Additional legal/interstitial modal containers to remove if huge/fixed
    # (we do not rely on exact class names; just size + z-index + position)
    big_fraction = "0.25" if aggressive else "0.33"
    max_ticks = "100" if aggressive else "40"

    return f"""
    (() => {{
      const BTN_TEXTS_ACCEPT = {accept_texts!r};
      const BTN_TEXTS_CLOSE  = {close_texts!r};
      const VENDOR_SELECTORS = {vendor_buttons!r};
      const BIG_FRAC = {big_fraction};
      const MAX_TICKS = {max_ticks};

      const isVisible = (el) => {{
        try {{
          const r = el.getBoundingClientRect();
          const cs = getComputedStyle(el);
          return r.width > 1 && r.height > 1 && cs.visibility !== 'hidden' && cs.display !== 'none';
        }} catch (_e) {{ return false; }}
      }};

      const clickIf = (el) => {{
        try {{
          if (!isVisible(el)) return false;
          el.click();
          return true;
        }} catch (_e) {{ return false; }}
      }};

      const textLike = (el) => {{
        try {{ return (el.innerText || el.textContent || "").trim().toLowerCase(); }}
        catch (_e) {{ return ""; }}
      }};

      const tryClickByText = (texts) => {{
        const btns = Array.from(document.querySelectorAll("button, [role='button'], a, input[type='button'], input[type='submit']"));
        let clicked = false;
        for (const b of btns) {{
          const t = textLike(b);
          for (const k of texts) {{
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
              if (clickIf(el)) clicked = true;
            }});
          }} catch (_e) {{}}
        }}
        return clicked;
      }};

      const containsLegal = (el) => {{
        try {{
          const t = textLike(el);
          return /\\b(terms\\s*(and\\s*conditions|of\\s*(use|service))|privacy\\s*policy|user\\s*generated\\s*content|ugc|royalty[-\\s]?free|perpetual|irrevoc|indemnif|jurisdiction|governed\\s*by)\\b/i.test(t);
        }} catch (_e) {{ return false; }}
      }};

      const removeBigOverlays = () => {{
        const vw = window.innerWidth || 1;
        const vh = window.innerHeight || 1;
        const nodes = Array.from(document.querySelectorAll('[role="dialog"],[aria-modal="true"],div,section,aside,footer,header'));
        for (const el of nodes) {{
          try {{
            const cs = getComputedStyle(el);
            const r = el.getBoundingClientRect();
            const big = r.width/vw > BIG_FRAC && r.height/vh > BIG_FRAC;
            const layered = (cs.position === 'fixed' || cs.position === 'sticky') && parseInt(cs.zIndex || '0', 10) >= 1000;
            if ((big && layered) || (containsLegal(el) && layered && big)) {{
              el.remove();
            }}
          }} catch (_e) {{}}
        }}
      }};

      let ticks = 0;
      const iv = setInterval(() => {{
        try {{
          // 1) obvious vendor buttons
          tryVendorSelectors();

          // 2) accept/agree text buttons
          tryClickByText(BTN_TEXTS_ACCEPT);

          // 3) sometimes modals need close first, then accept appears
          tryClickByText(BTN_TEXTS_CLOSE);

          // 4) remove large fixed overlays (cookie/consent/legal)
          removeBigOverlays();

          // 5) nudge the page for lazy content
          if (ticks % 2 === 0) window.scrollBy(0, 180);
          if (ticks % 10 === 0) window.scrollTo(0, 0);

          ticks++;
          if (ticks >= MAX_TICKS) clearInterval(iv);
        }} catch (_e) {{
          clearInterval(iv);
        }}
      }}, 120);
    }})();
    """


# ----------------------------
# Convenience for callers
# ----------------------------

def detect_and_reason(text: str, thresholds: Optional[InterstitialThresholds] = None) -> Tuple[InterstitialKind, str, Dict[str, float]]:
    kind, stats = detect_interstitial(text, thresholds=thresholds)
    return kind, reason_from_kind(kind), stats