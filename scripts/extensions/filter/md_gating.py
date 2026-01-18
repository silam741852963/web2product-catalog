from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Protocol, Tuple

from configs.language import default_language_factory

logger = logging.getLogger(__name__)

InterstitialKind = Literal[
    "none", "cookie", "legal", "age", "login", "paywall", "unknown"
]

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+\S")
_LIST_RE = re.compile(r"(?m)^\s*[-*+]\s+\S")
_MARKDOWN_TOKEN_RE = re.compile(r"[a-z0-9’']+", re.IGNORECASE)


def _word_count(s: str | None) -> int:
    return len(_WORD_RE.findall(s or ""))


# ---------------------------------------------------------------------------
# Interstitial detection core (static patterns + language-derived patterns)
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class InterstitialThresholds:
    # Minimum total words before we even consider dominance
    min_words: int = 80

    # Cookie dominance: short but decisive
    cookie_min_hits: int = 4
    cookie_share_threshold: float = 0.005  # 0.5%

    # Legal/UGC dominance: long-form
    legal_min_hits: int = 12
    legal_share_threshold: float = 0.015  # 1.5%

    # Age / login / paywall dominance: short but decisive
    age_min_hits: int = 2
    age_share_threshold: float = 0.003

    login_min_hits: int = 3
    login_share_threshold: float = 0.004


def _score(text: str, rx: re.Pattern[str]) -> Tuple[int, float]:
    total = max(1, _word_count(text))
    hits = len(rx.findall(text))
    share = hits / float(total)
    return hits, share


def _kind_reason_from_stats(
    t: str,
    th: InterstitialThresholds,
) -> Tuple[InterstitialKind, Dict[str, float]]:
    """
    More conservative logic to reduce false positives:
    - Require BOTH hit count AND share thresholds for dominance.
    - Also includes a language-derived interstitial regex (union of en + target).
    """
    t = (t or "").strip()
    if not t:
        return "none", {"total_words": 0.0}

    total_words = float(_word_count(t))
    stats: Dict[str, float] = {"total_words": total_words}

    if total_words < th.min_words:
        return "none", stats

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

    if (c_hits >= th.cookie_min_hits) and (c_share >= th.cookie_share_threshold):
        return "cookie", stats

    if (l_hits >= th.legal_min_hits) and (l_share >= th.legal_share_threshold):
        return "legal", stats

    if (a_hits >= th.age_min_hits) and (a_share >= th.age_share_threshold):
        return "age", stats

    if (g_hits >= th.login_min_hits) and (g_share >= th.login_share_threshold):
        return "login", stats

    # Unknown interstitial: language-derived patterns (union across effective langs).
    # This must be evaluated at runtime (no import-time caching) because language can change.
    interstitial_re = default_language_factory.interstitial_re()
    u_hits, u_share = _score(t, interstitial_re)
    stats.update({"unknown_hits": float(u_hits), "unknown_share": float(u_share)})

    return "none", stats


def reason_from_kind(kind: InterstitialKind) -> str:
    if kind in ("cookie", "legal", "age"):
        return f"{kind}-dominant"
    if kind in ("login", "paywall"):
        return f"{kind}-required"
    if kind == "unknown":
        return "interstitial"
    return "ok"


def detect_interstitial(
    text: str,
    *,
    thresholds: Optional[InterstitialThresholds] = None,
) -> Tuple[InterstitialKind, Dict[str, float]]:
    th = thresholds or InterstitialThresholds()
    return _kind_reason_from_stats(text, th)


def detect_and_reason(
    text: str,
    *,
    thresholds: Optional[InterstitialThresholds] = None,
) -> Tuple[InterstitialKind, str, Dict[str, float]]:
    kind, stats = detect_interstitial(text, thresholds=thresholds)
    return kind, reason_from_kind(kind), stats


# ---------------------------------------------------------------------------
# Gating Strategy (Configuration-as-Strategy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownGatingConfig:
    min_meaningful_words: int = 5
    cookie_max_fraction: float = 0.02
    require_structure: bool = True

    # legacy / heuristic knobs for simple gating
    interstitial_max_share: float = 0.70
    interstitial_min_hits: int = 2


class InterstitialDetector(Protocol):
    def __call__(
        self,
        text: str,
        *,
        thresholds: Optional[InterstitialThresholds] = None,
    ) -> Tuple[InterstitialKind, str, Dict[str, float]]: ...


def default_interstitial_detector(
    text: str,
    *,
    thresholds: Optional[InterstitialThresholds] = None,
) -> Tuple[InterstitialKind, str, Dict[str, float]]:
    return detect_and_reason(text, thresholds=thresholds)


def evaluate_markdown(
    markdown_text: str,
    *,
    min_meaningful_words: int,
    cookie_max_fraction: float,
    require_structure: bool,
    detector: InterstitialDetector = default_interstitial_detector,
) -> tuple[str, str, Dict[str, float]]:
    """
    Decide whether to SAVE or SUPPRESS markdown.

    Returns:
        (action, reason, md_stats)
        - action ∈ {"save","suppress"}
        - reason ∈ {"ok","cookie-dominant","legal-dominant","age-dominant",
                    "login-required","too-short","no-structure","empty",...}
        - md_stats: diagnostic counters (total_words, shares/hits, interstitial_kind)
    """
    text = (markdown_text or "").strip()
    stats: Dict[str, float] = {}

    if not text:
        return "suppress", "empty", {"total_words": 0.0}

    total_words = len(_MARKDOWN_TOKEN_RE.findall(text))
    stats["total_words"] = float(total_words)

    # Structure gate
    if require_structure:
        has_heading = _HEADING_RE.search(text) is not None
        has_list = _LIST_RE.search(text) is not None
        if not (
            has_heading or has_list or total_words >= max(5, int(min_meaningful_words))
        ):
            return "suppress", "no-structure", stats

    if total_words < max(5, int(min_meaningful_words)):
        return "suppress", "too-short", stats

    # Interstitial detection
    cookie_share_cap = min(0.02, max(0.001, float(cookie_max_fraction or 0.02)))
    th = InterstitialThresholds(cookie_share_threshold=cookie_share_cap)

    kind, reason, istats = detector(text, thresholds=th)
    stats.update(istats)
    stats["interstitial_kind"] = {
        "none": 0,
        "cookie": 1,
        "legal": 2,
        "age": 3,
        "login": 4,
        "paywall": 5,
        "unknown": 6,
    }.get(kind, -1)

    # If an interstitial is truly dominant, just suppress.
    if kind in ("cookie", "legal", "age"):
        return "suppress", reason, stats

    if kind in ("login", "paywall"):
        return "suppress", reason, stats

    return "save", "ok", stats


# ---------------------------------------------------------------------------
# Factory for DI (Configuration-as-Strategy)
# ---------------------------------------------------------------------------


def build_gating_config(
    *,
    min_meaningful_words: int = 30,
    cookie_max_fraction: float = 0.02,
    require_structure: bool = True,
    interstitial_max_share: float = 0.70,
    interstitial_min_hits: int = 2,
) -> MarkdownGatingConfig:
    """
    Factory used by run_crawl / orchestrator to create a gating config object
    that can be injected where needed.

    IMPORTANT: This function signature remains unchanged (drop-in).
    """
    return MarkdownGatingConfig(
        min_meaningful_words=min_meaningful_words,
        cookie_max_fraction=cookie_max_fraction,
        require_structure=require_structure,
        interstitial_max_share=interstitial_max_share,
        interstitial_min_hits=interstitial_min_hits,
    )
