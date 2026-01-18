from __future__ import annotations

from typing import Dict, Iterable, Tuple

from crawl4ai import CacheMode

# Keep the canonical retry exit code in one place (used by CLI defaults + exit logic).
RETRY_EXIT_CODE: int = 17

HTTP_LANG_MAP: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
}


def _primary_http_lang(code: str) -> str:
    c = (code or "").strip()
    if not c:
        raise ValueError("lang code is empty")

    lc = c.lower()
    mapped = HTTP_LANG_MAP.get(lc)
    if mapped is not None:
        return mapped

    # If already looks like a regional tag (xx-YY), pass through as-is.
    if "-" in c and len(c) >= 4:
        return c

    # Deterministic fallback.
    return f"{lc}-US"


def build_accept_language(
    *, target: str, effective: Tuple[str, ...] | Iterable[str]
) -> str:
    """
    Deterministic Accept-Language builder.

    Rules:
      - target=en  -> "en-US,en;q=0.9"
      - target!=en -> "<target-primary>,en-US;q=0.9,en;q=0.8"

    effective is included to keep the API explicit; behavior remains target-driven and stable.
    """
    t = (target or "").strip().lower()
    if not t:
        raise ValueError("lang target is empty")

    _ = tuple(effective)  # explicit dependency; not used for heuristics

    en_primary = _primary_http_lang("en")

    if t == "en":
        return f"{en_primary},en;q=0.9"

    t_primary = _primary_http_lang(t)
    return f"{t_primary},{en_primary};q=0.9,en;q=0.8"


CACHE_MODE_MAP: Dict[str, CacheMode] = {
    "enabled": CacheMode.ENABLED,
    "disabled": CacheMode.DISABLED,
    "read_only": CacheMode.READ_ONLY,
    "write_only": CacheMode.WRITE_ONLY,
    "bypass": CacheMode.BYPASS,
}
