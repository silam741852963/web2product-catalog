from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, Mapping, Tuple

from .base_en import DEFAULT_LANG_SPEC
from .types import LanguageSpec

_LANG_DIR = Path(__file__).parent / "lang"


def _normalized_lang(code: str | None) -> str:
    c = (code or "").strip().lower()
    if not c:
        raise ValueError("lang code is empty")
    return c


def discover_lang_modules() -> Dict[str, LanguageSpec]:
    """
    Discover configs.language.lang.<code> modules.

    Deterministic rules:
    - Scan configs/language/lang/*.py
    - Ignore __init__.py
    - Import each module by full module path
    - Require:
      - LANG_CODE: str, must equal filename stem
      - LANG_SPEC: dict
    - Fail loudly on any invalid module.
    """
    if not _LANG_DIR.exists():
        return {}

    files = sorted(p for p in _LANG_DIR.glob("*.py") if p.name != "__init__.py")
    out: Dict[str, LanguageSpec] = {}

    for path in files:
        code = path.stem.strip().lower()
        if not code:
            raise ValueError(f"invalid lang module filename: {path.name!r}")

        mod_name = f"configs.language.lang.{code}"
        mod = importlib.import_module(mod_name)

        lang_code = getattr(mod, "LANG_CODE", None)
        if not isinstance(lang_code, str) or lang_code.strip().lower() != code:
            raise ValueError(
                f"{mod_name}: LANG_CODE must be {code!r}, got {lang_code!r}"
            )

        lang_spec = getattr(mod, "LANG_SPEC", None)
        if not isinstance(lang_spec, dict):
            raise ValueError(f"{mod_name}: LANG_SPEC must be a dict")

        out[code] = lang_spec

    return out


# Cache at module import: deterministic. If a lang module is broken, fail loudly early.
_DISCOVERED: Dict[str, LanguageSpec] = discover_lang_modules()


def get_lang_additions(code: str) -> LanguageSpec:
    """
    Return per-language additive spec (delta), or empty dict if not implemented.
    Deterministic: never returns None.
    """
    c = _normalized_lang(code)
    if c == "en":
        return {}
    return dict(_DISCOVERED.get(c, {}))


def validate_lang_code(code: str | None) -> str:
    """
    Validate and normalize a language code.

    Acceptance:
    - "en"
    - any discovered module code in configs.language.lang.*
    - any code in DEFAULT_LANG_SPEC["LANG_CODES"] or ["REGIONAL_LANG_CODES"]

    Reject:
    - empty / whitespace
    - unknown codes not in any of the above
    """
    c = _normalized_lang(code)

    if c == "en":
        return "en"

    known = DEFAULT_LANG_SPEC.get("LANG_CODES", set()) or set()
    regional = DEFAULT_LANG_SPEC.get("REGIONAL_LANG_CODES", set()) or set()

    if c in _DISCOVERED:
        return c
    if c in known:
        return c
    if c in regional:
        return c

    raise ValueError(
        f"unknown lang code: {c!r} (not discovered in configs.language.lang and not in known code sets)"
    )


def resolve_effective_langs(target_lang: str | None) -> Tuple[str, ...]:
    """
    Effective language set:
    - target "en" -> ("en",)
    - else -> ("en", "<target>")

    Stable order, en first.
    """
    t = validate_lang_code(target_lang)
    if t == "en":
        return ("en",)
    return ("en", t)
