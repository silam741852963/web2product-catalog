from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Pattern, Protocol, Set

from .base_en import DEFAULT_LANG_SPEC
from .merge import (
    compile_or_join,
    iter_str_values,
    normalize_set_map,
    union_lang_sets,
    union_nested_lang_map_sets,
    union_strings_in_order,
)
from .registry import get_lang_additions, resolve_effective_langs, validate_lang_code
from .types import LanguageSpec


class LanguageSpecStrategy(Protocol):
    def load_spec(self, code: str) -> LanguageSpec:  # pragma: no cover
        ...


@dataclass(frozen=True)
class DefaultLanguageSpecStrategy(LanguageSpecStrategy):
    """
    Deterministic loader:

    - code == "en": returns DEFAULT_LANG_SPEC (full base spec)
    - code != "en": returns additive LANG_SPEC delta from configs.language.lang.<code>
      (or {} if not implemented)

    This ensures the target language cannot overwrite English by accident.
    """

    def load_spec(self, code: str) -> LanguageSpec:
        c = validate_lang_code(code)
        if c == "en":
            return dict(DEFAULT_LANG_SPEC)
        return get_lang_additions(c)


@dataclass
class LanguageConfigFactory:
    strategy: LanguageSpecStrategy

    active_target_code: str = "en"
    active_codes: tuple[str, ...] = ("en",)
    active_specs: Dict[str, LanguageSpec] = field(default_factory=dict)

    def set_language(self, code: str | None = "en") -> None:
        target = validate_lang_code(code)
        codes = resolve_effective_langs(target)

        specs: Dict[str, LanguageSpec] = {}
        for c in codes:
            specs[c] = self.strategy.load_spec(c)

        # Invariants:
        # - specs["en"] is full base spec
        # - specs[target] (if any) is delta-only additions
        self.active_target_code = target
        self.active_codes = codes
        self.active_specs = specs

    # --- Introspection --------------------------------------------------------

    def effective_langs(self) -> tuple[str, ...]:
        return self.active_codes

    def allowed_path_langs(self) -> Set[str]:
        return set(self.active_codes)

    # --- Union vocab/patterns across effective langs -------------------------

    def interstitial_patterns(self) -> List[str]:
        parts: List[Iterable[str]] = []
        for c in self.active_codes:
            spec = self.active_specs.get(c, {})
            parts.append(iter_str_values(spec.get("INTERSTITIAL_PATTERNS", [])))
        return union_strings_in_order(parts)

    def cookieish_patterns(self) -> List[str]:
        parts: List[Iterable[str]] = []
        for c in self.active_codes:
            spec = self.active_specs.get(c, {})
            parts.append(iter_str_values(spec.get("COOKIEISH_PATTERNS", [])))
        return union_strings_in_order(parts)

    def product_tokens(self) -> List[str]:
        parts: List[Iterable[str]] = []
        for c in self.active_codes:
            spec = self.active_specs.get(c, {})
            parts.append(iter_str_values(spec.get("PRODUCT_TOKENS", [])))
        return union_strings_in_order(parts)

    def exclude_tokens(self) -> List[str]:
        parts: List[Iterable[str]] = []
        for c in self.active_codes:
            spec = self.active_specs.get(c, {})
            parts.append(iter_str_values(spec.get("EXCLUDE_TOKENS", [])))
        return union_strings_in_order(parts)

    def stopwords(self) -> Set[str]:
        return union_lang_sets(self.active_specs, self.active_codes, "STOPWORDS")

    def cta_keywords(self) -> Set[str]:
        return union_lang_sets(self.active_specs, self.active_codes, "CTA_KEYWORDS")

    def interstitial_re(self) -> Pattern[str]:
        return compile_or_join(self.interstitial_patterns(), flags=re.IGNORECASE)

    def cookieish_re(self) -> Pattern[str]:
        return compile_or_join(self.cookieish_patterns(), flags=re.IGNORECASE)

    def product_tokens_re(self) -> Pattern[str]:
        toks = self.product_tokens()
        if not toks:
            return re.compile(r"(?!x)x")
        token_rx = r"\b(" + "|".join(re.escape(t) for t in toks) + r")\b"
        return re.compile(token_rx, flags=re.IGNORECASE)

    def default_product_bm25_query(self) -> str:
        return " ".join(self.product_tokens())

    # --- PATH_LANG_TOKENS is treated as shared mapping, unioned across specs ---

    def path_lang_tokens(self) -> Dict[str, Set[str]]:
        """
        Return normalized PATH_LANG_TOKENS mapping (dict[lang_code] -> set(tokens)).
        Deterministic: union across active specs by key.
        """
        merged: Dict[str, Set[str]] = {}

        for c in self.active_codes:
            spec = self.active_specs.get(c, {})
            m = normalize_set_map(spec.get("PATH_LANG_TOKENS", {}))
            for k, vals in m.items():
                if k not in merged:
                    merged[k] = set(vals)
                else:
                    merged[k] |= set(vals)

        return merged

    # --- URL allow/deny unions for effective langs ----------------------------

    def tld_allow_for_effective_langs(self) -> Set[str]:
        return union_nested_lang_map_sets(
            self.active_specs, self.active_codes, "LANG_TLD_ALLOW"
        )

    def tld_deny_for_effective_langs(self) -> Set[str]:
        return union_nested_lang_map_sets(
            self.active_specs, self.active_codes, "LANG_TLD_DENY"
        )

    def host_allow_suffixes_for_effective_langs(self) -> Set[str]:
        return union_nested_lang_map_sets(
            self.active_specs, self.active_codes, "LANG_HOST_ALLOW_SUFFIXES"
        )

    def host_block_suffixes_for_effective_langs(self) -> Set[str]:
        return union_nested_lang_map_sets(
            self.active_specs, self.active_codes, "LANG_HOST_BLOCK_SUFFIXES"
        )


default_language_strategy = DefaultLanguageSpecStrategy()
default_language_factory = LanguageConfigFactory(strategy=default_language_strategy)

# Set default at import time (safe: remains "en")
default_language_factory.set_language("en")

__all__ = [
    "LanguageSpecStrategy",
    "DefaultLanguageSpecStrategy",
    "LanguageConfigFactory",
    "default_language_strategy",
    "default_language_factory",
]
