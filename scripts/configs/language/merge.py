from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Pattern, Sequence, Set


def iter_str_values(v: Any) -> Iterable[str]:
    if v is None:
        return []
    if isinstance(v, (set, list, tuple)):
        return (str(x) for x in v)
    return [str(v)]


def union_strings_in_order(parts: Sequence[Iterable[str]]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for it in parts:
        for raw in it:
            s = (raw or "").strip()
            if not s:
                continue
            s_l = s.lower()
            if s_l in seen:
                continue
            seen.add(s_l)
            out.append(s_l)
    return out


def compile_or_join(patterns: Iterable[str], flags: int = 0) -> Pattern[str]:
    pats = [p for p in patterns if str(p).strip()]
    if not pats:
        return re.compile(r"(?!x)x")
    return re.compile("|".join(f"(?:{p})" for p in pats), flags=flags)


def normalize_set_map(m: Any) -> Dict[str, Set[str]]:
    """
    Normalize mapping[str, iterable[str]] into dict[str, set[str]] with lowercase values.
    """
    if not isinstance(m, dict):
        return {}
    out: Dict[str, Set[str]] = {}
    for k, v in m.items():
        kk = str(k).strip().lower()
        if not kk:
            continue
        vals = {str(x).strip().lower() for x in iter_str_values(v) if str(x).strip()}
        out[kk] = vals
    return out


def union_lang_sets(
    spec_by_code: Mapping[str, Mapping[str, Any]],
    codes: Sequence[str],
    key: str,
) -> Set[str]:
    parts: List[Iterable[str]] = []
    for c in codes:
        spec = spec_by_code.get(c, {})
        parts.append(iter_str_values(spec.get(key, [])))
    return set(union_strings_in_order(parts))


def union_nested_lang_map_sets(
    spec_by_code: Mapping[str, Mapping[str, Any]],
    codes: Sequence[str],
    key: str,
) -> Set[str]:
    """
    Union of (spec[key][lang] ...) across effective langs.
    Example: LANG_TLD_ALLOW / LANG_HOST_ALLOW_SUFFIXES
    """
    out: Set[str] = set()
    for c in codes:
        spec = spec_by_code.get(c, {})
        m = spec.get(key, {})
        if not isinstance(m, dict):
            continue
        v = m.get(c, None)
        out |= {
            str(x).strip().lower().lstrip(".")
            for x in iter_str_values(v)
            if str(x).strip()
        }
    return out
