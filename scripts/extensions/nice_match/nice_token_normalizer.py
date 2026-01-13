from __future__ import annotations

import re
from dataclasses import dataclass

from extensions.nice_match.nice_loader import NiceRow


_SPECIAL_STRIP_RE = re.compile(r"[*†‡•]+")
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


_PREPOSITIONS = {
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "from",
    "with",
    "without",
    "into",
    "onto",
    "under",
    "over",
    "within",
    "between",
    "among",
    "around",
    "through",
    "during",
    "after",
    "before",
}


@dataclass(frozen=True, slots=True)
class NiceToken:
    token_id: str  # Basic No. or Place / № de base ou endroit

    cl: str
    x: str
    prop_no: str
    action_en: str

    raw_goods_2024: str
    note: str
    aliases: list[str]


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_special(s: str) -> str:
    s2 = _SPECIAL_STRIP_RE.sub("", s)
    # normalize quotes
    s2 = re.sub(r"[“”]", '"', s2)
    s2 = re.sub(r"[‘’]", "'", s2)
    return s2.strip()


def _split_slash_variants(s: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*/\s*", s) if p.strip()]


def _contains_preposition_words(s: str) -> bool:
    words = set(re.findall(r"[A-Za-z]+", s.lower()))
    for w in _PREPOSITIONS:
        if w in words:
            return True
    return False


def _handle_brackets(base: str) -> tuple[str, list[str], list[str]]:
    """
    Returns (text_without_brackets, notes_from_brackets, aliases_from_brackets).

    Rules:
      - bracket content is removed from the base pattern for matching.
      - if there is suffix text after the bracket: bracket content forms an alias with suffix,
        and does NOT become note.
      - if bracket is at end:
          - default: bracket content becomes note
          - exception: if base has >=2 words AND bracket content contains a preposition-word,
            discard bracket content entirely (no note, no alias).
    """
    m = _BRACKET_RE.search(base)
    if m is None:
        return base, [], []

    content = (m.group(1) or "").strip()
    if content == "":
        without = _norm_spaces((base[: m.start()] + " " + base[m.end() :]).strip())
        return without, [], []

    before = base[: m.start()].strip()
    after = base[m.end() :].strip()

    without = _norm_spaces((before + " " + after).strip())

    # If there's suffix text, treat bracket content as alias-with-suffix
    if after != "":
        alias = _norm_spaces(f"{content} {after}")
        return without, [], [alias]

    base_word_count = len([w for w in re.findall(r"[A-Za-z0-9]+", before) if w.strip()])
    if base_word_count >= 2 and _contains_preposition_words(content):
        return without, [], []

    return without, [content], []


def _comma_patterns(s: str) -> tuple[list[str], list[str]]:
    """
    Returns (patterns, notes).

    - 1 comma: treat as note boundary: "X, note" -> pattern "X", note "note"
    - 2 commas: generate alternatives:
        A C
        B C
        A B C
      (no note)
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) <= 1:
        return [s.strip()], []

    if len(parts) == 2:
        base = parts[0]
        note = parts[1]
        return [base], [note]

    if len(parts) == 3:
        a, b, c = parts
        return [f"{a} {c}".strip(), f"{b} {c}".strip(), f"{a} {b} {c}".strip()], []

    return [" ".join(parts).strip()], []


def build_tokens(rows: list[NiceRow]) -> list[NiceToken]:
    out: list[NiceToken] = []

    for r in rows:
        raw = _strip_special(r.en_goods_2024.strip())
        variants = _split_slash_variants(raw)
        if len(variants) == 0:
            continue

        all_aliases: list[str] = []
        notes: list[str] = []

        for v in variants:
            v2 = _strip_special(v)
            w, bracket_notes, bracket_aliases = _handle_brackets(v2)
            notes.extend(bracket_notes)

            patterns, comma_notes = _comma_patterns(w)
            notes.extend(comma_notes)

            for p in patterns:
                p2 = _norm_spaces(_strip_special(p))
                if p2 != "":
                    all_aliases.append(p2)

            for a in bracket_aliases:
                a2 = _norm_spaces(_strip_special(a))
                if a2 != "":
                    all_aliases.append(a2)

        # Dedupe aliases (case-insensitive) preserving order
        seen: set[str] = set()
        aliases: list[str] = []
        for a in all_aliases:
            k = a.lower()
            if k in seen:
                continue
            seen.add(k)
            aliases.append(a)

        note_str = " | ".join([n.strip() for n in notes if n.strip()])

        if len(aliases) == 0:
            continue

        out.append(
            NiceToken(
                token_id=r.token_id,
                cl=r.cl,
                x=r.x,
                prop_no=r.prop_no,
                action_en=r.action_en,
                raw_goods_2024=r.en_goods_2024,
                note=note_str,
                aliases=aliases,
            )
        )

    if len(out) == 0:
        raise RuntimeError("No NICE tokens were built from NICE rows.")
    return out
