from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Base (shared) instructions — DRY
# --------------------------------------------------------------------------- #

BASE_PRESENCE_INSTRUCTION = (
    "Classify if the page MAIN CONTENT includes any offerings (products/brands/product lines or services/solutions).\n"
    "Ignore cookie/privacy banners, navigation/menus, and legal footers.\n"
    'Return ONLY JSON: {"r":1} or {"r":0}.'
)

BASE_FULL_INSTRUCTION = (
    "Extract the company's offerings from the page MAIN CONTENT.\n"
    "Ignore cookie/privacy banners, navigation/menus, and legal footers.\n"
    "Do NOT invent offerings.\n"
    "Return ONLY valid JSON matching the provided schema."
)


@dataclass(frozen=True)
class IndustryLLMProfile:
    """
    Industry profile: ONLY unique details per industry.
    Keep these addendums short and concrete (no schema duplication).
    """

    code: str
    label: str
    profile_id: str

    # These are appended to the shared base instructions.
    presence_addendum: str = ""
    full_addendum: str = ""


def _norm(s: str) -> str:
    return (s or "").strip()


def compose_presence_instruction(profile: IndustryLLMProfile) -> str:
    add = _norm(profile.presence_addendum)
    if not add:
        return BASE_PRESENCE_INSTRUCTION
    return (
        BASE_PRESENCE_INSTRUCTION
        + "\n\n"
        + f"Industry context: {profile.label} (code={profile.code}).\n"
        + add
    )


def compose_full_instruction(profile: IndustryLLMProfile) -> str:
    add = _norm(profile.full_addendum)
    if not add:
        return BASE_FULL_INSTRUCTION
    return (
        BASE_FULL_INSTRUCTION
        + "\n\n"
        + f"Industry context: {profile.label} (code={profile.code}).\n"
        + add
    )
