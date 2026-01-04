from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndustryContext:
    """
    Runtime-provided context computed upstream (load_source.py/run.py).

    Notes:
      - industry_label MAY be empty/unknown. In that case we emit a generic instruction.
      - industry/nace are included for transparency/audit only; no inference happens here.
    """

    industry_label: str
    industry: int
    nace: int


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


def _has_label(ctx: IndustryContext) -> bool:
    return bool((ctx.industry_label or "").strip())


def build_presence_instruction(ctx: IndustryContext) -> str:
    """
    If ctx.industry_label is empty, return the base instruction only (generic mode).
    Otherwise, include the industry context block to help disambiguate terms.
    """
    if not _has_label(ctx):
        return BASE_PRESENCE_INSTRUCTION

    return (
        BASE_PRESENCE_INSTRUCTION
        + "\n\n"
        + "Industry context (already resolved upstream):\n"
        + f"- industry_label: {ctx.industry_label.strip()}\n"
        + f"- industry: {ctx.industry}\n"
        + f"- nace: {ctx.nace}\n"
        + "\n"
        + "Use the industry_label to interpret ambiguous terms on the page.\n"
        + "Do not reinterpret or re-derive the industry_label."
    )


def build_full_instruction(ctx: IndustryContext) -> str:
    """
    If ctx.industry_label is empty, return the base instruction only (generic mode).
    Otherwise, include the industry context block to help disambiguate terms.
    """
    if not _has_label(ctx):
        return BASE_FULL_INSTRUCTION

    return (
        BASE_FULL_INSTRUCTION
        + "\n\n"
        + "Industry context (already resolved upstream):\n"
        + f"- industry_label: {ctx.industry_label.strip()}\n"
        + f"- industry: {ctx.industry}\n"
        + f"- nace: {ctx.nace}\n"
        + "\n"
        + "Use the industry_label to interpret ambiguous terms on the page.\n"
        + "Do not reinterpret or re-derive the industry_label."
    )
