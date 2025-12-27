from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="6",
    label="Pharma",
    profile_id="industry:06@v1",
    presence_addendum=(
        "Offerings include pharmaceuticals, APIs, medical products, and pharma services.\n"
        "Signals: API, formulation, dosage forms, generic drugs, pipeline, therapeutics, clinical, CDMO/CMO, GMP.\n"
        "Do NOT invent indications; only extract therapeutic areas if explicitly stated."
    ),
    full_addendum=(
        "Products: named drugs, product lines (e.g., tablets/capsules), APIs/intermediates, OTC categories, medical consumables if present.\n"
        "Services: CDMO/CMO, GMP manufacturing, packaging, analytical testing, stability studies, clinical supply, regulatory support.\n"
        "If pipeline is described, treat it as 'service/solution' ONLY if they offer development services; otherwise extract marketed products.\n"
        "Tags: 'API', 'GMP', 'CDMO', 'formulation', 'analytical testing', 'packaging'."
    ),
)
