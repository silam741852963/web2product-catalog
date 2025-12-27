from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="23",
    label="Professional Services",
    profile_id="industry:23@v1",
    presence_addendum=(
        "Offerings include professional services like consulting, legal, accounting, engineering, marketing, staffing.\n"
        "Signals: service lines, practice areas, 'what we do', casework, audits, advisory, compliance.\n"
        "Do not treat team bios as offerings."
    ),
    full_addendum=(
        "Services: clearly named service lines (tax advisory, audit, legal services, engineering consulting, marketing services, staffing/recruiting).\n"
        "Products: toolkits or software only if explicitly sold.\n"
        "Prefer grouping by practice area (3–8 offerings) rather than listing every micro-service.\n"
        "Tags: 'consulting', 'advisory', 'audit', 'compliance', 'legal', 'engineering', 'staffing'."
    ),
)
