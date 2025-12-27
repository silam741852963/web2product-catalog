from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="4",
    label="Wood & Paper",
    profile_id="industry:04@v1",
    presence_addendum=(
        "Offerings include wood products, paper/pulp products, packaging, and converting services.\n"
        "Signals: lumber, plywood, veneer, MDF/particleboard, pulp, paper grades, corrugated, cartons, tissue, printing paper.\n"
        "Also count services: converting, cutting, laminating, custom packaging design, printing (if offered)."
    ),
    full_addendum=(
        "Products: lumber types, engineered wood panels, pulp grades, paper grades, packaging formats (corrugated boxes, cartons), tissue products.\n"
        "Services: paper converting, custom packaging, printing on packaging, kitting/assembly, logistics for packaging supply.\n"
        "If sustainability claims appear, include only if tied to a product line (e.g., 'recycled kraft paper').\n"
        "Tags: 'corrugated', 'pulp', 'kraft', 'engineered wood', 'converting', 'packaging'."
    ),
)
