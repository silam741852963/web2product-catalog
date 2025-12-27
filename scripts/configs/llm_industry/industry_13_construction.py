from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="13",
    label="Construction",
    profile_id="industry:13@v1",
    presence_addendum=(
        "Offerings include construction services, contracting, project delivery, and construction products/materials.\n"
        "Signals: general contractor, design-build, EPC, civil works, roofing, MEP, renovations, project portfolio, 'services'.\n"
        "Also count products if sold: building materials, prefabricated modules."
    ),
    full_addendum=(
        "Services: general contracting, design-build, EPC, civil/structural works, MEP installation, renovation, maintenance.\n"
        "Products: prefabricated buildings, modular components, building materials if explicitly marketed.\n"
        "Do not treat 'projects we completed' as offerings unless the service type is stated—extract the service type.\n"
        "Tags: 'design-build', 'EPC', 'civil', 'MEP', 'renovation', 'maintenance'."
    ),
)
