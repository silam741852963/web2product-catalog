from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="8",
    label="Minerals",
    profile_id="industry:08@v1",
    presence_addendum=(
        "Offerings include non-metallic minerals/materials and related processing services.\n"
        "Signals: aggregates, sand, gravel, limestone, gypsum, clay, silica, cement raw materials, industrial minerals.\n"
        "Services may include crushing/screening, materials testing, delivery (if offered)."
    ),
    full_addendum=(
        "Products: named mineral products/grades, aggregates categories, industrial mineral powders, cementitious materials.\n"
        "Services: crushing/screening/processing, custom blends, materials testing, delivery/logistics (only if sold).\n"
        "If the page is project-based (quarry supplying a region), still extract supply as product offering.\n"
        "Tags: 'aggregates', 'industrial minerals', 'crushed stone', 'screening', 'bulk supply'."
    ),
)
