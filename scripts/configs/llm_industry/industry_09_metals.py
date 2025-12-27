from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="9",
    label="Metals",
    profile_id="industry:09@v1",
    presence_addendum=(
        "Offerings include metals, alloys, fabricated metal products, and metalworking services.\n"
        "Signals: steel, aluminum, copper, alloys, sheet/plate/bar/pipe, foundry, casting, forging, machining, welding, heat treatment.\n"
        "Do not count 'quality policy' pages unless actual products/services are described."
    ),
    full_addendum=(
        "Products: metal forms (sheet/plate/bar/tube), alloy families, castings/forgings, fabricated assemblies.\n"
        "Services: CNC machining, welding, heat treatment, surface finishing, fabrication, custom metalwork.\n"
        "If they are a distributor, extract 'metal distribution/supply' and key categories.\n"
        "Tags: 'CNC machining', 'fabrication', 'casting', 'forging', 'steel', 'aluminum', 'heat treatment'."
    ),
)
