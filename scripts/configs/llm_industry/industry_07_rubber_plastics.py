from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="7",
    label="Rubber & Plastics",
    profile_id="industry:07@v1",
    presence_addendum=(
        "Offerings include plastic/rubber materials, molded parts, packaging, and manufacturing services.\n"
        "Signals: injection molding, extrusion, blow molding, thermoforming, elastomers, seals/gaskets, films, pellets, compounds.\n"
        "Count services if explicit: custom molding, tooling, design-for-manufacture."
    ),
    full_addendum=(
        "Products: plastic components, rubber seals, hoses, films/sheets, pellets/compounds, packaging items.\n"
        "Services: injection molding/extrusion as a service, custom compounding, tooling/mold design, prototyping.\n"
        "If they list processes + end-markets, convert into offerings like 'Injection-molded components' rather than vague 'manufacturing'.\n"
        "Tags: 'injection molding', 'extrusion', 'thermoforming', 'elastomer', 'custom tooling', 'compounding'."
    ),
)
