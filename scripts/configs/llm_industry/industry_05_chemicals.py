from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="5",
    label="Chemicals",
    profile_id="industry:05@v1",
    presence_addendum=(
        "Offerings include chemical products/formulations and chemical services.\n"
        "Signals: resins, solvents, adhesives, coatings, surfactants, catalysts, additives, specialty chemicals, SDS/TDS, formulations.\n"
        "Services count if explicit: custom synthesis, toll manufacturing, blending, lab testing, formulation support."
    ),
    full_addendum=(
        "Products: specific chemical families or named formulations (e.g., epoxy resin systems, polyurethane coatings, water treatment chemicals).\n"
        "Services: custom synthesis, contract/toll manufacturing, blending, analytical testing, technical support for applications.\n"
        "Prefer grouping into a few offerings by application area if the page lists many chemicals.\n"
        "Tags: 'specialty chemicals', 'additives', 'resins', 'coatings', 'adhesives', 'custom synthesis', 'toll manufacturing'."
    ),
)
