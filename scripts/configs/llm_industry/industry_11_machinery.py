from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="11",
    label="Machinery",
    profile_id="industry:11@v1",
    presence_addendum=(
        "Offerings include industrial machinery/equipment, machine lines, and industrial services.\n"
        "Signals: equipment models/series, machines, tools, automation equipment, parts, spare parts, maintenance services.\n"
        "Also count rental/leasing if explicitly marketed."
    ),
    full_addendum=(
        "Products: machine models/series, equipment categories, spare parts/consumables, tooling.\n"
        "Services: installation, commissioning, maintenance, repair, retrofits, operator training.\n"
        "If the page lists industries served, only use them as tags, not as offerings.\n"
        "Tags: 'industrial equipment', 'automation', 'spare parts', 'maintenance', 'installation', 'training'."
    ),
)
