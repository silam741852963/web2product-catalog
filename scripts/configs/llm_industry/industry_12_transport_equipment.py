from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="12",
    label="Transport Equipment",
    profile_id="industry:12@v1",
    presence_addendum=(
        "Offerings include vehicles/transport equipment, parts, systems, and related services.\n"
        "Signals: automotive components, rail equipment, aerospace parts, marine equipment, trailers, fleets, powertrain, interiors.\n"
        "Services: maintenance, fleet management, retrofits, certification."
    ),
    full_addendum=(
        "Products: specific components/systems (brakes, axles, body parts), equipment categories, aftermarket parts.\n"
        "Services: maintenance/repair, fleet services, retrofitting, inspection/certification if offered.\n"
        "If they are a dealer, extract sales + service offerings separately.\n"
        "Tags: 'aftermarket', 'OEM', 'fleet', 'maintenance', 'automotive', 'rail', 'marine', 'aerospace'."
    ),
)
