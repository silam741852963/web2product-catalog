from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="14",
    label="Utilities",
    profile_id="industry:14@v1",
    presence_addendum=(
        "Offerings include utility services (electricity, gas, water, waste), plans/tariffs, and customer solutions.\n"
        "Signals: power generation/supply, transmission/distribution, water supply, wastewater, waste management, billing plans.\n"
        "Also count B2B services: energy management, renewable procurement, grid services, metering."
    ),
    full_addendum=(
        "Services: electricity/gas/water supply, waste collection/treatment, district heating, grid services, connection services.\n"
        "Products: meters/devices only if the company sells them (otherwise treat as part of service).\n"
        "If there are customer programs (green energy plans, demand response), extract as service offerings.\n"
        "Tags: 'electricity', 'water', 'wastewater', 'renewables', 'metering', 'demand response', 'energy management'."
    ),
)
