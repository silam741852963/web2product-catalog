from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="22",
    label="Real Estate",
    profile_id="industry:22@v1",
    presence_addendum=(
        "Offerings include property sales/leasing, property management, development, brokerage.\n"
        "Signals: listings, lease, rent, buy/sell, commercial/residential, property management, appraisal.\n"
        "Ignore neighborhood guides unless tied to services."
    ),
    full_addendum=(
        "Services: brokerage, leasing, property management, development services, appraisal/valuation if offered.\n"
        "Products: specific property categories (e.g., 'industrial warehouses for lease') as offerings; do not list every listing.\n"
        "If portfolio pages exist, extract as 'Commercial property leasing' etc.\n"
        "Tags: 'property management', 'brokerage', 'leasing', 'commercial', 'residential', 'development'."
    ),
)
