from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="16",
    label="Transport & Logistics",
    profile_id="industry:16@v1",
    presence_addendum=(
        "Offerings include logistics/transport services and related solutions.\n"
        "Signals: freight, shipping, trucking, air/sea freight, warehousing, 3PL, last-mile, customs brokerage, tracking.\n"
        "Ignore generic 'we deliver excellence' unless service types are specified."
    ),
    full_addendum=(
        "Services: freight forwarding, trucking, ocean/air shipping, warehousing, 3PL, fulfillment, last-mile delivery, customs brokerage.\n"
        "Products: logistics platforms/software only if marketed as a product.\n"
        "When many services are listed, group into 4–8 offerings (e.g., 'Freight forwarding', 'Warehousing & fulfillment').\n"
        "Tags: '3PL', 'warehousing', 'customs', 'freight forwarding', 'last-mile', 'cold chain' (if present)."
    ),
)
