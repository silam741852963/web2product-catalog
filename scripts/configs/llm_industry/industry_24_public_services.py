from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="24",
    label="Public Services",
    profile_id="industry:24@v1",
    presence_addendum=(
        "Offerings include public/municipal services or public-sector deliverables.\n"
        "Signals: government services, permits/licensing, public programs, community services, public safety, education, health services.\n"
        "Only mark presence=1 if the page describes a service/program delivered to the public or to organizations."
    ),
    full_addendum=(
        "Services: public programs, administrative services, permitting/licensing, public utilities run by agencies, community/health/education services.\n"
        "Products: publications/forms only if positioned as deliverables; usually treat as part of a service.\n"
        "Avoid extracting purely informational pages without an actual service described.\n"
        "Tags: 'government', 'permits', 'licensing', 'public program', 'community services'."
    ),
)
