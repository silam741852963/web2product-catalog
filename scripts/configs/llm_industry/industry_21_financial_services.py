from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="21",
    label="Financial Services",
    profile_id="industry:21@v1",
    presence_addendum=(
        "Offerings include financial products/services.\n"
        "Signals: loans, credit, insurance, brokerage, asset management, payments, cards, investment products, advisory.\n"
        "Do not invent rates/terms; only use what is shown."
    ),
    full_addendum=(
        "Services: lending, insurance services, investment advisory, brokerage, wealth management, payment processing, merchant services.\n"
        "Products: specific financial products (e.g., 'SME loans', 'mortgage loans', 'business insurance') as offerings.\n"
        "If they list many products, group into a few category offerings.\n"
        "Tags: 'lending', 'insurance', 'wealth management', 'payments', 'brokerage', 'advisory'."
    ),
)
