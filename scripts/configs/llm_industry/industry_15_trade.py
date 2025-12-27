from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="15",
    label="Trade",
    profile_id="industry:15@v1",
    presence_addendum=(
        "Offerings include distribution/wholesale/retail product categories and trade services.\n"
        "Signals: catalog, product categories, brands carried, 'shop', 'store', 'wholesale', 'distributor'.\n"
        "Services: sourcing, procurement, fulfillment, after-sales support if stated."
    ),
    full_addendum=(
        "Products: main categories and key brand/product lines (group; don’t enumerate every SKU).\n"
        "Services: distribution, wholesale supply, procurement/sourcing, fulfillment, after-sales/installation if offered.\n"
        "If the site is purely a storefront, extract the top-level categories as offerings.\n"
        "Tags: 'wholesale', 'distribution', 'catalog', 'fulfillment', 'retail'."
    ),
)
