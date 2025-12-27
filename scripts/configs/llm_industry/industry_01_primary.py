from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="1",
    label="Primary",
    profile_id="industry:01@v1",
    presence_addendum=(
        "Treat as offerings when main content mentions selling/producing/extracting primary goods or operating assets.\n"
        "Common offering signals: crop/produce, livestock, feed, seeds, fertilizer sales, timber/forestry products, mining/ore, quarrying, crude materials.\n"
        "Also count services if explicitly offered: drilling services, contract farming, harvesting, land management, assay/testing, exploration services.\n"
        "Do NOT count 'sustainability/ESG' pages unless they clearly describe goods/services provided."
    ),
    full_addendum=(
        "Extract offerings as concrete goods/services, not generic capabilities.\n"
        "Products: specific commodities/materials (e.g., corn, alfalfa hay, hardwood logs, aggregates, limestone, copper concentrate), branded lines, seed varieties.\n"
        "Services: contract harvesting, drilling, exploration, land leasing (if marketed), storage/handling (if offered), transport of own commodities (only if sold as service).\n"
        "If content describes owned assets (mines, farms) + outputs, convert outputs into product offerings.\n"
        "Prefer names from headings (e.g., 'Products', 'Our Crops', 'Metals & Minerals')."
    ),
)
