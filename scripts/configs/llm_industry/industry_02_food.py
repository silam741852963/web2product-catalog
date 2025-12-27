from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="2",
    label="Food",
    profile_id="industry:02@v1",
    presence_addendum=(
        "Count offerings when main content mentions food/beverage products, ingredients, brands, menu lines, or manufacturing/processing services.\n"
        "Signals: 'products', 'brands', 'menu', 'flavors', 'recipes', 'ingredients', 'packaging sizes', 'nutrition', 'private label'.\n"
        "Also count food services: catering, foodservice distribution, co-packing, contract manufacturing, QA/testing if marketed."
    ),
    full_addendum=(
        "Products: packaged foods, beverages, ingredient lines, product families, brand names, SKUs/categories (snacks, dairy, sauces, frozen, bakery).\n"
        "Services: co-manufacturing/co-packing, private label, formulation/R&D, food safety testing, cold-chain distribution (if offered).\n"
        "When a page is a menu/catalog: group into a few offering entries (by category) rather than listing every single item.\n"
        "Include useful tags like: 'frozen', 'organic', 'gluten-free', 'ready-to-eat', 'bulk ingredient', 'foodservice'."
    ),
)
