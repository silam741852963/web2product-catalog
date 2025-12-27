from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="3",
    label="Textiles",
    profile_id="industry:03@v1",
    presence_addendum=(
        "Offerings include textiles/apparel materials and related manufacturing services.\n"
        "Signals: fabrics, yarn, fiber, knit/woven, dyeing/printing/finishing, garments, uniforms, technical textiles, nonwovens.\n"
        "Do not treat 'about our craftsmanship' as offerings unless specific products/services are named."
    ),
    full_addendum=(
        "Products: fabric types (cotton denim, polyester knit), yarn/fiber lines, nonwoven rolls, apparel categories, uniform lines, technical textiles.\n"
        "Services: OEM/ODM garment manufacturing, weaving/knitting, dyeing, printing, finishing, pattern making, sampling.\n"
        "Capture specs if clearly stated (GSM, composition, coatings) but keep descriptions short and grounded.\n"
        "Tags: 'OEM/ODM', 'dyeing', 'printing', 'technical textile', 'nonwoven', 'uniforms'."
    ),
)
