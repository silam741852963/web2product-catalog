from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="19",
    label="Telecom",
    profile_id="industry:19@v1",
    presence_addendum=(
        "Offerings include telecom connectivity services and plans.\n"
        "Signals: mobile plans, broadband, fiber, voice, SIP trunking, data centers/colocation (if telecom operator), IoT connectivity.\n"
        "Also count enterprise services: SD-WAN, managed networks, unified communications."
    ),
    full_addendum=(
        "Services: mobile/voice/data plans, broadband/fiber internet, business connectivity, SD-WAN, managed network services, UCaaS.\n"
        "Products: routers/devices only if sold as products (otherwise part of service).\n"
        "If pricing/plan tables exist, summarize as 'Mobile plans' rather than enumerating every plan.\n"
        "Tags: 'fiber', 'broadband', 'mobile', 'SIP', 'SD-WAN', 'managed network', 'IoT connectivity'."
    ),
)
