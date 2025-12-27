from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="20",
    label="IT Services",
    profile_id="industry:20@v1",
    presence_addendum=(
        "Offerings include IT services, managed services, cloud, cybersecurity, and software platforms if positioned as offerings.\n"
        "Signals: consulting, implementation, managed services, DevOps, cloud migration, MSP, SOC, cybersecurity services, ERP/CRM services.\n"
        "Ignore generic blog posts unless they list service lines."
    ),
    full_addendum=(
        "Services: managed IT, cloud migration, DevOps, app development, data/AI services, cybersecurity, helpdesk, network management.\n"
        "Products: SaaS/platforms only if clearly named and sold as a product.\n"
        "Prefer concrete offering names like 'Managed Security Services' vs vague 'Solutions'.\n"
        "Tags: 'managed services', 'cloud', 'migration', 'cybersecurity', 'DevOps', 'data analytics', 'AI'."
    ),
)
