from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="10",
    label="Electronics",
    profile_id="industry:10@v1",
    presence_addendum=(
        "Offerings include electronic components/devices/systems and electronics services.\n"
        "Signals: PCB/PCBA, sensors, semiconductors, modules, power supplies, embedded systems, IoT devices, instrumentation.\n"
        "Services: design services, EMS/contract manufacturing, testing, calibration."
    ),
    full_addendum=(
        "Products: named devices/modules, component categories, PCB assemblies, firmware/software packaged products (if positioned as product).\n"
        "Services: electronics design, EMS/PCBA manufacturing, prototyping, testing/validation, calibration, repair.\n"
        "Avoid extracting generic 'solutions' unless they clearly map to a concrete system or offering.\n"
        "Tags: 'PCB', 'PCBA', 'EMS', 'sensor', 'IoT', 'embedded', 'testing', 'calibration'."
    ),
)
