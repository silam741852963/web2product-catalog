from __future__ import annotations

import re
from importlib import import_module
from typing import Dict, Optional

from .base import IndustryLLMProfile
from .industry_default import PROFILE as DEFAULT_PROFILE

# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #

_NAN_LIKE = {"", "none", "null", "nan", "na", "n/a", "unknown", "undef", "undefined"}
_FLOATY_RE = re.compile(r"^\s*(\d+)(?:\.0+)?\s*$")


def normalize_industry_code(v: object) -> str:
    """
    Normalize industry code to a stable string key:
      - None / NaN-like => "default"
      - "9.0" => "9"
      - 9 => "9"
      - " 09 " => "9"
    """
    if v is None:
        return "default"

    s = str(v).strip()
    if not s:
        return "default"

    s_lower = s.lower()
    if s_lower in _NAN_LIKE:
        return "default"

    m = _FLOATY_RE.match(s)
    if m:
        try:
            return str(int(m.group(1)))
        except Exception:
            return "default"

    # handle "09"
    if s.isdigit():
        try:
            return str(int(s))
        except Exception:
            return s

    return s


# --------------------------------------------------------------------------- #
# Module mapping (file names) — populate addendums later in each module.
# --------------------------------------------------------------------------- #

_CODE_TO_MODULE: Dict[str, str] = {
    "1": "industry_01_primary",
    "2": "industry_02_food",
    "3": "industry_03_textiles",
    "4": "industry_04_wood_paper",
    "5": "industry_05_chemicals",
    "6": "industry_06_pharma",
    "7": "industry_07_rubber_plastics",
    "8": "industry_08_minerals",
    "9": "industry_09_metals",
    "10": "industry_10_electronics",
    "11": "industry_11_machinery",
    "12": "industry_12_transport_equipment",
    "13": "industry_13_construction",
    "14": "industry_14_utilities",
    "15": "industry_15_trade",
    "16": "industry_16_transport_logistics",
    "17": "industry_17_hospitality",
    "18": "industry_18_media",
    "19": "industry_19_telecom",
    "20": "industry_20_it_services",
    "21": "industry_21_financial_services",
    "22": "industry_22_real_estate",
    "23": "industry_23_professional_services",
    "24": "industry_24_public_services",
}


def get_industry_profile(industry_code: object) -> IndustryLLMProfile:
    code = normalize_industry_code(industry_code)
    if code == "default":
        return DEFAULT_PROFILE

    mod_name = _CODE_TO_MODULE.get(code)
    if not mod_name:
        return DEFAULT_PROFILE

    try:
        mod = import_module(f"{__package__}.{mod_name}")
        prof = getattr(mod, "PROFILE", None)
        if isinstance(prof, IndustryLLMProfile):
            return prof
    except Exception:
        pass

    return DEFAULT_PROFILE
