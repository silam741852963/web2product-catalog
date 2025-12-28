from .base import IndustryLLMProfile
from .registry import get_industry_profile, normalize_industry_code
import regex as re
from typing import Any, Dict, Optional
# ---------------------------------------------------------------------------
# Industry normalization + label mapping
# ---------------------------------------------------------------------------

INDUSTRY_CODE_TO_LABEL: Dict[str, str] = {
    "1": "Primary",
    "2": "Food",
    "3": "Textile",
    "4": "Wood",
    "5": "Paper",
    "6": "Chemical",
    "7": "Pharmaceuticals",
    "8": "Rubber",
    "9": "Computer",
    "10": "Electrical",
    "11": "Machinery",
    "12": "Motor vehicles",
    "13": "Other transport",
    "14": "Utilities",
    "15": "Construction",
    "16": "Trade",
    "17": "Hotels",
    "18": "Transport",
    "19": "Post/Telecom",
    "20": "IT services",
    "21": "Financial services",
    "22": "Real estate",
    "23": "Professional services",
    "24": "Public admin",
}

UNCLASSIFIED_INDUSTRY_LABEL = "Unclassified"


def industry_label(code: Optional[str]) -> str:
    if not code:
        return UNCLASSIFIED_INDUSTRY_LABEL
    return INDUSTRY_CODE_TO_LABEL.get(code, UNCLASSIFIED_INDUSTRY_LABEL)


def normalize_industry_all(raw: Any) -> Optional[str]:
    """
    Accept:
      - "9|10|20"
      - ["9","10",20]
      - "9,10,20"
    Return canonical "9|10|20" sorted, unique where possible.
    """
    if raw is None:
        return None

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        # allow comma-separated
        parts = re.split(r"[|,;/\s]+", s)
        norm = [normalize_industry_code(p) for p in parts if p is not None]
        norm2 = sorted({x for x in norm if x})
        return "|".join(norm2) if norm2 else None

    if isinstance(raw, (list, tuple, set)):
        norm = [normalize_industry_code(x) for x in raw]
        norm2 = sorted({x for x in norm if x})
        return "|".join(norm2) if norm2 else None

    # fallback: treat as single code
    c = normalize_industry_code(raw)
    return c if c else None


__all__ = [
    "IndustryLLMProfile",
    "get_industry_profile",
    "normalize_industry_code",
    "industry_label",
    "normalize_industry_all",
]
