from __future__ import annotations

from typing import Any, List, Annotated

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
)

# --------------------------------------------------------------------------- #
# Constants / helpers (schema-related)
# --------------------------------------------------------------------------- #

NAME_MAX = 160
DESC_MAX = 520
TAG_MAX = 50

NameStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=NAME_MAX)
]
DescStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=DESC_MAX)
]
TagStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=TAG_MAX)
]


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="ignore")
    return str(v)


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split())


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (str, bytes, bytearray)):
        s = _clean_ws(_to_text(v))
        if not s:
            return []
        if "|" in s:
            return [x.strip() for x in s.split("|") if x.strip()]
        if ";" in s:
            return [x.strip() for x in s.split(";") if x.strip()]
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return [v]


def _dedup_norm_list(items: List[str], *, max_items: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        s = _clean_ws(_to_text(it))
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


# --------------------------------------------------------------------------- #
# Pydantic schemas (IMPORTANT: preserved descriptions)
# --------------------------------------------------------------------------- #


class Offering(BaseModel):
    type: str = Field(
        ...,
        description=(
            "Offering class. Must be 'product' or 'service'. "
            "Choose 'product' for goods, devices, materials, components, software products, product lines, brands. "
            "Choose 'service' for services, solutions, platforms provided as a service, consulting, manufacturing-as-a-service, "
            "installation, maintenance, training, logistics, R&D/testing, integration, managed services."
        ),
    )

    name: NameStr = Field(
        ...,
        description=(
            "Short on-page name/label for the offering (use headings/labels if available). "
            "Examples: brand name, product family, model series, platform name, solution name, service line."
        ),
    )

    description: DescStr = Field(
        ...,
        description=(
            "1–3 grounded sentences explaining what it is and what value it provides. "
            "Keep factual; avoid generic marketing claims; do not invent details not on the page."
        ),
    )

    tags: List[TagStr] = Field(
        default_factory=list,
        description=(
            "0–10 short tags to improve retrieval/grouping. Prefer domain nouns and capability keywords. "
            "Do not include navigation terms like 'home', 'about', 'contact'."
        ),
    )

    @field_validator("type", mode="before")
    @classmethod
    def _norm_type(cls, v: Any) -> str:
        s = _clean_ws(_to_text(v)).lower()
        if s in {
            "product",
            "products",
            "prod",
            "goods",
            "good",
            "brand",
            "line",
            "hardware",
        }:
            return "product"
        if s in {
            "service",
            "services",
            "svc",
            "solution",
            "solutions",
            "capability",
            "capabilities",
        }:
            return "service"
        if "product" in s or "brand" in s or "goods" in s:
            return "product"
        if "service" in s or "solution" in s or "capabilit" in s:
            return "service"
        return s or "product"

    @field_validator("name", mode="before")
    @classmethod
    def _norm_name(cls, v: Any) -> str:
        return _clean_ws(_to_text(v))[:NAME_MAX]

    @field_validator("description", mode="before")
    @classmethod
    def _norm_desc(cls, v: Any) -> str:
        return _clean_ws(_to_text(v))[:DESC_MAX]

    @field_validator("tags", mode="before")
    @classmethod
    def _norm_tags(cls, v: Any) -> List[str]:
        items = _as_list(v)
        normed = _dedup_norm_list(
            [(_clean_ws(_to_text(x))[:TAG_MAX]) for x in items],
            max_items=10,
        )
        junk = {
            "home",
            "about",
            "contact",
            "privacy",
            "terms",
            "cookie",
            "careers",
            "news",
            "blog",
            "sitemap",
        }
        return [t for t in normed if t.lower() not in junk]


class ExtractionPayload(BaseModel):
    offerings: List[Offering] = Field(default_factory=list)


class PresencePayload(BaseModel):
    r: int = Field(..., ge=0, le=1)
