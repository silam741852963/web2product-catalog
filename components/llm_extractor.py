# components/llm_extractor.py
from __future__ import annotations

import json
import os
from typing import List, Optional, Dict, Any, Union, Annotated, Tuple

from pydantic import BaseModel, Field, StringConstraints, ValidationError

# Crawl4AI imports
from crawl4ai import LLMExtractionStrategy  # for typing/usage hints
from crawl4ai import LLMConfig  # provider-agnostic config (LiteLLM under the hood)

# ---------- String constraints ----------
ShortStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# ---------- Minimal schema ----------
class Offering(BaseModel):
    """
    Minimal offering schema:
      - type: "product" | "service"
      - name: canonical name
      - description: comprehensive, concise description (features/coverage/specs/what it does/who it serves)
    """
    type: ShortStr = Field(..., description="Either 'product' or 'service'.")
    name: ShortStr = Field(..., description="Canonical item name.")
    description: ShortStr = Field(..., description="Comprehensive but concise description.")


class ExtractionPayload(BaseModel):
    offerings: List[Offering] = Field(default_factory=list, description="Zero or more product/service items.")


# ---- Presence (yes/no) classification schema ----
class PresenceResult(BaseModel):
    has_offering: bool = Field(..., description="True if page clearly markets sellable products/services.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional rough confidence [0..1].")
    rationale: Optional[str] = Field(
        default=None,
        description="Short reason: which cues indicate offerings (e.g., 'pricing', 'contact sales', 'get quote')."
    )


# ---------- Instructions (updated for minimal schema) ----------
_DEFAULT_INSTRUCTION = """
Extract COMMERCIAL OFFERINGS the company SELLS (products or paid services).
Return data ONLY if there is clear selling/contracting intent. Otherwise return empty.

INCLUDE
- Things customers can buy/subscribe/contract: software, platforms, hardware, consumables, insurance/coverage, consulting/managed services, commodities.
- Signals: “offer / sell / provide / underwrite / distribute / export / implement / subscribe / plans / pricing / get a quote / request demo / contact sales”.

EXCLUDE
- Editorial/corporate: news, press, blogs, articles, insights, whitepapers, case studies, awards, CSR/DEI/community, partnership thank-yous, leadership profiles.
- Careers, investor relations, cookie/privacy/terms/legal, navigation-only.

TYPE DECISION
- "product": a thing (including commodities) with features/grades/specs/plans/coverage.
- "service": paid professional/managed services delivered under contract.

OUTPUT — EXACTLY ONE JSON OBJECT (no extra keys, no arrays at the top level):
{
  "offerings": [
    {
      "type": "product" | "service",
      "name": "string",
      "description": "string (comprehensive but concise: what it is, key features/specs/coverage, intended buyers)"
    }
  ]
}
If nothing is relevant, return exactly:
{ "offerings": [] }
""".strip()

_PRESENCE_INSTRUCTION = """
You are a strict binary classifier.
Question: Does this page (markdown input) clearly market SELLABLE products or paid services?

Positive cues (any of these strongly indicates YES):
- Features/specifications of a product or coverage details of a plan
- Pricing, plans, SKUs, 'get a quote', 'contact sales', 'request demo', 'buy', 'add to cart', 'subscribe'
- 'Services we offer', 'managed services', 'consulting services', 'solutions' with commercial CTAs

Negative cues (indicates NO):
- News/press/blog/media/articles/insights/case studies
- Careers/jobs
- Investor relations/SEC filings
- Cookie/privacy/terms/legal notices
- Navigation-only or interstitial pages

Return a single JSON object:
{
  "has_offering": true|false,
  "confidence": 0.0..1.0 (optional),
  "rationale": "short reason" (optional)
}
Be conservative: if uncertain, answer false.
""".strip()


# ---------- LLM config helpers ----------
def _default_ollama_llm_config(
    model: str = "gemma3:12b-it-qat",
    base_url: str = "http://localhost:11434",
) -> LLMConfig:
    return LLMConfig(
        provider=f"ollama/{model}",
        api_token=None,
        base_url=base_url,
    )


def build_llm_extraction_strategy(
    *,
    use_local_ollama: bool = True,
    ollama_model: str = "gemma3:12b-it-qat",
    ollama_base_url: str | None = None,
    provider: Optional[str] = None,
    api_token: Optional[str] = None,
    base_url: Optional[str] = None,
    # legacy/back-compat flag (maps to `mode`)
    presence_only: Optional[bool] = None,
    # modern switch:
    mode: str = "schema",  # "schema" | "presence"
    # schema mode params:
    instruction: str = _DEFAULT_INSTRUCTION,
    input_format: str = "fit_markdown",
    extraction_type: str = "schema",
    # both modes:
    chunk_token_threshold: int = 1400,
    overlap_rate: float = 0.08,
    apply_chunking: bool = True,
    extra_args: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> LLMExtractionStrategy:
    """
    Build an LLMExtractionStrategy for either:
      - mode="schema": structured offerings extraction (default), or
      - mode="presence": boolean presence classification (has_offering yes/no).

    Back-compat:
      - If presence_only is not None, it overrides `mode`:
          presence_only=True  -> mode="presence"
          presence_only=False -> mode="schema"
    """
    # ---- Back-compat mapping ----
    if presence_only is not None:
        mode = "presence" if presence_only else "schema"

    # ---- LLM config ----
    if use_local_ollama and not provider:
        ollama_base = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        llm_cfg = _default_ollama_llm_config(model=ollama_model, base_url=ollama_base)
    else:
        if not provider:
            raise ValueError(
                "Remote LLM provider requested but 'provider' not set "
                "(e.g., provider='openai/gpt-4o-mini')."
            )
        llm_cfg = LLMConfig(
            provider=provider,
            api_token=api_token,
            base_url=base_url,
        )

    _extra = {
        "temperature": 0.0,
        "max_tokens": 900 if mode == "presence" else 1400,
        "response_format": {"type": "json_object"},
    }
    if extra_args:
        _extra.update(extra_args)

    # ---- Schema & instruction by mode ----
    if mode == "presence":
        schema_dict = PresenceResult.model_json_schema()
        used_instruction = _PRESENCE_INSTRUCTION
        used_extraction_type = "schema"  # still schema-constrained output
    else:
        schema_dict = ExtractionPayload.model_json_schema()
        used_instruction = instruction
        used_extraction_type = extraction_type

    return LLMExtractionStrategy(
        llm_config=llm_cfg,
        schema=schema_dict,
        extraction_type=used_extraction_type,
        instruction=used_instruction,
        chunk_token_threshold=chunk_token_threshold,
        overlap_rate=overlap_rate,
        apply_chunking=apply_chunking,
        input_format=input_format,
        extra_args=_extra,
        verbose=verbose,
    )


# ---------- Robust parsing / normalization ----------
def _looks_like_offering(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    t = (d.get("type") or "").strip().lower()
    n = (d.get("name") or "").strip()
    desc = (d.get("description") or "").strip()
    return bool(n) and bool(desc) and t in {"product", "service"}


def _extract_offerings_from_mixed_list(items: List[Any]) -> List[Dict[str, Any]]:
    """
    Accept a mixed list of dicts (possibly containing {company, offerings}, errors, or direct offering dicts).
    Returns a flat list of minimal offering dicts.
    """
    offerings: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("error") is True:
            continue
        if "offerings" in it and isinstance(it["offerings"], list):
            for o in it["offerings"]:
                if _looks_like_offering(o):
                    offerings.append({"type": o["type"], "name": o["name"], "description": o["description"]})
            continue
        if _looks_like_offering(it):
            offerings.append({"type": it["type"], "name": it["name"], "description": it["description"]})
    return offerings


def parse_extracted_payload(
    extracted_content: Union[str, bytes, Dict[str, Any], List[Any], None]
) -> ExtractionPayload:
    """
    Be liberal in what we accept, strict in what we emit.
    - Accept dict, list, stringified JSON, bytes.
    - Normalize to ExtractionPayload(company={}, offerings=[...]) with minimal schema.
    """
    if extracted_content is None:
        return ExtractionPayload(offerings=[])

    # 1) Load to Python
    if isinstance(extracted_content, (str, bytes)):
        try:
            data = json.loads(extracted_content)
        except Exception:
            return ExtractionPayload(offerings=[])
    else:
        data = extracted_content

    # 2) Normalize shapes
    payload_dict: Dict[str, Any] | None = None

    if isinstance(data, dict):
        if "offerings" in data:
            offs = []
            for o in data.get("offerings") or []:
                if _looks_like_offering(o):
                    offs.append({"type": o["type"], "name": o["name"], "description": o["description"]})
            payload_dict = {
                "company": data.get("company") or ({}),
                "offerings": offs,
            }
        elif _looks_like_offering(data):
            payload_dict = {
                "company": ({}),
                "offerings": [{"type": data["type"], "name": data["name"], "description": data["description"]}],
            }
        else:
            payload_dict = {
                "company": data.get("company") or ({}),
                "offerings": [],
            }

    elif isinstance(data, list):
        offs = _extract_offerings_from_mixed_list(data)
        payload_dict = {
            "company": ({}),
            "offerings": offs,
        }
    else:
        payload_dict = {
            "company": ({}),
            "offerings": [],
        }

    try:
        return ExtractionPayload.model_validate(payload_dict)
    except ValidationError:
        return ExtractionPayload(offerings=[])


# ---------- Optional post-filter to drop non-offerings ----------
_NEGATIVE_URL_TOKENS = {
    "/news/", "/press/", "/media/", "/blog/", "/insights/", "/stories/", "/story/",
    "/article/", "/events/", "/event/", "/careers/", "/jobs/", "/privacy/",
    "/terms/", "/legal/", "/cookie/", "/cookies/"
}

_NEGATIVE_NAME_HINTS = {
    "article", "articles", "news", "insight", "insights", "blog", "press",
    "media", "story", "stories", "webinar", "event", "career", "jobs",
    "anniversary", "cookie", "privacy", "terms", "legal"
}


def _has_negative_url_token(u: Optional[str]) -> bool:
    if not u:
        return False
    u = u.lower()
    return any(tok in u for tok in _NEGATIVE_URL_TOKENS)


def _looks_like_non_offering_text(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.lower()
    return any(h in s for h in _NEGATIVE_NAME_HINTS)


def strict_offering_postfilter(payload: ExtractionPayload,
                               source_url: Optional[str] = None) -> ExtractionPayload:
    """
    Drop items that still look editorial after model extraction.
    """
    if _has_negative_url_token(source_url):
        return ExtractionPayload(company=payload.company, offerings=[])
    kept: List[Offering] = []
    for off in payload.offerings:
        text_block = f"{off.name} {off.description}".lower()
        if _looks_like_non_offering_text(text_block):
            continue
        kept.append(off)
    return ExtractionPayload(company=payload.company, offerings=kept)


# ---------- Presence result parsing ----------
def parse_presence_result(
    extracted_content: Union[str, bytes, Dict[str, Any], None],
    *,
    default: bool = False
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Parse presence-mode output into (has_offering, confidence, rationale).
    If parsing fails, returns (default, None, None).
    """
    if extracted_content is None:
        return (default, None, None)
    try:
        data = json.loads(extracted_content) if isinstance(extracted_content, (str, bytes)) else extracted_content
        res = PresenceResult.model_validate(data)
        return (bool(res.has_offering), res.confidence, res.rationale)
    except Exception:
        return (default, None, None)