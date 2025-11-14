from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any, Union, Tuple, Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError

from crawl4ai import LLMExtractionStrategy
from crawl4ai import LLMConfig

from configs import language_settings as lang_cfg

logger = logging.getLogger(__name__)

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


# ---------- Presence Pydantic schema ----------
class PresencePayload(BaseModel):
    """
    Presence-only payload: single field 'r' with value 0 or 1.
    r = 0 -> no sellable offering present
    r = 1 -> sellable offering present
    """
    r: int = Field(..., description="0 or 1 (0=no offering, 1=has offering)", ge=0, le=1)


# ---------- Instructions (pull from language settings if available) ----------
_instructions = lang_cfg.get("INSTRUCTIONS", {}) or {}
_DEFAULT_INSTRUCTION = _instructions.get("full")

_PRESENCE_INSTRUCTION = _instructions.get("presence")


# ---------- LLM config helpers ----------

def _default_ollama_llm_config(
    model: str = "gemma3:12b-it-qat",
    base_url: str = "http://localhost:11434",
) -> LLMConfig:
    cfg = LLMConfig(
        provider=f"ollama/{model}",
        api_token=None,
        base_url=base_url,
    )
    logger.debug("[llm_extractor] default ollama config created: provider=%s base_url=%s", cfg.provider, cfg.base_url)
    return cfg


def build_llm_extraction_strategy(
    *,
    use_local_ollama: bool = True,
    ollama_model: str = "gemma3:12b-it-qat",
    ollama_base_url: str | None = None,
    provider: Optional[str] = None,
    api_token: Optional[str] = None,
    base_url: Optional[str] = None,
    presence_only: Optional[bool] = None,
    mode: str = "schema",  # "schema" | "presence"
    instruction: Optional[str] = None,  # if None, use language_settings
    input_format: str = "fit_markdown",
    extraction_type: str = "schema",
    chunk_token_threshold: int = 1400,
    overlap_rate: float = 0.08,
    apply_chunking: bool = True,
    extra_args: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> LLMExtractionStrategy:
    # Back-compat mapping
    if presence_only is not None:
        mode = "presence" if presence_only else "schema"

    # choose instruction from arg or language_settings
    if instruction is None:
        instruction = _PRESENCE_INSTRUCTION if mode == "presence" else _DEFAULT_INSTRUCTION

    # LLM config
    if use_local_ollama and not provider:
        ollama_base = ollama_base_url or "http://localhost:11434"
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

    # default extra args
    _extra = {
        # deterministic-ish
        "temperature": 0.0 if mode == "presence" else 0.2,
        # tokens
        "max_tokens": 900 if mode == "presence" else 1400,
        # prefer a simple JSON object response when in schema mode; presence also uses json_object
        "response_format": {"type": "json_object"},
    }
    if extra_args:
        _extra.update(extra_args)

    # Schema & instruction by mode
    if mode == "presence":
        # Use pydantic schema for presence (single-field object {"r": 0|1})
        schema_dict = PresencePayload.model_json_schema()
        used_instruction = _PRESENCE_INSTRUCTION
        used_extraction_type = "schema"
    else:
        schema_dict = ExtractionPayload.model_json_schema()
        used_instruction = instruction or _DEFAULT_INSTRUCTION
        used_extraction_type = extraction_type

    # Log a preview of the built strategy (non-sensitive)
    try:
        cfg_preview = {
            "mode": mode,
            "provider": getattr(llm_cfg, "provider", None),
            "base_url": getattr(llm_cfg, "base_url", None),
            "chunk_token_threshold": int(chunk_token_threshold),
            "overlap_rate": float(overlap_rate),
            "apply_chunking": bool(apply_chunking),
            "input_format": input_format,
            "extraction_type": used_extraction_type,
            "extra_keys": {k: v for k, v in (_extra or {}).items() if k != "api_token"},
            "verbose": bool(verbose),
        }
        logger.info("[llm_extractor] build strategy: %s", cfg_preview)
    except Exception:
        logger.debug("[llm_extractor] built llm strategy (preview suppressed)")

    strat = LLMExtractionStrategy(
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

    try:
        logger.debug("[llm_extractor] instruction_snippet=%s", used_instruction[:320].replace("\n", " "))
    except Exception:
        pass

    return strat


# ---------- Robust parsing / normalization ----------

def _looks_like_offering(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    t = (d.get("type") or "").strip().lower()
    n = (d.get("name") or "").strip()
    desc = (d.get("description") or "").strip()
    return bool(n) and bool(desc) and t in {"product", "service"}


def _extract_offerings_from_mixed_list(items: List[Any]) -> List[Dict[str, Any]]:
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
    if extracted_content is None:
        logger.debug("[llm_extractor.parse] empty extracted_content -> empty payload")
        return ExtractionPayload(offerings=[])

    logger.debug("[llm_extractor.parse] raw_extracted_preview=%s", _short_preview(extracted_content, length=800))

    # 1) Load to Python
    if isinstance(extracted_content, (str, bytes)):
        try:
            data = json.loads(extracted_content)
            logger.debug("[llm_extractor.parse] json.loads -> %s", type(data).__name__)
        except Exception as e:
            logger.debug("[llm_extractor.parse] json.loads failed: %s | preview=%s", e, _short_preview(extracted_content))
            return ExtractionPayload(offerings=[])
    else:
        data = extracted_content

    # 2) Normalize shapes
    payload_dict: Dict[str, Any] | None = None

    try:
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
    except Exception as e:
        logger.exception("[llm_extractor.parse] normalization error: %s", e)
        return ExtractionPayload(offerings=[])

    try:
        payload = ExtractionPayload.model_validate(payload_dict)
        logger.debug("[llm_extractor.parse] normalized payload offerings=%d", len(payload.offerings))
        return payload
    except ValidationError as e:
        logger.debug("[llm_extractor.parse] pydantic validation failed: %s", e)
        return ExtractionPayload(offerings=[])

# ---------- Presence result parsing ----------
def parse_presence_result(
    extracted_content: Union[str, bytes, int, Dict[str, Any], List[Any], None],
    *,
    default: bool = False
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Parse presence result (presence-only runs). Return (has_bool, confidence_or_none, raw_preview_or_none).

    Expected canonical shape (new): {"r": 0} or {"r": 1}

    Accepts legacy/various shapes for robustness:
      - integer 0 or 1
      - string "0" or "1"
      - dict with "r": 0|1
      - dict with "classification"/"result"/"presence"/"has_offering" : "0"/"1" or 0/1
      - list-wrapped variants: [{...}] or ["0"] or [0]
    Anything ambiguous falls back to `default`.
    """
    try:
        # produce a short preview and log it so operator can inspect outputs without debug files
        preview = _short_preview(extracted_content, length=800)
        logger.debug("[llm_extractor.presence] raw preview=%s", preview)

        if extracted_content is None:
            logger.debug("[llm_extractor.presence] empty content -> default=%s", default)
            return (default, None, preview)

        # Normalize bytes -> string -> attempt JSON load
        if isinstance(extracted_content, (bytes, bytearray)):
            s = extracted_content.decode(errors="ignore").strip()
            try:
                loaded = json.loads(s)
                logger.debug("[llm_extractor.presence] json.loads -> %s", type(loaded).__name__)
            except Exception:
                loaded = s
                logger.debug("[llm_extractor.presence] treated as raw string after decode")
        elif isinstance(extracted_content, str):
            s = extracted_content.strip()
            try:
                loaded = json.loads(s)
                logger.debug("[llm_extractor.presence] json.loads -> %s", type(loaded).__name__)
            except Exception:
                loaded = s
                logger.debug("[llm_extractor.presence] treated as raw string")
        else:
            loaded = extracted_content

        def _interpret_scalar(val: Any) -> Optional[int]:
            if isinstance(val, int) and val in (0, 1):
                return int(val)
            if isinstance(val, str) and val.strip() in ("0", "1"):
                return int(val.strip())
            return None

        # 1) Direct int / "0"/"1"
        isc = _interpret_scalar(loaded)
        if isc is not None:
            logger.debug("[llm_extractor.presence] parsed scalar presence=%d", isc)
            return (bool(isc), None, preview)

        # 2) Dict handling (canonical new schema: {"r": 0|1})
        if isinstance(loaded, dict):
            # direct r field
            if "r" in loaded:
                r_val = _interpret_scalar(loaded["r"])
                if r_val is not None:
                    logger.debug("[llm_extractor.presence] parsed 'r' field presence=%d", r_val)
                    return (bool(r_val), None, preview)
            # common legacy keys
            for key in ("has_offering", "presence", "present", "classification", "result", "type"):
                if key in loaded:
                    r_val = _interpret_scalar(loaded[key])
                    if r_val is not None:
                        logger.debug("[llm_extractor.presence] parsed '%s' presence=%s", key, r_val)
                        return (bool(r_val), None, preview)
            # No recognized numeric presence key
            logger.debug("[llm_extractor.presence] dict without recognized presence key -> default=%s", default)
            return (default, None, preview)

        # 3) List-wrapped values
        if isinstance(loaded, list) and len(loaded) > 0:
            # try simple scalar list first
            first = loaded[0]
            isc = _interpret_scalar(first)
            if isc is not None:
                logger.debug("[llm_extractor.presence] parsed list-wrapped scalar presence=%d", isc)
                return (bool(isc), None, preview)
            # if first is dict, try keys inside it (support list-of-dicts like [{ "classification": "0" }, ...])
            if isinstance(first, dict):
                if "r" in first:
                    r_val = _interpret_scalar(first["r"])
                    if r_val is not None:
                        logger.debug("[llm_extractor.presence] parsed list->dict 'r' presence=%d", r_val)
                        return (bool(r_val), None, preview)
                for key in ("has_offering", "presence", "present", "classification", "result", "type"):
                    if key in first:
                        r_val = _interpret_scalar(first[key])
                        if r_val is not None:
                            logger.debug("[llm_extractor.presence] parsed list->dict '%s' presence=%s", key, r_val)
                            return (bool(r_val), None, preview)
            # fallthrough: not interpretable
            logger.debug("[llm_extractor.presence] list returned but no interpretable presence; returning default=%s", default)
            return (default, None, preview)

        # fallback: unable to interpret
        logger.debug("[llm_extractor.presence] unable to interpret presence result; returning default=%s preview=%s", default, preview)
        return (default, None, preview)
    except Exception as e:
        logger.exception("[llm_extractor.presence] parse error: %s", e)
        return (default, None, None)


# ---------- Utilities: preview ----------
def _short_preview(x: Any, length: int = 400) -> str:
    try:
        if x is None:
            return "<none>"
        if isinstance(x, (bytes, bytearray)):
            s = x.decode(errors="ignore")
        else:
            s = str(x)
        s = s.replace("\n", " ")
        if len(s) <= length:
            return s
        return s[:length] + "…"
    except Exception:
        return "<preview-failed>"