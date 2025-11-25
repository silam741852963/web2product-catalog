from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError

from crawl4ai import LLMConfig, LLMExtractionStrategy

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pydantic schemas (domain-level, language-agnostic)
# --------------------------------------------------------------------------- #

ShortStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Offering(BaseModel):
    """
    Minimal offering schema:

    - type: "product" | "service"
    - name: canonical name
    - description: concise but comprehensive summary
      (features / specs / what it does / who it serves)
    """

    type: ShortStr = Field(..., description="Either 'product' or 'service'.")
    name: ShortStr = Field(..., description="Canonical item name.")
    description: ShortStr = Field(..., description="Comprehensive but concise description.")


class ExtractionPayload(BaseModel):
    """
    Canonical schema for full product/service extraction.
    """

    offerings: List[Offering] = Field(
        default_factory=list,
        description="Zero or more product/service items.",
    )


class PresencePayload(BaseModel):
    """
    Presence-only payload: single field 'r' with value 0 or 1.

    r = 0 -> no sellable offering present
    r = 1 -> at least one sellable offering present
    """

    r: int = Field(
        ...,
        description="0 or 1 (0=no offering, 1=has offering)",
        ge=0,
        le=1,
    )


# --------------------------------------------------------------------------- #
# LLM provider strategies (Configuration-as-Strategy)
# --------------------------------------------------------------------------- #


class LLMProviderStrategy(Protocol):
    """
    Strategy interface for building an LLMConfig.

    Implementations encapsulate where/how the model is hosted:
      - Local Ollama
      - Remote API (OpenAI, etc.)
    """

    def build_config(self) -> LLMConfig:  # pragma: no cover - interface
        ...


@dataclass
class OllamaProviderStrategy(LLMProviderStrategy):
    """
    Local provider strategy using Ollama.

    Example provider string: "ollama/gemma3:12b-it-qat"
    """

    model: str = "gemma3:12b-it-qat"
    base_url: str = "http://localhost:11434"

    def build_config(self) -> LLMConfig:
        cfg = LLMConfig(
            provider=f"ollama/{self.model}",
            api_token=None,
            base_url=self.base_url,
        )
        logger.debug(
            "[llm] OllamaProviderStrategy.build_config provider=%s base_url=%s",
            cfg.provider,
            cfg.base_url,
        )
        return cfg


@dataclass
class RemoteAPIProviderStrategy(LLMProviderStrategy):
    """
    Remote API strategy, e.g. OpenAI, Anthropic, etc. via LiteLLM.

    provider example: "openai/gpt-4o-mini"
    """

    provider: str
    api_token: Optional[str] = None
    base_url: Optional[str] = None

    def build_config(self) -> LLMConfig:
        if not self.provider:
            raise ValueError("RemoteAPIProviderStrategy requires a non-empty provider string.")

        cfg = LLMConfig(
            provider=self.provider,
            api_token=self.api_token,
            base_url=self.base_url,
        )
        logger.debug(
            "[llm] RemoteAPIProviderStrategy.build_config provider=%s base_url=%s",
            cfg.provider,
            cfg.base_url,
        )
        return cfg


# --------------------------------------------------------------------------- #
# LLM default instructions (shared by orchestrator)
# --------------------------------------------------------------------------- #

DEFAULT_PRESENCE_INSTRUCTION = (
    "You are a classifier. Read the page content and answer ONLY in JSON. "
    "Return {\"r\":1} if the page clearly presents at least one product or service "
    "offering that the company sells or provides. Otherwise return {\"r\":0}. "
    "Do not include any other fields."
)

DEFAULT_FULL_INSTRUCTION = (
    "You are an expert information extractor. Read the page content and extract "
    "all concrete products and services that the company sells or provides. "
    "Return a JSON object with a single field 'offerings', which is a list of objects. "
    "Each object must have fields: 'type' ('product' or 'service'), 'name', and "
    "'description'. 'description' should concisely summarize what it is, what it does, "
    "and who it is for. If there are no offerings, return {\"offerings\": []}."
)


# --------------------------------------------------------------------------- #
# Extraction factory (Strategy + Factory + DI)
# --------------------------------------------------------------------------- #


@dataclass
class LLMExtractionFactory:
    """
    Factory for building LLMExtractionStrategy instances.
    """

    provider_strategy: LLMProviderStrategy
    default_full_instruction: str
    default_presence_instruction: str

    # Chunking / token defaults
    default_chunk_token_threshold: int = 1400
    default_overlap_rate: float = 0.08
    default_apply_chunking: bool = True
    default_input_format: str = "fit_markdown"

    # LLM behavior defaults
    default_schema_temperature: float = 0.2
    default_presence_temperature: float = 0.0
    default_schema_max_tokens: int = 1400
    default_presence_max_tokens: int = 900

    def create(
        self,
        *,
        mode: str = "schema",  # "schema" | "presence"
        schema: Optional[Dict[str, Any]] = None,
        instruction: Optional[str] = None,
        extraction_type: str = "schema",  # typically "schema" or "block"
        input_format: Optional[str] = None,
        chunk_token_threshold: Optional[int] = None,
        overlap_rate: Optional[float] = None,
        apply_chunking: Optional[bool] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> LLMExtractionStrategy:
        """
        Build an LLMExtractionStrategy.
        """

        normalized_mode = (mode or "schema").strip().lower()
        if normalized_mode not in {"schema", "presence"}:
            raise ValueError(f"Unsupported mode={mode!r}; expected 'schema' or 'presence'.")

        llm_cfg = self.provider_strategy.build_config()

        # Instruction selection
        if instruction is not None:
            used_instruction = instruction
        else:
            used_instruction = (
                self.default_presence_instruction
                if normalized_mode == "presence"
                else self.default_full_instruction
            )

        # Schema selection
        if normalized_mode == "presence":
            used_schema = PresencePayload.model_json_schema()
            used_extraction_type = "schema"
            temperature = self.default_presence_temperature
            max_tokens = self.default_presence_max_tokens
        else:
            used_schema = schema or ExtractionPayload.model_json_schema()
            used_extraction_type = extraction_type
            temperature = self.default_schema_temperature
            max_tokens = self.default_schema_max_tokens

        # Chunking & input format
        used_chunk_threshold = (
            self.default_chunk_token_threshold
            if chunk_token_threshold is None
            else int(chunk_token_threshold)
        )
        used_overlap_rate = (
            self.default_overlap_rate
            if overlap_rate is None
            else float(overlap_rate)
        )
        used_apply_chunking = (
            self.default_apply_chunking
            if apply_chunking is None
            else bool(apply_chunking)
        )
        used_input_format = input_format or self.default_input_format

        # Extra args (LLM-level tuning)
        _extra: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if extra_args:
            _extra.update(extra_args)

        # Logging preview (without secrets)
        try:
            cfg_preview = {
                "mode": normalized_mode,
                "provider": getattr(llm_cfg, "provider", None),
                "base_url": getattr(llm_cfg, "base_url", None),
                "chunk_token_threshold": used_chunk_threshold,
                "overlap_rate": used_overlap_rate,
                "apply_chunking": used_apply_chunking,
                "input_format": used_input_format,
                "extraction_type": used_extraction_type,
                "verbose": bool(verbose),
            }
            logger.info("[llm] building LLMExtractionStrategy: %s", cfg_preview)
        except Exception:
            logger.debug("[llm] LLMExtractionStrategy preview suppressed")

        strat = LLMExtractionStrategy(
            llm_config=llm_cfg,
            schema=used_schema,
            extraction_type=used_extraction_type,
            instruction=used_instruction,
            chunk_token_threshold=used_chunk_threshold,
            overlap_rate=used_overlap_rate,
            apply_chunking=used_apply_chunking,
            input_format=used_input_format,
            extra_args=_extra,
            verbose=verbose,
        )

        try:
            logger.debug(
                "[llm] instruction_snippet=%s",
                (used_instruction or "")[:320].replace("\n", " "),
            )
        except Exception:
            pass

        return strat

# --------------------------------------------------------------------------- #
# Robust parsing / normalization for schema mode
# --------------------------------------------------------------------------- #


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

        # Nested under "offerings"
        if "offerings" in it and isinstance(it["offerings"], list):
            for o in it["offerings"]:
                if _looks_like_offering(o):
                    offerings.append(
                        {
                            "type": o["type"],
                            "name": o["name"],
                            "description": o["description"],
                        }
                    )
            continue

        # Direct object with type/name/description
        if _looks_like_offering(it):
            offerings.append(
                {
                    "type": it["type"],
                    "name": it["name"],
                    "description": it["description"],
                }
            )
    return offerings


def parse_extracted_payload(
    extracted_content: Union[str, bytes, Dict[str, Any], List[Any], None]
) -> ExtractionPayload:
    """
    Normalize the raw `extracted_content` from LLMExtractionStrategy into
    an ExtractionPayload Pydantic object.

    Robust against:
      - string/bytes JSON
      - dicts with/without "offerings"
      - lists of offerings / wrapper objects
      - malformed/non-JSON -> returns empty payload
    """
    if extracted_content is None:
        logger.debug("[llm.parse] empty extracted_content -> empty payload")
        return ExtractionPayload(offerings=[])

    logger.debug(
        "[llm.parse] raw_extracted_preview=%s",
        _short_preview(extracted_content, length=800),
    )

    # 1) Load JSON if string/bytes
    if isinstance(extracted_content, (str, bytes, bytearray)):
        try:
            data = json.loads(extracted_content)
            logger.debug("[llm.parse] json.loads -> %s", type(data).__name__)
        except Exception as e:
            logger.debug(
                "[llm.parse] json.loads failed: %s | preview=%s",
                e,
                _short_preview(extracted_content),
            )
            return ExtractionPayload(offerings=[])
    else:
        data = extracted_content

    # 2) Normalize shapes into a dict compatible with ExtractionPayload
    payload_dict: Dict[str, Any]

    try:
        if isinstance(data, dict):
            if "offerings" in data:
                offs = []
                for o in data.get("offerings") or []:
                    if _looks_like_offering(o):
                        offs.append(
                            {
                                "type": o["type"],
                                "name": o["name"],
                                "description": o["description"],
                            }
                        )
                payload_dict = {"offerings": offs}

            elif _looks_like_offering(data):
                payload_dict = {
                    "offerings": [
                        {
                            "type": data["type"],
                            "name": data["name"],
                            "description": data["description"],
                        }
                    ]
                }
            else:
                payload_dict = {"offerings": []}

        elif isinstance(data, list):
            offs = _extract_offerings_from_mixed_list(data)
            payload_dict = {"offerings": offs}

        else:
            payload_dict = {"offerings": []}
    except Exception as e:
        logger.exception("[llm.parse] normalization error: %s", e)
        return ExtractionPayload(offerings=[])

    # 3) Validate with Pydantic
    try:
        payload = ExtractionPayload.model_validate(payload_dict)
        logger.debug(
            "[llm.parse] normalized payload offerings=%d",
            len(payload.offerings),
        )
        return payload
    except ValidationError as e:
        logger.debug("[llm.parse] pydantic validation failed: %s", e)
        return ExtractionPayload(offerings=[])


# --------------------------------------------------------------------------- #
# Presence result parsing
# --------------------------------------------------------------------------- #


def parse_presence_result(
    extracted_content: Union[str, bytes, int, Dict[str, Any], List[Any], None],
    *,
    default: bool = False,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Parse presence-only result and return:
        (has_offering_bool, confidence_or_none, raw_preview_or_none)

    Canonical target shape:
        {"r": 0}  or  {"r": 1}

    Also accepts:
      - integer 0/1
      - string "0"/"1"
      - dict with {"r": ...} or legacy keys: classification/result/presence/etc.
      - list-wrapped variants

    Anything ambiguous falls back to `default`.
    """
    try:
        preview = _short_preview(extracted_content, length=800)
        logger.debug("[llm.presence] raw preview=%s", preview)

        if extracted_content is None:
            logger.debug("[llm.presence] empty content -> default=%s", default)
            return (default, None, preview)

        # Normalize bytes -> string -> try JSON
        if isinstance(extracted_content, (bytes, bytearray)):
            s = extracted_content.decode(errors="ignore").strip()
            try:
                loaded = json.loads(s)
                logger.debug("[llm.presence] json.loads -> %s", type(loaded).__name__)
            except Exception:
                loaded = s
                logger.debug("[llm.presence] treated as raw string after decode")
        elif isinstance(extracted_content, str):
            s = extracted_content.strip()
            try:
                loaded = json.loads(s)
                logger.debug("[llm.presence] json.loads -> %s", type(loaded).__name__)
            except Exception:
                loaded = s
                logger.debug("[llm.presence] treated as raw string")
        else:
            loaded = extracted_content

        def _interpret_scalar(val: Any) -> Optional[int]:
            if isinstance(val, int) and val in (0, 1):
                return int(val)
            if isinstance(val, str) and val.strip() in ("0", "1"):
                return int(val.strip())
            return None

        # 1) Direct scalar
        isc = _interpret_scalar(loaded)
        if isc is not None:
            logger.debug("[llm.presence] parsed scalar presence=%d", isc)
            return (bool(isc), None, preview)

        # 2) Dict
        if isinstance(loaded, dict):
            if "r" in loaded:
                r_val = _interpret_scalar(loaded["r"])
                if r_val is not None:
                    logger.debug("[llm.presence] parsed 'r' field presence=%d", r_val)
                    return (bool(r_val), None, preview)

            for key in (
                "has_offering",
                "presence",
                "present",
                "classification",
                "result",
                "type",
            ):
                if key in loaded:
                    r_val = _interpret_scalar(loaded[key])
                    if r_val is not None:
                        logger.debug(
                            "[llm.presence] parsed '%s' presence=%d",
                            key,
                            r_val,
                        )
                        return (bool(r_val), None, preview)

            logger.debug(
                "[llm.presence] dict without recognized presence key -> default=%s",
                default,
            )
            return (default, None, preview)

        # 3) List
        if isinstance(loaded, list) and loaded:
            first = loaded[0]
            isc = _interpret_scalar(first)
            if isc is not None:
                logger.debug(
                    "[llm.presence] parsed list-wrapped scalar presence=%d",
                    isc,
                )
                return (bool(isc), None, preview)

            if isinstance(first, dict):
                if "r" in first:
                    r_val = _interpret_scalar(first["r"])
                    if r_val is not None:
                        logger.debug(
                            "[llm.presence] parsed list->dict 'r' presence=%d",
                            r_val,
                        )
                        return (bool(r_val), None, preview)

                for key in (
                    "has_offering",
                    "presence",
                    "present",
                    "classification",
                    "result",
                    "type",
                ):
                    if key in first:
                        r_val = _interpret_scalar(first[key])
                        if r_val is not None:
                            logger.debug(
                                "[llm.presence] parsed list->dict '%s' presence=%d",
                                key,
                                r_val,
                            )
                            return (bool(r_val), None, preview)

            logger.debug(
                "[llm.presence] list returned but no interpretable presence; "
                "returning default=%s",
                default,
            )
            return (default, None, preview)

        # Fallback
        logger.debug(
            "[llm.presence] unable to interpret presence result; returning default=%s",
            default,
        )
        return (default, None, preview)
    except Exception as e:
        logger.exception("[llm.presence] parse error: %s", e)
        return (default, None, None)


# --------------------------------------------------------------------------- #
# Utility: short preview
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Default, injectable provider instance
# --------------------------------------------------------------------------- #

#: Default local Ollama provider strategy. Override model/base_url as needed.
default_ollama_provider_strategy = OllamaProviderStrategy()

__all__ = [
    "Offering",
    "ExtractionPayload",
    "PresencePayload",
    "LLMProviderStrategy",
    "OllamaProviderStrategy",
    "RemoteAPIProviderStrategy",
    "LLMExtractionFactory",
    "parse_extracted_payload",
    "parse_presence_result",
    "default_ollama_provider_strategy",
    "DEFAULT_PRESENCE_INSTRUCTION",
    "DEFAULT_FULL_INSTRUCTION",
]