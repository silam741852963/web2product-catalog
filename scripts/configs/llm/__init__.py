from __future__ import annotations

from .schema import Offering, ExtractionPayload, PresencePayload
from .prompting import (
    IndustryContext,
    BASE_FULL_INSTRUCTION,
    BASE_PRESENCE_INSTRUCTION,
    build_full_instruction,
    build_presence_instruction,
)
from .factory import (
    LLMProviderStrategy,
    OllamaProviderStrategy,
    RemoteAPIProviderStrategy,
    provider_strategy_from_llm_model_selector,
    LLMExtractionFactory,
    IndustryAwareStrategyCache,
)
from .parsing import (
    parse_extracted_payload,
    parse_presence_result,
    call_llm_extract,
)

__all__ = [
    "Offering",
    "ExtractionPayload",
    "PresencePayload",
    "IndustryContext",
    "BASE_FULL_INSTRUCTION",
    "BASE_PRESENCE_INSTRUCTION",
    "build_full_instruction",
    "build_presence_instruction",
    "LLMProviderStrategy",
    "OllamaProviderStrategy",
    "RemoteAPIProviderStrategy",
    "provider_strategy_from_llm_model_selector",
    "LLMExtractionFactory",
    "IndustryAwareStrategyCache",
    "parse_extracted_payload",
    "parse_presence_result",
    "call_llm_extract",
]
