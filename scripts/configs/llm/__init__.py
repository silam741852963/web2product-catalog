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

import litellm

# Explicitly register local vLLM model metadata
litellm.model_cost["hosted_vllm/google/gemma-3-270m-it"] = {
    # pricing irrelevant for local models
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    # Gemma 3 270M IT context window
    # adjust if you started vLLM with a different limit
    "max_input_tokens": 8192,
    "max_output_tokens": 8192,
    # Optional but helps debug clarity
    "mode": "chat",
}
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
