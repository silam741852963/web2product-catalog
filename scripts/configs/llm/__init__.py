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

_LOCAL_VLLM_MODELS = {
    "hosted_vllm/google/gemma-3-270m-it": 8192,
    "hosted_vllm/google/gemma-3-1b-it": 8192,
}

for name, ctx in _LOCAL_VLLM_MODELS.items():
    litellm.model_cost[name] = {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "max_input_tokens": ctx,
        "max_output_tokens": ctx,
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
