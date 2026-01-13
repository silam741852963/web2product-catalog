from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

from crawl4ai import LLMConfig, LLMExtractionStrategy

from configs.llm.prompting import (
    IndustryContext,
    BASE_FULL_INSTRUCTION,
    BASE_PRESENCE_INSTRUCTION,
    build_full_instruction,
    build_presence_instruction,
)
from configs.llm.schema import ExtractionPayload, PresencePayload

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="ignore")
    return str(v)


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split())


def _provider_prefix(provider: str) -> str:
    p = (provider or "").strip()
    if "/" in p:
        return p.split("/", 1)[0].strip().lower()
    return p.lower()


def _stable_json_sha1(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _ensure_non_empty(s: str, label: str) -> str:
    v = (s or "").strip()
    if not v:
        raise ValueError(f"{label} must be non-empty.")
    return v


def _validate_vllm_base_url(base_url: str) -> str:
    """
    vLLM OpenAI-compatible servers usually expose /v1.
    Accept both:
      - http://127.0.0.1:8000
      - http://127.0.0.1:8000/v1
    and normalize to end with /v1.
    """
    u = (base_url or "").strip()
    if not u:
        raise ValueError("VLLM base_url must be non-empty.")
    if u.endswith("/v1"):
        return u
    return u.rstrip("/") + "/v1"


def _strip_known_prefix(s: str, prefix: str) -> str:
    if not s.startswith(prefix):
        raise ValueError(f"Expected prefix {prefix!r}, got {s!r}")
    return s[len(prefix) :].lstrip()


def _infer_api_token_for_provider(provider: str) -> Optional[str]:
    """
    Only infer for providers we explicitly support inference for.
    Never infer OpenAI here.
    """
    pref = _provider_prefix(provider)
    if pref in {"gemini", "google"}:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return None


def _is_gemini_like_provider(provider: Optional[str]) -> bool:
    return _provider_prefix(provider or "") in {"gemini", "google"}


def _is_plain_model_id(s: str) -> bool:
    """
    Detect "bare model id" (HF-style repo id) like 'google/gemma-3-270m-it'
    that is NOT intended as a LiteLLM provider route.

    We treat these as LOCAL vLLM model_ids.
    """
    if not s or "/" not in s:
        return False

    pref = _provider_prefix(s)

    known_provider_prefixes = {
        "vllm",
        "hosted_vllm",
        "ollama",
        "openai",
        "azure",
        "anthropic",
        "bedrock",
        "vertex_ai",
        "mistral",
        "cohere",
        "groq",
        "together_ai",
        "huggingface",
        "replicate",
        "perplexity",
        "xai",
        "deepseek",
        "fireworks_ai",
        "openrouter",
        "gemini",
    }
    return pref not in known_provider_prefixes


# ---------------------------------------------------------------------------
# Provider strategies
# ---------------------------------------------------------------------------


class LLMProviderStrategy(Protocol):
    def build_config(self) -> LLMConfig:  # pragma: no cover
        ...


@dataclass(frozen=True, slots=True)
class OllamaProviderStrategy(LLMProviderStrategy):
    model: str = "mistral-3:14b"
    base_url: str = "http://localhost:11434"

    def build_config(self) -> LLMConfig:
        cfg = LLMConfig(
            provider=f"ollama/{self.model}",
            api_token=None,
            base_url=self.base_url,
        )
        logger.debug(
            "[llm] provider=ollama model=%s base_url=%s", self.model, self.base_url
        )
        return cfg


@dataclass(frozen=True, slots=True)
class VLLMProviderStrategy(LLMProviderStrategy):
    """
    Local vLLM via LiteLLM "hosted_vllm/<model_id>" routing.
    Default model is instruction-tuned to ensure tokenizer chat_template exists.
    """

    model_id: str = "google/gemma-3-270m-it"
    base_url: str = "http://127.0.0.1:8000/v1"
    placeholder_api_token: str = "EMPTY"

    def build_config(self) -> LLMConfig:
        mid = _ensure_non_empty(self.model_id, "VLLMProviderStrategy.model_id")
        bu = _validate_vllm_base_url(self.base_url)

        token = os.getenv("HOSTED_VLLM_API_KEY") or self.placeholder_api_token

        cfg = LLMConfig(
            provider=f"hosted_vllm/{mid}",
            api_token=token,
            base_url=bu,
        )
        logger.debug("[llm] provider=hosted_vllm model_id=%s base_url=%s", mid, bu)
        return cfg


@dataclass(frozen=True, slots=True)
class RemoteAPIProviderStrategy(LLMProviderStrategy):
    """
    Pass-through for true remote providers (openai/..., gemini/..., anthropic/..., etc.)
    No rewriting. No OpenAI fallback.
    """

    provider: str
    api_token: Optional[str] = None
    base_url: Optional[str] = None

    def build_config(self) -> LLMConfig:
        prov = _ensure_non_empty(self.provider, "RemoteAPIProviderStrategy.provider")

        token = self.api_token
        if token is None:
            token = _infer_api_token_for_provider(prov)

        cfg = LLMConfig(provider=prov, api_token=token, base_url=self.base_url)
        logger.debug("[llm] provider=%s base_url=%s", prov, self.base_url)
        return cfg


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def provider_strategy_from_llm_model_selector(
    llm_model: Optional[str],
    *,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    default_vllm_model_id: str = "google/gemma-3-270m-it",
    ollama_base_url: str = "http://localhost:11434",
    default_ollama_model: str = "mistral-3:14b",
    api_token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProviderStrategy:
    """
    Explicit, deterministic selection. NO silent OpenAI fallback.

    - Empty -> local vLLM default (hosted_vllm/<model_id>)
    - vllm/<model_id> or hosted_vllm/<model_id> -> local vLLM
    - ollama/<model> -> Ollama
    - no "/" -> legacy -> Ollama
    - bare org/model (plain model id) -> local vLLM
    - otherwise -> remote provider string as-is
    """
    s = (llm_model or "").strip()
    if not s:
        return VLLMProviderStrategy(
            model_id=default_vllm_model_id, base_url=vllm_base_url
        )

    low = s.lower()

    if low.startswith("vllm/"):
        model_id = _strip_known_prefix(s, s[:5])
        return VLLMProviderStrategy(model_id=model_id, base_url=vllm_base_url)

    if low.startswith("hosted_vllm/"):
        model_id = _strip_known_prefix(s, s[:12])
        return VLLMProviderStrategy(model_id=model_id, base_url=vllm_base_url)

    if low.startswith("ollama/"):
        model = _strip_known_prefix(s, s[:7])
        return OllamaProviderStrategy(model=model, base_url=ollama_base_url)

    if "/" not in s:
        return OllamaProviderStrategy(
            model=(s or default_ollama_model), base_url=ollama_base_url
        )

    if _is_plain_model_id(s):
        return VLLMProviderStrategy(model_id=s, base_url=vllm_base_url)

    return RemoteAPIProviderStrategy(provider=s, api_token=api_token, base_url=base_url)


# ---------------------------------------------------------------------------
# Extraction factory + caching
# ---------------------------------------------------------------------------


@dataclass
class LLMExtractionFactory:
    provider_strategy: LLMProviderStrategy

    default_chunk_token_threshold: int = 2200
    default_overlap_rate: float = 0.10
    default_apply_chunking: bool = True
    default_input_format: str = "fit_markdown"

    default_schema_temperature: float = 0.2
    default_presence_temperature: float = 0.0

    default_schema_max_tokens: int = 2200
    default_presence_max_tokens: int = 350

    def create(
        self,
        *,
        mode: str,
        instruction: str,
        schema: Optional[Dict[str, Any]] = None,
        extraction_type: str = "schema",
        input_format: Optional[str] = None,
        chunk_token_threshold: Optional[int] = None,
        overlap_rate: Optional[float] = None,
        apply_chunking: Optional[bool] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> LLMExtractionStrategy:
        m = (mode or "").strip().lower()
        if m not in {"schema", "presence"}:
            raise ValueError(
                f"Unsupported mode={mode!r}; expected 'schema' or 'presence'."
            )

        llm_cfg = self.provider_strategy.build_config()

        provider = _clean_ws(_to_text(getattr(llm_cfg, "provider", "")))
        if not provider:
            raise RuntimeError(
                "LLMConfig.provider is empty. Refusing to proceed to avoid hidden fallback."
            )

        used_input_format = (
            input_format or self.default_input_format or "markdown"
        ).strip()
        used_chunk_threshold = (
            self.default_chunk_token_threshold
            if chunk_token_threshold is None
            else int(chunk_token_threshold)
        )
        used_overlap_rate = (
            self.default_overlap_rate if overlap_rate is None else float(overlap_rate)
        )
        used_apply_chunking = (
            self.default_apply_chunking
            if apply_chunking is None
            else bool(apply_chunking)
        )

        if m == "presence":
            used_schema = PresencePayload.model_json_schema()
            temperature = self.default_presence_temperature
            max_tokens = self.default_presence_max_tokens
            used_extraction_type = "schema"
        else:
            used_schema = schema or ExtractionPayload.model_json_schema()
            temperature = self.default_schema_temperature
            max_tokens = self.default_schema_max_tokens
            used_extraction_type = extraction_type

        _extra: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_args:
            _extra.update(extra_args)

        if _is_gemini_like_provider(provider):
            _extra.setdefault("response_format", {"type": "json_object"})

        return LLMExtractionStrategy(
            llm_config=llm_cfg,
            schema=used_schema,
            extraction_type=used_extraction_type,
            instruction=instruction,
            chunk_token_threshold=used_chunk_threshold,
            overlap_rate=used_overlap_rate,
            apply_chunking=used_apply_chunking,
            input_format=used_input_format,
            extra_args=_extra,
            verbose=verbose,
        )


@dataclass
class IndustryAwareStrategyCache:
    """
    Cache strategies keyed by:
      - llm mode (presence/schema)
      - context key (industry_label + industry + nace) OR "default"
      - provider fingerprint (no secrets)
      - schema hash + factory knobs
    """

    factory: LLMExtractionFactory
    schema: Optional[Dict[str, Any]] = None
    extraction_type: str = "schema"
    input_format: Optional[str] = None
    extra_args: Optional[Dict[str, Any]] = None
    verbose: bool = False

    _cache: Dict[
        Tuple[str, str, str, str, str, str, str, bool], LLMExtractionStrategy
    ] = field(default_factory=dict)
    _provider_fp: Optional[str] = None
    _factory_fp: Optional[str] = None

    def _get_provider_fingerprint(self) -> str:
        if self._provider_fp is not None:
            return self._provider_fp
        cfg = self.factory.provider_strategy.build_config()
        provider = _clean_ws(_to_text(getattr(cfg, "provider", "")))
        base_url = _clean_ws(_to_text(getattr(cfg, "base_url", "")))
        if not provider:
            raise RuntimeError(
                "LLM provider is empty; provider_strategy.build_config() must provide .provider."
            )
        self._provider_fp = _stable_json_sha1(
            {"provider": provider, "base_url": base_url}
        )
        return self._provider_fp

    def _get_factory_fingerprint(self) -> str:
        if self._factory_fp is not None:
            return self._factory_fp
        f = self.factory
        payload = {
            "default_chunk_token_threshold": f.default_chunk_token_threshold,
            "default_overlap_rate": f.default_overlap_rate,
            "default_apply_chunking": f.default_apply_chunking,
            "default_input_format": f.default_input_format,
            "default_schema_temperature": f.default_schema_temperature,
            "default_presence_temperature": f.default_presence_temperature,
            "default_schema_max_tokens": f.default_schema_max_tokens,
            "default_presence_max_tokens": f.default_presence_max_tokens,
            "provider_fp": self._get_provider_fingerprint(),
        }
        self._factory_fp = _stable_json_sha1(payload)
        return self._factory_fp

    def get_default_strategy(self, *, mode: str) -> LLMExtractionStrategy:
        m = (mode or "").strip().lower()
        if m not in {"schema", "presence"}:
            raise ValueError(f"mode must be 'schema' or 'presence', got {mode!r}")

        used_input_format = (
            (self.input_format or self.factory.default_input_format or "markdown")
            .strip()
            .lower()
        )
        used_extraction_type = (self.extraction_type or "schema").strip().lower()

        if m == "presence":
            used_schema = PresencePayload.model_json_schema()
            instruction = BASE_PRESENCE_INSTRUCTION
        else:
            used_schema = self.schema or ExtractionPayload.model_json_schema()
            instruction = BASE_FULL_INSTRUCTION

        schema_hash = _stable_json_sha1(used_schema)
        extra_hash = _stable_json_sha1(self.extra_args or {})
        provider_fp = self._get_provider_fingerprint()
        _ = self._get_factory_fingerprint()

        ctx_key = "default"
        key = (
            m,
            ctx_key,
            provider_fp,
            schema_hash,
            used_extraction_type,
            used_input_format,
            extra_hash,
            bool(self.verbose),
        )

        st = self._cache.get(key)
        if st is not None:
            return st

        if m == "presence":
            st = self.factory.create(
                mode="presence",
                instruction=instruction,
                input_format=used_input_format,
                extra_args=self.extra_args,
                verbose=self.verbose,
            )
        else:
            st = self.factory.create(
                mode="schema",
                instruction=instruction,
                schema=used_schema,
                extraction_type=used_extraction_type,
                input_format=used_input_format,
                extra_args=self.extra_args,
                verbose=self.verbose,
            )

        self._cache[key] = st
        return st

    def get_strategy(self, *, mode: str, ctx: IndustryContext) -> LLMExtractionStrategy:
        m = (mode or "").strip().lower()
        if m not in {"schema", "presence"}:
            raise ValueError(f"mode must be 'schema' or 'presence', got {mode!r}")

        used_input_format = (
            (self.input_format or self.factory.default_input_format or "markdown")
            .strip()
            .lower()
        )
        used_extraction_type = (self.extraction_type or "schema").strip().lower()

        if m == "presence":
            used_schema = PresencePayload.model_json_schema()
            instruction = build_presence_instruction(ctx)
        else:
            used_schema = self.schema or ExtractionPayload.model_json_schema()
            instruction = build_full_instruction(ctx)

        schema_hash = _stable_json_sha1(used_schema)
        extra_hash = _stable_json_sha1(self.extra_args or {})
        provider_fp = self._get_provider_fingerprint()
        _ = self._get_factory_fingerprint()

        ctx_key = _stable_json_sha1(
            {
                "industry_label": (ctx.industry_label or "").strip(),
                "industry": ctx.industry,
                "nace": ctx.nace,
            }
        )

        key = (
            m,
            ctx_key,
            provider_fp,
            schema_hash,
            used_extraction_type,
            used_input_format,
            extra_hash,
            bool(self.verbose),
        )

        st = self._cache.get(key)
        if st is not None:
            return st

        if m == "presence":
            st = self.factory.create(
                mode="presence",
                instruction=instruction,
                input_format=used_input_format,
                extra_args=self.extra_args,
                verbose=self.verbose,
            )
        else:
            st = self.factory.create(
                mode="schema",
                instruction=instruction,
                schema=used_schema,
                extraction_type=used_extraction_type,
                input_format=used_input_format,
                extra_args=self.extra_args,
                verbose=self.verbose,
            )

        self._cache[key] = st
        return st
