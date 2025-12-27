from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    Annotated,
)

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from crawl4ai import LLMConfig, LLMExtractionStrategy

from configs.llm_industry import get_industry_profile, normalize_industry_code
from configs.llm_industry.base import (
    BASE_FULL_INSTRUCTION,
    BASE_PRESENCE_INSTRUCTION,
    compose_full_instruction,
    compose_presence_instruction,
    IndustryLLMProfile,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants / helpers
# --------------------------------------------------------------------------- #

NAME_MAX = 160
DESC_MAX = 520
TAG_MAX = 50

DEBUG_PREVIEW_CHARS = 900
DEBUG_RAW_CHARS = 7000

EVIDENCE_SNIPPET_MAX_CHARS = 240

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


def _short_preview(v: Any, *, max_chars: int = DEBUG_PREVIEW_CHARS) -> str:
    s = _clean_ws(_to_text(v))
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _head_tail(s: str, *, n: int = 260) -> Tuple[str, str]:
    ss = _clean_ws(s)
    head = ss[:n] + ("…" if len(ss) > n else "")
    tail = ("…" if len(ss) > n else "") + ss[-n:]
    return head, tail


def _estimate_tokens(text: str, word_token_rate: float = 0.75) -> int:
    words = len((text or "").split())
    return int(words / max(word_token_rate, 0.01))


def _chunk_text(
    text: str,
    *,
    chunk_token_threshold: int,
    overlap_rate: float,
    word_token_rate: float = 0.75,
) -> List[str]:
    """
    Approximate chunking by words so we can support Crawl4AI variants where
    extract(url, ix, <chunk>) is called per-chunk.
    """
    words = (text or "").split()
    if not words:
        return [""]

    max_words = max(80, int(chunk_token_threshold * word_token_rate))
    if len(words) <= max_words:
        return [" ".join(words)]

    overlap_words = int(max_words * max(0.0, min(overlap_rate, 0.5)))
    step = max(1, max_words - overlap_words)

    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks


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


def _dedup_norm_list(items: Iterable[str], *, max_items: int) -> List[str]:
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


def _provider_prefix(provider: str) -> str:
    p = (provider or "").strip()
    if "/" in p:
        return p.split("/", 1)[0].strip().lower()
    return p.lower()


def _infer_api_token(provider: str) -> Optional[str]:
    """
    LiteLLM commonly supports provider-specific env vars.
    We support both:
      - gemini/<model>
      - google/<model> (some setups pass this through LiteLLM)
    """
    pref = _provider_prefix(provider)
    if pref in {"gemini", "google"}:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return None


def _is_gemini_like_provider(provider: Optional[str]) -> bool:
    return _provider_prefix(provider or "") in {"gemini", "google"}


# --------------------------------------------------------------------------- #
# Pydantic schemas (SIMPLIFIED, higher recall)
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
            (_clean_ws(_to_text(x))[:TAG_MAX] for x in items), max_items=10
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


# --------------------------------------------------------------------------- #
# Provider strategies
# --------------------------------------------------------------------------- #


class LLMProviderStrategy(Protocol):
    def build_config(self) -> LLMConfig:  # pragma: no cover
        ...


@dataclass
class OllamaProviderStrategy(LLMProviderStrategy):
    # Align default with repo flag help text
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


@dataclass
class RemoteAPIProviderStrategy(LLMProviderStrategy):
    """
    Generic LiteLLM remote provider.
    provider is passed as-is (e.g. "gemini/gemini-2.0-flash" or "google/gemini-1.5-flash").
    """

    provider: str
    api_token: Optional[str] = None
    base_url: Optional[str] = None

    def build_config(self) -> LLMConfig:
        if not self.provider:
            raise ValueError(
                "RemoteAPIProviderStrategy requires non-empty provider string."
            )

        token = self.api_token
        if token is None:
            token = _infer_api_token(self.provider)
            if token and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[llm] inferred api_token from env for provider=%s",
                    self.provider,
                )

        cfg = LLMConfig(provider=self.provider, api_token=token, base_url=self.base_url)
        logger.debug("[llm] provider=%s base_url=%s", self.provider, self.base_url)
        return cfg


def provider_strategy_from_llm_model_selector(
    llm_model: Optional[str],
    *,
    ollama_base_url: str = "http://localhost:11434",
    default_ollama_model: str = "mistral-3:14b",
    api_token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProviderStrategy:
    """
    Repo convention:

      - If llm_model contains "/": treat as remote provider/model (passed through to LiteLLM).
        Example: "google/gemini-1.5-flash" or "gemini/gemini-2.0-flash".

      - Else: treat as ollama model name (e.g. "mistral-3:14b").

      - If omitted/None/empty: use default_ollama_model via Ollama.
    """
    s = (llm_model or "").strip()
    if not s:
        return OllamaProviderStrategy(
            model=default_ollama_model, base_url=ollama_base_url
        )

    if "/" in s:
        return RemoteAPIProviderStrategy(
            provider=s, api_token=api_token, base_url=base_url
        )

    return OllamaProviderStrategy(model=s, base_url=ollama_base_url)


# --------------------------------------------------------------------------- #
# Default instructions (DRY: sourced from configs/llm_industry/base.py)
# --------------------------------------------------------------------------- #

DEFAULT_PRESENCE_INSTRUCTION = BASE_PRESENCE_INSTRUCTION
DEFAULT_FULL_INSTRUCTION = BASE_FULL_INSTRUCTION


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


@dataclass
class LLMExtractionFactory:
    provider_strategy: LLMProviderStrategy
    default_full_instruction: str = DEFAULT_FULL_INSTRUCTION
    default_presence_instruction: str = DEFAULT_PRESENCE_INSTRUCTION

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
        mode: str = "schema",  # "schema" | "presence"
        schema: Optional[Dict[str, Any]] = None,
        instruction: Optional[str] = None,
        extraction_type: str = "schema",
        input_format: Optional[str] = None,
        chunk_token_threshold: Optional[int] = None,
        overlap_rate: Optional[float] = None,
        apply_chunking: Optional[bool] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> LLMExtractionStrategy:
        m = (mode or "schema").strip().lower()
        if m not in {"schema", "presence"}:
            raise ValueError(
                f"Unsupported mode={mode!r}; expected 'schema' or 'presence'."
            )

        llm_cfg = self.provider_strategy.build_config()

        used_instruction = (
            instruction
            if instruction is not None
            else (
                self.default_presence_instruction
                if m == "presence"
                else self.default_full_instruction
            )
        )

        if m == "presence":
            used_schema = PresencePayload.model_json_schema()
            used_extraction_type = "schema"
            temperature = self.default_presence_temperature
            max_tokens = self.default_presence_max_tokens
        else:
            used_schema = schema or ExtractionPayload.model_json_schema()
            used_extraction_type = extraction_type
            temperature = self.default_schema_temperature
            max_tokens = self.default_schema_max_tokens

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
        used_input_format = (
            input_format or self.default_input_format or "markdown"
        ).strip()

        _extra: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_args:
            _extra.update(extra_args)

        # Gemini-like models: force JSON object output for higher success rates.
        if _is_gemini_like_provider(getattr(llm_cfg, "provider", None)):
            _extra.setdefault("response_format", {"type": "json_object"})

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm.factory] mode=%s provider=%s input_format=%s chunk=%s overlap=%s apply_chunking=%s temp=%s max_tokens=%s verbose=%s",
                m,
                getattr(llm_cfg, "provider", None),
                used_input_format,
                used_chunk_threshold,
                used_overlap_rate,
                used_apply_chunking,
                temperature,
                max_tokens,
                bool(verbose),
            )

        return LLMExtractionStrategy(
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


# --------------------------------------------------------------------------- #
# Industry-aware strategy cache (build per-industry instructions DRY)
# --------------------------------------------------------------------------- #


@dataclass
class IndustryStrategyCache:
    """
    Build + cache LLMExtractionStrategy per (mode, industry_code).

    - Uses shared schema from llm.py
    - Uses shared base instructions from configs/llm_industry/base.py
    - Uses only industry addendums from configs/llm_industry/industry_XX_*.py
    """

    factory: LLMExtractionFactory
    schema: Optional[Dict[str, Any]] = None
    extraction_type: str = "schema"
    input_format: Optional[str] = None
    extra_args: Optional[Dict[str, Any]] = None
    verbose: bool = False

    _cache: Dict[Tuple[str, str], LLMExtractionStrategy] = field(default_factory=dict)

    def get_profile(self, industry_code: object) -> IndustryLLMProfile:
        return get_industry_profile(industry_code)

    def get_strategy(
        self, *, mode: str, industry_code: object
    ) -> LLMExtractionStrategy:
        m = (mode or "schema").strip().lower()
        if m not in {"schema", "presence"}:
            raise ValueError(f"mode must be 'schema' or 'presence', got {mode!r}")

        code = normalize_industry_code(industry_code)
        key = (m, code)

        st = self._cache.get(key)
        if st is not None:
            return st

        profile = self.get_profile(code)
        if m == "presence":
            instruction = compose_presence_instruction(profile)
            st = self.factory.create(
                mode="presence",
                instruction=instruction,
                input_format=self.input_format,
                extra_args=self.extra_args,
                verbose=self.verbose,
            )
        else:
            instruction = compose_full_instruction(profile)
            st = self.factory.create(
                mode="schema",
                schema=self.schema or ExtractionPayload.model_json_schema(),
                instruction=instruction,
                extraction_type=self.extraction_type,
                input_format=self.input_format,
                extra_args=self.extra_args,
                verbose=self.verbose,
            )

        self._cache[key] = st
        return st


# --------------------------------------------------------------------------- #
# JSON salvage WITHOUT recursive regex (Python 3.13 safe)
# --------------------------------------------------------------------------- #


def _iter_balanced_json_blobs(s: str) -> Iterable[str]:
    if not s:
        return

    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch not in "{[":
            i += 1
            continue

        open_ch = ch
        close_ch = "}" if open_ch == "{" else "]"
        start = i

        stack = [close_ch]
        i += 1

        in_str = False
        esc = False

        while i < n and stack:
            c = s[i]

            if in_str:
                if esc:
                    esc = False
                else:
                    if c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                i += 1
                continue

            if c == '"':
                in_str = True
                i += 1
                continue

            if c == "{":
                stack.append("}")
            elif c == "[":
                stack.append("]")
            elif c == "}" or c == "]":
                if stack and c == stack[-1]:
                    stack.pop()
                else:
                    stack.clear()
                    break

            i += 1

        if not stack:
            blob = s[start:i]
            yield blob
        i = start + 1


def _extract_first_json_blob(s: str) -> Optional[str]:
    if not s:
        return None

    ss = s.strip()
    if (ss.startswith("{") and ss.endswith("}")) or (
        ss.startswith("[") and ss.endswith("]")
    ):
        return ss

    best_any: Optional[str] = None
    for blob in _iter_balanced_json_blobs(ss):
        if best_any is None:
            best_any = blob
        if any(
            k in blob
            for k in ('"offerings"', '"items"', '"products"', '"services"', '"r"')
        ):
            return blob
    return best_any


# --------------------------------------------------------------------------- #
# Parsing (robust, tolerant) – simplified schema but accepts legacy shapes
# --------------------------------------------------------------------------- #


def _looks_like_offering_dict(d: Dict[str, Any]) -> bool:
    name = _clean_ws(_to_text(d.get("name") or d.get("title") or d.get("label")))
    desc = _clean_ws(
        _to_text(d.get("description") or d.get("summary") or d.get("details"))
    )
    typ = _clean_ws(
        _to_text(d.get("type") or d.get("kind") or d.get("category"))
    ).lower()
    return bool(name) and (bool(desc) or bool(typ) or isinstance(d.get("tags"), list))


def _guess_type_from_text(name: str, desc: str, tags: List[str]) -> str:
    s = (name + " " + desc + " " + " ".join(tags)).lower()
    svc_hints = (
        "service",
        "solution",
        "capabilit",
        "consult",
        "managed",
        "support",
        "maintenance",
        "repair",
        "installation",
        "deployment",
        "integration",
        "migration",
        "logistics",
        "fulfillment",
        "training",
        "certification",
        "testing",
        "inspection",
        "manufactur",
        "contract manufacturing",
    )
    return "service" if any(h in s for h in svc_hints) else "product"


def _sanitize_offering(d: Dict[str, Any]) -> Dict[str, Any]:
    name = _clean_ws(_to_text(d.get("name") or d.get("title") or d.get("label")))[
        :NAME_MAX
    ]
    desc = _clean_ws(
        _to_text(d.get("description") or d.get("summary") or d.get("details"))
    )[:DESC_MAX]
    typ = _clean_ws(
        _to_text(d.get("type") or d.get("kind") or d.get("category"))
    ).lower()

    tags = d.get("tags") or d.get("keywords") or d.get("key_terms") or d.get("tag")
    tag_list = _dedup_norm_list(
        (_clean_ws(_to_text(x))[:TAG_MAX] for x in _as_list(tags)), max_items=10
    )

    if typ in {"products", "product", "prod", "brand", "line", "goods"}:
        typ = "product"
    elif typ in {
        "services",
        "service",
        "svc",
        "solutions",
        "solution",
        "capabilities",
        "capability",
    }:
        typ = "service"
    elif typ not in {"product", "service"}:
        typ = _guess_type_from_text(name, desc, tag_list)

    if not desc:
        desc = name

    return {"type": typ, "name": name, "description": desc, "tags": tag_list}


def _collect_offering_candidates(obj: Any) -> Iterable[Dict[str, Any]]:
    if obj is None:
        return
    if isinstance(obj, str):
        return

    if isinstance(obj, dict):
        if isinstance(obj.get("offerings"), list):
            for o in obj["offerings"]:
                if isinstance(o, dict):
                    yield o
        if isinstance(obj.get("items"), list):
            for o in obj["items"]:
                if isinstance(o, dict):
                    yield o

        if isinstance(obj.get("products"), list):
            for p in obj["products"]:
                if isinstance(p, dict):
                    p2 = dict(p)
                    p2.setdefault("type", "product")
                    yield p2
                elif isinstance(p, str):
                    yield {"type": "product", "name": p, "description": p, "tags": []}

        if isinstance(obj.get("services"), list):
            for s in obj["services"]:
                if isinstance(s, dict):
                    s2 = dict(s)
                    s2.setdefault("type", "service")
                    yield s2
                elif isinstance(s, str):
                    yield {"type": "service", "name": s, "description": s, "tags": []}

        if _looks_like_offering_dict(obj):
            yield obj
        return

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield from _collect_offering_candidates(it)


def parse_extracted_payload(
    extracted_content: Union[str, bytes, Dict[str, Any], List[Any], None],
) -> ExtractionPayload:
    if extracted_content is None:
        logger.debug("[llm.parse] extracted_content=None -> empty payload")
        return ExtractionPayload(offerings=[])

    raw_preview = _short_preview(extracted_content, max_chars=DEBUG_PREVIEW_CHARS)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[llm.parse] raw_type=%s preview=%s",
            type(extracted_content).__name__,
            raw_preview,
        )

    data: Any = extracted_content

    if isinstance(extracted_content, (str, bytes, bytearray)):
        s0 = _to_text(extracted_content).strip()
        try:
            data = json.loads(s0)
        except Exception:
            blob = _extract_first_json_blob(s0)
            if blob:
                try:
                    data = json.loads(blob)
                except Exception as e2:
                    logger.debug(
                        "[llm.parse] salvage json.loads failed err=%s blob_preview=%s",
                        e2,
                        _short_preview(blob, max_chars=600),
                    )
                    return ExtractionPayload(offerings=[])
            else:
                logger.debug("[llm.parse] json missing; preview=%s", raw_preview)
                return ExtractionPayload(offerings=[])

    if isinstance(data, dict):
        for k in ("extracted_content", "content", "data", "result", "output"):
            if isinstance(data.get(k), (str, bytes, bytearray)):
                inner = _to_text(data[k]).strip()
                try:
                    data = json.loads(inner)
                    logger.debug(
                        "[llm.parse] unwrapped key=%s -> %s", k, type(data).__name__
                    )
                    break
                except Exception:
                    blob = _extract_first_json_blob(inner)
                    if blob:
                        try:
                            data = json.loads(blob)
                            logger.debug(
                                "[llm.parse] unwrapped+salvaged key=%s -> %s",
                                k,
                                type(data).__name__,
                            )
                            break
                        except Exception:
                            pass

    candidates = list(_collect_offering_candidates(data))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[llm.parse] candidates=%d", len(candidates))

    valid: List[Offering] = []
    seen: Dict[Tuple[str, str], int] = {}

    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            continue
        try:
            cleaned = _sanitize_offering(c)
            off = Offering.model_validate(cleaned)
        except ValidationError as e:
            logger.debug(
                "[llm.parse] drop idx=%d validation_err=%s raw=%s",
                i,
                e.errors()[:2],
                _short_preview(c, max_chars=500),
            )
            continue
        except Exception as e:
            logger.debug(
                "[llm.parse] drop idx=%d unexpected=%s raw=%s",
                i,
                e,
                _short_preview(c, max_chars=500),
            )
            continue

        key = (off.type, off.name.lower())
        if key in seen:
            j = seen[key]
            cur = valid[j]
            pick = False
            if len(off.description) > len(cur.description):
                pick = True
            elif len(off.description) == len(cur.description) and len(off.tags) > len(
                cur.tags
            ):
                pick = True
            if pick:
                valid[j] = off
        else:
            seen[key] = len(valid)
            valid.append(off)

    return ExtractionPayload(offerings=valid)


def parse_presence_result(
    extracted_content: Union[str, bytes, int, float, Dict[str, Any], List[Any], None],
    *,
    default: bool = False,
) -> Tuple[bool, Optional[float], Optional[str]]:
    preview = _short_preview(extracted_content, max_chars=DEBUG_PREVIEW_CHARS)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[llm.presence.parse] raw_type=%s preview=%s",
            type(extracted_content).__name__,
            preview,
        )

    if extracted_content is None:
        return (default, None, preview)

    loaded: Any = extracted_content
    if isinstance(extracted_content, (str, bytes, bytearray)):
        s = _to_text(extracted_content).strip()
        try:
            loaded = json.loads(s)
        except Exception:
            blob = _extract_first_json_blob(s)
            if blob:
                try:
                    loaded = json.loads(blob)
                except Exception:
                    loaded = s
            else:
                loaded = s

    def _scalar01(v: Any) -> Optional[int]:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int) and v in (0, 1):
            return v
        if isinstance(v, float) and v in (0.0, 1.0):
            return int(v)
        if isinstance(v, str):
            ss = v.strip().lower()
            if ss in {"0", "1"}:
                return int(ss)
            if ss in {"true", "yes"}:
                return 1
            if ss in {"false", "no"}:
                return 0
        return None

    def _conf(v: Any) -> Optional[float]:
        try:
            f = float(v)
            return f if 0.0 <= f <= 1.0 else None
        except Exception:
            return None

    s01 = _scalar01(loaded)
    if s01 is not None:
        return (bool(s01), None, preview)

    if isinstance(loaded, dict):
        conf = None
        for ck in ("confidence", "score", "prob", "p"):
            if ck in loaded:
                conf = _conf(loaded.get(ck))
                if conf is not None:
                    break

        if "r" in loaded:
            r = _scalar01(loaded["r"])
            if r is not None:
                return (bool(r), conf, preview)

        for key in (
            "has_offering",
            "presence",
            "present",
            "classification",
            "result",
            "value",
        ):
            if key in loaded:
                r = _scalar01(loaded[key])
                if r is not None:
                    return (bool(r), conf, preview)

        return (default, conf, preview)

    if isinstance(loaded, list) and loaded:
        first = loaded[0]
        s01 = _scalar01(first)
        if s01 is not None:
            return (bool(s01), None, preview)
        if isinstance(first, dict):
            return parse_presence_result(first, default=default)

    return (default, None, preview)


# --------------------------------------------------------------------------- #
# Evidence-based culling (SOFTER to avoid false negatives)
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "to",
    "a",
    "an",
    "for",
    "in",
    "on",
    "with",
    "by",
    "at",
    "from",
    "into",
    "our",
    "your",
    "their",
    "its",
    "is",
    "are",
    "be",
    "as",
    "this",
    "that",
    "these",
    "those",
    "we",
    "you",
    "they",
}

_JUNK_NAMES = {
    "home",
    "about",
    "contact",
    "privacy",
    "terms",
    "cookie",
    "cookies",
    "careers",
    "news",
    "blog",
    "sitemap",
    "login",
    "sign in",
    "signin",
    "register",
}

_TRADEMARK_CHARS_RE = re.compile(r"[®™©]+")


def _strip_trademarks(s: str) -> str:
    return _TRADEMARK_CHARS_RE.sub("", s or "").strip()


def _mainish_text(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            return "\n".join(lines[i:])
    return text


def _keyword_tokens(s: str) -> List[str]:
    s = _strip_trademarks(_clean_ws(s)).lower()
    toks = re.split(r"[^a-z0-9]+", s)
    out: List[str] = []
    for t in toks:
        if len(t) < 4:
            continue
        if t in _STOPWORDS:
            continue
        out.append(t)
    return out


def _find_snippet(haystack: str, needle: str) -> Optional[str]:
    if not haystack or not needle:
        return None
    m = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
    if not m:
        return None
    start = max(0, m.start() - (EVIDENCE_SNIPPET_MAX_CHARS // 2))
    end = min(len(haystack), m.end() + (EVIDENCE_SNIPPET_MAX_CHARS // 2))
    snip = _clean_ws(haystack[start:end])
    return snip[:EVIDENCE_SNIPPET_MAX_CHARS] if snip else None


def _has_any_evidence_soft(off: Offering, source_text: str) -> bool:
    t = _mainish_text(source_text)
    if not t:
        return True

    name = _clean_ws(off.name)
    if not name:
        return False

    name2 = _strip_trademarks(name)
    if _find_snippet(t, name) or (
        _find_snippet(t, name2) if name2 and name2 != name else None
    ):
        return True

    name_keys = _keyword_tokens(name2 or name)
    if name_keys:
        for k in set(name_keys):
            if re.search(rf"\b{re.escape(k)}\b", t, flags=re.IGNORECASE):
                return True

    desc_keys = _keyword_tokens(off.description)
    hits = 0
    for k in set(desc_keys):
        if re.search(rf"\b{re.escape(k)}\b", t, flags=re.IGNORECASE):
            hits += 1
            if hits >= 2:
                return True

    return False


def _filter_offerings_soft(
    offerings: List[Offering], *, source_text: str
) -> List[Offering]:
    kept: List[Offering] = []
    dropped = 0

    for off in offerings:
        nm = _clean_ws(off.name).lower()
        if nm in _JUNK_NAMES:
            dropped += 1
            continue

        if off.tags:
            kept.append(off)
            continue

        ok = _has_any_evidence_soft(off, source_text)
        if ok:
            kept.append(off)
            continue

        desc = _clean_ws(off.description).lower()
        if len(desc) < 40 or desc in _JUNK_NAMES or desc == nm:
            dropped += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[llm.filter] drop_soft type=%s name=%s desc=%s",
                    off.type,
                    off.name,
                    _short_preview(off.description, max_chars=220),
                )
            continue

        kept.append(off)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[llm.filter] kept=%d dropped=%d", len(kept), dropped)

    out: List[Offering] = []
    seen: Dict[Tuple[str, str], int] = {}
    for off in kept:
        key = (off.type, off.name.lower())
        if key in seen:
            j = seen[key]
            cur = out[j]
            pick = False
            if len(off.description) > len(cur.description):
                pick = True
            elif len(off.description) == len(cur.description) and len(off.tags) > len(
                cur.tags
            ):
                pick = True
            if pick:
                out[j] = off
        else:
            seen[key] = len(out)
            out.append(off)

    return out


# --------------------------------------------------------------------------- #
# Robust extract caller (handles Crawl4AI signature variants)
# --------------------------------------------------------------------------- #


def _debug_dump_raw(label: str, raw: Any) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    s = _to_text(raw)
    s = s if len(s) <= DEBUG_RAW_CHARS else s[:DEBUG_RAW_CHARS] + "…(truncated)"
    logger.debug("[%s] raw_dump=%s", label, s)


def _signature_summary(fn: Any) -> str:
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "<sig-unavailable>"


def _build_call_args(
    fn: Any,
    *,
    url: str,
    text: str,
    html: str,
    ix: int,
    strategy_input_format: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    sig = inspect.signature(fn)
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}

    primary = text
    if html and strategy_input_format.strip().lower() == "html":
        primary = html

    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue

        name = p.name.lower()

        def pick_value() -> Any:
            if name in {"url", "page_url"} or "url" in name:
                return url
            if name in {"ix", "idx", "index", "chunk_ix", "chunk_index"}:
                return ix
            if "fit_markdown" in name or ("fit" in name and "markdown" in name):
                return text
            if "markdown" in name or name in {
                "md",
                "text",
                "content",
                "document",
                "input",
            }:
                return text
            if "html" in name:
                return html if html else primary
            return "" if p.default is inspect._empty else p.default

        val = pick_value()
        if p.kind == p.KEYWORD_ONLY:
            kwargs[p.name] = val
        else:
            args.append(val)

    return args, kwargs


def call_llm_extract(
    strategy: Any,
    url: str,
    text: str,
    *,
    html: str = "",
    kind: str = "full",  # "full" | "presence"
    require_evidence: bool = True,
) -> Any:
    if text is None:
        text = ""
    text = _to_text(text)

    fn = getattr(strategy, "extract", None)
    if fn is None or not callable(fn):
        raise TypeError(
            f"strategy has no callable extract(): {type(strategy).__name__}"
        )

    sig_str = _signature_summary(fn)
    input_format = getattr(strategy, "input_format", "markdown") or "markdown"

    if logger.isEnabledFor(logging.DEBUG):
        head, tail = _head_tail(text, n=260)
        logger.debug(
            "[llm.call] url=%s text_len=%d text_sha1=%s head=%s tail=%s",
            url,
            len(text),
            _sha1_text(text),
            head,
            tail,
        )
        logger.debug("[llm.call] extract_signature=%s", sig_str)
        logger.debug("[llm.call] strategy_input_format=%s", input_format)

    requires_ix = False
    try:
        sig = inspect.signature(fn)
        requires_ix = any(
            p.name.lower() in {"ix", "idx", "index", "chunk_ix", "chunk_index"}
            for p in sig.parameters.values()
        )
    except Exception:
        requires_ix = False

    t0 = time.perf_counter()

    if requires_ix:
        apply_chunking = bool(getattr(strategy, "apply_chunking", True))
        chunk_threshold = int(getattr(strategy, "chunk_token_threshold", 2200))
        overlap_rate = float(getattr(strategy, "overlap_rate", 0.10))
        word_token_rate = float(getattr(strategy, "word_token_rate", 0.75))

        chunks = [text]
        if apply_chunking:
            est = _estimate_tokens(text, word_token_rate=word_token_rate)
            if est > chunk_threshold:
                chunks = _chunk_text(
                    text,
                    chunk_token_threshold=chunk_threshold,
                    overlap_rate=overlap_rate,
                    word_token_rate=word_token_rate,
                )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm.call] per_chunk_api=True chunks=%d apply_chunking=%s chunk_threshold=%s overlap=%s",
                len(chunks),
                apply_chunking,
                chunk_threshold,
                overlap_rate,
            )

        raw_results: List[Any] = []
        for ix, chunk in enumerate(chunks):
            args, kwargs = _build_call_args(
                fn,
                url=url,
                text=chunk,
                html=html,
                ix=ix,
                strategy_input_format=str(input_format),
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[llm.call] chunk_call ix=%d args_len=%d kwargs_keys=%s",
                    ix,
                    len(args),
                    sorted(kwargs.keys()),
                )
            raw_results.append(fn(*args, **kwargs))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm.call] done per_chunk elapsed_ms=%.1f last_raw_type=%s last_raw_preview=%s",
                elapsed_ms,
                type(raw_results[-1]).__name__,
                _short_preview(raw_results[-1]),
            )

        if kind == "presence":
            any_true = False
            best_conf: Optional[float] = None
            last_preview: Optional[str] = None
            for r in raw_results:
                has, conf, prev = parse_presence_result(r, default=False)
                any_true = any_true or has
                if conf is not None:
                    best_conf = conf if best_conf is None else max(best_conf, conf)
                last_preview = prev
            merged = {"r": 1 if any_true else 0}
            if logger.isEnabledFor(logging.DEBUG):
                _debug_dump_raw("llm.call.presence.merged", merged)
            return merged

        merged_payload = ExtractionPayload(offerings=[])
        for r in raw_results:
            p = parse_extracted_payload(r)
            if p.offerings:
                merged_payload.offerings.extend(p.offerings)

        offerings = merged_payload.offerings
        if require_evidence:
            offerings = _filter_offerings_soft(offerings, source_text=text)

        out_payload = ExtractionPayload(offerings=offerings)
        out_dict = out_payload.model_dump()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm.call] merged_offerings=%d final_offerings=%d",
                len(merged_payload.offerings),
                len(offerings),
            )
            _debug_dump_raw("llm.call.full.final", out_dict)

        return out_dict

    args, kwargs = _build_call_args(
        fn, url=url, text=text, html=html, ix=0, strategy_input_format=str(input_format)
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[llm.call] single_call args_len=%d kwargs_keys=%s",
            len(args),
            sorted(kwargs.keys()),
        )

    try:
        raw = fn(*args, **kwargs)
    except TypeError as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[llm.call] direct_call TypeError=%s; trying fallbacks", e)
        try:
            raw = fn(url, text)
        except Exception:
            try:
                raw = fn(url, text, html)
            except Exception:
                raw = fn(url, html, text)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[llm.call] done single elapsed_ms=%.1f raw_type=%s raw_preview=%s",
            elapsed_ms,
            type(raw).__name__,
            _short_preview(raw),
        )
        _debug_dump_raw("llm.call.raw", raw)

        try:
            total_usage = getattr(strategy, "total_usage", None)
            usages = getattr(strategy, "usages", None)
            if total_usage is not None or usages is not None:
                logger.debug(
                    "[llm.call] usage total=%s usages_preview=%s",
                    total_usage,
                    (usages[:3] if isinstance(usages, list) else usages),
                )
        except Exception:
            pass

    if kind == "presence":
        return raw

    payload = parse_extracted_payload(raw)
    offerings = payload.offerings
    if require_evidence:
        offerings = _filter_offerings_soft(offerings, source_text=text)

    out_payload = ExtractionPayload(offerings=offerings)
    out_dict = out_payload.model_dump()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[llm.call] single_offerings=%d final_offerings=%d",
            len(payload.offerings),
            len(offerings),
        )
        _debug_dump_raw("llm.call.full.final", out_dict)

    return out_dict


# --------------------------------------------------------------------------- #
# Defaults / exports
# --------------------------------------------------------------------------- #

default_ollama_provider_strategy = OllamaProviderStrategy()

__all__ = [
    "Offering",
    "ExtractionPayload",
    "PresencePayload",
    "LLMProviderStrategy",
    "OllamaProviderStrategy",
    "RemoteAPIProviderStrategy",
    "LLMExtractionFactory",
    "IndustryStrategyCache",
    "provider_strategy_from_llm_model_selector",
    "parse_extracted_payload",
    "parse_presence_result",
    "call_llm_extract",
    "default_ollama_provider_strategy",
    "DEFAULT_PRESENCE_INSTRUCTION",
    "DEFAULT_FULL_INSTRUCTION",
    # re-export useful profile helpers
    "IndustryLLMProfile",
    "get_industry_profile",
    "normalize_industry_code",
]
