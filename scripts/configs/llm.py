from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Annotated

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from crawl4ai import LLMConfig, LLMExtractionStrategy

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pydantic schemas (domain-level, language-agnostic)
# --------------------------------------------------------------------------- #

ShortStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)
]
EvidenceStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)
]


class Offering(BaseModel):
    """
    A single extracted offering (product or service).

    This is intentionally broad to support many industries and page types:
      - Products include: branded products, product lines, models, SKUs, packaged formats,
        and other concrete deliverables (including B2B formats/packaging when presented as an option).
      - Services include: solutions/programs/capabilities a customer can buy (e.g., private label,
        contract manufacturing, processing, implementation, sourcing, platform subscription).

    Fields:
      - type:
          "product" or "service". Use "product" for tangible deliverables or named product/brand lines.
          Use "service" for programs/capabilities/solutions provided to customers.
      - name:
          A short human-readable label for the offering. Prefer names/headings used on the page.
          If the page describes a capability without a formal name, create a concise name that is still
          clearly supported by the page (do not invent a brand).
      - description:
          1–3 sentences describing what it is, what it does, and (when present) who it is for.
          Must be grounded in the page content; avoid adding external facts.
      - evidence:
          1–3 short verbatim snippets copied from the page that directly support the offering.
          These are used to reduce hallucinations. Snippets should come from main content,
          not cookie banners/navigation/legal footers.
    """

    type: ShortStr = Field(
        ...,
        description="Offering kind: must normalize to exactly 'product' or 'service'.",
    )
    name: ShortStr = Field(
        ...,
        description="Short label for the offering; prefer on-page names/headings; may be concise but must be supported by evidence.",
    )
    description: ShortStr = Field(
        ...,
        description="Grounded 1–3 sentence summary of the offering (what it is/does/for whom), based only on page content.",
    )
    evidence: List[EvidenceStr] = Field(
        default_factory=list,
        description="1–3 verbatim snippets from the page that support this offering (short phrases/sentences, copied exactly).",
        max_length=3,
    )

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, v: Any) -> Any:
        if v is None:
            return v
        s = str(v).strip().lower()
        if s in {"products", "product", "prod", "good", "goods", "brand", "line"}:
            return "product"
        if s in {
            "services",
            "service",
            "svc",
            "solution",
            "solutions",
            "capability",
            "capabilities",
        }:
            return "service"
        # allow already-correct values
        if s in {"product", "service"}:
            return s
        # last resort: keep original to let validation fail upstream if needed
        return s

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            # some models return a single string; wrap it
            return [v]
        return v


class ExtractionPayload(BaseModel):
    """
    Full extraction payload for a page.

    - offerings:
        A (possibly empty) list of Offerings. Keep the list to the main offerings on the page:
        include brands/product lines listed under “Our Brands”, product/service sections, solution
        blocks, and format lists that represent real deliverables or purchasable options.
        Merge duplicates and avoid repeating the same offering with tiny wording changes.
    """

    offerings: List[Offering] = Field(
        default_factory=list,
        description="Zero or more extracted offerings (products/services) grounded by evidence.",
    )


class PresencePayload(BaseModel):
    """
    Presence-only payload:
      r = 0 -> no offerings present in the page's main content
      r = 1 -> at least one offering is present in the page's main content
    """

    r: int = Field(..., ge=0, le=1, description="0=no offering, 1=has offering")


# --------------------------------------------------------------------------- #
# Compatibility wrapper for Crawl4AI LLMExtractionStrategy.extract()
# --------------------------------------------------------------------------- #


class CompatLLMExtractionStrategy(LLMExtractionStrategy):
    """
    Crawl4AI LLMExtractionStrategy.extract() signatures vary across versions.

    Your offline pass calls:
        strategy.extract(url, text)

    Some Crawl4AI versions require:
        extract(url, markdown, html)
        extract(url, html, markdown)
        extract(url, markdown, html, fit_markdown=...)
        ...

    This subclass accepts the 2-arg call and forwards into the base implementation
    with safe defaults for missing fields, while preserving normal behavior for
    the crawler's multi-arg calls.
    """

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        base_extract = super().extract

        # If caller already supplies kwargs or 3+ args, just pass through.
        if kwargs or len(args) != 2:
            return base_extract(*args, **kwargs)

        url, text = args
        html = ""

        # Try to adapt to the base signature.
        try:
            sig = inspect.signature(base_extract)  # bound method: no "self"
            params = list(sig.parameters.values())
            names = [p.name.lower() for p in params]
        except Exception:
            params = []
            names = []

        # Common modern forms: (url, markdown, html) or (url, html, markdown)
        if len(params) >= 3:
            # Named mapping when possible
            try:
                if any("html" in n for n in names) and any(
                    ("markdown" in n)
                    or ("fit" in n)
                    or (n in {"md", "text", "content"})
                    for n in names
                ):
                    built: List[Any] = []
                    for n in names:
                        if n in {"url", "page_url"}:
                            built.append(url)
                        elif "html" in n:
                            built.append(html)
                        elif "fit_markdown" in n or ("fit" in n and "markdown" in n):
                            built.append(text)
                        elif "markdown" in n or n in {"md", "text", "content"}:
                            built.append(text)
                        else:
                            built.append("")
                    return base_extract(*built)
            except TypeError:
                pass

            # Positional fallbacks
            try:
                return base_extract(url, text, html)  # (url, markdown, html)
            except TypeError:
                return base_extract(url, html, text)  # (url, html, markdown)

        # Older/simpler forms: (url, something)
        return base_extract(url, text)


# --------------------------------------------------------------------------- #
# LLM provider strategies (Configuration-as-Strategy)
# --------------------------------------------------------------------------- #


class LLMProviderStrategy(Protocol):
    def build_config(self) -> LLMConfig:  # pragma: no cover
        ...


@dataclass
class OllamaProviderStrategy(LLMProviderStrategy):
    """
    Local provider strategy using Ollama.
    provider string becomes: "ollama/<model>"
    """

    model: str = "qwen3:30b"
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
    Remote API strategy (LiteLLM), e.g. "openai/gpt-4o-mini"
    """

    provider: str
    api_token: Optional[str] = None
    base_url: Optional[str] = None

    def build_config(self) -> LLMConfig:
        if not self.provider:
            raise ValueError(
                "RemoteAPIProviderStrategy requires a non-empty provider string."
            )

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
# NOTE: Keep instructions short; schema descriptions are the detailed contract.
# --------------------------------------------------------------------------- #

DEFAULT_PRESENCE_INSTRUCTION = (
    "Classify if the page's MAIN CONTENT contains any offerings (products/brands/product lines or services/solutions).\n"
    "Ignore cookie/privacy banners, accessibility boilerplate, navigation/menus, and legal footers.\n"
    'Return ONLY JSON: {"r":1} or {"r":0}. No other keys.'
)

DEFAULT_FULL_INSTRUCTION = (
    "Extract the company's offerings from the page.\n"
    "Ignore cookie/privacy banners, accessibility boilerplate, navigation/menus, and legal footers.\n"
    "Include products/brands/product lines and services/solutions that the company offers.\n"
    "Ground each item with 1–3 verbatim evidence snippets copied from the page; if you cannot ground it, omit it.\n"
    "Merge duplicates; keep descriptions concise and factual.\n"
    "Return ONLY valid JSON matching the provided schema."
)


# --------------------------------------------------------------------------- #
# Extraction factory (Strategy + Factory + DI)
# --------------------------------------------------------------------------- #


@dataclass
class LLMExtractionFactory:
    provider_strategy: LLMProviderStrategy
    default_full_instruction: str
    default_presence_instruction: str

    # Chunking / token defaults
    default_chunk_token_threshold: int = 1400
    default_overlap_rate: float = 0.08
    default_apply_chunking: bool = True

    # Always prefer fit_markdown for Crawl4AI LLM input
    default_input_format: str = "fit_markdown"

    # LLM behavior defaults
    # Slightly higher than 0.0 to reduce under-extraction, while evidence requirement limits hallucinations.
    default_schema_temperature: float = 0.25
    default_presence_temperature: float = 0.0
    default_schema_max_tokens: int = 1500
    default_presence_max_tokens: int = 600

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
        normalized_mode = (mode or "schema").strip().lower()
        if normalized_mode not in {"schema", "presence"}:
            raise ValueError(
                f"Unsupported mode={mode!r}; expected 'schema' or 'presence'."
            )

        llm_cfg = self.provider_strategy.build_config()

        used_instruction = (
            instruction
            if instruction is not None
            else (
                self.default_presence_instruction
                if normalized_mode == "presence"
                else self.default_full_instruction
            )
        )

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

        # Enforce fit_markdown
        if input_format is not None and str(input_format).strip() != "fit_markdown":
            logger.warning(
                "[llm] input_format override requested (%s) but forcing fit_markdown",
                input_format,
            )
        used_input_format = "fit_markdown"

        _extra: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not (extra_args and "response_format" in extra_args):
            _extra["response_format"] = {"type": "json_object"}
        if extra_args:
            _extra.update(extra_args)

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

        return CompatLLMExtractionStrategy(
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
# Robust parsing / normalization for schema mode
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
        return s if len(s) <= length else s[:length] + "…"
    except Exception:
        return "<preview-failed>"


def _looks_like_offering(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    t = (d.get("type") or d.get("kind") or "").strip().lower()
    n = (d.get("name") or "").strip()
    desc = (d.get("description") or d.get("summary") or "").strip()
    if t in {"products", "product", "prod"}:
        t = "product"
    if t in {"services", "service", "svc"}:
        t = "service"
    return bool(n) and bool(desc) and t in {"product", "service"}


def _normalize_offering_dict(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _looks_like_offering(d):
        return None
    t = (d.get("type") or d.get("kind") or "").strip().lower()
    if t in {"products", "product", "prod"}:
        t = "product"
    if t in {"services", "service", "svc"}:
        t = "service"
    out: Dict[str, Any] = {
        "type": t,
        "name": d.get("name", ""),
        "description": d.get("description") or d.get("summary") or "",
    }
    ev = d.get("evidence")
    if ev is None:
        out["evidence"] = []
    elif isinstance(ev, str):
        out["evidence"] = [ev]
    elif isinstance(ev, list):
        out["evidence"] = [x for x in ev if isinstance(x, (str, bytes, bytearray))][:3]
    else:
        out["evidence"] = []
    return out


def _extract_offerings_from_mixed_list(items: List[Any]) -> List[Dict[str, Any]]:
    offerings: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("error") is True:
            continue

        if "offerings" in it and isinstance(it["offerings"], list):
            for o in it["offerings"]:
                if isinstance(o, dict):
                    norm = _normalize_offering_dict(o)
                    if norm:
                        offerings.append(norm)
            continue

        norm = _normalize_offering_dict(it)
        if norm:
            offerings.append(norm)

    return offerings


def parse_extracted_payload(
    extracted_content: Union[str, bytes, Dict[str, Any], List[Any], None],
) -> ExtractionPayload:
    if extracted_content is None:
        return ExtractionPayload(offerings=[])

    logger.debug(
        "[llm.parse] raw_extracted_preview=%s",
        _short_preview(extracted_content, length=800),
    )

    # 1) Load JSON if string/bytes
    if isinstance(extracted_content, (str, bytes, bytearray)):
        try:
            data = json.loads(extracted_content)
        except Exception as e:
            logger.debug(
                "[llm.parse] json.loads failed: %s | preview=%s",
                e,
                _short_preview(extracted_content),
            )
            return ExtractionPayload(offerings=[])
    else:
        data = extracted_content

    # 1.5) unwrap common wrappers
    if isinstance(data, dict):
        for k in ("extracted_content", "content", "data", "result", "output"):
            if k in data and isinstance(data[k], (str, bytes, bytearray)):
                try:
                    data = json.loads(data[k])
                    break
                except Exception:
                    pass

    # 2) Normalize into payload dict
    payload_dict: Dict[str, Any] = {"offerings": []}
    try:
        if isinstance(data, dict):
            # common alternate key names
            if "offerings" in data and isinstance(data["offerings"], list):
                payload_dict["offerings"] = [
                    norm
                    for o in data["offerings"]
                    if isinstance(o, dict) and (norm := _normalize_offering_dict(o))
                ]
            elif "items" in data and isinstance(data["items"], list):
                payload_dict["offerings"] = _extract_offerings_from_mixed_list(
                    data["items"]
                )
            elif "products" in data or "services" in data:
                items: List[Any] = []
                if isinstance(data.get("products"), list):
                    for p in data["products"]:
                        if isinstance(p, dict):
                            p2 = dict(p)
                            p2.setdefault("type", "product")
                            items.append(p2)
                        elif isinstance(p, str):
                            items.append(
                                {"type": "product", "name": p, "description": p}
                            )
                if isinstance(data.get("services"), list):
                    for s in data["services"]:
                        if isinstance(s, dict):
                            s2 = dict(s)
                            s2.setdefault("type", "service")
                            items.append(s2)
                        elif isinstance(s, str):
                            items.append(
                                {"type": "service", "name": s, "description": s}
                            )
                payload_dict["offerings"] = _extract_offerings_from_mixed_list(items)
            else:
                # single offering dict?
                norm = _normalize_offering_dict(data)
                payload_dict["offerings"] = [norm] if norm else []

        elif isinstance(data, list):
            payload_dict["offerings"] = _extract_offerings_from_mixed_list(data)

        else:
            payload_dict["offerings"] = []

    except Exception as e:
        logger.exception("[llm.parse] normalization error: %s", e)
        return ExtractionPayload(offerings=[])

    # 3) Validate
    try:
        return ExtractionPayload.model_validate(payload_dict)
    except ValidationError as e:
        logger.debug("[llm.parse] pydantic validation failed: %s", e)
        return ExtractionPayload(offerings=[])


# --------------------------------------------------------------------------- #
# Presence result parsing
# --------------------------------------------------------------------------- #


def parse_presence_result(
    extracted_content: Union[str, bytes, int, float, Dict[str, Any], List[Any], None],
    *,
    default: bool = False,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Return: (has_offering_bool, confidence_or_none, preview)
    """
    try:
        preview = _short_preview(extracted_content, length=800)

        if extracted_content is None:
            return (default, None, preview)

        # Normalize bytes/str -> try JSON
        if isinstance(extracted_content, (bytes, bytearray)):
            s = extracted_content.decode(errors="ignore").strip()
            try:
                loaded = json.loads(s)
            except Exception:
                loaded = s
        elif isinstance(extracted_content, str):
            s = extracted_content.strip()
            try:
                loaded = json.loads(s)
            except Exception:
                loaded = s
        else:
            loaded = extracted_content

        def _interpret_scalar(val: Any) -> Optional[int]:
            if isinstance(val, bool):
                return int(val)
            if isinstance(val, int) and val in (0, 1):
                return int(val)
            if isinstance(val, float) and val in (0.0, 1.0):
                return int(val)
            if isinstance(val, str):
                ss = val.strip().lower()
                if ss in {"0", "1"}:
                    return int(ss)
                if ss in {"yes", "true"}:
                    return 1
                if ss in {"no", "false"}:
                    return 0
            return None

        def _interpret_conf(val: Any) -> Optional[float]:
            try:
                if isinstance(val, (int, float)):
                    f = float(val)
                elif isinstance(val, str):
                    f = float(val.strip())
                else:
                    return None
                return f if 0.0 <= f <= 1.0 else None
            except Exception:
                return None

        # scalar
        isc = _interpret_scalar(loaded)
        if isc is not None:
            return (bool(isc), None, preview)

        # dict
        if isinstance(loaded, dict):
            conf = None
            for ck in ("confidence", "score", "prob", "p"):
                if ck in loaded:
                    conf = _interpret_conf(loaded.get(ck))
                    if conf is not None:
                        break

            if "r" in loaded:
                r_val = _interpret_scalar(loaded["r"])
                if r_val is not None:
                    return (bool(r_val), conf, preview)

            for key in (
                "has_offering",
                "presence",
                "present",
                "classification",
                "result",
                "value",
            ):
                if key in loaded:
                    r_val = _interpret_scalar(loaded[key])
                    if r_val is not None:
                        return (bool(r_val), conf, preview)

            return (default, conf, preview)

        # list
        if isinstance(loaded, list) and loaded:
            first = loaded[0]
            isc = _interpret_scalar(first)
            if isc is not None:
                return (bool(isc), None, preview)
            if isinstance(first, dict):
                return parse_presence_result(first, default=default)

        return (default, None, preview)

    except Exception as e:
        logger.exception("[llm.presence] parse error: %s", e)
        return (default, None, None)


# --------------------------------------------------------------------------- #
# Default provider instance
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
    "parse_extracted_payload",
    "parse_presence_result",
    "default_ollama_provider_strategy",
    "DEFAULT_PRESENCE_INSTRUCTION",
    "DEFAULT_FULL_INSTRUCTION",
    "CompatLLMExtractionStrategy",
]
