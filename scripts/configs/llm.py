from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants / helpers
# --------------------------------------------------------------------------- #

SHORT_MAX = 400
DEBUG_PREVIEW_CHARS = 900
DEBUG_RAW_CHARS = 6000

# Internal (not emitted in schema) – used to validate/cull hallucinations/nav.
EVIDENCE_SNIPPET_MAX_CHARS = 240

ShortStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=SHORT_MAX)
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

    max_words = max(50, int(chunk_token_threshold * word_token_rate))
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


# --------------------------------------------------------------------------- #
# Pydantic schemas (NO evidence field — keep LLM output small)
# --------------------------------------------------------------------------- #


class Offering(BaseModel):
    type: ShortStr = Field(..., description="Must be 'product' or 'service'.")
    name: ShortStr = Field(..., description="Short label for the offering.")
    description: ShortStr = Field(..., description="1–3 sentence grounded summary.")

    @field_validator("type", mode="before")
    @classmethod
    def _norm_type(cls, v: Any) -> str:
        s = _clean_ws(_to_text(v)).lower()
        if s in {"product", "products", "prod", "goods", "good", "brand", "line"}:
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
        if "product" in s:
            return "product"
        if ("service" in s) or ("solution" in s) or ("capabilit" in s):
            return "service"
        return s

    @field_validator("name", mode="before")
    @classmethod
    def _norm_name(cls, v: Any) -> str:
        return _clean_ws(_to_text(v))[:SHORT_MAX]

    @field_validator("description", mode="before")
    @classmethod
    def _norm_desc(cls, v: Any) -> str:
        return _clean_ws(_to_text(v))[:SHORT_MAX]


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
    model: str = "ministral-3:14b"
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
    provider: str
    api_token: Optional[str] = None
    base_url: Optional[str] = None

    def build_config(self) -> LLMConfig:
        if not self.provider:
            raise ValueError(
                "RemoteAPIProviderStrategy requires non-empty provider string."
            )
        cfg = LLMConfig(
            provider=self.provider, api_token=self.api_token, base_url=self.base_url
        )
        logger.debug("[llm] provider=%s base_url=%s", self.provider, self.base_url)
        return cfg


# --------------------------------------------------------------------------- #
# Default instructions (NO evidence)
# --------------------------------------------------------------------------- #

DEFAULT_PRESENCE_INSTRUCTION = (
    "Classify if the page MAIN CONTENT includes any offerings (products/brands/product lines or services/solutions).\n"
    "Ignore cookie/privacy banners, accessibility boilerplate, navigation/menus, and legal footers.\n"
    'Return ONLY JSON: {"r":1} or {"r":0}. No other keys.'
)

DEFAULT_FULL_INSTRUCTION = (
    "Extract the company's offerings from the page MAIN CONTENT.\n"
    "Ignore cookie/privacy banners, accessibility boilerplate, navigation/menus, and legal footers.\n"
    "Include products/brands/product lines and services/solutions that the company offers.\n"
    "Do NOT invent offerings. Use names/headings used on the page when available.\n"
    "Merge duplicates; keep descriptions concise and factual.\n"
    "Return ONLY valid JSON matching the provided schema."
)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


@dataclass
class LLMExtractionFactory:
    provider_strategy: LLMProviderStrategy
    default_full_instruction: str = DEFAULT_FULL_INSTRUCTION
    default_presence_instruction: str = DEFAULT_PRESENCE_INSTRUCTION

    default_chunk_token_threshold: int = 1400
    default_overlap_rate: float = 0.08
    default_apply_chunking: bool = True
    default_input_format: str = "fit_markdown"

    # With evidence removed, outputs are smaller; you can reduce max_tokens safely.
    default_schema_temperature: float = 0.25
    default_presence_temperature: float = 0.0
    default_schema_max_tokens: int = 900
    default_presence_max_tokens: int = 300

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
# Parsing (robust, tolerant) – schema has no evidence
# --------------------------------------------------------------------------- #


def _looks_like_offering_dict(d: Dict[str, Any]) -> bool:
    name = _clean_ws(_to_text(d.get("name") or d.get("title") or d.get("label")))
    desc = _clean_ws(
        _to_text(d.get("description") or d.get("summary") or d.get("details"))
    )
    typ = _clean_ws(
        _to_text(d.get("type") or d.get("kind") or d.get("category"))
    ).lower()
    return bool(name) and (bool(desc) or bool(typ))


def _guess_type_from_text(name: str, desc: str) -> str:
    s = (name + " " + desc).lower()
    svc_hints = (
        "service",
        "solution",
        "capabilit",
        "consult",
        "platform",
        "subscription",
        "manufactur",
        "processing",
        "sourcing",
        "installation",
    )
    return "service" if any(h in s for h in svc_hints) else "product"


def _sanitize_offering(d: Dict[str, Any]) -> Dict[str, Any]:
    name = _clean_ws(_to_text(d.get("name") or d.get("title") or d.get("label")))[
        :SHORT_MAX
    ]
    desc = _clean_ws(
        _to_text(d.get("description") or d.get("summary") or d.get("details"))
    )[:SHORT_MAX]
    typ = _clean_ws(
        _to_text(d.get("type") or d.get("kind") or d.get("category"))
    ).lower()

    if typ in {"products", "product", "prod", "brand", "line"}:
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
        typ = _guess_type_from_text(name, desc)

    if not desc:
        desc = name

    return {"type": typ, "name": name, "description": desc}


def _collect_offering_candidates(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Walk common shapes:
      - {"offerings":[...]}
      - {"items":[...]}
      - {"products":[...], "services":[...]}
      - [ ... ]
      - single offering dict
    """
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
                    yield {"type": "product", "name": p, "description": p}
        if isinstance(obj.get("services"), list):
            for s in obj["services"]:
                if isinstance(s, dict):
                    s2 = dict(s)
                    s2.setdefault("type", "service")
                    yield s2
                elif isinstance(s, str):
                    yield {"type": "service", "name": s, "description": s}

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

    # Load JSON if needed
    data: Any = extracted_content
    if isinstance(extracted_content, (str, bytes, bytearray)):
        s = _to_text(extracted_content).strip()
        try:
            data = json.loads(s)
        except Exception as e:
            logger.debug(
                "[llm.parse] json.loads failed err=%s preview=%s", e, raw_preview
            )
            return ExtractionPayload(offerings=[])

    # Unwrap common wrappers containing JSON strings
    if isinstance(data, dict):
        for k in ("extracted_content", "content", "data", "result", "output"):
            if isinstance(data.get(k), (str, bytes, bytearray)):
                try:
                    data = json.loads(_to_text(data[k]))
                    logger.debug(
                        "[llm.parse] unwrapped key=%s -> %s", k, type(data).__name__
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
            best_desc = (
                off.description
                if len(off.description) > len(cur.description)
                else cur.description
            )
            valid[j] = Offering(type=cur.type, name=cur.name, description=best_desc)
        else:
            seen[key] = len(valid)
            valid.append(off)

    return ExtractionPayload(offerings=valid)


# --------------------------------------------------------------------------- #
# Presence parsing
# --------------------------------------------------------------------------- #


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
# Evidence-based culling (drop offerings with no evidence in MAIN-ish content)
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

_TRADEMARK_CHARS_RE = re.compile(r"[®™©]+")


def _strip_trademarks(s: str) -> str:
    return _TRADEMARK_CHARS_RE.sub("", s or "").strip()


def _mainish_text(text: str) -> str:
    """
    Heuristic: use content from the first Markdown H1 ("# ") onward.
    This intentionally removes cookie banners + global nav that usually precede H1.
    """
    if not text:
        return ""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("# "):  # first H1
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
    # case-insensitive search
    m = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
    if not m:
        return None
    start = max(0, m.start() - (EVIDENCE_SNIPPET_MAX_CHARS // 2))
    end = min(len(haystack), m.end() + (EVIDENCE_SNIPPET_MAX_CHARS // 2))
    snip = _clean_ws(haystack[start:end])
    return snip[:EVIDENCE_SNIPPET_MAX_CHARS] if snip else None


def _has_any_evidence(offer: Offering, source_text: str) -> Tuple[bool, Optional[str]]:
    """
    Return (has_evidence, snippet_preview).
    Evidence rule (strict-ish):
      - prefer finding the offering name (or name without trademarks) in main-ish text
      - else require >=2 keyword tokens from the name present in main-ish text
      - else require >=3 keyword tokens from the description present in main-ish text
    """
    t = _mainish_text(source_text)
    if not t:
        return (False, None)

    name = _clean_ws(offer.name)
    name2 = _strip_trademarks(name)

    # 1) exact-ish name match
    snip = _find_snippet(t, name) or (
        _find_snippet(t, name2) if name2 and name2 != name else None
    )
    if snip:
        return (True, snip)

    # 2) keyword matches from name (>=2)
    name_keys = _keyword_tokens(name2 or name)
    if name_keys:
        hits = sum(
            1
            for k in set(name_keys)
            if re.search(rf"\b{re.escape(k)}\b", t, flags=re.IGNORECASE)
        )
        if hits >= 2 or (hits >= 1 and len(name_keys) == 1):
            # produce a snippet from the first hit keyword
            for k in name_keys:
                snip2 = _find_snippet(t, k)
                if snip2:
                    return (True, snip2)
            return (True, None)

    # 3) keyword matches from description (>=3)
    desc_keys = _keyword_tokens(offer.description)
    if desc_keys:
        hits = sum(
            1
            for k in set(desc_keys)
            if re.search(rf"\b{re.escape(k)}\b", t, flags=re.IGNORECASE)
        )
        if hits >= 3:
            for k in desc_keys:
                snip3 = _find_snippet(t, k)
                if snip3:
                    return (True, snip3)
            return (True, None)

    return (False, None)


def _filter_offerings_require_evidence(
    offerings: List[Offering], *, source_text: str
) -> List[Offering]:
    kept: List[Offering] = []
    dropped = 0

    for off in offerings:
        ok, snip = _has_any_evidence(off, source_text)
        if ok:
            kept.append(off)
        else:
            dropped += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[llm.filter] drop_no_evidence type=%s name=%s desc=%s",
                    off.type,
                    off.name,
                    _short_preview(off.description, max_chars=220),
                )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[llm.filter] kept=%d dropped=%d", len(kept), dropped)

    # Dedup again after filtering (same rule as parse)
    out: List[Offering] = []
    seen: Dict[Tuple[str, str], int] = {}
    for off in kept:
        key = (off.type, off.name.lower())
        if key in seen:
            j = seen[key]
            cur = out[j]
            best_desc = (
                off.description
                if len(off.description) > len(cur.description)
                else cur.description
            )
            out[j] = Offering(type=cur.type, name=cur.name, description=best_desc)
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
    """
    Build positional args + kwargs in signature order.
    - url-like param -> url
    - ix/index-like param -> ix
    - markdown/text/content-like param -> text
    - html-like param -> html (fallback to text if html empty)
    """
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
) -> Any:
    """
    Offline-safe extraction wrapper.

    Supports Crawl4AI versions where:
      - extract(url, text)
      - extract(url, markdown, html)
      - extract(url, html, markdown)
      - extract(url, ix, html)   (per-chunk API)

    IMPORTANT BEHAVIOR CHANGE:
      - For kind="full": returns a *dict* already filtered to drop offerings with no evidence in main-ish content,
        and the dict does NOT contain an evidence field.
      - For kind="presence": returns raw (or merged {"r":0/1} for per-chunk variants).
    """
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

    # Detect if extract requires ix (per-chunk API)
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

    # ------------------------------------------------------------------ #
    # Per-chunk API path
    # ------------------------------------------------------------------ #
    if requires_ix:
        apply_chunking = bool(getattr(strategy, "apply_chunking", True))
        chunk_threshold = int(getattr(strategy, "chunk_token_threshold", 1400))
        overlap_rate = float(getattr(strategy, "overlap_rate", 0.08))
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

        # Evidence filter + final dict
        filtered = _filter_offerings_require_evidence(
            merged_payload.offerings, source_text=text
        )
        out_payload = ExtractionPayload(offerings=filtered)
        out_dict = out_payload.model_dump()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[llm.call] merged_offerings=%d filtered_offerings=%d",
                len(merged_payload.offerings),
                len(filtered),
            )
            _debug_dump_raw("llm.call.full.filtered", out_dict)

        return out_dict

    # ------------------------------------------------------------------ #
    # Single-call API path
    # ------------------------------------------------------------------ #
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

    # For presence we keep raw; for full we parse+filter+return dict (small + safe).
    if kind == "presence":
        return raw

    payload = parse_extracted_payload(raw)
    filtered = _filter_offerings_require_evidence(payload.offerings, source_text=text)
    out_payload = ExtractionPayload(offerings=filtered)
    out_dict = out_payload.model_dump()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[llm.call] single_offerings=%d filtered_offerings=%d",
            len(payload.offerings),
            len(filtered),
        )
        _debug_dump_raw("llm.call.full.filtered", out_dict)

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
    "parse_extracted_payload",
    "parse_presence_result",
    "call_llm_extract",
    "default_ollama_provider_strategy",
    "DEFAULT_PRESENCE_INSTRUCTION",
    "DEFAULT_FULL_INSTRUCTION",
]
