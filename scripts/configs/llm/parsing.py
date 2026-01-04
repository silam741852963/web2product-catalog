from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import ValidationError

from .schema import (
    Offering,
    ExtractionPayload,
    PresencePayload,
    NAME_MAX,
    DESC_MAX,
    TAG_MAX,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

DEBUG_PREVIEW_CHARS = 900
DEBUG_RAW_CHARS = 7000
EVIDENCE_SNIPPET_MAX_CHARS = 240


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


# --------------------------------------------------------------------------- #
# JSON salvage (Python 3.13 safe)
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

        close_ch = "}" if ch == "{" else "]"
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
            yield s[start:i]
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
# Parsing
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
        return ExtractionPayload(offerings=[])

    data: Any = extracted_content

    if isinstance(extracted_content, (str, bytes, bytearray)):
        s0 = _to_text(extracted_content).strip()
        try:
            data = json.loads(s0)
        except Exception:
            blob = _extract_first_json_blob(s0)
            if blob is None:
                return ExtractionPayload(offerings=[])
            data = json.loads(blob)

    if isinstance(data, dict):
        for k in ("extracted_content", "content", "data", "result", "output"):
            if isinstance(data.get(k), (str, bytes, bytearray)):
                inner = _to_text(data[k]).strip()
                try:
                    data = json.loads(inner)
                    break
                except Exception:
                    blob = _extract_first_json_blob(inner)
                    if blob is None:
                        continue
                    data = json.loads(blob)
                    break

    candidates = list(_collect_offering_candidates(data))

    valid: List[Offering] = []
    seen: Dict[Tuple[str, str], int] = {}

    for c in candidates:
        if not isinstance(c, dict):
            continue
        try:
            cleaned = _sanitize_offering(c)
            off = Offering.model_validate(cleaned)
        except ValidationError:
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
    """
    Returns: (has_offering, confidence, preview)

    Fix: PresencePayload is now the canonical fast-path parser (so the import is used),
    with tolerant fallbacks for messy/legacy output.
    """
    preview = _short_preview(extracted_content, max_chars=DEBUG_PREVIEW_CHARS)

    if extracted_content is None:
        return (default, None, preview)

    loaded: Any = extracted_content
    if isinstance(extracted_content, (str, bytes, bytearray)):
        s = _to_text(extracted_content).strip()
        try:
            loaded = json.loads(s)
        except Exception:
            blob = _extract_first_json_blob(s)
            if blob is not None:
                try:
                    loaded = json.loads(blob)
                except Exception:
                    loaded = s
            else:
                loaded = s

    # ---- Canonical schema fast-path (PresencePayload) ----
    # Accept either {"r": 0/1} or variants where caller merged {"r":..., "confidence":...}
    if isinstance(loaded, dict):
        try:
            pp = PresencePayload.model_validate(loaded)
            conf = None
            for ck in ("confidence", "score", "prob", "p"):
                if ck in loaded:
                    try:
                        f = float(loaded.get(ck))
                        conf = f if 0.0 <= f <= 1.0 else None
                    except Exception:
                        conf = None
                    break
            return (bool(pp.r), conf, preview)
        except ValidationError:
            pass

    # ---- Tolerant fallback parsing ----

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
# Evidence-based culling (kept as-is)
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

    for off in offerings:
        nm = _clean_ws(off.name).lower()
        if nm in _JUNK_NAMES:
            continue

        if off.tags:
            kept.append(off)
            continue

        ok = _has_any_evidence_soft(off, source_text=source_text)
        if ok:
            kept.append(off)
            continue

        desc = _clean_ws(off.description).lower()
        if len(desc) < 40 or desc in _JUNK_NAMES or desc == nm:
            continue

        kept.append(off)

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
# Robust extract caller (kept)
# --------------------------------------------------------------------------- #


def _debug_dump_raw(label: str, raw: Any) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    s = _to_text(raw)
    s = s if len(s) <= DEBUG_RAW_CHARS else s[:DEBUG_RAW_CHARS] + "…(truncated)"
    logger.debug("[%s] raw_dump=%s", label, s)


def _signature_summary(fn: Any) -> str:
    return str(inspect.signature(fn))


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
    kind: str = "full",
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
        logger.debug("[llm.call] extract_signature=%s", _signature_summary(fn))
        logger.debug("[llm.call] strategy_input_format=%s", input_format)

    sig = inspect.signature(fn)
    requires_ix = any(
        p.name.lower() in {"ix", "idx", "index", "chunk_ix", "chunk_index"}
        for p in sig.parameters.values()
    )

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
            raw_results.append(fn(*args, **kwargs))

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
            return {
                "r": 1 if any_true else 0,
                "confidence": best_conf,
                "preview": last_preview,
            }

        merged_payload = ExtractionPayload(offerings=[])
        for r in raw_results:
            p = parse_extracted_payload(r)
            if p.offerings:
                merged_payload.offerings.extend(p.offerings)

        offerings = merged_payload.offerings
        if require_evidence:
            offerings = _filter_offerings_soft(offerings, source_text=text)

        out_payload = ExtractionPayload(offerings=offerings)
        return out_payload.model_dump()

    args, kwargs = _build_call_args(
        fn, url=url, text=text, html=html, ix=0, strategy_input_format=str(input_format)
    )
    raw = fn(*args, **kwargs)

    if kind == "presence":
        return raw

    payload = parse_extracted_payload(raw)
    offerings = payload.offerings
    if require_evidence:
        offerings = _filter_offerings_soft(offerings, source_text=text)

    out_payload = ExtractionPayload(offerings=offerings)
    out_dict = out_payload.model_dump()

    if logger.isEnabledFor(logging.DEBUG):
        _debug_dump_raw("llm.call.full.final", out_dict)

    return out_dict
