"""
Shared utilities:
- URL helpers (normalization, domain checks)
- Robust retry decorators (sync & async) using tenacity
- HTML -> Markdown conversion and cleaning
- Text chunking for LLM token budgets
- File I/O helpers (atomic write)
- Simple robots.txt guard (opt-in)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable, Awaitable, Optional
from urllib.parse import urlparse, urlunparse
import json

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# ---- Optional BeautifulSoup (used by prune_html_for_markdown) ----
try:  # pragma: no cover - availability checked in tests
    from bs4 import BeautifulSoup, Comment  # type: ignore
    _BS4_AVAILABLE = True
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore
    Comment = None        # type: ignore
    _BS4_AVAILABLE = False

try:
    # markdownify is lightweight and preserves structure better than html2text for our use
    from markdownify import markdownify as _markdownify
except Exception:  # pragma: no cover
    _markdownify = None  # fail gracefully; caller can check


logger = logging.getLogger(__name__)


# ---------- URL helpers ----------

def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistency:
    - if input looks like a bare hostname (e.g., 'example.com'), coerce to http://example.com/
    - lowercase scheme/host
    - strip default ports
    - remove fragment
    - ensure trailing slash on empty path
    """
    url = url.strip()
    parsed = urlparse(url)

    # If no scheme/netloc, but path looks like a hostname (e.g., 'example.com' or 'example.com:8080')
    if not parsed.scheme and not parsed.netloc and parsed.path:
        m = re.match(r"^(?P<host>[A-Za-z0-9.-]+)(?::(?P<port>\d+))?$", parsed.path)
        if m and "." in m.group("host"):  # require at least one dot to look like a domain
            host = m.group("host").lower()
            port = m.group("port")
            netloc = f"{host}:{port}" if port else host
            return urlunparse(("http", netloc, "/", "", "", ""))

    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname.lower() if parsed.hostname else ""
    netloc = host
    if parsed.port:
        # keep non-default ports only
        if not ((scheme == "http" and parsed.port == 80) or (scheme == "https" and parsed.port == 443)):
            netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"
    query = parsed.query

    return urlunparse((scheme, netloc, path, "", query, ""))


def is_same_domain(a: str, b: str) -> bool:
    """Return True if URLs share the same (exact) hostname."""
    pa, pb = urlparse(a), urlparse(b)
    return (pa.hostname or "").lower() == (pb.hostname or "").lower()


def is_http_url(url: str) -> bool:
    s = urlparse(url).scheme.lower()
    return s in {"http", "https"}


def slugify(text: str, max_len: int = 80) -> str:
    """
    Turn arbitrary text into a filesystem-friendly slug.
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_.]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-._")
    return text or "untitled"


# ---------- Retry decorators ----------

class TransientHTTPError(Exception):
    """Marker for retryable transient HTTP/Net errors (429/5xx/timeouts)."""


def retry_sync(max_attempts: int, initial_delay_ms: int, max_delay_ms: int, jitter_ms: int):
    """
    Decorator factory for synchronous retry with exponential backoff + jitter.
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            initial=initial_delay_ms / 1000.0,
            max=max_delay_ms / 1000.0,
            jitter=jitter_ms / 1000.0,
        ),
        retry=retry_if_exception_type((TransientHTTPError, IOError, TimeoutError)),
    )


def retry_async(max_attempts: int, initial_delay_ms: int, max_delay_ms: int, jitter_ms: int):
    """
    Decorator factory for async retry with exponential backoff + jitter.
    Usage:
        @retry_async(4, 500, 8000, 300)
        async def fetch(...): ...
    """
    def _decorator(fn: Callable[..., Awaitable]):
        @retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(
                initial=initial_delay_ms / 1000.0,
                max=max_delay_ms / 1000.0,
                jitter=jitter_ms / 1000.0,
            ),
            retry=retry_if_exception_type((TransientHTTPError, IOError, TimeoutError)),
        )
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)

        return wrapper
    return _decorator


# ---------- HTML pruning (before Markdown) ----------

# Heuristics for stripping site chrome, compliance, and noisy blocks
_CHROME_ID_CLASS = re.compile(
    r"(?:^|[-_])(header|footer|nav|navbar|menu|mega|breadcrumb|aside|sidebar|"
    r"social|share|subscribe|newsletter|signup|modal|popup|overlay|cookie|gdpr|consent|"
    r"banner|intercom|chat|widget|toolbar|sticky|floating|drawer|offcanvas|"
    r"comments|related|pagination|pager|sitemap|language|locale|switcher|"
    r"searchbox|search-bar|promo|advert|ad-)\b",
    re.I,
)

_COMPLIANCE_TEXT = re.compile(
    r"(cookie|gdpr|consent|we use cookies|privacy\s+policy|terms\s+of\s+use|"
    r"copyright|all rights reserved|trademark|forward-looking statements)",
    re.I,
)

_CTA_TEXT = re.compile(
    r"(request (a )?demo|contact sales|start (your )?trial|get started|"
    r"join (our )?newsletter|subscribe)", re.I
)


def prune_html_for_markdown(html: str, *, keep_first_cta: bool = True) -> str:
    """
    Remove boilerplate & low-signal sections before HTML->Markdown.
    Conservative: tries not to nuke product content.

    If BeautifulSoup is not available, returns the original HTML.
    """
    if not html or not _BS4_AVAILABLE:
        return html

    soup = BeautifulSoup(html, "lxml")

    # 0) strip comments and obvious non-content tags
    for c in soup(text=lambda t: isinstance(t, Comment)):  # type: ignore
        c.extract()
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    # 1) remove elements by role/landmark (but keep <main>, <article>)
    for tag in soup.select("[role='navigation'], [role='banner'], [role='contentinfo'], nav, aside, header, footer"):
        tag.decompose()

    # 2) remove elements by id/class keywords
    for el in soup.find_all(attrs={"class": True}):
        classes = " ".join(el.get("class") or [])
        if _CHROME_ID_CLASS.search(classes):
            el.decompose()
    for el in soup.find_all(attrs={"id": True}):
        if _CHROME_ID_CLASS.search(el.get("id") or ""):
            el.decompose()

    # 3) remove compliance/marketing blocks by visible text
    for el in list(soup.find_all(True)):
        # quick skip for main/article/section with lots of text
        if el.name in ("main", "article", "section"):
            continue
        txt = (el.get_text(" ", strip=True) or "")[:500].lower()
        if not txt:
            continue
        if _COMPLIANCE_TEXT.search(txt):
            el.decompose()
            continue
        # CTA blocks (keep the first one if asked)
        if _CTA_TEXT.search(txt):
            if keep_first_cta:
                keep_first_cta = False
            else:
                el.decompose()

    # 4) remove empty containers
    for el in list(soup.find_all(True)):
        if el.name in ("main", "article", "section"):  # keep
            continue
        txt = el.get_text("", strip=True)
        if not txt:
            el.decompose()

    # 5) Fallback: if page lacks a <main>, prefer the largest text block
    if not soup.find("main"):
        candidates = sorted(
            (el for el in soup.find_all(["article", "section", "div"])),
            key=lambda e: len((e.get_text(" ", strip=True) or "")),
            reverse=True,
        )
        if candidates:
            keep = candidates[0]
            # remove siblings around the largest block to reduce noise
            parent = keep.parent
            if parent:
                for sib in list(parent.children):
                    if getattr(sib, "name", None) and sib is not keep:
                        try:
                            sib.decompose()
                        except Exception:
                            pass

    # 6) Clean anchor junk
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            a.attrs.pop("href", None)

    return str(soup)


# ---------- HTML → Markdown ----------

@dataclass
class MarkdownOptions:
    strip: bool = True                # trim leading/trailing whitespace
    heading_style: str = "ATX"        # for markdownify (if supported)
    code_language: Optional[str] = None  # not used by markdownify; reserved for future


def html_to_markdown(html: str, opts: MarkdownOptions = MarkdownOptions()) -> str:
    """
    Convert raw HTML to Markdown. Falls back to plain text if markdownify is unavailable.
    """
    if not html:
        return ""
    if _markdownify is None:
        logger.warning("markdownify not installed; returning plain text fallback")
        # naive strip tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    md = _markdownify(html, heading_style=opts.heading_style.lower())
    if opts.strip:
        md = md.strip()
    return md


def clean_markdown(md: str, remove_boilerplate: bool = True) -> str:
    """
    Light cleanup pass over Markdown:
    - collapse excessive blank lines
    - remove super-short nav/footer lines (heuristic)
    """
    if not md:
        return md
    lines = [ln.rstrip() for ln in md.splitlines()]
    if remove_boilerplate:
        filtered = []
        for ln in lines:
            # drop obvious crumbs/footers
            if len(ln) <= 2:
                continue
            if ln.lower().startswith(("cookie", "privacy policy", "terms of", "login", "subscribe")):
                continue
            filtered.append(ln)
        lines = filtered
    # collapse consecutive blanks
    cleaned: list[str] = []
    blank = False
    for ln in lines:
        if ln.strip() == "":
            if not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(ln)
            blank = False
    return "\n".join(cleaned).strip()


# ---------- LLM chunking helpers ----------

def chunk_text(text: str, max_chars: int) -> Iterable[str]:
    """
    Greedy chunker by characters (safe pre-tokenization approximation).
    Keeps paragraph boundaries where possible.
    """
    if not text:
        return []
    paragraphs = text.split("\n\n")
    buf: list[str] = []
    length = 0
    for para in paragraphs:
        # +2 for the double newline rejoin
        if length + len(para) + (2 if buf else 0) <= max_chars:
            buf.append(para)
            length += len(para) + (2 if buf else 0)
        else:
            if buf:
                yield "\n\n".join(buf)
            # if paragraph itself is huge, hard-split it
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    yield para[i:i + max_chars]
                buf, length = [], 0
            else:
                buf, length = [para], len(para)
    if buf:
        yield "\n\n".join(buf)


# ---------- File I/O ----------

def atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """
    Atomic write to avoid torn writes on crash (write to temp, then replace).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(data, encoding=encoding)
    tmp.replace(path)


def append_jsonl(path: Path, json_line: str, encoding: str = "utf-8") -> None:
    """
    Append a single JSON line to a .jsonl file (creates file if missing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=encoding) as f:
        f.write(json_line.rstrip() + "\n")


def save_markdown(base_dir: Path, host: str, url_path: str, url: str, md: str) -> Path:
    """
    Save markdown in a deterministic path mirroring HTML cache:
    data/markdown/{host}/{slug}-{sha}.md
    """
    from hashlib import sha1
    host = (host or "unknown-host").lower()
    if host.startswith("www."):
        host = host[4:]  # drop leading www.

    path = url_path or "/"
    if path.endswith("/"):
        path += "index"
    slug = slugify(path, max_len=80)
    h = sha1(url.encode("utf-8")).hexdigest()[:10]
    fname = f"{slug}-{h}.md"
    out_path = base_dir / host / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, md)
    return out_path


def safe_json_loads(s: str) -> Optional[dict]:
    """
    Try strict json.loads first; if it fails, attempt to extract the first
    top-level JSON object by scanning braces (robust to noisy wrappers).
    Returns None if nothing valid is found.
    """
    if not isinstance(s, str):
        return None
    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = s.find("{", start + 1)

    return None


# ---------- Robots (simple, optional) ----------

def should_crawl_url(respect_robots: bool, robots_txt_allowed: Optional[bool]) -> bool:
    """
    Gate to decide whether to crawl a URL. If respect_robots is False, always allow.
    Callers can integrate a proper robots.txt parser later and pass robots_txt_allowed=True/False.
    """
    if not respect_robots:
        return True
    return bool(robots_txt_allowed)