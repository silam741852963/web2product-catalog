# scraper/utils.py
from __future__ import annotations

import json
import logging
import os
import re
import string
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterable, Callable, Awaitable, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urljoin

import httpx
import tldextract
from bs4 import BeautifulSoup, Comment
from bs4.element import Tag
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

try:
    from markdownify import markdownify as _markdownify
except Exception as e:  # pragma: no cover
    _markdownify = None
    logging.getLogger(__name__).warning(
        "markdownify import failed (%s). Falling back to plain-text stripping. "
        "Ensure 'markdownify' and 'beautifulsoup4' are installed in this interpreter.",
        repr(e),
    )

logger = logging.getLogger(__name__)

# ========== Environment & Logging helpers (moved from config.py) ==========

def getenv_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() else default

def getenv_int(name: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if min_val is not None:
        val = max(min_val, val)
    if max_val is not None:
        val = min(max_val, val)
    return val

def getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def init_logging(log_path: Path, level: int = logging.INFO) -> None:
    """
    Simple file+console logger. Call once early (e.g., in run_scraper.py) with cfg.log_file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(levelname)s %(asctime)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

# ========== Exceptions & HTTP status mapping ==========

class TransientHTTPError(Exception):
    """Retryable transient HTTP/Net error (429/5xx/timeouts)."""

class NonRetryableHTTPError(Exception):
    """Non-retryable client error (e.g., 404) or policy block."""

class NeedsBrowser(Exception):
    """Signals that static fetch is insufficient and a browser is required."""

def http_status_to_exc(status: Optional[int]) -> Optional[Exception]:
    if status is None:
        return None
    if status == 404:
        return NonRetryableHTTPError("404 Not Found")
    if status >= 400:
        return TransientHTTPError(f"HTTP {status}")
    return None

# ========== URL & domain helpers ==========

_JS_APP_SIGNATURES = (
    "__NEXT_DATA__", 'id="__next"', "data-reactroot", "reactdom",
    "window.__NUXT__", 'id="app"', "ng-app", "sapper", "astro-island",
    "vite", "webpackjsonp",
)

# strong allow-hints: if the path has these, we DO NOT block
_PRODUCT_HINTS = (
    "product", "products", "solution", "solutions",
    "service", "services", "catalog", "portfolio",
    "platform", "features", "pricing", "specs", "datasheet",
    "store", "shop",
)

# Obvious non-product path fragments
_NON_PRODUCT_PATH_PARTS = (
    "privacy", "cookies", "cookie-policy", "cookie-preferences", "cookie-settings",
    "terms", "legal", "disclaimer", "compliance", "accessibility",
    "policy", "policies", "charter", "bylaws", "guidelines", "ethics",
    "anti-bribery", "sanctions", "human-rights", "code-of-conduct",
    "governance", "board", "leadership",
    "investor", "investors", "ir", "sec", "earnings", "transcript",
    "prospectus", "summaryprospectus", "factcard", "sai", "holdings",
    "annual-report", "semi-annual", "shareholder", "premiumdiscount",
    "newsroom", "press", "press-release", "press-releases", "media", "news",
    "information-request", "request-information", "request-info", "request",
    "contact", "contact-us", "form", "forms", "subscribe", "newsletter",
    "whitepaper", "whitepapers", "case-study", "case-studies", "ebook", "e-book",
    "resources", "resource-center", "brochure",
    "login", "log-in", "signin", "sign-in", "signup", "sign-up", "register",
    "account", "my-account", "profile", "auth", "sso", "oauth",
    "cart", "checkout", "wishlist", "compare",
    "sitemap", "search", "search-results", "results", "site-search",
    "preferences", "settings", "help", "support", "faq",
    "wp-admin", "wp-login", "wp-json", "xmlrpc.php", "feed", "rss", "atom",
    "translate", "lang", "locale",
    "careers", "jobs", "recruit", "hiring",
    "wp-content/uploads", "media-library", "document-library", "asset-library",
    "acquire/tel",
)

_NON_PRODUCT_QUERY_KEYS = (
    "gclid", "fbclid", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "session", "sessionid", "token", "auth", "sso", "oauth", "attachment", "attachment_id",
    "download", "redirect", "ref", "trk",
)

_NON_PRODUCT_HOST_TOKENS = (
    "investor", "investors", "ir", "press", "news", "careers", "jobs",
    "support", "help", "auth", "login", "accounts",
)

_TEL_PATH_RE = re.compile(r"(?:^|/)(?:tel|fax)\d{6,}(?:/|$)", re.IGNORECASE)

def looks_non_product_url(u: str) -> bool:
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        path = (p.path or "").lower()

        if path in ("", "/"):
            return False
        if any(tok in path for tok in _PRODUCT_HINTS):
            return False
        if _TEL_PATH_RE.search(path or ""):
            return True
        if any(hint in host.split(".") for hint in _NON_PRODUCT_HOST_TOKENS):
            return True
        for frag in _NON_PRODUCT_PATH_PARTS:
            if frag in path:
                return True
        if p.query:
            q = dict(parse_qsl(p.query, keep_blank_values=True))
            for k in _NON_PRODUCT_QUERY_KEYS:
                if k in q:
                    return True
        return False
    except Exception:
        return False

def get_base_domain(host: str) -> str:
    """
    Return registrable domain (eTLD+1); fall back to host if unknown.
    """
    if not host:
        return "unknown-host"
    host = host.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    try:
        ext = tldextract.extract(host)
        td = getattr(ext, "top_domain_under_public_suffix", None)
        if td:
            return td
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
    except Exception:
        pass
    return host

def normalize_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)

    if not parsed.scheme and not parsed.netloc and parsed.path:
        m = re.match(r"^(?P<host>[A-Za-z0-9.-]+)(?::(?P<port>\d+))?$", parsed.path)
        if m and "." in m.group("host"):
            host = m.group("host").lower()
            port = m.group("port")
            netloc = f"{host}:{port}" if port else host
            return urlunparse(("http", netloc, "/", "", "", ""))

    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname.lower() if parsed.hostname else ""
    netloc = host
    if parsed.port and not ((scheme == "http" and parsed.port == 80) or (scheme == "https" and parsed.port == 443)):
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"
    query = parsed.query
    return urlunparse((scheme, netloc, path, "", query, ""))

def is_same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.hostname or "").lower() == (pb.hostname or "").lower()

def is_http_url(url: str) -> bool:
    s = urlparse(url).scheme.lower()
    return s in {"http", "https"}

def same_site(url_a: str, url_b: str, allow_subdomains: bool) -> bool:
    if not allow_subdomains:
        return is_same_domain(url_a, url_b)
    ha = get_base_domain((urlparse(url_a).hostname or "").lower())
    hb = get_base_domain((urlparse(url_b).hostname or "").lower())
    return ha == hb

# ========== Retry decorators ==========

def retry_sync(max_attempts: int, initial_delay_ms: int, max_delay_ms: int, jitter_ms: int):
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

# ========== Static parsing / HTML utils ==========

def visible_text_len(html: str) -> int:
    if not html:
        return 0
    try:
        soup = BeautifulSoup(html, "lxml")
        return len((soup.get_text(" ", strip=True) or ""))
    except Exception:
        return 0

def looks_like_js_app(html: str, threshold: int) -> bool:
    if not html:
        return True
    lower = html.lower()
    if any(sig in lower for sig in _JS_APP_SIGNATURES) and visible_text_len(html) < threshold:
        return True
    if "enable javascript" in lower or "requires javascript" in lower:
        return True
    return False

def extract_links_static(html: str, base_url: str) -> list[str]:
    links: list[str] = []
    seen = set()
    try:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            if not href:
                continue
            abs_u = urljoin(base_url, href)
            abs_u = re.sub(r"#.*$", "", abs_u)
            if abs_u not in seen:
                seen.add(abs_u)
                links.append(abs_u)
        return links
    except Exception:
        return []

def extract_title_static(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, "lxml")
        t = soup.title.string if soup.title else None
        return (t or "").strip() or None
    except Exception:
        return None

# ========== HTML pruning (before Markdown) ==========

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

def _safe_classes(el) -> str:
    try:
        val = el.get("class")
        if isinstance(val, (list, tuple)):
            return " ".join(str(x) for x in val if x)
        if isinstance(val, str):
            return val
    except Exception:
        pass
    return ""

def _safe_id(el) -> str:
    try:
        v = el.get("id")
        return v if isinstance(v, str) else ""
    except Exception:
        return ""

def prune_html_for_markdown(html: str, *, keep_first_cta: bool = True) -> str:
    if not html:
        return html

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    try:
        for c in list(soup(string=lambda t: isinstance(t, Comment))):  # type: ignore
            try:
                c.extract()
            except Exception:
                pass
    except Exception:
        pass

    for tag in list(soup.find_all(["script", "style", "noscript", "svg", "canvas"])):
        if isinstance(tag, Tag):
            try:
                tag.decompose()
            except Exception:
                pass

    for tag in list(soup.select("[role='navigation'], [role='banner'], [role='contentinfo'], nav, aside, header, footer")):
        if isinstance(tag, Tag):
            try:
                tag.decompose()
            except Exception:
                pass

    for el in list(soup.find_all(attrs={"class": True})):
        if not isinstance(el, Tag):
            continue
        if _CHROME_ID_CLASS.search(_safe_classes(el)):
            try:
                el.decompose()
            except Exception:
                pass

    for el in list(soup.find_all(attrs={"id": True})):
        if not isinstance(el, Tag):
            continue
        if _CHROME_ID_CLASS.search(_safe_id(el)):
            try:
                el.decompose()
            except Exception:
                pass

    for el in list(soup.find_all(True)):
        if not isinstance(el, Tag):
            continue
        if el.name in ("main", "article", "section"):
            continue
        try:
            txt = (el.get_text(" ", strip=True) or "")[:500].lower()
        except Exception:
            continue
        if not txt:
            continue
        if _COMPLIANCE_TEXT.search(txt):
            try:
                el.decompose()
            except Exception:
                pass
            continue
        if _CTA_TEXT.search(txt):
            if keep_first_cta:
                keep_first_cta = False
            else:
                try:
                    el.decompose()
                except Exception:
                    pass

    for el in list(soup.find_all(True)):
        if not isinstance(el, Tag):
            continue
        if el.name in ("main", "article", "section"):
            continue
        try:
            txt = el.get_text("", strip=True)
        except Exception:
            txt = ""
        if not txt:
            try:
                el.decompose()
            except Exception:
                pass

    try:
        if not soup.find("main"):
            candidates = sorted(
                (el for el in list(soup.find_all(["article", "section", "div"])) if isinstance(el, Tag)),
                key=lambda e: len((e.get_text(" ", strip=True) or "")),
                reverse=True,
            )
            if candidates:
                keep = candidates[0]
                parent = keep.parent
                if parent:
                    for sib in list(getattr(parent, "children", []) or []):
                        if isinstance(sib, Tag) and sib is not keep:
                            try:
                                sib.decompose()
                            except Exception:
                                pass
    except Exception:
        pass

    for a in list(soup.find_all("a")):
        if not isinstance(a, Tag):
            continue
        try:
            href = (a.get("href") or "").strip()
        except Exception:
            href = ""
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            try:
                a.attrs.pop("href", None)
            except Exception:
                pass

    return str(soup)

# ========== HTML → Markdown ==========

@dataclass
class MarkdownOptions:
    strip: bool = True
    heading_style: str = "ATX"
    code_language: Optional[str] = None

def html_to_markdown(html: str, opts: MarkdownOptions = MarkdownOptions()) -> str:
    if not html:
        return ""
    if _markdownify is None:
        logger.warning("markdownify not installed; returning plain text fallback")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    md = _markdownify(html, heading_style=opts.heading_style.lower())
    if opts.strip:
        md = md.strip()
    return md

def clean_markdown(md: str, remove_boilerplate: bool = True) -> str:
    if not md:
        return md
    lines = [ln.rstrip() for ln in md.splitlines()]
    if remove_boilerplate:
        filtered = []
        for ln in lines:
            if len(ln) <= 2:
                continue
            if ln.lower().startswith(("cookie", "privacy policy", "terms of", "login", "subscribe")):
                continue
            filtered.append(ln)
        lines = filtered
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

# ========== LLM chunking helpers ==========

def chunk_text(text: str, max_chars: int) -> Iterable[str]:
    if not text:
        return []
    paragraphs = text.split("\n\n")
    buf: list[str] = []
    length = 0
    for para in paragraphs:
        add_len = len(para) + (2 if buf else 0)
        if length + add_len <= max_chars:
            buf.append(para)
            length += add_len
        else:
            if buf:
                yield "\n\n".join(buf)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    yield para[i:i + max_chars]
                buf, length = [], 0
            else:
                buf, length = [para], len(para)
    if buf:
        yield "\n\n".join(buf)

# ========== File I/O ==========

def atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(data, encoding=encoding)
    tmp.replace(path)

def append_jsonl(path: Path, json_line: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=encoding) as f:
        f.write(json_line.rstrip() + "\n")

def save_markdown(base_dir: Path, host: str, url_path: str, url: str, md: str) -> Path:
    """
    Save markdown under the registrable domain folder (eTLD+1),
    collapsing subdomains (investors.example.com -> example.com).
    """
    base_host = get_base_domain(host or "unknown-host")
    path = url_path or "/"
    if path.endswith("/"):
        path += "index"
    slug = slugify(path, max_len=80)
    h = sha1(url.encode("utf-8")).hexdigest()[:10]
    fname = f"{slug}-{h}.md"
    out_path = base_dir / base_host / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, md)
    return out_path

def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_.]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-._")
    return text or "untitled"

def safe_json_loads(s: str) -> Optional[dict]:
    if not isinstance(s, str):
        return None
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

def markdown_quality(md: str) -> dict:
    if not isinstance(md, str):
        return {"chars": 0, "words": 0, "alpha": 0, "lines": 0, "uniq_words": 0}
    chars = len(md)
    words_list = [w.strip(string.punctuation) for w in md.split()]
    words = len(words_list)
    uniq_words = len(set(w.lower() for w in words_list if w))
    alpha = sum(ch.isalnum() for ch in md)
    lines = md.count("\n") + 1
    return {"chars": chars, "words": words, "alpha": alpha, "lines": lines, "uniq_words": uniq_words}

def is_meaningful_markdown(md: str,
                           *,
                           min_chars: int = 400,
                           min_words: int = 80,
                           min_uniq_words: int = 50,
                           min_lines: int = 8) -> bool:
    q = markdown_quality(md)
    if q["chars"] < min_chars:
        return False
    if q["words"] < min_words:
        return False
    if q["uniq_words"] < min_uniq_words:
        return False
    if q["lines"] < min_lines:
        return False
    return True

# ========== Robots gate ==========

def should_crawl_url(respect_robots: bool, robots_txt_allowed: Optional[bool]) -> bool:
    if not respect_robots:
        return True
    return bool(robots_txt_allowed)

# ========== HTTPX client (shared static fetch wiring) ==========

def httpx_client(cfg) -> httpx.AsyncClient:
    """
    Return a preconfigured AsyncClient honoring cfg static settings.
    """
    limits = httpx.Limits(max_keepalive_connections=16, max_connections=64)
    timeout = httpx.Timeout(cfg.static_timeout_ms / 1000.0)
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }
    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        headers=headers,
        follow_redirects=True,
        http2=cfg.static_http2,
        max_redirects=cfg.static_max_redirects,
    )

# ========== Playwright helpers (shared dynamic fetch wiring) ==========

async def play_goto(page, url: str, wait_chain: list[str], timeout_ms: int) -> tuple[Optional[int], str]:
    """
    Try a chain of wait_until strategies; return (status, html).
    """
    last_status = None
    for wu in wait_chain:
        try:
            resp = await page.goto(url, wait_until=wu, timeout=timeout_ms)
            last_status = resp.status if resp else None
            break
        except Exception:
            continue
    # Ensure body exists, then pull html
    try:
        await page.wait_for_selector("body", timeout=timeout_ms)
    except Exception:
        pass
    try:
        html = await page.content()
    except Exception:
        html = ""
    return last_status, html

async def play_title(page) -> Optional[str]:
    try:
        t = await page.title()
        return (t or "").strip() or None
    except Exception:
        return None

async def play_links(page) -> list[str]:
    links: list[str] = []
    try:
        anchors = await page.locator("a[href]").element_handles()
        seen = set()
        for a in anchors:
            try:
                href = await a.get_attribute("href")
            except Exception:
                href = None
            if not href:
                continue
            try:
                abs_u = await a.evaluate("(el, base) => new URL(el.getAttribute('href'), base).toString()", page.url)
            except Exception:
                # fallback: best effort
                abs_u = urljoin(page.url, href)
            abs_u = re.sub(r"#.*$", "", abs_u)
            if abs_u not in seen:
                seen.add(abs_u)
                links.append(abs_u)
    except Exception:
        pass
    return links

# ========== Uniform link filtering ==========

def filter_links(
    source_url: str,
    links: list[str],
    cfg,
    *,
    allow_re: re.Pattern | None = None,
    deny_re: re.Pattern | None = None,
    product_only: bool = True,
) -> list[str]:
    """
    Apply same-site/language/junk/product filters uniformly.
    """
    out: list[str] = []
    seen = set()
    for u in links:
        if not is_http_url(u):
            continue

        # same-site gate
        if not same_site(source_url, u, cfg.allow_subdomains):
            continue

        # language policy
        low = u.lower()
        # subdomain deny
        host = (urlparse(u).hostname or "").lower()
        if any(sub and host.startswith(sub.strip().lower()) for sub in cfg.lang_subdomain_deny):
            continue
        # path deny
        path = (urlparse(u).path or "").lower()
        if any(p and p in path for p in cfg.lang_path_deny):
            continue
        # query keys deny
        if urlparse(u).query:
            q = dict(parse_qsl(urlparse(u).query, keep_blank_values=True))
            if any(k in q for k in cfg.lang_query_keys):
                continue

        # allow/deny regex policy
        if allow_re and not allow_re.search(u):
            continue
        if deny_re and deny_re.search(u):
            continue

        # product-likeness gate
        if product_only and looks_non_product_url(u):
            continue

        if u not in seen:
            seen.add(u)
            out.append(u)
    return out