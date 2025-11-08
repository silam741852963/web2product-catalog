"""
Central language configuration for crawling / extraction / md-generation.

Two usage modes:
1) Single-file mode: edit the ENGLISH defaults below.
2) Multi-language mode: add files scripts/config/lang_<code>.py
   Each file should define a dict named LANG_SPEC with the same structure
   as DEFAULT_LANG_SPEC. Then pass --lang <code> to run_crawl.py to pick it.

At import-time this module exposes:
 - get_lang_spec(code="en") -> dict
 - get(key, default=None) -> shorthand to current active spec (set by load_lang)
 - a default active spec (english)
"""

from __future__ import annotations
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Pattern

_BASE_DIR = Path(__file__).parent

# ---------------------------
# Default English spec object
# ---------------------------
DEFAULT_LANG_SPEC: Dict[str, Any] = {
    # MD / interstitial patterns (list of regex strings)
    "INTERSTITIAL_PATTERNS": [
        r"\bforbidden\b",
        r"\baccess\s+denied\b",
        r"\byou\s+don'?t\s+have\s+permission\b",
        r"\bnot\s+authorized\b",
        r"\bunauthorized\b",
        r"\berror\s*(401|403)\b",
        r"\btemporarily\s+blocked\b",
        r"\bchecking\s+your\s+browser\b",
        r"\bverify\s+you\s+are\s+human\b",
        r"\bcaptcha\b",
        r"\bcookies?\s+(policy|preferences|settings)\b",
        r"\bwe\s+use\s+cookies\b",
        r"\baccept\s+all\s+cookies\b",
        r"\bmanage\s+cookies\b",
        r"\bsubscribe\s+to\s+continue\b",
        r"\bsign\s*in\s+to\s+continue\b",
        r"\blog\s*in\s+to\s+continue\b",
        r"\benable\s+javascript\b",
        r"\bprivacy\s+policy\b",
        r"\bterms\s+(and|&)\s+conditions\b",
    ],
    "COOKIEISH_PATTERNS": [
        r"\bcookies?\b",
        r"\bconsent\b",
        r"\bprivacy\b",
        r"\byour\s+device\b",
        r"\bpreferences?\b",
        r"\bpersonalized\s+web\s+experience\b",
        r"\bopt\s+(in|out)\b",
    ],

    # Product-ish tokens used in md gating and seeding
    "PRODUCT_TOKENS": [
        "product", "products",
        "service", "services",
        "solution", "solutions",
        "platform", "platforms",
        "pricing", "catalog", "catalogue",
        "features", "specifications", "specs",
        "datasheet", "SDS", "MSDS",
        "buy", "quote",
        "equipment", "industrial", "manufacturing", "technology",
    ],

    # SMART include tokens used by URL seeder heuristics
    "SMART_INCLUDE_TOKENS": [
        "product", "products",
        "service", "services",
        "solution", "solutions",
        "platform", "platforms",
        "pricing",
        "catalog", "catalogue",
        "features", "specs", "specifications",
        "datasheet", "data-sheet", "sds", "msds",
        "buy", "quote",
        "equipment", "industrial", "manufacturing", "technology",
    ],

    # Default include / exclude patterns (URL globs)
    "DEFAULT_INCLUDE_PATTERNS": [
        "*/product/*", "*/products/*",
        "*/service/*", "*/services/*",
        "*/solution/*", "*/solutions/*",
        "*/platform/*", "*/platforms/*",
        "*/capabilities/*", "*/offerings/*",
        "*/pricing/*", "*/catalog/*",
        "*/what-we-do/*", "*/technology/*",
        "*/industr*", "*/manufactur*", "*/equipment*",
        "*/catalogue/*","*/portfolio/*","*/features/*","*/specs/*",
        "*/datasheet/*", "*/datasheets/*", "*/data-sheet/*",
        "*/store/*", "*/shop/*", "*/plans/*", "*/buy/*", "*/quote/*",
    ],

    "DEFAULT_EXCLUDE_PATTERNS": [
        "*/blog/*", "*/news/*", "*/press/*",
        "*/investor/*", "*/investors/*", "*/ir/*",
        "*/privacy*", "*/terms*", "*/legal*",
        "*/login*", "*/signin*", "*/account*",
        "*/support/*", "*/help/*", "*/docs/*", "*/developer/*", "*/api/*",
        "*/sitemap*", "*/robots.txt*", "*/rss*", "*/xmlrpc.php*",
        "*.pdf", "*.jpg", "*.jpeg", "*.png", "*.gif", "*.svg",
        "*.zip", "*.rar", "*.7z", "*.gz", "*.csv", "*.json",
    ],

    # URL query keys that should be ignored / treated as noise
    "DEFAULT_EXCLUDE_QUERY_KEYS": [
        "gclid", "fbclid", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
        "session", "sessionid", "token", "auth", "sso", "oauth", "redirect", "ref", "trk", "_gl",
        "download", "attachment", "attachment_id", "page_id", "p", "post_type", "orderby", "order",
        "lang", "hl", "locale", "lc", "lr", "region", "optanonconsent", "consent", "gdpr", "cookie",
    ],

    # language codes / maps (defaults)
    "LANG_CODES": {
        "ar","bg","bn","bs","ca","cs","da","de","el","es","et","fa","fi","fil","fr","he","hi",
        "hr","hu","id","is","it","ja","ko","lt","lv","ms","nb","nl","no","pl","pt","ro","ru",
        "sk","sl","sq","sr","sv","ta","th","tr","uk","ur","vi","zh"
    },
    "REGIONAL_LANG_CODES": {
        "en-us","en-gb","en-au","en-ca","en-nz","es-mx","pt-br","zh-cn","zh-tw","ja-jp", "ko-kr"
    },
    "ENGLISH_REGIONS": {"us","uk","gb","au","ca","nz","ie","sg","ph","in","za"},

    # country-code -> language mapping for heuristics
    "CC_TO_LANG": {
        "cn": "zh", "tw": "zh", "hk": "zh", "mo": "zh",
        "kr": "ko", "jp": "ja",
        "vn": "vi", "th": "th", "id": "id", "my": "ms",
        "ru": "ru", "ua": "uk", "cz": "cs", "gr": "el", "pl": "pl",
        "pt": "pt", "br": "pt", "es": "es", "mx": "es",
        "de": "de", "fr": "fr", "it": "it", "nl": "nl", "be": "nl",
    },

    # pattern compilation helpers (compiled objects are created on demand)
    # Ccompiled regexes via helper functions below.

    # stopwords & file-ext stop (for tokenization / dual_bm25)
    "FILE_EXT_STOP": {
        "pdf","ps","rtf","doc","docx","ppt","pptx","xls","xlsx","csv","tsv","xml",
        "json","yml","yaml","md","rst","zip","rar","7z","gz","bz2","xz","tar","tgz",
        "exe","apk","jpg","jpeg","png","gif","svg","webp","mp3","mp4","mov"
    },

    "STOPWORDS": {
        "and","or","the","a","an","to","of","for","on","in","by","with","at","from","as",
        "our","your","my","we","you","us","it","is","are","be","this","that","these","those",
        "www","http","https","com","net","org","io","co"
    },

    # Extraction instructions (English default) — you can override in lang_xx.py
    "INSTRUCTIONS": {
        "full": """
Extract COMMERCIAL OFFERINGS the company SELLS (products or paid services).
Return data ONLY if there is clear selling/contracting intent. Otherwise return empty.

INCLUDE
- Things customers can buy/subscribe/contract: software, platforms, hardware, consumables, insurance/coverage, consulting/managed services, commodities.
- Signals: “offer / sell / provide / underwrite / distribute / export / implement / subscribe / plans / pricing / get a quote / request demo / contact sales”.

EXCLUDE
- Editorial/corporate: news, press, blogs, articles, insights, whitepapers, case studies, awards, CSR/DEI/community, partnership thank-yous, leadership profiles.
- Careers, investor relations, cookie/privacy/terms/legal, navigation-only.

TYPE DECISION
- "product": a thing (including commodities) with features/grades/specs/plans/coverage.
- "service": paid professional/managed services delivered under contract.

OUTPUT — EXACTLY ONE JSON OBJECT (no extra keys, no arrays at the top level):
{ "offerings": [ { "type": "product" | "service", "name": "string", "description": "string" } ] }
If nothing is relevant, return exactly: { "offerings": [] }
""".strip(),
        "presence": """
You are a strict binary classifier for whether the page clearly MARKETS products or paid services.

Return EXACTLY one JSON OBJECT with a single integer field named "r" that is either 0 or 1 and nothing else.

Examples (must follow this format exactly):
- Page without sellable offerings -> {"r": 0}
- Page clearly marketing sellable offerings -> {"r": 1}

Do not include any additional keys, arrays, comments, or surrounding text. Be conservative: if uncertain, return {"r": 0}.
""".strip()
    },
        
    # stopwords (for tokenization / dual_bm25)

    "STOPWORDS": {
        "and","or","the","a","an","to","of","for","on","in","by","with","at","from","as",
        "our","your","my","we","you","us","it","is","are","be","this","that","these","those",
        "www","http","https","com","net","org","io","co","wp","json","api","xmlrpc","cdn",
        "rest","graphql","oembed","feed","feeds","search","results","result","help","support",
    },
    "CTA_KEYWORDS": {
        "pricing","plans","plan","quote","get a quote","contact sales","request demo",
        "book a demo","buy","purchase","subscribe","trial","free trial",
        "datasheet","brochure","specification","download","get started","get in touch"
    }
    
}

# -------------------------
# runtime 'active' spec
# -------------------------
_ACTIVE_SPEC: Dict[str, Any] = DEFAULT_LANG_SPEC.copy()


# -------------------------
# helper: compile regexes
# -------------------------
def _compile_or_join(patterns: Iterable[str], flags=0) -> Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=flags)

def get_lang_spec(code: str = "en") -> Dict[str, Any]:
    """
    Try to load a per-language override file scripts/config/lang_<code>.py that
    defines LANG_SPEC dict. If not found, return DEFAULT_LANG_SPEC.
    """
    code = (code or "en").lower()
    path = _BASE_DIR / f"lang_{code}.py"
    if path.exists():
        spec_mod_name = f"config.lang_{code}"
        # load via importlib from file path
        spec = importlib.util.spec_from_file_location(spec_mod_name, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "LANG_SPEC") and isinstance(mod.LANG_SPEC, dict):
                return {**DEFAULT_LANG_SPEC, **mod.LANG_SPEC}
    # fallback to english default
    return DEFAULT_LANG_SPEC.copy()

def load_lang(code: str = "en") -> None:
    """Set the active runtime spec to the given language code."""
    global _ACTIVE_SPEC
    _ACTIVE_SPEC = get_lang_spec(code)

def get(key: str, default: Any = None) -> Any:
    """Read from the active spec."""
    return _ACTIVE_SPEC.get(key, default)

# convenience compiled regex getters (use at runtime)
def get_interstitial_re() -> Pattern:
    return _compile_or_join(get("INTERSTITIAL_PATTERNS", []), flags=re.IGNORECASE)

def get_cookieish_re() -> Pattern:
    return _compile_or_join(get("COOKIEISH_PATTERNS", []), flags=re.IGNORECASE)

def get_product_tokens_re() -> Pattern:
    # compile tokens into an inside-word style regex
    tokens = get("PRODUCT_TOKENS", [])
    token_rx = r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b"
    return re.compile(token_rx, flags=re.IGNORECASE)

def default_product_bm25_query() -> str:
    return " ".join(get("PRODUCT_TOKENS", []))

def get_stopwords() -> set[str]:
    """Return the active STOPWORDS set (as strings)."""
    s = get("STOPWORDS", [])
    return set(s) if s else set()