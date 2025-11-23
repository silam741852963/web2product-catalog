from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Pattern, Protocol, Set

_BASE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Default English language spec (data-only)
# --------------------------------------------------------------------------- #

DEFAULT_LANG_SPEC: Dict[str, Any] = {
    # ------------------------------------------------------------------ #
    # Interstitial / cookie-ish patterns (markdown or text snippets)
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Dual-BM25 vocab buckets
    #   - PRODUCT_TOKENS: commercial / offering / CTA-ish words
    #   - EXCLUDE_TOKENS: corporate / editorial / infra / account / dev
    # ------------------------------------------------------------------ #
    "PRODUCT_TOKENS": [
        # Core offerings
        "product", "products",
        "service", "services",
        "solution", "solutions",
        "platform", "platforms",
        "offering", "offerings",
        "capability", "capabilities",
        "technology", "technologies",
        "equipment", "industrial", "manufacturing",
        "system", "systems",
        "software", "hardware",
        "application", "applications",
        "package", "packages",
        "suite", "suites",

        # Commercial framing / catalog-ish
        "portfolio",
        "catalog", "catalogs",
        "catalogue", "catalogues",
        "range", "ranges",
        "line", "lines",          # product line

        # Specs / datasheets
        "feature", "features",
        "spec", "specs",
        "specification", "specifications",
        "datasheet", "datasheets",
        "data-sheet", "data-sheets",
        "brochure", "brochures",
        "model", "models",
        "grade", "grades",
        "variant", "variants",
        "configuration", "configurations",

        # Performance / sizing
        "capacity", "capacities",
        "performance",
        "ratings", "rating",
        "options", "option",

        # Commerce / CTAs
        "pricing", "price", "prices",
        "plan", "plans",
        "subscription", "subscriptions",
        "license", "licensing",
        "store", "shop",
        "buy", "purchase", "order",
        "quote", "rfq", "tender",
        "demo", "trial", "free", "free-trial",
        "subscription", "subscriptions",
        "package", "packages",

        # Safety / compliance docs (often tied to SKUs)
        "sds", "msds",
        "safety-data-sheet", "safety-data-sheets",

        # Distribution / channels
        "distributor", "distributors",
        "reseller", "resellers",
        "wholesale", "wholesaler", "wholesalers",

        # Verticalized offerings (common on B2B sites)
        "solution-portfolio",    # breaks into tokens but still helpful
        "industry-solutions",
        "products-and-services",
    ],

    # Tokens we treat as strongly "non-product": legal, editorial, careers, dev, etc.
    # These seed the negative_query for DualBM25.
    "EXCLUDE_TOKENS": [
        # Editorial / marketing content types
        "blog", "blogs",
        "news", "press", "media",
        "story", "stories",
        "insight", "insights",
        "article", "articles",
        "resource", "resources",
        "library",
        "ebook", "ebooks",
        "whitepaper", "whitepapers",
        "case-study", "case-studies",
        "casestudy", "casestudies",
        "testimonial", "testimonials",

        # Events & webinars
        "event", "events",
        "webinar", "webinars",
        "conference", "conferences",
        "seminar", "seminars",
        "expo", "exhibition",

        # Corporate / about / CSR
        "about", "about-us",
        "company", "corporate",
        "overview", "who-we-are",
        "history", "our-story",
        "leadership", "management",
        "team", "teams",
        "board", "governance",
        "sustainability", "csr",
        "esg", "environment",
        "community", "communities",
        "foundation", "philanthropy",

        # Careers / HR
        "career", "careers",
        "job", "jobs",
        "recruit", "recruitment",
        "talent", "vacancies",

        # Investor relations
        "investor", "investors",
        "ir", "shareholder", "shareholders",
        "stock", "stocks",
        "financial", "financials",
        "report", "reports",
        "annual-report", "annual-reports",

        # Legal / privacy / cookies
        "privacy", "privacy-policy",
        "cookie", "cookies", "cookie-policy",
        "terms", "terms-and-conditions",
        "legal", "legal-notice",
        "compliance",
        "gdpr",

        # Support / docs / help
        "support", "support-center",
        "help", "help-center",
        "faq", "faqs",
        "knowledge-base", "knowledgebase",
        "documentation", "docs",
        "manual", "manuals",
        "guide", "guides",
        "tutorial", "tutorials",

        # Developer / API / technical docs
        "developer", "developers",
        "api", "apis",
        "sdk", "sdks",
        "cli",
        "reference", "references",

        # Accounts / auth
        "login", "log-in",
        "signin", "sign-in",
        "signup", "sign-up",
        "register", "registration",
        "account", "accounts",
        "profile", "profiles",

        # Navigation / plumbing
        "sitemap",
        "robots", "robots.txt",
        "search", "search-results",
        "result", "results",
        "feed", "feeds",
        "rss",
        "xmlrpc", "xmlrpc.php",
        "wp",            # wordpress internals commonly non-product
        "admin",

        # Contact / generic pages (not necessarily purely product)
        "contact", "contact-us",
        "locations", "location",
        "office", "offices",

        # File / download oriented (often docs/resources rather than offerings)
        "download", "downloads",
        "attachment", "attachments",
        "asset", "assets",

        # Query-ish / tracking-ish tokens that appear in URLs and sometimes text
        "gclid", "fbclid",
        "utm", "utm_source", "utm_medium", "utm_campaign",
        "utm_term", "utm_content", "utm_id",
        "session", "sessionid",
        "token", "auth",
        "sso", "oauth",
        "redirect", "ref", "trk",
        "_gl",
        "lang", "locale", "region",
        "consent",
    ],
    
    # ------------------------------------------------------------------ #
    # Language codes & mappings (heuristics)
    # ------------------------------------------------------------------ #
    "LANG_CODES": {
        "ar",
        "bg",
        "bn",
        "bs",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "es",
        "et",
        "fa",
        "fi",
        "fil",
        "fr",
        "he",
        "hi",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ja",
        "ko",
        "lt",
        "lv",
        "ms",
        "nb",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sq",
        "sr",
        "sv",
        "ta",
        "th",
        "tr",
        "uk",
        "ur",
        "vi",
        "zh",
    },
    "REGIONAL_LANG_CODES": {
        "en-us",
        "en-gb",
        "en-au",
        "en-ca",
        "en-nz",
        "es-mx",
        "pt-br",
        "zh-cn",
        "zh-tw",
        "ja-jp",
        "ko-kr",
    },
    "ENGLISH_REGIONS": {
        "us",
        "uk",
        "gb",
        "au",
        "ca",
        "nz",
        "ie",
        "sg",
        "ph",
        "in",
        "za",
    },

    # country-code -> language mapping for heuristics
    "CC_TO_LANG": {
        "cn": "zh",
        "tw": "zh",
        "hk": "zh",
        "mo": "zh",
        "kr": "ko",
        "jp": "ja",
        "vn": "vi",
        "th": "th",
        "id": "id",
        "my": "ms",
        "ru": "ru",
        "ua": "uk",
        "cz": "cs",
        "gr": "el",
        "pl": "pl",
        "pt": "pt",
        "br": "pt",
        "es": "es",
        "mx": "es",
        "de": "de",
        "fr": "fr",
        "it": "it",
        "nl": "nl",
        "be": "nl",
    },

    "PATH_LANG_TOKENS": {
        "en": {
            "en", "en-us", "en-gb", "en-au", "en-ca", "en-nz",
        },
        "ja": {
            "ja", "jp", "ja-jp",
        },
        "de": {
            "de", "de-de",
        },
        "fr": {
            "fr", "fr-fr",
        },
        "es": {
            "es", "es-es", "es-mx",
        },
        "pt": {
            "pt", "pt-br", "pt-pt",
        },
        "zh": {
            "zh", "zh-cn", "zh-tw", "zh-hk",
        },
        "ko": {
            "ko", "ko-kr",
        },
        "vi": {
            "vi", "vn",
        },
    },

    # ------------------------------------------------------------------ #
    # File-extension stop list & lexical stopwords
    # ------------------------------------------------------------------ #
    "FILE_EXT_STOP": {
        "pdf",
        "ps",
        "rtf",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "xls",
        "xlsx",
        "csv",
        "tsv",
        "xml",
        "json",
        "yml",
        "yaml",
        "md",
        "rst",
        "zip",
        "rar",
        "7z",
        "gz",
        "bz2",
        "xz",
        "tar",
        "tgz",
        "exe",
        "apk",
        "jpg",
        "jpeg",
        "png",
        "gif",
        "svg",
        "webp",
        "mp3",
        "mp4",
        "mov",
    },

    "STOPWORDS": {
        "and",
        "or",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "on",
        "in",
        "by",
        "with",
        "at",
        "from",
        "as",
        "our",
        "your",
        "my",
        "we",
        "you",
        "us",
        "it",
        "is",
        "are",
        "be",
        "this",
        "that",
        "these",
        "those",
        "www",
        "http",
        "https",
        "com",
        "net",
        "org",
        "io",
        "co",
        "wp",
        "json",
        "api",
        "xmlrpc",
        "cdn",
        "rest",
        "graphql",
        "oembed",
        "feed",
        "feeds",
        "search",
        "results",
        "result",
        "help",
        "support",
    },

    "CTA_KEYWORDS": {
        "pricing",
        "plans",
        "plan",
        "quote",
        "get a quote",
        "contact sales",
        "request demo",
        "book a demo",
        "buy",
        "purchase",
        "subscribe",
        "trial",
        "free trial",
        "datasheet",
        "brochure",
        "specification",
        "download",
        "get started",
        "get in touch",
    },

    # ------------------------------------------------------------------ #
    # Extraction instructions (English defaults)
    # ------------------------------------------------------------------ #
    "INSTRUCTIONS": {
        "full": """
Extract COMMERCIAL OFFERINGS the company SELLS (products or paid services).
Return data ONLY if there is clear selling/contracting intent. Otherwise return empty.

INCLUDE
- Things customers can buy/subscribe/contract: software, platforms, hardware, consumables,
  insurance/coverage, consulting/managed services, commodities.
- Signals: “offer / sell / provide / underwrite / distribute / export / implement / subscribe
  / plans / pricing / get a quote / request demo / contact sales”.

EXCLUDE
- Editorial/corporate: news, press, blogs, articles, insights, whitepapers, case studies,
  awards, CSR/DEI/community, partnership thank-yous, leadership profiles.
- Careers, investor relations, cookie/privacy/terms/legal, navigation-only.

TYPE DECISION
- "product": a thing (including commodities) with features/grades/specs/plans/coverage.
- "service": paid professional/managed services delivered under contract.

OUTPUT — EXACTLY ONE JSON OBJECT (no extra keys, no arrays at the top level):
{ "offerings": [ { "type": "product" | "service", "name": "string", "description": "string" } ] }

If nothing is relevant, return exactly:
{ "offerings": [] }
""".strip(),
        "presence": """
You are a strict binary classifier for whether the page clearly MARKETS products or paid services.

Return EXACTLY one JSON OBJECT with a single integer field named "r" that is either 0 or 1 and nothing else.

Examples (must follow this format exactly):
- Page without sellable offerings -> {"r": 0}
- Page clearly marketing sellable offerings -> {"r": 1}

Do not include any additional keys, arrays, comments, or surrounding text.
Be conservative: if uncertain, return {"r": 0}.
""".strip(),
    },

    # ------------------------------------------------------------------ #
    # Language-specific TLD & host rules for URL filtering
    # These are *hints*; override/extend per language in lang_<code>.py.
    # ------------------------------------------------------------------ #
    "LANG_TLD_ALLOW": {
        "en": {"com", "net", "org", "io", "ai", "co"},
        "ja": {"jp", "co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp"},
    },
    "LANG_TLD_DENY": {
        "en": {"gov"},
    },
    "LANG_HOST_ALLOW_SUFFIXES": {
        "ja": {"co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp"},
    },
    "LANG_HOST_BLOCK_SUFFIXES": {
        "en": set(),
        "ja": set(),
    },
}

# --------------------------------------------------------------------------- #
# Strategy interface for language specs
# --------------------------------------------------------------------------- #


class LanguageSpecStrategy(Protocol):
    """
    Strategy interface for loading a language specification.

    Implementations are responsible for:
      - locating & loading lang_<code>.py if present
      - merging it with a base spec (e.g., DEFAULT_LANG_SPEC)

    This keeps the loading logic swappable while the Factory holds state.
    """

    def load_spec(self, code: str) -> Dict[str, Any]:  # pragma: no cover - interface
        ...


@dataclass
class DefaultLanguageSpecStrategy(LanguageSpecStrategy):
    """
    Default strategy:

    - Uses DEFAULT_LANG_SPEC as base.
    - Looks for sidecar lang_<code>.py with LANG_SPEC: Dict[str, Any].
    """

    base_spec: Dict[str, Any] = field(default_factory=lambda: DEFAULT_LANG_SPEC)
    base_dir: Path = _BASE_DIR

    def load_spec(self, code: str) -> Dict[str, Any]:
        code = (code or "en").lower()
        path = self.base_dir / f"lang_{code}.py"
        if path.exists():
            mod_name = f"language_{code}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "LANG_SPEC") and isinstance(mod.LANG_SPEC, dict):
                    merged = {**self.base_spec, **mod.LANG_SPEC}
                    return merged
        # Fallback to base-only copy
        return dict(self.base_spec)


# --------------------------------------------------------------------------- #
# Factory holding active language + helpers
# --------------------------------------------------------------------------- #


def _compile_or_join(patterns: Iterable[str], flags: int = 0) -> Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=flags)


@dataclass
class LanguageConfigFactory:
    """
    Factory that uses a LanguageSpecStrategy and maintains active runtime spec.

    Higher-level code should depend on this abstraction:
      - Call .set_language("en"/"ja"/...) at startup.
      - Use .get(), .get_interstitial_re(), etc. for language-sensitive behavior.
    """

    strategy: LanguageSpecStrategy
    active_code: str = "en"
    active_spec: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LANG_SPEC))

    # --- Core config management ------------------------------------------------

    def set_language(self, code: str = "en") -> None:
        """
        Set (and load) the active runtime language spec via the underlying strategy.
        """
        self.active_code = (code or "en").lower()
        self.active_spec = self.strategy.load_spec(self.active_code)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Read a key from the active spec.
        """
        return self.active_spec.get(key, default)

    # --- Convenience regex/lexical helpers ------------------------------------

    def interstitial_re(self) -> Pattern:
        return _compile_or_join(self.get("INTERSTITIAL_PATTERNS", []), flags=re.IGNORECASE)

    def cookieish_re(self) -> Pattern:
        return _compile_or_join(self.get("COOKIEISH_PATTERNS", []), flags=re.IGNORECASE)

    def product_tokens_re(self) -> Pattern:
        tokens: List[str] = self.get("PRODUCT_TOKENS", []) or []
        token_rx = r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b"
        return re.compile(token_rx, flags=re.IGNORECASE)

    def default_product_bm25_query(self) -> str:
        return " ".join(self.get("PRODUCT_TOKENS", []) or [])

    def stopwords(self) -> Set[str]:
        s = self.get("STOPWORDS", [])
        return set(s) if s else set()


# --------------------------------------------------------------------------- #
# Default, injectable instances
# --------------------------------------------------------------------------- #

#: Default language spec strategy: base English + lang_<code>.py overrides.
default_language_strategy = DefaultLanguageSpecStrategy()

#: Default language factory used across the project.
default_language_factory = LanguageConfigFactory(strategy=default_language_strategy)


# --------------------------------------------------------------------------- #
# Backwards-compatible module-level helpers
# --------------------------------------------------------------------------- #


def get_lang_spec(code: str = "en") -> Dict[str, Any]:
    """
    Load a per-language override file lang_<code>.py if present.

    That file must define LANG_SPEC: Dict[str, Any].
    Returned spec is DEFAULT_LANG_SPEC overlaid with LANG_SPEC.
    """
    # Use strategy directly to keep behavior identical to previous version.
    return default_language_strategy.load_spec(code)


def load_lang(code: str = "en") -> None:
    """
    Set the active runtime language spec.

    Thin wrapper over default_language_factory.set_language().
    """
    default_language_factory.set_language(code)


def get(key: str, default: Any = None) -> Any:
    """
    Read a key from the active spec.

    Thin wrapper over default_language_factory.get().
    """
    return default_language_factory.get(key, default)


def get_interstitial_re() -> Pattern:
    return default_language_factory.interstitial_re()


def get_cookieish_re() -> Pattern:
    return default_language_factory.cookieish_re()


def get_product_tokens_re() -> Pattern:
    return default_language_factory.product_tokens_re()


def default_product_bm25_query() -> str:
    return default_language_factory.default_product_bm25_query()

def get_stopwords() -> Set[str]:
    return default_language_factory.stopwords()

def get_path_lang_tokens() -> Dict[str, Set[str]]:
    """
    Return the PATH_LANG_TOKENS mapping from the active language spec.

    Keys: canonical language codes (e.g., "en", "ja")
    Values: sets of strings that can appear as path segments indicating that language.
    """
    tokens = default_language_factory.get("PATH_LANG_TOKENS", {}) or {}
    # Ensure all values are sets of lowercase strings
    out: Dict[str, Set[str]] = {}
    for code, vals in tokens.items():
        out[code.lower()] = {str(v).lower() for v in vals}
    return out

__all__ = [
    "DEFAULT_LANG_SPEC",
    "LanguageSpecStrategy",
    "DefaultLanguageSpecStrategy",
    "LanguageConfigFactory",
    "default_language_strategy",
    "default_language_factory",
    "get_lang_spec",
    "load_lang",
    "get",
    "get_interstitial_re",
    "get_cookieish_re",
    "get_product_tokens_re",
    "default_product_bm25_query",
    "get_stopwords",
    "get_path_lang_tokens",
]