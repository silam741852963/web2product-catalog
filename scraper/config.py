"""
Centralized configuration for the web2product-catalog project.

- Loads environment variables (with sane defaults)
- Normalizes and ensures data/log directories exist
- Exposes a strongly-typed Config object for use across modules
- Sets up standard logging via dictConfig (rotating file + console)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from logging.config import dictConfig


# ---------- Project Paths ----------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# subfolders
SCRAPED_HTML_DIR: Path = DATA_DIR / "scraped_html"
MARKDOWN_DIR: Path = DATA_DIR / "markdown"
OUTPUT_JSONL: Path = DATA_DIR / "output.jsonl"
INPUT_URLS: Path = DATA_DIR / "input_urls.csv"
LOG_FILE: Path = LOG_DIR / "scraper.log"
CANDIDATES_DIR: Path = DATA_DIR / "candidates"
EVIDENCE_DIR: Path = DATA_DIR / "evidence"
ENTITIES_DIR: Path = DATA_DIR / "entities"
COMPANY_SUMMARIES_DIR: Path = DATA_DIR / "company_summaries"
PAGE_META_DIR: Path = DATA_DIR / "page_meta"
CHECKPOINTS_DIR: Path = DATA_DIR / "checkpoints"
EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"

# Ensure directories exist at import time (idempotent)
for p in (
    DATA_DIR, LOG_DIR, SCRAPED_HTML_DIR, MARKDOWN_DIR,
    CANDIDATES_DIR, EVIDENCE_DIR, ENTITIES_DIR,
    COMPANY_SUMMARIES_DIR, PAGE_META_DIR, CHECKPOINTS_DIR, EMBEDDINGS_DIR,
):
    p.mkdir(parents=True, exist_ok=True)


# ---------- Environment helpers ----------

def _getenv_str(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value if isinstance(value, str) and value.strip() else default


def _getenv_int(name: str, default: int, min_val: Optional[int] = None,
                max_val: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        return val
    except ValueError:
        return default


def _getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------- Config dataclass ----------

@dataclass(frozen=True)
class Config:
    # Runtime
    timezone: str
    env: Literal["dev", "staging", "prod"]

    # Concurrency & timeouts
    max_companies_parallel: int
    max_pages_per_domain_parallel: int
    request_timeout_ms: int
    page_load_timeout_ms: int
    navigation_wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"]

    # Retry / backoff
    retry_max_attempts: int
    retry_initial_delay_ms: int
    retry_max_delay_ms: int
    retry_jitter_ms: int

    # Robots / ethics / blocking
    respect_robots_txt: bool
    user_agent: str
    block_heavy_resources: bool

    # LLM / Ollama
    ollama_base_url: str
    ollama_model: str
    llm_max_input_tokens: int
    llm_target_json_schema_name: str

    # Paths
    project_root: Path
    data_dir: Path
    scraped_html_dir: Path
    markdown_dir: Path
    output_jsonl: Path
    input_urls_csv: Path
    log_file: Path

    # Misc
    cache_html: bool
    sanitize_markdown: bool

    # Crawler extras
    crawler_max_retries: int
    per_page_delay_ms: int  # polite delay per fetched page (0 = disabled)
    allow_subdomains: bool  # include subdomains as same "site" when True
    max_pages_per_company: int  # soft cap per company (adaptive logic can stop earlier)

    # Language/translation policy
    primary_lang: str
    lang_path_deny: tuple[str, ...]
    lang_query_keys: tuple[str, ...]
    lang_subdomain_deny: tuple[str, ...]

    # Sectionizer / classifier defaults
    min_section_chars: int
    max_section_chars: int
    product_like_url_keywords: tuple[str, ...]
    non_product_keywords: tuple[str, ...]
    prefer_detail_url_keywords: tuple[str, ...]  # used later in consolidation

    # Paths
    candidates_dir: Path
    evidence_dir: Path
    entities_dir: Path
    company_summaries_dir: Path
    page_meta_dir: Path
    checkpoints_dir: Path
    embeddings_dir: Path


def load_config() -> Config:
    cfg = Config(
        timezone=_getenv_str("APP_TZ", "Asia/Singapore"),
        env=_getenv_str("APP_ENV", "dev"),

        # concurrency/timeouts
        max_companies_parallel=_getenv_int("MAX_COMPANIES_PARALLEL", 14, min_val=1, max_val=64),
        max_pages_per_domain_parallel=_getenv_int("MAX_PAGES_PER_DOMAIN_PARALLEL", 4, min_val=1, max_val=32),
        request_timeout_ms=_getenv_int("REQUEST_TIMEOUT_MS", 30000, min_val=5000, max_val=120000),
        page_load_timeout_ms=_getenv_int("PAGE_LOAD_TIMEOUT_MS", 45000, min_val=5000, max_val=180000),
        navigation_wait_until=_getenv_str("NAV_WAIT_UNTIL", "domcontentloaded"),

        # retry/backoff
        retry_max_attempts=_getenv_int("RETRY_MAX_ATTEMPTS", 4, min_val=1, max_val=10),
        retry_initial_delay_ms=_getenv_int("RETRY_INITIAL_DELAY_MS", 500, min_val=100, max_val=10000),
        retry_max_delay_ms=_getenv_int("RETRY_MAX_DELAY_MS", 8000, min_val=1000, max_val=60000),
        retry_jitter_ms=_getenv_int("RETRY_JITTER_MS", 300, min_val=0, max_val=2000),

        # robots / blocking
        respect_robots_txt=_getenv_bool("RESPECT_ROBOTS", False),
        user_agent=_getenv_str(
            "SCRAPER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        block_heavy_resources=_getenv_bool("BLOCK_HEAVY_RESOURCES", True),

        # LLM / Ollama
        ollama_base_url=_getenv_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_getenv_str("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
        llm_max_input_tokens=_getenv_int("LLM_MAX_INPUT_TOKENS", 8000, min_val=2048, max_val=32768),
        llm_target_json_schema_name=_getenv_str("LLM_SCHEMA_NAME", "product_catalog_schema"),

        # paths
        project_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        scraped_html_dir=SCRAPED_HTML_DIR,
        markdown_dir=MARKDOWN_DIR,
        output_jsonl=OUTPUT_JSONL,
        input_urls_csv=INPUT_URLS,
        log_file=LOG_FILE,
        candidates_dir=CANDIDATES_DIR,
        evidence_dir=EVIDENCE_DIR,
        entities_dir=ENTITIES_DIR,
        company_summaries_dir=COMPANY_SUMMARIES_DIR,
        page_meta_dir=PAGE_META_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        embeddings_dir=EMBEDDINGS_DIR,

        # misc
        cache_html=_getenv_bool("CACHE_HTML", True),
        sanitize_markdown=_getenv_bool("SANITIZE_MARKDOWN", True),

        # crawler extras
        crawler_max_retries=_getenv_int("CRAWLER_MAX_RETRIES", 3, 0, 10),
        per_page_delay_ms=_getenv_int("PER_PAGE_DELAY_MS", 50, 0, 2000),
        allow_subdomains=_getenv_bool("ALLOW_SUBDOMAINS", True),
        max_pages_per_company=_getenv_int("MAX_PAGES_PER_COMPANY", 150, 50, 2000),

        # language policy
        primary_lang=_getenv_str("PRIMARY_LANG", "en"),
        lang_path_deny=tuple(
            _getenv_str("LANG_PATH_DENY", "/fr,/de,/es,/pt,/it,/ru,/zh,/zh-cn,/ja,/ko").split(",")
        ),
        lang_query_keys=tuple(
            _getenv_str("LANG_QUERY_KEYS", "lang,locale,hl").split(",")
        ),
        lang_subdomain_deny=tuple(
            _getenv_str("LANG_SUBDOMAIN_DENY", "fr.,de.,es.,pt.,it.,ru.,zh.,cn.,jp.,kr.").split(",")
        ),

        # sectionizer/classifier
        min_section_chars=_getenv_int("MIN_SECTION_CHARS", 180, 50, 2000),
        max_section_chars=_getenv_int("MAX_SECTION_CHARS", 2400, 400, 6000),
        product_like_url_keywords=tuple(
            _getenv_str("PRODUCT_URL_KEYWORDS", "/product,/products,/solutions,/services,/catalog").split(",")
        ),
        non_product_keywords=tuple(
            _getenv_str("NON_PRODUCT_KEYWORDS", "/blog,/news,/legal,/privacy,/careers,/investors").split(",")
        ),
        prefer_detail_url_keywords=tuple(
            _getenv_str("PREFER_DETAIL_URL_KEYWORDS", "/product,/products").split(",")
        ),
    )
    _init_logging(cfg)
    return cfg


# ---------- Logging ----------

def _init_logging(cfg: Config) -> None:
    """
    Configure structured logging:
    - Rotating file handler for long runs
    - Console handler for local dev
    """
    log_level = "INFO" if cfg.env != "dev" else "DEBUG"

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "[{levelname}] {asctime} {name}: {message}",
                "style": "{",
            },
            "file": {
                "format": "{asctime} | {levelname:<8} | {name:<24} | {message}",
                "style": "{",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "console",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "file",
                "filename": str(cfg.log_file),
                "maxBytes": 5 * 1024 * 1024,  # 5 MB
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file"],
        },
    })