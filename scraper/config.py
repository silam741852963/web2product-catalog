from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple
from .utils import getenv_bool, getenv_int, getenv_str, getenv_float, getenv_csv

# ---------- Project Paths ----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Subfolders & files (paths only; no logging init here)
SCRAPED_HTML_DIR: Path = DATA_DIR / "scraped_html"
MARKDOWN_DIR: Path = DATA_DIR / "markdown"
OUTPUT_JSONL: Path = DATA_DIR / "output.jsonl"
INPUT_URLS: Path = DATA_DIR / "input_urls.csv"
INPUT_ROOT_DIR: Path = DATA_DIR / "input"
LOG_FILE: Path = LOG_DIR / "scraper.log"
CANDIDATES_DIR: Path = DATA_DIR / "candidates"
EVIDENCE_DIR: Path = DATA_DIR / "evidence"
ENTITIES_DIR: Path = DATA_DIR / "entities"
COMPANY_SUMMARIES_DIR: Path = DATA_DIR / "company_summaries"
PAGE_META_DIR: Path = DATA_DIR / "page_meta"
CHECKPOINTS_DIR: Path = DATA_DIR / "checkpoints"
EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"

# Ensure directories exist at import time (but do NOT create/log to files here)
for p in (
    DATA_DIR, LOG_DIR, INPUT_ROOT_DIR, SCRAPED_HTML_DIR, MARKDOWN_DIR,
    CANDIDATES_DIR, EVIDENCE_DIR, ENTITIES_DIR,
    COMPANY_SUMMARIES_DIR, PAGE_META_DIR, CHECKPOINTS_DIR, EMBEDDINGS_DIR,
):
    p.mkdir(parents=True, exist_ok=True)


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
    input_root: Path
    input_glob: str
    log_file: Path

    # Misc
    cache_html: bool
    sanitize_markdown: bool

    # Crawler extras
    crawler_max_retries: int
    per_page_delay_ms: int
    allow_subdomains: bool
    max_pages_per_company: int

    # Language/translation policy
    primary_lang: str
    lang_path_deny: tuple[str, ...]
    lang_query_keys: tuple[str, ...]
    lang_subdomain_deny: tuple[str, ...]

    # Static-first HTTP client knobs
    enable_static_first: bool
    static_timeout_ms: int
    static_max_bytes: int
    static_http2: bool
    static_max_redirects: int
    static_js_app_text_threshold: int

    # Sectionizer / classifier defaults
    min_section_chars: int
    max_section_chars: int
    product_like_url_keywords: tuple[str, ...]
    non_product_keywords: tuple[str, ...]
    prefer_detail_url_keywords: tuple[str, ...]

    # Extra output dirs
    candidates_dir: Path
    evidence_dir: Path
    entities_dir: Path
    company_summaries_dir: Path
    page_meta_dir: Path
    checkpoints_dir: Path
    embeddings_dir: Path

    # --------- Redirect/migration & filtering defaults ----------
    # Default regexes used by crawler if CLI doesn't supply --allow/--deny
    default_allow_regex: str | None
    default_deny_regex: str | None

    # Domain migration / redirect handling
    migration_threshold: int                    # e.g., 2 off-site 301/308 (or homepage once)
    migration_forbid_hosts: tuple[str, ...]     # known sinkholes, never adopt

    # Auth & backoff policies
    deny_on_auth: bool                          # 401/403: suppress frontier (non-homepage)
    backoff_on_429: float                       # multiply delay (and/or reduce seed batch) next pass

    # Same-site expansion for the current run (runner can set per company)
    extra_same_site_hosts: tuple[str, ...]      # additional eTLD+1 treated as same-site this run

    # --------- Fuse / Stall controls (read-only; set via env or defaults) ----------
    forbidden_done_threshold: int               # 403s ≥ this → mark company done
    stall_pending_max: int                      # pending ≤ this AND…
    stall_repeat_passes: int                    # …same fingerprint for ≥ this many passes → done
    stall_fingerprint_window: int               # history length to compare fingerprints

    # --------- Per-host throttling / 429 handling (crawler uses directly) ----------
    host_min_interval_ms: int                   # minimum spacing between requests to a host
    throttle_penalty_initial_ms: int            # penalty injected after the first 429
    throttle_penalty_max_ms: int                # upper bound for penalty sleep
    throttle_penalty_decay_mult: float          # multiply penalty by this on 2xx success (0<mult<1)

    # ---------- Global browser/page limits & health ----------
    max_global_pages_open: int                  # cap Playwright pages across the whole run
    page_close_timeout_ms: int                  # how long to wait when closing a page
    browser_recycle_after_pages: int            # recycle Chromium after this many page opens
    browser_recycle_after_seconds: int          # or after this many seconds (whichever first)
    watchdog_interval_seconds: int              # watchdog logging/pressure interval
    max_httpx_clients: int                      # number of shared httpx AsyncClients

    # ---------- Browser extras ----------
    proxy_server: str | None
    browser_slow_mo_ms: int
    browser_bypass_csp: bool
    browser_args_extra: tuple[str, ...]

    # GPU
    browser_enable_gpu: bool


# ---------- Loader ----------
def load_config() -> Config:

    cfg = Config(
        timezone=getenv_str("APP_TZ", "Asia/Singapore"),
        env=getenv_str("APP_ENV", "dev"),

        # ---------- Tuned for high-throughput workstation ----------
        # Cross-company parallelism is the main throughput lever.
        max_companies_parallel=getenv_int("MAX_COMPANIES_PARALLEL", 128, 1, 256),
        # Keep per-domain parallelism modest to avoid 429 bursts.
        max_pages_per_domain_parallel=getenv_int("MAX_PAGES_PER_DOMAIN_PARALLEL", 12, 1, 64),
        request_timeout_ms=getenv_int("REQUEST_TIMEOUT_MS", 30000, 5000, 120000),
        # Slightly tighter page timeout to avoid hanging SPAs.
        page_load_timeout_ms=getenv_int("PAGE_LOAD_TIMEOUT_MS", 25000, 5000, 180000),
        navigation_wait_until=getenv_str("NAV_WAIT_UNTIL", "domcontentloaded"),

        # retry/backoff
        retry_max_attempts=getenv_int("RETRY_MAX_ATTEMPTS", 3, 1, 10),
        retry_initial_delay_ms=getenv_int("RETRY_INITIAL_DELAY_MS", 10000, 100, 10000),
        retry_max_delay_ms=getenv_int("RETRY_MAX_DELAY_MS", 60000, 1000, 60000),
        retry_jitter_ms=getenv_int("RETRY_JITTER_MS", 300, 0, 2000),

        # robots / blocking
        respect_robots_txt=getenv_bool("RESPECT_ROBOTS", False),
        user_agent=getenv_str(
            "SCRAPER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        block_heavy_resources=getenv_bool("BLOCK_HEAVY_RESOURCES", True),

        # LLM / Ollama
        ollama_base_url=getenv_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=getenv_str("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M"),
        llm_max_input_tokens=getenv_int("LLM_MAX_INPUT_TOKENS", 8000, 2048, 32768),
        llm_target_json_schema_name=getenv_str("LLM_SCHEMA_NAME", "product_catalog_schema"),

        # paths
        project_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        scraped_html_dir=SCRAPED_HTML_DIR,
        markdown_dir=MARKDOWN_DIR,
        output_jsonl=OUTPUT_JSONL,
        input_urls_csv=INPUT_URLS,
        input_root=INPUT_ROOT_DIR,
        input_glob=getenv_str("INPUT_GLOB", "data/input/us/*.csv"),
        log_file=LOG_FILE,
        candidates_dir=CANDIDATES_DIR,
        evidence_dir=EVIDENCE_DIR,
        entities_dir=ENTITIES_DIR,
        company_summaries_dir=COMPANY_SUMMARIES_DIR,
        page_meta_dir=PAGE_META_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        embeddings_dir=EMBEDDINGS_DIR,

        # misc
        cache_html=getenv_bool("CACHE_HTML", True),
        sanitize_markdown=getenv_bool("SANITIZE_MARKDOWN", True),

        # crawler extras
        crawler_max_retries=getenv_int("CRAWLER_MAX_RETRIES", 3, 0, 10),
        # Gentle pacing between pages to reduce burstiness.
        per_page_delay_ms=getenv_int("PER_PAGE_DELAY_MS", 15, 0, 2000),
        allow_subdomains=getenv_bool("ALLOW_SUBDOMAINS", True),
        max_pages_per_company=getenv_int("MAX_PAGES_PER_COMPANY", 200, 1, 2000),

        # language policy
        primary_lang=getenv_str("PRIMARY_LANG", "en"),
        lang_path_deny=tuple(getenv_str("LANG_PATH_DENY", "/fr,/de,/es,/pt,/it,/ru,/zh,/zh-cn,/ja,/ko").split(",")),
        lang_query_keys=tuple(getenv_str("LANG_QUERY_KEYS", "lang,locale,hl").split(",")),
        lang_subdomain_deny=tuple(getenv_str("LANG_SUBDOMAIN_DENY", "fr.,de.,es.,pt.,it.,ru.,zh.,cn.,jp.,kr.").split(",")),

        # Static-first HTTP client
        static_timeout_ms=getenv_int("STATIC_TIMEOUT_MS", 9000, 1000, 60000),
        static_max_bytes=getenv_int("STATIC_MAX_BYTES", 2_000_000, 200_000, 8_000_000),
        enable_static_first=getenv_bool("ENABLE_STATIC_FIRST", True),
        static_http2=getenv_bool("STATIC_HTTP2", True),
        static_max_redirects=getenv_int("STATIC_MAX_REDIRECTS", 8, 1, 20),
        static_js_app_text_threshold=getenv_int("STATIC_JS_APP_TEXT_THRESHOLD", 800, 200, 4000),

        # sectionizer/classifier
        min_section_chars=getenv_int("MIN_SECTION_CHARS", 180, 50, 2000),
        max_section_chars=getenv_int("MAX_SECTION_CHARS", 2400, 400, 6000),
        product_like_url_keywords=tuple(getenv_str("PRODUCT_URL_KEYWORDS", "/product,/products,/solutions,/services,/catalog").split(",")),
        non_product_keywords=tuple(getenv_str("NON_PRODUCT_KEYWORDS", "/blog,/news,/legal,/privacy,/careers,/investors").split(",")),
        prefer_detail_url_keywords=tuple(getenv_str("PREFER_DETAIL_URL_KEYWORDS", "/product,/products").split(",")),

        # --------- Redirect/migration & filtering defaults ----------
        default_allow_regex=None,
        default_deny_regex=None,

        migration_threshold=getenv_int("MIGRATION_THRESHOLD", 2, 1, 10),
        migration_forbid_hosts=getenv_csv(
            "MIGRATION_FORBID_HOSTS",
            "youtube.com,facebook.com,instagram.com,tiktok.com,vimeo.com,shop.app,amazon.com,medium.com,linktr.ee,mailchi.mp,bit.ly"
        ),

        deny_on_auth=getenv_bool("DENY_ON_AUTH", True),
        # Stronger next-pass backoff factor when we keep hitting auth walls/429.
        backoff_on_429=getenv_float("BACKOFF_ON_429", 2.5),

        # Runner can set this per company (comma-separated eTLD+1); default empty
        extra_same_site_hosts=getenv_csv("EXTRA_SAME_SITE_HOSTS", ""),

        # --------- Fuse / Stall controls ----------
        # Lower threshold so 403-heavy companies finish faster.
        forbidden_done_threshold=getenv_int("FORBIDDEN_DONE_THRESHOLD", 25, 1, 10_000),
        stall_pending_max=getenv_int("STALL_PENDING_MAX", 2, 0, 50),
        # Fast-finish after 2 consecutive no-progress passes with tiny frontier.
        stall_repeat_passes=getenv_int("STALL_REPEAT_PASSES", 2, 1, 50),
        stall_fingerprint_window=getenv_int("STALL_FINGERPRINT_WINDOW", 4, 2, 50),

        # --------- Per-host throttling / 429 handling ----------
        # Gentle per-host pacing reduces 429 bursts while keeping overall QPS high.
        host_min_interval_ms=getenv_int("HOST_MIN_INTERVAL_MS", 150, 0, 60000),
        # Stronger penalty after a 429; decays on success.
        throttle_penalty_initial_ms=getenv_int("THROTTLE_PENALTY_INITIAL_MS", 8000, 0, 120000),
        throttle_penalty_max_ms=getenv_int("THROTTLE_PENALTY_MAX_MS", 60000, 100, 300000),
        throttle_penalty_decay_mult=getenv_float("THROTTLE_PENALTY_DECAY_MULT", 0.66),

        # ---------- Global browser/page limits & health ----------
        # 32GB RAM default: keep this conservative to prevent RAM runaway.
        max_global_pages_open=getenv_int("MAX_GLOBAL_PAGES_OPEN", 128, 32, 1024),
        page_close_timeout_ms=getenv_int("PAGE_CLOSE_TIMEOUT_MS", 1500, 100, 10000),
        browser_recycle_after_pages=getenv_int("BROWSER_RECYCLE_AFTER_PAGES", 10_000, 1_000, 1_000_000),
        browser_recycle_after_seconds=getenv_int("BROWSER_RECYCLE_AFTER_SECONDS", 21_600, 600, 172_800),
        watchdog_interval_seconds=getenv_int("WATCHDOG_INTERVAL_SECONDS", 30, 5, 600),
        max_httpx_clients=getenv_int("MAX_HTTPX_CLIENTS", 3, 1, 16),

        # ---------- Browser extras ----------
        proxy_server=getenv_str("PROXY_SERVER", "") or None,
        browser_slow_mo_ms=getenv_int("BROWSER_SLOW_MO_MS", 0, 0, 5000),
        browser_bypass_csp=getenv_bool("BROWSER_BYPASS_CSP", False),
        browser_args_extra=getenv_csv("BROWSER_ARGS_EXTRA", ""),

        # GPU
        browser_enable_gpu=getenv_bool("BROWSER_ENABLE_GPU", True)
    )
    return cfg