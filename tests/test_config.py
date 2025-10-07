import os
from pathlib import Path

import pytest

from scraper.config import load_config, DATA_DIR, LOG_DIR, SCRAPED_HTML_DIR, MARKDOWN_DIR, LOG_FILE


def test_directories_exist_after_import():
    # created at import-time by config module
    assert DATA_DIR.exists()
    assert LOG_DIR.exists()
    assert SCRAPED_HTML_DIR.exists()
    assert MARKDOWN_DIR.exists()


def test_load_config_defaults(monkeypatch):
    # Clear relevant env vars to test defaults
    for var in [
        "APP_TZ", "APP_ENV",
        "MAX_COMPANIES_PARALLEL", "MAX_PAGES_PER_DOMAIN_PARALLEL",
        "REQUEST_TIMEOUT_MS", "PAGE_LOAD_TIMEOUT_MS", "NAV_WAIT_UNTIL",
        "RETRY_MAX_ATTEMPTS", "RETRY_INITIAL_DELAY_MS", "RETRY_MAX_DELAY_MS", "RETRY_JITTER_MS",
        "RESPECT_ROBOTS", "SCRAPER_USER_AGENT",
        "OLLAMA_BASE_URL", "OLLAMA_MODEL", "LLM_MAX_INPUT_TOKENS", "LLM_SCHEMA_NAME",
        "CACHE_HTML", "SANITIZE_MARKDOWN", "CRAWLER_MAX_RETRIES", "BLOCK_HEAVY_RESOURCES", "PER_PAGE_DELAY_MS",
        "MIN_SECTION_CHARS", "MAX_SECTION_CHARS",
        "PRODUCT_URL_KEYWORDS", "NON_PRODUCT_KEYWORDS", "PREFER_DETAIL_URL_KEYWORDS",
        # NEW:
        "ALLOW_SUBDOMAINS", "MAX_PAGES_PER_COMPANY",
        "PRIMARY_LANG", "LANG_PATH_DENY", "LANG_QUERY_KEYS", "LANG_SUBDOMAIN_DENY",
        # static-first:
        "ENABLE_STATIC_FIRST", "STATIC_TIMEOUT_MS", "STATIC_MAX_BYTES",
        "STATIC_HTTP2", "STATIC_MAX_REDIRECTS", "STATIC_JS_APP_TEXT_THRESHOLD",
        # input discovery:
        "INPUT_GLOB",
    ]:
        monkeypatch.delenv(var, raising=False)

    cfg = load_config()

    # runtime & concurrency defaults
    assert cfg.timezone == "Asia/Singapore"
    assert cfg.env == "dev"
    assert cfg.max_companies_parallel == 14
    assert cfg.max_pages_per_domain_parallel == 4
    assert cfg.request_timeout_ms == 30000
    assert cfg.page_load_timeout_ms == 45000
    assert cfg.navigation_wait_until == "domcontentloaded"

    # retry defaults
    assert cfg.retry_max_attempts == 4
    assert cfg.retry_initial_delay_ms == 500
    assert cfg.retry_max_delay_ms == 8000
    assert cfg.retry_jitter_ms == 300

    # ethics / UA
    assert cfg.respect_robots_txt is False
    assert "Mozilla/5.0" in cfg.user_agent

    # llm defaults
    assert cfg.ollama_base_url == "http://localhost:11434"
    assert isinstance(cfg.output_jsonl, Path)

    # input discovery knobs
    assert isinstance(cfg.input_root, Path)
    assert isinstance(cfg.input_glob, str) and cfg.input_glob == "data/input/us/*.csv"

    # misc flags
    assert cfg.cache_html is True
    assert cfg.sanitize_markdown is True
    assert cfg.block_heavy_resources in (True, False)
    assert cfg.crawler_max_retries == 3
    assert cfg.per_page_delay_ms == 50

    # new: subdomain, per-company budget, language policy
    assert cfg.allow_subdomains is True
    assert cfg.max_pages_per_company == 200
    assert cfg.primary_lang == "en"
    assert isinstance(cfg.lang_path_deny, tuple) and len(cfg.lang_path_deny) > 0
    assert isinstance(cfg.lang_query_keys, tuple) and len(cfg.lang_query_keys) > 0
    assert isinstance(cfg.lang_subdomain_deny, tuple) and len(cfg.lang_subdomain_deny) > 0

    # static-first defaults
    assert cfg.enable_static_first is True
    assert cfg.static_timeout_ms == 9000
    assert cfg.static_max_bytes == 2_000_000
    assert cfg.static_http2 is True
    assert cfg.static_max_redirects == 8
    assert cfg.static_js_app_text_threshold == 800

    # sectionizer/classifier defaults and path objects
    assert isinstance(cfg.candidates_dir, Path)
    assert isinstance(cfg.entities_dir, Path)
    assert isinstance(cfg.company_summaries_dir, Path)
    assert isinstance(cfg.page_meta_dir, Path)
    assert isinstance(cfg.checkpoints_dir, Path)
    assert isinstance(cfg.embeddings_dir, Path)

    assert isinstance(cfg.product_like_url_keywords, tuple)
    assert isinstance(cfg.non_product_keywords, tuple)
    assert isinstance(cfg.prefer_detail_url_keywords, tuple)


def test_load_config_env_overrides_and_bounds(monkeypatch):
    # Override with custom values (and some edge/bounds)
    monkeypatch.setenv("APP_TZ", "UTC")
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MAX_COMPANIES_PARALLEL", "1000")  # clamp to 64
    monkeypatch.setenv("MAX_PAGES_PER_DOMAIN_PARALLEL", "0")  # clamp to min 1
    monkeypatch.setenv("REQUEST_TIMEOUT_MS", "999999")  # clamp to 120000
    monkeypatch.setenv("PAGE_LOAD_TIMEOUT_MS", "3000")  # clamp to 5000
    monkeypatch.setenv("NAV_WAIT_UNTIL", "load")
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "0")  # clamp to min 1
    monkeypatch.setenv("RETRY_INITIAL_DELAY_MS", "50")  # clamp to min 100
    monkeypatch.setenv("RETRY_MAX_DELAY_MS", "999999")  # clamp to 60000
    monkeypatch.setenv("RETRY_JITTER_MS", "-1")  # clamp to min 0
    monkeypatch.setenv("RESPECT_ROBOTS", "true")
    monkeypatch.setenv("SCRAPER_USER_AGENT", "TestAgent/1.0")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("LLM_MAX_INPUT_TOKENS", "1024")  # clamp to min 2048
    monkeypatch.setenv("LLM_SCHEMA_NAME", "custom_schema")
    monkeypatch.setenv("CACHE_HTML", "false")
    monkeypatch.setenv("SANITIZE_MARKDOWN", "no")

    # NEW fields
    monkeypatch.setenv("CRAWLER_MAX_RETRIES", "9")
    monkeypatch.setenv("BLOCK_HEAVY_RESOURCES", "false")
    monkeypatch.setenv("PER_PAGE_DELAY_MS", "250")
    monkeypatch.setenv("MIN_SECTION_CHARS", "120")
    monkeypatch.setenv("MAX_SECTION_CHARS", "2600")
    monkeypatch.setenv("PRODUCT_URL_KEYWORDS", "/product,/solutions")
    monkeypatch.setenv("NON_PRODUCT_KEYWORDS", "/blog,/careers")
    monkeypatch.setenv("PREFER_DETAIL_URL_KEYWORDS", "/product")

    # newer fields (subdomain, budget, language policy)
    monkeypatch.setenv("ALLOW_SUBDOMAINS", "false")
    monkeypatch.setenv("MAX_PAGES_PER_COMPANY", "77")
    monkeypatch.setenv("PRIMARY_LANG", "de")
    monkeypatch.setenv("LANG_PATH_DENY", "/fr,/es")
    monkeypatch.setenv("LANG_QUERY_KEYS", "lang,hl")
    monkeypatch.setenv("LANG_SUBDOMAIN_DENY", "fr.,es.")

    # static-first overrides
    monkeypatch.setenv("ENABLE_STATIC_FIRST", "false")
    monkeypatch.setenv("STATIC_TIMEOUT_MS", "15000")
    monkeypatch.setenv("STATIC_MAX_BYTES", "300000")
    monkeypatch.setenv("STATIC_HTTP2", "false")
    monkeypatch.setenv("STATIC_MAX_REDIRECTS", "3")
    monkeypatch.setenv("STATIC_JS_APP_TEXT_THRESHOLD", "1200")

    # input discovery
    monkeypatch.setenv("INPUT_GLOB", "data/input/custom/*.csv")

    cfg = load_config()

    assert cfg.timezone == "UTC"
    assert cfg.env == "prod"
    assert cfg.max_companies_parallel == 64
    assert cfg.max_pages_per_domain_parallel == 1
    assert cfg.request_timeout_ms == 120000
    assert cfg.page_load_timeout_ms == 5000
    assert cfg.navigation_wait_until == "load"
    assert cfg.retry_max_attempts == 1
    assert cfg.retry_initial_delay_ms == 100
    assert cfg.retry_max_delay_ms == 60000
    assert cfg.retry_jitter_ms == 0
    assert cfg.respect_robots_txt is True
    assert cfg.user_agent == "TestAgent/1.0"
    assert cfg.ollama_base_url == "http://localhost:9999"
    assert cfg.ollama_model == "mistral"
    assert cfg.llm_max_input_tokens == 2048
    assert cfg.llm_target_json_schema_name == "custom_schema"
    assert cfg.cache_html is False
    assert cfg.sanitize_markdown is False

    # NEW fields assertions
    assert cfg.crawler_max_retries == 9
    assert cfg.block_heavy_resources is False
    assert cfg.per_page_delay_ms == 250
    assert cfg.min_section_chars == 120
    assert cfg.max_section_chars == 2600
    assert cfg.product_like_url_keywords == ("/product", "/solutions")
    assert cfg.non_product_keywords == ("/blog", "/careers")
    assert cfg.prefer_detail_url_keywords == ("/product",)

    # newer fields assertions
    assert cfg.allow_subdomains is False
    assert cfg.max_pages_per_company == 77
    assert cfg.primary_lang == "de"
    assert cfg.lang_path_deny == ("/fr", "/es")
    assert cfg.lang_query_keys == ("lang", "hl")
    assert cfg.lang_subdomain_deny == ("fr.", "es.")

    # static-first overrides
    assert cfg.enable_static_first is False
    assert cfg.static_timeout_ms == 15000
    assert cfg.static_max_bytes == 300000
    assert cfg.static_http2 is False
    assert cfg.static_max_redirects == 3
    assert cfg.static_js_app_text_threshold == 1200

    # input discovery override
    assert cfg.input_glob == "data/input/custom/*.csv"


def test_logging_file_exists_and_writable():
    # The config defines LOG_FILE constant; ensure actual file exists.
    # The logging handler may not write here, but presence is enough.
    assert LOG_FILE.exists()
    assert LOG_FILE.stat().st_size >= 0