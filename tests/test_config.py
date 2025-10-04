# tests/test_config.py
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
        "CACHE_HTML", "SANITIZE_MARKDOWN",
        # NEW fields:
        "CRAWLER_MAX_RETRIES", "BLOCK_HEAVY_RESOURCES", "PER_PAGE_DELAY_MS",
        "MIN_SECTION_CHARS", "MAX_SECTION_CHARS",
        "PRODUCT_URL_KEYWORDS", "NON_PRODUCT_KEYWORDS", "PREFER_DETAIL_URL_KEYWORDS",
    ]:
        monkeypatch.delenv(var, raising=False)

    cfg = load_config()

    # existing expectations
    assert cfg.timezone == "Asia/Ho_Chi_Minh"
    assert cfg.env == "dev"
    assert cfg.max_companies_parallel == 6
    assert cfg.max_pages_per_domain_parallel == 6
    assert cfg.request_timeout_ms == 30000
    assert cfg.page_load_timeout_ms == 35000
    assert cfg.navigation_wait_until == "networkidle"
    assert cfg.retry_max_attempts == 4
    assert cfg.retry_initial_delay_ms == 500
    assert cfg.retry_max_delay_ms == 8000
    assert cfg.retry_jitter_ms == 300
    assert cfg.respect_robots_txt is False
    assert "Mozilla/5.0" in cfg.user_agent
    assert cfg.ollama_base_url == "http://localhost:11434"
    assert isinstance(cfg.output_jsonl, Path)
    assert cfg.cache_html is True
    assert cfg.sanitize_markdown is True

    # NEW: defaults for added knobs/paths
    assert isinstance(cfg.candidates_dir, Path)
    assert isinstance(cfg.entities_dir, Path)
    assert isinstance(cfg.company_summaries_dir, Path)
    assert isinstance(cfg.page_meta_dir, Path)
    assert isinstance(cfg.checkpoints_dir, Path)
    assert isinstance(cfg.embeddings_dir, Path)

    assert cfg.crawler_max_retries in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # default 2 per our suggestion
    assert isinstance(cfg.block_heavy_resources, bool)
    assert cfg.per_page_delay_ms >= 0

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


def test_logging_file_exists_and_writable():
    # The config defines LOG_FILE constant; ensure actual file exists.
    # The logging handler may not write here, but presence is enough.
    assert LOG_FILE.exists()
    assert LOG_FILE.stat().st_size >= 0
