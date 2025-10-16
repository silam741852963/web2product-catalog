import os
import pytest

from scraper.config import load_config


def _clear_env(keys):
    for k in keys:
        os.environ.pop(k, None)


def test_load_config_defaults(monkeypatch):
    # clear potentially noisy env vars
    _clear_env([
        "MAX_COMPANIES_PARALLEL",
        "MAX_PAGES_PER_DOMAIN_PARALLEL",
        "HOST_MIN_INTERVAL_MS",
        "BROWSER_ENABLE_GPU",
        "BLOCK_HEAVY_RESOURCES",
        "FORBIDDEN_DONE_THRESHOLD",
        "STALL_PENDING_MAX",
        "STALL_REPEAT_PASSES",
        "STALL_FINGERPRINT_WINDOW",
        "ENABLE_STATIC_FIRST",
        "STATIC_TIMEOUT_MS",
        "STATIC_MAX_BYTES",
        "STATIC_HTTP2",
        "STATIC_MAX_REDIRECTS",
        "STATIC_JS_APP_TEXT_THRESHOLD",
        "MIN_SECTION_CHARS",
        "MAX_SECTION_CHARS",
        "PRODUCT_URL_KEYWORDS",
        "NON_PRODUCT_KEYWORDS",
        "PREFER_DETAIL_URL_KEYWORDS",
        "MIGRATION_THRESHOLD",
        "MIGRATION_FORBID_HOSTS",
        "DENY_ON_AUTH",
        "BACKOFF_ON_429",
        "EXTRA_SAME_SITE_HOSTS",
        # global page pool / recycle / logging
        "MAX_GLOBAL_PAGES_OPEN",
        "PAGE_CLOSE_TIMEOUT_MS",
        "BROWSER_RECYCLE_AFTER_PAGES",
        "BROWSER_RECYCLE_AFTER_SECONDS",
        "WATCHDOG_INTERVAL_SECONDS",
        "MAX_HTTPX_CLIENTS",
    ])

    cfg = load_config()

    # Core concurrency defaults (as tuned in your loader)
    assert cfg.max_companies_parallel == 128
    assert cfg.max_pages_per_domain_parallel == 12

    # Static-first client knobs exist with sensible defaults
    assert isinstance(cfg.enable_static_first, bool)
    assert isinstance(cfg.static_timeout_ms, int)
    assert cfg.static_timeout_ms > 0
    assert isinstance(cfg.static_max_bytes, int)
    assert cfg.static_max_bytes >= 200_000

    # Global page-pool / recycle knobs should exist and be within sane ranges
    assert hasattr(cfg, "max_global_pages_open")
    assert isinstance(cfg.max_global_pages_open, int)
    assert cfg.max_global_pages_open >= 64

    assert hasattr(cfg, "page_close_timeout_ms")
    assert isinstance(cfg.page_close_timeout_ms, int)
    assert 200 <= cfg.page_close_timeout_ms <= 10_000

    assert hasattr(cfg, "browser_recycle_after_pages")
    assert isinstance(cfg.browser_recycle_after_pages, int)
    assert cfg.browser_recycle_after_pages >= 1_000

    assert hasattr(cfg, "browser_recycle_after_seconds")
    assert isinstance(cfg.browser_recycle_after_seconds, int)
    assert cfg.browser_recycle_after_seconds >= 3600

    assert hasattr(cfg, "watchdog_interval_seconds")
    assert isinstance(cfg.watchdog_interval_seconds, int)
    assert 5 <= cfg.watchdog_interval_seconds <= 600

    assert hasattr(cfg, "max_httpx_clients")
    assert isinstance(cfg.max_httpx_clients, int)
    assert 1 <= cfg.max_httpx_clients <= 8

    # GPU default: enabled-by-default for headless acceleration
    assert isinstance(cfg.browser_enable_gpu, bool)


def test_load_config_env_overrides_and_bounds(monkeypatch):
    # push extremes to test clamping
    monkeypatch.setenv("MAX_COMPANIES_PARALLEL", "999")
    monkeypatch.setenv("MAX_PAGES_PER_DOMAIN_PARALLEL", "999")
    monkeypatch.setenv("HOST_MIN_INTERVAL_MS", "-10")
    monkeypatch.setenv("BROWSER_ENABLE_GPU", "false")
    monkeypatch.setenv("BLOCK_HEAVY_RESOURCES", "false")
    monkeypatch.setenv("STATIC_TIMEOUT_MS", "500")
    monkeypatch.setenv("STATIC_MAX_BYTES", "100000")
    monkeypatch.setenv("WATCHDOG_INTERVAL_SECONDS", "5")

    cfg = load_config()

    # check clamps
    assert cfg.max_companies_parallel == 256
    assert cfg.max_pages_per_domain_parallel == 64
    assert cfg.host_min_interval_ms == 0

    # booleans honored
    assert cfg.browser_enable_gpu is False
    assert cfg.block_heavy_resources is False

    # static-first clamps
    assert cfg.static_timeout_ms >= 1000
    assert cfg.static_max_bytes >= 200_000

    # global watchdog & clients
    assert cfg.watchdog_interval_seconds == 5
    assert cfg.max_httpx_clients == 3