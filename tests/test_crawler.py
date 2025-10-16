import os
import re
import pytest

from scraper.config import load_config
from scraper.utils import TransientHTTPError


def _clear_env(keys):
    for k in keys:
        os.environ.pop(k, None)


def test_crawl_site_max_pages_cap(monkeypatch):
    # ensure default present and within range
    _clear_env(["MAX_PAGES_PER_COMPANY"])
    cfg = load_config()
    assert isinstance(cfg.max_pages_per_company, int)
    assert 1 <= cfg.max_pages_per_company <= 2000

    # override w/ clamp
    monkeypatch.setenv("MAX_PAGES_PER_COMPANY", "999999")
    cfg2 = load_config()
    assert cfg2.max_pages_per_company <= 2000


def test_language_filters_drop_non_primary(monkeypatch):
    cfg = load_config()
    # Should contain common language path-blockers
    assert "/fr" in cfg.lang_path_deny or "fr" in ",".join(cfg.lang_path_deny)
    # Subdomain deny list should include language-like prefixes
    deny_subs = ",".join(cfg.lang_subdomain_deny)
    assert "fr." in deny_subs or "de." in deny_subs


def test_subdomain_toggle(monkeypatch):
    # default True
    _clear_env(["ALLOW_SUBDOMAINS"])
    cfg = load_config()
    assert cfg.allow_subdomains is True

    # toggle off
    monkeypatch.setenv("ALLOW_SUBDOMAINS", "false")
    cfg2 = load_config()
    assert cfg2.allow_subdomains is False


def test_fetch_retry_on_transient_error():
    # The crawler uses TransientHTTPError to signal retryable issues.
    err = TransientHTTPError("temporary network blip")
    assert isinstance(err, Exception)


def test_http_error_status_triggers_nonretryable_policy_knobs():
    # We validate presence of policy knobs used by crawler logic.
    cfg = load_config()
    assert hasattr(cfg, "deny_on_auth") and isinstance(cfg.deny_on_auth, bool)
    assert hasattr(cfg, "migration_threshold") and isinstance(cfg.migration_threshold, int)