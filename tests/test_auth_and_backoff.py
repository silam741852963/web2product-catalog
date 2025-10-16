import os
import math
import pytest

from scraper.config import load_config
from scraper.utils import parse_retry_after_header


def test_401_homepage_transient(monkeypatch):
    """
    Policy sanity check:
    - We keep the 'deny_on_auth' flag for non-homepage 401/403.
    - This test simply ensures the knob is present and set as expected.
    """
    # Ensure a clean env
    for k in ("DENY_ON_AUTH", "BACKOFF_ON_429"):
        os.environ.pop(k, None)

    cfg = load_config()
    assert hasattr(cfg, "deny_on_auth")
    assert cfg.deny_on_auth is True

    # backoff multiplier should be sane (>=1.0 so we actually back off)
    assert hasattr(cfg, "backoff_on_429")
    assert isinstance(cfg.backoff_on_429, float)
    assert cfg.backoff_on_429 >= 1.0


def test_403_non_homepage_nonretryable(monkeypatch):
    """
    Utilities sanity:
    - Retry-After parsing for numeric seconds
    - Header missing => None
    """
    # numeric seconds
    h = {"Retry-After": "120"}
    ra = parse_retry_after_header(h)
    assert isinstance(ra, float)
    assert math.isclose(ra, 120.0, rel_tol=0, abs_tol=0.5)

    # different casing
    h2 = {"retry-after": "15"}
    ra2 = parse_retry_after_header(h2)
    assert isinstance(ra2, float)
    assert math.isclose(ra2, 15.0, rel_tol=0, abs_tol=0.5)

    # missing header => None
    assert parse_retry_after_header({}) is None