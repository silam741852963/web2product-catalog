from pathlib import Path
from types import SimpleNamespace

from scripts.run_scraper import (
    _record_redirect,
    _save_company_state,
)

def _mk_cfg(**over):
    base = dict(
        allow_subdomains=True,
        migration_threshold=2,
        migration_forbid_hosts=tuple(),
    )
    base.update(over)
    return SimpleNamespace(**base)

def test_301_triggers_migration_and_rebuild(tmp_path: Path):
    cfg = _mk_cfg()
    state_path = tmp_path / "company.json"
    state = {
        "company": "ACME",
        "homepage": "https://www.old.com/",
        "visited": [],
        "pending": ["https://www.old.com/a", "https://www.old.com/b?x=1", "https://off.example/x"],
        "done": False,
    }
    _save_company_state(state_path, state)

    _record_redirect(cfg, state_path, state,
                     src="https://www.old.com/",
                     dst="https://new.com/",
                     code=301,
                     is_homepage=True)

    assert state["migrated_to"] == "new.com"
    assert "old.com" in state.get("alias_domains", [])
    assert state["homepage"].startswith("https://new.com/")
    assert all("off.example" not in u for u in state["pending"])
    assert any(u.startswith("https://new.com") for u in state["pending"])

def test_308_adopt_after_threshold_from_non_homepage(tmp_path: Path):
    cfg = _mk_cfg(migration_threshold=2)
    state_path = tmp_path / "company.json"
    state = {
        "company": "ACME",
        "homepage": "https://old.com/",
        "visited": [],
        "pending": ["https://old.com/"],
        "done": False,
    }
    _save_company_state(state_path, state)

    _record_redirect(cfg, state_path, state,
                     src="https://old.com/about",
                     dst="https://new.com/about",
                     code=308,
                     is_homepage=False)
    assert state.get("migrated_to") in (None, "",)

    _record_redirect(cfg, state_path, state,
                     src="https://old.com/contact",
                     dst="https://new.com/contact",
                     code=308,
                     is_homepage=False)
    assert state["migrated_to"] == "new.com"
    assert "old.com" in state.get("alias_domains", [])
