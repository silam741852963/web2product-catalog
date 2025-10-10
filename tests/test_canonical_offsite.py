from pathlib import Path
from types import SimpleNamespace

from scripts.run_scraper import _record_canonical, _save_company_state

def _mk_cfg(**over):
    base = dict(
        allow_subdomains=True,
        migration_threshold=1,
        migration_forbid_hosts=("youtube.com",),
    )
    base.update(over)
    return SimpleNamespace(**base)

def test_canonical_offsite_counts_as_migration(tmp_path: Path):
    cfg = _mk_cfg(migration_threshold=1)
    state_path = tmp_path / "company.json"
    state = {
        "company": "ACME",
        "homepage": "https://old.com/",
        "visited": [],
        "pending": ["https://old.com/"],
        "done": False,
    }
    _save_company_state(state_path, state)

    _record_canonical(cfg, state_path, state,
                      src="https://old.com/",
                      dst="https://new.com/",
                      is_homepage=True)

    assert state["migrated_to"] == "new.com"
    assert "old.com" in state.get("alias_domains", [])

def test_canonical_forbidden_host_ignored(tmp_path: Path):
    cfg = _mk_cfg(migration_threshold=1, migration_forbid_hosts=("youtube.com",))
    state_path = tmp_path / "company.json"
    state = {
        "company": "ACME",
        "homepage": "https://old.com/",
        "visited": [],
        "pending": ["https://old.com/"],
        "done": False,
    }
    _save_company_state(state_path, state)

    _record_canonical(cfg, state_path, state,
                      src="https://old.com/",
                      dst="https://youtube.com/somechannel",
                      is_homepage=True)

    assert state.get("migrated_to") in (None, "",)
    assert "old.com" not in set(state.get("alias_domains", []))