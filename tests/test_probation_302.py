from pathlib import Path
from types import SimpleNamespace

from scripts.run_scraper import _record_redirect, _save_company_state

def _mk_cfg(**over):
    base = dict(
        allow_subdomains=True,
        migration_threshold=3,
        migration_forbid_hosts=tuple(),
    )
    base.update(over)
    return SimpleNamespace(**base)

def test_homepage_302_adds_probation_and_enqueues(tmp_path: Path):
    cfg = _mk_cfg()
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
                     src="https://old.com/",
                     dst="https://beta.new.com/welcome",
                     code=302,
                     is_homepage=True)

    assert "new.com" in set(state.get("temp_allowed_hosts", []))
    assert any(u.startswith("https://beta.new.com/welcome") for u in state.get("pending", []))
    assert state.get("migrated_to") in (None, "",)