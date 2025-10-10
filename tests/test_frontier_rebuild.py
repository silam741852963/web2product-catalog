from pathlib import Path
from scripts.run_scraper import _rebuild_frontier_after_migration

def test_frontier_rebuild_maps_old_to_new_and_drops_offdomain(tmp_path: Path):
    state = {
        "homepage": "https://old.com/",
        "visited": ["https://old.com/a"],
        "pending": [
            "https://old.com/a",
            "https://old.com/b?x=1",
            "https://off.example/x",
            "https://new.com/c",
        ],
    }
    _rebuild_frontier_after_migration(state, new_base="new.com", allow_subdomains=True)

    pend = set(state["pending"])
    assert "https://new.com/b?x=1" in pend
    assert "https://new.com/c" in pend
    assert "https://off.example/x" not in pend
    assert not any(u.startswith("https://old.com") for u in pend)