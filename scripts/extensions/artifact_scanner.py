from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional, Set

from extensions.output_paths import OUTPUT_ROOT as OUTPUTS_DIR, sanitize_bvdid

_HTML_DIR = "html"
_MD_DIR = "markdown"
_JSON_DIR = "json"
_CHECKPOINTS_DIR = "checkpoints"
_URL_INDEX_NAME = "url_index.json"
_CRAWL_META_NAME = "crawl_meta.json"
_SESSION_CHECKPOINT_NAME = "session_checkpoint.json"

# ----------------------------
# Hashing must match run_crawl
# ----------------------------

def _url_hash(url: str) -> str:
    return sha1(url.encode()).hexdigest()[:8]

# ----------------------------
# Datamodel
# ----------------------------

@dataclass
class ArtifactReport:
    bvdid: str
    base_dir: str
    has_url_index: bool
    url_index_count: int
    saved_html_total: int
    saved_md_total: int
    saved_json_total: int
    completed_urls_by_stage: Dict[str, List[str]]
    notes: List[str]

    def as_dict(self) -> Dict:
        return asdict(self)

# ----------------------------
# Core scanner
# ----------------------------

class ArtifactScanner:
    """
    Scans outputs/{safe_bvdid}/ to infer what work is already done and
    which URLs can be skipped on resume.

    Rules:
      - If url_index.json exists, we trust it as the canonical URL list and
        use its recorded artifact paths to test completion quickly.
      - If crawl_meta.json (merged progress) exists, use it for higher-level
        counts and notes.
      - If url_index.json does not exist, we fall back to counting artifacts
        and (best-effort) mapping URLs by filename hash patterns.
    """
    def __init__(self, outputs_dir: Path = OUTPUTS_DIR) -> None:
        self.outputs_dir = outputs_dir

    # ---- path helpers ----

    def _company_root(self, bvdid: str) -> Path:
        safe = sanitize_bvdid(bvdid)
        return self.outputs_dir / safe

    def _dir_for_stage(self, bvdid: str, stage: str) -> Path:
        name = {"html": _HTML_DIR, "markdown": _MD_DIR, "json": _JSON_DIR}[stage]
        return self._company_root(bvdid) / name

    def _checkpoints_dir(self, bvdid: str) -> Path:
        return self._company_root(bvdid) / _CHECKPOINTS_DIR

    def _url_index_path(self, bvdid: str) -> Path:
        return self._checkpoints_dir(bvdid) / _URL_INDEX_NAME

    def _crawl_meta_path(self, bvdid: str) -> Path:
        return self._checkpoints_dir(bvdid) / _CRAWL_META_NAME

    def _session_checkpoint_path(self) -> Path:
        return self.outputs_dir / _CHECKPOINTS_DIR / _SESSION_CHECKPOINT_NAME

    # ---- loaders ----

    def load_url_index(self, bvdid: str) -> Dict[str, Dict[str, str]]:
        p = self._url_index_path(bvdid)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def load_crawl_meta(self, bvdid: str) -> Dict:
        p = self._crawl_meta_path(bvdid)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_session_checkpoint(self) -> Dict:
        p = self._session_checkpoint_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ---- existence checks ----

    def _artifact_exists(self, path_str: Optional[str]) -> bool:
        if not path_str:
            return False
        p = Path(path_str)
        return p.exists()

    def _infer_artifact_by_hash(self, bvdid: str, url: str, stage: str) -> Optional[Path]:
        """
        If url_index.json is missing-or-incomplete, try to find an artifact by hash.
        This matches run_crawl's naming of '*{hash}.ext' in the stage directory.
        """
        d = self._dir_for_stage(bvdid, stage)
        h = _url_hash(url)
        ext = {"html": ".html", "markdown": ".md", "json": ".json"}[stage]
        if not d.exists():
            return None
        for p in d.glob(f"*{h}{ext}"):
            return p
        return None

    # ---- public API ----

    def scan_company(self, bvdid: str) -> ArtifactReport:
        root = self._company_root(bvdid)
        notes: List[str] = []
        if not root.exists():
            return ArtifactReport(
                bvdid=bvdid,
                base_dir=str(root),
                has_url_index=False,
                url_index_count=0,
                saved_html_total=0,
                saved_md_total=0,
                saved_json_total=0,
                completed_urls_by_stage={"html": [], "markdown": [], "llm": []},
                notes=["no-company-root"],
            )

        html_dir = self._dir_for_stage(bvdid, "html")
        md_dir = self._dir_for_stage(bvdid, "markdown")
        json_dir = self._dir_for_stage(bvdid, "json")

        html_count = sum(1 for _ in html_dir.glob("*.html")) if html_dir.exists() else 0
        md_count = sum(1 for _ in md_dir.glob("*.md")) if md_dir.exists() else 0
        json_count = sum(1 for _ in json_dir.glob("*.json")) if json_dir.exists() else 0

        idx = self.load_url_index(bvdid)
        has_idx = bool(idx)
        by_stage: Dict[str, Set[str]] = {"html": set(), "markdown": set(), "llm": set()}

        # Prefer url_index.json if present - it's canonical and contains saved paths/timestamps
        if has_idx:
            for url, ent in idx.items():
                # HTML present?
                if self._artifact_exists(ent.get("html_path")):
                    by_stage["html"].add(url)
                else:
                    guessed = self._infer_artifact_by_hash(bvdid, url, "html")
                    if guessed and guessed.exists():
                        by_stage["html"].add(url)
                # Markdown present?
                if self._artifact_exists(ent.get("markdown_path")):
                    by_stage["markdown"].add(url)
                else:
                    guessed = self._infer_artifact_by_hash(bvdid, url, "markdown")
                    if guessed and guessed.exists():
                        by_stage["markdown"].add(url)
                # LLM (json) present?
                if self._artifact_exists(ent.get("json_path")):
                    by_stage["llm"].add(url)
                else:
                    guessed = self._infer_artifact_by_hash(bvdid, url, "json")
                    if guessed and guessed.exists():
                        by_stage["llm"].add(url)
        else:
            # url_index missing: fall back to crawl_meta.json for counts, then guess by artifact filenames
            meta = self.load_crawl_meta(bvdid)
            if meta:
                notes.append("crawl_meta_loaded")
            # best-effort: try to infer urls by hashing artifacts found in directories
            # collect hashes found and map back to URL by name matching is impossible without index,
            # so we only return counts and leave completed_urls_by_stage empty in this fallback case.
            # However still try to map using filename hash pattern if possible.
            # Build map hash -> filename for each stage
            hash_to_file = {}
            for d, ext in ((html_dir, ".html"), (md_dir, ".md"), (json_dir, ".json")):
                if not d.exists():
                    continue
                for p in d.glob(f"*{ext}"):
                    # filename could end with -{hash}.ext or contain the 8-char hash; we attempt to find 8 hex chars
                    name = p.name
                    # naive scan for 8-hex substring
                    for i in range(len(name) - 7):
                        chunk = name[i:i+8]
                        if all(c in "0123456789abcdef" for c in chunk.lower()):
                            hash_to_file.setdefault(chunk, []).append(str(p))
                            break

            # Attempt to associate any url in crawl_meta's seed list by recomputing hash
            meta = self.load_crawl_meta(bvdid)
            if meta:
                seeded = []
                seeding = meta.get("seeding") or {}
                # crawl_meta may have 'seed_counts_by_root' or 'seed_roots' but not full url list.
                # We can't reconstruct seeded URLs reliably; note that and return counts.
                notes.append("no-url-index-cannot-map-urls")
            else:
                notes.append("no-crawl-meta-no-url-index")

        return ArtifactReport(
            bvdid=bvdid,
            base_dir=str(root),
            has_url_index=has_idx,
            url_index_count=len(idx),
            saved_html_total=html_count,
            saved_md_total=md_count,
            saved_json_total=json_count,
            completed_urls_by_stage={
                "html": sorted(by_stage["html"]),
                "markdown": sorted(by_stage["markdown"]),
                "llm": sorted(by_stage["llm"]),
            },
            notes=notes,
        )

    def compute_resume_set(
        self,
        bvdid: str,
        seeded_urls: List[str],
        completion_stage: str,
    ) -> List[str]:
        """
        Given a set of seeded URLs and a target completion stage
        ('html' | 'markdown' | 'llm'), return URLs that still need work.
        """
        report = self.scan_company(bvdid)
        if completion_stage == "html":
            done = set(report.completed_urls_by_stage["html"])
        elif completion_stage == "markdown":
            done = set(report.completed_urls_by_stage["markdown"])
        else:
            done = set(report.completed_urls_by_stage["llm"])

        todo: List[str] = []
        for u in seeded_urls:
            if u in done:
                continue
            guessed = self._infer_artifact_by_hash(bvdid, u, completion_stage)
            if guessed and guessed.exists():
                # artifact exists on disk even if url_index doesn't list it
                continue
            todo.append(u)
        return todo