from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from extensions.io.output_paths import get_output_root, sanitize_bvdid


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class MarkdownPage:
    url: str
    text: str  # cleaned text for sentence splitting


def _clean_markdown_to_text(md: str) -> str:
    s = md

    s = _CODE_FENCE_RE.sub(" ", s)
    s = _MD_IMAGE_RE.sub(" ", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = _INLINE_CODE_RE.sub(r"\1", s)

    s = re.sub(r"^\s{0,3}#+\s+", "", s, flags=re.MULTILINE)  # headings
    s = re.sub(r"^\s{0,3}[-*+]\s+", "", s, flags=re.MULTILINE)  # list bullets
    s = re.sub(r"^\s{0,3}\d+\.\s+", "", s, flags=re.MULTILINE)  # numbered list
    s = _HTML_TAG_RE.sub(" ", s)

    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _company_base_dir(bvd_id: str) -> Path:
    out_root = Path(get_output_root()).resolve()
    safe = sanitize_bvdid(bvd_id)
    return (out_root / safe).resolve()


def company_markdown_dir(bvd_id: str) -> Path:
    return (_company_base_dir(bvd_id) / "markdown").resolve()


def company_metadata_dir(bvd_id: str) -> Path:
    return (_company_base_dir(bvd_id) / "metadata").resolve()


def company_url_index_path(bvd_id: str) -> Path:
    return (company_metadata_dir(bvd_id) / "url_index.json").resolve()


def _load_markdown_filename_to_url_map(bvd_id: str) -> dict[str, str]:
    p = company_url_index_path(bvd_id)
    if not p.exists():
        raise FileNotFoundError(f"Missing url_index.json: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"url_index.json must be a JSON object: {p}")

    out: dict[str, str] = {}
    for k, v in data.items():
        if k == "__meta__":
            continue
        if not isinstance(v, dict):
            raise RuntimeError(
                f"Invalid url_index entry type for key={k!r} (expected object): {p}"
            )

        md_path = v.get("markdown_path")
        url = v.get("url")

        if md_path is None:
            continue

        if not isinstance(md_path, str) or md_path.strip() == "":
            raise RuntimeError(f"Invalid markdown_path in url_index for key={k!r}: {p}")

        if not isinstance(url, str) or url.strip() == "":
            raise RuntimeError(f"Invalid url in url_index for key={k!r}: {p}")

        fname = Path(md_path).name
        if fname.strip() == "":
            raise RuntimeError(
                f"Could not derive markdown filename from markdown_path={md_path!r}: {p}"
            )

        prev = out.get(fname)
        if prev is None:
            out[fname] = url.strip()
        else:
            if prev != url.strip():
                raise RuntimeError(
                    "Conflicting URLs for the same markdown filename in url_index.json: "
                    f"bvd_id={bvd_id} filename={fname!r} url1={prev!r} url2={url.strip()!r} file={p}"
                )

    if len(out) == 0:
        raise RuntimeError(f"url_index.json produced an empty markdown->url map: {p}")

    return out


def iter_company_markdown_pages(bvd_id: str) -> list[MarkdownPage]:
    md_dir = company_markdown_dir(bvd_id)
    if not md_dir.exists():
        raise FileNotFoundError(f"Missing markdown directory: {md_dir}")

    md_files = sorted(md_dir.glob("*.md"))
    if len(md_files) == 0:
        raise FileNotFoundError(f"No markdown files found under: {md_dir}")

    md_to_url = _load_markdown_filename_to_url_map(bvd_id)

    pages: list[MarkdownPage] = []
    for p in md_files:
        url = md_to_url.get(p.name)
        if url is None:
            raise RuntimeError(
                "markdown file has no corresponding entry in url_index.json "
                f"(match is by filename). bvd_id={bvd_id} md_file={p} url_index={company_url_index_path(bvd_id)}"
            )

        raw = p.read_text(encoding="utf-8")
        text = _clean_markdown_to_text(raw)

        pages.append(MarkdownPage(url=url, text=text))

    return pages
