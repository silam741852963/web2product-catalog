from __future__ import annotations

import re


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_into_sentences(text: str) -> list[str]:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)

    # First split on blank lines to avoid joining unrelated blocks.
    blocks = [b.strip() for b in re.split(r"\n{2,}", s) if b.strip()]

    out: list[str] = []
    for b in blocks:
        # Keep single newlines as spaces inside a block.
        b2 = re.sub(r"\n+", " ", b).strip()
        if not b2:
            continue
        parts = [p.strip() for p in _SENT_SPLIT_RE.split(b2) if p.strip()]
        out.extend(parts)

    return out
