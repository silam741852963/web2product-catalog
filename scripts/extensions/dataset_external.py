from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Two-level public suffixes that require one extra label to form the registrable domain
_TWO_LEVEL_SUFFIXES: Set[str] = {
    "co.uk",
    "org.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.jp",
    "ne.jp",
    "or.jp",
    "com.cn",
    "com.sg",
    "com.br",
}

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _registrable_domain(host: str) -> str:
    h = (host or "").lower().strip(".")
    if not h:
        return ""
    labels = h.split(".")
    if len(labels) < 2:
        return h

    last2 = ".".join(labels[-2:])
    if last2 in _TWO_LEVEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last2


def _host_from_url(u: str) -> str:
    """
    Robustly extract a host from:
      - Full URLs (with scheme)
      - Scheme-less: "example.com/path"
      - Protocol-relative: "//example.com"
    """
    s = (u or "").strip()
    if not s:
        return ""

    if not _SCHEME_RE.match(s):
        if s.startswith("//"):
            s = "http:" + s
        else:
            s = "http://" + s

    try:
        host = (urlparse(s).hostname or "").lower().strip(".")
        if host.startswith("www.") and len(host) > 4:
            host = host[4:]
        return host
    except Exception:
        return ""


def _add_host_variants(host: str, out: set[str]) -> None:
    h = (host or "").lower().strip(".")
    if not h:
        return
    if h.startswith("www.") and len(h) > 4:
        h = h[4:]
    out.add(h)
    out.add(f"www.{h}")


def registrable_domain_from_url(url: str) -> str:
    h = _host_from_url(url)
    return _registrable_domain(h) if h else ""


def build_dataset_externals(
    *, args: argparse.Namespace, companies: list[Any]
) -> frozenset[str]:
    """
    Build dataset_externals for UniversalExternalFilter.

    Behavior:
      - If --dataset-file is provided: derive externals from that full dataset source.
      - Else: derive externals from the current crawl list (`companies`).

    Returns:
      frozenset[str] of hosts (includes both www/non-www variants).
    """
    dataset_hosts: set[str] = set()

    dataset_file = getattr(args, "dataset_file", None)
    if dataset_file:
        try:
            # Use the same loader as your crawl list (supports file or directory)
            from extensions.load_source import load_companies_from_source

            inputs = load_companies_from_source(Path(dataset_file))
            for ci in inputs or []:
                u = getattr(ci, "url", "") or ""
                h = _host_from_url(str(u))
                if h:
                    _add_host_variants(h, dataset_hosts)

            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "[dataset_external] derived externals from dataset_file=%s hosts=%d",
                    dataset_file,
                    len(dataset_hosts),
                )

            return frozenset(dataset_hosts)

        except Exception as e:
            logger.exception(
                "[dataset_external] failed to load dataset_file=%s; falling back to crawl list: %s",
                dataset_file,
                e,
            )

    # Fallback: derive from the current crawl list
    for c in companies or []:
        u = getattr(c, "domain_url", None) or getattr(c, "url", None) or ""
        h = _host_from_url(str(u))
        if h:
            _add_host_variants(h, dataset_hosts)

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "[dataset_external] derived externals from crawl list hosts=%d",
            len(dataset_hosts),
        )

    return frozenset(dataset_hosts)
