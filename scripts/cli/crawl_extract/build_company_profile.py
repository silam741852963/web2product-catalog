from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from extensions.company_profile.builder import build_company_profile_for_company
from extensions.io import output_paths

runner_logger = logging.getLogger("company_profile")


def _setup_root_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _is_company_dir(p: Path) -> bool:
    # Must match your directory contract
    return p.is_dir() and (p / "product").is_dir() and (p / "metadata").is_dir()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build company_profile.json from <output_root>/<company_id>/product/*.json"
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="(Optional) Output root directory. If empty, uses extensions.io.output_paths.get_output_root().",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--company-id", type=str, help="Run for one company folder name")
    g.add_argument(
        "--all", action="store_true", help="Run all company folders under output root"
    )

    p.add_argument(
        "--embed-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Embedding device used for clustering/dedup (always on): cpu (default), cuda (NVIDIA), mps (Apple).",
    )

    p.add_argument(
        "--embed-text-output",
        action="store_true",
        help="If set, also embed company_profile.md and per-offering text into company_profile.json. Default: off.",
    )

    p.add_argument("--log-level", type=str, default="INFO")

    args = p.parse_args(argv)
    _setup_root_logger(args.log_level)

    # Resolve out root
    output_root = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir or "").strip()
        else Path(output_paths.get_output_root()).resolve()
    )

    if not output_root.exists():
        runner_logger.error("output_root does not exist: %s", output_root)
        return 1

    # ✅ CRITICAL FIX:
    # Make output_root the canonical global root so every downstream module
    # (output_paths.ensure_company_dirs, crawl_state.load_*, etc.) uses it.
    output_paths.ensure_output_root(output_root)

    if args.all:
        company_ids = sorted(
            [d.name for d in output_root.iterdir() if _is_company_dir(d)]
        )
        runner_logger.info(
            "mode=all output_root=%s companies=%d", output_root, len(company_ids)
        )
        if not company_ids:
            runner_logger.warning(
                "No valid company folders found under %s", output_root
            )
            return 0
    else:
        company_id = (args.company_id or "").strip()
        if not company_id:
            runner_logger.error("company_id is empty")
            return 1
        company_ids = [company_id]

    ok_all = True
    for cid in company_ids:
        ok = build_company_profile_for_company(
            company_id=cid,
            outputs_dir=output_root,  # still passed for clarity/sanity
            embed_device=args.embed_device,
            embed_text_output=bool(args.embed_text_output),
        )
        if not ok:
            ok_all = False

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
