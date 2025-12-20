from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from extensions.company_profile_builder import build_company_profile_for_company

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
        description="Build company_profile.json from outputs/<company_id>/product/*.json"
    )
    p.add_argument("--outputs-dir", type=str, default="outputs")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--company-id", type=str, help="Run for one company folder name")
    g.add_argument(
        "--all", action="store_true", help="Run all company folders under outputs/"
    )

    # The only embedding-related flag:
    p.add_argument(
        "--embed-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Embedding device: cpu (default), cuda (NVIDIA), mps (Apple).",
    )

    p.add_argument("--log-level", type=str, default="INFO")

    args = p.parse_args(argv)
    _setup_root_logger(args.log_level)

    outputs_dir = Path(args.outputs_dir).resolve()
    if not outputs_dir.exists():
        runner_logger.error("outputs_dir does not exist: %s", outputs_dir)
        return 1

    if args.all:
        company_ids = sorted(
            [d.name for d in outputs_dir.iterdir() if _is_company_dir(d)]
        )
        runner_logger.info(
            "mode=all outputs_dir=%s companies=%d", outputs_dir, len(company_ids)
        )
        if not company_ids:
            runner_logger.warning(
                "No valid company folders found under %s", outputs_dir
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
            outputs_dir=outputs_dir,
            company_id=cid,
            embed_device=args.embed_device,
        )
        if not ok:
            ok_all = False

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
