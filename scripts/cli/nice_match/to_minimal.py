from __future__ import annotations

import argparse
import csv
from pathlib import Path


_REQUIRED_COLS = ("bvd_id", "sentence_id", "token_id")
_OUT_COLS = list(_REQUIRED_COLS)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nice_match_to_minimal",
        description="Convert NICE matches CSV to minimal columns: bvd_id,sentence_id,token_id",
    )
    p.add_argument(
        "--in-csv", required=True, help="Input matches CSV (full or already minimal)."
    )
    p.add_argument("--out-csv", required=True, help="Output minimal matches CSV.")
    return p.parse_args()


def _validate_header(fieldnames: list[str] | None, in_path: Path) -> list[str]:
    if fieldnames is None:
        raise RuntimeError(f"CSV has no header: {in_path}")

    missing = [c for c in _REQUIRED_COLS if c not in fieldnames]
    if missing:
        raise RuntimeError(
            f"Input CSV missing required columns {missing}. "
            f"Available columns: {fieldnames}. File: {in_path}"
        )

    return fieldnames


def main() -> int:
    ns = _parse_args()

    in_path = Path(ns.in_csv).expanduser().resolve()
    out_path = Path(ns.out_csv).expanduser().resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"--in-csv not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        _validate_header(reader.fieldnames, in_path)

        with out_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=_OUT_COLS)
            writer.writeheader()

            for row in reader:
                writer.writerow(
                    {
                        "bvd_id": (row.get("bvd_id") or "").strip(),
                        "sentence_id": (row.get("sentence_id") or "").strip(),
                        "token_id": (row.get("token_id") or "").strip(),
                    }
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
