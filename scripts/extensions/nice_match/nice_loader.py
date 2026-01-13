from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class NiceRow:
    # token_id is taken from "Basic No. or Place / № de base ou endroit"
    token_id: str

    cl: str
    x: str
    prop_no: str
    action_en: str
    en_goods_2024: str


_REQUIRED_COLS = [
    "Cl.",
    "Basic No. or Place / № de base ou endroit",
    "X",
    "Prop. No./№",
    "Action EN",
    "EN - Goods and Services NCL (12-2024)",
]


def load_nice_rows(xlsx_path: str | Path, *, sheet: int | str = 0) -> list[NiceRow]:
    p = Path(xlsx_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"NICE xlsx not found: {p}")

    df = pd.read_excel(p, sheet_name=sheet, engine="openpyxl")

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if len(missing) != 0:
        raise RuntimeError(f"NICE xlsx missing required columns: {missing}")

    if len(df) == 0:
        raise RuntimeError("NICE xlsx has no rows.")

    # Skip the first data row (explanatory row).
    df = df.iloc[1:].copy()

    def _s(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""
        return str(v).strip()

    rows: list[NiceRow] = []
    for _, r in df.iterrows():
        basic = _s(r["Basic No. or Place / № de base ou endroit"])
        if basic == "":
            continue

        # Skip explanatory entries mid-way.
        if basic == "Class Heading and Explanatory Note":
            continue

        en_goods = _s(r["EN - Goods and Services NCL (12-2024)"])
        if en_goods == "":
            continue

        token_id = (
            basic  # explicit: token_id is Basic No. or Place / № de base ou endroit
        )

        rows.append(
            NiceRow(
                token_id=token_id,
                cl=_s(r["Cl."]),
                x=_s(r["X"]),
                prop_no=_s(r["Prop. No./№"]),
                action_en=_s(r["Action EN"]),
                en_goods_2024=en_goods,
            )
        )

    if len(rows) == 0:
        raise RuntimeError("After filtering, NICE rows are empty.")
    return rows
