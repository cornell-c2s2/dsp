# convert_xlsx_to_csv.py
# Converts your custom Excel files (with ARM/LEG/NECK groups and ax/ay/az/gx/gy/gz under each)
# into CSV files suitable for the classifier.
#
# Usage examples:
#   python convert_xlsx_to_csv.py --root data --group ARM
#   python convert_xlsx_to_csv.py --root data --group ALL
#
# Output: For each .xlsx in the tree, writes a .csv next to it.
#   - If --group ARM/LEG/NECK: columns -> ax,ay,az,gx,gy,gz
#   - If --group ALL: columns -> ax_arm,ay_arm,...,gz_arm, ax_leg,...,gz_leg, ax_neck,...,gz_neck

import os
import re
import argparse
import pandas as pd
from typing import Dict, List, Tuple

AXES = ["ax","ay","az","gx","gy","gz"]
GROUPS_CANON = ["ARM","LEG","NECK"]


def _detect_header_rows(df: pd.DataFrame) -> Tuple[int, int]:
    """Return (group_row_idx, axis_row_idx) in the Excel sheet.
    axis_row: the row that contains tokens like ax/ay/az/gx/gy/gz in repeated fashion
    group_row: typically one row above axis_row (contains ARM/LEG/NECK over blocks)
    """
    axis_row = None
    # search first ~20 rows for axis labels
    for r in range(min(20, len(df))):
        vals = df.iloc[r].astype(str).str.strip().str.lower()
        hits = sum(v in set(AXES) for v in vals)
        if hits >= 3:  # enough to believe this is the axis header row
            axis_row = r
            break
    if axis_row is None:
        raise ValueError("Could not locate an axis header row (ax/ay/az/gx/gy/gz)")

    group_row = max(0, axis_row - 1)
    return group_row, axis_row


def _multiindex_columns(raw: pd.DataFrame, group_row: int, axis_row: int) -> List[str]:
    # Build names like ARM_ax, LEG_gx, etc.; later we will canonicalize to ax_arm, ...
    g = raw.iloc[group_row].copy()
    a = raw.iloc[axis_row].copy()

    # forward-fill across columns to propagate group names
    g = g.fillna(method="ffill")

    cols = []
    for gg, aa in zip(g, a):
        if pd.isna(aa):
            cols.append(None)
            continue
        gg_s = str(gg).strip().upper() if not pd.isna(gg) else ""  # e.g., ARM/LEG/NECK
        aa_s = str(aa).strip().lower()  # e.g., ax
        if aa_s in AXES and gg_s:
            cols.append(f"{aa_s}_{gg_s}")
        elif aa_s in AXES:
            # no group name; default to ARM if absent
            cols.append(f"{aa_s}_ARM")
        else:
            cols.append(None)
    return cols


def _read_excel_groups(path: str) -> Dict[str, pd.DataFrame]:
    """Parse Excel, return dict like {"ARM": df6, "LEG": df6, "NECK": df6},
    where each df6 has columns ax,ay,az,gx,gy,gz when available. Missing groups omitted."""
    raw = pd.read_excel(path, header=None)
    group_row, axis_row = _detect_header_rows(raw)
    colnames = _multiindex_columns(raw, group_row, axis_row)

    data = raw.iloc[axis_row+1:].reset_index(drop=True)
    data.columns = colnames
    # keep only recognized columns
    keep = [c for c in data.columns if c is not None]
    data = data[keep]

    # coerce numerics
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(how="all")

    out: Dict[str, pd.DataFrame] = {}
    for G in GROUPS_CANON:
        colsG = [f"{ax}_{G}" for ax in AXES]
        avail = [c for c in colsG if c in data.columns]
        if len(avail) >= 4:  # require at least some channels
            sub = data.reindex(columns=colsG).copy()
            sub.columns = AXES  # rename to ax..gz for this group
            out[G] = sub
    return out


def convert_one(path: str, group: str) -> str:
    groups = _read_excel_groups(path)
    if group == "ALL":
        # concat groups horizontally with suffixed names
        pieces = []
        for G, dfG in groups.items():
            dfGi = dfG.copy()
            dfGi.columns = [f"{c}_{G.lower()}" for c in dfGi.columns]
            pieces.append(dfGi)
        if not pieces:
            raise ValueError(f"{path}: no recognizable IMU groups found (ARM/LEG/NECK)")
        out = pd.concat(pieces, axis=1)
    else:
        G = group.upper()
        if G not in groups:
            raise ValueError(f"{path}: requested group {group} not found; available: {list(groups.keys())}")
        out = groups[G]

    csv_path = re.sub(r"\.xlsx$", ".csv", path, flags=re.IGNORECASE)
    out.to_csv(csv_path, index=False)
    return csv_path


def main_convert():
    ap = argparse.ArgumentParser(description="Convert custom IMU Excel files to CSV")
    ap.add_argument("--root", default="data", help="Root directory to scan for .xlsx")
    ap.add_argument("--group", default="ARM", choices=["ARM","LEG","NECK","ALL"],
                    help="Which group of sensors to export. ALL exports all groups with suffixes.")
    args = ap.parse_args()

    written = 0
    for root, _, files in os.walk(args.root):
        for fn in files:
            if fn.lower().endswith(".xlsx"):
                path = os.path.join(root, fn)
                try:
                    outp = convert_one(path, args.group)
                    print(f"✓ {fn} -> {os.path.basename(outp)}")
                    written += 1
                except Exception as e:
                    print(f"! {fn}: {e}")
    print(f"Done. Converted {written} Excel file(s).")


if __name__ == "__main__":
    main_convert()