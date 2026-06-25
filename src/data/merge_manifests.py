"""
Combines invasoras_manifest.csv, floralens_manifest.csv, and
gbif_dl_manifest.csv (whichever exist) into one manifest, then computes
the grid_cell / group columns splits.py needs.

Floralens and gbif_dl rows often won't have real lat/lon (see notes in
those download scripts) — rows without coordinates get grouped by
source+region only (no grid refinement possible), which is coarser but
still prevents a same-platform-and-region leak; revisit if this leaves
too few groups once real data is in hand.

Usage:
    python -m src.data.merge_manifests
"""
from pathlib import Path

import pandas as pd

from config import (
    GRID_CELL_SIZE_DEG, MACRO_REGION_MAP, MANIFEST_DIR,
    MIN_DISTRICT_SIZE, MIN_GRID_SIZE,
)
from src.data.paths import manifest_path_for
from src.data.splits import assign_grid_cell, hierarchical_group


def _deduplicate_manifest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact file duplicates and repeated source/GBIF records."""
    before = len(df)
    df = df.drop_duplicates(subset=["filepath"]).copy()

    if {"source", "gbif_id"}.issubset(df.columns):
        has_key = df["gbif_id"].notna() & df["source"].notna()
        repeated_key = df.loc[has_key].duplicated(subset=["source", "gbif_id"], keep="first")
        duplicate_index = repeated_key[repeated_key].index
        if len(duplicate_index):
            df = df.drop(index=duplicate_index)

    dropped = before - len(df)
    if dropped:
        print(f"  removed {dropped} duplicate manifest rows")
    return df.reset_index(drop=True)


def merge_all_manifests(manifest_dir: Path = MANIFEST_DIR) -> pd.DataFrame:
    parts = []
    for name in [
        "invasoras_manifest.csv",
        "floralens_manifest.csv",
        "gbif_dl_manifest.csv",
        "external_manifest.csv",
        "synthetic_manifest.csv",
    ]:
        path = manifest_dir / name
        if path.exists():
            df = pd.read_csv(path)
            parts.append(df)
            print(f"  loaded {len(df)} rows from {name}")
        else:
            print(f"  (skipping {name} — not found yet)")

    if not parts:
        raise FileNotFoundError(
            f"No manifest CSVs found in {manifest_dir}. Run the download_* "
            f"scripts first."
        )
    combined = pd.concat(parts, ignore_index=True)
    combined["filepath"] = combined["filepath"].apply(manifest_path_for)
    combined = _deduplicate_manifest_rows(combined)

    has_coords = combined["latitude"].notna() & combined["longitude"].notna() \
        if "latitude" in combined.columns else pd.Series(False, index=combined.index)

    combined["grid_cell"] = "no_coords"
    if has_coords.any():
        combined.loc[has_coords, "grid_cell"] = assign_grid_cell(
            combined.loc[has_coords], cell_size_deg=GRID_CELL_SIZE_DEG
        )

    combined["group"] = hierarchical_group(
        combined, grid_col="grid_cell", district_col="region",
        macro_map=MACRO_REGION_MAP, min_grid_size=MIN_GRID_SIZE,
        min_district_size=MIN_DISTRICT_SIZE,
    )
    # Rows with no coordinates fall back to "macro:<region or source>" via
    # hierarchical_group's district-size check; tag them explicitly so
    # it's obvious in downstream analysis which rows had no real location.
    combined.loc[~has_coords, "group"] = (
        "no_coords:" + combined.loc[~has_coords, "source"].astype(str)
        + ":" + combined.loc[~has_coords, "region"].astype(str)
    )

    return combined


def main():
    print("Merging manifests...")
    combined = merge_all_manifests()
    out_path = MANIFEST_DIR / "combined_manifest.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {len(combined)} total rows to {out_path}")
    print("\nPer-class counts:")
    print(combined["label"].value_counts().to_string())
    print("\nPer-source counts:")
    print(combined["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
