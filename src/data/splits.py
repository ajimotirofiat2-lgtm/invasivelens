"""
Geographic stratification for InvasiveLens — v2.

WHY THIS CHANGED FROM v1 (kept here as a paper trail, not just for show):
v1 grouped strictly by Portuguese administrative district. Testing it on
data shaped like the real INVASORAS.PT distribution (~60% of all records
in the Lisboa district alone) showed the obvious-in-hindsight problem:
a group-based k-fold CANNOT split a single group across folds, so if one
district holds 60% of the data, at least one fold is forced to be >=60%
of the dataset. District-level grouping doesn't actually solve spatial
leakage at any usable fold balance once one district dominates.

v2 groups by spatial grid cell (derived from lat/lon, ~20km cells by
default), with a fallback hierarchy for sparse cells:
    grid cell  ->  district  ->  macro-region (Norte/Centro/Sul/Ilhas)  ->  "Other"
Dense areas like metropolitan Lisbon span many grid cells (so they CAN be
split across folds), while remote single-observation locations fall back
to coarser, still-spatially-meaningful groups. This is the standard
"spatial blocking" approach used in spatial cross-validation literature.

McNemar's test: rather than carving out a second, separate held-out set
(which has its own overshoot problems for the same reason above), fold 0
of the same grouped k-fold split is designated as the fixed comparison
set. Both compared models are trained on folds 1-4 and evaluated on fold
0 for McNemar's; all 5 folds are used (in rotation) for the primary
accuracy / macro-F1 / per-class metrics.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_grid_cell(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    cell_size_deg: float = 0.2,
) -> pd.Series:
    """~0.2 degrees is roughly 20km at Portugal's latitude — coarse enough
    that a single plant population won't span a fold boundary by accident,
    fine enough that a whole metro area spans many cells."""
    lat_bin = (df[lat_col] // cell_size_deg) * cell_size_deg
    lon_bin = (df[lon_col] // cell_size_deg) * cell_size_deg
    return "grid_" + lat_bin.round(2).astype(str) + "_" + lon_bin.round(2).astype(str)


def hierarchical_group(
    df: pd.DataFrame,
    grid_col: str = "grid_cell",
    district_col: str = "region",
    macro_map: Optional[dict] = None,
    min_grid_size: int = 15,
    min_district_size: int = 30,
) -> pd.Series:
    """
    Returns one group label per row, falling back from grid cell -> district
    -> macro-region -> 'Other' depending on how much data backs each level.
    """
    macro_map = macro_map or {}
    grid_counts = df[grid_col].value_counts()
    district_counts = df[district_col].value_counts()

    def pick(row):
        if grid_counts.get(row[grid_col], 0) >= min_grid_size:
            return f"grid:{row[grid_col]}"
        if district_counts.get(row[district_col], 0) >= min_district_size:
            return f"district:{row[district_col]}"
        macro = macro_map.get(row[district_col], "Other")
        return f"macro:{macro}"

    return df.apply(pick, axis=1)


def make_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "group",
    n_folds: int = 5,
    seed: int = 42,
    mcnemar_fold_index: int = 0,
) -> dict:
    """
    df must already have a `group_col` (typically the output of
    hierarchical_group). Returns:
      - 'cv_folds': list of (train_idx, val_idx) across n_folds, for primary
        metrics (average accuracy / macro-F1 / per-class P&R over folds).
      - 'mcnemar_split': the (train_idx, test_idx) pair for whichever fold
        is designated the fixed comparison set — use this, and only this,
        for McNemar's test between two models.
      - 'fold_sizes': row count per fold, so you can sanity-check balance
        before trusting the averaged metrics.
    """
    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_folds = list(sgkf.split(df, df[label_col], groups=df[group_col]))

    fold_sizes = [len(val_idx) for _, val_idx in cv_folds]

    return {
        "cv_folds": cv_folds,
        "mcnemar_split": cv_folds[mcnemar_fold_index],
        "mcnemar_fold_index": mcnemar_fold_index,
        "fold_sizes": fold_sizes,
    }
