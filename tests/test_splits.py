"""
Tests for src/data/splits.py (v2: grid-cell grouping with hierarchical
fallback). Synthetic data mimics the real INVASORAS.PT shape: a couple of
metro areas (Lisboa, Porto) holding most of the records, scattered over a
wide enough area to span several grid cells, plus several remote districts
with only a handful of tightly clustered records each.

Run with:  pytest tests/test_splits.py -v
"""
import numpy as np
import pandas as pd

from src.data.splits import assign_grid_cell, hierarchical_group, make_splits

MACRO_MAP = {
    "Lisboa": "Centro-Litoral", "Setúbal": "Centro-Litoral",
    "Porto": "Norte-Litoral", "Braga": "Norte-Litoral", "Bragança": "Norte-Litoral",
    "Faro": "Sul", "Beja": "Sul",
    "Madeira": "Ilhas",
}

# (center_lat, center_lon, jitter_degrees) — jitter controls how spread out
# a "district" is. Lisboa/Porto get a wide jitter so they span several grid
# cells; remote districts get a tight jitter so they collapse to ~1 cell.
DISTRICT_GEO = {
    "Lisboa": (38.72, -9.14, 0.35),
    "Porto": (41.15, -8.61, 0.30),
    "Setúbal": (38.52, -8.89, 0.03),
    "Braga": (41.55, -8.42, 0.03),
    "Faro": (37.02, -7.93, 0.03),
    "Beja": (38.02, -7.86, 0.03),
    "Bragança": (41.80, -6.76, 0.03),
    "Madeira": (32.65, -16.90, 0.03),
}

CLASSES_DENSE = ["Acacia dealbata", "Cortaderia selloana", "Arundo donax",
                 "Phragmites australis", "Ammophila arenaria"]
CLASSES_SPARSE = ["Acacia dealbata", "Cortaderia selloana", "Arundo donax"]


def make_synthetic_df(seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    sizes = {"Lisboa": 800, "Porto": 500, "Setúbal": 8, "Braga": 5,
             "Faro": 12, "Beja": 3, "Bragança": 2, "Madeira": 6}
    for district, n in sizes.items():
        lat0, lon0, jitter = DISTRICT_GEO[district]
        classes = CLASSES_DENSE if n >= 30 else CLASSES_SPARSE
        labels = rng.choice(classes, size=n)
        lats = lat0 + rng.uniform(-jitter, jitter, size=n)
        lons = lon0 + rng.uniform(-jitter, jitter, size=n)
        for lab, lat, lon in zip(labels, lats, lons):
            rows.append({"filepath": "x.jpg", "label": lab, "region": district,
                         "latitude": lat, "longitude": lon, "source": "invasoras"})
    return pd.DataFrame(rows)


def build_groups(df):
    df = df.copy()
    df["grid_cell"] = assign_grid_cell(df, cell_size_deg=0.2)
    df["group"] = hierarchical_group(
        df, macro_map=MACRO_MAP, min_grid_size=15, min_district_size=30
    )
    return df


def test_dominant_district_is_split_into_multiple_groups():
    """Regression test for the v1 bug: Lisboa (60% of data) must NOT collapse
    into a single unsplittable group."""
    df = build_groups(make_synthetic_df())
    lisboa_groups = df.loc[df["region"] == "Lisboa", "group"].unique()
    assert len(lisboa_groups) > 1, (
        "Lisboa collapsed into a single group — the dominant-district bug is back."
    )
    print(f"Lisboa split into {len(lisboa_groups)} distinct spatial groups.")


def test_sparse_districts_fall_back_to_macro_region():
    df = build_groups(make_synthetic_df())
    for district in ["Beja", "Bragança", "Madeira"]:
        groups = df.loc[df["region"] == district, "group"].unique()
        assert all(g.startswith("macro:") for g in groups), (
            f"{district} should have fallen back to macro-region, got {groups}"
        )
    print("Sparse remote districts correctly fell back to macro-region groups.")


def test_no_group_split_across_folds():
    df = build_groups(make_synthetic_df())
    result = make_splits(df, n_folds=5, seed=42)
    for fold_i, (train_idx, val_idx) in enumerate(result["cv_folds"]):
        train_groups = set(df.loc[train_idx, "group"])
        val_groups = set(df.loc[val_idx, "group"])
        assert not (train_groups & val_groups), f"Fold {fold_i} leaks a group across train/val"
    print("No spatial group appears on both sides of any fold. No leakage.")


def test_fold_sizes_are_reasonably_balanced():
    """With v1 (district grouping), one fold would have been forced to
    >=60% of the data. With v2 (grid grouping), folds should be much closer
    to the ideal 1/n_folds share."""
    df = build_groups(make_synthetic_df())
    result = make_splits(df, n_folds=5, seed=42)
    n = len(df)
    shares = [size / n for size in result["fold_sizes"]]
    print("Fold shares of total data:", [f"{s:.2f}" for s in shares])
    assert max(shares) < 0.45, (
        f"A fold holds {max(shares):.2f} of the data — grouping is still too coarse."
    )


def test_mcnemar_split_is_one_specific_fold():
    df = build_groups(make_synthetic_df())
    result = make_splits(df, n_folds=5, seed=42, mcnemar_fold_index=0)
    assert result["mcnemar_split"] == result["cv_folds"][0]
    train_idx, test_idx = result["mcnemar_split"]
    print(f"McNemar comparison set: {len(test_idx)} rows, "
          f"trained on the other {len(train_idx)} rows.")


if __name__ == "__main__":
    test_dominant_district_is_split_into_multiple_groups()
    test_sparse_districts_fall_back_to_macro_region()
    test_no_group_split_across_folds()
    test_fold_sizes_are_reasonably_balanced()
    test_mcnemar_split_is_one_specific_fold()
    print("\nAll splits.py tests passed.")
