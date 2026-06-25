"""Manifest quality checks for data readiness.

This does not decide whether a model is good. It answers the earlier
question: is the dataset clean enough that model metrics are worth taking
seriously?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import MANIFEST_DIR, RESULTS_DIR
from src.data.paths import resolve_manifest_path


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def audit_manifest(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    n_rows = len(df)

    missing_columns = sorted({"filepath", "label", "region", "source"} - set(df.columns))
    if missing_columns:
        return {
            "manifest": str(manifest_path),
            "n_rows": n_rows,
            "missing_required_columns": missing_columns,
            "issues": [f"Missing required columns: {missing_columns}"],
        }

    resolved_paths = df["filepath"].apply(lambda p: resolve_manifest_path(p, manifest_path))
    exists_mask = resolved_paths.apply(Path.exists)

    has_lat_lon = {"latitude", "longitude"}.issubset(df.columns)
    if has_lat_lon:
        coord_mask = df["latitude"].notna() & df["longitude"].notna()
    else:
        coord_mask = pd.Series(False, index=df.index)

    duplicate_filepaths = int(df["filepath"].duplicated().sum())
    duplicate_source_gbif = 0
    if {"source", "gbif_id"}.issubset(df.columns):
        keyed = df[df["source"].notna() & df["gbif_id"].notna()]
        duplicate_source_gbif = int(keyed.duplicated(subset=["source", "gbif_id"]).sum())

    class_counts = _value_counts(df["label"])
    source_counts = _value_counts(df["source"])
    group_counts = _value_counts(df["group"]) if "group" in df.columns else {}
    largest_group = max(group_counts.values(), default=0)
    smallest_class = min(class_counts.values(), default=0)

    issues = []
    missing_files = int((~exists_mask).sum())
    no_coords = int((~coord_mask).sum())
    no_coords_ratio = float(no_coords / n_rows) if n_rows else 0.0
    largest_group_share = float(largest_group / n_rows) if n_rows else 0.0

    if missing_files:
        issues.append(f"{missing_files} manifest paths do not exist on disk.")
    if duplicate_filepaths:
        issues.append(f"{duplicate_filepaths} duplicate filepath rows found.")
    if duplicate_source_gbif:
        issues.append(f"{duplicate_source_gbif} duplicate source/gbif_id rows found.")
    if no_coords_ratio > 0.25:
        issues.append(f"{no_coords_ratio:.0%} of rows lack latitude/longitude.")
    if largest_group_share > 0.30:
        issues.append(f"Largest validation group holds {largest_group_share:.0%} of rows.")
    if smallest_class and smallest_class < 100:
        issues.append(f"Smallest class has only {smallest_class} images.")

    return {
        "manifest": str(manifest_path),
        "n_rows": int(n_rows),
        "n_classes": int(df["label"].nunique()),
        "n_sources": int(df["source"].nunique()),
        "n_groups": int(df["group"].nunique()) if "group" in df.columns else None,
        "missing_required_columns": [],
        "missing_files": missing_files,
        "duplicate_filepaths": duplicate_filepaths,
        "duplicate_source_gbif": duplicate_source_gbif,
        "rows_with_coordinates": int(coord_mask.sum()),
        "rows_without_coordinates": no_coords,
        "rows_without_coordinates_ratio": no_coords_ratio,
        "largest_group_share": largest_group_share,
        "smallest_class_count": int(smallest_class),
        "class_counts": class_counts,
        "source_counts": source_counts,
        "top_groups": dict(sorted(group_counts.items(), key=lambda item: item[1], reverse=True)[:10]),
        "issues": issues,
    }


def _print_report(report: dict) -> None:
    print(f"Manifest: {report['manifest']}")
    print(f"Rows: {report['n_rows']}  Classes: {report.get('n_classes')}  Sources: {report.get('n_sources')}")
    if report.get("n_groups") is not None:
        print(f"Groups: {report['n_groups']}  Largest group share: {report['largest_group_share']:.1%}")
    print(f"Missing files: {report.get('missing_files', 0)}")
    print(f"Rows without coordinates: {report.get('rows_without_coordinates', 0)} "
          f"({report.get('rows_without_coordinates_ratio', 0):.1%})")

    print("\nClass counts:")
    for label, count in report.get("class_counts", {}).items():
        print(f"  {label}: {count}")

    print("\nSource counts:")
    for source, count in report.get("source_counts", {}).items():
        print(f"  {source}: {count}")

    if report.get("issues"):
        print("\nIssues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
    else:
        print("\nNo blocking data-quality issues found.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_DIR / "combined_manifest.csv"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--write", default=str(RESULTS_DIR / "data_quality_report.json"))
    args = parser.parse_args()

    report = audit_manifest(args.manifest)
    out_path = Path(args.write)
    out_path.write_text(json.dumps(report, indent=2))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
        print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
