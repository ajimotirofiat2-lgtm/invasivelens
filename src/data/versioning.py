"""
Data versioning and artifact tracking for InvasiveLens.

Computes checksums for manifests and tracks data versions to ensure
reproducibility and detect data drift.

Usage:
    python -m src.data.versioning --manifest manifests/combined_manifest.csv
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from config import MANIFEST_DIR, RESULTS_DIR
from src.data.download_utils import compute_md5, compute_sha256


class ManifestVersion:
    """Track version information for a manifest."""

    def __init__(self, manifest_path: Path | str):
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(self.manifest_path)
        self.version_file = self.manifest_path.with_suffix(".version.json")

    def compute_checksums(self) -> dict:
        """Compute MD5 and SHA256 for the manifest."""
        return {
            "md5": compute_md5(self.manifest_path),
            "sha256": compute_sha256(self.manifest_path),
        }

    def compute_data_hash(self) -> str:
        """
        Compute content-based hash of manifest data (order-independent).
        Useful for detecting data changes even if file format changes.
        """
        # Sort by all columns to make hash stable across row reorderings
        sorted_df = self.df.sort_values(by=list(self.df.columns)).reset_index(drop=True)
        content = sorted_df.to_csv(index=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_statistics(self) -> dict:
        """Compute manifest statistics."""
        return {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "n_classes": int(self.df["label"].nunique()) if "label" in self.df.columns else None,
            "n_sources": int(self.df["source"].nunique()) if "source" in self.df.columns else None,
            "n_groups": int(self.df["group"].nunique()) if "group" in self.df.columns else None,
            "columns": list(self.df.columns),
            "class_distribution": (
                self.df["label"].value_counts().to_dict() if "label" in self.df.columns else {}
            ),
            "source_distribution": (
                self.df["source"].value_counts().to_dict() if "source" in self.df.columns else {}
            ),
        }

    def create_version(self, description: str = "") -> dict:
        """
        Create a version record for this manifest.

        Returns:
            Dictionary with version metadata
        """
        import time

        checksums = self.compute_checksums()
        stats = self.get_statistics()

        version = {
            "timestamp": time.time(),
            "description": description,
            "checksums": checksums,
            "data_hash": self.compute_data_hash(),
            "statistics": stats,
        }

        return version

    def save_version(self, description: str = "") -> Path:
        """Save version metadata to .version.json file."""
        version = self.create_version(description)
        self.version_file.write_text(json.dumps(version, indent=2))
        return self.version_file

    def load_version(self) -> Optional[dict]:
        """Load version metadata if it exists."""
        if not self.version_file.exists():
            return None
        return json.loads(self.version_file.read_text())

    def verify_checksum(self) -> tuple[bool, Optional[str]]:
        """
        Verify that current manifest matches saved version checksums.

        Returns:
            (matches, error_message)
        """
        version = self.load_version()
        if version is None:
            return True, None  # No version to compare against

        current_checksums = self.compute_checksums()
        saved_checksums = version.get("checksums", {})

        if current_checksums["md5"] != saved_checksums.get("md5"):
            return False, "MD5 mismatch — manifest file has changed"

        if current_checksums["sha256"] != saved_checksums.get("sha256"):
            return False, "SHA256 mismatch — manifest file has changed"

        return True, None

    def detect_data_drift(self) -> dict:
        """
        Compare current statistics to saved version.

        Returns:
            Dictionary describing changes
        """
        version = self.load_version()
        if version is None:
            return {"status": "no_previous_version"}

        old_stats = version.get("statistics", {})
        current_stats = self.get_statistics()

        drift = {}
        for key in old_stats:
            if old_stats[key] != current_stats.get(key):
                drift[key] = {
                    "old": old_stats[key],
                    "new": current_stats[key],
                }

        return {
            "has_drift": len(drift) > 0,
            "changes": drift,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_DIR / "combined_manifest.csv"))
    parser.add_argument("--verify", action="store_true",
                        help="Verify checksum matches saved version")
    parser.add_argument("--detect-drift", action="store_true",
                        help="Detect changes from previous version")
    parser.add_argument("--update", action="store_true",
                        help="Save current manifest as new version")
    parser.add_argument("--description", default="",
                        help="Description for new version")
    args = parser.parse_args()

    manifest = ManifestVersion(args.manifest)

    if args.verify:
        matches, error = manifest.verify_checksum()
        print(f"Manifest: {args.manifest}")
        if matches:
            print("[OK] Checksum matches saved version")
        else:
            print(f"[FAIL] Checksum mismatch: {error}")

    if args.detect_drift:
        drift_info = manifest.detect_data_drift()
        print(f"Manifest: {args.manifest}")
        if drift_info.get("has_drift"):
            print("[FAIL] Data drift detected:")
            for key, change in drift_info.get("changes", {}).items():
                print(f"  {key}: {change['old']} → {change['new']}")
        else:
            print("[OK] No data drift detected")

    if args.update:
        version_file = manifest.save_version(args.description)
        stats = manifest.get_statistics()
        print(f"Manifest: {args.manifest}")
        print(f"Version saved to: {version_file}")
        print(f"\nStatistics:")
        for key, value in stats.items():
            if key != "class_distribution" and key != "source_distribution":
                print(f"  {key}: {value}")
        print(f"\nClass distribution: {dict(list(stats['class_distribution'].items())[:5])}...")

    if not (args.verify or args.detect_drift or args.update):
        # Default: show current version info
        current_version = manifest.load_version()
        if current_version:
            print(f"Manifest: {args.manifest}")
            print(f"Version info (saved):")
            print(json.dumps(current_version, indent=2, default=str))
        else:
            print(f"Manifest: {args.manifest}")
            print("No version metadata found. Run with --update to create one.")


if __name__ == "__main__":
    main()
