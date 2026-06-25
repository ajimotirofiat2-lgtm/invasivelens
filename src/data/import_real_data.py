"""
Import real datasets and build training manifests for invasive plant detection.

This module processes:
- GBIF occurrence data (species distribution records)
- INVASORAS database (native vs invasive plant records)
- Floralens visual data (plant images)

Creates unified manifests for training ML models.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from config import CANDIDATE_PAIRS, MANIFEST_DIR, RESULTS_DIR

# Target invasive species and their native comparisons
TARGET_SPECIES = {
    sp["name"]: sp["native"] for sp in CANDIDATE_PAIRS if sp["native"] is not None
}

# Mapping common name variants to scientific names
SPECIES_ALIASES = {
    # Acacia dealbata
    "Acacia dealbata": ["silver wattle", "mimosa", "acacia dealbata"],
    "Acacia longifolia": ["longleaf wattle", "acacia longifolia"],
    # Ammophila arenaria
    "Ammophila arenaria": ["marram grass", "beach grass", "ammophila arenaria"],
    # Arundo donax
    "Arundo donax": ["giant reed", "arundo", "arundo donax"],
    # Cortaderia selloana
    "Cortaderia selloana": ["pampas grass", "cortaderia", "cortaderia selloana"],
    # Phragmites australis
    "Phragmites australis": ["common reed", "reed", "phragmites australis"],
}


class GBIFImporter:
    """Import occurrence records from GBIF CSV."""

    def __init__(self, csv_path: Path | str):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, sep="\t", low_memory=False)
        print(f"Loaded {len(self.df)} GBIF records")

    def filter_plants(self) -> "GBIFImporter":
        """Keep only plant records."""
        if "phylum" in self.df.columns:
            self.df = self.df[self.df["phylum"] == "Tracheophyta"]
            print(f"Filtered to {len(self.df)} plant records")
        return self

    def filter_target_species(self) -> "GBIFImporter":
        """Keep only target invasive and native species."""
        target_scientific = set(TARGET_SPECIES.keys()) | set(TARGET_SPECIES.values())
        if "scientificName" in self.df.columns:
            # Normalize names and match
            self.df = self.df[
                self.df["scientificName"].str.lower().isin(
                    {s.lower() for s in target_scientific}
                )
            ]
            print(f"Filtered to {len(self.df)} target species records")
        return self

    def get_manifest_rows(self) -> List[Dict]:
        """Convert to manifest format."""
        rows = []

        for _, record in self.df.iterrows():
            species = record.get("scientificName", "").strip()
            if not species:
                continue

            # Determine if invasive or native
            invasive_status = "invasive" if species in TARGET_SPECIES else "native"

            rows.append({
                "filepath": f"gbif://{record['gbifID']}",  # Virtual reference
                "label": species,
                "source": "gbif",
                "region": record.get("stateProvince", "unknown"),
                "latitude": record.get("decimalLatitude"),
                "longitude": record.get("decimalLongitude"),
                "group": f"{record.get('stateProvince', 'unknown')}_{int(record.get('decimalLatitude', 0))}",
                "status": invasive_status,
            })

        print(f"Generated {len(rows)} manifest rows from GBIF")
        return rows


class InvasorasImporter:
    """Import occurrence records from INVASORAS Darwin Core Archive."""

    def __init__(self, csv_path: Path | str):
        self.csv_path = Path(csv_path)
        try:
            self.df = pd.read_csv(self.csv_path, sep="\t", low_memory=False)
            print(f"Loaded {len(self.df)} INVASORAS records")
        except Exception as e:
            print(f"Error loading INVASORAS: {e}")
            self.df = pd.DataFrame()

    def filter_target_species(self) -> "InvasorasImporter":
        """Keep only target species."""
        if "scientificName" not in self.df.columns:
            return self

        target_scientific = set(TARGET_SPECIES.keys()) | set(TARGET_SPECIES.values())
        self.df = self.df[
            self.df["scientificName"].str.lower().isin(
                {s.lower() for s in target_scientific}
            )
        ]
        print(f"Filtered to {len(self.df)} target species in INVASORAS")
        return self

    def get_manifest_rows(self) -> List[Dict]:
        """Convert to manifest format."""
        rows = []

        for _, record in self.df.iterrows():
            species = record.get("scientificName", "").strip()
            if not species:
                continue

            # INVASORAS tracks invasive status explicitly
            invasive_status = "invasive" if species in TARGET_SPECIES else "native"

            rows.append({
                "filepath": f"invasoras://{record.get('occurrenceID', '')}",
                "label": species,
                "source": "invasoras",
                "region": record.get("stateProvince", "unknown"),
                "latitude": record.get("decimalLatitude"),
                "longitude": record.get("decimalLongitude"),
                "group": f"{record.get('stateProvince', 'unknown')}_{int(record.get('decimalLatitude', 0))}",
                "status": invasive_status,
            })

        print(f"Generated {len(rows)} manifest rows from INVASORAS")
        return rows


def create_occurrence_manifest():
    """
    Build a manifest of occurrence records (non-image) for spatial analysis.

    This is useful for:
    - Spatial distribution analysis
    - Risk mapping
    - Ecological modeling
    """
    rows = []

    # Import GBIF data
    gbif_0067492 = Path("C:/Users/USER/Downloads/0067492-260519110011954/0067492-260519110011954.csv")
    if gbif_0067492.exists():
        gbif = GBIFImporter(gbif_0067492).filter_plants().filter_target_species()
        rows.extend(gbif.get_manifest_rows())

    gbif_0067463 = Path("C:/Users/USER/Downloads/0067463-260519110011954/0067463-260519110011954.csv")
    if gbif_0067463.exists():
        gbif = GBIFImporter(gbif_0067463).filter_plants().filter_target_species()
        rows.extend(gbif.get_manifest_rows())

    # Import INVASORAS data
    invasoras_csv = Path("C:/Users/USER/Downloads/dwca-invasoras-v2.10/occurrence.txt")
    if invasoras_csv.exists():
        inv = InvasorasImporter(invasoras_csv).filter_target_species()
        rows.extend(inv.get_manifest_rows())

    # Save manifest
    if rows:
        df = pd.DataFrame(rows)
        manifest_path = MANIFEST_DIR / "occurrence_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, index=False)
        print(f"\n[OK] Saved {len(df)} occurrence records to {manifest_path}")

        # Summary statistics
        print("\nOccurrence Manifest Summary:")
        print(f"  Total records: {len(df)}")
        print(f"  Species: {df['label'].nunique()}")
        print(f"  Regions: {df['region'].nunique()}")
        print(f"  Invasive vs Native: {df['status'].value_counts().to_dict()}")
        print(f"\n  By species:")
        for sp, count in df["label"].value_counts().items():
            print(f"    {sp}: {count}")

        return manifest_path
    else:
        print("[WARN] No occurrence records found")
        return None


if __name__ == "__main__":
    create_occurrence_manifest()
