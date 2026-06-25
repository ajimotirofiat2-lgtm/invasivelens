#!/usr/bin/env python3
"""
Analyze available datasets and create import plan.
"""
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def analyze_floralens(base_path: Path) -> Dict:
    """Analyze Floralens dataset structure."""
    print("\n=== FLORALENS ===")
    species_dirs = list(base_path.glob("*/"))
    species_counts = {}
    
    for species_dir in species_dirs:
        if species_dir.is_dir():
            images = list(species_dir.glob("*.jpg")) + list(species_dir.glob("*.JPG"))
            species_name = species_dir.name
            species_counts[species_name] = len(images)
            print(f"  {species_name}: {len(images)} images")
    
    total = sum(species_counts.values())
    print(f"Total: {total} images across {len(species_counts)} species")
    return species_counts


def analyze_gbif_csv(csv_path: Path) -> Dict:
    """Analyze GBIF CSV dataset."""
    print("\n=== GBIF CSV ===")
    df = pd.read_csv(csv_path, sep="\t", low_memory=False)
    
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns[:10])}")
    
    # Check for species names
    if "scientificName" in df.columns:
        species = df["scientificName"].value_counts()
        print(f"  Species: {len(species)}")
        for sp, count in species.head(5).items():
            print(f"    {sp}: {count}")
    
    # Check for image URLs
    if "multimedia" in df.columns:
        has_images = df["multimedia"].notna().sum()
        print(f"  Rows with images: {has_images}")
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "columns_list": list(df.columns)
    }


def analyze_dwca_invasoras(base_path: Path) -> Dict:
    """Analyze Darwin Core Archive (INVASORAS)."""
    print("\n=== INVASORAS DWCA ===")
    files = list(base_path.rglob("*.csv"))
    
    for csv_file in files:
        print(f"  File: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            print(f"    Rows: {len(df)}")
            print(f"    Columns: {list(df.columns[:5])}")
        except Exception as e:
            print(f"    Error: {e}")
    
    return {"files": len(files)}


def main():
    """Analyze all available datasets."""
    base_dir = Path("C:/Users/USER/Downloads")
    
    floralens_path = base_dir / "floralens-data-v0.3"
    gbif_0067492 = base_dir / "0067492-260519110011954/0067492-260519110011954.csv"
    gbif_0067463 = base_dir / "0067463-260519110011954/0067463-260519110011954.csv"
    invasoras_path = base_dir / "dwca-invasoras-v2.10"
    
    results = {}
    
    if floralens_path.exists():
        results["floralens"] = analyze_floralens(floralens_path)
    
    if gbif_0067492.exists():
        results["gbif_0067492"] = analyze_gbif_csv(gbif_0067492)
    
    if gbif_0067463.exists():
        results["gbif_0067463"] = analyze_gbif_csv(gbif_0067463)
    
    if invasoras_path.exists():
        results["invasoras"] = analyze_dwca_invasoras(invasoras_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
