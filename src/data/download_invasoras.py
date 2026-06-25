"""
Downloads occurrence records (with images) for the INVASORAS.PT dataset
specifically, restricted to a target species list, and writes them into
this project's unified manifest format (filepath, label, region, source,
latitude, longitude).

Needs real internet access to api.gbif.org — confirmed BLOCKED in this
sandbox's egress allowlist (HTTP 403). Run on your own machine or Colab.

Usage:
    python -m src.data.download_invasoras --species "Acacia dealbata" "Cortaderia selloana"
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import DATA_ROOT, MANIFEST_DIR
from src.data.inspect_counts import INVASORAS_DATASET_KEY
from src.data.paths import manifest_path_for

GBIF_OCCURRENCE_SEARCH = "https://api.gbif.org/v1/occurrence/search"


def fetch_occurrence_records(species: str, page_limit: int = 300) -> list[dict]:
    """Pages through GBIF occurrence/search for one species within the
    INVASORAS.PT dataset, restricted to records that have both an image
    and coordinates (both required for this project)."""
    records, offset = [], 0
    while True:
        params = {
            "scientificName": species,
            "datasetKey": INVASORAS_DATASET_KEY,
            "country": "PT",
            "hasCoordinate": "true",
            "mediaType": "StillImage",
            "limit": page_limit,
            "offset": offset,
        }
        resp = requests.get(GBIF_OCCURRENCE_SEARCH, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        records.extend(payload.get("results", []))
        if payload.get("endOfRecords", True):
            break
        offset += page_limit
        time.sleep(0.2)  # be polite to the API
    return records


def download_images_for_species(species: str, records: list[dict], out_dir: Path) -> list[dict]:
    """Downloads each record's first image to disk; returns manifest rows."""
    species_dir = out_dir / species.replace(" ", "_")
    species_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for rec in tqdm(records, desc=species, unit="img"):
        media = rec.get("media", [])
        image_url = next((m["identifier"] for m in media if m.get("type") == "StillImage"), None)
        if not image_url:
            continue
        lat, lon = rec.get("decimalLatitude"), rec.get("decimalLongitude")
        if lat is None or lon is None:
            continue

        gbif_id = rec.get("gbifID", rec.get("key"))
        ext = Path(image_url).suffix or ".jpg"
        filepath = species_dir / f"{gbif_id}{ext}"

        if not filepath.exists():
            try:
                img_resp = requests.get(image_url, timeout=20)
                img_resp.raise_for_status()
                filepath.write_bytes(img_resp.content)
            except requests.RequestException as exc:
                print(f"  skip {gbif_id}: {exc}")
                continue

        rows.append({
            "filepath": manifest_path_for(filepath),
            "label": species,
            "region": rec.get("stateProvince") or rec.get("county") or "Unknown",
            "latitude": lat,
            "longitude": lon,
            "source": "invasoras",
            "gbif_id": gbif_id,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", required=True)
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "invasoras"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    all_rows = []
    for species in args.species:
        print(f"Fetching occurrence records for {species} ...")
        records = fetch_occurrence_records(species)
        print(f"  {len(records)} candidate records with image + coordinates")
        all_rows.extend(download_images_for_species(species, records, out_dir))

    manifest_path = MANIFEST_DIR / "invasoras_manifest.csv"
    pd.DataFrame(all_rows).to_csv(manifest_path, index=False)
    print(f"\nWrote {len(all_rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
