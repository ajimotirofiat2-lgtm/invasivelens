"""
Downloads the Floralens dataset (Portuguese native flora) from Zenodo and
converts it into this project's unified manifest format.

IMPORTANT — verify before running:
  1. Zenodo deposits are versioned; each version gets its own record ID.
     The proposal cites DOI 10.5281/zenodo.10679622 (record 10679622).
     A search done while building this script found a newer-looking
     record (10.5281/zenodo.15148991, "Floralens supplementary material")
     — check https://zenodo.org/search?q=floralens for the CURRENT
     version before relying on the default below.
  2. Per the Floralens paper, the Zenodo deposit is a MAPPING file (image
     labels + source image URLs + GBIF IDs), not a pre-packaged folder of
     images — this script downloads that mapping file first, then
     downloads each image from its listed URL. The exact column names in
     that mapping file could not be confirmed from this sandbox (no
     network access to zenodo.org here); inspect the downloaded file's
     header once you have it and adjust COLUMN_MAP below if needed.

Needs real internet access to zenodo.org — confirmed BLOCKED in this
sandbox's egress allowlist. Run on your own machine or Colab.

Usage:
    python -m src.data.download_floralens --species "Phragmites australis" "Ammophila arenaria"
    python -m src.data.download_floralens --record-id 15148991   # to use a newer version
"""
import argparse
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import DATA_ROOT, MANIFEST_DIR
from src.data.paths import manifest_path_for

DEFAULT_RECORD_ID = "10679622"  # the version cited in the proposal; see note above
ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"

# Adjust these once you've inspected the real mapping file's header.
COLUMN_MAP = {
    "label": "species",          # expected ground-truth species name column
    "image_url": "image_url",    # expected source image URL column
    "gbif_id": "gbif_id",        # expected GBIF identifier column, if present
}


def get_record_files(record_id: str) -> list[dict]:
    resp = requests.get(ZENODO_RECORD_API.format(record_id=record_id), timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def download_mapping_file(record_id: str, out_dir: Path) -> Path:
    files = get_record_files(record_id)
    candidates = [f for f in files if f["key"].lower().endswith((".csv", ".json", ".tsv"))]
    if not candidates:
        raise RuntimeError(
            f"No CSV/JSON/TSV file found in Zenodo record {record_id}. "
            f"Files present: {[f['key'] for f in files]}"
        )
    # Prefer a file with "mapping", "metadata", or "dataset" in the name if
    # there are several candidates.
    candidates.sort(key=lambda f: ("map" not in f["key"].lower()
                                    and "data" not in f["key"].lower()))
    target = candidates[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / target["key"]

    print(f"Downloading mapping file: {target['key']} ...")
    resp = requests.get(target["links"]["self"], timeout=60)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return local_path


def load_mapping(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return pd.read_json(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def download_images(df: pd.DataFrame, species_list: list[str], out_dir: Path) -> list[dict]:
    rows = []
    df = df[df[COLUMN_MAP["label"]].isin(species_list)]
    for _, rec in tqdm(df.iterrows(), total=len(df), desc="floralens", unit="img"):
        species = rec[COLUMN_MAP["label"]]
        url = rec[COLUMN_MAP["image_url"]]
        species_dir = out_dir / species.replace(" ", "_")
        species_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(url).name or f"{abs(hash(url))}.jpg"
        filepath = species_dir / filename

        if not filepath.exists():
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                filepath.write_bytes(resp.content)
            except requests.RequestException as exc:
                print(f"  skip {url}: {exc}")
                continue

        rows.append({
            "filepath": manifest_path_for(filepath),
            "label": species,
            "region": "Unknown",  # Floralens prioritizes curated sources over
                                   # geolocation; cross-reference gbif_id with
                                   # GBIF occurrence/search if coordinates are
                                   # needed for the geographic split.
            "source": "floralens",
            "gbif_id": rec.get(COLUMN_MAP["gbif_id"]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", required=True)
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "floralens"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    mapping_path = download_mapping_file(args.record_id, out_dir / "_mapping")
    df = load_mapping(mapping_path)

    missing_cols = set(COLUMN_MAP.values()) - set(df.columns)
    if missing_cols:
        raise RuntimeError(
            f"Expected columns {missing_cols} not found in {mapping_path}. "
            f"Actual columns: {list(df.columns)}. Update COLUMN_MAP at the "
            f"top of this script to match."
        )

    rows = download_images(df, args.species, out_dir)
    manifest_path = MANIFEST_DIR / "floralens_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"\nWrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
