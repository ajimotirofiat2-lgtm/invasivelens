"""
Wraps gbif-dl (https://github.com/plantnet/gbif-dl) to top up species
whose INVASORAS.PT/Floralens counts fall short of the per-class target —
the "heavy use of gbif-dl" referenced in the proposal's Class Selection &
Feasibility Check section.

API notes (verified against the installed package, since this sandbox
can't reach api.gbif.org to test the calls live):
  - gbif_dl.api.generate_urls(...) returns a (sync) generator of dicts
    with at least 'url' and 'label'.
  - gbif_dl.dl_async.download(items, root=...) is a regular function (NOT
    a coroutine — don't asyncio.run() it) that handles the async download
    internally and writes files directly to root/<label>/<hash>.<ext>. It
    does not hand back a list of local paths, so this script rebuilds the
    manifest by walking that directory structure afterwards.

Needs real internet access to api.gbif.org — confirmed BLOCKED in this
sandbox's egress allowlist. Run on your own machine or Colab.

Usage:
    python -m src.data.download_gbif_dl --species "Acacia dealbata" --nb-samples 300
"""
import argparse
from pathlib import Path

import gbif_dl
import pandas as pd

from config import DATA_ROOT, MANIFEST_DIR
from src.data.paths import manifest_path_for

# Restrict to the same well-curated, plant-focused datasets the Floralens
# paper prioritized, rather than all of GBIF, to keep label quality
# consistent with the rest of the project. Verify these keys are still
# current at https://www.gbif.org/dataset/search.
PREFERRED_SOURCE_DATASETS = {
    "iNaturalist": "50c9509d-22c7-4a22-a47d-8c48425ef4a7",
    "Pl@ntNet": "7a3679ef-5582-4aaa-81f0-8c2545cafc81",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def build_queries(species_list: list[str]) -> dict:
    """gbif-dl expects one dict of query params (lists expand to a product)."""
    return {
        "scientificName": species_list,
        "country": "PT",
        "datasetKey": list(PREFERRED_SOURCE_DATASETS.values()),
    }


def run_download(species_list: list[str], nb_samples: int, out_dir: Path) -> None:
    queries = build_queries(species_list)
    data_generator = gbif_dl.api.generate_urls(
        queries=queries,
        label="scientificName",
        nb_samples=nb_samples,
        split_streams_by=["datasetKey", "scientificName"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    gbif_dl.dl_async.download(data_generator, root=str(out_dir))


def build_manifest_from_disk(out_dir: Path) -> list[dict]:
    """download() organizes files as out_dir/<species>/<hash>.<ext> —
    rebuild the manifest by walking that structure rather than relying on
    a return value from download()."""
    rows = []
    for species_dir in out_dir.iterdir():
        if not species_dir.is_dir():
            continue
        for img_path in species_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append({
                "filepath": manifest_path_for(img_path),
                "label": species_dir.name.replace("_", " "),
                "region": "Unknown",  # generate_urls doesn't surface lat/lon
                                       # by default; re-query
                                       # occurrences.search per gbifID if
                                       # the geographic split needs it.
                "source": "gbif_dl",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", required=True)
    parser.add_argument("--nb-samples", type=int, default=200,
                         help="Target images per species (across the preferred datasets)")
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "gbif_dl"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    run_download(args.species, args.nb_samples, out_dir)
    rows = build_manifest_from_disk(out_dir)

    manifest_path = MANIFEST_DIR / "gbif_dl_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"\nWrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
