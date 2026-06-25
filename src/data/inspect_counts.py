"""
The Week 1-2 timeline task: pull REAL per-species counts before finalising
the class list, instead of trusting the proposal's 10,000-20,000 image
estimate. See the "Class Selection & Feasibility Check" section of the
proposal for why this matters (INVASORAS.PT totals ~9,478 records across
~43 species, very unevenly distributed; Floralens caps native species at
50-200 images by design).

Needs real internet access to api.gbif.org — confirmed BLOCKED in this
sandbox's egress allowlist (HTTP 403, "Host not in allowlist"). Run this
on your own machine or Colab, not in this container.

Usage:
    python -m src.data.inspect_counts
    python -m src.data.inspect_counts --species "Acacia dealbata" "Cortaderia selloana"
"""
import argparse

import pandas as pd
from pygbif import occurrences

from config import CANDIDATE_PAIRS, RESULTS_DIR

# The real GBIF dataset key for INVASORAS.PT's "Sightings Map of Invasive
# Plants in Portugal" (published by CFE - Centre for Functional Ecology,
# University of Coimbra; confirmed via https://www.gbif.org and
# http://ipt.gbif.pt/ipt/resource?r=invasoras). Worth re-confirming the
# dataset is still registered under this key before relying on it long-term.
INVASORAS_DATASET_KEY = "feb41318-374b-4ed6-b61e-0369993abedc"


def count_species_in_portugal(scientific_name: str, dataset_key: str | None = None) -> dict:
    """Returns occurrence counts for one species, with and without an
    image attached, restricted to Portugal (and optionally one dataset)."""
    base_kwargs = dict(scientificName=scientific_name, country="PT", limit=0)
    if dataset_key:
        base_kwargs["datasetKey"] = dataset_key

    total = occurrences.search(**base_kwargs)["count"]
    with_image = occurrences.search(**base_kwargs, mediatype="StillImage")["count"]
    with_coords = occurrences.search(**base_kwargs, hasCoordinate=True)["count"]

    return {
        "species": scientific_name,
        "dataset_key": dataset_key or "any",
        "total_records_PT": total,
        "records_with_image_PT": with_image,
        "records_with_coordinates_PT": with_coords,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species", nargs="*", default=None,
        help="Species to check (default: every species in config.CANDIDATE_PAIRS)",
    )
    parser.add_argument(
        "--invasoras-only", action="store_true",
        help="Restrict counts to the INVASORAS.PT dataset specifically",
    )
    args = parser.parse_args()

    if args.species:
        species_list = args.species
    else:
        species_list = []
        for pair in CANDIDATE_PAIRS:
            species_list.append(pair["invasive"])
            if pair["native"]:
                species_list.append(pair["native"])

    dataset_key = INVASORAS_DATASET_KEY if args.invasoras_only else None

    rows = []
    for sp in species_list:
        print(f"Querying GBIF for: {sp} ...")
        try:
            rows.append(count_species_in_portugal(sp, dataset_key=dataset_key))
        except Exception as exc:  # noqa: BLE001 — report and keep going
            print(f"  FAILED for {sp}: {exc}")
            rows.append({"species": sp, "error": str(exc)})

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))

    out_path = RESULTS_DIR / "feasibility_check_counts.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(
        "\nNext step: compare records_with_image_PT against the per-class "
        "image counts the proposal assumed, and drop/replace any species "
        "that can't realistically support the target class size before "
        "locking in the class list."
    )


if __name__ == "__main__":
    main()
