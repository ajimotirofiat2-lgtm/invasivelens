"""
Creates a fully offline image dataset + manifest so the entire InvasiveLens
pipeline (merge -> train -> compare -> Grad-CAM) can run without GBIF/Zenodo.

Each species gets a distinct color patch on a noisy background, spread across
Portugal-like coordinates so geographic CV splits behave realistically.

Usage:
    python -m src.data.generate_synthetic_fixture
    python -m src.data.generate_synthetic_fixture --per-species 40 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import CANDIDATE_PAIRS, DATA_ROOT, MANIFEST_DIR, SEED
from src.data.paths import manifest_path_for

DISTRICT_GEO = {
    "Lisboa": (38.72, -9.14, 0.35),
    "Porto": (41.15, -8.61, 0.30),
    "Setúbal": (38.52, -8.89, 0.08),
    "Faro": (37.02, -7.93, 0.08),
    "Braga": (41.55, -8.42, 0.08),
    "Madeira": (32.65, -16.90, 0.05),
}

SPECIES_COLORS = {
    "Arundo donax": (180, 40, 40),
    "Phragmites australis": (40, 140, 60),
    "Cortaderia selloana": (200, 160, 40),
    "Ammophila arenaria": (60, 100, 200),
    "Acacia dealbata": (140, 60, 180),
}


def _species_list() -> list[str]:
    names = []
    for pair in CANDIDATE_PAIRS:
        names.append(pair["invasive"])
        if pair["native"]:
            names.append(pair["native"])
    return names


def _make_image(color: tuple[int, int, int], rng: np.random.RandomState, size: int = 224) -> Image.Image:
    arr = rng.randint(20, 60, size=(size, size, 3), dtype=np.uint8)
    margin = size // 5
    arr[margin:-margin, margin:-margin] = np.array(color, dtype=np.uint8)
    return Image.fromarray(arr)


def generate_fixture(
    per_species: int = 30,
    seed: int = SEED,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    out_dir = out_dir or (DATA_ROOT / "synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    species_list = _species_list()
    districts = list(DISTRICT_GEO.keys())
    district_weights = [0.45, 0.25, 0.10, 0.08, 0.07, 0.05]

    rows = []
    for species in species_list:
        species_dir = out_dir / species.replace(" ", "_")
        species_dir.mkdir(parents=True, exist_ok=True)
        color = SPECIES_COLORS.get(species)
        if color is None:
            color = tuple(int(x) for x in rng.randint(80, 200, size=3))

        for i in range(per_species):
            district = rng.choice(districts, p=district_weights)
            lat0, lon0, jitter = DISTRICT_GEO[district]
            lat = lat0 + rng.uniform(-jitter, jitter)
            lon = lon0 + rng.uniform(-jitter, jitter)

            filepath = species_dir / f"{species.replace(' ', '_')}_{i:04d}.jpg"
            _make_image(color, rng).save(filepath, quality=90)

            rows.append({
                "filepath": manifest_path_for(filepath),
                "label": species,
                "region": district,
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "source": "synthetic",
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-species", type=int, default=30,
                        help="Images per species (default 30 -> 150 total for 5 species)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "synthetic"))
    args = parser.parse_args()

    df = generate_fixture(per_species=args.per_species, seed=args.seed, out_dir=Path(args.out_dir))
    manifest_path = MANIFEST_DIR / "synthetic_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Wrote {len(df)} synthetic images to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
