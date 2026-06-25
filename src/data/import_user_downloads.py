"""
Import the user's pre-downloaded GBIF / Floralens / INVASORAS archives.

Maps your four folders:
  dwca-invasoras-v2.10     -> data_raw/invasoras/     (direct image URLs in associatedMedia)
  floralens-data-v0.3      -> data_raw/floralens/     (url column in floralens_dataset.tsv)
  0067492-... (iNaturalist GBIF export) -> data_raw/gbif_dl/inaturalist/
  0067463-... (Pl@ntNet GBIF export)    -> data_raw/gbif_dl/plantnet/

Usage:
    python -m src.data.import_user_downloads
    python -m src.data.import_user_downloads --max-per-species 500 --merge
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from pygbif import occurrences as gbif_occurrences
from tqdm import tqdm

from config import CANDIDATE_PAIRS, DATA_ROOT, MANIFEST_DIR, SEED
from src.data.paths import manifest_path_for

# Default paths — your Downloads folder
DEFAULT_PATHS = {
    "invasoras_dwca": Path(r"C:\Users\USER\Downloads\dwca-invasoras-v2.10"),
    "floralens_root": Path(r"C:\Users\USER\Downloads\floralens-data-v0.3\edrdo-floralens-data-5c0726f"),
    "inaturalist_csv": Path(r"C:\Users\USER\Downloads\0067492-260519110011954\0067492-260519110011954.csv"),
    "plantnet_csv": Path(r"C:\Users\USER\Downloads\0067463-260519110011954\0067463-260519110011954.csv"),
}

INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
PLANTNET_DATASET = "7a3679ef-5582-4aaa-81f0-8c2545cafc81"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".JPG"}


def _target_species() -> tuple[list[str], list[str]]:
    invasives, natives = [], []
    for pair in CANDIDATE_PAIRS:
        invasives.append(pair["invasive"])
        if pair["native"]:
            natives.append(pair["native"])
    return invasives, natives


def _species_dir(out_root: Path, species: str) -> Path:
    d = out_root / species.replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_url(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as exc:
        print(f"    skip {dest.name}: {exc}")
        return False


def _ext_from_url(url: str) -> str:
    path = urlparse(url).path
    ext = Path(path).suffix
    return ext if ext else ".jpg"


def import_invasoras_dwca(
    dwca_dir: Path,
    out_dir: Path,
    invasives: list[str],
    max_per_species: int | None,
    session: requests.Session,
) -> list[dict]:
    occ_path = dwca_dir / "occurrence.txt"
    if not occ_path.exists():
        print(f"  (skip invasoras — {occ_path} not found)")
        return []

    df = pd.read_csv(occ_path, sep="\t", low_memory=False)
    df = df[df["associatedMedia"].notna()].copy()
    df = df[df["scientificName"].astype(str).apply(
        lambda n: any(n.startswith(sp) for sp in invasives)
    )]

    rows = []
    for species in invasives:
        sub = df[df["scientificName"].str.startswith(species)]
        if max_per_species:
            sub = sub.head(max_per_species)
        species_dir = _species_dir(out_dir, species)

        for _, rec in tqdm(sub.iterrows(), total=len(sub), desc=f"invasoras {species}", unit="img"):
            url = str(rec["associatedMedia"]).strip()
            ext = _ext_from_url(url)
            fname = hashlib.md5(url.encode()).hexdigest()[:16] + ext
            filepath = species_dir / fname
            if not _download_url(url, filepath, session):
                continue
            rows.append({
                "filepath": manifest_path_for(filepath),
                "label": species,
                "region": rec.get("stateProvince") or rec.get("county") or "Unknown",
                "latitude": rec.get("decimalLatitude"),
                "longitude": rec.get("decimalLongitude"),
                "source": "invasoras",
                "gbif_id": rec.get("id"),
            })
    return rows


def import_floralens_tsv(
    floralens_root: Path,
    out_dir: Path,
    species_list: list[str],
    max_per_species: int | None,
    session: requests.Session,
) -> list[dict]:
    tsv = floralens_root / "floralens" / "floralens_dataset.tsv"
    if not tsv.exists():
        print(f"  (skip floralens — {tsv} not found)")
        return []

    df = pd.read_csv(tsv, sep="\t")
    df = df[df["species"].isin(species_list)]

    rows = []
    for species in species_list:
        sub = df[df["species"] == species]
        if max_per_species:
            sub = sub.head(max_per_species)
        species_dir = _species_dir(out_dir, species)

        for _, rec in tqdm(sub.iterrows(), total=len(sub), desc=f"floralens {species}", unit="img"):
            url = str(rec["url"]).strip()
            fname = rec.get("filename") or (hashlib.md5(url.encode()).hexdigest()[:16] + ".jpg")
            filepath = species_dir / Path(str(fname)).name
            if not _download_url(url, filepath, session):
                continue
            rows.append({
                "filepath": manifest_path_for(filepath),
                "label": species,
                "region": "Unknown",
                "source": "floralens",
                "gbif_id": rec.get("gbif_id"),
                "floralens_set": rec.get("set"),
                "floralens_repos": rec.get("repos"),
            })
    return rows


def _gbif_media_url(gbif_id: int, cache: dict) -> str | None:
    if gbif_id in cache:
        return cache[gbif_id]
    try:
        rec = gbif_occurrences.get(gbif_id)
        media = rec.get("media") or []
        url = next(
            (m["identifier"] for m in media if m.get("type") == "StillImage" and m.get("identifier")),
            None,
        )
        cache[gbif_id] = url
        time.sleep(0.05)
        return url
    except Exception:
        cache[gbif_id] = None
        return None


def import_gbif_csv(
    csv_path: Path,
    out_dir: Path,
    source_tag: str,
    dataset_key: str,
    species_list: list[str],
    max_per_species: int | None,
    session: requests.Session,
) -> list[dict]:
    if not csv_path.exists():
        print(f"  (skip {source_tag} — {csv_path} not found)")
        return []

    print(f"  Loading {csv_path.name} (large file, may take a minute)...")
    df = pd.read_csv(csv_path, sep="\t", low_memory=False)
    df = df[
        (df["datasetKey"] == dataset_key)
        & (df["countryCode"] == "PT")
        & (df["kingdom"] == "Plantae")
        & (df["mediaType"] == "StillImage")
    ]
    df = df[df["scientificName"].astype(str).apply(
        lambda n: any(n.startswith(sp) for sp in species_list)
    )]

    rows = []
    cache: dict = {}
    for species in species_list:
        sub = df[df["scientificName"].str.startswith(species)]
        if max_per_species:
            sub = sub.head(max_per_species)
        species_dir = _species_dir(out_dir / source_tag, species)

        for _, rec in tqdm(sub.iterrows(), total=len(sub), desc=f"{source_tag} {species}", unit="img"):
            gbif_id = int(rec["gbifID"])
            url = _gbif_media_url(gbif_id, cache)
            if not url:
                continue
            ext = _ext_from_url(url)
            filepath = species_dir / f"{gbif_id}{ext}"
            if not _download_url(url, filepath, session):
                continue
            rows.append({
                "filepath": manifest_path_for(filepath),
                "label": species,
                "region": rec.get("stateProvince") or "Unknown",
                "latitude": rec.get("decimalLatitude"),
                "longitude": rec.get("decimalLongitude"),
                "source": "gbif_dl",
                "gbif_id": gbif_id,
                "gbif_platform": source_tag,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--invasoras-dwca", type=Path, default=DEFAULT_PATHS["invasoras_dwca"])
    parser.add_argument("--floralens-root", type=Path, default=DEFAULT_PATHS["floralens_root"])
    parser.add_argument("--inaturalist-csv", type=Path, default=DEFAULT_PATHS["inaturalist_csv"])
    parser.add_argument("--plantnet-csv", type=Path, default=DEFAULT_PATHS["plantnet_csv"])
    parser.add_argument("--max-per-species", type=int, default=500,
                        help="Cap per species per source (default 500; floralens has ~200/species)")
    parser.add_argument("--skip-gbif-csv", action="store_true",
                        help="Skip iNaturalist/Pl@ntNet CSV import (slow — many API calls)")
    parser.add_argument("--skip-invasoras", action="store_true",
                        help="Skip INVASORAS image download (use if app.invasoras.pt is unreachable)")
    parser.add_argument("--merge", action="store_true", help="Run merge_manifests after import")
    args = parser.parse_args()

    invasives, natives = _target_species()
    all_species = sorted(set(invasives + natives))
    session = requests.Session()
    session.headers.update({"User-Agent": "InvasiveLens/1.0 (research)"})

    print("Importing your downloaded archives into InvasiveLens...")
    print(f"  Target invasives: {invasives}")
    print(f"  Target natives:   {natives}")
    print(f"  Max per species per source: {args.max_per_species}")

    inv_rows = []
    if not args.skip_invasoras:
        inv_rows = import_invasoras_dwca(
            args.invasoras_dwca, DATA_ROOT / "invasoras", invasives, args.max_per_species, session,
        )
        if inv_rows:
            path = MANIFEST_DIR / "invasoras_manifest.csv"
            pd.DataFrame(inv_rows).to_csv(path, index=False)
            print(f"  invasoras: {len(inv_rows)} images -> {path}")
    else:
        print("  (skipping invasoras — app.invasoras.pt unreachable or --skip-invasoras)")

    fl_rows = import_floralens_tsv(
        args.floralens_root, DATA_ROOT / "floralens", natives, args.max_per_species, session,
    )
    if fl_rows:
        path = MANIFEST_DIR / "floralens_manifest.csv"
        pd.DataFrame(fl_rows).to_csv(path, index=False)
        print(f"  floralens: {len(fl_rows)} images -> {path}")

    gbif_rows = []
    if not args.skip_gbif_csv:
        gbif_rows.extend(import_gbif_csv(
            args.inaturalist_csv, DATA_ROOT / "gbif_dl", "inaturalist",
            INAT_DATASET, all_species, args.max_per_species, session,
        ))
        gbif_rows.extend(import_gbif_csv(
            args.plantnet_csv, DATA_ROOT / "gbif_dl", "plantnet",
            PLANTNET_DATASET, all_species, args.max_per_species, session,
        ))
        if gbif_rows:
            path = MANIFEST_DIR / "gbif_dl_manifest.csv"
            pd.DataFrame(gbif_rows).to_csv(path, index=False)
            print(f"  gbif_dl: {len(gbif_rows)} images -> {path}")

    total = len(inv_rows) + len(fl_rows) + len(gbif_rows)
    print(f"\nDone — {total} images imported to {DATA_ROOT}")

    if args.merge:
        from src.data.merge_manifests import main as merge_main  # noqa: WPS433
        merge_main()
    else:
        print("\nNext: python -m src.data.merge_manifests")


if __name__ == "__main__":
    main()
