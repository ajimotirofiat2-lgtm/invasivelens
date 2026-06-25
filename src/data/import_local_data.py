"""
Bring already-downloaded images into InvasiveLens.

Use this when you downloaded data on another machine, or have image folders
but no manifest CSVs yet.

Expected layout (copy your folders to match):

    data_raw/
      invasoras/
        Arundo_donax/*.jpg
        Cortaderia_selloana/*.jpg
      floralens/
        Phragmites_australis/*.jpg
      gbif_dl/
        Acacia_dealbata/*.jpg

Then run:

    python -m src.data.import_local_data
    python -m src.data.merge_manifests

If your data lives elsewhere, copy it in one shot:

    python -m src.data.import_local_data --copy-from "D:\my_downloads\invasoras" --source invasoras

Or point at an external tree without copying (paths in manifest stay where they are):

    python -m src.data.import_local_data --scan "D:\all_my_plant_images" --source external
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import DATA_ROOT, MANIFEST_DIR
from src.data.paths import manifest_path_for

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
SOURCE_DIRS = {
    "invasoras": DATA_ROOT / "invasoras",
    "floralens": DATA_ROOT / "floralens",
    "gbif_dl": DATA_ROOT / "gbif_dl",
}
MANIFEST_NAMES = {
    "invasoras": "invasoras_manifest.csv",
    "floralens": "floralens_manifest.csv",
    "gbif_dl": "gbif_dl_manifest.csv",
    "external": "external_manifest.csv",
}


def _species_from_dirname(name: str) -> str:
    return name.replace("_", " ")


def _scan_species_tree(root: Path, source: str) -> list[dict]:
    """Walk root/<Species_Name>/*.jpg and build manifest rows."""
    if not root.exists():
        return []

    rows = []
    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir():
            continue
        label = _species_from_dirname(species_dir.name)
        for img_path in sorted(species_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append({
                "filepath": manifest_path_for(img_path),
                "label": label,
                "region": "Unknown",
                "source": source,
            })
    return rows


def _copy_tree(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            rel = item.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(item, target)
            count += 1
    return count


def build_manifest_for_source(source: str, root: Path | None = None) -> pd.DataFrame:
    root = root or SOURCE_DIRS.get(source)
    if root is None:
        raise ValueError(f"Unknown source '{source}'. Choose from {list(SOURCE_DIRS)}")
    rows = _scan_species_tree(root, source)
    return pd.DataFrame(rows)


def import_all(scan_external: Path | None = None) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    for source, root in SOURCE_DIRS.items():
        df = build_manifest_for_source(source, root)
        if df.empty:
            print(f"  (no images in {root})")
            continue
        out = MANIFEST_DIR / MANIFEST_NAMES[source]
        df.to_csv(out, index=False)
        print(f"  {source}: {len(df)} images -> {out.name}")

    if scan_external:
        df = build_manifest_for_source("external", scan_external)
        if not df.empty:
            out = MANIFEST_DIR / MANIFEST_NAMES["external"]
            df.to_csv(out, index=False)
            print(f"  external: {len(df)} images -> {out.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copy-from", type=Path, default=None,
                        help="Copy images from this folder into data_raw/<source>/")
    parser.add_argument("--source", choices=[*SOURCE_DIRS, "external"], default="invasoras",
                        help="Which source tag to use with --copy-from (default: invasoras)")
    parser.add_argument("--scan", type=Path, default=None,
                        help="Scan an external folder tree in place (no copy)")
    parser.add_argument("--merge", action="store_true",
                        help="Run merge_manifests after importing")
    args = parser.parse_args()

    if args.copy_from:
        dest = SOURCE_DIRS.get(args.source, DATA_ROOT / args.source)
        n = _copy_tree(args.copy_from, dest)
        print(f"Copied {n} images from {args.copy_from} -> {dest}")

    print("Scanning local images and building manifests...")
    import_all(scan_external=args.scan)

    if args.merge:
        from src.data.merge_manifests import main as merge_main  # noqa: WPS433
        merge_main()
    else:
        print("\nNext: python -m src.data.merge_manifests")


if __name__ == "__main__":
    main()
