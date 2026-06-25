import pandas as pd
from pathlib import Path

from src.data.quality import audit_manifest


def test_audit_manifest_reports_core_data_hygiene():
    work_dir = Path(__file__).parent / "_tmp_quality"
    work_dir.mkdir(exist_ok=True)
    existing = work_dir / "img.jpg"
    existing.write_bytes(b"not a real image but present")
    missing = work_dir / "missing.jpg"
    if missing.exists():
        missing.unlink()
    manifest = work_dir / "manifest.csv"

    pd.DataFrame([
        {
            "filepath": str(existing),
            "label": "A",
            "region": "Lisboa",
            "source": "gbif_dl",
            "latitude": 38.7,
            "longitude": -9.1,
            "group": "grid:a",
            "gbif_id": "1",
        },
        {
            "filepath": str(missing),
            "label": "B",
            "region": "Unknown",
            "source": "floralens",
            "latitude": None,
            "longitude": None,
            "group": "no_coords:floralens:Unknown",
            "gbif_id": "2",
        },
    ]).to_csv(manifest, index=False)

    report = audit_manifest(manifest)

    assert report["n_rows"] == 2
    assert report["missing_files"] == 1
    assert report["rows_without_coordinates"] == 1
    assert report["class_counts"] == {"A": 1, "B": 1}
    assert report["issues"]

    existing.unlink()
    manifest.unlink()
    work_dir.rmdir()
