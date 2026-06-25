"""
Integration test: synthetic fixture -> merge -> train baseline 1 epoch.
Run with:  pytest tests/test_godmode.py -v
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "manifests" / "combined_manifest.csv"


@pytest.mark.integration
def test_synthetic_fixture_and_merge():
    subprocess.run(
        [sys.executable, "-m", "src.data.generate_synthetic_fixture", "--per-species", "20"],
        cwd=PROJECT_ROOT, check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "src.data.merge_manifests"],
        cwd=PROJECT_ROOT, check=True,
    )
    df = pd.read_csv(MANIFEST)
    assert len(df) >= 100
    assert "group" in df.columns
    assert df["label"].nunique() >= 5


@pytest.mark.integration
def test_baseline_train_one_epoch():
    if not MANIFEST.exists():
        test_synthetic_fixture_and_merge()

    result = subprocess.run(
        [sys.executable, "-m", "src.train",
         "--manifest", str(MANIFEST),
         "--model", "baseline",
         "--epochs", "1",
         "--no-pretrained"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    metrics_path = PROJECT_ROOT / "results" / "baseline_metrics.json"
    assert metrics_path.exists()
