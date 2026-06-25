"""
A single PyTorch Dataset that doesn't care whether an image came from
INVASORAS.PT, Floralens, or a gbif-dl pull — it just reads a manifest CSV
with the columns: filepath, label, region, source.

Keeping the three data sources behind a common manifest format means the
download scripts (download_invasoras.py, download_floralens.py,
download_gbif_dl.py) can be developed/swapped independently of the
training code.
"""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.paths import resolve_manifest_path

REQUIRED_COLUMNS = {"filepath", "label", "region", "source"}


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None,
    ):
        if isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
            self.manifest_path = None
        else:
            self.manifest_path = Path(manifest)
            df = pd.read_csv(self.manifest_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {missing}. "
                f"Expected at least: {sorted(REQUIRED_COLUMNS)}"
            )

        self.df = df.reset_index(drop=True)
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(self.df["label"].unique())
            class_to_idx = {c: i for i, c in enumerate(classes)}
        self.class_to_idx = class_to_idx
        self.idx_to_class = {i: c for c, i in class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_manifest_path(row["filepath"], self.manifest_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label

    def region_array(self):
        """Used by src/data/splits.py for geographic stratification."""
        return self.df["region"].to_numpy()

    def label_array(self):
        return self.df["label"].to_numpy()
