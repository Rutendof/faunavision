"""PyTorch dataset over downloaded crops."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from faunavision.data.classes import CLASSES
from faunavision.data.crops import crop_path

# Fixed class order: every model and report uses these indices.
CLASS_NAMES: list[str] = sorted(CLASSES)
CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}


def available_rows(manifest: pd.DataFrame, crops_root: Path) -> pd.DataFrame:
    """Manifest rows whose crop exists on disk (downloads can fail or be partial)."""
    exists = [
        crop_path(crops_root, r.image_id, r.split, r.cls).exists()
        for r in manifest.itertuples()
    ]
    return manifest[exists].reset_index(drop=True)


class CropDataset:
    """Yields (image_tensor, label_index, row_index)."""

    def __init__(self, rows: pd.DataFrame, crops_root: Path, transform):
        self.rows = rows.reset_index(drop=True)
        self.root = Path(crops_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        img = Image.open(crop_path(self.root, r.image_id, r.split, r.cls)).convert("RGB")
        return self.transform(img), CLASS_INDEX[r.cls], i
