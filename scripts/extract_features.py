"""Extract frozen backbone features for every crop (model 1: DINOv2 baseline).

Example (Kaggle GPU):
    python scripts/extract_features.py --crops /kaggle/working/crops --out /kaggle/working/features

Laptop smoke test (small model, CPU):
    python scripts/extract_features.py --crops data/crops_smoke --model vit_small_patch14_dinov2.lvd142m

Writes one <split>.npz per split with: features (N x D float16), labels,
image_ids. Features are computed once; every probe experiment reuses them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.dataset import CropDataset, available_rows  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "manifest.csv")
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "features")
    p.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=args.img_size)
    model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    # The pretrained config says 518 px; we run at --img-size. Crops are already
    # square around the animal, so resize straight to size (no centre crop).
    cfg.update(input_size=(3, args.img_size, args.img_size), crop_pct=1.0)
    transform = timm.data.create_transform(**cfg, is_training=False)
    print(f"{args.model} on {device}, input {cfg['input_size']}, feature dim {model.num_features}")

    manifest = pd.read_csv(args.manifest)
    rows = available_rows(manifest, args.crops)
    print(f"{len(rows):,} of {len(manifest):,} manifest crops found on disk")
    args.out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        part = rows[rows.split == split]
        if part.empty:
            continue
        loader = DataLoader(CropDataset(part, args.crops, transform), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=device == "cuda")
        feats, labels, idx = [], [], []
        t0 = time.time()
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            for x, y, i in loader:
                feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy().astype(np.float16))
                labels.append(y.numpy())
                idx.append(i.numpy())
        idx = np.concatenate(idx)
        np.savez(
            args.out / f"{split}.npz",
            features=np.concatenate(feats),
            labels=np.concatenate(labels),
            image_ids=part.image_id.to_numpy()[idx].astype(str),  # fixed-width str: loads without pickle
            model=np.array(args.model),
        )
        print(f"{split}: {len(part):,} crops in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
