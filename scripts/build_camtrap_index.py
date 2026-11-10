"""Build the flat camera-trap index (labels + best MegaDetector box per image).

Example:
    python scripts/build_camtrap_index.py

Reads data/raw/lila/{ss2024_meta.zip, md_ss2024_rde.json.zip} and writes
data/processed/camtrap_index.csv.gz. Takes several minutes (it streams ~3 GB
of JSON) but only needs to run once.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.classes import EMPTY_CATEGORY, map_category  # noqa: E402
from faunavision.data.lila import read_labels, read_md_best_boxes  # noqa: E402

# Southern African Snapshot Safari projects (Kenya and Tanzania left out).
SOUTHERN_PROJECTS = {
    "APN", "CDB", "KAR", "KGA", "KRU", "LEC", "MAD", "MTZ", "PLN", "TSW",
    "Snapshot South Africa", "Snapshot Cameo",
}


def main() -> None:
    p = argparse.ArgumentParser()
    raw = ROOT / "data" / "raw" / "lila"
    p.add_argument("--labels", type=Path, default=raw / "ss2024_meta.zip")
    p.add_argument("--md", type=Path, default=raw / "md_ss2024_rde.json.zip")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "camtrap_index.csv.gz")
    args = p.parse_args()

    t0 = time.time()
    images = read_labels(args.labels, SOUTHERN_PROJECTS)
    print(f"{len(images):,} labelled images read ({time.time() - t0:.0f}s)")

    boxes = read_md_best_boxes(args.md, keep=set(images))
    print(f"{len(boxes):,} images matched to MegaDetector results ({time.time() - t0:.0f}s)")

    rows = []
    for image_id, rec in images.items():
        labels = rec["labels"]
        raw_label = next(iter(labels)) if len(labels) == 1 else None
        md = boxes.get(image_id)
        bbox = md["bbox"] if md and md["bbox"] else (None, None, None, None)
        rows.append({
            "image_id": image_id,
            "project": rec["project"],
            "location": rec["location"],
            "seq_id": rec["seq_id"],
            "frame_num": rec["frame_num"],
            "raw_label": raw_label if raw_label else "|".join(sorted(labels)),
            "n_labels": len(labels),
            "cls": map_category(raw_label) if raw_label else None,
            "is_empty": labels == {EMPTY_CATEGORY},
            "md_found": md is not None,
            "md_conf": md["md_conf"] if md else None,
            "bx": bbox[0], "by": bbox[1], "bw": bbox[2], "bh": bbox[3],
            "md_n_animals": md["md_n_animals"] if md else None,
            "md_person": md["md_person"] if md else None,
        })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    print(f"Wrote {args.out} ({len(df):,} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
