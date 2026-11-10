"""Download manifest images from LILA's public bucket and save animal crops.

Only the crop around MegaDetector's box is kept (resized so its longer side
is at most --size px), so ~100k images become a few GB instead of hundreds.
Resumable: existing crops are skipped. Failures go to failures.csv.

Run on Kaggle for speed (free, fast internet):
    python scripts/download_crops.py --manifest data/processed/manifest.csv --out /kaggle/working/crops

Local smoke test:
    python scripts/download_crops.py --limit 20
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.crops import crop_path, square_crop_box  # noqa: E402

BASE_URL = "https://storage.googleapis.com/public-datasets-lila/snapshot-safari-2024-expansion/"


def fetch_and_crop(session, row, out: Path, size: int) -> None:
    dest = crop_path(out, row.image_id, row.split, row.cls)
    if dest.exists():
        return
    resp = session.get(BASE_URL + requests.utils.quote(row.image_id), timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    box = square_crop_box(row.bx, row.by, row.bw, row.bh, img.width, img.height)
    crop = img.crop(box)
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    crop.save(tmp, "JPEG", quality=90)
    tmp.replace(dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "manifest.csv")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "crops")
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)
    session = requests.Session()
    session.headers["User-Agent"] = "FaunaVision-research/0.1 (portfolio project)"
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=args.workers, max_retries=3)
    session.mount("https://", adapter)

    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_and_crop, session, r, args.out, args.size): r for r in df.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:  # keep going; record and report at the end
                failures.append((futures[fut].image_id, repr(e)))
            if i % 1000 == 0 or i == len(futures):
                print(f"{i:,}/{len(futures):,} done, {len(failures)} failed", flush=True)

    if failures:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "failures.csv", "w", newline="") as f:
            csv.writer(f).writerows([("image_id", "error"), *failures])
        print(f"{len(failures)} failures written to {args.out / 'failures.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
