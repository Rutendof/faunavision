"""Create the train/val/test manifest from the camera-trap index.

Example:
    python scripts/make_manifest.py --test-projects KRU,KGA,LEC

Writes data/processed/manifest.csv and reports/data/coverage.md
(images per class per split, with gaps flagged).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.classes import CLASSES, COMMON_NAMES  # noqa: E402
from faunavision.data.splits import build_manifest, coverage  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=ROOT / "data" / "processed" / "camtrap_index.csv.gz")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "manifest.csv")
    p.add_argument("--report", type=Path, default=ROOT / "reports" / "data" / "coverage.md")
    p.add_argument("--test-projects", required=True, help="comma-separated reserve codes held out for testing")
    p.add_argument("--cap-train", type=int, default=4000)
    p.add_argument("--cap-eval", type=int, default=1500)
    p.add_argument("--min-md-conf", type=float, default=0.5)
    args = p.parse_args()

    test_projects = set(args.test_projects.split(","))
    index = pd.read_csv(args.index, low_memory=False)
    m = build_manifest(index, test_projects, args.min_md_conf, args.cap_train, args.cap_eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)

    cov = coverage(m).reindex(sorted(CLASSES), fill_value=0)
    cov["cluster"] = [CLASSES[c].cluster for c in cov.index]
    cov["priority"] = [CLASSES[c].priority for c in cov.index]
    cov = cov.sort_values(["cluster", "train"], ascending=[True, False])

    lines = [
        "# Camera-trap dataset coverage",
        "",
        f"- Test reserves (never seen in training): {', '.join(sorted(test_projects))}",
        f"- One frame per sequence, MegaDetector confidence >= {args.min_md_conf}",
        f"- Caps: {args.cap_train} train / {args.cap_eval} val and test images per class",
        f"- Totals: {len(m):,} images; " + ", ".join(f"{s} {n:,}" for s, n in m.split.value_counts().items()),
        "",
        "| Cluster | Class | Priority | Train | Val | Test | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, r in cov.iterrows():
        note = []
        if r.train < 100:
            note.append("too few to train: needs iNaturalist top-up")
        if r.test == 0:
            note.append("absent from test reserves")
        lines.append(
            f"| {r.cluster} | {COMMON_NAMES[cls]} | {'yes' if r.priority else ''} | "
            f"{r.train} | {r.val} | {r.test} | {'; '.join(note)} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(m):,} images in manifest; report at {args.report}")


if __name__ == "__main__":
    main()
