"""Stage 1 triage: how well does the detector filter empty frames?

Example:
    python scripts/triage_eval.py --projects PLN,KRU,CDB

Scored per *sequence* (one camera trigger burst), because a ranger reviews
events, not individual frames: a sequence is kept if any of its frames has
an animal box at or above the threshold.

For each threshold it reports:
- empty sequences removed (work saved)
- animal sequences kept (recall), overall and for priority classes

Caveat: MegaDetector saw some Snapshot Safari data in training, so these
numbers are likely optimistic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.classes import CLASSES, COMMON_NAMES  # noqa: E402

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def sequence_table(index: pd.DataFrame, projects: set[str]) -> pd.DataFrame:
    df = index[index.project.isin(projects)].copy()
    df["md_conf"] = df["md_conf"].fillna(0.0)
    df["priority"] = df["cls"].map(lambda c: bool(isinstance(c, str) and CLASSES[c].priority))
    df["animal"] = df["cls"].notna()
    return df.groupby("seq_id").agg(
        project=("project", "first"),
        max_conf=("md_conf", "max"),
        no_animal=("is_empty", "all"),
        animal=("animal", "any"),
        priority=("priority", "any"),
        cls=("cls", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        frames=("image_id", "size"),
    ).reset_index()


def evaluate(seqs: pd.DataFrame) -> list[dict]:
    rows = []
    empty, animal, prio = seqs[seqs.no_animal], seqs[seqs.animal], seqs[seqs.priority]
    for t in THRESHOLDS:
        kept = seqs.max_conf >= t
        rows.append({
            "threshold": t,
            "empty_removed": float((~kept[empty.index]).mean()),
            "animal_kept": float(kept[animal.index].mean()),
            "priority_kept": float(kept[prio.index].mean()),
            "frames_to_review": float(seqs.loc[kept, "frames"].sum() / seqs.frames.sum()),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=ROOT / "data" / "processed" / "camtrap_index.csv.gz")
    p.add_argument("--projects", default="PLN,KRU,CDB")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "results" / "triage_megadetector")
    args = p.parse_args()

    projects = set(args.projects.split(","))
    cols = ["image_id", "project", "seq_id", "cls", "is_empty", "md_conf"]
    seqs = sequence_table(pd.read_csv(args.index, usecols=cols, low_memory=False), projects)
    rows = evaluate(seqs)

    # Which priority species are hardest to detect at the 0.2 threshold?
    prio = seqs[seqs.priority]
    per_cls = (prio.assign(kept=prio.max_conf >= 0.2).groupby("cls")
               .agg(sequences=("kept", "size"), kept=("kept", "mean")).sort_values("kept"))

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"projects": sorted(projects), "sequences": int(len(seqs)),
               "empty_sequences": int(seqs.no_animal.sum()), "animal_sequences": int(seqs.animal.sum()),
               "priority_sequences": int(seqs.priority.sum()), "thresholds": rows,
               "priority_by_class_at_0.2": per_cls.reset_index().to_dict("records")}
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    pct = lambda v: f"{v:.1%}"  # noqa: E731
    lines = [
        "# Triage with MegaDetector on unseen reserves",
        "",
        f"Reserves: {', '.join(sorted(projects))}. {len(seqs):,} sequences "
        f"({summary['empty_sequences']:,} empty, {summary['animal_sequences']:,} with a FaunaVision animal, "
        f"{summary['priority_sequences']:,} with a priority species). A sequence is kept if any frame "
        "has an animal box at or above the threshold.",
        "",
        "| Threshold | Empty removed | Animals kept | Priority kept | Frames left to review |",
        "|---|---|---|---|---|",
        *[f"| {r['threshold']} | {pct(r['empty_removed'])} | {pct(r['animal_kept'])} | "
          f"{pct(r['priority_kept'])} | {pct(r['frames_to_review'])} |" for r in rows],
        "",
        "## Priority species kept at threshold 0.2",
        "",
        "| Species | Sequences | Kept |",
        "|---|---|---|",
        *[f"| {COMMON_NAMES[c]} | {int(r.sequences)} | {pct(r.kept)} |" for c, r in per_cls.iterrows()],
        "",
        "Caveat: MegaDetector was partly trained on Snapshot Safari data, so these numbers are likely optimistic.",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:6 + 2 + len(rows)]))


if __name__ == "__main__":
    main()
