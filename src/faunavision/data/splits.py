"""Leakage-aware splits and the training manifest.

Split rules:
- **Test** = whole reserves the model never sees in training. This measures
  what matters for a new conservancy: does the model transfer to new
  cameras, habitats and lighting?
- **Validation** = whole camera locations held out from the training
  reserves (chosen by a stable hash, so the split never changes).
- Frames from the same burst (sequence) always share a camera, so they can
  never straddle splits.

Manifest rules:
- Only images with exactly one label that maps to a FaunaVision class.
- One frame per sequence (the one where MegaDetector is most confident),
  because burst frames are near-duplicates.
- A MegaDetector box is required, so we can crop the animal.
- Per-class caps stop impala and elephant from dominating.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def stable_fraction(key: str) -> float:
    """Deterministic value in [0, 1) for a string."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0x1_0000_0000


def assign_splits(df: pd.DataFrame, test_projects: set[str], val_frac: float = 0.15) -> pd.Series:
    def split_for(project: str, location: str) -> str:
        if project in test_projects:
            return "test"
        return "val" if stable_fraction(f"{project}/{location}") < val_frac else "train"

    return pd.Series(
        [split_for(p, l) for p, l in zip(df["project"], df["location"].astype(str))],
        index=df.index,
        name="split",
    )


def one_frame_per_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most confidently detected frame of each sequence."""
    df = df.sort_values(["md_conf", "frame_num"], ascending=[False, True])
    return df.drop_duplicates(subset=["seq_id", "cls"], keep="first")


def build_manifest(
    index: pd.DataFrame,
    test_projects: set[str],
    min_md_conf: float = 0.5,
    cap_train: int = 4000,
    cap_eval: int = 1500,
    val_frac: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    df = index[(index["n_labels"] == 1) & index["cls"].notna()].copy()
    df = df[df["md_conf"].fillna(0) >= min_md_conf]
    df["split"] = assign_splits(df, test_projects, val_frac)
    df = one_frame_per_sequence(df)

    parts = []
    for (split, _cls), g in df.groupby(["split", "cls"], sort=True):
        cap = cap_train if split == "train" else cap_eval
        parts.append(g.sample(n=min(cap, len(g)), random_state=seed))
    out = pd.concat(parts).sort_values(["split", "cls", "image_id"]).reset_index(drop=True)
    return out


def coverage(manifest: pd.DataFrame) -> pd.DataFrame:
    """Images per class per split (0 where a class is missing)."""
    return (
        manifest.pivot_table(index="cls", columns="split", values="image_id", aggfunc="count", fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
        .sort_values("train", ascending=False)
    )
