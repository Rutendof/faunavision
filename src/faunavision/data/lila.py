"""Stream LILA camera-trap metadata into one flat index table.

The Snapshot Safari 2024 metadata is a 2 GB COCO Camera Traps JSON, and the
MegaDetector results another ~0.9 GB. Both are read incrementally with ijson
so memory stays well below what ``json.load`` would need.

Output: one row per image with its location, sequence, label(s) and the
best MegaDetector animal box.
"""

from __future__ import annotations

import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

import ijson

MD_ANIMAL = "1"  # MegaDetector categories: 1 animal, 2 person, 3 vehicle


@contextmanager
def open_json(path: Path) -> Iterator[IO[bytes]]:
    """Open a .json file, or the single .json inside a .zip."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".json"))
            with z.open(name) as f:
                yield f
    else:
        with open(path, "rb") as f:
            yield f


def project_of(image_id: str) -> str:
    return image_id.split("/", 1)[0]


def read_labels(path: Path, projects: set[str] | None = None) -> dict[str, dict]:
    """Image metadata and label set from a COCO Camera Traps file.

    Returns {image_id: {project, location, seq_id, frame_num, labels:set}}.
    Two passes over the file (images, then annotations), so the categories
    block at the end never has to be held with the rest.
    """
    with open_json(path) as f:
        categories = {
            c["id"]: c["name"] for c in ijson.items(f, "categories.item")
        }

    images: dict[str, dict] = {}
    with open_json(path) as f:
        for im in ijson.items(f, "images.item"):
            proj = project_of(im["id"])
            if projects and proj not in projects:
                continue
            images[im["id"]] = {
                "project": proj,
                "location": im.get("location"),
                "seq_id": im.get("seq_id"),
                "frame_num": im.get("frame_num"),
                "labels": set(),
            }

    with open_json(path) as f:
        for ann in ijson.items(f, "annotations.item"):
            rec = images.get(ann["image_id"])
            if rec is not None:
                rec["labels"].add(categories[ann["category_id"]])
    return images


def read_md_best_boxes(path: Path, keep: set[str], min_conf: float = 0.1) -> dict[str, dict]:
    """Best animal detection per image from a MegaDetector results file.

    Returns {file: {md_conf, bbox (x, y, w, h, normalised), md_n_animals,
    md_person}} for files in ``keep``. Images with no animal box at or above
    ``min_conf`` get md_conf 0.
    """
    out: dict[str, dict] = {}
    with open_json(path) as f:
        for im in ijson.items(f, "images.item", use_float=True):
            name = im["file"]
            if name not in keep:
                continue
            dets = im.get("detections") or []
            animals = [d for d in dets if d["category"] == MD_ANIMAL and d["conf"] >= min_conf]
            best = max(animals, key=lambda d: d["conf"], default=None)
            out[name] = {
                "md_conf": best["conf"] if best else 0.0,
                "bbox": tuple(best["bbox"]) if best else None,
                "md_n_animals": sum(d["conf"] >= 0.5 for d in animals),
                "md_person": any(d["category"] == "2" and d["conf"] >= 0.5 for d in dets),
            }
    return out

