"""Crop geometry and on-disk layout, shared by the downloader and the loaders."""

from __future__ import annotations

from pathlib import Path


def square_crop_box(bx, by, bw, bh, width, height, pad=0.1):
    """Pixel box: the detection padded by ``pad`` and grown to a square
    around its centre (no stretching), clipped to the image."""
    side = max(bw * width, bh * height) * (1 + 2 * pad)
    cx, cy = (bx + bw / 2) * width, (by + bh / 2) * height
    left, top = max(0, cx - side / 2), max(0, cy - side / 2)
    right, bottom = min(width, cx + side / 2), min(height, cy + side / 2)
    return int(left), int(top), int(round(right)), int(round(bottom))


def crop_path(root: Path, image_id: str, split: str, cls: str) -> Path:
    safe = image_id.replace("/", "__").replace(" ", "_")
    return Path(root) / split / cls / safe
