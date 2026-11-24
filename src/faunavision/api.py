"""FaunaVision inference API: ONNX Runtime on CPU, no PyTorch needed.

Run:
    FAUNAVISION_MODEL=exports/student_b0.onnx uvicorn faunavision.api:app --port 8000

The model is a .onnx from scripts/export_onnx.py with its JSON sidecar
(classes, input size, mean/std) next to it. Send one animal crop per
request (e.g. a MegaDetector box); the response is the top-k species with
priority and lookalike-cluster information for ranger triage.
"""

from __future__ import annotations

import io
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from faunavision.data.classes import CLASSES, COMMON_NAMES

log = logging.getLogger("faunavision.api")

MAX_UPLOAD_BYTES = 10 * 2**20
Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard (PIL raises above 2x this)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
DEFAULT_REVIEW_MARGIN = 0.2  # used only if the sidecar has no margin tuned on validation data


class Classifier:
    def __init__(self, onnx_path: Path):
        meta_path = onnx_path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        unknown = set(self.meta["classes"]) - set(CLASSES)
        if unknown:
            raise ValueError(f"model classes not in this codebase: {sorted(unknown)}")
        self.size = int(self.meta["img_size"])
        self.mean = np.asarray(self.meta["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.asarray(self.meta["std"], np.float32).reshape(3, 1, 1)
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.review_margin = float(self.meta.get("review_margin", DEFAULT_REVIEW_MARGIN))
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Match the training eval transform: centre square, resized to size (bicubic), normalised.

        The square is cut in source coordinates before resizing, so a long
        thin image never becomes a huge intermediate (same result as
        shorter-side resize then centre crop, up to rounding).
        """
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.resize((self.size, self.size), Image.BICUBIC, box=(left, top, left + side, top + side))
        x = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None]

    def predict(self, img: Image.Image, top_k: int = 5) -> dict:
        logits = self.session.run([self.meta["output"]], {self.meta["input"]: self.preprocess(img)})[0][0]
        z = logits / self.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)[:top_k]
        names = self.meta["classes"]
        preds = [{"class": names[i], "common_name": COMMON_NAMES[names[i]], "probability": round(float(p[i]), 4),
                  "cluster": CLASSES[names[i]].cluster, "priority": CLASSES[names[i]].priority} for i in order]
        first, second = (preds + [None])[:2]
        lookalike = bool(second and first["cluster"] == second["cluster"]
                         and float(p[order[0]] - p[order[1]]) < self.review_margin)
        return {"predictions": preds, "priority_alert": first["priority"], "lookalike_review": lookalike}


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    path = os.environ.get("FAUNAVISION_MODEL")
    if not path:
        raise RuntimeError("set FAUNAVISION_MODEL to the exported .onnx file")
    return Classifier(Path(path))


app = FastAPI(title="FaunaVision", description="Lookalike-aware species classification for camera-trap crops")


@app.get("/health")
def health() -> dict:
    try:
        clf = get_classifier()
    except Exception as e:  # details go to the server log, not to the client
        log.exception("model failed to load")
        raise HTTPException(503, "model not loaded") from e
    return {"status": "ok", "model": clf.meta["model_name"], "img_size": clf.size, "classes": len(clf.meta["classes"])}


@app.post("/predict")
def predict(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=10)) -> dict:
    # sync endpoint: FastAPI runs it in a worker thread, so CPU-bound inference doesn't block the event loop
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(415, f"unsupported image format {img.format}; use JPEG, PNG or WebP")
        if img.width * img.height > Image.MAX_IMAGE_PIXELS:  # PIL itself only raises above 2x the limit
            raise HTTPException(413, "image has too many pixels")
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise HTTPException(400, "file is not a readable image") from e
    return get_classifier().predict(img, top_k)
