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
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from faunavision.data.classes import CLASSES, COMMON_NAMES

MAX_UPLOAD_BYTES = 10 * 2**20
Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard (PIL raises above 2x this)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
LOOKALIKE_MARGIN = 0.2  # top-2 in one cluster and closer than this -> flag for review


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
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Match the training eval transform: shorter side to size (bicubic), centre crop, normalise."""
        img = img.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
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
                         and first["probability"] - second["probability"] < LOOKALIKE_MARGIN)
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
    except Exception as e:  # report, don't crash: lets a platform health check show the cause
        raise HTTPException(503, f"model not loaded: {e}") from e
    return {"status": "ok", "model": clf.meta["model_name"], "img_size": clf.size, "classes": len(clf.meta["classes"])}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=10)) -> dict:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(415, f"unsupported image format {img.format}; use JPEG, PNG or WebP")
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise HTTPException(400, "file is not a readable image") from e
    return get_classifier().predict(img, top_k)
