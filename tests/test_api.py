import io
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

CLASSES = ["white_rhino", "black_rhino", "elephant"]


def _tiny_model(path):
    """Mean colour -> logits: red favours white_rhino, green black_rhino, blue elephant."""
    w = np.eye(3, dtype=np.float32) * 10
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
         helper.make_node("Flatten", ["pooled"], ["flat"]),
         helper.make_node("Gemm", ["flat", "W"], ["logits"])],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.flatten())],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    meta = {"model_name": "tiny", "img_size": 32, "classes": CLASSES, "mean": [0.0] * 3, "std": [1.0] * 3,
            "input": "input", "output": "logits"}
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from faunavision import api

    _tiny_model(tmp_path / "m.onnx")
    monkeypatch.setenv("FAUNAVISION_MODEL", str(tmp_path / "m.onnx"))
    api.get_classifier.cache_clear()
    yield TestClient(api.app)
    api.get_classifier.cache_clear()


def _png(color, size=(50, 40), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["classes"] == 3


def test_predict_ranks_and_flags(client):
    r = client.post("/predict?top_k=2", files={"file": ("a.png", _png((255, 0, 0)), "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert [p["class"] for p in body["predictions"]][0] == "white_rhino" and len(body["predictions"]) == 2
    assert body["priority_alert"] is True and body["predictions"][0]["common_name"] == "White rhinoceros"
    assert body["lookalike_review"] is False  # confident


def test_lookalike_review_when_rhinos_are_close(client):
    r = client.post("/predict", files={"file": ("a.png", _png((200, 199, 0)), "image/png")})
    assert r.json()["lookalike_review"] is True


def test_rejects_non_images_and_bad_formats(client):
    assert client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400
    assert client.post("/predict", files={"file": ("x.gif", _png((0, 0, 0), fmt="GIF"), "image/gif")}).status_code == 415


def test_rejects_oversized_upload(client, monkeypatch):
    from faunavision import api

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    assert client.post("/predict", files={"file": ("a.png", _png((1, 2, 3), (200, 200)), "image/png")}).status_code == 413


def test_health_reports_missing_model(monkeypatch):
    from fastapi.testclient import TestClient

    from faunavision import api

    monkeypatch.delenv("FAUNAVISION_MODEL", raising=False)
    api.get_classifier.cache_clear()
    assert TestClient(api.app).get("/health").status_code == 503


def test_review_margin_comes_from_sidecar(client, tmp_path):
    from faunavision import api

    meta_path = tmp_path / "m.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["review_margin"] = 1e-9  # effectively never flag
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    api.get_classifier.cache_clear()
    r = client.post("/predict", files={"file": ("a.png", _png((200, 199, 0)), "image/png")})
    assert r.json()["lookalike_review"] is False


def test_extreme_aspect_ratio_is_cheap_and_correct(client):
    from faunavision import api

    clf = api.get_classifier()
    x = clf.preprocess(Image.new("RGB", (20_000, 3), (255, 0, 0)))
    assert x.shape == (1, 3, 32, 32) and np.allclose(x[0, 0], 1.0)


def test_rejects_too_many_pixels(client, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    assert client.post("/predict", files={"file": ("a.png", _png((1, 2, 3), (40, 40)), "image/png")}).status_code == 413


def test_health_hides_error_details(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from faunavision import api

    monkeypatch.setenv("FAUNAVISION_MODEL", str(tmp_path / "secret_dir" / "missing.onnx"))
    api.get_classifier.cache_clear()
    r = TestClient(api.app).get("/health")
    assert r.status_code == 503 and "secret_dir" not in r.text
    api.get_classifier.cache_clear()
