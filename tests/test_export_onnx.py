import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

_spec = importlib.util.spec_from_file_location(
    "export_onnx", Path(__file__).resolve().parents[1] / "scripts" / "export_onnx.py")
export_onnx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_onnx)


def test_export_round_trip_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=7).eval()
    out = tmp_path / "m.onnx"
    export_onnx.export(model, 64, out)
    assert export_onnx.verify(model, out, 64, batch=3) < 1e-3  # batch 3 also checks the dynamic axis
