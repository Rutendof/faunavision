"""Export a trained checkpoint (usually the model 6 student) to ONNX and verify it.

Example:
    python scripts/export_onnx.py --ckpt checkpoints/student_b0/best.pt --out exports/student_b0.onnx \\
        --metrics reports/results/student_b0/metrics.json

The graph takes a normalised float32 batch (N, 3, S, S) and returns logits
(N, num_classes); the batch axis is dynamic. A sidecar JSON next to the .onnx
records class names, input size and mean/std so the API can preprocess
without importing timm. After export, ONNX Runtime outputs are compared with
PyTorch on random inputs and the script fails if they diverge.

Requires the optional extra: pip install -e ".[export]"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.dataset import CLASS_NAMES  # noqa: E402


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("classes") != CLASS_NAMES:
        raise ValueError(f"{path} was trained on a different class list")
    name, size = ck["model_name"], ck["img_size"]
    kwargs = {"img_size": size} if "vit" in name else {}
    model = timm.create_model(name, pretrained=False, num_classes=len(CLASS_NAMES), **kwargs)
    model.load_state_dict(ck["model"])
    cfg = timm.data.resolve_model_data_config(model)
    meta = {"model_name": name, "img_size": size, "classes": CLASS_NAMES,
            "mean": list(cfg["mean"]), "std": list(cfg["std"]), "input": "input", "output": "logits"}
    return model.eval(), meta


def export(model: torch.nn.Module, img_size: int, out: Path, opset: int = 17) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, (dummy,), str(out), input_names=["input"], output_names=["logits"],
                      dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},), opset_version=opset,
                      external_data=False)  # one self-contained file (fine below protobuf's 2 GB limit)


def verify(model: torch.nn.Module, onnx_path: Path, img_size: int, batch: int = 4,
           atol: float = 1e-3) -> float:
    """Max |logit difference| between PyTorch and ONNX Runtime; raises if above atol or top-1 differs."""
    import onnxruntime as ort

    x = torch.randn(batch, 3, img_size, img_size)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > atol or not (ref.argmax(1) == got.argmax(1)).all():
        raise RuntimeError(f"ONNX output diverges from PyTorch (max diff {diff:.2e})")
    return diff


def main() -> None:
    # torch.onnx's progress messages contain emoji, which crash cp1252 Windows consoles
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    p =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True, help="best.pt written by scripts/train.py")
    p.add_argument("--out", type=Path, required=True, help="output .onnx path")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--metrics", type=Path, default=None,
                   help="the run's reports/results/<run>/metrics.json: copies the val-fitted temperature")
    args = p.parse_args()

    model, meta = load_checkpoint(args.ckpt)
    if args.metrics:
        meta["temperature"] = json.loads(args.metrics.read_text(encoding="utf-8"))["val"]["temperature"]
    export(model, meta["img_size"], args.out, args.opset)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    diff = verify(model, args.out, meta["img_size"])
    size_mb = args.out.stat().st_size / 2**20
    print(f"{meta['model_name']} -> {args.out} ({size_mb:.1f} MB), max |diff| vs PyTorch {diff:.2e}")


if __name__ == "__main__":
    main()
