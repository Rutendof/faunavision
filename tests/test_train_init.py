import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

_spec = importlib.util.spec_from_file_location("train", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def test_init_from_lower_resolution_vit_resamples_pos_embed():
    src = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=5, img_size=224)
    dst = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=5, img_size=448)
    dst.load_state_dict(train.adapt_state_to_model(src.state_dict(), dst))  # raised size mismatch before
    assert torch.equal(dst.pos_embed[:, :1], src.pos_embed[:, :1])  # class-token embedding kept as is
    dst.eval()(torch.randn(1, 3, 448, 448))


def test_init_same_shape_state_is_unchanged():
    m = timm.create_model("resnet18", pretrained=False, num_classes=5)
    state = m.state_dict()
    assert train.adapt_state_to_model(state, m) is state
