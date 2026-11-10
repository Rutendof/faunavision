import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from faunavision.explain import contrastive_gradcam, overlay  # noqa: E402


@pytest.mark.parametrize("name", ["resnet18", "vit_tiny_patch16_224"])
def test_contrastive_gradcam_shape_and_range(name):
    torch.manual_seed(0)
    model = timm.create_model(name, pretrained=False, num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    cam = contrastive_gradcam(model, x, class_a=1, class_b=2)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0 and cam.max() <= 1


def test_contrast_differs_from_plain_gradcam():
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    plain = contrastive_gradcam(model, x, class_a=1)
    contrast = contrastive_gradcam(model, x, class_a=1, class_b=2)
    assert not np.allclose(plain, contrast)


def test_overlay_keeps_image_size():
    from PIL import Image
    img = Image.new("RGB", (300, 200), "grey")
    out = overlay(img, np.random.rand(224, 224))
    assert out.size == (300, 200)
