"""Contrastive Grad-CAM: *why species A and not lookalike B?*

Standard Grad-CAM explains one class score. Here the target is the
difference ``logit[A] - logit[B]``, so the heatmap highlights regions that
favour A **over** B (e.g. the square lip of a white rhino versus the hooked
lip of a black rhino), not just "where the rhino is".

Works with CNNs (last conv feature map) and ViTs (last block's patch
tokens, reshaped to the patch grid). These maps are diagnostic aids, not
proof that the model reasons like an expert.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "blocks"):  # timm ViT
        return model.blocks[-1]
    if hasattr(model, "layer4"):  # ResNet
        return model.layer4
    if hasattr(model, "stages"):  # ConvNeXt / EfficientNet-style
        return model.stages[-1]
    raise ValueError("Pass target_layer explicitly for this architecture")


def _to_grid(act: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """(B, C, H, W) for CNNs; ViT tokens (B, N, C) -> (B, C, h, w)."""
    if act.ndim == 4:
        return act
    prefix = getattr(model, "num_prefix_tokens", 1)
    tokens = act[:, prefix:, :]
    side = int(round(tokens.shape[1] ** 0.5))
    return tokens.transpose(1, 2).reshape(act.shape[0], act.shape[2], side, side)


def contrastive_gradcam(model, x: torch.Tensor, class_a: int, class_b: int | None = None,
                        target_layer: torch.nn.Module | None = None) -> np.ndarray:
    """Heatmap in [0, 1] (H x W of the input) for evidence of A over B.

    With ``class_b=None`` it falls back to ordinary Grad-CAM for A.
    """
    model.eval()
    layer = target_layer or default_target_layer(model)
    store: dict[str, torch.Tensor] = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x)
        target = logits[0, class_a] - (logits[0, class_b] if class_b is not None else 0)
        model.zero_grad(set_to_none=True)
        target.backward()
    finally:
        h1.remove()
        h2.remove()

    act, grad = _to_grid(store["act"].detach(), model), _to_grid(store["grad"].detach(), model)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max()).numpy() if cam.max() > 0 else cam.numpy()


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Heatmap (ochre-to-red) blended over the image, at the image's size."""
    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.Resampling.BILINEAR)
    h = np.asarray(heat, dtype=np.float32)[..., None] / 255.0
    colour = np.concatenate([np.ones_like(h) * 220, 60 + 120 * (1 - h), np.ones_like(h) * 30], axis=2)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    mixed = base * (1 - alpha * h) + colour * alpha * h
    return Image.fromarray(mixed.clip(0, 255).astype(np.uint8))
