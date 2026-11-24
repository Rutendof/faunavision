"""Knowledge distillation helpers (model 6: small student learns from model 5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
            alpha: float = 0.7, temperature: float = 2.0, label_smoothing: float = 0.0) -> torch.Tensor:
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, labels).

    The T^2 factor keeps the soft-target gradient scale independent of T
    (Hinton et al., 2015).
    """
    t = temperature
    soft = F.kl_div(F.log_softmax(student_logits.float() / t, dim=1),
                    F.log_softmax(teacher_logits.float() / t, dim=1),
                    reduction="batchmean", log_target=True) * (t * t)
    hard = F.cross_entropy(student_logits.float(), labels, label_smoothing=label_smoothing)
    return alpha * soft + (1 - alpha) * hard


class TeacherAdapter(torch.nn.Module):
    """Frozen teacher fed the student's batch: re-normalised and resized to its own input.

    Student and teacher may use different mean/std and resolutions (e.g. a
    DINOv2 teacher at 448 px, an EfficientNet-B0 student at 224 px).
    """

    def __init__(self, teacher: torch.nn.Module, student_mean, student_std,
                 teacher_mean, teacher_std, teacher_size: int):
        super().__init__()
        self.teacher = teacher.eval().requires_grad_(False)
        shape = (1, 3, 1, 1)
        self.register_buffer("s_mean", torch.tensor(student_mean).view(shape))
        self.register_buffer("s_std", torch.tensor(student_std).view(shape))
        self.register_buffer("t_mean", torch.tensor(teacher_mean).view(shape))
        self.register_buffer("t_std", torch.tensor(teacher_std).view(shape))
        self.size = teacher_size

    def train(self, mode: bool = True):  # the teacher always stays in eval mode
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.s_std + self.s_mean - self.t_mean) / self.t_std
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        return self.teacher(x)
