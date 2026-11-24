import pytest

torch = pytest.importorskip("torch")

from faunavision.distill import TeacherAdapter, kd_loss  # noqa: E402


def test_kd_loss_reduces_to_ce_when_alpha_zero():
    s, t = torch.randn(8, 5), torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    assert torch.allclose(kd_loss(s, t, y, alpha=0.0), torch.nn.functional.cross_entropy(s, y))


def test_kd_soft_term_is_zero_when_student_matches_teacher():
    s = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    assert kd_loss(s, s.clone(), y, alpha=1.0, temperature=3.0).abs() < 1e-6


def test_kd_loss_gradient_pulls_student_towards_teacher():
    t = torch.tensor([[4.0, 0.0, 0.0]])
    s = torch.zeros(1, 3, requires_grad=True)
    kd_loss(s, t, torch.tensor([0]), alpha=1.0).backward()
    assert s.grad[0, 0] < 0 < s.grad[0, 1]  # descending raises class 0, lowers the others


def test_teacher_adapter_renormalises_resizes_and_stays_frozen():
    seen = {}

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            seen["x"] = x
            return x.mean(dim=(2, 3)) * self.w

    raw = torch.rand(2, 3, 16, 16)
    s_mean, s_std, t_mean, t_std = [0.5] * 3, [0.25] * 3, [0.4] * 3, [0.2] * 3
    student_in = (raw - 0.5) / 0.25
    adapter = TeacherAdapter(Probe(), s_mean, s_std, t_mean, t_std, teacher_size=16)
    adapter.train()
    out = adapter(student_in)
    assert torch.allclose(seen["x"], (raw - 0.4) / 0.2, atol=1e-5)
    assert not out.requires_grad and not adapter.teacher.training

    TeacherAdapter(Probe(), s_mean, s_std, t_mean, t_std, teacher_size=8)(student_in)
    assert seen["x"].shape[-2:] == (8, 8)
