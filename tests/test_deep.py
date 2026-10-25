import pytest

torch = pytest.importorskip("torch")
from owerf_deep_encoders import (OWERFEncoder, episodic_loss, gaussian_nll,
                                 inconsistency_ratio)


def _batch(B=8, L=10, Lc=12, T=6):
    torch.manual_seed(0)
    hist = torch.randn(B, L, 6)
    codes = torch.randint(1, 500, (B, Lc))
    ctx = torch.randn(B, T, 16)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, -2:] = True
    exo = torch.randn(B, 3)
    return hist, codes, ctx, pad, exo


def test_forward_shapes():
    model = OWERFEncoder()
    h, logits, mu, log_sigma = model(*_batch())
    assert h.shape == (8, 64) and logits.shape == (8, 4)
    assert mu.shape == (8, 2) and log_sigma.shape == (8, 2)


def test_backward_gives_gradients():
    model = OWERFEncoder()
    h, logits, mu, log_sigma = model(*_batch())
    y = torch.randint(0, 4, (8,))
    loss = episodic_loss(logits, y, [1, 2, 3]) + gaussian_nll(mu, log_sigma, torch.randn(8, 2))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_nll_prefers_correct_mean_and_ratio_nonnegative():
    target = torch.zeros(5, 2)
    log_sigma = torch.zeros(5, 2)
    good = gaussian_nll(torch.zeros(5, 2), log_sigma, target)
    bad = gaussian_nll(torch.full((5, 2), 3.0), log_sigma, target)
    assert good < bad
    assert (inconsistency_ratio(torch.zeros(5, 2), log_sigma, torch.randn(5, 2)) >= 0).all()
