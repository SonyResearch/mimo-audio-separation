"""Spectral-loss regressions; no model weights or training data are required."""

import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "spectral_loss", Path(__file__).resolve().parents[1] / "src/model/loss/spec.py"
)
spectral_loss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectral_loss)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("signal", ["silence", "random", "mixed"])
def test_rmse_exact_matches_have_finite_gradients(dtype, signal):
    torch.manual_seed(0)
    x = torch.randn(2, 2, 2, 512, dtype=dtype)
    if signal == "silence":
        x.zero_()
    y = x.clone()
    if signal == "mixed":
        y[:, 1] += torch.randn_like(y[:, 1])
    x.requires_grad_()
    y.requires_grad_()
    criterion = spectral_loss.MRSTFT_RMSELoss(
        n_ffts=[64, 128], hop_sizes=[16, 32]
    )
    loss = criterion(x, y)["mrstft/rmse"]
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()
    if signal == "mixed":
        assert torch.count_nonzero(x.grad[:, 0]) == 0
        assert torch.count_nonzero(x.grad[:, 1]) > 0
    else:
        torch.testing.assert_close(loss, loss.new_tensor(criterion.eps))
        assert torch.count_nonzero(x.grad) == 0
        assert torch.count_nonzero(y.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rmse_preserves_nonzero_loss_and_gradients(dtype):
    torch.manual_seed(1)
    x = torch.randn(2, 3, 2, 512, dtype=dtype, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    criterion = spectral_loss.MRSTFT_RMSELoss(
        n_ffts=[64, 128], hop_sizes=[16, 32], win_sizes=[48, 96]
    )
    actual = criterion(x, y)["mrstft/rmse"]

    # Independently reproduce the original per-source, multi-resolution RMSE.
    expected = x.new_zeros(())
    for n_fft, hop, win in zip(
        criterion.n_ffts, criterion.hop_sizes, criterion.win_sizes
    ):
        window = torch.hann_window(win)
        kwargs = dict(
            n_fft=n_fft, hop_length=hop, win_length=win, window=window,
            center=True, normalized=False, onesided=True, return_complex=True
        )
        sx = torch.view_as_real(torch.stft(x.reshape(-1, 512), **kwargs))
        sy = torch.view_as_real(torch.stft(y.reshape(-1, 512), **kwargs))
        squared = (sx - sy).square().reshape(2, 3, -1)
        expected = expected + (squared.mean(-1).sqrt() + criterion.eps).mean()
    expected = expected / len(criterion.n_ffts)

    actual_grads = torch.autograd.grad(actual, (x, y), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, y))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
