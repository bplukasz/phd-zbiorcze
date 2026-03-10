"""Tests for WaveletStatRegularizer (Etap 6) and FFTStatRegularizer (Etap 7)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
R3GAN_SOURCE_PATH = EXPERIMENT_ROOT / "r3gan-source.py"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

spec = importlib.util.spec_from_file_location("e001_02_r3gan_source", R3GAN_SOURCE_PATH)
assert spec is not None and spec.loader is not None
r3gan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r3gan
spec.loader.exec_module(r3gan)

WaveletStatRegularizer = r3gan.WaveletStatRegularizer
FFTStatRegularizer = r3gan.FFTStatRegularizer
WaveReg = r3gan.WaveReg
FFTReg = r3gan.FFTReg


# ---- Aliases ---------------------------------------------------------------

def test_aliases_are_correct_classes():
    assert WaveReg is WaveletStatRegularizer
    assert FFTReg is FFTStatRegularizer


# ---- WaveletStatRegularizer ------------------------------------------------

class TestWaveletStatRegularizer:

    def _make_reg(self, **kwargs):
        defaults = dict(weight=0.02, ema_beta=0.99, eps=1e-8, in_channels=3)
        defaults.update(kwargs)
        return WaveletStatRegularizer(**defaults)

    def test_returns_tuple_of_loss_and_dict(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        result = reg(fake, real)
        assert isinstance(result, tuple) and len(result) == 2
        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_loss_is_scalar(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        loss, _ = reg(fake, real)
        assert loss.shape == ()

    def test_metrics_keys_present(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        _, metrics = reg(fake, real)
        expected_keys = {
            "wave_reg_total", "wave_mu_loss", "wave_std_loss",
            "wave_fake_mu_lh", "wave_fake_mu_hl", "wave_fake_mu_hh",
            "wave_real_mu_lh", "wave_real_mu_hl", "wave_real_mu_hh",
        }
        assert expected_keys <= set(metrics.keys()), \
            f"Missing keys: {expected_keys - set(metrics.keys())}"

    def test_metrics_are_floats(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        _, metrics = reg(fake, real)
        for k, v in metrics.items():
            assert isinstance(v, float), f"metric '{k}' is {type(v)}, expected float"

    def test_loss_has_gradient(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32, requires_grad=True)
        real = torch.randn(4, 3, 32, 32)
        loss, _ = reg(fake, real)
        loss.backward()
        assert fake.grad is not None
        assert fake.grad.abs().sum().item() > 0.0

    def test_ema_buffers_initialized_after_first_forward(self):
        reg = self._make_reg()
        assert not reg._initialized.item()  # type: ignore[union-attr]
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        reg(fake, real)
        assert reg._initialized.item()  # type: ignore[union-attr]

    def test_ema_buffers_update_over_calls(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        reg(fake, real)
        mu_after_1 = reg.mu_real_ema.clone()
        # Second call with different real
        real2 = torch.randn(4, 3, 32, 32) * 3.0  # much higher energy
        reg(fake, real2)
        mu_after_2 = reg.mu_real_ema.clone()
        # EMA should have changed
        assert not torch.allclose(mu_after_1, mu_after_2)

    def test_loss_approaches_zero_when_distributions_match(self):
        """When fake and real come from the same distribution, loss should be small."""
        torch.manual_seed(0)
        reg = self._make_reg(ema_beta=0.0)  # ema_beta=0 → EMA = current batch immediately
        x = torch.randn(16, 3, 32, 32)
        loss, _ = reg(x.clone(), x.clone())
        assert loss.item() < 0.1, f"loss too high: {loss.item()}"

    def test_loss_larger_when_distributions_differ(self):
        torch.manual_seed(0)
        reg = self._make_reg(ema_beta=0.0)
        fake = torch.randn(16, 3, 32, 32) * 0.01   # very low energy
        real = torch.randn(16, 3, 32, 32) * 10.0   # very high energy
        # Warm up real EMA
        reg(fake.detach(), real)
        loss, _ = reg(fake, real)
        assert loss.item() > 0.01, f"loss unexpectedly small: {loss.item()}"

    def test_works_with_single_channel(self):
        reg = WaveletStatRegularizer(in_channels=1)
        fake = torch.randn(4, 1, 32, 32)
        real = torch.randn(4, 1, 32, 32)
        loss, metrics = reg(fake, real)
        assert loss.shape == ()


# ---- FFTStatRegularizer ----------------------------------------------------

class TestFFTStatRegularizer:

    def _make_reg(self, **kwargs):
        defaults = dict(weight=0.02, ema_beta=0.99, eps=1e-8, num_bins=16)
        defaults.update(kwargs)
        return FFTStatRegularizer(**defaults)

    def test_returns_tuple_of_loss_and_dict(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        result = reg(fake, real)
        assert isinstance(result, tuple) and len(result) == 2
        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_loss_is_scalar(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        loss, _ = reg(fake, real)
        assert loss.shape == ()

    def test_metrics_keys_present(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        _, metrics = reg(fake, real)
        expected_keys = {"fft_reg_total", "fft_mu_loss", "fft_std_loss"}
        assert expected_keys <= set(metrics.keys()), \
            f"Missing keys: {expected_keys - set(metrics.keys())}"

    def test_metrics_are_floats(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        _, metrics = reg(fake, real)
        for k, v in metrics.items():
            assert isinstance(v, float), f"metric '{k}' is {type(v)}, expected float"

    def test_loss_has_gradient(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32, requires_grad=True)
        real = torch.randn(4, 3, 32, 32)
        loss, _ = reg(fake, real)
        loss.backward()
        assert fake.grad is not None
        assert fake.grad.abs().sum().item() > 0.0

    def test_ema_buffers_initialized_after_first_forward(self):
        reg = self._make_reg()
        assert not reg._initialized.item()  # type: ignore[union-attr]
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        reg(fake, real)
        assert reg._initialized.item()  # type: ignore[union-attr]

    def test_ema_buffers_update_over_calls(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        reg(fake, real)
        mu_after_1 = reg.mu_real_ema.clone()
        real2 = torch.randn(4, 3, 32, 32) * 5.0
        reg(fake, real2)
        mu_after_2 = reg.mu_real_ema.clone()
        assert not torch.allclose(mu_after_1, mu_after_2)

    def test_loss_approaches_zero_when_distributions_match(self):
        torch.manual_seed(0)
        reg = self._make_reg(ema_beta=0.0)
        x = torch.randn(16, 3, 32, 32)
        loss, _ = reg(x.clone(), x.clone())
        assert loss.item() < 0.1, f"loss too high: {loss.item()}"

    def test_loss_larger_when_distributions_differ(self):
        torch.manual_seed(0)
        reg = self._make_reg(ema_beta=0.0)
        fake = torch.randn(16, 3, 32, 32) * 0.01
        real = torch.randn(16, 3, 32, 32) * 10.0
        reg(fake.detach(), real)  # warm up EMA
        loss, _ = reg(fake, real)
        assert loss.item() > 0.01, f"loss unexpectedly small: {loss.item()}"

    def test_num_bins_param_respected(self):
        reg = FFTStatRegularizer(num_bins=8)
        assert reg.mu_real_ema.shape == (8,)
        assert reg.std_real_ema.shape == (8,)
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        loss, _ = reg(fake, real)
        assert loss.shape == ()

    def test_bin_map_is_cached(self):
        reg = self._make_reg()
        fake = torch.randn(4, 3, 32, 32)
        real = torch.randn(4, 3, 32, 32)
        reg(fake, real)
        bm1 = reg._bin_map
        reg(fake, real)
        bm2 = reg._bin_map
        assert bm1 is bm2  # same object — cache hit


# ---- Parallel structure check ----------------------------------------------

def test_both_regularizers_same_interface():
    """Confirm WaveletStatReg and FFTStatReg both return (Tensor, dict[str, float])."""
    wave = WaveletStatRegularizer()
    fft = FFTStatRegularizer()
    fake = torch.randn(4, 3, 32, 32)
    real = torch.randn(4, 3, 32, 32)
    for reg in (wave, fft):
        out = reg(fake, real)
        assert isinstance(out, tuple) and len(out) == 2
        loss, metrics = out
        assert isinstance(loss, torch.Tensor) and loss.shape == ()
        assert all(isinstance(v, float) for v in metrics.values())
