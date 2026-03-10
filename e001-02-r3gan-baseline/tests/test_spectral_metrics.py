"""
Tests for RPSE and WBED spectral metrics in gan_metrics.py

Run with:
    cd e001-02-r3gan-baseline && pytest tests/test_spectral_metrics.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gan_metrics import (
    GANMetricsConfig,
    compute_radial_power_spectrum,
    compute_rpse,
    compute_wavelet_band_energies,
    compute_wbed,
    _haar_dwt2d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(b: int = 8, c: int = 3, h: int = 32, w: int = 32, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


# ===========================================================================
# _haar_dwt2d
# ===========================================================================

class TestHaarDWT2d:
    def test_output_shape(self):
        x = make_batch(4, 3, 32, 32)
        out = _haar_dwt2d(x)
        assert set(out.keys()) == {"LL", "LH", "HL", "HH"}
        for band, tensor in out.items():
            assert tensor.shape == (4, 3, 16, 16), f"{band} wrong shape {tensor.shape}"

    def test_single_channel(self):
        x = make_batch(2, 1, 16, 16)
        out = _haar_dwt2d(x)
        for band in ("LL", "LH", "HL", "HH"):
            assert out[band].shape == (2, 1, 8, 8)

    def test_rejects_odd_size(self):
        x = torch.rand(2, 3, 31, 32)
        with pytest.raises(ValueError, match="even"):
            _haar_dwt2d(x)

    def test_rejects_3d_input(self):
        x = torch.rand(3, 32, 32)
        with pytest.raises(ValueError, match="B, C, H, W"):
            _haar_dwt2d(x)

    def test_ll_preserves_mean(self):
        """LL subband (normalised) should be close to image mean because it
        approximates a downsampled average."""
        x = torch.ones(2, 1, 8, 8) * 0.5
        out = _haar_dwt2d(x)
        # With constant input, LL = 0.5 * 0.5 + 0.5 * 0.5 + 0.5 * 0.5 + 0.5 * 0.5 = 1.0
        # (sum of four 0.5 * weight contributions where weights sum to 2)
        assert out["LL"].allclose(torch.ones_like(out["LL"]), atol=1e-5)

    def test_hf_bands_zero_for_constant_image(self):
        """High-frequency subbands must be exactly zero for a constant image."""
        x = torch.ones(2, 3, 16, 16) * 0.42
        out = _haar_dwt2d(x)
        for band in ("LH", "HL", "HH"):
            assert out[band].abs().max().item() < 1e-6, f"{band} should be 0 for constant image"

    def test_perfect_reconstruction(self):
        """Verify that DWT coefficients carry all signal energy (Parseval-like)."""
        x = make_batch(4, 3, 32, 32)
        out = _haar_dwt2d(x)
        # Sum of squared coefficients across all bands should equal sum of input squared
        # (the Haar filters form an orthogonal frame so energy is preserved)
        input_energy = (x ** 2).sum().item()
        coeff_energy = sum((v ** 2).sum().item() for v in out.values())
        assert abs(input_energy - coeff_energy) / (input_energy + 1e-8) < 1e-4


# ===========================================================================
# compute_radial_power_spectrum
# ===========================================================================

class TestRadialPowerSpectrum:
    def test_output_shape_default_bins(self):
        x = make_batch(8, 3, 32, 32)
        profile = compute_radial_power_spectrum(x)
        assert profile.shape == (16,)  # min(32,32)//2

    def test_output_shape_custom_bins(self):
        x = make_batch(4, 3, 64, 64)
        profile = compute_radial_power_spectrum(x, num_bins=20)
        assert profile.shape == (20,)

    def test_all_positive(self):
        x = make_batch(8, 3, 32, 32)
        profile = compute_radial_power_spectrum(x)
        assert (profile >= 0).all(), "Profile values must be non-negative"

    def test_constant_image_has_dc_peak(self):
        """Constant image: all energy at DC (bin 0 = lowest freq)."""
        x = torch.ones(4, 3, 32, 32) * 0.5
        profile = compute_radial_power_spectrum(x)
        assert profile[0].item() > profile[-1].item()

    def test_same_input_gives_zero_rpse(self):
        """Same profile → RPSE = 0."""
        x = make_batch(16, 3, 32, 32)
        p = compute_radial_power_spectrum(x)
        rpse = compute_rpse(p, p)
        assert rpse == pytest.approx(0.0, abs=1e-6)

    def test_different_input_gives_positive_rpse(self):
        """Inputs with different spectral content → RPSE > 0."""
        # Low-freq dominated
        x_low = torch.ones(8, 1, 32, 32) * 0.5
        # High-freq dominated (white noise)
        torch.manual_seed(99)
        x_noise = torch.rand(8, 1, 32, 32)
        p_low = compute_radial_power_spectrum(x_low)
        p_noise = compute_radial_power_spectrum(x_noise)
        rpse = compute_rpse(p_low, p_noise)
        assert rpse > 1e-4, f"Expected positive RPSE, got {rpse}"

    def test_rpse_symmetric(self):
        """RPSE(a,b) == RPSE(b,a)."""
        a = make_batch(8, 3, 32, 32, seed=1)
        b = make_batch(8, 3, 32, 32, seed=2)
        pa = compute_radial_power_spectrum(a)
        pb = compute_radial_power_spectrum(b)
        assert compute_rpse(pa, pb) == pytest.approx(compute_rpse(pb, pa), rel=1e-5)

    def test_rpse_upper_bound(self):
        """Normalised profiles sum to 1, so max L2 dist is sqrt(2)."""
        a = make_batch(8, 3, 32, 32, seed=3)
        b = make_batch(8, 3, 32, 32, seed=4)
        pa = compute_radial_power_spectrum(a)
        pb = compute_radial_power_spectrum(b)
        assert compute_rpse(pa, pb) <= 2.0 ** 0.5 + 1e-5


# ===========================================================================
# compute_wavelet_band_energies
# ===========================================================================

class TestWaveletBandEnergies:
    def test_output_keys_and_shapes(self):
        x = make_batch(8, 3, 32, 32)
        energies = compute_wavelet_band_energies(x)
        assert set(energies.keys()) == {"LL", "LH", "HL", "HH"}
        for band, e in energies.items():
            assert e.shape == (8,), f"{band} energy should be (B,), got {e.shape}"

    def test_all_positive(self):
        x = make_batch(8, 3, 32, 32)
        energies = compute_wavelet_band_energies(x)
        for band, e in energies.items():
            assert (e >= 0).all(), f"{band} energy must be non-negative"

    def test_constant_image_hf_zero(self):
        """Constant image → LH/HL/HH energy = 0 per image."""
        x = torch.ones(4, 3, 32, 32) * 0.7
        energies = compute_wavelet_band_energies(x)
        for band in ("LH", "HL", "HH"):
            assert energies[band].abs().max().item() < 1e-6


# ===========================================================================
# compute_wbed
# ===========================================================================

class TestWBED:
    def test_identical_inputs_give_zero(self):
        """Same distribution → WBED = 0."""
        x = make_batch(32, 3, 32, 32, seed=0)
        e = compute_wavelet_band_energies(x)
        result = compute_wbed(e, e)
        assert result["wbed_total"] == pytest.approx(0.0, abs=1e-6)
        for band in ("ll", "lh", "hl", "hh"):
            assert result[f"wbed_{band}_dist"] == pytest.approx(0.0, abs=1e-6)

    def test_different_inputs_give_positive(self):
        """Different distributions → WBED > 0."""
        # Constant vs noisy
        x_const = torch.ones(32, 3, 32, 32) * 0.5
        torch.manual_seed(7)
        x_noise = torch.rand(32, 3, 32, 32)
        e_real = compute_wavelet_band_energies(x_const)
        e_fake = compute_wavelet_band_energies(x_noise)
        result = compute_wbed(e_real, e_fake)
        assert result["wbed_total"] > 1e-4, f"Expected positive WBED, got {result['wbed_total']}"

    def test_output_keys(self):
        x = make_batch(16, 3, 32, 32)
        e = compute_wavelet_band_energies(x)
        result = compute_wbed(e, e)
        expected_keys = {"wbed_total"}
        for band in ("ll", "lh", "hl", "hh"):
            expected_keys |= {f"wbed_{band}_mean_diff", f"wbed_{band}_std_diff", f"wbed_{band}_dist"}
        assert set(result.keys()) == expected_keys

    def test_wbed_total_equals_sum_of_band_dists(self):
        a = make_batch(32, 3, 32, 32, seed=10)
        b = make_batch(32, 3, 32, 32, seed=11)
        ea = compute_wavelet_band_energies(a)
        eb = compute_wavelet_band_energies(b)
        result = compute_wbed(ea, eb)
        band_sum = sum(result[f"wbed_{band}_dist"] for band in ("ll", "lh", "hl", "hh"))
        assert result["wbed_total"] == pytest.approx(band_sum, rel=1e-5)

    def test_wbed_non_negative(self):
        a = make_batch(32, 3, 32, 32, seed=20)
        b = make_batch(32, 3, 32, 32, seed=21)
        ea = compute_wavelet_band_energies(a)
        eb = compute_wavelet_band_energies(b)
        result = compute_wbed(ea, eb)
        for k, v in result.items():
            assert v >= 0.0, f"Key {k} has negative value {v}"


# ===========================================================================
# GANMetricsConfig: spectral flags
# ===========================================================================

class TestGANMetricsConfigSpectral:
    def test_defaults_disabled(self):
        cfg = GANMetricsConfig()
        assert cfg.spectral_enabled is False

    def test_can_enable(self):
        cfg = GANMetricsConfig(spectral_enabled=True, spectral_num_images=512)
        assert cfg.spectral_enabled is True
        assert cfg.spectral_num_images == 512

    def test_custom_bins(self):
        cfg = GANMetricsConfig(spectral_enabled=True, spectral_rpse_num_bins=32)
        assert cfg.spectral_rpse_num_bins == 32

    def test_none_bins_means_auto(self):
        cfg = GANMetricsConfig(spectral_rpse_num_bins=None)
        assert cfg.spectral_rpse_num_bins is None


# ===========================================================================
# End-to-end: prepare_real + evaluate_generator with spectral_enabled
# ===========================================================================

class TestSpectralEndToEnd:
    """Lightweight CPU-only integration test using mocked FID/KID/LPIPS."""

    def _make_suite(self, num_images: int = 64) -> "GANMetricsSuite":  # noqa: F821
        from unittest.mock import MagicMock, patch
        from src.gan_metrics import GANMetricsSuite

        cfg = GANMetricsConfig(
            device="cpu",
            input_range="zero_one",
            fid_feature=64,
            kid_feature=64,
            kid_subsets=2,
            kid_subset_size=20,
            max_real_images_fid_kid=num_images,
            pr_num_samples=num_images,
            pr_k=3,
            lpips_num_pairs=8,
            lpips_pool_size=16,
            use_amp_for_feature_extractor=False,
            use_channels_last=False,
            spectral_enabled=True,
            spectral_num_images=num_images,
        )

        suite = GANMetricsSuite.__new__(GANMetricsSuite)
        suite.cfg = cfg
        suite.device = torch.device("cpu")
        suite.pr_device = torch.device("cpu")

        # Stub heavy dependencies
        suite.fid = MagicMock()
        suite.fid.compute.return_value = torch.tensor(10.0)
        suite.kid = MagicMock()
        suite.kid.compute.return_value = (torch.tensor(0.01), torch.tensor(0.001))
        suite.lpips_metric = MagicMock()
        suite.lpips_metric.return_value = torch.tensor(0.3)

        # Stub PR extractor with random 2048-dim features
        def fake_extractor(imgs):
            return torch.randn(imgs.shape[0], 2048)
        suite.pr_extractor = MagicMock(side_effect=fake_extractor)

        suite._real_prepared = False
        suite._real_feature_cache = None
        suite._real_image_count_fid_kid = 0
        suite._last_fake_feature_cache = None
        suite._last_fake_lpips_pool = None
        suite._real_spectral_images = None
        import random
        suite._rng = random.Random(42)

        return suite

    def test_spectral_keys_present_in_output(self):
        n = 64
        suite = self._make_suite(n)

        real_imgs = [torch.rand(n, 3, 32, 32)]
        suite.prepare_real(iter(real_imgs))

        def sample_fn(bs):
            return torch.rand(bs, 3, 32, 32)

        metrics = suite.evaluate_generator(sample_fn, num_fake_images=n, fake_batch_size=n)
        assert "rpse" in metrics, "RPSE missing from output"
        assert "wbed" in metrics, "WBED missing from output"
        assert metrics["rpse"] >= 0.0
        assert metrics["wbed"] >= 0.0

    def test_spectral_not_present_when_disabled(self):
        from src.gan_metrics import GANMetricsSuite
        from unittest.mock import MagicMock
        n = 64
        suite = self._make_suite(n)
        suite.cfg.spectral_enabled = False

        real_imgs = [torch.rand(n, 3, 32, 32)]
        suite.prepare_real(iter(real_imgs))

        def sample_fn(bs):
            return torch.rand(bs, 3, 32, 32)

        metrics = suite.evaluate_generator(sample_fn, num_fake_images=n, fake_batch_size=n)
        assert "rpse" not in metrics
        assert "wbed" not in metrics

    def test_identical_generator_gives_low_rpse(self):
        """If fake images are identical to real images, RPSE should be near 0."""
        n = 64
        torch.manual_seed(42)
        fixed_imgs = torch.rand(n, 3, 32, 32)

        suite = self._make_suite(n)
        suite.prepare_real(iter([fixed_imgs]))

        # Generator returns same images
        idx = [0]
        def sample_fn(bs):
            return fixed_imgs[:bs]

        metrics = suite.evaluate_generator(sample_fn, num_fake_images=n, fake_batch_size=n)
        assert metrics["rpse"] < 0.05, f"RPSE={metrics['rpse']:.6f} too high for identical images"
        assert metrics["wbed"] < 0.1, f"WBED={metrics['wbed']:.6f} too high for identical images"

    def test_different_generator_gives_higher_rpse(self):
        """Constant fake vs noisy real → RPSE should be higher than identical case."""
        n = 64
        torch.manual_seed(0)
        real_imgs = torch.rand(n, 3, 32, 32)
        fake_imgs = torch.ones(n, 3, 32, 32) * 0.5  # constant = DC-only spectrum

        suite = self._make_suite(n)
        suite.prepare_real(iter([real_imgs]))

        def sample_fn(bs):
            return fake_imgs[:bs]

        metrics = suite.evaluate_generator(sample_fn, num_fake_images=n, fake_batch_size=n)
        assert metrics["rpse"] > 0.01, f"Expected RPSE > 0 for different inputs, got {metrics['rpse']}"

    def test_real_spectral_images_shape(self):
        n = 64
        suite = self._make_suite(n)
        real_imgs = [torch.rand(n, 3, 32, 32)]
        suite.prepare_real(iter(real_imgs))
        assert suite._real_spectral_images is not None
        assert suite._real_spectral_images.shape == (n, 3, 32, 32)

