from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
R3GAN_SOURCE_PATH = EXPERIMENT_ROOT / "r3gan-source.py"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

spec = importlib.util.spec_from_file_location("e001_02_r3gan_source", R3GAN_SOURCE_PATH)
assert spec is not None and spec.loader is not None
r3gan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r3gan
spec.loader.exec_module(r3gan)

from src.config_loader import RunConfig
import src.experiment as experiment_module

WaveletHFBranch = r3gan.WaveletHFBranch
MatchedCapacityBranch = r3gan.MatchedCapacityBranch
R3GANDiscriminator = r3gan.R3GANDiscriminator
WaveletR3GANDiscriminator = r3gan.WaveletR3GANDiscriminator
MatchedCapacityR3GANDiscriminator = r3gan.MatchedCapacityR3GANDiscriminator
build_stage_channels = r3gan.build_stage_channels
build_models = getattr(experiment_module, "_build_models")


class RecordingBranch(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        batch, _, height, width = x.shape
        return torch.zeros(batch, self.out_channels, height // 2, width // 2, dtype=x.dtype, device=x.device)


def test_wavelet_hf_branch_matches_first_stage_shape_and_starts_at_zero() -> None:
    torch.manual_seed(0)
    branch = WaveletHFBranch(in_channels=3, out_channels=64, mid_channels=32, init_gate=0.0)
    x = torch.randn(2, 3, 64, 64)

    y = branch(x)

    assert y.shape == (2, 64, 32, 32)
    assert torch.allclose(y, torch.zeros_like(y), atol=0.0, rtol=0.0)
    assert branch.conv2.conv.weight.abs().sum().item() == 0.0
    assert branch.gate.item() == 0.0


def test_wavelet_hf_branch_produces_signal_once_gate_is_opened() -> None:
    torch.manual_seed(1)
    branch = WaveletHFBranch(in_channels=3, out_channels=16, mid_channels=8, init_gate=1.0)
    with torch.no_grad():
        branch.conv2.conv.weight.fill_(0.05)
    x = torch.randn(2, 3, 32, 32)

    y = branch(x)

    assert y.shape == (2, 16, 16, 16)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_matched_capacity_branch_matches_first_stage_shape_and_starts_at_zero() -> None:
    torch.manual_seed(0)
    branch = MatchedCapacityBranch(in_channels=3, out_channels=64, mid_channels=32, init_gate=0.0)
    x = torch.randn(2, 3, 64, 64)

    y = branch(x)

    assert y.shape == (2, 64, 32, 32)
    assert torch.allclose(y, torch.zeros_like(y), atol=0.0, rtol=0.0)
    assert branch.conv2.conv.weight.abs().sum().item() == 0.0
    assert branch.gate.item() == 0.0


def test_matched_capacity_branch_produces_signal_once_gate_is_opened() -> None:
    torch.manual_seed(2)
    branch = MatchedCapacityBranch(in_channels=3, out_channels=16, mid_channels=8, init_gate=1.0)
    with torch.no_grad():
        branch.conv2.conv.weight.fill_(0.05)
    x = torch.randn(2, 3, 32, 32)

    y = branch(x)

    assert y.shape == (2, 16, 16, 16)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_wavelet_discriminator_preserves_baseline_at_init() -> None:
    torch.manual_seed(123)
    x = torch.randn(4, 3, 64, 64)
    stage_channels = list(reversed(build_stage_channels(img_resolution=64, base_channels=32, channel_max=128)))

    torch.manual_seed(321)
    baseline = R3GANDiscriminator(
        img_resolution=64,
        stage_channels=stage_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=16,
        in_channels=3,
    )
    torch.manual_seed(321)
    with_wavelet = WaveletR3GANDiscriminator(
        img_resolution=64,
        stage_channels=stage_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=16,
        in_channels=3,
        wavelet_branch_mid_scale=0.5,
        wavelet_init_gate=0.0,
    )

    y_baseline = baseline(x)
    y_wavelet = with_wavelet(x)

    torch.testing.assert_close(y_wavelet, y_baseline, atol=1e-6, rtol=1e-6)


def test_matched_capacity_discriminator_preserves_baseline_at_init() -> None:
    torch.manual_seed(123)
    x = torch.randn(4, 3, 64, 64)
    stage_channels = list(reversed(build_stage_channels(img_resolution=64, base_channels=32, channel_max=128)))

    torch.manual_seed(321)
    baseline = R3GANDiscriminator(
        img_resolution=64,
        stage_channels=stage_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=16,
        in_channels=3,
    )
    torch.manual_seed(321)
    control = MatchedCapacityR3GANDiscriminator(
        img_resolution=64,
        stage_channels=stage_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=16,
        in_channels=3,
        wavelet_branch_mid_scale=0.5,
        wavelet_init_gate=0.0,
    )

    y_baseline = baseline(x)
    y_control = control(x)

    torch.testing.assert_close(y_control, y_baseline, atol=1e-6, rtol=1e-6)


def test_wavelet_discriminator_branch_sees_exact_input_and_supports_conditional_projection() -> None:
    torch.manual_seed(7)
    x = torch.randn(3, 3, 64, 64)
    cond = F.one_hot(torch.tensor([0, 1, 2]), num_classes=5).to(dtype=torch.float32)
    stage_channels = list(reversed(build_stage_channels(img_resolution=64, base_channels=32, channel_max=128)))

    model = WaveletR3GANDiscriminator(
        img_resolution=64,
        stage_channels=stage_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=16,
        in_channels=3,
        cond_dim=5,
        cond_embed_dim=16,
        wavelet_branch_mid_scale=0.5,
        wavelet_init_gate=0.0,
    )
    recorder = RecordingBranch(out_channels=stage_channels[1])
    model.wavelet_branch = recorder

    y = model(x, cond)

    assert y.shape == (3,)
    assert recorder.last_input is not None
    torch.testing.assert_close(recorder.last_input, x)


def test_matched_capacity_discriminator_branch_sees_exact_input_and_supports_conditional_projection() -> None:
    torch.manual_seed(9)
    x = torch.randn(3, 3, 64, 64)
    cond = F.one_hot(torch.tensor([0, 1, 2]), num_classes=5).to(dtype=torch.float32)
    stage_channels = list(reversed(build_stage_channels(img_resolution=64, base_channels=32, channel_max=128)))

    model = MatchedCapacityR3GANDiscriminator(
        img_resolution=64,
        stage_channels=stage_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=16,
        in_channels=3,
        cond_dim=5,
        cond_embed_dim=16,
        wavelet_branch_mid_scale=0.5,
        wavelet_init_gate=0.0,
    )
    recorder = RecordingBranch(out_channels=stage_channels[1])
    model.matched_capacity_branch = recorder

    y = model(x, cond)

    assert y.shape == (3,)
    assert recorder.last_input is not None
    torch.testing.assert_close(recorder.last_input, x)


def test_wavelet_discriminator_rejects_unsupported_wavelet_settings() -> None:
    stage_channels = list(reversed(build_stage_channels(img_resolution=64, base_channels=32, channel_max=128)))

    try:
        WaveletR3GANDiscriminator(
            img_resolution=64,
            stage_channels=stage_channels,
            wavelet_type="db2",
        )
    except ValueError as exc:
        assert "Only haar wavelets are supported" in str(exc)
    else:
        raise AssertionError("Expected unsupported wavelet_type to raise ValueError")


def test_build_models_selects_matched_capacity_variant() -> None:
    cfg = RunConfig()
    cfg.img_resolution = 64
    cfg.base_channels = 32
    cfg.channel_max = 128
    cfg.blocks_per_stage = 1
    cfg.matched_capacity_enabled = True
    cfg.metrics_every = 0

    _, d_model = build_models(cfg)

    assert type(d_model).__name__ == "MatchedCapacityR3GANDiscriminator"


def test_build_models_rejects_enabling_both_aux_variants() -> None:
    cfg = RunConfig()
    cfg.img_resolution = 64
    cfg.base_channels = 32
    cfg.channel_max = 128
    cfg.blocks_per_stage = 1
    cfg.wavelet_enabled = True
    cfg.matched_capacity_enabled = True
    cfg.metrics_every = 0

    try:
        build_models(cfg)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected mutually exclusive aux variants to raise ValueError")
