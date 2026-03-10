from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WAVELETS_PATH = EXPERIMENT_ROOT / "src" / "wavelets.py"
REAL_SAMPLES_DIR = EXPERIMENT_ROOT / "artifacts" / "real_samples"

spec = importlib.util.spec_from_file_location("e001_02_wavelets", WAVELETS_PATH)
assert spec is not None and spec.loader is not None
wavelets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wavelets)

FixedHaarDWT2d = wavelets.FixedHaarDWT2d
FixedHaarIDWT2d = wavelets.FixedHaarIDWT2d

MAX_ABS_ERR_THRESHOLD = 1e-6
MAE_THRESHOLD = 1e-7
PSNR_THRESHOLD = 120.0


def _reconstruct(x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dwt = FixedHaarDWT2d(in_channels=x.shape[1])
    idwt = FixedHaarIDWT2d(in_channels=x.shape[1])
    bands = dwt(x)
    x_rec = idwt(bands)
    return bands, x_rec


def _compute_metrics(x: torch.Tensor, x_rec: torch.Tensor) -> dict[str, float]:
    diff = (x - x_rec).abs()
    mse = torch.mean((x - x_rec) ** 2).item()
    return {
        "max_abs_err": diff.max().item(),
        "mae": diff.mean().item(),
        "psnr": float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
    }


def _load_real_batch(batch_size: int = 8) -> torch.Tensor:
    image_paths = sorted(REAL_SAMPLES_DIR.glob("*.png"))[:batch_size]
    if len(image_paths) < batch_size:
        raise AssertionError(f"Expected at least {batch_size} PNG files in {REAL_SAMPLES_DIR}")

    batch = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        width -= width % 2
        height -= height % 2
        image = image.crop((0, 0, width, height))
        batch.append(pil_to_tensor(image).float() / 255.0)

    return torch.stack(batch, dim=0)


def test_idwt_of_dwt_reconstructs_input() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 48)

    bands, x_rec = _reconstruct(x)

    assert set(bands) == {"LL", "LH", "HL", "HH"}
    assert bands["LL"].shape == (4, 3, 16, 24)
    assert torch.allclose(x_rec, x, atol=1e-6, rtol=1e-6)


def test_random_tensor_metrics_are_practically_perfect() -> None:
    torch.manual_seed(123)
    x = torch.rand(2, 3, 64, 64)

    _, x_rec = _reconstruct(x)
    metrics = _compute_metrics(x, x_rec)

    assert metrics["max_abs_err"] < MAX_ABS_ERR_THRESHOLD
    assert metrics["mae"] < MAE_THRESHOLD
    assert metrics["psnr"] > PSNR_THRESHOLD


def test_real_image_batch_metrics_are_practically_perfect() -> None:
    x = _load_real_batch(batch_size=8)

    _, x_rec = _reconstruct(x)
    metrics = _compute_metrics(x, x_rec)

    assert metrics["max_abs_err"] < MAX_ABS_ERR_THRESHOLD
    assert metrics["mae"] < MAE_THRESHOLD
    assert metrics["psnr"] > PSNR_THRESHOLD

