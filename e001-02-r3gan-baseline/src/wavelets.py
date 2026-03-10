from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FixedHaarDWT2d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")

        self.in_channels = int(in_channels)
        base_filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
                [[0.5, -0.5], [0.5, -0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("weight", base_filters.repeat(self.in_channels, 1, 1, 1))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, C, H, W), got {tuple(x.shape)}")

        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Expected even H and W, got H={h}, W={w}")

        coeffs = F.conv2d(
            x,
            self.weight.to(device=x.device, dtype=x.dtype),
            stride=2,
            padding=0,
            groups=self.in_channels,
        )
        coeffs = coeffs.view(b, self.in_channels, 4, h // 2, w // 2)
        return {
            "LL": coeffs[:, :, 0],
            "LH": coeffs[:, :, 1],
            "HL": coeffs[:, :, 2],
            "HH": coeffs[:, :, 3],
        }


class FixedHaarIDWT2d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")

        self.in_channels = int(in_channels)
        base_filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
                [[0.5, -0.5], [0.5, -0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("weight", base_filters.repeat(self.in_channels, 1, 1, 1))

    def forward(self, bands: dict[str, Tensor]) -> Tensor:
        expected_keys = {"LL", "LH", "HL", "HH"}
        if set(bands) != expected_keys:
            raise ValueError(f"Expected band keys {sorted(expected_keys)}, got {sorted(bands)}")

        ll = bands["LL"]
        if ll.ndim != 4:
            raise ValueError(f"Expected bands with shape (B, C, H, W), got {tuple(ll.shape)}")

        b, c, h, w = ll.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")

        for name in ("LH", "HL", "HH"):
            if bands[name].shape != ll.shape:
                raise ValueError(f"Band {name} shape {tuple(bands[name].shape)} does not match LL shape {tuple(ll.shape)}")

        coeffs = torch.stack([bands["LL"], bands["LH"], bands["HL"], bands["HH"]], dim=2)
        coeffs = coeffs.view(b, 4 * self.in_channels, h, w)
        return F.conv_transpose2d(
            coeffs,
            self.weight.to(device=coeffs.device, dtype=coeffs.dtype),
            stride=2,
            padding=0,
            groups=self.in_channels,
        )

