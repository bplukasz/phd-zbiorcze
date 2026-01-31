"""
Model architectures: Generator, Discriminator, EMA
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Generator
# ============================================================================

class ResBlockG(nn.Module):
    """Residual block dla Generatora z upsamplingiem."""

    def __init__(self, in_ch: int, out_ch: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn1(x)
        h = F.relu(h, inplace=True)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        return h + self.skip(x)


class Generator(nn.Module):
    """ResNet Generator - dynamiczny rozmiar obrazu."""

    def __init__(self, z_dim: int = 128, ch: int = 64, img_channels: int = 3, img_size: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.ch = ch
        self.img_size = img_size

        # Oblicz liczbę bloków upsampling na podstawie rozmiaru obrazu
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 (5 bloków dla 128x128)
        self.n_blocks = int(math.log2(img_size)) - 2

        # z -> 4x4 feature map
        self.fc = nn.Linear(z_dim, ch * 16 * 4 * 4)

        # Dynamiczna liczba bloków z channel multipliers
        blocks = []
        in_ch = ch * 16
        ch_mults = [16, 16, 8, 4, 2, 1]

        for i in range(self.n_blocks):
            out_mult = ch_mults[min(i + 1, len(ch_mults) - 1)]
            out_ch = max(ch * out_mult, ch)
            blocks.append(ResBlockG(in_ch, out_ch, upsample=True))
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.final_ch = in_ch

        self.bn_out = nn.BatchNorm2d(self.final_ch)
        self.conv_out = nn.Conv2d(self.final_ch, img_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), self.ch * 16, 4, 4)

        for block in self.blocks:
            h = block(h)

        h = self.bn_out(h)
        h = F.relu(h, inplace=True)
        h = self.conv_out(h)
        return torch.tanh(h)


# ============================================================================
# Discriminator
# ============================================================================

def spectral_norm(module: nn.Module) -> nn.Module:
    """Wrapper dla SpectralNorm."""
    return nn.utils.spectral_norm(module)


class ResBlockD(nn.Module):
    """Residual block dla Discriminatora z SpectralNorm i downsamplingiem."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True, first: bool = False):
        super().__init__()
        self.downsample = downsample
        self.first = first

        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))

        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0)) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if not self.first:
            h = F.relu(h, inplace=True)
        h = self.conv1(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)

        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)

        return h + self.skip(x)


class WaveletBranch(nn.Module):
    """Osobny model/gałąź przetwarzająca cechy falkowe (DWT + conv+SN+LeakyReLU)."""

    def __init__(
        self,
        *,
        img_channels: int,
        out_ch: int,
        hf_only: bool = False,
        wavelet_type: str = "haar",
        wavelet_level: int = 1,
    ):
        super().__init__()

        from .wavelets import DWT2D, split_subbands

        if wavelet_level < 1:
            raise ValueError("wavelet_level musi być >= 1")
        if wavelet_type not in ("haar", "db2"):
            raise ValueError(f"Nieznany wavelet_type={wavelet_type}. Dozwolone: haar, db2")

        self.img_channels = img_channels
        self.hf_only = hf_only
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level

        self._split_subbands = split_subbands
        self.dwt = DWT2D(wavelet=wavelet_type)

        in_ch = img_channels * (3 if hf_only else 4)

        # 3 bloki conv+SN+LeakyReLU (bez dodatkowego downsamplingu)
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _dwt_multilevel(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = x
        for i in range(self.wavelet_level):
            coeffs = self.dwt(coeffs)
            if i < self.wavelet_level - 1:
                # kolejne poziomy liczone na LL
                C_prev = coeffs.shape[1] // 4
                LL, _, _, _ = self._split_subbands(coeffs, C=C_prev)
                coeffs = LL
        return coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self._dwt_multilevel(x)
        C = self.img_channels
        LL, LH, HL, HH = self._split_subbands(coeffs, C=C)
        if self.hf_only:
            w_in = torch.cat([LH, HL, HH], dim=1)
        else:
            w_in = torch.cat([LL, LH, HL, HH], dim=1)
        return self.net(w_in)


class WaveletFusion(nn.Module):
    """Concat(main, wavelet) + 1x1 conv (SN) do fuzji cech."""

    def __init__(self, *, main_ch: int, wavelet_ch: int, out_ch: int):
        super().__init__()
        self.fuse = spectral_norm(nn.Conv2d(main_ch + wavelet_ch, out_ch, 1, 1, 0))

    def forward(self, main: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
        H = min(main.shape[2], wavelet.shape[2])
        W = min(main.shape[3], wavelet.shape[3])
        main = main[:, :, :H, :W]
        wavelet = wavelet[:, :, :H, :W]
        return self.fuse(torch.cat([main, wavelet], dim=1))


class Discriminator(nn.Module):
    """ResNet Discriminator z SpectralNorm - dynamiczny rozmiar obrazu."""

    def __init__(
        self,
        ch: int = 64,
        img_channels: int = 3,
        img_size: int = 128,
        *,
        use_wavelet_branch: bool = False,
        wavelet_hf_only: bool = False,
        wavelet_type: str = "haar",
        wavelet_level: int = 1,
    ):
        super().__init__()
        self.ch = ch
        self.img_size = img_size
        self.img_channels = img_channels

        self.use_wavelet_branch = use_wavelet_branch

        if self.use_wavelet_branch:
            wb_ch = ch
            self.wavelet_branch = WaveletBranch(
                img_channels=img_channels,
                out_ch=wb_ch,
                hf_only=wavelet_hf_only,
                wavelet_type=wavelet_type,
                wavelet_level=wavelet_level,
            )
            # main: BxC x (H/2) x (W/2), wavelet: Bxwb_ch x H' x W'
            self.wavelet_fusion = WaveletFusion(main_ch=img_channels, wavelet_ch=wb_ch, out_ch=img_channels)

        # Oblicz liczbę bloków downsampling
        self.n_blocks = int(math.log2(img_size)) - 2  # -> do 4x4

        # Channel multipliers
        ch_mults = [1, 2, 4, 8, 16, 16]

        blocks = []
        in_ch = img_channels

        for i in range(self.n_blocks):
            out_mult = ch_mults[min(i, len(ch_mults) - 1)]
            out_ch = ch * out_mult
            blocks.append(ResBlockD(in_ch, out_ch, downsample=True, first=(i == 0)))
            in_ch = out_ch

        # Ostatni blok bez downsamplingu
        blocks.append(ResBlockD(in_ch, ch * 16, downsample=False))

        self.blocks = nn.ModuleList(blocks)
        self.fc = spectral_norm(nn.Linear(ch * 16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        if self.use_wavelet_branch:
            h_main = F.avg_pool2d(h, 2)
            h_w = self.wavelet_branch(h)
            h = self.wavelet_fusion(h_main, h_w)

        for block in self.blocks:
            h = block(h)

        h = F.relu(h, inplace=True)
        h = h.sum(dim=[2, 3])  # Global sum pooling
        return self.fc(h)


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Exponential Moving Average dla modelu.

    WAŻNE: Ta wersja prawidłowo kopiuje bufory BatchNorm (running_mean, running_var),
    nie tylko parametry. Bez tego EMA-generator może generować szum nawet jeśli
    oryginalny G się uczy.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

        # Cache maps for fast updates
        self._shadow_params = dict(self.shadow.named_parameters())
        self._shadow_bufs = dict(self.shadow.named_buffers())

    @torch.no_grad()
    def update(self, model: nn.Module):
        # EMA for parameters
        for name, p in model.named_parameters():
            self._shadow_params[name].data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

        # COPY buffers (BatchNorm running stats etc.) - bez EMA, bezpośrednia kopia!
        for name, b in model.named_buffers():
            if name in self._shadow_bufs:
                self._shadow_bufs[name].copy_(b)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
