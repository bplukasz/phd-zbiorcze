"""
Eksperyment: e001-01-wavelets-baseline
Opis: ResNet GAN baseline z hinge loss, SpectralNorm, EMA, DiffAugment.
      Training pipeline dla CelebA 128x128.
"""

import os
import csv
import time
import copy
import math
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# Optional: notebook live display
try:
    from IPython.display import clear_output, display
    _HAS_IPYTHON = True
except Exception:
    _HAS_IPYTHON = False

import matplotlib.pyplot as plt

# Optional: Weights & Biases
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# Import configuration system
from .config_loader import RunConfig, get_config


# ============================================================================
# DiffAugment
# ============================================================================

def DiffAugment(x: torch.Tensor, policy: str = '') -> torch.Tensor:
    """Differentiable Augmentation for Data-Efficient GAN Training."""
    if policy:
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
    return x


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


# ============================================================================
# Model: ResNet Generator
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
    """ResNet Generator dla 128x128."""

    def __init__(self, z_dim: int = 128, ch: int = 64, img_channels: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.ch = ch

        # z -> 4x4 feature map
        self.fc = nn.Linear(z_dim, ch * 16 * 4 * 4)

        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.blocks = nn.ModuleList([
            ResBlockG(ch * 16, ch * 16, upsample=True),  # 4 -> 8
            ResBlockG(ch * 16, ch * 8, upsample=True),   # 8 -> 16
            ResBlockG(ch * 8, ch * 4, upsample=True),    # 16 -> 32
            ResBlockG(ch * 4, ch * 2, upsample=True),    # 32 -> 64
            ResBlockG(ch * 2, ch, upsample=True),        # 64 -> 128
        ])

        self.bn_out = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, img_channels, 3, 1, 1)

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
# Model: ResNet Discriminator with SpectralNorm
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


class Discriminator(nn.Module):
    """ResNet Discriminator z SpectralNorm dla 128x128."""

    def __init__(self, ch: int = 64, img_channels: int = 3):
        super().__init__()
        self.ch = ch

        # 128x128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.blocks = nn.ModuleList([
            ResBlockD(img_channels, ch, downsample=True, first=True),  # 128 -> 64
            ResBlockD(ch, ch * 2, downsample=True),                     # 64 -> 32
            ResBlockD(ch * 2, ch * 4, downsample=True),                 # 32 -> 16
            ResBlockD(ch * 4, ch * 8, downsample=True),                 # 16 -> 8
            ResBlockD(ch * 8, ch * 16, downsample=True),                # 8 -> 4
            ResBlockD(ch * 16, ch * 16, downsample=False),              # 4 -> 4
        ])

        self.fc = spectral_norm(nn.Linear(ch * 16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)

        h = F.relu(h, inplace=True)
        h = h.sum(dim=[2, 3])  # Global sum pooling
        return self.fc(h)


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Exponential Moving Average dla modelu."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ============================================================================
# Hinge Loss
# ============================================================================

def hinge_loss_d(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss dla Discriminatora."""
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def hinge_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss dla Generatora."""
    return -fake_logits.mean()


def r1_penalty(D: nn.Module, real_imgs: torch.Tensor) -> torch.Tensor:
    """
    R1 gradient penalty (Mescheder et al., 2018).
    Karze Discriminator za zbyt duże gradienty względem danych wejściowych.
    Stabilizuje trening i zapobiega mode collapse.

    Args:
        D: Discriminator model
        real_imgs: Prawdziwe obrazy

    Returns:
        Gradient penalty scalar
    """
    real_imgs = real_imgs.detach().requires_grad_(True)
    real_logits = D(real_imgs)

    # Backward przez sumę logits (nie mean, bo chcemy gradienty dla każdej próbki)
    grad_outputs = torch.ones_like(real_logits)
    gradients = torch.autograd.grad(
        outputs=real_logits,
        inputs=real_imgs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # L2 norm of gradients
    penalty = (gradients ** 2).sum([1, 2, 3]).mean()
    return penalty


def hinge_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss dla Generatora."""
    return -fake_logits.mean()


# ============================================================================
# Gradient Norm
# ============================================================================

def compute_grad_norm(model: nn.Module) -> float:
    """Oblicza normę gradientów modelu."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ============================================================================
# Logging
# ============================================================================

class CSVLogger:
    """Prosty logger do pliku CSV."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ============================================================================
# Evaluation: FID/KID
# ============================================================================

def generate_samples(G: nn.Module, n_samples: int, z_dim: int, batch_size: int,
                     device: torch.device, out_dir: str) -> str:
    """Generuje próbki i zapisuje do katalogu."""
    os.makedirs(out_dir, exist_ok=True)
    G.eval()

    n_batches = math.ceil(n_samples / batch_size)
    idx = 0

    print(f"    Generowanie {n_samples} próbek w {n_batches} partiach...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            # Progress feedback co 20 batchy lub ostatni
            if batch_idx % 20 == 0 or batch_idx == n_batches - 1:
                progress = (idx / n_samples) * 100
                print(f"    -> {idx}/{n_samples} próbek ({progress:.1f}%)")

            z = torch.randn(min(batch_size, n_samples - idx), z_dim, device=device)
            imgs = G(z)
            imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]

            for img in imgs:
                if idx >= n_samples:
                    break
                save_image(img, os.path.join(out_dir, f'{idx:06d}.png'))
                idx += 1

    print(f"    ✓ Wygenerowano wszystkie {n_samples} próbek")
    G.train()
    return out_dir


def compute_fid_kid(real_dir: str, fake_dir: str, fid_samples: int = 10000) -> Dict[str, float]:
    """Oblicza FID i KID używając torch-fidelity."""
    try:
        from torch_fidelity import calculate_metrics

        print(f"    Obliczanie FID/KID dla {fid_samples} próbek...")
        print(f"    (To może potrwać 5-15 minut, obliczenia Inception...)")

        metrics = calculate_metrics(
            input1=fake_dir,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            kid=True,
            kid_subset_size=min(1000, fid_samples),
            verbose=False,  # torch-fidelity ma swój progress, ale jest spammy
        )

        print(f"    ✓ Metryki obliczone!")

        return {
            'fid': metrics.get('frechet_inception_distance', float('nan')),
            'kid': metrics.get('kernel_inception_distance_mean', float('nan')) * 1000,  # x1000 dla czytelności
        }
    except Exception as e:
        print(f"    ✗ Błąd przy obliczaniu FID/KID: {e}")
        return {'fid': float('nan'), 'kid': float('nan')}


# ============================================================================
# Dataset
# ============================================================================

def get_dataloader(data_dir: str, img_size: int, batch_size: int,
                   num_workers: int = 4) -> DataLoader:
    """Tworzy DataLoader dla CelebA."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(
        root=os.path.dirname(data_dir),
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


# ============================================================================
# Training Loop
# ============================================================================

def train(profile: str = "preview", overrides: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, List[float]]:
    """
    Główna funkcja treningowa.

    Args:
        profile: "preview" dla notebooka, "train" dla pełnego treningu, "smoke" dla smoke test
        overrides: Dodatkowe nadpisania konfiguracji z CLI

    Returns:
        Tuple z modelem (EMA Generator) i historią loss_G
    """
    cfg = get_config(profile, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=" * 60)
    print(f"Eksperyment: e001-01-wavelets-baseline")
    print(f"Profile: {cfg.name}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"=" * 60)

    # Directories
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save actual config for reproducibility
    from .config_loader import ConfigLoader
    loader = ConfigLoader()
    loader.save_config(cfg, os.path.join(cfg.out_dir, "config_used.yaml"))
    grid_dir = os.path.join(cfg.out_dir, "grids")
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    samples_dir = os.path.join(cfg.out_dir, "samples")
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Models
    G = Generator(z_dim=cfg.z_dim, ch=cfg.g_ch, img_channels=cfg.img_channels).to(device)
    D = Discriminator(ch=cfg.d_ch, img_channels=cfg.img_channels).to(device)
    G_ema = EMA(G, decay=cfg.ema_decay)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_G, betas=cfg.betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_D, betas=cfg.betas)

    # DataLoader
    try:
        dataloader = get_dataloader(cfg.data_dir, cfg.img_size, cfg.batch_size)
        data_iter = iter(dataloader)
    except Exception as e:
        print(f"Błąd ładowania danych: {e}")
        print("Używam dummy data dla testów...")
        dataloader = None
        data_iter = None

    # Logging
    csv_logger = CSVLogger(
        os.path.join(cfg.out_dir, "logs.csv"),
        fieldnames=['step', 'loss_D', 'loss_G', 'grad_norm_D', 'grad_norm_G',
                    'sec_per_iter', 'vram_peak_mb', 'fid', 'kid']
    )

    # W&B
    if cfg.use_wandb and _HAS_WANDB:
        try:
            wandb.init(
                project="e001-wavelets-baseline",
                name=cfg.name,
                config=cfg.__dict__,
            )
            print("W&B logging enabled")
        except Exception as e:
            print(f"Warning: Could not initialize W&B: {e}")
            print("Continuing without W&B logging...")
            cfg.use_wandb = False

    # Fixed noise for visualization
    fixed_z = torch.randn(64, cfg.z_dim, device=device)

    # Training
    t0 = time.time()
    losses_G: List[float] = []
    best_fid = float('inf')
    fid_history: List[Tuple[int, float]] = []  # (step, fid)

    print(f"\nRozpoczynam trening: {cfg.steps} iteracji")
    print(f"Batch size: {cfg.batch_size}, LR: {cfg.lr_G}")
    print(f"DiffAugment: {cfg.diffaug_policy}")
    print("-" * 60)

    for step in range(1, cfg.steps + 1):
        iter_start = time.time()

        # Get real data
        if data_iter is not None:
            try:
                real_imgs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                real_imgs, _ = next(data_iter)
            real_imgs = real_imgs.to(device)
        else:
            # Dummy data for testing
            real_imgs = torch.randn(cfg.batch_size, cfg.img_channels, cfg.img_size, cfg.img_size, device=device)

        # ==================== Train D ====================
        D.zero_grad()

        # Real
        real_aug = DiffAugment(real_imgs, cfg.diffaug_policy)
        real_logits = D(real_aug)

        # Fake
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake_imgs = G(z).detach()
        fake_aug = DiffAugment(fake_imgs, cfg.diffaug_policy)
        fake_logits = D(fake_aug)

        loss_D = hinge_loss_d(real_logits, fake_logits)

        # R1 gradient penalty (optional, costly but stabilizing)
        if hasattr(cfg, 'use_r1_penalty') and cfg.use_r1_penalty:
            r1_every = getattr(cfg, 'r1_every', 16)
            if step % r1_every == 0:
                r1_lambda = getattr(cfg, 'r1_lambda', 10.0)
                gp = r1_penalty(D, real_imgs)
                loss_D = loss_D + r1_lambda * gp

        loss_D.backward()
        grad_norm_D = compute_grad_norm(D)
        opt_D.step()

        # ==================== Train G ====================
        G.zero_grad()

        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake_imgs = G(z)
        fake_aug = DiffAugment(fake_imgs, cfg.diffaug_policy)
        fake_logits = D(fake_aug)

        loss_G = hinge_loss_g(fake_logits)
        loss_G.backward()
        grad_norm_G = compute_grad_norm(G)
        opt_G.step()

        # Update EMA
        G_ema.update(G)

        # Metrics
        iter_time = time.time() - iter_start
        vram_peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0

        losses_G.append(loss_G.item())

        # Detect potential mode collapse (sudden loss spikes)
        if len(losses_G) > 10:
            recent_avg = sum(losses_G[-10:-1]) / 9
            if abs(loss_G.item() - recent_avg) > 5.0:
                print(f"  ⚠️  [UWAGA krok {step}] Nagły skok w loss_G: {loss_G.item():.2f} "
                      f"(średnia z 10: {recent_avg:.2f}). Możliwy mode collapse!")

        # ==================== Logging ====================
        if cfg.log_every > 0 and step % cfg.log_every == 0:
            log_data = {
                'step': step,
                'loss_D': loss_D.item(),
                'loss_G': loss_G.item(),
                'grad_norm_D': grad_norm_D,
                'grad_norm_G': grad_norm_G,
                'sec_per_iter': iter_time,
                'vram_peak_mb': vram_peak,
                'fid': None,
                'kid': None,
            }
            csv_logger.log(log_data)

            if cfg.use_wandb and _HAS_WANDB:
                wandb.log({k: v for k, v in log_data.items() if v is not None}, step=step)

            print(f"[{step:06d}/{cfg.steps}] D:{loss_D.item():.4f} G:{loss_G.item():.4f} "
                  f"gD:{grad_norm_D:.2f} gG:{grad_norm_G:.2f} "
                  f"t:{iter_time:.3f}s VRAM:{vram_peak:.0f}MB")

        # ==================== Sample Grid ====================
        if cfg.grid_every > 0 and step % cfg.grid_every == 0:
            G_ema.shadow.eval()
            with torch.no_grad():
                fake_grid = G_ema(fixed_z)
                fake_grid = (fake_grid + 1) / 2
            grid = make_grid(fake_grid, nrow=8, padding=2)
            save_image(grid, os.path.join(grid_dir, f"grid_{step:06d}.png"))

            if cfg.use_wandb and _HAS_WANDB:
                wandb.log({"samples": wandb.Image(grid)}, step=step)

            if cfg.live and _HAS_IPYTHON:
                clear_output(wait=True)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.title(f"Step {step}")
                plt.show()

            print(f"  -> Saved grid: grid_{step:06d}.png")

        # ==================== Checkpoint ====================
        if cfg.ckpt_every > 0 and step % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{step:06d}.pt")
            torch.save({
                'step': step,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'G_ema': G_ema.shadow.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
            }, ckpt_path)
            print(f"  -> Saved checkpoint: ckpt_{step:06d}.pt")

        # ==================== Evaluation ====================
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            print(f"  -> Rozpoczynam ewaluację na kroku {step}...")
            print(f"     (1/2) Generowanie {cfg.fid_samples} próbek testowych...")

            # Generate samples
            eval_samples_dir = os.path.join(samples_dir, f"step_{step:06d}")
            generate_samples(
                G_ema.shadow,
                n_samples=cfg.fid_samples,
                z_dim=cfg.z_dim,
                batch_size=cfg.batch_size,
                device=device,
                out_dir=eval_samples_dir
            )

            # Compute FID/KID
            if dataloader is not None:
                print(f"     (2/2) Obliczanie metryk FID/KID...")
                metrics = compute_fid_kid(cfg.data_dir, eval_samples_dir, cfg.fid_samples)
                current_fid = metrics['fid']
                print(f"  -> FID: {current_fid:.2f}, KID: {metrics['kid']:.4f}")

                # Track FID improvements/degradations
                fid_history.append((step, current_fid))
                if current_fid < best_fid:
                    improvement = best_fid - current_fid
                    best_fid = current_fid
                    print(f"  ✓ Nowy najlepszy FID! (poprawa: {improvement:.2f})")
                elif len(fid_history) > 1:
                    prev_fid = fid_history[-2][1]
                    degradation = current_fid - prev_fid
                    print(f"  ⚠️  FID pogorszył się o {degradation:.2f} (poprzedni: {prev_fid:.2f})")
                    if degradation > 20:
                        print(f"  ⚠️  UWAGA: Znaczące pogorszenie! Możliwy mode collapse lub overfitting.")

                if cfg.use_wandb and _HAS_WANDB:
                    wandb.log({'fid': metrics['fid'], 'kid': metrics['kid']}, step=step)

    # ==================== Final ====================
    total_time = time.time() - t0
    print("=" * 60)
    print(f"Trening zakończony w {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Średni czas/iter: {total_time/cfg.steps:.3f}s")

    # Final checkpoint
    final_ckpt_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        'step': cfg.steps,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'G_ema': G_ema.shadow.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
    }, final_ckpt_path)
    print(f"Zapisano finalny checkpoint: {final_ckpt_path}")

    # Final evaluation
    if cfg.eval_every > 0:
        print("\n" + "=" * 60)
        print("FINALNA EWALUACJA")
        print("=" * 60)
        print(f"Generowanie finalnych {cfg.eval_samples} próbek...")
        print(f"(To może potrwać 5-10 minut w zależności od rozmiaru)")

        final_samples_dir = os.path.join(samples_dir, "final_50k")
        generate_samples(
            G_ema.shadow,
            n_samples=cfg.eval_samples,
            z_dim=cfg.z_dim,
            batch_size=cfg.batch_size,
            device=device,
            out_dir=final_samples_dir
        )
        print(f"✓ Zapisano {cfg.eval_samples} próbek do: {final_samples_dir}")

    if cfg.use_wandb and _HAS_WANDB:
        wandb.finish()

    print("\n" + "=" * 60)
    print("🎉 EKSPERYMENT ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 60)

    return G_ema.shadow, losses_G

