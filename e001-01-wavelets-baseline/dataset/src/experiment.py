"""
Eksperyment: e001-01-wavelets-baseline
Opis: ResNet GAN baseline z hinge loss, SpectralNorm, EMA, DiffAugment.
      Training pipeline dla CelebA 128x128.
"""

print("WERSJA 31.01.2026-02")

import os
import time
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
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

# Re-export wszystkich potrzebnych komponentów dla zachowania kompatybilności wstecznej
from .augmentations import DiffAugment, AUGMENT_FNS
from .models import Generator, Discriminator, EMA, ResBlockG, ResBlockD, spectral_norm
from .losses import (
    hinge_loss_d,
    hinge_loss_g,
    r1_penalty,
    compute_grad_norm,
    wavelet_energy_matching_loss,
    fft_energy_matching_loss,
)
from .data import get_dataloader
from .metrics import (
    export_real_images,
    generate_samples,
    compute_fid_kid,
    compute_radial_power_spectrum,
    compute_rpse,
    load_images_from_folder,
    compute_rpse_from_folders,
    compute_wavelet_band_energies,
    compute_wbed,
    compute_wbed_from_folders,
    compute_all_spectral_metrics
)
from .utils import CSVLogger
from .config_loader import get_config, ConfigLoader


# Dla kompatybilności - exportuj wszystko co było w oryginalnym pliku
__all__ = [
    # Augmentations
    'DiffAugment',
    'AUGMENT_FNS',

    # Models
    'Generator',
    'Discriminator',
    'EMA',
    'ResBlockG',
    'ResBlockD',
    'spectral_norm',

    # Losses
    'hinge_loss_d',
    'hinge_loss_g',
    'r1_penalty',
    'compute_grad_norm',

    # Data
    'get_dataloader',

    # Metrics
    'export_real_images',
    'generate_samples',
    'compute_fid_kid',
    'compute_radial_power_spectrum',
    'compute_rpse',
    'load_images_from_folder',
    'compute_rpse_from_folders',
    'compute_wavelet_band_energies',
    'compute_wbed',
    'compute_wbed_from_folders',
    'compute_all_spectral_metrics',

    # Utils
    'CSVLogger',

    # Training
    'train',

    # Config
    'get_config',
    'ConfigLoader',
]


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
    dataset_name = getattr(cfg, 'dataset_name', 'celeba')
    print(f"Dataset: {dataset_name} ({cfg.img_size}x{cfg.img_size})")
    print(f"=" * 60)

    # Directories
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save actual config for reproducibility
    ConfigLoader().save_config(cfg, os.path.join(cfg.out_dir, "config_used.yaml"))

    grid_dir = os.path.join(cfg.out_dir, "grids")
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    samples_dir = os.path.join(cfg.out_dir, "samples")
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Models
    G = Generator(z_dim=cfg.z_dim, ch=cfg.g_ch, img_channels=cfg.img_channels, img_size=cfg.img_size).to(device)
    D = Discriminator(
        ch=cfg.d_ch,
        img_channels=cfg.img_channels,
        img_size=cfg.img_size,
        use_wavelet_branch=getattr(cfg, 'use_wavelet_branch', False),
        wavelet_hf_only=getattr(cfg, 'wavelet_hf_only', False),
        wavelet_type=getattr(cfg, 'wavelet_type', 'haar'),
        wavelet_level=getattr(cfg, 'wavelet_level', 1),
    ).to(device)
    G_ema = EMA(G, decay=cfg.ema_decay)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_G, betas=cfg.betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_D, betas=cfg.betas)

    # DataLoader
    try:
        dataloader = get_dataloader(
            cfg.data_dir, cfg.img_size, cfg.batch_size,
            dataset_name=dataset_name, img_channels=cfg.img_channels
        )
    except Exception as e:
        raise RuntimeError(
            f"Data loading failed, stop training (otherwise you'll train on dummy noise). "
            f"Error: {e}"
        )

    # Sanity check: zapisz real_grid.png na starcie żeby sprawdzić czy dane są OK
    data_iter = iter(dataloader)
    real_imgs_check, _ = next(data_iter)
    print(f"REAL data stats: min={real_imgs_check.min().item():.3f}, "
          f"max={real_imgs_check.max().item():.3f}, "
          f"mean={real_imgs_check.mean().item():.3f}")
    save_image(
        (real_imgs_check[:64] + 1) / 2,  # [-1,1] -> [0,1]
        os.path.join(cfg.out_dir, "real_grid.png"),
        nrow=8
    )
    print(f"Zapisano real_grid.png - sprawdź czy dane wyglądają sensownie!")

    # Przygotuj real_samples_dir dla FID/KID
    if dataset_name.lower() == "celeba":
        real_samples_dir = cfg.data_dir
    else:
        # Export real images dla datasetu bez ImageFolder (CIFAR, MNIST, etc.)
        real_samples_dir = os.path.join(cfg.out_dir, "real_samples")
        if cfg.eval_every > 0:
            print(f"\nPrzygotowuję real samples dla FID/KID...")
            export_real_images(dataloader, cfg.fid_samples, real_samples_dir)

    # Logging
    csv_logger = CSVLogger(
        os.path.join(cfg.out_dir, "logs.csv"),
        fieldnames=[
            'step', 'loss_D', 'loss_G', 'grad_norm_D', 'grad_norm_G',
            'sec_per_iter', 'vram_peak_mb',
            'fid', 'kid',
            'rpse', 'wbed_total',
            # Wavelet-energy matching regularization
            'wavereg_loss',
            'wavereg_mu_real_LL', 'wavereg_std_real_LL', 'wavereg_mu_fake_LL', 'wavereg_std_fake_LL',
            'wavereg_mu_real_LH', 'wavereg_std_real_LH', 'wavereg_mu_fake_LH', 'wavereg_std_fake_LH',
            'wavereg_mu_real_HL', 'wavereg_std_real_HL', 'wavereg_mu_fake_HL', 'wavereg_std_fake_HL',
            'wavereg_mu_real_HH', 'wavereg_std_real_HH', 'wavereg_mu_fake_HH', 'wavereg_std_fake_HH',
            # Fourier (FFT) energy matching regularization
            'fftreg_loss',
            'fftreg_time_ms',
        ]
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

    print(f"\nRozpoczynam trening: {cfg.steps} iteracji")
    print(f"Batch size: {cfg.batch_size}, LR: {cfg.lr_G}")
    print(f"DiffAugment: {cfg.diffaug_policy}")
    print("-" * 60)

    for step in range(1, cfg.steps + 1):
        iter_start = time.time()

        # Get real data
        try:
            real_imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_imgs, _ = next(data_iter)
        real_imgs = real_imgs.to(device)

        wavereg_stats: Dict[str, float] = {}
        wavereg_loss_val: Optional[torch.Tensor] = None
        fftreg_stats: Dict[str, float] = {}
        fftreg_loss_val: Optional[torch.Tensor] = None
        fftreg_time_ms: Optional[float] = None

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

        # Wavelet-energy matching regularization (optional)
        if getattr(cfg, 'use_wavereg', False) and getattr(cfg, 'lambda_wavereg', 0.0) > 0:
            apply_to = str(getattr(cfg, 'wavereg_apply_to', 'g')).lower()
            if apply_to in ('d', 'both'):
                reg, reg_logs = wavelet_energy_matching_loss(
                    real_imgs=real_imgs,
                    fake_imgs=fake_imgs,
                    wavelet=str(getattr(cfg, 'wavereg_wavelet', 'haar')),
                    eps=float(getattr(cfg, 'wavereg_eps', 1e-8)),
                )
                wavereg_loss_val = reg
                wavereg_stats.update(reg_logs)
                loss_D = loss_D + float(cfg.lambda_wavereg) * reg

        # Fourier (FFT) energy matching regularization (optional baseline)
        if (
            getattr(cfg, 'use_fftreg', False)
            and getattr(cfg, 'lambda_fftreg', 0.0) > 0
            and (step % int(getattr(cfg, 'fftreg_every', 1)) == 0)
        ):
            apply_to = str(getattr(cfg, 'fftreg_apply_to', 'g')).lower()
            if apply_to in ('d', 'both'):
                t_reg0 = time.time()
                reg, reg_logs = fft_energy_matching_loss(
                    real_imgs=real_imgs,
                    fake_imgs=fake_imgs,
                    num_bins=int(getattr(cfg, 'fftreg_num_bins', 16)),
                    downsample_to=int(getattr(cfg, 'fftreg_downsample_to', 64)),
                    eps=float(getattr(cfg, 'fftreg_eps', 1e-8)),
                )
                fftreg_time_ms = (time.time() - t_reg0) * 1000.0
                fftreg_loss_val = reg
                fftreg_stats.update(reg_logs)
                loss_D = loss_D + float(cfg.lambda_fftreg) * reg

        # R1 gradient penalty (optional, costly but stabilizing)
        if cfg.use_r1_penalty and step % cfg.r1_every == 0:
            gp = r1_penalty(D, real_imgs)
            loss_D = loss_D + cfg.r1_lambda * gp

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

        # Wavelet-energy matching regularization (optional)
        if getattr(cfg, 'use_wavereg', False) and getattr(cfg, 'lambda_wavereg', 0.0) > 0:
            apply_to = str(getattr(cfg, 'wavereg_apply_to', 'g')).lower()
            if apply_to in ('g', 'both'):
                reg, reg_logs = wavelet_energy_matching_loss(
                    real_imgs=real_imgs,
                    fake_imgs=fake_imgs,
                    wavelet=str(getattr(cfg, 'wavereg_wavelet', 'haar')),
                    eps=float(getattr(cfg, 'wavereg_eps', 1e-8)),
                )
                wavereg_loss_val = reg
                wavereg_stats.update(reg_logs)
                loss_G = loss_G + float(cfg.lambda_wavereg) * reg

        # Fourier (FFT) energy matching regularization (optional baseline)
        if (
            getattr(cfg, 'use_fftreg', False)
            and getattr(cfg, 'lambda_fftreg', 0.0) > 0
            and (step % int(getattr(cfg, 'fftreg_every', 1)) == 0)
        ):
            apply_to = str(getattr(cfg, 'fftreg_apply_to', 'g')).lower()
            if apply_to in ('g', 'both'):
                t_reg0 = time.time()
                reg, reg_logs = fft_energy_matching_loss(
                    real_imgs=real_imgs,
                    fake_imgs=fake_imgs,
                    num_bins=int(getattr(cfg, 'fftreg_num_bins', 16)),
                    downsample_to=int(getattr(cfg, 'fftreg_downsample_to', 64)),
                    eps=float(getattr(cfg, 'fftreg_eps', 1e-8)),
                )
                fftreg_time_ms = (time.time() - t_reg0) * 1000.0
                fftreg_loss_val = reg
                fftreg_stats.update(reg_logs)
                loss_G = loss_G + float(cfg.lambda_fftreg) * reg

        loss_G.backward()
        grad_norm_G = compute_grad_norm(G)
        opt_G.step()

        # Update EMA
        G_ema.update(G)

        # Metrics
        iter_time = time.time() - iter_start
        vram_peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0
        losses_G.append(loss_G.item())

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
                'rpse': None,
                'wbed_total': None,
                'wavereg_loss': float(wavereg_stats.get('wavereg_loss', 0.0)) if wavereg_stats else None,
                'fftreg_loss': float(fftreg_stats.get('fftreg_loss', 0.0)) if fftreg_stats else None,
                'fftreg_time_ms': fftreg_time_ms,
            }
            log_data.update(wavereg_stats)
            log_data.update(fftreg_stats)
            csv_logger.log(log_data)

            if cfg.use_wandb and _HAS_WANDB:
                wb = {
                    'loss_D': loss_D.item(),
                    'loss_G': loss_G.item(),
                    'grad_norm_D': grad_norm_D,
                    'grad_norm_G': grad_norm_G,
                    'sec_per_iter': iter_time,
                    'vram_peak_mb': vram_peak,
                }
                if wavereg_stats:
                    wb.update(wavereg_stats)
                if fftreg_stats:
                    wb.update(fftreg_stats)
                if fftreg_time_ms is not None:
                    wb['fftreg_time_ms'] = fftreg_time_ms
                wandb.log(wb, step=step)

            if getattr(cfg, 'use_fftreg', False) and getattr(cfg, 'lambda_fftreg', 0.0) > 0 and fftreg_stats:
                print(f"    fftreg: loss={fftreg_stats.get('fftreg_loss', 0.0):.6f} "
                      f"time={fftreg_time_ms if fftreg_time_ms is not None else float('nan'):.2f}ms "
                      f"(bins={getattr(cfg, 'fftreg_num_bins', 16)}, downsample={getattr(cfg, 'fftreg_downsample_to', 64)}, "
                      f"apply_to={getattr(cfg, 'fftreg_apply_to', 'g')}, lambda={getattr(cfg, 'lambda_fftreg', 0.0)})")

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
            print(f"     (2/2) Obliczanie metryk FID/KID...")
            metrics = compute_fid_kid(real_samples_dir, eval_samples_dir, cfg.fid_samples)
            current_fid = metrics['fid']
            current_kid = metrics['kid']
            print(f"  -> FID: {current_fid:.2f}, KID: {current_kid:.4f}")

            # Spectral metrics (RPSE + WBED)
            rpse_val = None
            wbed_total = None
            if getattr(cfg, 'spectral_metrics', True):
                try:
                    max_imgs = int(getattr(cfg, 'spectral_max_images', 1000))
                    img_size = getattr(cfg, 'spectral_img_size', None)
                    wavelet = getattr(cfg, 'wbed_wavelet', 'haar')

                    print(f"     (extra) Obliczanie metryk spektralnych (RPSE/WBED) na max_images={max_imgs}...")
                    spectral = compute_all_spectral_metrics(
                        real_folder=real_samples_dir,
                        fake_folder=eval_samples_dir,
                        max_images=max_imgs,
                        num_bins=None,
                        wavelet=wavelet,
                        img_size=img_size,
                        device=str(device),
                    )
                    rpse_val = spectral.get('rpse')
                    wbed_total = spectral.get('wbed_total')
                except Exception as e:
                    print(f"     ✗ Nie udało się policzyć RPSE/WBED: {e}")

            # Log evaluation row to CSV
            csv_logger.log({
                'step': step,
                'loss_D': loss_D.item(),
                'loss_G': loss_G.item(),
                'grad_norm_D': grad_norm_D,
                'grad_norm_G': grad_norm_G,
                'sec_per_iter': iter_time,
                'vram_peak_mb': vram_peak,
                'fid': current_fid,
                'kid': current_kid,
                'rpse': rpse_val,
                'wbed_total': wbed_total,
            })

            # Track best FID
            if current_fid < best_fid:
                improvement = best_fid - current_fid
                best_fid = current_fid
                print(f"  ✓ Nowy najlepszy FID! (poprawa: {improvement:.2f})")

            if cfg.use_wandb and _HAS_WANDB:
                wandb.log({'fid': current_fid, 'kid': current_kid, 'rpse': rpse_val, 'wbed_total': wbed_total}, step=step)

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
