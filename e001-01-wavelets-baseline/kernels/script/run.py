#!/usr/bin/env python3
"""
e001-01-wavelets-baseline - Script Runner
ResNet GAN z hinge loss, SpectralNorm, EMA, DiffAugment na CelebA 128x128
"""

import os
import sys
import subprocess

# ============================================================================
# Auto-instalacja zależności
# ============================================================================

def install_dependencies():
    """Instaluje wymagane pakiety."""
    packages = [
        "torch-fidelity",  # FID/KID
        "wandb",           # Weights & Biases logging
    ]

    for pkg in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Uwaga: nie udało się zainstalować {pkg}: {e}")

print("=" * 60)
print("e001-01-wavelets-baseline - Script Runner")
print("=" * 60)

# Instaluj zależności
print("Instalowanie zależności...")
install_dependencies()

# ============================================================================
# Paths setup
# ============================================================================

print("\nInputs:", os.listdir("/kaggle/input"))

CODE_DIR = "/kaggle/input/e001-01-wavelets-baseline-lib"
sys.path.insert(0, CODE_DIR)

SHARED_DIR = "/kaggle/input/shared-lib"
sys.path.insert(0, SHARED_DIR)

# Verify paths
print(f"CODE_DIR exists: {os.path.exists(CODE_DIR)}")
print(f"SHARED_DIR exists: {os.path.exists(SHARED_DIR)}")

# Check CelebA dataset
CELEBA_DIR = "/kaggle/input/celeba-dataset"
print(f"CelebA exists: {os.path.exists(CELEBA_DIR)}")
if os.path.exists(CELEBA_DIR):
    print(f"CelebA contents: {os.listdir(CELEBA_DIR)}")

# ============================================================================
# Configuration
# ============================================================================

# Wybierz profil treningu:
# - "preview": 200 kroków, szybki test
# - "smoke": 500 kroków, weryfikacja pipeline
# - "train": pełny trening (30k kroków)
PROFILE = "fast"

# Opcjonalne nadpisania konfiguracji:
# UWAGA: Na Kaggle W&B wymaga API key. Jeśli go nie masz, ustaw use_wandb: False
OVERRIDES = {
    'use_wandb': False,  # Wyłącz W&B jeśli brak API key
    # 'steps': 10000,
    # 'batch_size': 128,
    # 'eval_every': 5000,
}

# ============================================================================
# Import and run
# ============================================================================

from src import train, get_config
from src.wavelets import run_all_tests
from src.experiment import (
    compute_radial_power_spectrum,
    compute_rpse,
    compute_wavelet_band_energies,
    compute_wbed,
    load_images_from_folder,
    compute_rpse_from_folders,
    compute_wbed_from_folders,
    compute_all_spectral_metrics,
)


def run_training():
    """Uruchamia trening modelu GAN."""
    cfg = get_config(PROFILE, OVERRIDES)

    print(f"\nKonfiguracja:")
    print(f"  Profile: {cfg.name}")
    print(f"  Steps: {cfg.steps}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  LR_G: {cfg.lr_G}")
    print(f"  W&B: {cfg.use_wandb}")
    print(f"  Eval every: {cfg.eval_every}")
    print()

    # Run training
    model, losses = train(PROFILE, OVERRIDES)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final loss_G: {losses[-1]:.4f}")
    print(f"Artifacts saved to: {cfg.out_dir}")
    print("=" * 60)


def run_wavelet_tests():
    """Uruchamia testy transformaty falkowej DWT2D/IDWT2D."""
    print("\n" + "=" * 60)
    print("Uruchamianie testów transformaty falkowej DWT2D/IDWT2D")
    print("=" * 60)

    # Uruchom wszystkie testy
    output_dir = '/kaggle/working/dwt_test_output'

    # Wybierz przykładowy obraz z CelebA (jeśli dostępny)
    celeba_img_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
    real_image = None
    if os.path.exists(celeba_img_dir):
        # Użyj pierwszego dostępnego obrazu
        celeba_images = sorted([f for f in os.listdir(celeba_img_dir) if f.endswith('.jpg')])
        if celeba_images:
            real_image = os.path.join(celeba_img_dir, celeba_images[0])
            print(f"Użyty obraz testowy: {celeba_images[0]}")

    results = run_all_tests(output_dir=output_dir, real_image_path=real_image)

    print("\n" + "=" * 60)
    print("Testy zakończone!")
    print(f"Wyniki zapisane w: {output_dir}")
    print("=" * 60)


def run_spectral_metrics_test():
    """
    Testuje metryki spektralne RPSE i WBED na syntetycznych lub prawdziwych danych.
    """
    import torch
    import numpy as np

    print("\n" + "=" * 60)
    print("TEST METRYK SPEKTRALNYCH: RPSE + WBED")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # =========================================================================
    # Test 1: Syntetyczne dane (szum gaussowski)
    # =========================================================================
    print("\n" + "-" * 60)
    print("[1/3] Test na syntetycznych danych (szum gaussowski)")
    print("-" * 60)

    torch.manual_seed(42)

    # Real = szum gaussowski o std=1.0
    real_imgs = torch.randn(100, 3, 64, 64, device=device)

    # Fake = szum gaussowski o std=0.5 (inny rozkład)
    fake_imgs = torch.randn(100, 3, 64, 64, device=device) * 0.5

    print(f"Real images: {real_imgs.shape}, range=[{real_imgs.min():.2f}, {real_imgs.max():.2f}]")
    print(f"Fake images: {fake_imgs.shape}, range=[{fake_imgs.min():.2f}, {fake_imgs.max():.2f}]")

    # RPSE
    print("\n>>> RPSE (Radial Power Spectrum Error):")
    real_profile = compute_radial_power_spectrum(real_imgs)
    fake_profile = compute_radial_power_spectrum(fake_imgs)
    rpse = compute_rpse(real_profile, fake_profile)
    print(f"    Profil real: {real_profile.shape}, sum={real_profile.sum():.4f}")
    print(f"    Profil fake: {fake_profile.shape}, sum={fake_profile.sum():.4f}")
    print(f"    RPSE: {rpse:.6f}")

    # RPSE dla identycznych obrazów (powinno być ~0)
    rpse_same = compute_rpse(real_profile, real_profile)
    print(f"    RPSE (same vs same): {rpse_same:.6f} (oczekiwane: ~0)")

    # WBED
    print("\n>>> WBED (Wavelet Band Energy Distance):")
    for wavelet in ['haar', 'db2']:
        print(f"\n    Wavelet: {wavelet}")
        real_energies = compute_wavelet_band_energies(real_imgs, wavelet=wavelet)
        fake_energies = compute_wavelet_band_energies(fake_imgs, wavelet=wavelet)

        wbed_result = compute_wbed(real_energies, fake_energies)
        print(f"    WBED total: {wbed_result['wbed_total']:.6f}")
        for band in ['LL', 'LH', 'HL', 'HH']:
            print(f"      {band}: dist={wbed_result[f'{band}_dist']:.6f} "
                  f"(mean_diff={wbed_result[f'{band}_mean_diff']:.6f}, "
                  f"std_diff={wbed_result[f'{band}_std_diff']:.6f})")

    # =========================================================================
    # Test 2: Syntetyczne obrazy (wzory geometryczne)
    # =========================================================================
    print("\n" + "-" * 60)
    print("[2/3] Test na syntetycznych obrazach (wzory geometryczne)")
    print("-" * 60)

    # Real = szachownica
    checkerboard = torch.zeros(1, 3, 64, 64, device=device)
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                checkerboard[0, :, i, j] = 1.0
    real_pattern = checkerboard.repeat(50, 1, 1, 1)

    # Fake = paski poziome
    stripes = torch.zeros(1, 3, 64, 64, device=device)
    for i in range(64):
        if (i // 8) % 2 == 0:
            stripes[0, :, i, :] = 1.0
    fake_pattern = stripes.repeat(50, 1, 1, 1)

    # Dodaj lekki szum
    real_pattern = real_pattern + torch.randn_like(real_pattern) * 0.1
    fake_pattern = fake_pattern + torch.randn_like(fake_pattern) * 0.1

    print(f"Real pattern (szachownica + szum): {real_pattern.shape}")
    print(f"Fake pattern (paski + szum): {fake_pattern.shape}")

    # RPSE
    real_profile_p = compute_radial_power_spectrum(real_pattern)
    fake_profile_p = compute_radial_power_spectrum(fake_pattern)
    rpse_pattern = compute_rpse(real_profile_p, fake_profile_p)
    print(f"\n>>> RPSE: {rpse_pattern:.6f}")

    # WBED
    real_e = compute_wavelet_band_energies(real_pattern, wavelet='haar')
    fake_e = compute_wavelet_band_energies(fake_pattern, wavelet='haar')
    wbed_pattern = compute_wbed(real_e, fake_e)
    print(f">>> WBED total (haar): {wbed_pattern['wbed_total']:.6f}")

    # =========================================================================
    # Test 3: Prawdziwe obrazy z CelebA (jeśli dostępne)
    # =========================================================================
    print("\n" + "-" * 60)
    print("[3/3] Test na prawdziwych obrazach z CelebA")
    print("-" * 60)

    celeba_img_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'

    if os.path.exists(celeba_img_dir):
        print(f"CelebA znalezione: {celeba_img_dir}")

        # Wczytaj obrazy
        real_celeba = load_images_from_folder(
            celeba_img_dir,
            max_images=500,
            img_size=64,
            device=device
        )
        print(f"Real CelebA: {real_celeba.shape}")

        # Fake = zaburzone obrazy (blur + noise)
        # Symulacja złej jakości generatora
        import torch.nn.functional as F

        # Rozmycie (avg pool + upsample)
        fake_celeba = F.avg_pool2d(real_celeba, kernel_size=4, stride=1, padding=2)
        fake_celeba = F.interpolate(fake_celeba, size=(64, 64), mode='bilinear')
        fake_celeba = fake_celeba + torch.randn_like(fake_celeba) * 0.05
        fake_celeba = fake_celeba.clamp(0, 1)

        print(f"Fake (blurred CelebA): {fake_celeba.shape}")

        # RPSE
        real_profile_c = compute_radial_power_spectrum(real_celeba)
        fake_profile_c = compute_radial_power_spectrum(fake_celeba)
        rpse_celeba = compute_rpse(real_profile_c, fake_profile_c)
        print(f"\n>>> RPSE (CelebA vs blurred): {rpse_celeba:.6f}")

        # WBED
        for wavelet in ['haar', 'db2']:
            real_e_c = compute_wavelet_band_energies(real_celeba, wavelet=wavelet)
            fake_e_c = compute_wavelet_band_energies(fake_celeba, wavelet=wavelet)
            wbed_celeba = compute_wbed(real_e_c, fake_e_c)
            print(f">>> WBED total ({wavelet}): {wbed_celeba['wbed_total']:.6f}")

        # Zapisz wizualizację profili
        output_dir = '/kaggle/working/spectral_metrics_output'
        os.makedirs(output_dir, exist_ok=True)

        # Plot profili
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # RPSE profile
            ax1 = axes[0]
            x = np.arange(len(real_profile_c.cpu().numpy()))
            ax1.plot(x, real_profile_c.cpu().numpy(), label='Real CelebA', linewidth=2)
            ax1.plot(x, fake_profile_c.cpu().numpy(), label='Fake (blurred)', linewidth=2, linestyle='--')
            ax1.set_xlabel('Radial frequency bin')
            ax1.set_ylabel('Power')
            ax1.set_title(f'Radial Power Spectrum (RPSE={rpse_celeba:.4f})')
            ax1.legend()
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)

            # WBED energies
            ax2 = axes[1]
            bands = ['LL', 'LH', 'HL', 'HH']
            x_pos = np.arange(len(bands))
            width = 0.35

            real_means = [real_e_c[b].mean().item() for b in bands]
            fake_means = [fake_e_c[b].mean().item() for b in bands]

            ax2.bar(x_pos - width/2, real_means, width, label='Real CelebA', alpha=0.8)
            ax2.bar(x_pos + width/2, fake_means, width, label='Fake (blurred)', alpha=0.8)
            ax2.set_xlabel('Wavelet Band')
            ax2.set_ylabel('Mean Energy')
            ax2.set_title(f'Wavelet Band Energies (WBED={wbed_celeba["wbed_total"]:.4f})')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(bands)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'spectral_metrics_comparison.png'), dpi=150)
            plt.close()

            print(f"\nWizualizacja zapisana: {output_dir}/spectral_metrics_comparison.png")

        except Exception as e:
            print(f"Nie udało się utworzyć wizualizacji: {e}")

    else:
        print(f"CelebA nie znalezione pod: {celeba_img_dir}")
        print("Pomijam test na prawdziwych obrazach.")

    # =========================================================================
    # Podsumowanie
    # =========================================================================
    print("\n" + "=" * 60)
    print("PODSUMOWANIE TESTÓW METRYK SPEKTRALNYCH")
    print("=" * 60)
    print(f"✓ RPSE (szum gaussian):     {rpse:.6f}")
    print(f"✓ RPSE (wzory geometryczne): {rpse_pattern:.6f}")
    if os.path.exists(celeba_img_dir):
        print(f"✓ RPSE (CelebA vs blurred): {rpse_celeba:.6f}")
    print()
    print("Testy zakończone pomyślnie!")
    print("=" * 60)


if __name__ == "__main__":
    # Wybierz tryb działania:
    # - "training": pełny trening GAN
    # - "wavelet_tests": testy DWT2D/IDWT2D
    # - "spectral_metrics": testy metryk RPSE i WBED
    MODE = "spectral_metrics"

    if MODE == "training":
        run_training()
    elif MODE == "wavelet_tests":
        run_wavelet_tests()
    elif MODE == "spectral_metrics":
        run_spectral_metrics_test()
    else:
        print(f"Nieznany tryb: {MODE}")
        print("Dostępne tryby: 'training', 'wavelet_tests', 'spectral_metrics'")

