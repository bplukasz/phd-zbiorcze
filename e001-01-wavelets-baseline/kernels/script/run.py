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


if __name__ == "__main__":
    # Wybierz tryb działania:
    MODE = "wavelet_tests"  # "training" lub "wavelet_tests"

    if MODE == "training":
        run_training()
    elif MODE == "wavelet_tests":
        run_wavelet_tests()
    else:
        print(f"Nieznany tryb: {MODE}")
        print("Dostępne tryby: 'training', 'wavelet_tests'")

