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
PROFILE = "smoke"

# Opcjonalne nadpisania konfiguracji (ustaw None żeby nie nadpisywać):
OVERRIDES = {
    # 'steps': 10000,
    # 'batch_size': 128,
    # 'use_wandb': False,
    # 'eval_every': 5000,
}

# ============================================================================
# Import and run
# ============================================================================

from src import train, get_config

if __name__ == "__main__":
    # Display config
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



