#!/usr/bin/env python3
"""
e001-01-wavelets-baseline - Script Runner
ResNet GAN z hinge loss, SpectralNorm, EMA, DiffAugment na CelebA 128x128
"""

import os
import sys
import subprocess
import argparse

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
# Import and run
# ============================================================================

from src import train, get_config

def main():
    parser = argparse.ArgumentParser(
        description="ResNet GAN baseline training on CelebA 128x128"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="smoke",
        choices=["preview", "train", "smoke"],
        help="Training profile: preview (200 steps), smoke (500 steps), train (30k steps)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Override FID/KID evaluation interval"
    )

    args = parser.parse_args()

    # Prepare overrides from CLI arguments
    overrides = {}
    if args.steps is not None:
        overrides['steps'] = args.steps
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.no_wandb:
        overrides['use_wandb'] = False
    if args.eval_every is not None:
        overrides['eval_every'] = args.eval_every

    # Get config with overrides
    cfg = get_config(args.profile, overrides)

    print(f"\nKonfiguracja:")
    print(f"  Profile: {cfg.name}")
    print(f"  Steps: {cfg.steps}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  LR: {cfg.lr_G}")
    print(f"  W&B: {cfg.use_wandb}")
    print(f"  Eval every: {cfg.eval_every}")
    print()

    # Run training
    model, losses = train(args.profile, overrides)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final loss_G: {losses[-1]:.4f}")
    print(f"Artifacts saved to: {cfg.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

