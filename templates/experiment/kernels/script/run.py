#!/usr/bin/env python3
"""
{{FULL_NAME}} - Script Runner
"""

import os
import sys

print("=" * 60)
print("{{FULL_NAME}} - Script Runner")
print("=" * 60)

# Paths setup
print("\nInputs:", os.listdir("/kaggle/input"))

CODE_DIR = "/kaggle/input/{{FULL_NAME}}-lib"
sys.path.insert(0, CODE_DIR)

{{SHARED_IMPORT}}

# Verify paths
print(f"CODE_DIR exists: {os.path.exists(CODE_DIR)}")

# ============================================================================
# Configuration
# ============================================================================

# Wybierz profil treningu:
# - "preview": 200 kroków, szybki test
# - "smoke": 500 kroków, weryfikacja pipeline
# - "train": pełny trening (5k kroków)
PROFILE = "train"

# Opcjonalne nadpisania konfiguracji:
# UWAGA: Na Kaggle W&B wymaga API key. Jeśli go nie masz, wyłącz:
OVERRIDES = {
    # 'use_wandb': False,  # Wyłącz W&B jeśli brak API key
    # 'steps': 10000,
    # 'batch_size': 128,
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
    print(f"  LR: {cfg.lr}")
    print(f"  W&B: {cfg.use_wandb}")
    print()

    # Run training
    model, losses = train(PROFILE, OVERRIDES)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Artifacts saved to: {cfg.out_dir}")
    print("=" * 60)


