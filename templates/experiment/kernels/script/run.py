#!/usr/bin/env python3
"""
{{FULL_NAME}} - Script Runner
"""

import os
import sys
import argparse

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

# Import and run
from src import train, get_config

def main():
    parser = argparse.ArgumentParser(
        description="{{FULL_NAME}} training"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="train",
        choices=["preview", "train", "smoke"],
        help="Training profile: preview (200 steps), smoke (500 steps), train (5k steps)"
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

    args = parser.parse_args()

    # Prepare overrides from CLI arguments
    overrides = {}
    if args.steps is not None:
        overrides['steps'] = args.steps
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.no_wandb:
        overrides['use_wandb'] = False

    # Get config with overrides
    cfg = get_config(args.profile, overrides)

    print(f"\nKonfiguracja:")
    print(f"  Profile: {cfg.name}")
    print(f"  Steps: {cfg.steps}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  LR: {cfg.lr}")
    print(f"  W&B: {cfg.use_wandb}")
    print()

    # Run training
    model, losses = train(args.profile, overrides)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Artifacts saved to: {cfg.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

