#!/usr/bin/env python3
"""
e001-01-wavelets-baseline — local runner (MSI EDGEXPERT)

Usage:
    python run.py --profile train
    python run.py --profile fast --override steps=500 batch_size=32
    python run.py --profile train --data-dir /path/to/celeba/img_align_celeba
    DATA_DIR=/data/celeba python run.py --profile train

Profiles: preview, smoke, train, fast, fast64, fast-e13, fast-e13-base, ...
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="e001-01-wavelets-baseline — local runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile", "-p",
        default="preview",
        help="Config profile name (default: preview)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data_dir (path to dataset). Also reads DATA_DIR env var.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override out_dir (path to artifacts output).",
    )
    parser.add_argument(
        "--override", "-o",
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs, e.g. steps=500 batch_size=32",
    )
    parser.add_argument(
        "--mode",
        choices=["training", "wavelet_tests", "spectral_metrics"],
        default="training",
        help="Run mode (default: training)",
    )
    args = parser.parse_args()

    # Build overrides dict
    overrides = {}
    for item in args.override:
        if "=" not in item:
            print(f"⚠ Invalid override (expected key=value): {item}")
            continue
        key, value = item.split("=", 1)
        # Try to parse as int/float/bool
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        overrides[key] = value

    # data_dir: CLI > env var > config
    data_dir = args.data_dir or os.environ.get("DATA_DIR")
    if data_dir:
        overrides["data_dir"] = data_dir

    # out_dir: CLI > config
    if args.out_dir:
        overrides["out_dir"] = args.out_dir

    print("=" * 60)
    print("e001-01-wavelets-baseline — local runner")
    print("=" * 60)
    print(f"Profile: {args.profile}")
    print(f"Mode:    {args.mode}")
    if overrides:
        print(f"Overrides: {overrides}")
    print()

    from src import train, get_config

    if args.mode == "training":
        cfg = get_config(args.profile, overrides)
        if not cfg.data_dir:
            print("❌ data_dir is not set!")
            print("   Set it via: --data-dir, DATA_DIR env var, or in configs/base.yaml")
            sys.exit(1)
        model, losses = train(args.profile, overrides)
        print(f"\nFinal loss_G: {losses[-1]:.4f}")

    elif args.mode == "wavelet_tests":
        from src.wavelets import run_all_tests
        cfg = get_config(args.profile, overrides)
        output_dir = os.path.join(cfg.out_dir, "dwt_test_output")
        results = run_all_tests(output_dir=output_dir)
        print(f"Results saved to: {output_dir}")

    elif args.mode == "spectral_metrics":
        from src.metrics import (
            compute_radial_power_spectrum,
            compute_rpse,
            compute_wavelet_band_energies,
            compute_wbed,
        )
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        # Quick synthetic test
        torch.manual_seed(42)
        real_imgs = torch.randn(100, 3, 64, 64, device=device)
        fake_imgs = torch.randn(100, 3, 64, 64, device=device) * 0.5

        real_profile = compute_radial_power_spectrum(real_imgs)
        fake_profile = compute_radial_power_spectrum(fake_imgs)
        rpse = compute_rpse(real_profile, fake_profile)
        print(f"RPSE (gaussian noise test): {rpse:.6f}")

        real_energies = compute_wavelet_band_energies(real_imgs, wavelet="haar")
        fake_energies = compute_wavelet_band_energies(fake_imgs, wavelet="haar")
        wbed_result = compute_wbed(real_energies, fake_energies)
        print(f"WBED total (haar): {wbed_result['wbed_total']:.6f}")


if __name__ == "__main__":
    main()

