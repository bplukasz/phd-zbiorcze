#!/usr/bin/env python3
"""
e001-02-r3gan-baseline — local runner

Usage:
    python run.py --profile fast
    python run.py --profile overnight --data-dir /data/celeba/img_align_celeba
    python run.py --profile smoke
    python run.py --profile fast --override steps=5000 batch_size=64

Profiles: smoke, fast, overnight  (+ base)
"""

import argparse
import os
import sys


def _parse_overrides(items):
    overrides = {}
    for item in (items or []):
        if "=" not in item:
            print(f"⚠  Invalid override (expected key=value): {item}")
            continue
        key, value = item.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        overrides[key] = value
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="e001-02-r3gan-baseline — local runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--profile", "-p", default="base",
                        help="Config profile (base | fast | overnight | smoke)")
    parser.add_argument("--data-dir", default=None,
                        help="Override data_dir. Also reads DATA_DIR env var.")
    parser.add_argument("--out-dir", default=None,
                        help="Override out_dir.")
    parser.add_argument("--override", "-o", nargs="*", default=[],
                        help="Extra key=value overrides, e.g. steps=5000 batch_size=64")
    args = parser.parse_args()

    overrides = _parse_overrides(args.override)

    data_dir = args.data_dir or os.environ.get("DATA_DIR")
    if data_dir:
        overrides["data_dir"] = data_dir
    if args.out_dir:
        overrides["out_dir"] = args.out_dir

    print("=" * 60)
    print("e001-02-r3gan-baseline")
    print("=" * 60)
    print(f"Profile  : {args.profile}")
    if overrides:
        print(f"Overrides: {overrides}")
    print()

    # Add repo root to path so r3gan-source.py is importable
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src import train, get_config

    cfg = get_config(args.profile, overrides)
    if cfg.dataset_name == "celeba" and not cfg.data_dir:
        print("❌  data_dir is required for CelebA.")
        print("    Use: --data-dir /path/to/img_align_celeba  or DATA_DIR env var.")
        sys.exit(1)

    train(args.profile, overrides)


if __name__ == "__main__":
    main()

