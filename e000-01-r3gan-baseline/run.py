#!/usr/bin/env python3
"""
e000-01-r3gan-baseline — local runner

Usage:
    python run.py --profile fast
    python run.py --profile overnight --data-dir /data/celeba/img_align_celeba
    python run.py --profile smoke
    python run.py --profile fast --override steps=5000 batch_size=64
    python run.py --profile overnight --resume latest

Profiles: smoke, fast, overnight  (+ base)
"""

import argparse
import glob
import os
import sys
import yaml
from dataclasses import fields
from typing import Any, Dict, Tuple, Union, get_args, get_origin, get_type_hints


def _coerce_override_value(value: Any, expected_type: Any) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return value

    if origin is Union:
        last_error = None
        for variant in args:
            try:
                return _coerce_override_value(value, variant)
            except ValueError as exc:
                last_error = exc
        raise ValueError(str(last_error) if last_error is not None else "invalid value")

    if expected_type is type(None):
        if value is None:
            return None
        raise ValueError("expected null")

    if expected_type is bool:
        if isinstance(value, bool):
            return value
        raise ValueError("expected bool (true/false)")

    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("expected int")
        return value

    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected float")
        return float(value)

    if expected_type is str:
        if not isinstance(value, str):
            raise ValueError("expected string")
        return value

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected tuple/list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_override_value(v, args[0]) for v in value)
        if args and len(value) != len(args):
            raise ValueError(f"expected tuple length {len(args)}")
        if not args:
            return tuple(value)
        return tuple(_coerce_override_value(v, t) for v, t in zip(value, args))

    if isinstance(expected_type, type):
        if not isinstance(value, expected_type):
            raise ValueError(f"expected {expected_type.__name__}")
        return value

    return value


def _parse_overrides(items, *, valid_keys, type_hints):
    overrides: Dict[str, Any] = {}
    for item in (items or []):
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}' (expected key=value)")

        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in valid_keys:
            raise ValueError(f"Unknown override key: {key}")

        lowered = raw_value.strip().lower()
        if lowered in {"none", "null", "~"}:
            parsed_value = None
        else:
            parsed_value = yaml.safe_load(raw_value)
            if parsed_value is None:
                parsed_value = raw_value

        expected_type = type_hints.get(key, Any)
        try:
            overrides[key] = _coerce_override_value(parsed_value, expected_type)
        except ValueError as exc:
            raise ValueError(
                f"Invalid override value for '{key}': {raw_value!r} ({exc})"
            ) from exc

    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="e000-01-r3gan-baseline — local runner",
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
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from checkpoint path, checkpoint dir, or 'latest' from out_dir/checkpoints.",
    )
    args = parser.parse_args()

    from src import train, get_config
    from src.config_loader import RunConfig

    type_hints = get_type_hints(RunConfig)
    valid_keys = {f.name for f in fields(RunConfig)}

    try:
        overrides = _parse_overrides(args.override, valid_keys=valid_keys, type_hints=type_hints)
    except ValueError as exc:
        parser.error(str(exc))

    data_dir = args.data_dir or os.environ.get("DATA_DIR")
    if data_dir:
        overrides["data_dir"] = data_dir
    if args.out_dir:
        overrides["out_dir"] = args.out_dir
    out_dir_explicit = "out_dir" in overrides

    print("=" * 60)
    print("e000-01-r3gan-baseline")
    print("=" * 60)
    print(f"Profile  : {args.profile}")
    if overrides:
        print(f"Overrides: {overrides}")
    print()

    try:
        cfg = get_config(args.profile, overrides)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if cfg.dataset_name == "celeba" and not cfg.data_dir:
        print("❌  data_dir is required for CelebA.")
        print("    Use: --data-dir /path/to/img_align_celeba  or DATA_DIR env var.")
        sys.exit(1)

    if args.resume and args.resume.strip().lower() == "latest" and not out_dir_explicit:
        ckpt_pattern = os.path.join(cfg.out_dir, "checkpoints", "ckpt_*.pt")
        if not glob.glob(ckpt_pattern):
            parser.error(
                "--resume latest looked in auto-generated out_dir with no checkpoints: "
                f"{cfg.out_dir}. "
                "Pass --out-dir <previous_artifacts_dir> to point at an existing run "
                "or use --resume <checkpoint_path_or_dir>."
            )

    train(args.profile, overrides, resume=args.resume)


if __name__ == "__main__":
    main()


