"""
e000-01-r3gan-baseline — config loader
Hierarchiczne ładowanie YAML: base.yaml + {profile}.yaml + CLI overrides
"""

import os
import re
import yaml
from dataclasses import dataclass, asdict, fields
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, get_args, get_origin, get_type_hints


def _to_yaml_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    return value


def _auto_out_dir(profile: str, base_dir: Optional[Path] = None) -> str:
    """
    Generuje nazwę katalogu artifacts w formacie:
        artifacts-MM-DD-{LP:02d}-{profile}
    LP to kolejny numer z danego dnia (01, 02, …), wyznaczany na podstawie
    istniejących podfolderów pasujących do wzorca w katalogu eksperymentu.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]

    today = date.today()
    mm = today.strftime("%m")
    dd = today.strftime("%d")
    prefix = f"artifacts-{mm}-{dd}-"

    pattern = re.compile(rf"^artifacts-{mm}-{dd}-(\d{{2}})-")
    max_lp = 0
    for entry in base_dir.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                max_lp = max(max_lp, int(m.group(1)))

    lp = max_lp + 1
    safe_profile = re.sub(r"[^a-zA-Z0-9_-]", "-", profile)
    dir_name = f"{prefix}{lp:02d}-{safe_profile}"
    return str(base_dir / dir_name)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"expected bool, got {type(value).__name__}")


def _coerce_value(value: Any, expected_type: Any) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return value

    if origin is Union:
        last_error = None
        for variant in args:
            try:
                return _coerce_value(value, variant)
            except ValueError as exc:
                last_error = exc
        raise ValueError(str(last_error) if last_error is not None else "invalid union value")

    if expected_type is type(None):
        if value is None:
            return None
        raise ValueError("expected null")

    if expected_type is bool:
        return _coerce_bool(value)

    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"expected int, got {type(value).__name__}")
        return value

    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"expected float, got {type(value).__name__}")
        return float(value)

    if expected_type is str:
        if not isinstance(value, str):
            raise ValueError(f"expected str, got {type(value).__name__}")
        return value

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"expected tuple/list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(v, args[0]) for v in value)
        if args and len(value) != len(args):
            raise ValueError(f"expected tuple of len={len(args)}, got len={len(value)}")
        if not args:
            return tuple(value)
        return tuple(_coerce_value(v, t) for v, t in zip(value, args))

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"expected list, got {type(value).__name__}")
        item_type = args[0] if args else Any
        return [_coerce_value(v, item_type) for v in value]

    if isinstance(expected_type, type):
        if not isinstance(value, expected_type):
            raise ValueError(f"expected {expected_type.__name__}, got {type(value).__name__}")
        return value

    return value


@dataclass
class RunConfig:
    """Konfiguracja eksperymentu e000-01-r3gan-baseline."""

    name: str = "base"

    seed: int = 42
    deterministic: bool = False

    steps: int = 100_000
    batch_size: int = 64
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.0, 0.99)
    gamma: float = 10.0
    ema_beta: float = 0.999

    use_amp_for_g: bool = True
    use_amp_for_d: bool = False
    channels_last: bool = True
    grad_clip: Optional[float] = None

    z_dim: int = 256
    img_resolution: int = 64
    base_channels: int = 96
    channel_max: int = 768
    blocks_per_stage: int = 2
    expansion_factor: int = 2
    group_size: int = 16
    resample_mode: str = "bilinear"
    out_channels: int = 3
    in_channels: int = 3

    dataset_name: str = "cifar10"
    img_channels: int = 3

    log_every: int = 100
    grid_every: int = 1000
    ckpt_every: int = 10_000
    save_n_samples: int = 64
    real_grid_samples: int = 64

    metrics_every: int = 10_000
    metrics_num_fake: int = 10_000
    metrics_fake_batch_size: int = 256
    metrics_fid_feature: int = 2048
    metrics_kid_feature: int = 2048
    metrics_kid_subsets: int = 100
    metrics_kid_subset_size: int = 1000
    metrics_max_real: int = 50_000
    metrics_pr_num_samples: int = 10_000
    metrics_pr_k: int = 3
    metrics_lpips_num_pairs: int = 2048
    metrics_lpips_pool_size: int = 4096
    metrics_amp_dtype: str = "bf16"

    out_dir: str = "./artifacts"
    data_dir: str = ""

    def update_from_dict(self, d: Dict[str, Any], *, source: str = "override", strict: bool = True) -> None:
        type_hints = get_type_hints(type(self))
        known_keys = {f.name for f in fields(self)}
        unknown_keys = sorted(k for k in d if k not in known_keys)
        if strict and unknown_keys:
            raise ValueError(f"Unknown config keys in {source}: {', '.join(unknown_keys)}")

        for key, value in d.items():
            if key not in known_keys:
                continue
            expected_type = type_hints.get(key, Any)
            try:
                coerced = _coerce_value(value, expected_type)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value for key '{key}' in {source}: {exc}. Received: {value!r}"
                ) from exc
            setattr(self, key, coerced)

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.log_every <= 0:
            raise ValueError("log_every must be > 0")
        if self.grid_every <= 0:
            raise ValueError("grid_every must be > 0")
        if self.ckpt_every <= 0:
            raise ValueError("ckpt_every must be > 0")
        if self.save_n_samples <= 0:
            raise ValueError("save_n_samples must be > 0")
        if self.real_grid_samples <= 0:
            raise ValueError("real_grid_samples must be > 0")
        if self.img_resolution < 4 or (self.img_resolution & (self.img_resolution - 1)) != 0:
            raise ValueError("img_resolution must be a power of two and >= 4")
        if self.img_channels <= 0:
            raise ValueError("img_channels must be > 0")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if self.dataset_name.strip().lower() == "celeba" and not self.data_dir:
            raise ValueError("data_dir is required when dataset_name=celeba")
        if len(self.betas) != 2:
            raise ValueError("betas must contain exactly two values")
        b1, b2 = self.betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("betas values must be in [0, 1)")

        if self.metrics_every < 0:
            raise ValueError("metrics_every must be >= 0")
        if self.metrics_every == 0:
            return
        if self.metrics_num_fake <= 0:
            raise ValueError("metrics_num_fake must be > 0 when metrics are enabled")
        if self.metrics_fake_batch_size <= 0:
            raise ValueError("metrics_fake_batch_size must be > 0 when metrics are enabled")
        if self.metrics_kid_subsets <= 0:
            raise ValueError("metrics_kid_subsets must be > 0")
        if self.metrics_kid_subset_size <= 0:
            raise ValueError("metrics_kid_subset_size must be > 0")
        if self.metrics_kid_subset_size > self.metrics_num_fake:
            raise ValueError("metrics_kid_subset_size cannot be larger than metrics_num_fake")
        if self.metrics_max_real < self.metrics_kid_subset_size:
            raise ValueError("metrics_max_real cannot be smaller than metrics_kid_subset_size")
        if self.metrics_pr_k <= 0:
            raise ValueError("metrics_pr_k must be > 0")
        if self.metrics_pr_num_samples <= self.metrics_pr_k:
            raise ValueError("metrics_pr_num_samples must be greater than metrics_pr_k")
        if self.metrics_lpips_pool_size < 2:
            raise ValueError("metrics_lpips_pool_size must be >= 2")
        if self.metrics_lpips_num_pairs <= 0:
            raise ValueError("metrics_lpips_num_pairs must be > 0")
        if self.metrics_amp_dtype not in {"bf16", "fp16"}:
            raise ValueError("metrics_amp_dtype must be one of: bf16, fp16")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigLoader:
    def __init__(self, config_dir: Union[str, Path, None] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        assert config_dir is not None
        self.config_dir = Path(config_dir)

    def _load_yaml(self, path: Path, *, source: str) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {source}, got {type(data).__name__}")
        return data

    def get_config(self, profile: str = "base", overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
        profile_name = profile.strip().lower()
        cfg = RunConfig()

        cfg.update_from_dict(
            self._load_yaml(self.config_dir / "base.yaml", source="base"),
            source="base",
            strict=True,
        )

        if profile_name != "base":
            profile_path = self.config_dir / f"{profile_name}.yaml"
            if not profile_path.exists():
                raise FileNotFoundError(f"Profile config not found: {profile_path}")
            cfg.update_from_dict(
                self._load_yaml(profile_path, source="profile"),
                source="profile",
                strict=True,
            )

        out_dir_overridden = overrides is not None and "out_dir" in overrides
        if overrides:
            cfg.update_from_dict(overrides, source="override", strict=True)

        if not out_dir_overridden and cfg.out_dir == "./artifacts":
            cfg.out_dir = _auto_out_dir(profile_name)

        cfg.validate()
        return cfg

    def save_config(self, cfg: RunConfig, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_to_yaml_safe(cfg.to_dict()), f, default_flow_style=False, sort_keys=False)


def get_config(profile: str = "base", overrides: Optional[Dict[str, Any]] = None,
               config_dir: Union[str, Path, None] = None) -> RunConfig:
    return ConfigLoader(config_dir).get_config(profile, overrides)
