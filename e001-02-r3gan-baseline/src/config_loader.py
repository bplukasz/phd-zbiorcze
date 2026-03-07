"""
e001-02-r3gan-baseline — config loader
Hierarchiczne ładowanie YAML: base.yaml + {profile}.yaml + CLI overrides
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path


@dataclass
class RunConfig:
    """Konfiguracja eksperymentu e001-02-r3gan-baseline."""

    # Metadata
    name: str = "base"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Training
    steps: int = 100_000
    batch_size: int = 64
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.0, 0.99)
    gamma: float = 10.0
    ema_beta: float = 0.999

    # AMP / performance
    use_amp_for_g: bool = True
    use_amp_for_d: bool = False
    channels_last: bool = True
    grad_clip: Optional[float] = None

    # Model architecture
    z_dim: int = 256
    img_resolution: int = 64
    base_channels: int = 96
    channel_max: int = 768
    blocks_per_stage: int = 2
    expansion_factor: int = 2
    group_size: int = 16
    resample_mode: str = "bilinear"
    out_channels: int = 3    # output image channels
    in_channels: int = 3     # input image channels for discriminator

    # Dataset
    dataset_name: str = "cifar10"   # celeba | cifar10 | cifar100 | mnist | fashion_mnist
    img_channels: int = 3
    img_size: int = 64

    # Logging
    log_every: int = 100
    grid_every: int = 1000
    ckpt_every: int = 10_000
    save_n_samples: int = 64      # grid images to generate each grid step
    real_grid_samples: int = 64   # real image grid on startup

    # GAN Metrics (FID, KID, PR, LPIPS)
    metrics_every: int = 10_000           # 0 = disabled
    metrics_num_fake: int = 10_000        # fake images to generate per evaluation
    metrics_fake_batch_size: int = 256    # batch size for generator during eval
    metrics_fid_feature: int = 2048
    metrics_kid_feature: int = 2048
    metrics_kid_subsets: int = 100
    metrics_kid_subset_size: int = 1000
    metrics_max_real: int = 50_000        # max real images for FID/KID
    metrics_pr_num_samples: int = 10_000  # features for Precision/Recall
    metrics_pr_k: int = 3
    metrics_lpips_num_pairs: int = 2048
    metrics_lpips_pool_size: int = 4096
    metrics_amp_dtype: str = "bf16"       # "bf16" or "fp16"

    # Output
    out_dir: str = "./artifacts"
    data_dir: str = ""

    def update_from_dict(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                if k == "betas" and isinstance(v, list):
                    v = tuple(v)
                setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigLoader:
    def __init__(self, config_dir: Union[str, Path, None] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        assert config_dir is not None
        self.config_dir = Path(config_dir)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def get_config(self, profile: str = "base", overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
        cfg = RunConfig()
        cfg.update_from_dict(self._load_yaml(self.config_dir / "base.yaml"))
        cfg.update_from_dict(self._load_yaml(self.config_dir / f"{profile.strip().lower()}.yaml"))
        if overrides:
            cfg.update_from_dict(overrides)
        return cfg

    def save_config(self, cfg: RunConfig, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_config(profile: str = "base", overrides: Optional[Dict[str, Any]] = None,
               config_dir: Union[str, Path, None] = None) -> RunConfig:
    return ConfigLoader(config_dir).get_config(profile, overrides)
