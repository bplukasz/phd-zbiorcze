"""
Moduł do zarządzania konfiguracją eksperymentów.
Wspiera hierarchiczne ładowanie konfiguracji: base + profile-specific overrides.
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class RunConfig:
    """Konfiguracja eksperymentu."""

    # Metadata
    name: str = "preview"

    # Training
    steps: int = 30000
    batch_size: int = 64
    lr_G: float = 2e-4
    lr_D: float = 2e-4
    betas: Tuple[float, float] = (0.0, 0.99)

    # Model
    z_dim: int = 128
    img_size: int = 128
    img_channels: int = 3
    g_ch: int = 64
    d_ch: int = 64

    # Discriminator wavelet branch (optional)
    use_wavelet_branch: bool = False
    wavelet_hf_only: bool = False
    wavelet_type: str = "haar"  # haar | db2
    wavelet_level: int = 1

    # EMA
    ema_decay: float = 0.999

    # DiffAugment
    diffaug_policy: str = "color,translation,cutout"

    # Regularization
    use_r1_penalty: bool = False
    r1_lambda: float = 10.0
    r1_every: int = 16

    # --- Wavelet-energy matching regularization (optional) ---
    # Dopasowanie statystyk energii pasm DWT (LL/LH/HL/HH) pomiędzy real i fake.
    use_wavereg: bool = False
    lambda_wavereg: float = 0.0
    # gdzie dodać karę: "g" | "d" | "both"
    wavereg_apply_to: str = "g"
    wavereg_wavelet: str = "haar"  # haar | db2
    wavereg_eps: float = 1e-8

    # --- Fourier (FFT) energy matching regularization (baseline, optional) ---
    use_fftreg: bool = False
    lambda_fftreg: float = 0.0
    fftreg_apply_to: str = "g"  # g | d | both
    fftreg_num_bins: int = 16
    fftreg_downsample_to: int = 64
    fftreg_every: int = 1
    fftreg_eps: float = 1e-8

    # Logging
    log_every: int = 1
    grid_every: int = 1000
    ckpt_every: int = 5000
    eval_every: int = 10000
    live: bool = False
    use_wandb: bool = True

    # Evaluation
    eval_samples: int = 50000
    fid_samples: int = 10000

    # --- Spectral / wavelet metrics (RPSE, WBED) ---
    # Uwaga: te metryki są dużo tańsze niż FID/KID, ale nadal warto limitować liczbę obrazów.
    spectral_metrics: bool = True
    spectral_max_images: int = 1000
    spectral_img_size: Optional[int] = None  # np. 128; None = bez resize
    wbed_wavelet: str = "haar"  # haar | db2

    # Dataset
    dataset_name: str = "celeba"  # celeba, cifar10, cifar100, mnist, fashion_mnist

    # Output
    out_dir: str = "/kaggle/working/artifacts"
    data_dir: str = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Aktualizuje konfigurację wartościami ze słownika."""
        for key, value in updates.items():
            if hasattr(self, key):
                # Konwertuj listy na tuple dla pola betas
                if key == 'betas' and isinstance(value, list):
                    value = tuple(value)
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje konfigurację do słownika."""
        return asdict(self)


class ConfigLoader:
    """Klasa do ładowania konfiguracji z plików YAML."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: Katalog z plikami konfiguracyjnymi. Jeśli None, używa katalogu 'configs'
                       w tym samym miejscu co ten plik.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        self.config_dir = Path(config_dir)

    def load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Ładuje plik YAML."""
        if not filepath.exists():
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}

    def get_config(self, profile: str = "preview",
                   overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
        """
        Ładuje konfigurację dla danego profilu.

        Args:
            profile: Nazwa profilu (np. "preview", "train", "smoke")
            overrides: Dodatkowe nadpisania z argumentów CLI

        Returns:
            RunConfig z połączoną konfiguracją
        """
        config = RunConfig()

        # 1. Załaduj bazową konfigurację
        base_config = self.load_yaml(self.config_dir / "base.yaml")
        if base_config:
            config.update_from_dict(base_config)

        # 2. Załaduj konfigurację profilu
        profile = profile.strip().lower()
        profile_config = self.load_yaml(self.config_dir / f"{profile}.yaml")
        if profile_config:
            config.update_from_dict(profile_config)

        # 3. Zastosuj CLI overrides
        if overrides:
            config.update_from_dict(overrides)

        return config

    def save_config(self, config: RunConfig, output_path: str) -> None:
        """Zapisuje aktualną konfigurację do pliku YAML."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_config(profile: str = "preview",
               overrides: Optional[Dict[str, Any]] = None,
               config_dir: Optional[str] = None) -> RunConfig:
    """
    Wygodna funkcja do ładowania konfiguracji.

    Args:
        profile: Nazwa profilu
        overrides: Dodatkowe nadpisania
        config_dir: Katalog z konfiguracją (opcjonalnie)

    Returns:
        RunConfig z załadowaną konfiguracją
    """
    loader = ConfigLoader(config_dir)
    return loader.get_config(profile, overrides)
