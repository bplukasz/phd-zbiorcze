"""
Moduł do zarządzania konfiguracją eksperymentów.
Wspiera hierarchiczne ładowanie konfiguracji: base + profile-specific overrides.
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class RunConfig:
    """Konfiguracja eksperymentu."""

    # Metadata
    name: str = "preview"

    # Training
    steps: int = 5000
    batch_size: int = 64
    lr: float = 0.001

    # Logging
    log_every: int = 10
    viz_every: int = 100
    ckpt_every: int = 1000
    live: bool = False
    use_wandb: bool = True

    # Output
    out_dir: str = "/kaggle/working/artifacts"
    data_dir: str = "/kaggle/input/your-dataset"

    # CUSTOM: Dodaj tutaj własne pola specyficzne dla eksperymentu
    # np.: model_dim: int = 256
    #      dropout: float = 0.1

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Aktualizuje konfigurację wartościami ze słownika."""
        for key, value in updates.items():
            if hasattr(self, key):
                # Zachowaj typ pola jeśli to tuple
                field_type = type(getattr(self, key))
                if field_type == tuple and isinstance(value, list):
                    value = tuple(value)
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' ignored")

    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje konfigurację do słownika."""
        data = asdict(self)
        # Konwertuj tuple na listy dla kompatybilności z YAML
        for key, value in data.items():
            if isinstance(value, tuple):
                data[key] = list(value)
        return data


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

        Proces ładowania:
        1. Załaduj base.yaml (domyślne wartości)
        2. Załaduj {profile}.yaml (nadpisuje base)
        3. Zastosuj overrides z argumentów (nadpisuje wszystko)

        Args:
            profile: Nazwa profilu (np. "preview", "train", "smoke")
            overrides: Dodatkowe nadpisania z argumentów CLI

        Returns:
            RunConfig z połączoną konfiguracją
        """
        # Zacznij od pustej konfiguracji
        config = RunConfig()

        # 1. Załaduj bazową konfigurację
        base_path = self.config_dir / "base.yaml"
        base_config = self.load_yaml(base_path)
        if base_config:
            config.update_from_dict(base_config)

        # 2. Załaduj konfigurację profilu
        profile = profile.strip().lower()
        profile_path = self.config_dir / f"{profile}.yaml"
        profile_config = self.load_yaml(profile_path)
        if profile_config:
            config.update_from_dict(profile_config)
        elif profile not in ["custom", ""]:
            print(f"Warning: Profile config '{profile}.yaml' not found, using base config")

        # 3. Zastosuj CLI overrides
        if overrides:
            config.update_from_dict(overrides)

        # Upewnij się że nazwa profilu jest ustawiona
        if not profile_config and profile:
            config.name = profile

        return config

    def save_config(self, config: RunConfig, output_path: str) -> None:
        """Zapisuje aktualną konfigurację do pliku YAML."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


# Singleton instance
_default_loader: Optional[ConfigLoader] = None


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
    global _default_loader

    if config_dir is not None or _default_loader is None:
        _default_loader = ConfigLoader(config_dir)

    return _default_loader.get_config(profile, overrides)
