#!/usr/bin/env python3
"""
Przykład użycia systemu konfiguracji
"""

import sys
import os

# Dodaj ścieżkę do modułu (w prawdziwym użyciu nie potrzeba - jest w PYTHONPATH)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset'))

from src import get_config

def demo_basic():
    """Podstawowe użycie"""
    print("=" * 60)
    print("PODSTAWOWE UŻYCIE")
    print("=" * 60)

    # Załaduj profil
    cfg = get_config("preview")

    print(f"\nProfil: {cfg.name}")
    print(f"Steps: {cfg.steps}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate G: {cfg.lr_G}")
    print(f"Use W&B: {cfg.use_wandb}")
    print(f"Live mode: {cfg.live}")

def demo_overrides():
    """Użycie z nadpisaniami"""
    print("\n" + "=" * 60)
    print("Z NADPISANIAMI")
    print("=" * 60)

    # Załaduj z nadpisaniami
    cfg = get_config("train", overrides={
        "steps": 50000,
        "batch_size": 128,
        "lr_G": 1e-4,
        "use_wandb": False
    })

    print(f"\nProfil: {cfg.name}")
    print(f"Steps: {cfg.steps} <- OVERRIDE")
    print(f"Batch size: {cfg.batch_size} <- OVERRIDE")
    print(f"Learning rate G: {cfg.lr_G} <- OVERRIDE")
    print(f"Use W&B: {cfg.use_wandb} <- OVERRIDE")
    print(f"EMA decay: {cfg.ema_decay} (z base)")

def demo_all_profiles():
    """Porównanie wszystkich profili"""
    print("\n" + "=" * 60)
    print("PORÓWNANIE PROFILI")
    print("=" * 60)

    profiles = ["preview", "smoke", "train"]

    print(f"\n{'Parametr':<20} {'preview':<15} {'smoke':<15} {'train':<15}")
    print("-" * 65)

    params = ["steps", "batch_size", "live", "use_wandb", "eval_every", "fid_samples"]

    for param in params:
        values = []
        for profile in profiles:
            cfg = get_config(profile)
            values.append(str(getattr(cfg, param)))
        print(f"{param:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")

def demo_save_and_load():
    """Zapisywanie i wczytywanie konfiguracji"""
    print("\n" + "=" * 60)
    print("ZAPISYWANIE I WCZYTYWANIE")
    print("=" * 60)

    from src import ConfigLoader
    import yaml

    # Załaduj i zapisz
    cfg = get_config("smoke", overrides={"steps": 1000})

    output_file = "/tmp/demo_config.yaml"
    loader = ConfigLoader()
    loader.save_config(cfg, output_file)

    print(f"\n✓ Zapisano konfigurację do: {output_file}")

    # Wczytaj z powrotem
    with open(output_file, 'r') as f:
        data = yaml.safe_load(f)

    print(f"\n📄 Zawartość pliku:")
    print(yaml.dump(data, default_flow_style=False, sort_keys=False))

def demo_hierarchical():
    """Demonstracja hierarchii"""
    print("\n" + "=" * 60)
    print("HIERARCHIA KONFIGURACJI")
    print("=" * 60)

    print("""
    Hierarchia (od najniższego do najwyższego priorytetu):
    
    1. base.yaml       <- domyślne wartości
    2. {profile}.yaml  <- nadpisuje wybrane wartości z base
    3. overrides       <- nadpisuje wszystko
    """)

    print("Przykład dla 'preview' z override steps=100:")
    cfg = get_config("preview", overrides={"steps": 100})

    print(f"\n  steps: {cfg.steps}")
    print(f"    └─ źródło: override (100)")

    print(f"\n  batch_size: {cfg.batch_size}")
    print(f"    └─ źródło: preview.yaml (16)")

    print(f"\n  lr_G: {cfg.lr_G}")
    print(f"    └─ źródło: base.yaml (0.0002)")

    print(f"\n  live: {cfg.live}")
    print(f"    └─ źródło: preview.yaml (True)")

def main():
    """Uruchom wszystkie demo"""
    print("\n" + "=" * 60)
    print("DEMO SYSTEMU KONFIGURACJI")
    print("=" * 60)

    demo_basic()
    demo_overrides()
    demo_all_profiles()
    demo_save_and_load()
    demo_hierarchical()

    print("\n" + "=" * 60)
    print("✓ DEMO ZAKOŃCZONE")
    print("=" * 60)
    print("\nSprawdź też:")
    print("  - CONFIG_SYSTEM.md - pełna dokumentacja")
    print("  - test_config.py - testy systemu")
    print()

if __name__ == "__main__":
    main()
