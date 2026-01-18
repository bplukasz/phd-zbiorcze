#!/usr/bin/env python3
"""
Test systemu konfiguracji
"""

import sys
import os

# Dodaj ścieżkę do modułu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset'))

from src import get_config, RunConfig, ConfigLoader

def test_base_config():
    """Test ładowania bazowej konfiguracji"""
    print("=" * 60)
    print("Test 1: Ładowanie base config")
    print("=" * 60)

    cfg = get_config("custom")  # custom nie istnieje, więc używa tylko base
    print(f"Name: {cfg.name}")
    print(f"Steps: {cfg.steps}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"LR_G: {cfg.lr_G}")
    print(f"Use W&B: {cfg.use_wandb}")
    print("✓ Base config OK\n")

def test_profiles():
    """Test wszystkich profili"""
    print("=" * 60)
    print("Test 2: Profile")
    print("=" * 60)

    profiles = ["preview", "smoke", "train"]

    for profile in profiles:
        print(f"\n{profile.upper()}:")
        cfg = get_config(profile)
        print(f"  name: {cfg.name}")
        print(f"  steps: {cfg.steps}")
        print(f"  batch_size: {cfg.batch_size}")
        print(f"  live: {cfg.live}")
        print(f"  use_wandb: {cfg.use_wandb}")
        print(f"  fid_samples: {cfg.fid_samples}")

    print("\n✓ Wszystkie profile OK\n")

def test_overrides():
    """Test nadpisywania konfiguracji"""
    print("=" * 60)
    print("Test 3: Overrides")
    print("=" * 60)

    cfg = get_config("train", overrides={
        "steps": 50000,
        "batch_size": 128,
        "lr_G": 1e-4,
        "use_wandb": False
    })

    print(f"Name: {cfg.name}")
    print(f"Steps: {cfg.steps} (expected: 50000)")
    print(f"Batch size: {cfg.batch_size} (expected: 128)")
    print(f"LR_G: {cfg.lr_G} (expected: 0.0001)")
    print(f"Use W&B: {cfg.use_wandb} (expected: False)")

    assert cfg.steps == 50000
    assert cfg.batch_size == 128
    assert cfg.lr_G == 1e-4
    assert cfg.use_wandb == False

    print("✓ Overrides OK\n")

def test_save_config():
    """Test zapisywania konfiguracji"""
    print("=" * 60)
    print("Test 4: Zapisywanie konfiguracji")
    print("=" * 60)

    cfg = get_config("smoke")

    output_path = "/tmp/test_config.yaml"
    loader = ConfigLoader()
    loader.save_config(cfg, output_path)

    print(f"Zapisano do: {output_path}")

    # Sprawdź czy plik istnieje
    assert os.path.exists(output_path)

    # Wczytaj z powrotem
    import yaml
    with open(output_path, 'r') as f:
        data = yaml.safe_load(f)

    print(f"Wczytano: {len(data)} pól")
    print(f"Name: {data['name']}")
    print(f"Steps: {data['steps']}")

    assert data['name'] == 'smoke'
    assert data['steps'] == 500

    print("✓ Zapisywanie OK\n")

def test_to_dict():
    """Test konwersji do słownika"""
    print("=" * 60)
    print("Test 5: Konwersja do dict")
    print("=" * 60)

    cfg = get_config("preview")
    cfg_dict = cfg.to_dict()

    print(f"Liczba pól: {len(cfg_dict)}")
    print(f"Klucze: {list(cfg_dict.keys())[:5]}...")

    assert isinstance(cfg_dict, dict)
    assert 'steps' in cfg_dict
    assert 'batch_size' in cfg_dict
    assert cfg_dict['steps'] == 200

    print("✓ to_dict OK\n")

def test_hierarchical_override():
    """Test hierarchicznego nadpisywania: base -> profile -> overrides"""
    print("=" * 60)
    print("Test 6: Hierarchiczne nadpisywanie")
    print("=" * 60)

    # Base: steps=30000, batch_size=64, lr_G=2e-4
    # Preview: steps=200, batch_size=16 (pozostałe z base)
    # Override: steps=100 (pozostałe z preview)

    cfg = get_config("preview", overrides={"steps": 100})

    print(f"Steps: {cfg.steps} (expected: 100 from override)")
    print(f"Batch size: {cfg.batch_size} (expected: 16 from preview)")
    print(f"LR_G: {cfg.lr_G} (expected: 0.0002 from base)")
    print(f"Live: {cfg.live} (expected: True from preview)")

    assert cfg.steps == 100  # override
    assert cfg.batch_size == 16  # preview
    assert cfg.lr_G == 2e-4  # base
    assert cfg.live == True  # preview

    print("✓ Hierarchiczne nadpisywanie OK\n")

def main():
    """Uruchom wszystkie testy"""
    print("\n" + "=" * 60)
    print("TEST SYSTEMU KONFIGURACJI")
    print("=" * 60 + "\n")

    try:
        test_base_config()
        test_profiles()
        test_overrides()
        test_save_config()
        test_to_dict()
        test_hierarchical_override()

        print("=" * 60)
        print("✓ WSZYSTKIE TESTY PRZESZŁY POMYŚLNIE")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST NIEPOWODZENIE")
        print("=" * 60)
        print(f"Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
