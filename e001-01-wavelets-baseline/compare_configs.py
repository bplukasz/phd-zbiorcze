#!/usr/bin/env python3
"""
Skrypt pomocniczy: Porównanie konfiguracji przed/po poprawkach
Użycie: python compare_configs.py
"""

import yaml
from pathlib import Path

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def print_comparison(title, before, after, changed_keys):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    for key in changed_keys:
        before_val = before.get(key, "N/A")
        after_val = after.get(key, "N/A")

        if before_val != after_val:
            symbol = "✅" if after_val != "N/A" else "⚠️"
            print(f"{symbol} {key:20s}: {str(before_val):15s} → {str(after_val):15s}")
        else:
            print(f"  {key:20s}: {str(before_val):15s} (bez zmian)")

def main():
    configs_dir = Path(__file__).parent / "dataset/src/configs"

    # Odtworzenie starych wartości dla porównania
    old_smoke = {
        'name': 'smoke',
        'steps': 500,
        'batch_size': 32,
        'lr_D': 0.0002,  # inherited from base
        'lr_G': 0.0002,
        'eval_every': 250,
        'use_r1_penalty': False,
        'r1_lambda': None,
    }

    old_train = {
        'name': 'train',
        'steps': 30000,
        'batch_size': 64,
        'lr_D': 0.0002,
        'lr_G': 0.0002,
        'eval_every': 10000,
        'use_r1_penalty': False,
        'r1_lambda': None,
    }

    # Nowe wartości
    new_smoke = load_yaml(configs_dir / "smoke.yaml")
    new_train = load_yaml(configs_dir / "train.yaml")

    # Porównanie smoke
    print_comparison(
        "SMOKE.YAML - Smoke Test",
        old_smoke,
        new_smoke,
        ['lr_D', 'lr_G', 'eval_every', 'use_r1_penalty']
    )

    # Porównanie train
    print_comparison(
        "TRAIN.YAML - Pełny Trening",
        old_train,
        new_train,
        ['lr_D', 'lr_G', 'eval_every', 'use_r1_penalty', 'r1_lambda']
    )

    # Podsumowanie
    print(f"\n{'='*60}")
    print("  PODSUMOWANIE ZMIAN")
    print(f"{'='*60}")
    print("\n🎯 Główne poprawki:")
    print("  1. lr_D: 0.0002 → 0.0001 (zmniejszenie o 50%)")
    print("     Powód: D dominował nad G, FID rósł")
    print()
    print("  2. eval_every (smoke): 250 → 100")
    print("     Powód: Wcześniejsza detekcja problemów")
    print()
    print("  3. R1 penalty: włączony w train profile")
    print("     Powód: Stabilizacja, zapobieganie mode collapse")
    print()
    print("\n📊 Oczekiwane rezultaty:")
    print("  • Smoke (500 kroków): FID < 300 (było 425)")
    print("  • Train (30k kroków): FID < 50")
    print()
    print("📖 Szczegóły: ANALIZA_WYNIKOW.md, TO_NAPRAWIENIE.md")
    print()

if __name__ == "__main__":
    main()
