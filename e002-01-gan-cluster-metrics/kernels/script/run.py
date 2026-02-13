#!/usr/bin/env python3
"""
e002-01-gan-cluster-metrics - Script Runner
ResNet GAN z Cluster-FID i E11 metrics na CIFAR-10
"""

import os
import sys
import subprocess

# ============================================================================
# Auto-instalacja zależności
# ============================================================================

def install_dependencies():
    """Instaluje wymagane pakiety."""
    packages = [
        "open_clip_torch==2.24.0",
        "scikit-learn==1.4.2",
        "scipy==1.11.4",
        "pandas",
    ]

    for pkg in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Uwaga: nie udało się zainstalować {pkg}: {e}")

print("=" * 60)
print("e002-01-gan-cluster-metrics - Script Runner")
print("=" * 60)

# ============================================================================
# Paths setup
# ============================================================================

print("\nInputs:", os.listdir("/kaggle/input"))

# Ścieżki do katalogów z kodem
# Uwaga: Kaggle montuje zawartość katalogu 'dataset' bezpośrednio
CODE_DIR = "/kaggle/input/e002-01-gan-cluster-metrics-lib"
SHARED_DIR = "/kaggle/input/shared-lib"

# Dodaj do sys.path - CODE_DIR/src musi być PRZED SHARED_DIR
# aby lokalne moduły (utils, data, models, eval) miały pierwszeństwo
# Uwaga: insert(0, x) wstawia na początek, więc ostatni insert jest pierwszy w liście
sys.path.insert(0, SHARED_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "src"))

print(f"CODE_DIR: {CODE_DIR}")
print(f"CODE_DIR contents: {os.listdir(CODE_DIR) if os.path.exists(CODE_DIR) else 'NOT FOUND'}")
print(f"sys.path (top 3): {sys.path[:3]}")

# Instaluj zależności
print("Instalowanie zależności...")
install_dependencies()

# ============================================================================
# Import kodu eksperymentu
# ============================================================================

print("Importowanie modułów...")
from src.train import main as train_main
from src.eval.precompute_real import main as precompute_main
from src.eval.eval_e11 import main as eval_e11_main

# ============================================================================
# Główny pipeline
# ============================================================================

def main():
    """Główna funkcja uruchamiająca eksperyment."""
    import argparse

    parser = argparse.ArgumentParser(description="E002-01 GAN Cluster Metrics Runner")
    parser.add_argument("--mode", type=str, default="full",
                       choices=["train", "precompute", "eval", "full"],
                       help="Tryb uruchomienia")
    parser.add_argument("--config", type=str,
                       default="/kaggle/input/e002-01-gan-cluster-metrics-lib/src/configs/cifar10_baseline.yml",
                       help="Ścieżka do pliku konfiguracyjnego")
    parser.add_argument("--ckpt", type=str, default=None,
                       help="Ścieżka do checkpointu (dla eval)")

    args = parser.parse_args()

    print(f"\nTryb: {args.mode}")
    print(f"Config: {args.config}")

    if args.mode == "train" or args.mode == "full":
        print("\n" + "=" * 60)
        print("TRENING")
        print("=" * 60)
        sys.argv = ["train.py", "--config", args.config]
        train_main()

    if args.mode == "precompute" or args.mode == "full":
        print("\n" + "=" * 60)
        print("PRECOMPUTE REAL FEATURES")
        print("=" * 60)
        precompute_config = "/kaggle/input/e002-01-gan-cluster-metrics-lib/src/configs/e11_precompute_cifar10.yml"
        sys.argv = ["precompute_real.py", "--config", precompute_config]
        precompute_main()

    if args.mode == "eval":
        if args.ckpt is None:
            print("BŁĄD: Tryb eval wymaga parametru --ckpt")
            return
        print("\n" + "=" * 60)
        print("EVALUACJA E11")
        print("=" * 60)
        eval_config = "/kaggle/input/e002-01-gan-cluster-metrics-lib/src/configs/e11_eval_cifar10.yml"
        sys.argv = ["eval_e11.py", "--config", eval_config, "--ckpt", args.ckpt]
        eval_e11_main()

    print("\n" + "=" * 60)
    print("GOTOWE!")
    print("=" * 60)

if __name__ == "__main__":
    main()

