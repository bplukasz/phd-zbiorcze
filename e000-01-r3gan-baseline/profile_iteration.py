#!/usr/bin/env python3
"""
Script do demonstracji profilowania bottlenecków iteracji treningowej.

Usage:
    python profile_iteration.py --profile fast --steps 10

To będzie wypisać szczegółowe podsumowanie każdej sekcji iteracji.
"""

import sys
import os

# Dodaj src do ścieżki
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from profiler import get_global_profiler, reset_global_profiler


def main():
    """Pokazuje co się dzieje w jednej iteracji."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PROFILER BOTTLENECKÓW R3GAN                              ║
║                                                                              ║
║ Skrypt automatycznie profiluje każdą sekcję iteracji treningowej:            ║
║                                                                              ║
║ 1. dataloader.fetch        - pobieranie batcha z DataLoadera                ║
║ 2. batch.parse             - parsowanie i przygotowanie batcha              ║
║ 3. move_images_to_device   - transfer danych na GPU                         ║
║ 4. prepare_condition       - przygotowanie warunków (labels)                ║
║ 5. trainer.discriminator_step:                                              ║
║    - d_step.zero_grad      - zerowanie gradientów D                         ║
║    - d_step.noise_generation - generacja szumu                              ║
║    - d_step.loss_computation - forward pass D i loss                        ║
║    - d_step.backward       - backward pass D                                ║
║    - d_step.grad_clip      - clipping gradientów                            ║
║    - d_step.optimizer_step - update wag D                                   ║
║ 6. trainer.generator_step:                                                  ║
║    - g_step.zero_grad      - zerowanie gradientów G                         ║
║    - g_step.noise_generation - generacja szumu                              ║
║    - g_step.forward_and_discriminate - forward G + D discriminate           ║
║    - g_step.backward       - backward pass G                                ║
║    - g_step.grad_clip      - clipping gradientów                            ║
║    - g_step.optimizer_step - update wag G                                   ║
║ 7. update_ema              - update EMA kopii G                             ║
║ 8. metrics computation     - obliczanie metryk (FID, KID, etc.) gdy aktywne  ║
║                                                                              ║
║ INTERPRETACJA WYNIKÓW:                                                       ║
║ - Szukaj operacji z najwyższym %                                            ║
║ - Typowe wąskie gardła:                                                     ║
║   * d_step.loss_computation / g_step.forward_and_discriminate (GPU compute) ║
║   * dataloader.fetch (I/O bound)                                            ║
║   * metrics computation (ewaluacja na dużym zbiorze)                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    print("\nKRY MASZYNY I UŻYTE OPCJE:")
    print("=" * 80)
    
    # Importuj konfigurację i experiment
    import argparse
    from experiment import run_experiment
    from config_loader import get_config
    
    # Parsuj argumenty jak w run.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke", help="Profile name")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--override", nargs="+", default=[], help="Config overrides")
    parser.add_argument("--steps", type=int, default=None, help="Override steps for quick test")
    args = parser.parse_args()
    
    # Jeśli podano --steps, to override steps dla szybkiego testowania
    if args.steps:
        args.override.extend([f"steps={args.steps}"])
    
    print(f"Profile: {args.profile}")
    print(f"Data dir: {args.data_dir}")
    print(f"Config overrides: {args.override}")
    
    # Reset profiler
    reset_global_profiler()
    profiler = get_global_profiler()
    
    print("\nURUCHAMIAM TRENING Z PROFILOWANIEM...")
    print("=" * 80)
    
    try:
        # Uruchom experiment - będzie automatycznie profilować
        run_experiment(
            profile_name=args.profile,
            data_dir=args.data_dir,
            config_overrides=args.override,
        )
    except Exception as e:
        print(f"\nBŁĘD PODCZAS TRENINGU: {e}")
        import traceback
        traceback.print_exc()
    
    # Wydrukuj podsumowanie (też wydrukuje się automatycznie w pętli)
    print("\n\nPODSUMOWANIE PROFILOWANIA:")
    print(profiler.get_summary())


if __name__ == "__main__":
    main()

