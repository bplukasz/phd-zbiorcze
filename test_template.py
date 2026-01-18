#!/usr/bin/env python3
"""
Test szablonu konfiguracji - symuluje wygenerowanie nowego eksperymentu
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_template():
    """Test czy szablon działa poprawnie"""
    print("=" * 60)
    print("TEST SZABLONU KONFIGURACJI")
    print("=" * 60)

    # Utwórz tymczasowy katalog
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nKatalog testowy: {tmpdir}")

        # Skopiuj szablon
        template_src = Path("/Users/lukasztymoszuk/Documents/Projekty/phd/zbiorcze/templates/experiment/dataset/src")
        test_src = Path(tmpdir) / "src"
        shutil.copytree(template_src, test_src)

        print(f"✓ Skopiowano szablon")

        # Dodaj do PYTHONPATH
        sys.path.insert(0, str(Path(tmpdir)))

        try:
            # Importuj moduły
            from src import get_config, RunConfig, ConfigLoader
            print(f"✓ Import config_loader pomyślny")

            # Test ładowania profili
            print(f"\nTest profili:")
            for profile in ["preview", "smoke", "train"]:
                cfg = get_config(profile)
                print(f"  {profile}: steps={cfg.steps}, batch={cfg.batch_size}, lr={cfg.lr}")

            print(f"✓ Wszystkie profile działają")

            # Test overrides
            cfg = get_config("train", overrides={"steps": 100, "lr": 0.01})
            assert cfg.steps == 100
            assert cfg.lr == 0.01
            print(f"✓ Overrides działają")

            # Test zapisywania
            loader = ConfigLoader()
            output_path = Path(tmpdir) / "test_config.yaml"
            loader.save_config(cfg, str(output_path))
            assert output_path.exists()
            print(f"✓ Zapisywanie działa")

            print("\n" + "=" * 60)
            print("✓ SZABLON DZIAŁA POPRAWNIE")
            print("=" * 60)

        except Exception as e:
            print("\n" + "=" * 60)
            print("✗ BŁĄD W SZABLONIE")
            print("=" * 60)
            print(f"Błąd: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Usuń z PYTHONPATH
            sys.path.remove(str(Path(tmpdir)))

    return True

if __name__ == "__main__":
    success = test_template()
    sys.exit(0 if success else 1)
