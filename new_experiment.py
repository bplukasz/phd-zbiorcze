#!/usr/bin/env python3
"""
Generator szablonów eksperymentów Kaggle.

Użycie:
    python new_experiment.py E001-01 wavelets-base
    python new_experiment.py E001-02 wavelets-advanced --no-shared
    python new_experiment.py --shared  # tylko folder shared/

Tworzy strukturę:
    e001-01-wavelets-base/
    ├── push.sh
    ├── dataset/
    │   ├── __init__.py
    │   ├── dataset-metadata.json
    │   └── src/
    │       ├── __init__.py
    │       └── experiment.py
    └── kernels/
        ├── notebook/
        │   ├── kernel-metadata.json
        │   └── runner.ipynb
        └── script/
            ├── kernel-metadata.json
            └── run.py
"""

import argparse
import json
import os
import stat
import shutil
from pathlib import Path
from typing import Dict, Any

# === Konfiguracja ===
KAGGLE_USERNAME = "bplukasz"


def replace_placeholders(content: str, placeholders: Dict[str, str]) -> str:
    """
    Zamienia placeholdery w treści pliku.

    Args:
        content: Treść pliku
        placeholders: Słownik z placeholderami i wartościami

    Returns:
        Treść z zamienionymi placeholderami
    """
    result = content
    for key, value in placeholders.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


def copy_template_dir(
    src_dir: Path,
    dst_dir: Path,
    placeholders: Dict[str, str],
    make_executable: list = None
) -> None:
    """
    Kopiuje katalog z szablonami, podmieniając placeholdery.

    Args:
        src_dir: Katalog źródłowy z szablonami
        dst_dir: Katalog docelowy
        placeholders: Słownik z placeholderami
        make_executable: Lista plików do uczynienia wykonalnymi
    """
    make_executable = make_executable or []

    for item in src_dir.rglob("*"):
        if item.is_file():
            # Oblicz względną ścieżkę
            rel_path = item.relative_to(src_dir)
            dst_path = dst_dir / rel_path

            # Utwórz katalog docelowy jeśli nie istnieje
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Wczytaj zawartość
            try:
                with open(item, "r", encoding="utf-8") as f:
                    content = f.read()

                # Zamień placeholdery
                content = replace_placeholders(content, placeholders)

                # Dla plików .ipynb - dodatkowo parsuj i zapisz JSON
                if str(item).endswith('.ipynb'):
                    try:
                        notebook_data = json.loads(content)
                        with open(dst_path, "w", encoding="utf-8") as f:
                            json.dump(notebook_data, f, indent=1, ensure_ascii=False)
                    except json.JSONDecodeError:
                        # Jeśli nie da się sparsować, zapisz surowy tekst
                        with open(dst_path, "w", encoding="utf-8") as f:
                            f.write(content)
                else:
                    # Zapisz do pliku docelowego
                    with open(dst_path, "w", encoding="utf-8") as f:
                        f.write(content)

                # Ustaw bit wykonalności jeśli potrzeba
                if any(str(rel_path).endswith(pattern) for pattern in make_executable):
                    st = os.stat(dst_path)
                    os.chmod(dst_path, st.st_mode | stat.S_IEXEC)

            except UnicodeDecodeError:
                # Dla plików binarnych po prostu skopiuj
                shutil.copy2(item, dst_path)


def create_experiment(exp_id: str, exp_name: str, use_shared: bool = True):
    """
    Tworzy pełną strukturę eksperymentu na podstawie szablonów.

    Args:
        exp_id: Identyfikator eksperymentu, np. "E001-01"
        exp_name: Nazwa eksperymentu, np. "wavelets-base"
        use_shared: Czy dodać zależność od shared dataset
    """
    exp_id = exp_id.lower()
    exp_name = exp_name.lower()
    full_name = f"{exp_id}-{exp_name}"

    root = Path(__file__).parent
    exp_dir = root / full_name
    template_dir = root / "templates" / "experiment"

    if exp_dir.exists():
        print(f"❌ Folder {full_name} już istnieje!")
        return False

    if not template_dir.exists():
        print(f"❌ Folder z szablonami nie istnieje: {template_dir}")
        return False

    print(f"📁 Tworzę eksperyment: {full_name}")

    # === Przygotowanie placeholderów ===
    dataset_id = f"{KAGGLE_USERNAME}/{full_name}-lib"
    kernel_sources = [dataset_id]
    if use_shared:
        kernel_sources.append(f"{KAGGLE_USERNAME}/shared-lib")

    shared_code = ""
    shared_import = ""
    if use_shared:
        shared_code = '''\n\nSHARED_DIR = "/kaggle/input/shared-lib"\nsys.path.insert(0, SHARED_DIR)\nprint("SHARED_DIR exists:", os.path.exists(SHARED_DIR))'''
        shared_import = '''
SHARED_DIR = "/kaggle/input/shared-lib"
sys.path.insert(0, SHARED_DIR)
'''

    placeholders = {
        "KAGGLE_USERNAME": KAGGLE_USERNAME,
        "FULL_NAME": full_name,
        "KERNEL_SOURCES": json.dumps(kernel_sources),
        "SHARED_CODE": shared_code,
        "SHARED_IMPORT": shared_import,
    }

    # === Kopiowanie szablonów ===
    copy_template_dir(
        template_dir,
        exp_dir,
        placeholders,
        make_executable=["push.sh"]
    )


    print(f"✅ Eksperyment {full_name} utworzony!")
    print(f"\n📂 Struktura:")
    print(f"   {full_name}/")
    print(f"   ├── push.sh")
    print(f"   ├── dataset/")
    print(f"   │   ├── dataset-metadata.json")
    print(f"   │   └── src/")
    print(f"   │       ├── __init__.py")
    print(f"   │       └── experiment.py")
    print(f"   └── kernels/")
    print(f"       ├── notebook/")
    print(f"       └── script/")
    print(f"\n🔧 Następne kroki:")
    print(f"   1. Edytuj: {full_name}/dataset/src/experiment.py")
    print(f"   2. Utwórz dataset na Kaggle: kaggle datasets create -p {full_name}/dataset")
    print(f"   3. Push: cd {full_name} && ./push.sh")

    return True


def create_shared():
    """Tworzy folder shared/ na wspólny kod między eksperymentami."""
    root = Path(__file__).parent
    shared_dir = root / "shared"
    template_dir = root / "templates" / "shared"

    if shared_dir.exists():
        print("ℹ️  Folder shared/ już istnieje")
        return

    if not template_dir.exists():
        print(f"❌ Folder z szablonami nie istnieje: {template_dir}")
        return

    print("📁 Tworzę folder shared/ na wspólny kod")

    # === Przygotowanie placeholderów ===
    placeholders = {
        "KAGGLE_USERNAME": KAGGLE_USERNAME,
    }

    # === Kopiowanie szablonów ===
    copy_template_dir(
        template_dir,
        shared_dir,
        placeholders,
        make_executable=["push_shared.sh"]
    )

    print("✅ Folder shared/ utworzony!")
    print(f"\n📂 Struktura:")
    print(f"   shared/")
    print(f"   ├── dataset-metadata.json")
    print(f"   ├── push_shared.sh")
    print(f"   └── utils/")
    print(f"       ├── __init__.py")
    print(f"       ├── logging.py")
    print(f"       ├── visualization.py")
    print(f"       └── checkpoints.py")
    print(f"\n🔧 Aby utworzyć dataset na Kaggle:")
    print(f"   kaggle datasets create -p shared/")


def main():
    parser = argparse.ArgumentParser(
        description="Generator szablonów eksperymentów Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  %(prog)s E001-01 wavelets-base       # Tworzy e001-01-wavelets-base/
  %(prog)s E001-02 wavelets-advanced   # Tworzy e001-02-wavelets-advanced/
  %(prog)s --shared                    # Tworzy tylko folder shared/
  %(prog)s E002-01 gan-test --no-shared  # Bez zależności od shared
        """
    )

    parser.add_argument("exp_id", nargs="?", help="ID eksperymentu, np. E001-01")
    parser.add_argument("exp_name", nargs="?", help="Nazwa eksperymentu, np. wavelets-base")
    parser.add_argument("--shared", action="store_true", help="Utwórz tylko folder shared/")
    parser.add_argument("--no-shared", action="store_true", help="Nie dodawaj zależności od shared")

    args = parser.parse_args()

    if args.shared:
        create_shared()
        return

    if not args.exp_id or not args.exp_name:
        parser.print_help()
        print("\n❌ Podaj exp_id i exp_name, lub użyj --shared")
        return

    # Zawsze upewnij się, że shared/ istnieje (jeśli używamy shared)
    if not args.no_shared:
        create_shared()

    create_experiment(args.exp_id, args.exp_name, use_shared=not args.no_shared)


if __name__ == "__main__":
    main()

