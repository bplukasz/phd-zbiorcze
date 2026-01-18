#!/usr/bin/env python3
"""
Narzędzie do naprawy notebooków Jupyter przed pushem na Kaggle.
Dodaje wymagane metadane kernelspec i language_info.
"""

import json
import sys
from pathlib import Path


def fix_notebook(notebook_path: Path) -> bool:
    """
    Naprawia notebook dodając wymagane metadane.

    Returns:
        True jeśli notebook został naprawiony, False jeśli był już poprawny
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Błąd parsowania JSON: {e}")
        return False

    fixed = False

    # Upewnij się, że metadata istnieje
    if 'metadata' not in nb:
        nb['metadata'] = {}
        fixed = True

    # Napraw kernelspec
    if 'kernelspec' not in nb['metadata']:
        nb['metadata']['kernelspec'] = {}
        fixed = True

    kernelspec = nb['metadata']['kernelspec']
    required_kernelspec = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }

    for key, value in required_kernelspec.items():
        if key not in kernelspec or not kernelspec[key]:
            kernelspec[key] = value
            fixed = True

    # Napraw language_info
    if 'language_info' not in nb['metadata']:
        nb['metadata']['language_info'] = {}
        fixed = True

    language_info = nb['metadata']['language_info']
    required_language_info = {
        'codemirror_mode': {'name': 'ipython', 'version': 3},
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.10.0'
    }

    for key, value in required_language_info.items():
        if key not in language_info:
            language_info[key] = value
            fixed = True

    # Dodaj nbformat jeśli brakuje
    if 'nbformat' not in nb:
        nb['nbformat'] = 4
        fixed = True
    if 'nbformat_minor' not in nb:
        nb['nbformat_minor'] = 5
        fixed = True

    # Zapisz naprawiony notebook
    if fixed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✅ Naprawiono: {notebook_path}")
    else:
        print(f"✓  OK: {notebook_path}")

    return fixed


def main():
    if len(sys.argv) < 2:
        print("Użycie: fix_notebook.py <notebook.ipynb> [<notebook2.ipynb> ...]")
        sys.exit(1)

    fixed_count = 0
    for notebook_path in sys.argv[1:]:
        path = Path(notebook_path)
        if not path.exists():
            print(f"❌ Nie znaleziono: {path}")
            continue

        if fix_notebook(path):
            fixed_count += 1

    print(f"\n📊 Naprawiono {fixed_count} notebook(ów)")


if __name__ == "__main__":
    main()

