# Dataset downloader (GAN)

Skrypt: `scripts/download_datasets.py`

Domyslny katalog danych:
- `~/edx/datasets`

## Co obsluguje
- `mnist@native` (28x28)
- `cifar10@native` (32x32)
- `cifar100@native` (32x32)
- `celeba@aligned` (178x218)
- `ffhq@256`, `ffhq@1024` (Kaggle CLI)
- `celebahq@256`, `celebahq@512` (Kaggle CLI)

Mozesz podawac sam dataset bez wariantu (np. `ffhq`) - wtedy skrypt wybierze wariant domyslny.

## Uzycie

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --dataset mnist
python scripts/download_datasets.py --dataset ffhq@256
python scripts/download_datasets.py --dataset ffhq@256 --dataset ffhq@1024
python scripts/download_datasets.py --dataset cifar10 --dataset cifar100
python scripts/download_datasets.py --dataset all
python scripts/download_datasets.py --dataset all --all-variants
python scripts/download_datasets.py --dataset ffhq --dry-run
python scripts/download_datasets.py --dataset ffhq@1024 --validate-only
python scripts/download_datasets.py --dataset all --validate-only
python scripts/download_datasets.py --dataset all --skip-validate


## Walidacja po pobraniu
- Domyslnie po pobraniu skrypt waliduje:
  - liczbe obrazow,
  - rozdzielczosc obrazow zgodna z wybranym wariantem.
- Przy niezgodnosci skrypt zwraca blad (np. zly wariant lub niepelne dane).
- `--validate-only` pozwala sprawdzic juz pobrane dane bez ponownego sciagania.
- `--skip-validate` pomija walidacje (niezalecane poza awaryjnymi przypadkami).
Przyklad instalacji Kaggle CLI:

```bash
python -m pip install kaggle
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

## Uwaga o rozmiarach
Rozmiary sa orientacyjne i moga sie roznic zaleznie od zrodla, kompresji i wersji datasetu.


