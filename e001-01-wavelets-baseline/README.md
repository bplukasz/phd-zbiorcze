# e001-01-wavelets-baseline

ResNet GAN baseline z hinge loss, SpectralNorm, EMA, DiffAugment na CelebA 128x128.

## 🔧 WAŻNE - Aktualizacja po pierwszym runie (2026-01-18)

**Problem:** FID wzrósł (419→425), mode collapse na kroku 224, brak feedbacku o postępie

**Status:** ✅ NAPRAWIONO

**Kluczowe zmiany:**
- ✅ lr_D zmniejszone: 0.0002 → 0.0001
- ✅ R1 gradient penalty dodany (stabilizacja)
- ✅ Progress tracking dla długich operacji
- ✅ Real-time monitoring FID i mode collapse
- ✅ Częstsza ewaluacja (100 zamiast 250 kroków)

**📖 Przeczytaj:**
- **`QUICK_FIX.md`** - szybki start z poprawkami
- **`ANALIZA_WYNIKOW.md`** - szczegółowa analiza problemu
- **`TO_NAPRAWIENIE.md`** - pełna dokumentacja zmian

---

## Opis eksperymentu

- **Dataset**: CelebA 128×128 (faces)
- **Architektura**: ResNet GAN (5 bloków residual w G i D)
- **Loss**: Hinge loss
- **Regularizacja**: 
  - SpectralNorm w Discriminatorze
  - DiffAugment (color, translation, cutout)
  - **R1 Gradient Penalty** (NOWE - włączone w train profile)
- **EMA**: decay 0.999 dla stabilizacji generatora
- **Optimizer**: Adam (lr_G=2e-4, lr_D=1e-4, betas=(0.0, 0.99))

## System konfiguracji

Eksperyment używa **hierarchicznego systemu konfiguracji opartego na plikach YAML**:

```
dataset/src/configs/
├── base.yaml       # Wspólne ustawienia
├── preview.yaml    # Profil: podgląd (200 kroków)
├── smoke.yaml      # Profil: smoke test (500 kroków)
└── train.yaml      # Profil: pełny trening (30k kroków)
```

### Użycie

```python
from src import get_config, train

# Załaduj konfigurację
cfg = get_config("train")

# Z nadpisaniami
cfg = get_config("train", overrides={"steps": 50000})

# Uruchom trening
model, losses = train("train")
```

Zobacz szczegóły w [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md).

## Profile

### preview
- **200 kroków**, batch 16
- Live display w notebooku
- Bez W&B
- Do szybkiego testowania

### smoke
- **500 kroków**, batch 32
- Z W&B
- Do weryfikacji pipeline'u

### train
- **30k kroków**, batch 64
- Pełne metryki (FID/KID)
- Produkcyjny trening

## Uruchomienie

### Notebook (Kaggle)
```python
from src import train
model, losses = train("preview")  # lub "smoke", "train"
```

### Script (Kaggle)
```bash
python run.py --profile train
python run.py --profile smoke --steps 1000 --no-wandb
```

## Struktura

```
e001-01-wavelets-baseline/
├── dataset/
│   └── src/
│       ├── config_loader.py      # System konfiguracji
│       ├── experiment.py         # Główny kod
│       └── configs/              # Pliki konfiguracyjne
│           ├── base.yaml
│           ├── preview.yaml
│           ├── smoke.yaml
│           └── train.yaml
├── kernels/
│   ├── notebook/
│   │   └── runner.ipynb         # Notebook runner
│   └── script/
│       └── run.py               # Script runner
├── CONFIG_SYSTEM.md             # Dokumentacja systemu konfiguracji
├── requirements.txt
└── test_config.py               # Testy systemu konfiguracji
```

## Artefakty

Trening zapisuje:
- `config_used.yaml` - użyta konfiguracja (reproducibility)
- `logs.csv` - metryki treningowe
- `grids/*.png` - sample gridy
- `checkpoints/*.pt` - checkpointy modelu
- `samples/` - wygenerowane próbki do FID/KID

## Metryki

- Co iterację: `loss_D`, `loss_G`, `grad_norm_D/G`, `sec/iter`, `VRAM peak`
- Co 1k: sample grid
- Co 5k: checkpoint
- Co 10k: FID/KID

## Testy

```bash
python test_config.py
```

Testuje:
- Ładowanie base config
- Wszystkie profile
- Nadpisywanie konfiguracji
- Zapisywanie/wczytywanie
- Hierarchiczne łączenie

## Przenoszenie do szablonu

System konfiguracji jest gotowy do użycia w `templates/experiment/`:

1. Pliki są już skopiowane do `templates/experiment/dataset/src/`
2. Dostosuj `base.yaml` do swojego eksperymentu
3. Dostosuj profile w plikach `.yaml`
4. Użyj `get_config()` w swoim kodzie

Zobacz [templates/experiment/CONFIG_README.md](../templates/experiment/CONFIG_README.md) dla szczegółów.

## Dependencies

```
torch
torchvision
torch-fidelity
wandb
pyyaml
matplotlib
```

## 📚 Dokumentacja Poprawek v1.1.0

### Szybki start:
- **[INDEX.md](INDEX.md)** - przewodnik po całej dokumentacji
- **[QUICK_FIX.md](QUICK_FIX.md)** - 2-minutowe podsumowanie (START TUTAJ!)
- **[CHANGELOG.md](CHANGELOG.md)** - lista zmian wersja po wersji

### Szczegółowa analiza:
- **[ANALIZA_WYNIKOW.md](ANALIZA_WYNIKOW.md)** - techniczne wyjaśnienie problemu (15 min)
- **[VISUALIZATION.md](VISUALIZATION.md)** - wykresy przed/po (5 min)
- **[TO_NAPRAWIENIE.md](TO_NAPRAWIENIE.md)** - pełna dokumentacja zmian (20 min)

### Operacyjne:
- **[CHECKLIST.md](CHECKLIST.md)** - procedura przed i podczas treningu
- **[compare_configs.py](compare_configs.py)** - narzędzie do porównania parametrów

## Notatki

- System konfiguracji automatycznie zapisuje użytą konfigurację dla reproducibility
- Profile dziedziczą z `base.yaml` i nadpisują tylko różnice
- CLI overrides mają najwyższy priorytet
- **v1.1.0:** Dodano R1 penalty, progress tracking, FID monitoring
- **Target FID:** < 50 po 30k kroków (smoke test: < 300 po 500 kroków)
