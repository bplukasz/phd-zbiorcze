# System konfiguracji - Szablon eksperymentu

Ten szablon zawiera gotowy system konfiguracji oparty na plikach YAML.

## ✨ Co jest gotowe out-of-the-box

Po wygenerowaniu nowego eksperymentu otrzymujesz:

- **config_loader.py** - moduł do ładowania hierarchicznej konfiguracji
- **configs/base.yaml** - bazowa konfiguracja
- **configs/preview.yaml** - profil preview (200 kroków, live)
- **configs/smoke.yaml** - profil smoke test (500 kroków)
- **configs/train.yaml** - profil pełnego treningu (5k kroków)
- **experiment.py** - szablon z integracją konfiguracji
- **run.py** - script runner z obsługą CLI
- **requirements.txt** - zależności (torch, pyyaml, etc.)

## 🚀 Szybki start

### 1. Dostosuj base.yaml do swojego eksperymentu

Otwórz `dataset/src/configs/base.yaml` i dodaj swoje parametry:

```yaml
# === TRAINING ===
steps: 5000
batch_size: 64
lr: 0.001

# === LOGGING ===
log_every: 10
viz_every: 100
ckpt_every: 1000
live: false
use_wandb: true

# === PATHS ===
out_dir: "/kaggle/working/artifacts"
data_dir: "/kaggle/input/your-dataset"

# === CUSTOM ===
# Dodaj swoje parametry:
model_dim: 256
num_layers: 4
dropout: 0.1
```

### 2. Zaktualizuj RunConfig w config_loader.py

Dodaj swoje pola w klasie `RunConfig`:

```python
@dataclass
class RunConfig:
    """Konfiguracja eksperymentu."""
    
    # ...existing fields...
    
    # CUSTOM: Twoje parametry
    model_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
```

### 3. Dostosuj profile (opcjonalnie)

Jeśli potrzebujesz innych wartości dla różnych profili:

```yaml
# preview.yaml
name: "preview"
steps: 100
model_dim: 128  # mniejszy model dla preview
```

### 4. Użyj w kodzie

W `experiment.py` konfiguracja jest już zintegrowana:

```python
def train(profile: str = "preview", overrides: Optional[Dict[str, Any]] = None):
    cfg = get_config(profile, overrides)
    
    # Użyj konfiguracji
    model = MyModel(dim=cfg.model_dim, layers=cfg.num_layers)
    # ...
```

## 📚 Przykłady użycia

### Notebook
```python
from src import get_config, train

# Podgląd konfiguracji
cfg = get_config("preview")
print(f"Steps: {cfg.steps}, Batch: {cfg.batch_size}")

# Uruchom
model, losses = train("preview")
```

### CLI Script
```bash
# Użyj profilu
python run.py --profile train

# Z nadpisaniami
python run.py --profile train --steps 10000 --batch-size 128 --no-wandb
```

### Z kodem
```python
from src import get_config, train

# Z nadpisaniami
cfg = get_config("train", overrides={
    "steps": 10000,
    "lr": 0.0001,
    "model_dim": 512
})

# Lub bezpośrednio w train
model, losses = train("train", overrides={"steps": 10000})
```

## 🎯 Hierarchia konfiguracji

System działa na 3 poziomach (priorytet rosnący):

1. **base.yaml** ← domyślne wartości
2. **{profile}.yaml** ← nadpisuje wybrane wartości
3. **overrides** ← nadpisuje wszystko (CLI lub kod)

### Przykład

**base.yaml:**
```yaml
steps: 5000
batch_size: 64
lr: 0.001
```

**preview.yaml:**
```yaml
steps: 200       # nadpisuje base
batch_size: 16   # nadpisuje base
# lr: 0.001      # dziedziczy z base
```

**Użycie:**
```python
# Profil: steps=200, batch=16, lr=0.001
cfg = get_config("preview")

# Override: steps=100, batch=16, lr=0.001
cfg = get_config("preview", overrides={"steps": 100})
```

## ✅ Profile out-of-the-box

| Profil   | Steps | Batch | Live | W&B | Przeznaczenie |
|----------|-------|-------|------|-----|---------------|
| preview  | 200   | 16    | ✓    | ✗   | Szybki test w notebooku |
| smoke    | 500   | 32    | ✗    | ✓   | Weryfikacja pipeline |
| train    | 5000  | 64    | ✗    | ✓   | Pełny trening |

## 🔧 Dodawanie nowych parametrów

### Krok 1: Dodaj do RunConfig
```python
@dataclass
class RunConfig:
    # ...existing...
    new_param: int = 42
```

### Krok 2: Dodaj do base.yaml
```yaml
# === CUSTOM ===
new_param: 42
```

### Krok 3: Nadpisz w profilach (opcjonalnie)
```yaml
# preview.yaml
new_param: 10
```

### Krok 4: Użyj w kodzie
```python
def train(profile: str = "preview", overrides=None):
    cfg = get_config(profile, overrides)
    print(f"New param: {cfg.new_param}")
```

## 📦 Reproducibility

Przy każdym treningu konfiguracja jest automatycznie zapisywana:

```
{out_dir}/config_used.yaml
```

To pozwala dokładnie odtworzyć warunki eksperymentu.

## 🎨 Zalety tego rozwiązania

✅ **Gotowe od razu** - działa bez dodatkowej konfiguracji
✅ **Separacja od kodu** - łatwa edycja w YAML
✅ **DRY** - base.yaml eliminuje duplikację
✅ **Hierarchiczne** - elastyczny system priorytetów
✅ **Reproducible** - automatyczne zapisywanie konfiguracji
✅ **Skalowalne** - łatwo dodawać nowe parametry i profile
✅ **CLI-friendly** - pełna obsługa argumentów

## 📖 Dokumentacja

Szczegółowa dokumentacja dostępna w eksperymencie `e001-01-wavelets-baseline`:
- `CONFIG_SYSTEM.md` - pełna dokumentacja systemu
- `test_config.py` - testy systemu
- `demo_config.py` - demo użycia

## 💡 Przykład kompletnego workflow

```python
# 1. Wygeneruj nowy eksperyment
# python new_experiment.py E002-01 my-experiment

# 2. Dostosuj configs/base.yaml
# (dodaj swoje parametry)

# 3. Zaktualizuj RunConfig w config_loader.py
# (dodaj pola dla swoich parametrów)

# 4. Zaimplementuj logikę w experiment.py
def train(profile: str = "preview", overrides=None):
    cfg = get_config(profile, overrides)
    
    # Twój kod używający cfg
    model = MyModel(cfg.model_dim)
    for step in range(cfg.steps):
        # training loop
        pass

# 5. Uruchom
# python run.py --profile train
# lub w notebooku: train("preview")
```

System jest gotowy do użycia od razu po wygenerowaniu! 🎉
