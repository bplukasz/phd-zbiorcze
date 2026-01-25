# System konfiguracji eksperymentów

## Przegląd

System konfiguracji oparty na plikach YAML z hierarchicznym ładowaniem:
1. **base.yaml** - wspólne ustawienia dla wszystkich profili
2. **{profile}.yaml** - specyficzne ustawienia dla danego profilu (nadpisują base)
3. **CLI overrides** - nadpisania z argumentów wiersza poleceń (nadpisują wszystko)

## Struktura plików

```
dataset/src/
├── config_loader.py         # Moduł do ładowania konfiguracji
├── configs/
│   ├── base.yaml           # Bazowa konfiguracja
│   ├── preview.yaml        # Profil: podgląd (200 kroków, live mode)
│   ├── smoke.yaml          # Profil: smoke test (500 kroków)
│   └── train.yaml          # Profil: pełny trening (30k kroków)
└── experiment.py           # Główny kod eksperymentu
```

## Użycie

### W kodzie Python

```python
from src import get_config, train

# Załaduj konfigurację dla profilu
cfg = get_config("preview")

# Z nadpisaniami
cfg = get_config("train", overrides={"steps": 50000, "batch_size": 128})

# Uruchom trening
model, losses = train("train")

# Z nadpisaniami
model, losses = train("train", overrides={"steps": 50000})
```

### W notebooku

```python
from src import get_config, train

# Podgląd konfiguracji
cfg = get_config("preview")
print(f"Steps: {cfg.steps}, Batch: {cfg.batch_size}")

# Uruchom trening
model, losses = train("preview")
```

### W skrypcie CLI

```bash
# Użyj profilu
python run.py --profile train

# Z nadpisaniami
python run.py --profile train --steps 50000 --batch-size 128 --no-wandb

# Smoke test
python run.py --profile smoke
```

## Profile

### preview
- **Przeznaczenie**: Szybki podgląd w notebooku
- **Kroki**: 200
- **Batch size**: 16
- **Live mode**: Tak
- **W&B**: Nie
- **Użycie**: Testowanie w środowisku interaktywnym

### smoke
- **Przeznaczenie**: Smoke test - weryfikacja pipeline'u
- **Kroki**: 500
- **Batch size**: 32
- **Live mode**: Nie
- **W&B**: Tak
- **Użycie**: Szybka weryfikacja że kod działa end-to-end

### train
- **Przeznaczenie**: Pełny trening produkcyjny
- **Kroki**: 30000
- **Batch size**: 64
- **Live mode**: Nie
- **W&B**: Tak
- **Użycie**: Produkcyjny trening z pełnymi metrykami

### fast ⚡
- **Przeznaczenie**: Szybkie prototypowanie
- **Dataset**: CIFAR-10 (32x32, auto-download)
- **Kroki**: 10000
- **Batch size**: 128
- **Przyspieszenie**: ~4-8x szybciej niż train
- **Użycie**: Testowanie nowych pomysłów

### fast64 ⚡
- **Przeznaczenie**: Kompromis szybkość/jakość
- **Dataset**: CelebA (64x64)
- **Kroki**: 20000
- **Batch size**: 64
- **Przyspieszenie**: ~2-3x szybciej niż train
- **Użycie**: Prototypowanie z twarzami

## Struktura konfiguracji

### base.yaml
Wspólne ustawienia dla wszystkich profili:

```yaml
# TRAINING
steps: 30000
batch_size: 64
lr_G: 0.0002
lr_D: 0.0002
betas: [0.0, 0.99]

# MODEL
z_dim: 128
img_size: 128
img_channels: 3
g_ch: 64
d_ch: 64

# EMA
ema_decay: 0.999

# AUGMENTATION
diffaug_policy: "color,translation,cutout"

# REGULARIZATION
use_r1_penalty: false
r1_lambda: 10.0
r1_every: 16

# LOGGING
log_every: 1
grid_every: 1000
ckpt_every: 5000
eval_every: 10000
live: false
use_wandb: true

# EVALUATION
eval_samples: 50000
fid_samples: 10000

# DATASET
dataset_name: "celeba"  # celeba, cifar10, cifar100, mnist, fashion_mnist

# PATHS
out_dir: "/kaggle/working/artifacts"
data_dir: "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
```

## Dostępne datasety

| Dataset | Rozmiar | Kanały | Auto-download | Opis |
|---------|---------|--------|---------------|------|
| `celeba` | 128x128* | RGB | Nie | Twarze celebrytów (domyślny) |
| `cifar10` | 32x32* | RGB | Tak | 10 klas obiektów |
| `cifar100` | 32x32* | RGB | Tak | 100 klas obiektów |
| `mnist` | 28x28* | Gray→RGB | Tak | Cyfry 0-9 |
| `fashion_mnist` | 28x28* | Gray→RGB | Tak | Ubrania |

\* Rozmiar natywny, można przeskalować przez `img_size`

Przykład zmiany datasetu:
```yaml
dataset_name: "cifar10"
img_size: 32
```

### Profil (np. preview.yaml)
Nadpisuje tylko te wartości, które się różnią:

```yaml
name: "preview"

# TRAINING
steps: 200
batch_size: 16

# LOGGING
log_every: 10
grid_every: 50
ckpt_every: 100
eval_every: 100
live: true
use_wandb: false

# EVALUATION
fid_samples: 1000
```

## Dodawanie nowego profilu

1. Utwórz nowy plik w `dataset/src/configs/{nazwa}.yaml`
2. Zdefiniuj tylko wartości różniące się od base.yaml
3. Użyj profilu: `train("nazwa")` lub `--profile nazwa`

Przykład (`custom.yaml`):
```yaml
name: "custom"
steps: 10000
batch_size: 48
use_wandb: false
```

## API

### ConfigLoader

Klasa do ładowania konfiguracji z plików YAML.

```python
from src.config_loader import ConfigLoader

loader = ConfigLoader()  # Domyślnie: ./configs
loader = ConfigLoader("/path/to/configs")  # Custom path

cfg = loader.get_config("train")
cfg = loader.get_config("train", overrides={"steps": 50000})

# Zapisz konfigurację
loader.save_config(cfg, "/path/to/output.yaml")
```

### get_config()

Wygodna funkcja do szybkiego ładowania konfiguracji.

```python
from src import get_config

cfg = get_config("train")
cfg = get_config("train", overrides={"steps": 50000})
cfg = get_config("train", config_dir="/custom/path")
```

### RunConfig

Klasa dataclass z definicją konfiguracji.

```python
from src import RunConfig

cfg = RunConfig(name="custom", steps=10000)
cfg.update_from_dict({"batch_size": 128})
config_dict = cfg.to_dict()
```

## Reproducibility

Przy każdym uruchomieniu treningu, aktualna konfiguracja jest zapisywana do:
```
{out_dir}/config_used.yaml
```

To pozwala na:
- Dokładne odtworzenie warunków eksperymentu
- Śledzenie zmian w konfiguracji między eksperymentami
- Łatwe porównanie różnic między eksperymentami

## Przenoszenie do szablonu

Cały system konfiguracji jest gotowy do użycia w szablonie `templates/experiment/`:

```bash
templates/experiment/
├── dataset/src/
│   ├── config_loader.py
│   └── configs/
│       ├── base.yaml
│       ├── preview.yaml
│       ├── smoke.yaml
│       └── train.yaml
```

Przy tworzeniu nowego eksperymentu:
1. Skopiuj template
2. Dostosuj `base.yaml` do swojego eksperymentu
3. Dostosuj profile w plikach `.yaml`
4. Użyj `get_config()` w swoim kodzie

## Najlepsze praktyki

1. **Minimalizm w profilach**: W plikach profili nadpisuj tylko to, co się różni od base
2. **Dokumentuj zmiany**: Dodaj komentarze w YAML wyjaśniające nietypowe wartości
3. **Spójne nazewnictwo**: Używaj spójnych nazw profili (preview, smoke, train)
4. **Zapisuj konfigurację**: System automatycznie zapisuje użytą konfigurację
5. **Wersjonuj YAML**: Trzymaj pliki konfiguracyjne w git

## Rozszerzanie

### Dodawanie nowych pól

1. Dodaj pole do `RunConfig` w `config_loader.py`:
```python
@dataclass
class RunConfig:
    # ...existing fields...
    new_field: int = 42
```

2. Dodaj wartość w `base.yaml`:
```yaml
new_field: 42
```

3. Nadpisz w profilach jeśli potrzeba:
```yaml
# preview.yaml
new_field: 10
```

### Walidacja konfiguracji

Możesz dodać walidację w `RunConfig.update_from_dict()`:

```python
def update_from_dict(self, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if hasattr(self, key):
            # Custom validation
            if key == "steps" and value <= 0:
                raise ValueError(f"steps must be positive, got {value}")
            setattr(self, key, value)
```

## Przykład kompletnego workflow

```python
# 1. Załaduj i wyświetl konfigurację
from src import get_config

cfg = get_config("train")
print(f"Training for {cfg.steps} steps with batch size {cfg.batch_size}")

# 2. Uruchom z nadpisaniami
from src import train

model, losses = train("train", overrides={
    "steps": 50000,
    "lr_G": 1e-4,
    "use_wandb": False
})

# 3. Sprawdź zapisaną konfigurację
import yaml
with open("/kaggle/working/artifacts/config_used.yaml") as f:
    used_config = yaml.safe_load(f)
    print(used_config)
```
