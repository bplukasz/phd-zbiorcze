# Przewodnik migracji po uproszczeniach

## Zmiany w API - co może przestać działać?

### ✅ Kompatybilność wsteczna - działa bez zmian:

```python
# Podstawowe użycie
from src.config_loader import get_config
cfg = get_config('preview')

# Z overrides
cfg = get_config('train', overrides={'steps': 50000})

# Uruchomienie treningu
from src.experiment import train
G, losses = train('preview')
```

### ⚠️ Drobne zmiany (łatwe do naprawienia):

#### 1. Nieznane klucze w YAML są teraz cicho ignorowane

**Przed:**
```
Warning: Unknown config key 'typo_key' ignored
```

**Po:**
```
(brak komunikatu)
```

**Rozwiązanie:** Jeśli potrzebujesz walidacji, dodaj manualnie:
```python
cfg = get_config('train')
if not hasattr(cfg, 'my_custom_key'):
    print("Warning: my_custom_key not found")
```

#### 2. Automatyczna detekcja mode collapse została usunięta

**Przed:**
```
⚠️ [UWAGA krok 5000] Nagły skok w loss_G: 7.23 (średnia z 10: 2.15)
⚠️ FID pogorszył się o 25.3 (poprzedni: 45.2)
```

**Po:**
```
(brak automatycznych alertów)
```

**Rozwiązanie:** Monitoruj metryki w WandB lub CSV:
```python
import pandas as pd
df = pd.read_csv('artifacts/metrics.csv')
df.plot(x='step', y='loss_G')
```

#### 3. base.yaml jest teraz minimalny

**Przed:** 48 linii ze wszystkimi parametrami  
**Po:** 5 linii tylko z paths

**Rozwiązanie:** Wartości domyślne są teraz w `RunConfig` dataclass.
Jeśli chcesz zmienić domyślne wartości:
- Dla jednego profilu: modyfikuj plik profilu (np. `train.yaml`)
- Globalnie: modyfikuj wartości w `RunConfig` w `config_loader.py`

### 🚫 Breaking changes - nie ma!

Wszystkie publiczne API pozostają bez zmian. Kod działający przed uproszczeniem będzie działał dalej.

## Zalecane zmiany w kodzie użytkownika

### 1. Zamiast sprawdzać `hasattr()`, używaj bezpośredniego dostępu

**Przed:**
```python
if hasattr(cfg, 'use_r1_penalty') and cfg.use_r1_penalty:
    r1_lambda = getattr(cfg, 'r1_lambda', 10.0)
```

**Po:**
```python
if cfg.use_r1_penalty:  # Zawsze istnieje w dataclass
    penalty = cfg.r1_lambda * compute_r1(...)
```

### 2. Bezpośrednie tworzenie ConfigLoader

**Przed:**
```python
from src.config_loader import ConfigLoader
loader = ConfigLoader()
cfg = loader.get_config('train')
loader.save_config(cfg, 'output.yaml')
```

**Po:**
```python
from src.config_loader import get_config, ConfigLoader
cfg = get_config('train')
ConfigLoader().save_config(cfg, 'output.yaml')
```

### 3. Uproszczone profle YAML

**Przed (base.yaml):**
```yaml
steps: 30000
batch_size: 64
# ... 40+ linii
```

**Po (train.yaml):**
```yaml
name: "train"
steps: 100000
use_r1_penalty: true  # Tylko overrides
```

## Testowanie po migracji

1. **Test konfiguracji:**
```bash
cd dataset/src
python -c "from config_loader import get_config; print(get_config('preview'))"
```

2. **Smoke test treningu:**
```python
from src.experiment import train
G, losses = train('smoke')  # Krótki test
```

3. **Sprawdź czy WandB działa:**
```python
cfg = get_config('preview', overrides={'use_wandb': True, 'live': True})
```

## Co dalej?

- ✅ Kod jest prostszy i łatwiejszy w utrzymaniu
- ✅ Mniej duplikacji = mniej bugów
- ✅ Logika biznesowa nie uległa zmianie
- ✅ Performance bez zmian (lub lepszy)

Jeśli masz problemy, sprawdź:
1. `UPROSZCZENIA.md` - szczegółowy opis zmian
2. GitHub issues (jeśli projekt publiczny)
3. Commit history - możesz porównać przed/po
