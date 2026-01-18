# Kaggle Experiments Generator

Generator szablonów eksperymentów dla Kaggle z systemem opartym na plikach szablonów i placeholderach.

## Nowe: System konfiguracji YAML

Eksperymenty używają **hierarchicznego systemu konfiguracji** opartego na plikach YAML:

- **base.yaml** - wspólne ustawienia dla wszystkich profili
- **{profile}.yaml** - specyficzne ustawienia dla profilu (preview, smoke, train)
- **CLI overrides** - nadpisania z argumentów wiersza poleceń

```python
from src import get_config, train

# Załaduj konfigurację z profilu
cfg = get_config("train")

# Z nadpisaniami
cfg = get_config("train", overrides={"steps": 50000})

# Uruchom trening
model, losses = train("train")
```

Zobacz szczegóły w `e001-01-wavelets-baseline/CONFIG_SYSTEM.md` i `templates/experiment/CONFIG_README.md`.

## Struktura projektu

```
.
├── new_experiment.py          # Główny skrypt generujący eksperymenty
├── templates/                 # Szablony z placeholderami
│   ├── README.md             # Dokumentacja szablonów
│   ├── experiment/           # Szablon eksperymentu
│   │   ├── __init__.py
│   │   ├── push.sh
│   │   ├── dataset/
│   │   │   ├── dataset-metadata.json
│   │   │   └── src/
│   │   │       ├── __init__.py
│   │   │       └── experiment.py
│   │   └── kernels/
│   │       ├── notebook/
│   │       │   ├── kernel-metadata.json
│   │       │   └── runner.ipynb
│   │       └── script/
│   │           ├── kernel-metadata.json
│   │           └── run.py
│   └── shared/               # Szablon biblioteki wspólnej
│       ├── __init__.py
│       ├── dataset-metadata.json
│       ├── push_shared.sh
│       └── utils/
│           ├── __init__.py
│           ├── checkpoints.py
│           ├── logging.py
│           └── visualization.py
├── shared/                   # Wygenerowana biblioteka wspólna
└── eXXX-YY-nazwa/           # Wygenerowane eksperymenty
```

## Użycie

### Tworzenie nowego eksperymentu

```bash
# Standardowy eksperyment ze shared library
python new_experiment.py E001-01 wavelets-base

# Eksperyment bez shared library
python new_experiment.py E001-02 standalone-test --no-shared
```

### Tworzenie/aktualizacja shared library

```bash
python new_experiment.py --shared
```

### Po utworzeniu eksperymentu

```bash
cd e001-01-wavelets-base

# 1. Edytuj kod eksperymentu
vim dataset/src/experiment.py

# 2. Utwórz dataset na Kaggle (tylko za pierwszym razem)
kaggle datasets create -p dataset/

# 3. Wrzuć zmiany na Kaggle
./push.sh
```

## System szablonów

Wszystkie pliki w folderze `templates/` są automatycznie kopiowane i przetwarzane przez `new_experiment.py`. Placeholdery w formacie `{{NAZWA}}` są zastępowane odpowiednimi wartościami.

### Dostępne placeholdery

#### Dla experiment/

- `{{KAGGLE_USERNAME}}` - nazwa użytkownika Kaggle (domyślnie: "bplukasz")
- `{{FULL_NAME}}` - pełna nazwa eksperymentu (np. "e001-01-wavelets-base")
- `{{KERNEL_SOURCES}}` - JSON lista źródeł danych dla kerneli
- `{{SHARED_CODE}}` - kod do importu shared library (dla notebooka)
- `{{SHARED_IMPORT}}` - kod do importu shared library (dla skryptu)

#### Dla shared/

- `{{KAGGLE_USERNAME}}` - nazwa użytkownika Kaggle

## Modyfikacja szablonów

Aby zmienić domyślną strukturę eksperymentów:

1. Edytuj pliki w `templates/experiment/` lub `templates/shared/`
2. Użyj placeholderów w formacie `{{NAZWA_PLACEHOLDERA}}`
3. Przy następnym użyciu `new_experiment.py` nowe eksperymenty będą używały zaktualizowanych szablonów

**Uwagi:**
- Pliki `.ipynb` są parsowane jako JSON i zapisywane z formatowaniem
- Pliki `push.sh` i `push_shared.sh` automatycznie otrzymują bit wykonalności
- Wszystkie pliki są przetwarzane jako UTF-8

## Konfiguracja

Edytuj zmienną `KAGGLE_USERNAME` w `new_experiment.py`:

```python
KAGGLE_USERNAME = "twoja-nazwa-uzytkownika"
```

## Przykładowy workflow

```bash
# 1. Utwórz nowy eksperyment
python new_experiment.py E001-01 gan-mnist

# 2. Implementuj logikę
cd e001-01-gan-mnist
vim dataset/src/experiment.py

# 3. Utwórz dataset (pierwsza publikacja)
kaggle datasets create -p dataset/

# 4. Wrzuć na Kaggle
./push.sh

# 5. Obserwuj wyniki w notebooku na Kaggle
# Kernel będzie automatycznie uruchomiony
```

## Struktura eksperymentu

Każdy wygenerowany eksperyment zawiera:

- **dataset/** - kod eksperymentu pakowany jako Kaggle Dataset
  - **src/experiment.py** - główna logika eksperymentu
  - Funkcja `train(profile)` z profilami "preview" i "train"
  
- **kernels/notebook/** - interaktywny notebook do szybkich testów
  - Profil "preview" - krótki trening z live wykresami
  
- **kernels/script/** - skrypt do długiego treningu
  - Profil "train" - pełny trening
  
- **push.sh** - skrypt automatyzujący deploy na Kaggle

## Biblioteka wspólna (shared/)

Folder `shared/` zawiera utilities współdzielone między eksperymentami:

- **utils/logging.py** - konfiguracja logowania
- **utils/visualization.py** - wykresy i live rendering
- **utils/checkpoints.py** - zapis/odczyt modeli

Eksperymenty automatycznie importują shared library, chyba że utworzone z flagą `--no-shared`.

