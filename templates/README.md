# Szablony eksperymentów Kaggle

Ten folder zawiera szablony używane przez skrypt `new_experiment.py` do generowania nowych eksperymentów.

## Struktura

### `experiment/` - Szablon eksperymentu
Struktura pojedynczego eksperymentu z kodem, kernelami notebook i script.

**Placeholdery:**
- `{{KAGGLE_USERNAME}}` - nazwa użytkownika Kaggle (np. "bplukasz")
- `{{FULL_NAME}}` - pełna nazwa eksperymentu (np. "e001-01-wavelets-base")
- `{{KERNEL_SOURCES}}` - JSON lista źródeł danych dla kerneli
- `{{SHARED_CODE}}` - kod do importu shared library (dla notebooka)
- `{{SHARED_IMPORT}}` - kod do importu shared library (dla skryptu)

### `shared/` - Szablon wspólnej biblioteki
Zawiera utilities współdzielone między wszystkimi eksperymentami.

**Placeholdery:**
- `{{KAGGLE_USERNAME}}` - nazwa użytkownika Kaggle

## Użycie

Szablony są automatycznie kopiowane i przetwarzane przez `new_experiment.py`:

```bash
# Utwórz nowy eksperyment
python new_experiment.py E001-01 wavelets-base

# Utwórz eksperyment bez shared
python new_experiment.py E001-02 test --no-shared

# Utwórz tylko folder shared
python new_experiment.py --shared
```

## Modyfikacja szablonów

Aby zmienić domyślną strukturę eksperymentów:

1. Edytuj pliki w `templates/experiment/` lub `templates/shared/`
2. Użyj placeholderów w formacie `{{NAZWA_PLACEHOLDERA}}`
3. Placeholdery zostaną automatycznie zastąpione podczas tworzenia nowego eksperymentu

## Uwagi

- Wszystkie pliki są kopiowane i przetwarzane jako tekst UTF-8
- Pliki `.ipynb` są dodatkowo parsowane jako JSON i zapisywane z formatowaniem
- Pliki wskazane w `make_executable` otrzymują bit wykonalności (np. `push.sh`)

