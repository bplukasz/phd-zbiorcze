# ✅ NAPRAWIONO: Obsługa W&B na Kaggle

## Problem
```
wandb.errors.errors.UsageError: api_key not configured (no-tty). 
call wandb.login(key=[your_api_key])
```

Kaggle uruchamia skrypty w środowisku no-tty (bez terminala interaktywnego), więc W&B nie może poprosić o API key.

## Rozwiązanie

### 1. Bezpieczna inicjalizacja W&B
Dodano try-except przy `wandb.init()`:

```python
# W experiment.py:
if cfg.use_wandb and _HAS_WANDB:
    try:
        wandb.init(
            project="project-name",
            name=cfg.name,
            config=cfg.to_dict(),
        )
        print("W&B logging enabled")
    except Exception as e:
        print(f"Warning: Could not initialize W&B: {e}")
        print("Continuing without W&B logging...")
        cfg.use_wandb = False
```

### 2. Domyślnie wyłączony W&B w run.py (e001-01)
```python
PROFILE = "smoke"

OVERRIDES = {
    'use_wandb': False,  # Wyłącz W&B jeśli brak API key
}
```

### 3. Jasne komentarze w run.py
```python
# UWAGA: Na Kaggle W&B wymaga API key. Jeśli go nie masz, wyłącz:
OVERRIDES = {
    'use_wandb': False,  # Wyłącz W&B jeśli brak API key
}
```

## Co się zmienia

### Przed:
- ❌ Skrypt crashuje przy braku W&B API key
- ❌ Trzeba ręcznie łapać wyjątek
- ❌ Brak jasnej informacji co zrobić

### Po:
- ✅ Skrypt kontynuuje bez W&B
- ✅ Automatyczne wykrycie i obsługa błędu
- ✅ Jasny komunikat w konsoli
- ✅ Domyślnie wyłączony w smoke test
- ✅ Komentarze w run.py wyjaśniają jak używać

## Użycie

### Opcja 1: Bez W&B (domyślne dla smoke)
```python
# W run.py:
PROFILE = "smoke"
OVERRIDES = {
    'use_wandb': False,
}
```

### Opcja 2: Z W&B (wymagany API key)
```python
# W run.py:
PROFILE = "train"
OVERRIDES = {}  # use_wandb=True z profilu

# Dodaj w Kaggle Secrets lub na początku skryptu:
# import wandb
# wandb.login(key="your_api_key")
```

### Opcja 3: Automatyczne wykrycie
```python
# System automatycznie:
# 1. Próbuje zainicjalizować W&B
# 2. Jeśli fail -> wyłącza i kontynuuje
# 3. Loguje ostrzeżenie
# 4. Trening działa normalnie
```

## Pliki zaktualizowane

- ✅ `e001-01-wavelets-baseline/dataset/src/experiment.py` - try-except dla wandb.init()
- ✅ `e001-01-wavelets-baseline/kernels/script/run.py` - use_wandb: False domyślnie
- ✅ `templates/experiment/dataset/src/experiment.py` - try-except + obsługa W&B
- ✅ `templates/experiment/kernels/script/run.py` - komentarz o W&B
- ✅ `templates/experiment/CONFIG_README.md` - sekcja o W&B
- ✅ `templates/experiment/requirements.txt` - wandb dodany

## Teraz powinno działać!

Uruchom ponownie na Kaggle - skrypt:
1. ✅ Załaduje konfigurację
2. ✅ Spróbuje zainicjalizować W&B
3. ✅ Jeśli fail → wyłączy i kontynuuje
4. ✅ Trening przebiegnie normalnie
5. ✅ Zapisze wyniki bez W&B

**Problem rozwiązany! 🎉**
