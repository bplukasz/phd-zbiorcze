# 🔧 Poprawki po pierwszym runie - Quick Start

## 🚨 TL;DR - Co było nie tak?

**FID wzrósł z 419 → 425** (powinien maleć!)  
**Mode collapse** na kroku 224-227  
**11 minut ciszy** podczas ewaluacji (wyglądało jak zawieszenie)

## ✅ Co naprawiono?

1. ✅ **Progress tracking** - widzisz co się dzieje
2. ✅ **FID monitoring** - ostrzeżenia gdy rośnie
3. ✅ **Mode collapse detection** - wykrywa w czasie rzeczywistym
4. ✅ **lr_D: 0.0001** (było 0.0002) - mniej dominacji D
5. ✅ **R1 gradient penalty** - stabilizacja treningu
6. ✅ **Częstsza ewaluacja** (co 100 zamiast 250)

## 🚀 Jak uruchomić poprawiony test?

```bash
cd e001-01-wavelets-baseline
python -m dataset.src.experiment --profile smoke
```

**Teraz zobaczysz:**
```
[000100/500] ...
  -> Rozpoczynam ewaluację na kroku 100...
     (1/2) Generowanie 2048 próbek testowych...
    -> 0/2048 próbek (0.0%)
    -> 640/2048 próbek (31.2%)
    -> 1280/2048 próbek (62.5%)
    ✓ Wygenerowano wszystkie 2048 próbek
     (2/2) Obliczanie metryk FID/KID...
    (To może potrwać 5-15 minut...)
    ✓ Metryki obliczone!
  -> FID: 280.12, KID: 380.22
  ✓ Nowy najlepszy FID! (poprawa: 40.33)
```

## 📊 Oczekiwane wyniki

### Smoke test (500 kroków):
- **Cel:** FID < 300 (było 425)
- **Oznaka sukcesu:** FID maleje przy każdej ewaluacji

### Pełny trening (30k kroków):
```bash
python -m dataset.src.experiment --profile train
```
- **10k kroków:** FID ~50-100
- **30k kroków:** FID ~20-50

## 📁 Przeczytaj więcej

- **`ANALIZA_WYNIKOW.md`** - szczegółowa analiza problemu i benchmark
- **`TO_NAPRAWIENIE.md`** - pełna dokumentacja wszystkich zmian
- **`dataset/src/configs/smoke-fixed.yaml`** - alternatywna konfiguracja z komentarzami

## 🎯 Kluczowe zmiany w kodzie

### experiment.py:
```python
# NOWE: R1 gradient penalty
def r1_penalty(D, real_imgs):
    """Stabilizuje Discriminator, zapobiega mode collapse"""
    ...

# NOWE: Progress tracking
print(f"    -> {idx}/{n_samples} próbek ({progress:.1f}%)")

# NOWE: FID monitoring
if current_fid < best_fid:
    print(f"  ✓ Nowy najlepszy FID!")
else:
    print(f"  ⚠️  FID pogorszył się o {degradation:.2f}")

# NOWE: Mode collapse detection
if abs(loss_G - recent_avg) > 5.0:
    print(f"  ⚠️  Nagły skok w loss_G - możliwy mode collapse!")
```

### smoke.yaml:
```yaml
lr_D: 0.0001      # było 0.0002
eval_every: 100   # było 250
```

### train.yaml:
```yaml
lr_D: 0.0001
use_r1_penalty: true
r1_lambda: 10.0
```

## 🔍 Jak monitorować trening?

### Dobre znaki ✅:
- FID systematycznie maleje
- loss_D stabilne ~2.0
- Różnorodność w obrazach

### Złe znaki ❌:
- FID rośnie 2x z rzędu → **STOP**, zmień parametry
- loss_G skacze > 5.0 → **Mode collapse**
- Wszystkie obrazy podobne → **Restart**

## 💡 Jeśli nadal są problemy

1. Sprawdź `ANALIZA_WYNIKOW.md` sekcja "Plan B"
2. Rozważ silniejsze augmentacje
3. Dodaj diversity loss
4. Zwiększ model capacity (g_ch: 96)

---

**Pytania?** Zobacz pełną dokumentację w `TO_NAPRAWIENIE.md`  
**Data:** 2026-01-18
