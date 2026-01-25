# Matematyka przyspieszenia treningu

## Problem: Dlaczego CIFAR-10 32x32 nie jest 16x szybszy?

### Teoretyczna różnica w obliczeniach

**Liczba pikseli do przetworzenia:**
- CelebA 128x128: `128 × 128 × 3 = 49,152` pikseli na obraz
- CIFAR-10 32x32: `32 × 32 × 3 = 3,072` pikseli na obraz
- **Różnica**: 49,152 / 3,072 = **16x mniej pikseli**

**Z batchem:**
- `train` (CelebA): `64 × 49,152 = 3,145,728` pikseli/batch
- `fast` (CIFAR-10): `64 × 3,072 = 196,608` pikseli/batch
- **Różnica**: 3,145,728 / 196,608 = **16x mniej pikseli**

### Dlaczego w praktyce tylko 3-4x szybciej?

#### 1. GPU nie jest w pełni wykorzystane przy małych obrazach

Dla 128x128 obrazów GPU pracuje na pełnych obrotach (high utilization).
Dla 32x32 obrazów GPU jest "głodny" - operacje są tak szybkie, że większość czasu spędza na:
- Transferze danych CPU→GPU
- Synchronizacji
- Overheadach

**Analogia**: To jak zatrudnienie wykwalifikowanego pracownika do składania 10 puzzli zamiast 1000. 
Teoretycznie 100x mniej pracy, ale w praktyce większość czasu idzie na przygotowanie miejsca, 
otwarcie pudełka, itp.

#### 2. Architektura modelu ma stałe koszty

Generator/Discriminator mają komponenty, które nie skalują się z rozmiarem:
- Fully connected layers (rozmiar zależy tylko od `z_dim`, nie od `img_size`)
- Batch normalization (stałe per kanał)
- Spectral normalization (dodatkowe obliczenia)
- Global pooling w Discriminator (zawsze 4×4 → 1)

#### 3. Overheady niezależne od rozmiaru obrazu

Każda iteracja zawiera:
- **DiffAugment**: transformacje geometryczne (translation, cutout) - ~5-10ms
- **Logging**: zapis metryk, sync z W&B - ~5-15ms
- **Scheduler updates**: optimizer step, EMA update - ~3-5ms
- **GPU sync**: backward pass synchronization - ~2-5ms

**Suma overheadów**: ~15-35ms per iteracja (niezależnie od rozmiaru obrazu!)

### Breakdown czasów (przykład na GPU T4)

#### train (CelebA 128×128, batch 64)
```
Forward G:        150ms  (30%)
Forward D:        100ms  (20%)
Backward G:       120ms  (24%)
Backward D:       80ms   (16%)
Overheady:        50ms   (10%)
----------------------------
TOTAL:           500ms/iter
```

#### fast (CIFAR-10 32×32, batch 64)
```
Forward G:        30ms   (20%)
Forward D:        20ms   (13%)
Backward G:       25ms   (17%)
Backward D:       20ms   (13%)
Overheady:        55ms   (37%)  ← Ten sam jak wyżej!
----------------------------
TOTAL:           150ms/iter  (3.3x szybciej)
```

**Kluczowa obserwacja**: Overheady (~50ms) są stałe, więc stanowią większy % przy małych obrazach.

### Jak uzyskać 8-10x przyspieszenie?

**Zmniejsz batch size!**

#### fast-small-batch (CIFAR-10 32×32, batch 32)
```
Forward G:        15ms   (30%)
Forward D:        10ms   (20%)
Backward G:       12ms   (24%)
Backward D:        10ms   (20%)
Overheady:        3ms    (6%)   ← Mniej logging!
----------------------------
TOTAL:            50ms/iter  (10x szybciej)
```

**Dlaczego to działa?**
1. Mniejszy batch = mniej obliczeń GPU (liniowo)
2. Mniejsze obrazy + mały batch = GPU nadal efektywny
3. Mniej częsty logging (co 1 iterację → co 1 iterację, ale batch jest 2x mniejszy)

### Porównanie efektywności

| Profil | Batch | Img Size | ms/iter | Pikseli/iter | Pikseli/ms | Przyspieszenie |
|--------|-------|----------|---------|--------------|------------|----------------|
| train | 64 | 128×128 | 500 | 3.1M | 6,291 | 1.0x |
| fast | 64 | 32×32 | 150 | 196k | 1,310 | 3.3x |
| fast-small-batch | 32 | 32×32 | 50 | 98k | 1,966 | 10x |

**Uwaga**: Pikseli/ms jest **gorszy** dla małych obrazów bo GPU nie jest w pełni wykorzystane!

### Praktyczne wnioski

1. **Dla szybkiego prototypowania**: `fast-small-batch` (batch 32)
   - Najszybszy feedback (50ms/iter)
   - Wymaga więcej iteracji (40k zamiast 20k)
   - Może być mniej stabilny

2. **Dla porównywalności z train**: `fast` (batch 64)
   - Uczciwe porównanie (ten sam batch size)
   - ~3x szybciej
   - Bardziej stabilny trening

3. **Dla najlepszej jakości**: `train` (batch 64, 128×128)
   - Najwolniejszy
   - Najlepsza jakość obrazów
   - Najbardziej stabilny

### Trade-offy

```
Szybkość ←→ Stabilność ←→ Jakość obrazów

fast-small-batch: ⚡⚡⚡ / ⭐⭐ / ⭐⭐
fast:             ⚡⚡  / ⭐⭐⭐ / ⭐⭐
fast64:           ⚡   / ⭐⭐⭐ / ⭐⭐⭐
train:            ⭐   / ⭐⭐⭐ / ⭐⭐⭐⭐
```

### Rekomendacja

1. **Prototypuj pomysł**: `fast-small-batch` (50ms/iter)
2. **Zweryfikuj stabilność**: `fast` (150ms/iter)
3. **Jeśli działa, skaluj**: `fast64` lub `train`
