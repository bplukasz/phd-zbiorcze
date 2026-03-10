# Zestaw konfiguracji eksperymentalnych — frequency-aware R3GAN

Wygenerowano: 2026-03-10  
Dotyczy: `e001-02-r3gan-baseline`

---

## Struktura eksperymentu

Warianty porównawcze (wg roadmapy):

| Wariant | Opis |
|---------|------|
| **R0** | Czysty baseline R3GAN — bez żadnych zmian |
| **R1** | Matched-capacity control — gałąź bez DWT (avg\_pool2d + wieża conv) |
| **R2** | WaveD — dyskryminator z gałęzią HF (Haar, L=1, LH/HL/HH) |
| **R3** | WaveD + WaveReg — jak R2 + regularizacja statystyk HF generatora |
| **R4** | WaveD + FFTReg — jak R2 + regularizacja FFT generatora (kontrola) |

Wszystkie warianty: identyczny loss adversarialny, optymizer, gamma, EMA, preprocessing, budżet kimg.

---

## Kolejność uruchamiania

### 🔹 Faza A — infrastruktura (przed wszystkim)
Przed uruchomieniem eksperymentów upewnij się że:
1. Przechodzą testy: `pytest tests/test_wavelets.py tests/test_wavelet_branch.py`
2. Smoke testy: `smoke`, `smoke_matched_capacity`, `smoke_waved_wavereg`, `smoke_waved_fftreg`
3. Baseline działa stabilnie (loss converges, metryki liczą się poprawnie)

### 🔹 Faza B — szybki test sensu naukowego (1 seed, 32×32)
**Cel:** sprawdzić czy WaveD daje sygnał powyżej matched-capacity.  
Uruchamiaj równolegle (po przejściu Fazy A):

```
R0: phase_b_r0_baseline_32
R1: phase_b_r1_matched_capacity_32
R2: phase_b_r2_waved_32
```

**Decyzja po Fazie B:**  
- Jeśli R2 ≤ R1 pod FID/KID → rewizja przed przejściem do Fazy C  
- Jeśli R2 > R1 → przejdź do Fazy C

### 🔹 Faza B — na 64×64 (równolegle lub po 32×32)
```
R0: phase_b_r0_baseline_64
R1: phase_b_r1_matched_capacity_64
R2: phase_b_r2_waved_64
```

### 🔹 Faza C — pełna metoda (2 seedy, 32×32)
**Cel:** wybrać finalny wariant (R2/R3/R4).  
Seed 42 (wszystkie 5 wariantów):

```
R0: phase_b_r0_baseline_32          (seed 42, już masz z Fazy B)
R1: phase_b_r1_matched_capacity_32  (seed 42)
R2: phase_b_r2_waved_32             (seed 42)
R3: phase_c_r3_waved_wavereg_32     (seed 42)
R4: phase_c_r4_waved_fftreg_32      (seed 42)
```

Seed 43 (R0 i najlepszy kandydat z seed 42):

```
R0: phase_c_r0_baseline_32_seed43
R3: phase_c_r3_waved_wavereg_32_seed43
```

Lambda sweep dla R3 (jeśli wygrał):

```
phase_c_sweep_wavereg_lambda_005_32   (λ=0.005)
phase_c_r3_waved_wavereg_32          (λ=0.02  — już masz)
phase_c_sweep_wavereg_lambda_050_32   (λ=0.050)
```

### 🔹 Faza C — ablacje (po wyborze wariantu)
Uruchamiaj **tylko po** wybraniu R_best w Fazie C:

```
ablation_allbands_waved_32           (HF-only vs all-bands)
ablation_fuse_stage1_waved_32        (fuzja po stage[0] vs stage[1])
ablation_wavereg_mean_only_32        (WaveReg mean+std vs mean only)
```

### 🔹 Faza D — eksperyment potwierdzający (3 seedy, 128×128)
**Cel:** potwierdzenie wyniku na większej rozdzielczości i 3 seedach.  
Przed uruchomieniem: wypełnij `phase_d_rbest_128_*` prawidłowym wariantem z Fazy C.

```
R0 seed 42: phase_d_r0_baseline_128_seed42
R0 seed 43: phase_d_r0_baseline_128_seed43
R0 seed 44: phase_d_r0_baseline_128_seed44

R_best seed 42: phase_d_rbest_128_seed42
R_best seed 43: phase_d_rbest_128_seed43
R_best seed 44: phase_d_rbest_128_seed44
```

---

## Przegląd wszystkich plików konfiguracyjnych

### Smoke testy (istniejące)
| Plik | Opis |
|------|------|
| `smoke.yaml` | R0 — smoke, 32×32, 200 steps |
| `smoke_matched_capacity.yaml` | R1 — smoke, 32×32 |
| `smoke_waved_wavereg.yaml` | R3 — smoke, 32×32 |
| `smoke_waved_fftreg.yaml` | R4 — smoke, 32×32 |

### Faza B — 32×32 (CIFAR-10, 1 seed, ~100 kimg)
| Plik | Wariant | Seed |
|------|---------|------|
| `phase_b_r0_baseline_32.yaml` | R0 Baseline | 42 |
| `phase_b_r1_matched_capacity_32.yaml` | R1 Matched-Cap | 42 |
| `phase_b_r2_waved_32.yaml` | R2 WaveD | 42 |

### Faza B — 64×64 (CIFAR-10, 1 seed, ~200 kimg)
| Plik | Wariant | Seed |
|------|---------|------|
| `phase_b_r0_baseline_64.yaml` | R0 Baseline | 42 |
| `phase_b_r1_matched_capacity_64.yaml` | R1 Matched-Cap | 42 |
| `phase_b_r2_waved_64.yaml` | R2 WaveD | 42 |

### Faza C — 32×32 (CIFAR-10, 2 seedy, ~100 kimg)
| Plik | Wariant | Seed |
|------|---------|------|
| `phase_c_r3_waved_wavereg_32.yaml` | R3 WaveD+WaveReg λ=0.02 | 42 |
| `phase_c_r4_waved_fftreg_32.yaml` | R4 WaveD+FFTReg λ=0.02 | 42 |
| `phase_c_r0_baseline_32_seed43.yaml` | R0 Baseline | 43 |
| `phase_c_r3_waved_wavereg_32_seed43.yaml` | R3 WaveD+WaveReg λ=0.02 | 43 |

### Faza C — 64×64 (CIFAR-10, 1 seed, ~200 kimg)
| Plik | Wariant | Seed |
|------|---------|------|
| `phase_c_r3_waved_wavereg_64.yaml` | R3 WaveD+WaveReg λ=0.02 | 42 |
| `phase_c_r4_waved_fftreg_64.yaml` | R4 WaveD+FFTReg λ=0.02 | 42 |

### Lambda sweep — WaveReg (32×32)
| Plik | λ |
|------|---|
| `phase_c_sweep_wavereg_lambda_005_32.yaml` | 0.005 |
| `phase_c_r3_waved_wavereg_32.yaml` | 0.020 (bazowy) |
| `phase_c_sweep_wavereg_lambda_050_32.yaml` | 0.050 |

### Ablacje (32×32, po wyborze R_best)
| Plik | Co bada |
|------|---------|
| `ablation_allbands_waved_32.yaml` | HF-only vs all-bands (LL+LH+HL+HH) |
| `ablation_fuse_stage1_waved_32.yaml` | Fuzja po stage[0] vs stage[1] |
| `ablation_wavereg_mean_only_32.yaml` | WaveReg: mean+std vs mean only |

### Faza D — 128×128 (CelebA, 3 seedy, ~400 kimg)
| Plik | Wariant | Seed |
|------|---------|------|
| `phase_d_r0_baseline_128_seed42.yaml` | R0 Baseline | 42 |
| `phase_d_r0_baseline_128_seed43.yaml` | R0 Baseline | 43 |
| `phase_d_r0_baseline_128_seed44.yaml` | R0 Baseline | 44 |
| `phase_d_rbest_128_seed42.yaml` | R_best (template R3) | 42 |
| `phase_d_rbest_128_seed43.yaml` | R_best (template R3) | 43 |
| `phase_d_rbest_128_seed44.yaml` | R_best (template R3) | 44 |

---

## Parametry zamrożone (nie zmieniać między wariantami)

```yaml
lr_g: 2.0e-4
lr_d: 2.0e-4
betas: [0.0, 0.99]
gamma: 10.0
use_amp_for_g: true
use_amp_for_d: false
channels_last: true
```

Przy tym samym budżecie kimg:
- 32×32: `steps=100000, batch_size=128` → ~100 kimg
- 64×64: `steps=200000, batch_size=64` → ~200 kimg  
- 128×128: `steps=400000, batch_size=32` → ~400 kimg

---

## Punkty decyzyjne

```
Faza A passed? → start Fazy B
Faza B: R2 > R1? → start Fazy C  |  R2 ≤ R1 → rewizja
Faza C: wybrać R_best → uzupełnić phase_d_rbest_128_*.yaml → start Fazy D
Faza D: potwierdzenie na 128×128, 3 seedy → wynik końcowy
```

---

## Kryteria realnej poprawy (z roadmapy)

Poprawa jest realna gdy jednocześnie:
1. FID/KID poprawia się względem R0
2. Precision/Recall nie pogarsza się wyraźnie
3. LPIPS-diversity nie spada
4. RPSE/WBED poprawiają się zgodnie z mechanizmem HF
5. R1 (matched-capacity) **nie osiąga tego samego** → efekt pochodzi z waveletów
6. Wynik powtarzalny między seedami (mean ± std)

