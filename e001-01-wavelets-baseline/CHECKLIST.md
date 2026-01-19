# ✅ CHECKLIST przed następnym treningiem

## 📋 Pre-flight Check

### 1. Konfiguracja
- [ ] Sprawdzono parametry w `smoke.yaml` lub `train.yaml`
- [ ] `lr_D = 0.0001` (zmniejszone!)
- [ ] `eval_every` ustawione odpowiednio (100 dla smoke, 10000 dla train)
- [ ] `use_r1_penalty = true` dla train profile (stabilizacja)
- [ ] Ścieżki do danych są poprawne (`data_dir`)

### 2. Kod
- [ ] Najnowsza wersja `experiment.py` z progress tracking
- [ ] R1 penalty zaimplementowany
- [ ] Mode collapse detection aktywny
- [ ] FID monitoring z ostrzeżeniami

### 3. Środowisko
- [ ] GPU dostępne (`torch.cuda.is_available()`)
- [ ] VRAM > 6GB (dla batch_size=32)
- [ ] Miejsce na dysku > 50GB (dla próbek i checkpointów)
- [ ] torch-fidelity zainstalowany

### 4. Monitoring
- [ ] Terminal gotowy do obserwacji logów
- [ ] Plan na sprawdzanie `grid_*.png` co 1-2h
- [ ] Dostęp do katalogu z artifacts
- [ ] (opcjonalnie) W&B zalogowane

---

## 🎯 Kryteria sukcesu

### Smoke Test (500 kroków, ~30 min)
- [ ] **FID na kroku 100:** < 350
- [ ] **FID na kroku 200:** < 320
- [ ] **FID na kroku 500:** < 300
- [ ] **Trend:** FID maleje przy każdym pomiarze
- [ ] **Brak mode collapse** (brak skoków loss > 5.0)
- [ ] **Obrazy różnorodne** w grid_*.png

### Pełny Trening (30k kroków, ~2-3 dni)
- [ ] **FID po 5k kroków:** < 200
- [ ] **FID po 10k kroków:** < 100
- [ ] **FID po 20k kroków:** < 70
- [ ] **FID po 30k kroków:** < 50
- [ ] **Stabilność:** max 1 mode collapse na 10k kroków
- [ ] **Jakość wizualna:** czytelne twarze, naturalne kolory

---

## 🚨 Akcje w razie problemów

### FID rośnie przy pierwszym pomiarze
```yaml
# Zmniejsz jeszcze bardziej lr_D
lr_D: 0.00005  # bardzo ostrożne
```

### Mode collapse mimo R1 penalty
```yaml
# Zwiększ R1 lambda
r1_lambda: 20.0  # było 10.0

# Lub dodaj silniejsze augmentacje
diffaug_policy: "color,translation,cutout,flip,rotation"
```

### Discriminator zbyt słaby (loss_D >> 2.0)
```yaml
# Zwiększ lekko lr_D
lr_D: 0.00015

# Lub zmniejsz strength augmentacji
diffaug_policy: "color,translation"
```

### Generator nie uczy się (grad_norm_G < 0.01)
```yaml
# Zwiększ learning rate
lr_G: 0.0003

# Lub zmniejsz R1 penalty
r1_lambda: 5.0
```

---

## 📊 Monitoring Dashboard (ręczny)

### Co 100 kroków (1-2 min):
- [ ] Sprawdź console output
- [ ] Obserwuj loss_D i loss_G
- [ ] Sprawdź grad_norm (czy nie za małe/duże)

### Co 1000 kroków (~15 min):
- [ ] Obejrzyj `grid_*.png`
- [ ] Sprawdź czy twarze są różnorodne
- [ ] Oceń jakość wizualną

### Przy każdej ewaluacji:
- [ ] Zanotuj FID i KID
- [ ] Porównaj z poprzednim pomiarem
- [ ] Sprawdź czy jest trend wzrostowy/spadkowy
- [ ] W razie wzrostu > 20 punktów → rozważ stop

### Daily (1x dziennie dla train):
- [ ] Backup checkpointów
- [ ] Sprawdź wolne miejsce na dysku
- [ ] Oceń postęp vs oczekiwania
- [ ] Rozważ early stopping jeśli FID stabilny > 5k kroków

---

## 📝 Template notatek z treningu

```markdown
## Run: [DATA] - [PROFILE]

### Setup
- Profile: smoke / train
- GPU: [nazwa]
- Start time: [czas]
- Config changes: [lista zmian]

### Checkpoints
| Krok | FID | KID | Loss_D | Loss_G | Notatki |
|------|-----|-----|--------|--------|---------|
| 100  |     |     |        |        |         |
| 500  |     |     |        |        |         |
| 1000 |     |     |        |        |         |

### Observations
- [ ] Mode collapse events: [liczba, kroki]
- [ ] Best FID: [wartość] @ step [krok]
- [ ] Visual quality: [1-10]
- [ ] Grid diversity: [ocena]

### Issues
- [lista problemów]

### Conclusions
- [wnioski]
- [co zmienić w następnym runie]
```

---

## 🔄 Po zakończeniu treningu

### Analiza wyników
- [ ] Sprawdzono logs.csv
- [ ] Wygenerowano wykres FID vs steps
- [ ] Oceiono najlepsze checkpoint
- [ ] Przeanalizowano `grid_*.png` timeline
- [ ] Sprawdzono próbki z final_50k

### Dokumentacja
- [ ] Zapisano parametry użyte w tym runie
- [ ] Zanotowano wszystkie problemy
- [ ] Udokumentowano FID progression
- [ ] Zapisano wnioski do następnego eksperymentu

### Cleanup (opcjonalnie)
- [ ] Usunięto pośrednie checkpointy (zostaw co 5000)
- [ ] Skompresowano gridy
- [ ] Zachowano logs.csv i final checkpoint
- [ ] Zarchiwizowano najlepsze próbki

---

## 🎓 Lessons Learned (template)

### Co działało:
-

### Co nie działało:
-

### Co zmienić następnym razem:
-

### Odkrycia:
-

---

**Data utworzenia:** 2026-01-18  
**Ostatnia aktualizacja:** 2026-01-18  
**Status:** ✅ Gotowe do użycia
