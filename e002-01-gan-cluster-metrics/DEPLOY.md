# Instrukcja wdrożenia E002-01 na Kaggle

## Krok 1: Przygotowanie datasetu

```bash
cd e002-01-gan-cluster-metrics/dataset
kaggle datasets create -p .
```

To utworzy dataset `bplukasz/e002-01-gan-cluster-metrics-lib`.

## Krok 2: Aktualizacja datasetu (przy zmianach)

```bash
cd e002-01-gan-cluster-metrics/dataset
kaggle datasets version -p . -m "Updated code"
```

## Krok 3: Push kerneli

### Script kernel
```bash
cd e002-01-gan-cluster-metrics/kernels/script
kaggle kernels push
```

### Notebook kernel
```bash
cd e002-01-gan-cluster-metrics/kernels/notebook
kaggle kernels push
```

## Krok 4: Uruchomienie na Kaggle

### Opcja A: Script kernel

1. Otwórz kernel na Kaggle: https://www.kaggle.com/code/bplukasz/e002-01-gan-cluster-metrics-script
2. Kliknij "Edit"
3. Upewnij się, że GPU jest włączone (Settings → Accelerator → GPU T4 x2)
4. Kliknij "Run All"

Możesz też uruchomić przez API:
```bash
kaggle kernels output bplukasz/e002-01-gan-cluster-metrics-script -p /output/path
```

### Opcja B: Notebook kernel

1. Otwórz notebook: https://www.kaggle.com/code/bplukasz/e002-01-gan-cluster-metrics-notebook
2. Włącz GPU
3. Uruchamiaj komórki po kolei lub "Run All"

## Tryby uruchomienia

W script kernel możesz ustawić parametr `--mode`:

```python
# W run.py lub przez notebook parameters:
--mode train       # Tylko trening
--mode precompute  # Tylko precompute real features
--mode eval        # Tylko evaluacja (wymaga --ckpt)
--mode full        # Train + precompute
```

## Monitorowanie

### Logi treningowe
```
/kaggle/working/runs/<run_name>/log.csv
```

Kolumny: step, loss_d, loss_g, fid, time

### Sample gridy
```
/kaggle/working/runs/<run_name>/samples/grid_*.png
```

### Checkpointy
```
/kaggle/working/runs/<run_name>/checkpoints/ckpt_*.pt
```

## Pobieranie wyników

### Przez API
```bash
# Pobierz output całego kernela
kaggle kernels output bplukasz/e002-01-gan-cluster-metrics-script -p ./results

# Pobierz tylko logi
kaggle kernels output bplukasz/e002-01-gan-cluster-metrics-script -p ./results --file log.csv
```

### Przez UI
1. Otwórz kernel
2. Output → Download

## Troubleshooting

### OOM (Out of Memory)
Edytuj config i zmniejsz:
- `batch_size: 128` → `64`
- `n_eval_fake: 10000` → `5000`

### Zbyt długi czas precompute
W `e11_precompute_cifar10.yml`:
- Zwiększ `batch_size: 256` → `512`
- Zmniejsz `K_list: [20, 50, 100]` → `[50]`

### Brak CIFAR-10
Dataset powinien być automatycznie pobrany. Jeśli nie, dodaj dataset input:
1. Edit kernel
2. Data → Add input → Search "CIFAR-10"
3. Dodaj oficjalny dataset CIFAR-10

### Brak convergencji
Sprawdź logi:
- Czy loss_d i loss_g nie eksplodują?
- Czy FID spada w czasie?

Jeśli problem:
- Zmniejsz LR: `lr_g: 0.0001`, `lr_d: 0.0001`
- Zwiększ `n_critic: 2`

## Szacowany czas wykonania

Na GPU T4 x2:
- **Precompute** (CIFAR-10, K=50): ~5 min
- **Training** (200k kroków): ~12-15 godz
- **Eval E11** (K=50, 20k fake): ~15 min

Łącznie: **~16 godzin**

## Wykorzystanie zasobów Kaggle

- 1 run (train + eval) ≈ 16h GPU
- Limit Kaggle: 30h/tydzień GPU
- Zalecane: Uruchom 1 baseline run, potem ablacje

## Kolejne eksperymenty

Po baseline, utwórz nowe runy z ablacjami:

### Run 2: Bez Spectral Norm
```yaml
model:
  spectral_norm: false
```

### Run 3: Bez EMA
```yaml
train:
  ema:
    enabled: false
```

### Run 4: Mode dropping
```yaml
data:
  drop_clusters: [7, 12]
  drop_fraction: 1.0
```

Każdy run zapisz jako osobny kernel lub zmień `run.name` w config.

## Backup

Regularnie pobieraj checkpointy:
```bash
kaggle kernels output bplukasz/e002-01-gan-cluster-metrics-script -p ./backup
```

Najważniejsze:
- `checkpoints/ckpt_200000.pt` (końcowy model)
- `log.csv` (historia treningowa)
- `report_aggregates.csv` (metryki E11)

## Support

W razie problemów sprawdź:
1. Czy dataset `e002-01-gan-cluster-metrics-lib` jest public/private correctly?
2. Czy w kernel settings masz GPU enabled?
3. Czy `shared-lib` jest dostępny jako input?

Logi błędów znajdziesz w kernel output na Kaggle.

