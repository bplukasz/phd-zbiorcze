# Expected Outputs - E002-01

## Struktura katalogów output

```
/kaggle/working/
├── runs/
│   └── cifar10_baseline_seed0/
│       ├── checkpoints/
│       │   ├── ckpt_005000.pt
│       │   ├── ckpt_010000.pt
│       │   ├── ...
│       │   └── ckpt_200000.pt
│       ├── samples/
│       │   ├── grid_001000.png
│       │   ├── grid_002000.png
│       │   ├── ...
│       │   └── grid_200000.png
│       ├── log.csv
│       └── done.json
├── e11_cache/
│   ├── cifar10_real_clip_feats.npy
│   ├── cifar10_meta.json
│   ├── cifar10_K20_real_labels.npy
│   ├── cifar10_K20_centroids.npy
│   ├── cifar10_K20_sizes.json
│   ├── cifar10_K50_real_labels.npy
│   ├── cifar10_K50_centroids.npy
│   ├── cifar10_K50_sizes.json
│   ├── cifar10_K100_real_labels.npy
│   ├── cifar10_K100_centroids.npy
│   └── cifar10_K100_sizes.json
├── e11_results/
│   ├── e11_cifar10_K20/
│   │   └── metrics.json
│   ├── e11_cifar10_K50/
│   │   └── metrics.json
│   ├── e11_cifar10_K100/
│   │   └── metrics.json
│   └── aggregates_allK.json
├── report_aggregates.csv
└── report_per_cluster.csv
```

## Format plików

### log.csv (training)
```csv
step,loss_d,loss_g,fid,time
1000,0.4523,1.2341,,45.23
2000,0.4012,1.1892,,90.45
...
10000,0.3521,1.0234,85.23,450.12
20000,0.3102,0.9876,72.45,900.34
```

### done.json
```json
{
  "status": "done",
  "outdir": "/kaggle/working/runs/cifar10_baseline_seed0"
}
```

### cifar10_meta.json
```json
{
  "N": 50000,
  "d": 512,
  "K_list": [20, 50, 100]
}
```

### cifar10_K50_sizes.json
```json
{
  "K": 50,
  "sizes": [1024, 987, 1102, ..., 945]
}
```

### metrics.json (E11 per K)
```json
{
  "ckpt": "/kaggle/working/runs/.../ckpt_200000.pt",
  "K": 50,
  "aggregates": {
    "K": 50,
    "js_hist": 0.0234,
    "cluster_fid_mean_w": 45.23,
    "cluster_fid_worst": 89.45,
    "cluster_fid_tail10": 67.89,
    "cluster_kid_mean_w": 0.0123,
    "cluster_kid_worst": 0.0456,
    "cluster_kid_tail10": 0.0298,
    "coverage_mean_w": 0.87,
    "coverage_worst": 0.42,
    "coverage_tail10": 0.58
  },
  "per_cluster": [
    {
      "K": 50,
      "cluster": 0,
      "n_real": 1024,
      "n_fake": 412,
      "fid": 42.3,
      "kid": 0.011,
      "kid_std": 0.002,
      "coverage": 0.89
    },
    ...
  ]
}
```

### aggregates_allK.json
```json
{
  "ckpt": "/kaggle/working/runs/.../ckpt_200000.pt",
  "results": [
    {
      "K": 20,
      "js_hist": 0.0189,
      "cluster_fid_mean_w": 48.12,
      ...
    },
    {
      "K": 50,
      "js_hist": 0.0234,
      "cluster_fid_mean_w": 45.23,
      ...
    },
    {
      "K": 100,
      "js_hist": 0.0312,
      "cluster_fid_mean_w": 47.89,
      ...
    }
  ]
}
```

### report_aggregates.csv
```csv
K,js_hist,cluster_fid_mean_w,cluster_fid_worst,cluster_fid_tail10,cluster_kid_mean_w,cluster_kid_worst,cluster_kid_tail10,coverage_mean_w,coverage_worst,coverage_tail10,ckpt
20,0.0189,48.12,92.34,71.23,0.0145,0.0512,0.0334,0.85,0.38,0.54,/kaggle/working/runs/.../ckpt_200000.pt
50,0.0234,45.23,89.45,67.89,0.0123,0.0456,0.0298,0.87,0.42,0.58,/kaggle/working/runs/.../ckpt_200000.pt
100,0.0312,47.89,95.67,73.45,0.0156,0.0489,0.0321,0.84,0.35,0.52,/kaggle/working/runs/.../ckpt_200000.pt
```

### report_per_cluster.csv
```csv
K,cluster,n_real,n_fake,fid,kid,kid_std,coverage,ckpt
50,0,1024,412,42.3,0.011,0.002,0.89,/kaggle/working/runs/.../ckpt_200000.pt
50,1,987,398,45.7,0.013,0.003,0.86,/kaggle/working/runs/.../ckpt_200000.pt
50,2,1102,445,38.9,0.009,0.001,0.92,/kaggle/working/runs/.../ckpt_200000.pt
...
```

## Checkpoint format

```python
ckpt = {
    "step": 200000,
    "G": {...},           # Generator state_dict
    "D": {...},           # Discriminator state_dict
    "optG": {...},        # Optimizer G state_dict
    "optD": {...},        # Optimizer D state_dict
    "ema": {...},         # EMA shadow weights
    "cfg": {...}          # Config dict
}
```

## Sample grid format

- Format: PNG
- Rozdzielczość: 32x32 per obraz
- Grid: 8x8 = 64 obrazy
- Value range: [0, 1] (znormalizowane)
- Rozmiar pliku: ~50-100 KB

## Rozmiary plików (przybliżone)

- `log.csv`: ~200 KB (200k kroków)
- `ckpt_*.pt`: ~50-100 MB each
- `sample grid`: ~50-100 KB each
- `real_clip_feats.npy`: ~200 MB (50k images, float16)
- `K50_centroids.npy`: ~100 KB
- `metrics.json`: ~50-200 KB (zależy od K)
- `report*.csv`: ~10-50 KB

**Total storage**: ~5-10 GB dla pełnego runu

## Oczekiwane wartości metryk (CIFAR-10)

### Baseline po 200k kroków:
- **Global FID**: 20-40 (dobre), 40-60 (ok), >60 (słabe)
- **JS divergence**: <0.05 (dobre), 0.05-0.1 (ok), >0.1 (problemy z modami)
- **Cluster-FID mean_w**: Zazwyczaj wyższe niż global FID (+10-20 punktów)
- **Coverage mean_w**: >0.8 (dobre), 0.6-0.8 (ok), <0.6 (mode collapse)
- **worst vs mean_w**: Różnica >30 punktów sugeruje nierówną jakość

### Mode dropping (sanity check):
- JS divergence powinno wzrosnąć >0.15
- Coverage dla usuniętych klastrów → 0
- FID dla usuniętych klastrów → bardzo wysokie (>200)

## Typowe problemy i ich oznaki w outputach

### Mode collapse:
- `coverage_worst` < 0.3
- `js_hist` > 0.2
- Niektóre klastry z `n_fake` = 0 lub bardzo małe

### Nie-konwergencja:
- `loss_g` i `loss_d` oscylują chaotycznie
- `fid` nie spada lub rośnie
- Sample gridy pokazują artifacts/noise

### Overfitting D:
- `loss_g` >> `loss_d`
- `loss_d` → 0
- Sample gridy pokazują powtarzające się wzory

### Underfitting:
- `fid` > 100 przez większość treningu
- Sample gridy nierozpoznawalne

## Validacja wyników

Sprawdź czy:
1. ✅ `log.csv` ma wszystkie 200k kroków
2. ✅ `done.json` istnieje
3. ✅ FID końcowe < 60
4. ✅ JS divergence < 0.1
5. ✅ Coverage mean > 0.7
6. ✅ Żaden klaster nie ma n_fake = 0
7. ✅ Sample gridy pokazują rozpoznawalne obiekty
8. ✅ Report CSV zawiera dane dla wszystkich K

Jeśli wszystkie warunki spełnione → eksperyment sukces! ✅

