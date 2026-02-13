# E002-01: GAN Cluster Metrics

## Opis

Eksperyment implementujący ResNet GAN z metrykami klasterowymi E11:
- **Cluster-FID**: FID per klaster w przestrzeni CLIP
- **Cluster-KID**: Kernel Inception Distance per klaster
- **Coverage**: Pokrycie klastrów przez wygenerowane próbki
- **JS Divergence**: Rozbieżność rozkładów histogramów

## Struktura

```
e002-01-gan-cluster-metrics/
├── dataset/
│   ├── src/
│   │   ├── models/          # Generator i Discriminator
│   │   ├── data/            # Loadery danych + ClusterDropWrapper
│   │   ├── utils/           # Seed, IO, Logger
│   │   ├── eval/            # Metryki: FID, CLIP, E11
│   │   ├── configs/         # Konfiguracje YAML
│   │   └── train.py         # Główny skrypt treningu
│   └── dataset-metadata.json
├── kernels/
│   ├── script/              # Kaggle script kernel
│   └── notebook/            # Kaggle notebook kernel
└── requirements.txt
```

## Użycie

### 1. Precompute Real Features (raz na dataset)

```bash
python src/eval/precompute_real.py --config src/configs/e11_precompute_cifar10.yml
```

To:
- Wyekstraktuje cechy CLIP dla wszystkich real images
- Wykona klasteryzację KMeans dla K ∈ {20, 50, 100}
- Zapisze centroids i labels

### 2. Trening

```bash
python src/train.py --config src/configs/cifar10_baseline.yml
```

Tryby:
- `cifar10_baseline.yml`: Pełny trening 200k kroków
- `cifar10_drop.yml`: Sanity-check mode dropping (wymaga precompute)

### 3. Evaluacja E11

```bash
python src/eval/eval_e11.py \
  --config src/configs/e11_eval_cifar10.yml \
  --ckpt /path/to/ckpt_200000.pt
```

### 4. Raport

```bash
python src/eval/report.py \
  --e11_dir /kaggle/working/e11_results \
  --out /kaggle/working/report.csv
```

## Konfiguracja

### Trening (`cifar10_baseline.yml`)

```yaml
model:
  z_dim: 128
  g_ch: 128
  d_ch: 128
  spectral_norm: true

train:
  steps: 200000
  lr_g: 0.0002
  lr_d: 0.0002
  ema:
    enabled: true
    beta: 0.999
```

### E11 Eval (`e11_eval_cifar10.yml`)

```yaml
e11:
  K_list: [20, 50, 100]
  n_fake: 20000
  assignment: hard  # lub topM
  kid:
    enabled: true
  coverage:
    enabled: true
```

## Metryki

### Agregaty (per K)

- `cluster_fid_mean_w`: Średnia ważona FID
- `cluster_fid_worst`: Najgorszy klaster
- `cluster_fid_tail10`: Średnia z 10% najgorszych
- `js_hist`: JS divergence rozkładów

### Per-klaster

- `fid`, `kid`, `coverage` dla każdego klastra
- `n_real`, `n_fake` - rozmiary klastrów

## Na Kaggle

### Script Kernel

```python
# W run.py możesz ustawić tryb:
# --mode train       # tylko trening
# --mode precompute  # tylko precompute real
# --mode eval        # tylko evaluacja (wymaga --ckpt)
# --mode full        # train + precompute
```

### Notebook

Otwórz `runner.ipynb` i uruchom komórki.

## Zależności

- PyTorch >= 2.0.0
- open_clip_torch == 2.24.0
- scikit-learn == 1.4.2
- scipy == 1.11.4
- pandas

## Autor

Łukasz Tymoszuk

## Licencja

CC0-1.0

