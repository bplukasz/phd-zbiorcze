# E002-01: GAN Cluster Metrics - Podsumowanie

## Status
✅ **Implementacja zakończona**

## Zaimplementowano

### 1. Architektura modelu
- **Generator**: ResNet-based z BatchNorm, upsampling
- **Discriminator**: ResNet-based z Spectral Normalization
- **Loss**: Hinge loss (non-saturating)
- **Optimizer**: Adam (lr=2e-4, beta=(0.0, 0.9))
- **EMA**: Exponential Moving Average (beta=0.999) dla stabilności

### 2. Metryki
#### Global metrics
- **FID (Inception)**: Frechet Inception Distance na pełnym datasecie
- Streaming computation (nie trzyma wszystkich features w RAM)

#### Cluster metrics (E11)
- **Cluster-FID**: FID per klaster w przestrzeni CLIP
- **Cluster-KID**: Kernel Inception Distance per klaster
- **Coverage**: Pokrycie klastrów przez wygenerowane próbki
- **JS Divergence**: Rozbieżność rozkładów histogramów klastrów

#### Agregaty
- `mean_w`: Średnia ważona (wagowane rozmiarem klastrów)
- `worst`: Najgorszy klaster
- `tail10`: Średnia z 10% najgorszych klastrów

### 3. Pipeline
1. **Precompute**: Ekstrakcja CLIP features + KMeans clustering (K ∈ {20, 50, 100})
2. **Training**: Trening GAN z periodic checkpointing i global FID evaluation
3. **Evaluation**: Obliczenie cluster metrics dla checkpointów
4. **Report**: Generowanie CSV z wynikami (aggregates + per-cluster)

### 4. Struktura kodu
```
e002-01-gan-cluster-metrics/
├── dataset/src/
│   ├── models/
│   │   ├── gan_resnet.py    # Generator + Discriminator
│   │   └── ema.py            # Exponential Moving Average
│   ├── data/
│   │   └── datasets.py       # CIFAR-10/CelebA loaders + ClusterDropWrapper
│   ├── utils/
│   │   ├── seed.py           # Reproducibility
│   │   ├── io.py             # YAML/JSON utils
│   │   └── logger.py         # CSV logging
│   ├── eval/
│   │   ├── inception_feat.py # Inception features dla FID
│   │   ├── clip_feat.py      # CLIP features dla clustering
│   │   ├── fid_utils.py      # FID computation + streaming stats
│   │   ├── precompute_real.py # Precompute pipeline
│   │   ├── eval_e11.py       # E11 metrics evaluation
│   │   └── report.py         # CSV report generation
│   ├── configs/
│   │   ├── cifar10_baseline.yml
│   │   ├── cifar10_drop.yml
│   │   ├── e11_precompute_cifar10.yml
│   │   └── e11_eval_cifar10.yml
│   └── train.py              # Main training script
├── kernels/
│   ├── script/run.py         # Kaggle script kernel
│   └── notebook/runner.ipynb # Kaggle notebook
├── run_pipeline.sh           # Full pipeline bash script
└── README.md
```

### 5. Konfiguracje YAML
- **cifar10_baseline.yml**: Pełny trening 200k kroków
- **cifar10_drop.yml**: Mode dropping sanity check
- **e11_precompute_cifar10.yml**: Precompute real features + clustering
- **e11_eval_cifar10.yml**: E11 metrics evaluation

### 6. Sanity checks
- **Mode dropping**: ClusterDropWrapper pozwala usunąć wybrane klastry z datasetu treningowego
- Służy do weryfikacji czy metryki wykrywają brakujące mody

## Użycie

### Lokalnie (po pobraniu z Kaggle)
```bash
# Pełny pipeline
./run_pipeline.sh

# Lub krok po kroku:
python dataset/src/eval/precompute_real.py --config dataset/src/configs/e11_precompute_cifar10.yml
python dataset/src/train.py --config dataset/src/configs/cifar10_baseline.yml
python dataset/src/eval/eval_e11.py --config dataset/src/configs/e11_eval_cifar10.yml --ckpt path/to/ckpt.pt
python dataset/src/eval/report.py --e11_dir /path/to/e11_results --out report.csv
```

### Na Kaggle
```bash
# Script kernel
kaggle kernels push -p kernels/script/

# Notebook kernel
kaggle kernels push -p kernels/notebook/
```

## Wyniki

### Format output
- **Training logs**: CSV z loss_d, loss_g, fid, time
- **Checkpoints**: PyTorch state_dict (G, D, optG, optD, EMA)
- **Samples**: Grid obrazów co 1k kroków
- **E11 metrics**: JSON per K + CSV aggregates
- **Report**: CSV z metrykami (agregaty + per-cluster)

### Katalogi wyjściowe
- `/kaggle/working/runs/<run_name>/` - training outputs
- `/kaggle/working/e11_cache/` - precomputed features + clusters
- `/kaggle/working/e11_results/` - E11 evaluation results

## Zależności
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- open_clip_torch == 2.24.0
- scikit-learn == 1.4.2
- scipy == 1.11.4
- pandas >= 2.0.0

## Następne kroki (opcjonalne rozszerzenia)

### 1. Resume training
Dodaj w `train.py` obsługę `--resume` do wczytywania checkpointu

### 2. Multi-checkpoint sweep
Stwórz `eval/sweep_e11.py` iterujący po wszystkich checkpointach

### 3. Wizualizacje
- Heatmapa coverage per cluster vs checkpoint
- Histogram rozkładów klastrów (real vs fake)
- t-SNE projekcja CLIP features

### 4. Ablacje
- Spectral Norm on/off
- EMA on/off
- Different cluster counts K
- Different CLIP models (ViT-B-32 vs ViT-L-14)

### 5. CelebA support
Dataset jest już zaimplementowany, wystarczy:
```yaml
data:
  dataset: celeba
  resolution: 64
```

## Zgodność z instrukcjami (dino.md)

✅ Wszystkie komponenty z dino.md zaimplementowane:
- [x] ResNet GAN (Generator + Discriminator)
- [x] Hinge loss
- [x] Spectral Normalization
- [x] EMA
- [x] Global FID (Inception, streaming)
- [x] CLIP feature extraction
- [x] KMeans clustering
- [x] Cluster-FID
- [x] Cluster-KID
- [x] Coverage metric
- [x] JS divergence
- [x] Agregaty (mean_w, worst, tail10)
- [x] ClusterDropWrapper (mode dropping)
- [x] Config system (YAML)
- [x] CSV logging
- [x] Kaggle kernels (script + notebook)

## Autor
Łukasz Tymoszuk

Data utworzenia: 12 lutego 2026

## Licencja
CC0-1.0

