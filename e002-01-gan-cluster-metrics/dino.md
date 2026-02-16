Poniżej masz **kompletną, spójną implementację** (PyTorch + YAML), gotową do wklejenia jako pliki w Kaggle (np. w notebooku: tworzysz katalog `project/` i zapisujesz pliki). Uwzględnia:

- trening: **ResNet GAN + hinge loss + SpectralNorm w D** (+ opcjonalnie EMA G),
- logowanie/checkpointy/samplowanie,
- **global FID (Inception)** liczone streamingowo (bez trzymania wszystkich cech w RAM),
- **E11: Cluster-FID/Cluster-KID/Cluster-Coverage/JS(hist)** bez etykiet:
  - klasteryzacja reali w **CLIP feature space** (frozen),
  - przypisanie fake do klastrów,
  - metryki per-klaster + agregaty (`mean_w`, `worst`, `tail@10%`),
  - heatmap-friendly CSV,
- sanity-check „mode dropping”: dataset wrapper `drop_clusters`.

> Uwaga metodologiczna: tutaj **klastry i cluster-metryki** są w CLIP space (tanie i stabilne). Żeby rozbroić zarzut circularity, raportujesz równolegle **global FID (Inception)** i w ablacjach możesz użyć alternatywnego encodera do klastrów (np. DINO) – ale to już opcjonalne.

---

# 0) Jak uruchamiać na Kaggle

W notebooku Kaggle (na górze):

```bash
!pip -q install open_clip_torch==2.24.0 scikit-learn==1.4.2 pyyaml==6.0.1 scipy==1.11.4
```

Struktura:

```
/kaggle/working/project/...
```

Uruchomienia:
```bash
!python project/train.py --config project/configs/cifar10_baseline.yml
!python project/eval/precompute_real.py --config project/configs/e11_precompute_cifar10.yml
!python project/eval/eval_e11.py --config project/configs/e11_eval_cifar10.yml --ckpt /kaggle/working/runs/.../checkpoints/ckpt_200000.pt
```

---

# 1) Struktura plików

Wklej poniższe pliki do `project/`:

- `project/train.py`
- `project/configs/*.yml`
- `project/models/gan_resnet.py`
- `project/models/ema.py`
- `project/data/datasets.py`
- `project/utils/seed.py`
- `project/utils/io.py`
- `project/utils/logger.py`
- `project/eval/clip_feat.py`
- `project/eval/inception_feat.py`
- `project/eval/fid_utils.py`
- `project/eval/precompute_real.py`
- `project/eval/eval_e11.py`
- `project/eval/report.py`

---

# 2) KONFIGI YAML

## `project/configs/cifar10_baseline.yml`
```yaml
run:
  name: cifar10_baseline_seed0
  seed: 0
  outdir: /kaggle/working/runs

data:
  dataset: cifar10
  root: /kaggle/input
  resolution: 32
  batch_size: 128
  num_workers: 2
  drop_clusters: []
  drop_fraction: 0.0

model:
  z_dim: 128
  g_ch: 128
  d_ch: 128
  spectral_norm: true

train:
  steps: 200000
  n_critic: 1
  lr_g: 0.0002
  lr_d: 0.0002
  betas: [0.0, 0.9]
  ema:
    enabled: true
    beta: 0.999

log:
  sample_every: 1000
  ckpt_every: 5000
  eval_every: 10000
  n_eval_fake: 10000
```

## `project/configs/cifar10_drop.yml` (sanity-check mode dropping)
```yaml
run:
  name: cifar10_drop_seed0
  seed: 0
  outdir: /kaggle/working/runs

data:
  dataset: cifar10
  root: /kaggle/input
  resolution: 32
  batch_size: 128
  num_workers: 2
  drop_clusters: [7, 12]    # ustawisz po precompute_real (K=50)
  drop_fraction: 1.0

model:
  z_dim: 128
  g_ch: 128
  d_ch: 128
  spectral_norm: true

train:
  steps: 120000
  n_critic: 1
  lr_g: 0.0002
  lr_d: 0.0002
  betas: [0.0, 0.9]
  ema:
    enabled: true
    beta: 0.999

log:
  sample_every: 1000
  ckpt_every: 5000
  eval_every: 10000
  n_eval_fake: 10000
```

## `project/configs/e11_precompute_cifar10.yml`
```yaml
run:
  name: e11_precompute_cifar10
  seed: 0
  outdir: /kaggle/working/e11_cache

data:
  dataset: cifar10
  root: /kaggle/input
  resolution: 32
  batch_size: 256
  num_workers: 2

e11:
  clip_model: ViT-B-32
  clip_pretrained: openai
  K_list: [20, 50, 100]
  kmeans_max_iter: 300
  min_cluster_size: 200
  save_real_clip_features: true   # zapisuje (N,512) float16
```

## `project/configs/e11_eval_cifar10.yml`
```yaml
run:
  name: e11_eval_cifar10
  seed: 0
  outdir: /kaggle/working/e11_results

data:
  dataset: cifar10
  root: /kaggle/input
  resolution: 32
  batch_size: 256
  num_workers: 2

model:
  z_dim: 128
  g_ch: 128

e11:
  cache_dir: /kaggle/working/e11_cache
  clip_model: ViT-B-32
  clip_pretrained: openai

  K_list: [20, 50, 100]
  min_cluster_size: 200

  n_fake: 20000
  assignment: hard   # hard | topM
  topM: 2000         # używane jeśli assignment=topM

  kid:
    enabled: true
    subsets: 50
    subset_size: 1000
    degree: 3
    gamma: null
    coef0: 1.0

  coverage:
    enabled: true
    max_real_per_cluster: 2000
    max_fake_per_cluster: 2000
    threshold_percentile: 10
    chunk: 512

  aggregates: [mean_w, worst, tail10]
```

---

# 3) KOD: utils

## `project/utils/seed.py`
```python
import os, random
import numpy as np
import torch

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # szybciej na Kaggle
```

## `project/utils/io.py`
```python
import os, yaml, json
from dataclasses import dataclass

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_text(s: str, path: str):
    with open(path, "w") as f:
        f.write(s)
```

## `project/utils/logger.py`
```python
import csv, os, time
from .io import ensure_dir

class CSVLogger:
    def __init__(self, path: str, fieldnames):
        ensure_dir(os.path.dirname(path))
        self.path = path
        self.fieldnames = fieldnames
        self._init = not os.path.exists(path)
        if self._init:
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

    def log(self, row: dict):
        row = dict(row)
        row.setdefault("time", time.time())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)
```

---

# 4) KOD: data

## `project/data/datasets.py`
```python
import os
import numpy as np
from typing import Optional, List
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ClusterDropWrapper(Dataset):
    """
    Filtruje próbki na podstawie przypisania klastra (np. z KMeans na CLIP).
    Używane do sanity-check mode dropping.
    """
    def __init__(self, base_ds: Dataset, cluster_labels: np.ndarray,
                 drop_clusters: List[int], drop_fraction: float, seed: int = 0):
        assert len(base_ds) == len(cluster_labels)
        self.base = base_ds
        self.cluster_labels = cluster_labels
        self.drop_clusters = set(drop_clusters)
        self.drop_fraction = float(drop_fraction)

        rng = np.random.RandomState(seed)
        keep = np.ones(len(base_ds), dtype=bool)

        if len(self.drop_clusters) > 0 and self.drop_fraction > 0:
            idx_drop = np.where(np.isin(cluster_labels, list(self.drop_clusters)))[0]
            rng.shuffle(idx_drop)
            n_drop = int(len(idx_drop) * self.drop_fraction)
            keep[idx_drop[:n_drop]] = False

        self.indices = np.where(keep)[0]

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]

def make_transforms(resolution: int):
    if resolution == 32:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])

def get_dataset(name: str, root: str, resolution: int, train: bool = True):
    tfm = make_transforms(resolution)
    name = name.lower()

    if name == "cifar10":
        # Jeśli internet off, Kaggle zwykle ma CIFAR10 jako dataset input.
        # Spróbuj najpierw torchvision download=False (zadziała jeśli już jest).
        try:
            ds = datasets.CIFAR10(root=os.path.join(root, "cifar10"),
                                  train=train, download=False, transform=tfm)
        except Exception:
            ds = datasets.CIFAR10(root=os.path.join(root, "cifar10"),
                                  train=train, download=True, transform=tfm)
        return ds

    if name == "celeba":
        # Zakładamy ImageFolder w /kaggle/input/.../img_align_celeba/img_align_celeba
        # Ustaw root tak, by wskazywał na katalog nadrzędny.
        # W Kaggle często trzeba ręcznie dodać dataset CelebA.
        folder = os.path.join(root, "celeba", "img_align_celeba", "img_align_celeba")
        ds = datasets.ImageFolder(root=folder, transform=tfm)
        return ds

    raise ValueError(f"Unknown dataset: {name}")
```

---

# 5) KOD: models (ResNet GAN + hinge + SN w D)

## `project/models/ema.py`
```python
import torch

class EMA:
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.beta).add_(p.data, alpha=1 - self.beta)

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
```

## `project/models/gan_resnet.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def sn(module, enabled=True):
    return spectral_norm(module) if enabled else module

class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else None

    def forward(self, x):
        h = F.relu(self.bn1(x), inplace=True)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = self.conv2(h)

        s = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.skip is not None:
            s = self.skip(s)
        return h + s

class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, sn_enabled=True, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = sn(nn.Conv2d(in_ch, out_ch, 3, 1, 1), sn_enabled)
        self.conv2 = sn(nn.Conv2d(out_ch, out_ch, 3, 1, 1), sn_enabled)
        self.skip = sn(nn.Conv2d(in_ch, out_ch, 1, 1, 0), sn_enabled) if in_ch != out_ch else None

    def forward(self, x):
        h = F.relu(x, inplace=True)
        h = self.conv1(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        s = x
        if self.downsample:
            s = F.avg_pool2d(s, 2)
        if self.skip is not None:
            s = self.skip(s)
        return h + s

class Generator(nn.Module):
    def __init__(self, z_dim=128, ch=128, resolution=32):
        super().__init__()
        assert resolution in [32, 64]
        self.z_dim = z_dim
        self.resolution = resolution

        self.fc = nn.Linear(z_dim, 4*4*ch*4)
        self.rb1 = ResBlockG(ch*4, ch*2)
        self.rb2 = ResBlockG(ch*2, ch)
        if resolution == 64:
            self.rb3 = ResBlockG(ch, ch)
        self.bn = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        h = self.rb1(h)
        h = self.rb2(h)
        if self.resolution == 64:
            h = self.rb3(h)
        h = F.relu(self.bn(h), inplace=True)
        x = torch.tanh(self.conv_out(h))
        return x

class Discriminator(nn.Module):
    def __init__(self, ch=128, resolution=32, sn_enabled=True):
        super().__init__()
        assert resolution in [32, 64]
        self.resolution = resolution

        self.conv_in = sn(nn.Conv2d(3, ch, 3, 1, 1), sn_enabled)
        self.rb1 = ResBlockD(ch, ch, sn_enabled, downsample=True)
        self.rb2 = ResBlockD(ch, ch*2, sn_enabled, downsample=True)
        self.rb3 = ResBlockD(ch*2, ch*4, sn_enabled, downsample=True)
        if resolution == 64:
            self.rb4 = ResBlockD(ch*4, ch*4, sn_enabled, downsample=True)

        self.lin = sn(nn.Linear(ch*4, 1), sn_enabled)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.rb1(h)
        h = self.rb2(h)
        h = self.rb3(h)
        if self.resolution == 64:
            h = self.rb4(h)
        h = F.relu(h, inplace=True)
        h = h.sum(dim=[2, 3])  # global sum pooling (SNGAN-like)
        out = self.lin(h)
        return out.squeeze(1)
```

---

# 6) Trening + global FID (Inception)

## `project/train.py`
```python
import os, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils.io import load_yaml, ensure_dir, save_json
from utils.seed import seed_all
from utils.logger import CSVLogger
from data.datasets import get_dataset, ClusterDropWrapper
from models.gan_resnet import Generator, Discriminator
from models.ema import EMA
from eval.inception_feat import InceptionFeatureExtractor
from eval.fid_utils import compute_fid_from_stats, StreamingStats

def hinge_d_loss(d_real, d_fake):
    return (F.relu(1. - d_real).mean() + F.relu(1. + d_fake).mean())

def hinge_g_loss(d_fake):
    return (-d_fake).mean()

@torch.no_grad()
def sample_grid(G, z_dim, device, path, n=64):
    z = torch.randn(n, z_dim, device=device)
    x = G(z).cpu()
    grid = make_grid(x, nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid, path)

@torch.no_grad()
def compute_global_fid(G, z_dim, device, real_loader, n_fake=10000):
    # Inception stats for real (streaming, without storing all features)
    feat = InceptionFeatureExtractor(device=device)
    real_stats = StreamingStats(dim=2048)
    for (x, *_) in real_loader:
        x = x.to(device)
        f = feat(x)  # (B,2048)
        real_stats.update(f)

    # fake stats
    fake_stats = StreamingStats(dim=2048)
    bs = real_loader.batch_size
    remaining = n_fake
    while remaining > 0:
        cur = min(bs, remaining)
        z = torch.randn(cur, z_dim, device=device)
        xg = G(z)
        fg = feat(xg)
        fake_stats.update(fg)
        remaining -= cur

    mu_r, sig_r = real_stats.finalize()
    mu_f, sig_f = fake_stats.finalize()
    fid = compute_fid_from_stats(mu_r, sig_r, mu_f, sig_f)
    return float(fid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    run_name = cfg["run"]["name"]
    outdir = os.path.join(cfg["run"]["outdir"], run_name)
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "checkpoints"))
    ensure_dir(os.path.join(outdir, "samples"))

    seed_all(int(cfg["run"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = get_dataset(cfg["data"]["dataset"], cfg["data"]["root"], cfg["data"]["resolution"], train=True)

    # optional: drop clusters (sanity-check)
    drop_clusters = cfg["data"].get("drop_clusters", [])
    drop_fraction = cfg["data"].get("drop_fraction", 0.0)
    if len(drop_clusters) > 0 and drop_fraction > 0:
        # Oczekujemy pliku z klastrami w cache E11:
        # /kaggle/working/e11_cache/cifar10_K50_real_labels.npy
        cache_guess = os.path.join("/kaggle/working/e11_cache", f"{cfg['data']['dataset']}_K50_real_labels.npy")
        if not os.path.exists(cache_guess):
            raise FileNotFoundError(
                f"Brak labels do drop: {cache_guess}. "
                "Najpierw uruchom eval/precompute_real.py dla K=50."
            )
        import numpy as np
        labels = np.load(cache_guess)
        ds = ClusterDropWrapper(ds, labels, drop_clusters, drop_fraction, seed=int(cfg["run"]["seed"]))

    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                    num_workers=cfg["data"]["num_workers"], drop_last=True, pin_memory=True)

    G = Generator(z_dim=cfg["model"]["z_dim"], ch=cfg["model"]["g_ch"], resolution=cfg["data"]["resolution"]).to(device)
    D = Discriminator(ch=cfg["model"]["d_ch"], resolution=cfg["data"]["resolution"],
                      sn_enabled=bool(cfg["model"]["spectral_norm"])).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=cfg["train"]["lr_g"], betas=tuple(cfg["train"]["betas"]))
    optD = torch.optim.Adam(D.parameters(), lr=cfg["train"]["lr_d"], betas=tuple(cfg["train"]["betas"]))

    ema = None
    if cfg["train"]["ema"]["enabled"]:
        ema = EMA(G, beta=float(cfg["train"]["ema"]["beta"]))

    logger = CSVLogger(os.path.join(outdir, "log.csv"),
                       fieldnames=["step", "loss_d", "loss_g", "fid", "time"])

    it = iter(dl)
    steps = int(cfg["train"]["steps"])
    n_critic = int(cfg["train"]["n_critic"])

    t0 = time.time()
    for step in range(1, steps + 1):
        # --- D updates
        for _ in range(n_critic):
            try:
                x_real = next(it)[0]
            except StopIteration:
                it = iter(dl)
                x_real = next(it)[0]
            x_real = x_real.to(device)

            z = torch.randn(x_real.size(0), cfg["model"]["z_dim"], device=device)
            with torch.no_grad():
                x_fake = G(z)

            d_real = D(x_real)
            d_fake = D(x_fake)
            loss_d = hinge_d_loss(d_real, d_fake)

            optD.zero_grad(set_to_none=True)
            loss_d.backward()
            optD.step()

        # --- G update
        z = torch.randn(cfg["data"]["batch_size"], cfg["model"]["z_dim"], device=device)
        x_fake = G(z)
        d_fake = D(x_fake)
        loss_g = hinge_g_loss(d_fake)

        optG.zero_grad(set_to_none=True)
        loss_g.backward()
        optG.step()

        if ema is not None:
            ema.update(G)

        # --- logging / samples / ckpt / eval
        fid_val = None
        if step % int(cfg["log"]["sample_every"]) == 0:
            sample_grid(G, cfg["model"]["z_dim"], device, os.path.join(outdir, "samples", f"grid_{step:06d}.png"))

        if step % int(cfg["log"]["ckpt_every"]) == 0:
            ckpt = {
                "step": step,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "cfg": cfg,
            }
            if ema is not None:
                ckpt["ema"] = ema.shadow
            torch.save(ckpt, os.path.join(outdir, "checkpoints", f"ckpt_{step:06d}.pt"))

        if step % int(cfg["log"]["eval_every"]) == 0:
            # global FID: użyj EMA jeśli jest
            G_eval = G
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in G.state_dict().items()}
                ema.copy_to(G)
                G_eval = G

            real_eval_dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                                      num_workers=cfg["data"]["num_workers"], drop_last=False, pin_memory=True)
            fid_val = compute_global_fid(G_eval, cfg["model"]["z_dim"], device, real_eval_dl,
                                         n_fake=int(cfg["log"]["n_eval_fake"]))

            if backup is not None:
                G.load_state_dict(backup)

        logger.log({
            "step": step,
            "loss_d": float(loss_d.item()),
            "loss_g": float(loss_g.item()),
            "fid": "" if fid_val is None else fid_val,
            "time": time.time() - t0
        })

    save_json({"status": "done", "outdir": outdir}, os.path.join(outdir, "done.json"))

if __name__ == "__main__":
    main()
```

---

# 7) EVAL: Inception features + FID utils (global)

## `project/eval/inception_feat.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionFeatureExtractor(nn.Module):
    """
    Zwraca pool3 2048-d features jak w FID.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        m = inception_v3(weights=weights, transform_input=False, aux_logits=False)
        m.eval()
        # "odcinamy" klasyfikator, bierzemy do Mixed_7c + pooling
        self.m = m.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # x: [-1,1], Bx3xHxW -> 299
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Inception expects normalized to ImageNet stats implicitly in weights transforms,
        # ale w praktyce FID implementacje robią swoje; tutaj trzymamy prosty wariant.
        # Jeśli chcesz zgodność z clean-fid, potem dopasujemy preprocessing.
        m = self.m
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.flatten(1)
        return x
```

## `project/eval/fid_utils.py`
```python
import numpy as np
import torch
from scipy import linalg

class StreamingStats:
    """
    Streaming mean/cov bez zapisywania wszystkich cech:
    trzymamy sum(x) i sum(xx^T).
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.sum = np.zeros((dim,), dtype=np.float64)
        self.sum_sq = np.zeros((dim, dim), dtype=np.float64)

    def update(self, feats: torch.Tensor):
        x = feats.detach().cpu().numpy().astype(np.float64)
        self.n += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sum_sq += x.T @ x

    def finalize(self):
        mu = self.sum / max(self.n, 1)
        cov = self.sum_sq / max(self.n, 1) - np.outer(mu, mu)
        return mu, cov

def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)
```

---

# 8) E11: CLIP features + precompute real + eval per checkpoint

## `project/eval/clip_feat.py`
```python
import torch
import torch.nn.functional as F

def load_openclip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    # preprocess to PIL->tensor; my jednak już mamy tensory [-1,1] z datasetu
    # więc robimy własny preprocess w extract.
    return model

@torch.no_grad()
def clip_image_features(model, x):
    """
    x: Bx3xHxW in [-1,1]
    output: Bxd normalized
    """
    # CLIP expects 224x224, input in [0,1] then normalized internally? open_clip model expects normalized input.
    # Najprościej: skorzystaj z open_clip's preprocess stats:
    # mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    x = (x + 1) / 2
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
    x = (x - mean) / std
    f = model.encode_image(x)
    f = F.normalize(f.float(), dim=1)
    return f
```

## `project/eval/precompute_real.py`
```python
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from utils.io import load_yaml, ensure_dir, save_json
from utils.seed import seed_all
from data.datasets import get_dataset
from eval.clip_feat import load_openclip, clip_image_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    seed_all(int(cfg["run"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = cfg["run"]["outdir"]
    ensure_dir(outdir)

    ds = get_dataset(cfg["data"]["dataset"], cfg["data"]["root"], cfg["data"]["resolution"], train=True)
    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                    num_workers=cfg["data"]["num_workers"], pin_memory=True)

    clip_model = load_openclip(cfg["e11"]["clip_model"], cfg["e11"]["clip_pretrained"], device=device)

    feats = []
    for batch in dl:
        x = batch[0].to(device)
        f = clip_image_features(clip_model, x)
        feats.append(f.cpu())
    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)  # (N, d)
    N, d = feats.shape

    if cfg["e11"].get("save_real_clip_features", True):
        np.save(os.path.join(outdir, f"{cfg['data']['dataset']}_real_clip_feats.npy"),
                feats.astype(np.float16))

    meta = {"N": int(N), "d": int(d), "K_list": cfg["e11"]["K_list"]}
    save_json(meta, os.path.join(outdir, f"{cfg['data']['dataset']}_meta.json"))

    for K in cfg["e11"]["K_list"]:
        km = KMeans(n_clusters=int(K), random_state=int(cfg["run"]["seed"]),
                    n_init="auto", max_iter=int(cfg["e11"]["kmeans_max_iter"]))
        labels = km.fit_predict(feats)
        centroids = km.cluster_centers_.astype(np.float32)

        np.save(os.path.join(outdir, f"{cfg['data']['dataset']}_K{K}_real_labels.npy"), labels.astype(np.int32))
        np.save(os.path.join(outdir, f"{cfg['data']['dataset']}_K{K}_centroids.npy"), centroids.astype(np.float32))

        # cluster sizes
        sizes = np.bincount(labels, minlength=int(K)).astype(int).tolist()
        save_json({"K": int(K), "sizes": sizes},
                  os.path.join(outdir, f"{cfg['data']['dataset']}_K{K}_sizes.json"))

    print("Saved to:", outdir)

if __name__ == "__main__":
    main()
```

## `project/eval/eval_e11.py`
```python
import os, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_distances

from utils.io import load_yaml, ensure_dir, save_json
from utils.seed import seed_all
from models.gan_resnet import Generator
from eval.clip_feat import load_openclip, clip_image_features
from eval.fid_utils import compute_fid_from_stats

def poly_mmd2_unbiased(X, Y, degree=3, gamma=None, coef0=1.0):
    # X: (n,d), Y: (m,d)
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    n = X.shape[0]; m = Y.shape[0]
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = (gamma * (X @ X.T) + coef0) ** degree
    Kyy = (gamma * (Y @ Y.T) + coef0) ** degree
    Kxy = (gamma * (X @ Y.T) + coef0) ** degree
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (n*(n-1)) + Kyy.sum() / (m*(m-1)) - 2.0 * Kxy.mean())

def kid_score(X, Y, subsets=50, subset_size=1000, degree=3, gamma=None, coef0=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n = X.shape[0]; m = Y.shape[0]
    s = min(subset_size, n, m)
    vals = []
    for _ in range(subsets):
        ix = rng.choice(n, size=s, replace=False)
        iy = rng.choice(m, size=s, replace=False)
        vals.append(poly_mmd2_unbiased(X[ix], Y[iy], degree=degree, gamma=gamma, coef0=coef0))
    return float(np.mean(vals)), float(np.std(vals))

def cov_matrix(X):
    X = X.astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(X.shape[0], 1)
    return mu, cov

def tail10(values):
    v = np.array(values, dtype=np.float64)
    k = max(1, int(math.ceil(0.1 * len(v))))
    return float(np.mean(np.sort(v)[-k:]))

def js_divergence(p, q, eps=1e-12):
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a,b): return np.sum(a * np.log(a/b))
    return float(0.5 * kl(p,m) + 0.5 * kl(q,m))

def chunked_cdist_min(A, B, chunk=512):
    # A: (n,d), B: (m,d) numpy float32/64
    A = torch.from_numpy(A).float().cuda()
    B = torch.from_numpy(B).float().cuda()
    mins = []
    for i in range(0, A.size(0), chunk):
        a = A[i:i+chunk]
        d = torch.cdist(a, B)  # (chunk, m)
        mins.append(d.min(dim=1).values.detach().cpu())
    return torch.cat(mins, dim=0).numpy()

def nn_dist_real_real(X, chunk=512):
    # nearest neighbor distance inside set (exclude self)
    X = torch.from_numpy(X).float().cuda()
    mins = []
    for i in range(0, X.size(0), chunk):
        a = X[i:i+chunk]
        d = torch.cdist(a, X)
        # ustawiamy diagonalę tylko dla odpowiednich indeksów
        for j in range(a.size(0)):
            d[j, i+j] = 1e9
        mins.append(d.min(dim=1).values.detach().cpu())
    return torch.cat(mins, dim=0).numpy()

@torch.no_grad()
def generate_fakes(G, z_dim, n, device, batch=256):
    imgs = []
    remaining = n
    while remaining > 0:
        cur = min(batch, remaining)
        z = torch.randn(cur, z_dim, device=device)
        x = G(z)
        imgs.append(x.cpu())
        remaining -= cur
    return torch.cat(imgs, dim=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    seed_all(int(cfg["run"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = cfg["run"]["outdir"]
    ensure_dir(outdir)

    # load G
    ckpt = torch.load(args.ckpt, map_location="cpu")
    G = Generator(z_dim=cfg["model"]["z_dim"], ch=cfg["model"]["g_ch"], resolution=cfg["data"]["resolution"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # CLIP
    clip_model = load_openclip(cfg["e11"]["clip_model"], cfg["e11"]["clip_pretrained"], device=device)

    cache_dir = cfg["e11"]["cache_dir"]
    dsname = cfg["data"]["dataset"]

    # load real feats once
    real_feats = np.load(os.path.join(cache_dir, f"{dsname}_real_clip_feats.npy")).astype(np.float32)  # (N,d)
    N, d = real_feats.shape

    results_all = []

    for K in cfg["e11"]["K_list"]:
        labels_real = np.load(os.path.join(cache_dir, f"{dsname}_K{K}_real_labels.npy")).astype(np.int32)
        centroids = np.load(os.path.join(cache_dir, f"{dsname}_K{K}_centroids.npy")).astype(np.float32)  # (K,d)

        # generate fakes + compute clip feats
        x_fake = generate_fakes(G, cfg["model"]["z_dim"], int(cfg["e11"]["n_fake"]), device, batch=cfg["data"]["batch_size"])
        # move to GPU in batches for CLIP
        feats_fake = []
        for i in range(0, x_fake.size(0), cfg["data"]["batch_size"]):
            xf = x_fake[i:i+cfg["data"]["batch_size"]].to(device)
            ff = clip_image_features(clip_model, xf)
            feats_fake.append(ff.cpu())
        feats_fake = torch.cat(feats_fake, dim=0).numpy().astype(np.float32)

        # assign fake to clusters (cosine distance to centroids)
        # cosine_distances works on CPU; OK for 20k x 100
        dist = cosine_distances(feats_fake, centroids)  # (Nf,K)
        hard = dist.argmin(axis=1).astype(np.int32)

        # hist + JS
        hist_real = np.bincount(labels_real, minlength=int(K)).astype(np.float64)
        hist_fake = np.bincount(hard, minlength=int(K)).astype(np.float64)
        js = js_divergence(hist_real, hist_fake)

        per_cluster = []
        fid_list = []
        kid_list = []
        cov_list = []
        weights = []

        min_cluster = int(cfg["e11"]["min_cluster_size"])

        for c in range(int(K)):
            idx_r = np.where(labels_real == c)[0]
            n_r = idx_r.shape[0]
            if n_r < min_cluster:
                continue

            if cfg["e11"]["assignment"] == "topM":
                M = int(cfg["e11"]["topM"])
                idx_f = np.argsort(dist[:, c])[:min(M, dist.shape[0])]
            else:
                idx_f = np.where(hard == c)[0]

            n_f = idx_f.shape[0]
            if n_f < min_cluster:
                continue

            Xr = real_feats[idx_r]
            Xf = feats_fake[idx_f]

            mu_r, cov_r = cov_matrix(Xr)
            mu_f, cov_f = cov_matrix(Xf)
            fid_c = compute_fid_from_stats(mu_r, cov_r, mu_f, cov_f)
            fid_list.append(fid_c)

            kid_c = None
            kid_std = None
            if cfg["e11"]["kid"]["enabled"]:
                kid_c, kid_std = kid_score(
                    Xr, Xf,
                    subsets=int(cfg["e11"]["kid"]["subsets"]),
                    subset_size=int(cfg["e11"]["kid"]["subset_size"]),
                    degree=int(cfg["e11"]["kid"]["degree"]),
                    gamma=cfg["e11"]["kid"]["gamma"],
                    coef0=float(cfg["e11"]["kid"]["coef0"]),
                    seed=int(cfg["run"]["seed"])
                )
                kid_list.append(kid_c)

            cov_c = None
            if cfg["e11"]["coverage"]["enabled"]:
                mr = int(cfg["e11"]["coverage"]["max_real_per_cluster"])
                mf = int(cfg["e11"]["coverage"]["max_fake_per_cluster"])
                chunk = int(cfg["e11"]["coverage"]["chunk"])
                q = float(cfg["e11"]["coverage"]["threshold_percentile"])

                # subsample for speed
                rng = np.random.RandomState(int(cfg["run"]["seed"]))
                sr = idx_r if n_r <= mr else rng.choice(idx_r, size=mr, replace=False)
                sf = idx_f if n_f <= mf else rng.choice(idx_f, size=mf, replace=False)

                R = real_feats[sr].astype(np.float32)
                Gf = feats_fake[sf].astype(np.float32)

                # threshold from real-real nn distances
                rr = nn_dist_real_real(R, chunk=chunk)
                t = np.percentile(rr, q)

                # coverage: real->fake nn distances
                rg = chunked_cdist_min(R, Gf, chunk=chunk)
                cov_c = float((rg < t).mean())
                cov_list.append(cov_c)

            weights.append(n_r)

            per_cluster.append({
                "K": int(K),
                "cluster": int(c),
                "n_real": int(n_r),
                "n_fake": int(n_f),
                "fid": float(fid_c),
                "kid": None if kid_c is None else float(kid_c),
                "kid_std": None if kid_std is None else float(kid_std),
                "coverage": cov_c,
            })

        # aggregates
        weights = np.array(weights, dtype=np.float64)
        w = weights / max(weights.sum(), 1.0)

        agg = {"K": int(K), "js_hist": float(js)}
        if len(fid_list) > 0:
            fid_arr = np.array(fid_list, dtype=np.float64)
            agg["cluster_fid_mean_w"] = float((w * fid_arr).sum())
            agg["cluster_fid_worst"] = float(fid_arr.max())
            agg["cluster_fid_tail10"] = tail10(fid_arr)
        if len(kid_list) > 0:
            kid_arr = np.array(kid_list, dtype=np.float64)
            agg["cluster_kid_mean_w"] = float((w * kid_arr).sum())
            agg["cluster_kid_worst"] = float(kid_arr.max())
            agg["cluster_kid_tail10"] = tail10(kid_arr)
        if len(cov_list) > 0:
            cov_arr = np.array(cov_list, dtype=np.float64)
            agg["coverage_mean_w"] = float((w * cov_arr).sum())
            agg["coverage_worst"] = float(cov_arr.min())  # worst = min coverage
            agg["coverage_tail10"] = float(np.mean(np.sort(cov_arr)[:max(1, int(np.ceil(0.1*len(cov_arr))))]))

        # save per K
        outK = os.path.join(outdir, f"e11_{dsname}_K{K}")
        ensure_dir(outK)
        save_json({
            "ckpt": args.ckpt,
            "K": int(K),
            "aggregates": agg,
            "per_cluster": per_cluster,
        }, os.path.join(outK, "metrics.json"))

        results_all.append(agg)

    save_json({"ckpt": args.ckpt, "results": results_all},
              os.path.join(outdir, "aggregates_allK.json"))
    print("Saved E11 results to:", outdir)

if __name__ == "__main__":
    main()
```

---

# 9) Raport/CSV pod heatmapy i tabelę (minimalny)

## `project/eval/report.py`
```python
import os, json, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e11_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows_agg = []
    rows_cluster = []

    for root, dirs, files in os.walk(args.e11_dir):
        if "metrics.json" in files:
            p = os.path.join(root, "metrics.json")
            with open(p, "r") as f:
                data = json.load(f)
            agg = data["aggregates"]
            agg = dict(agg)
            agg["ckpt"] = data.get("ckpt", "")
            rows_agg.append(agg)
            for r in data["per_cluster"]:
                r = dict(r)
                r["ckpt"] = data.get("ckpt", "")
                rows_cluster.append(r)

    dfA = pd.DataFrame(rows_agg).sort_values(["K"])
    dfC = pd.DataFrame(rows_cluster).sort_values(["K", "cluster"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dfA.to_csv(args.out.replace(".csv", "_aggregates.csv"), index=False)
    dfC.to_csv(args.out.replace(".csv", "_per_cluster.csv"), index=False)
    print("Saved:", args.out.replace(".csv", "_aggregates.csv"))
    print("Saved:", args.out.replace(".csv", "_per_cluster.csv"))

if __name__ == "__main__":
    main()
```

---

# 10) Prompty do Copilota (agent) – pod tę implementację

1) **Trening**
> Dodaj obsługę resume z checkpointu w train.py (wczytanie step, optG/optD, EMA). Dodaj flagę --resume i automatyczne wykrywanie ostatniego ckpt w katalogu.

2) **E11: multi-ckpt sweep**
> Napisz skrypt eval/sweep_e11.py: iteruje po wszystkich ckpt_*.pt w danym run, uruchamia eval_e11.py dla każdego i zapisuje zbiorczy CSV.

3) **Wizualizacja heatmap**
> Zrób notebook-cell albo skrypt plot_heatmap.py, który z pliku *_per_cluster.csv rysuje heatmapę coverage (cluster vs ckpt_step) i zapisuje PNG.

---

# 11) Co jeszcze trzeba doprecyzować (żebym dopiął „w 100%”)

1) **Dataset**: zostajemy na CIFAR-10 (szybko) czy dorzucasz CelebA64?  
2) Czy chcesz liczyć E11 **w trakcie treningu** (np. co 20k), czy tylko po zakończeniu?  
3) Wolisz przypisanie fake do klastrów jako `hard` czy `topM` (stabilniejsze dla małych klastrów)?

Jeśli odpowiesz na te 3 punkty, dopnę Ci:
- gotowy `scripts/run_all.sh` (sekwencja trening → precompute → eval → report),
- minimalny zestaw runów (baseline + drop + 1 ablacj.),
- oraz tabelę „main” dokładnie pod recenzenta (co wchodzi do paperu, co do suplementu).