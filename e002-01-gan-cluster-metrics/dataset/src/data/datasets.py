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

