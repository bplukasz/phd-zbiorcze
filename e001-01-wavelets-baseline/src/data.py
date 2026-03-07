"""
Data loading and preprocessing
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(data_dir: str, img_size: int, batch_size: int,
                   num_workers: int = 4, dataset_name: str = "celeba",
                   img_channels: int = 3,
                   seed: int = 42) -> DataLoader:
    """
    Tworzy DataLoader dla wybranego datasetu.

    Args:
        data_dir: Ścieżka do danych lub katalog do pobrania
        img_size: Rozmiar obrazu
        batch_size: Rozmiar batcha
        num_workers: Liczba workerów
        dataset_name: "celeba", "cifar10", "cifar100", "mnist", "fashion_mnist"
        img_channels: Liczba kanałów (3 dla RGB, 1 dla grayscale)
        seed: Seed do shuffle/workerów (dla powtarzalności)

    Returns:
        DataLoader
    """
    # Normalizacja
    normalize = transforms.Normalize([0.5] * img_channels, [0.5] * img_channels)

    # Wybór datasetu
    dataset_name = dataset_name.lower()

    if dataset_name == "celeba":
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.ImageFolder(root=os.path.dirname(data_dir), transform=transform)

    elif dataset_name in ["cifar10", "cifar100"]:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])
        download_dir = data_dir if os.path.isdir(data_dir) else f"/tmp/{dataset_name}"
        dataset_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
        dataset = dataset_class(root=download_dir, train=True, download=True, transform=transform)

    elif dataset_name in ["mnist", "fashion_mnist"]:
        base_transforms = [transforms.Resize(img_size), transforms.ToTensor()]

        # Konwertuj grayscale do RGB jeśli potrzeba
        if img_channels == 3:
            base_transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        base_transforms.append(normalize)
        transform = transforms.Compose(base_transforms)

        download_dir = data_dir if os.path.isdir(data_dir) else f"/tmp/{dataset_name}"
        dataset_class = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        dataset = dataset_class(root=download_dir, train=True, download=True, transform=transform)

    else:
        raise ValueError(f"Nieznany dataset: {dataset_name}. "
                        f"Dostępne: celeba, cifar10, cifar100, mnist, fashion_mnist")

    def _seed_worker(worker_id: int) -> None:
        # Każdy worker dostaje inny seed, ale deterministycznie od bazowego seeda.
        worker_seed = (seed + worker_id) % 2**32
        try:
            import random
            random.seed(worker_seed)
        except Exception:
            pass
        try:
            import numpy as np  # type: ignore
            np.random.seed(worker_seed)
        except Exception:
            pass
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
        worker_init_fn=_seed_worker if num_workers and num_workers > 0 else None,
        generator=g,
    )

    print(f"Dataset: {dataset_name}, rozmiar: {len(dataset)}, "
          f"img_size: {img_size}, channels: {img_channels}")

    return dataloader
