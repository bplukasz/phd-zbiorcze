"""Data loading for e001-02-r3gan-baseline — reuses logic from e001-01."""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(data_dir: str, img_size: int, batch_size: int,
                   num_workers: int = 4, dataset_name: str = "cifar10",
                   img_channels: int = 3, seed: int = 42) -> DataLoader:
    normalize = transforms.Normalize([0.5] * img_channels, [0.5] * img_channels)
    dataset_name = dataset_name.lower()

    if dataset_name == "celeba":
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.ImageFolder(root=os.path.dirname(data_dir), transform=transform)

    elif dataset_name in ("cifar10", "cifar100"):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])
        download_dir = data_dir if os.path.isdir(data_dir) else f"/tmp/{dataset_name}"
        cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
        dataset = cls(root=download_dir, train=True, download=True, transform=transform)

    elif dataset_name in ("mnist", "fashion_mnist"):
        steps = [transforms.Resize(img_size), transforms.ToTensor()]
        if img_channels == 3:
            steps.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        steps.append(normalize)
        transform = transforms.Compose(steps)
        download_dir = data_dir if os.path.isdir(data_dir) else f"/tmp/{dataset_name}"
        cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        dataset = cls(root=download_dir, train=True, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2 ** 32
        import random; random.seed(worker_seed)
        try:
            import numpy as np; np.random.seed(worker_seed)
        except ImportError:
            pass
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    print(f"Dataset: {dataset_name}, size: {len(dataset)}, img_size: {img_size}, ch: {img_channels}")
    return loader

