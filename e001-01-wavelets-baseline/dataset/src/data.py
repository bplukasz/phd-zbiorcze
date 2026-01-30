"""
Data loading and preprocessing
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(data_dir: str, img_size: int, batch_size: int,
                   num_workers: int = 4, dataset_name: str = "celeba",
                   img_channels: int = 3) -> DataLoader:
    """
    Tworzy DataLoader dla wybranego datasetu.

    Args:
        data_dir: Ścieżka do danych lub katalog do pobrania
        img_size: Rozmiar obrazu
        batch_size: Rozmiar batcha
        num_workers: Liczba workerów
        dataset_name: "celeba", "cifar10", "cifar100", "mnist", "fashion_mnist"
        img_channels: Liczba kanałów (3 dla RGB, 1 dla grayscale)

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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset: {dataset_name}, rozmiar: {len(dataset)}, "
          f"img_size: {img_size}, channels: {img_channels}")

    return dataloader
