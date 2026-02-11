"""Common evaluation metrics for generative models.

Uniwersalne metryki ewaluacyjne (FID, KID, utilities).
"""

import os
import math
from typing import Dict, Optional
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


def export_real_images(dataloader: DataLoader, n_samples: int, out_dir: str) -> str:
    """
    Eksportuje prawdziwe obrazy z DataLoadera do osobnych plików PNG.

    Potrzebne dla torch-fidelity gdy dataset nie jest w formacie ImageFolder.

    Args:
        dataloader: DataLoader z prawdziwymi danymi
        n_samples: Liczba próbek do wyeksportowania
        out_dir: Katalog wyjściowy

    Returns:
        Ścieżka do katalogu z obrazami
    """
    os.makedirs(out_dir, exist_ok=True)

    # Sprawdź czy już eksportowano
    existing_files = [f for f in os.listdir(out_dir) if f.endswith('.png')]
    if len(existing_files) >= n_samples:
        print(f"    ✓ Real samples już wyeksportowane ({len(existing_files)} plików)")
        return out_dir

    print(f"    Eksportowanie {n_samples} real samples do {out_dir}...")
    idx = 0

    for imgs, _ in dataloader:
        if idx >= n_samples:
            break
        for img in imgs:
            if idx >= n_samples:
                break
            # Denormalizuj [-1, 1] -> [0, 1]
            img = (img + 1) / 2
            save_image(img, os.path.join(out_dir, f'{idx:06d}.png'))
            idx += 1

    print(f"    ✓ Wyeksportowano {idx} real samples")
    return out_dir


def generate_samples(
    G: nn.Module,
    n_samples: int,
    z_dim: int,
    batch_size: int,
    device: torch.device,
    out_dir: str,
) -> str:
    """
    Generuje próbki z modelu i zapisuje do katalogu.

    Args:
        G: Model Generatora
        n_samples: Liczba próbek do wygenerowania
        z_dim: Wymiar przestrzeni latentnej
        batch_size: Rozmiar batcha
        device: Device (cpu/cuda)
        out_dir: Katalog wyjściowy

    Returns:
        Ścieżka do katalogu z wygenerowanymi obrazami
    """
    os.makedirs(out_dir, exist_ok=True)
    G.eval()

    n_batches = math.ceil(n_samples / batch_size)
    idx = 0

    print(f"    Generowanie {n_samples} próbek w {n_batches} partiach...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            if batch_idx % 20 == 0 or batch_idx == n_batches - 1:
                print(f"    -> {idx}/{n_samples} próbek ({idx/n_samples*100:.1f}%)")

            z = torch.randn(min(batch_size, n_samples - idx), z_dim, device=device)
            imgs = G(z)
            imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]

            for img in imgs:
                if idx >= n_samples:
                    break
                save_image(img, os.path.join(out_dir, f'{idx:06d}.png'))
                idx += 1

    print(f"    ✓ Wygenerowano wszystkie {n_samples} próbek")

    return out_dir


def compute_fid_kid(real_dir: str, fake_dir: str, fid_samples: int = 10000) -> Dict[str, float]:
    """
    Oblicza FID i KID używając torch-fidelity.

    Args:
        real_dir: Katalog z prawdziwymi obrazami
        fake_dir: Katalog z wygenerowanymi obrazami
        fid_samples: Liczba próbek do użycia

    Returns:
        Dict z metrykami {'fid': float, 'kid': float}
    """
    try:
        from torch_fidelity import calculate_metrics

        print(f"    Obliczanie FID/KID dla {fid_samples} próbek (może potrwać 5-15 min)...")

        metrics = calculate_metrics(
            input1=fake_dir,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            kid=True,
            kid_subset_size=min(1000, fid_samples),
            verbose=False,
        )

        return {
            'fid': metrics.get('frechet_inception_distance', float('nan')),
            'kid': metrics.get('kernel_inception_distance_mean', float('nan')) * 1000,
        }
    except Exception as e:
        print(f"    ✗ Błąd podczas obliczania FID/KID: {e}")
        return {'fid': float('nan'), 'kid': float('nan')}


def load_images_from_folder(
    folder: str,
    max_images: int = 1000,
    img_size: Optional[int] = None,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Wczytuje obrazy z folderu do tensora PyTorch.

    Args:
        folder: Ścieżka do folderu z obrazami
        max_images: Maksymalna liczba obrazów do wczytania
        img_size: Opcjonalny resize (jeśli None, użyje oryginalnego rozmiaru)
        device: Urządzenie docelowe

    Returns:
        Tensor BxCxHxW z obrazami w zakresie [0, 1]
    """
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    image_files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ])[:max_images]

    if len(image_files) == 0:
        raise ValueError(f"Brak obrazów w folderze: {folder}")

    images = []
    transform_list = []
    if img_size is not None:
        transform_list.append(transforms.Resize((img_size, img_size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)

    batch = torch.stack(images, dim=0).to(device)
    return batch

