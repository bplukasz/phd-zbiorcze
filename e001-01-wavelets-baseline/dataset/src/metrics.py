"""
Evaluation metrics: FID, KID, RPSE, WBED
"""

import os
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# ============================================================================
# Sample Generation and Export
# ============================================================================

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


def generate_samples(G: nn.Module, n_samples: int, z_dim: int, batch_size: int,
                     device: torch.device, out_dir: str) -> str:
    """Generuje próbki i zapisuje do katalogu."""
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


# ============================================================================
# FID and KID
# ============================================================================

def compute_fid_kid(real_dir: str, fake_dir: str, fid_samples: int = 10000) -> Dict[str, float]:
    """Oblicza FID i KID używając torch-fidelity."""
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
        print(f"    ✗ Błąd: {e}")
        return {'fid': float('nan'), 'kid': float('nan')}


# ============================================================================
# Radial Power Spectrum (FFT-based)
# ============================================================================

def compute_radial_power_spectrum(imgs: torch.Tensor, num_bins: Optional[int] = None) -> torch.Tensor:
    """
    Oblicza radialny profil widma mocy z FFT dla batcha obrazów.

    Uśrednia po kanałach i batchu. Wykonuje radial binning na podstawie
    odległości od centrum w przestrzeni częstotliwości.

    Args:
        imgs: Tensor BxCxHxW z obrazami (znormalizowane do [-1,1] lub [0,1])
        num_bins: Liczba binów radialnych (domyślnie min(H,W)//2)

    Returns:
        Tensor 1D z radialnym profilem mocy [num_bins]
    """
    B, C, H, W = imgs.shape
    device = imgs.device

    if num_bins is None:
        num_bins = min(H, W) // 2

    # Oblicz 2D FFT dla każdego obrazu/kanału
    # Shift DC do centrum
    fft = torch.fft.fft2(imgs, norm='ortho')
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Power spectrum (magnitude squared)
    power = (fft_shifted.real ** 2 + fft_shifted.imag ** 2)

    # Uśrednij po batchu i kanałach -> HxW
    power_mean = power.mean(dim=(0, 1))

    # Stwórz mapę odległości od centrum
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=device, dtype=torch.float32) - cy
    x = torch.arange(W, device=device, dtype=torch.float32) - cx
    Y, X = torch.meshgrid(y, x, indexing='ij')
    radius = torch.sqrt(X ** 2 + Y ** 2)

    # Radial binning
    max_radius = min(cy, cx)
    bin_edges = torch.linspace(0, max_radius, num_bins + 1, device=device)

    profile = torch.zeros(num_bins, device=device)
    counts = torch.zeros(num_bins, device=device)

    for i in range(num_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.sum() > 0:
            profile[i] = power_mean[mask].mean()
            counts[i] = mask.sum().float()

    # Uzupełnij puste biny interpolacją liniową
    for i in range(num_bins):
        if counts[i] == 0:
            # Znajdź najbliższe niepuste biny
            left_idx, right_idx = i - 1, i + 1
            while left_idx >= 0 and counts[left_idx] == 0:
                left_idx -= 1
            while right_idx < num_bins and counts[right_idx] == 0:
                right_idx += 1

            if left_idx >= 0 and right_idx < num_bins:
                # Interpolacja liniowa
                t = (i - left_idx) / (right_idx - left_idx)
                profile[i] = profile[left_idx] * (1 - t) + profile[right_idx] * t
            elif left_idx >= 0:
                profile[i] = profile[left_idx]
            elif right_idx < num_bins:
                profile[i] = profile[right_idx]

    return profile


def compute_rpse(real_profile: torch.Tensor, fake_profile: torch.Tensor) -> float:
    """
    Oblicza Radial Power Spectrum Error (RPSE) jako L2 między profilami.

    Args:
        real_profile: Radialny profil mocy dla real images [num_bins]
        fake_profile: Radialny profil mocy dla fake images [num_bins]

    Returns:
        RPSE (L2 distance)
    """
    # Normalizuj profile przed porównaniem
    real_norm = real_profile / (real_profile.sum() + 1e-8)
    fake_norm = fake_profile / (fake_profile.sum() + 1e-8)

    rpse = torch.sqrt(((real_norm - fake_norm) ** 2).sum()).item()
    return rpse


def load_images_from_folder(folder: str, max_images: int = 1000,
                            img_size: Optional[int] = None,
                            device: str = 'cpu') -> torch.Tensor:
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
    from PIL import Image

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


def compute_rpse_from_folders(real_folder: str, fake_folder: str,
                               max_images: int = 1000,
                               num_bins: Optional[int] = None,
                               img_size: Optional[int] = None,
                               device: str = 'cpu') -> Dict[str, Any]:
    """
    Oblicza RPSE między obrazami z dwóch folderów.

    Args:
        real_folder: Folder z prawdziwymi obrazami
        fake_folder: Folder z wygenerowanymi obrazami
        max_images: Maksymalna liczba obrazów do wczytania z każdego folderu
        num_bins: Liczba binów radialnych
        img_size: Opcjonalny resize obrazów
        device: Urządzenie

    Returns:
        Dict z RPSE i profilami
    """
    print(f"    Wczytywanie obrazów real z {real_folder}...")
    real_imgs = load_images_from_folder(real_folder, max_images, img_size, device)

    print(f"    Wczytywanie obrazów fake z {fake_folder}...")
    fake_imgs = load_images_from_folder(fake_folder, max_images, img_size, device)

    print(f"    Obliczanie radialnych profili mocy (real: {real_imgs.shape[0]}, fake: {fake_imgs.shape[0]})...")
    real_profile = compute_radial_power_spectrum(real_imgs, num_bins)
    fake_profile = compute_radial_power_spectrum(fake_imgs, num_bins)

    rpse = compute_rpse(real_profile, fake_profile)

    print(f"    ✓ RPSE: {rpse:.6f}")

    return {
        'rpse': rpse,
        'real_profile': real_profile.cpu().numpy(),
        'fake_profile': fake_profile.cpu().numpy(),
    }


# ============================================================================
# Wavelet Band Energy Distance (WBED)
# ============================================================================

def compute_wavelet_band_energies(imgs: torch.Tensor, wavelet: str = 'haar') -> Dict[str, torch.Tensor]:
    """
    Oblicza energie pasm DWT dla batcha obrazów.

    Energia pasma = mean(coeff^2) per band per image.

    Args:
        imgs: Tensor BxCxHxW z obrazami
        wavelet: Nazwa falki ('haar' lub 'db2')

    Returns:
        Dict z energiami per band: {'LL': [B], 'LH': [B], 'HL': [B], 'HH': [B]}
    """
    from .wavelets import DWT2D, split_subbands

    B, C, H, W = imgs.shape
    device = imgs.device

    # DWT
    dwt = DWT2D(wavelet=wavelet).to(device)
    coeffs = dwt(imgs)

    # Rozdziel na podpasma
    LL, LH, HL, HH = split_subbands(coeffs, C)

    # Oblicz energie (mean(coeff^2) per image, uśrednione po kanałach i przestrzennie)
    energies = {
        'LL': (LL ** 2).mean(dim=(1, 2, 3)),  # [B]
        'LH': (LH ** 2).mean(dim=(1, 2, 3)),  # [B]
        'HL': (HL ** 2).mean(dim=(1, 2, 3)),  # [B]
        'HH': (HH ** 2).mean(dim=(1, 2, 3)),  # [B]
    }

    return energies


def compute_wbed(real_energies: Dict[str, torch.Tensor],
                 fake_energies: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Oblicza Wavelet Band Energy Distance między rozkładami energii.

    Dystans = |mean_real - mean_fake| + |std_real - std_fake| per band.
    Całkowity WBED = suma dystansów po wszystkich pasmach.

    Args:
        real_energies: Energie pasm dla real images
        fake_energies: Energie pasm dla fake images

    Returns:
        Dict z dystansami per band i całkowitym WBED
    """
    bands = ['LL', 'LH', 'HL', 'HH']
    distances = {}
    total_wbed = 0.0

    for band in bands:
        real_e = real_energies[band]
        fake_e = fake_energies[band]

        # Statystyki
        real_mean = real_e.mean().item()
        real_std = real_e.std().item()
        fake_mean = fake_e.mean().item()
        fake_std = fake_e.std().item()

        # Dystans
        mean_diff = abs(real_mean - fake_mean)
        std_diff = abs(real_std - fake_std)
        band_dist = mean_diff + std_diff

        distances[f'{band}_mean_diff'] = mean_diff
        distances[f'{band}_std_diff'] = std_diff
        distances[f'{band}_dist'] = band_dist

        total_wbed += band_dist

    distances['wbed_total'] = total_wbed

    return distances


def compute_wbed_from_folders(real_folder: str, fake_folder: str,
                               max_images: int = 1000,
                               wavelet: str = 'haar',
                               img_size: Optional[int] = None,
                               device: str = 'cpu') -> Dict[str, Any]:
    """
    Oblicza WBED między obrazami z dwóch folderów.

    Args:
        real_folder: Folder z prawdziwymi obrazami
        fake_folder: Folder z wygenerowanymi obrazami
        max_images: Maksymalna liczba obrazów do wczytania
        wavelet: Nazwa falki ('haar' lub 'db2')
        img_size: Opcjonalny resize obrazów
        device: Urządzenie

    Returns:
        Dict z WBED i szczegółami per band
    """
    print(f"    Wczytywanie obrazów real z {real_folder}...")
    real_imgs = load_images_from_folder(real_folder, max_images, img_size, device)

    print(f"    Wczytywanie obrazów fake z {fake_folder}...")
    fake_imgs = load_images_from_folder(fake_folder, max_images, img_size, device)

    print(f"    Obliczanie energii pasm DWT (wavelet={wavelet})...")
    real_energies = compute_wavelet_band_energies(real_imgs, wavelet)
    fake_energies = compute_wavelet_band_energies(fake_imgs, wavelet)

    distances = compute_wbed(real_energies, fake_energies)

    print(f"    ✓ WBED total: {distances['wbed_total']:.6f}")
    for band in ['LL', 'LH', 'HL', 'HH']:
        print(f"      {band}: dist={distances[f'{band}_dist']:.6f} "
              f"(mean_diff={distances[f'{band}_mean_diff']:.6f}, "
              f"std_diff={distances[f'{band}_std_diff']:.6f})")

    # Dodaj surowe energie do wyników
    distances['real_energies'] = {k: v.cpu().numpy() for k, v in real_energies.items()}
    distances['fake_energies'] = {k: v.cpu().numpy() for k, v in fake_energies.items()}

    return distances


def compute_all_spectral_metrics(real_folder: str, fake_folder: str,
                                  max_images: int = 1000,
                                  num_bins: Optional[int] = None,
                                  wavelet: str = 'haar',
                                  img_size: Optional[int] = None,
                                  device: str = 'cpu') -> Dict[str, Any]:
    """
    Oblicza wszystkie metryki spektralne (RPSE + WBED) między obrazami z folderów.

    Args:
        real_folder: Folder z prawdziwymi obrazami
        fake_folder: Folder z wygenerowanymi obrazami
        max_images: Maksymalna liczba obrazów do wczytania
        num_bins: Liczba binów radialnych dla RPSE
        wavelet: Nazwa falki dla WBED ('haar' lub 'db2')
        img_size: Opcjonalny resize obrazów
        device: Urządzenie

    Returns:
        Dict z wszystkimi metrykami
    """
    print(f"\n{'='*60}")
    print("METRYKI SPEKTRALNE")
    print(f"{'='*60}")

    # Wczytaj obrazy raz
    print(f"Wczytywanie obrazów...")
    real_imgs = load_images_from_folder(real_folder, max_images, img_size, device)
    fake_imgs = load_images_from_folder(fake_folder, max_images, img_size, device)
    print(f"  Real: {real_imgs.shape}, Fake: {fake_imgs.shape}")

    results = {}

    # RPSE
    print(f"\n[1/2] Radial Power Spectrum Error (RPSE)...")
    real_profile = compute_radial_power_spectrum(real_imgs, num_bins)
    fake_profile = compute_radial_power_spectrum(fake_imgs, num_bins)
    rpse = compute_rpse(real_profile, fake_profile)
    results['rpse'] = rpse
    results['real_rps_profile'] = real_profile.cpu().numpy()
    results['fake_rps_profile'] = fake_profile.cpu().numpy()
    print(f"  RPSE: {rpse:.6f}")

    # WBED
    print(f"\n[2/2] Wavelet Band Energy Distance (WBED, wavelet={wavelet})...")
    real_energies = compute_wavelet_band_energies(real_imgs, wavelet)
    fake_energies = compute_wavelet_band_energies(fake_imgs, wavelet)
    wbed_results = compute_wbed(real_energies, fake_energies)

    results['wbed_total'] = wbed_results['wbed_total']
    for key, value in wbed_results.items():
        if isinstance(value, (int, float)):
            results[f'wbed_{key}'] = value

    results['real_energies'] = {k: v.cpu().numpy() for k, v in real_energies.items()}
    results['fake_energies'] = {k: v.cpu().numpy() for k, v in fake_energies.items()}

    print(f"  WBED total: {wbed_results['wbed_total']:.6f}")

    print(f"\n{'='*60}")
    print("PODSUMOWANIE METRYK SPEKTRALNYCH")
    print(f"{'='*60}")
    print(f"  RPSE:       {rpse:.6f}")
    print(f"  WBED total: {wbed_results['wbed_total']:.6f}")

    return results
