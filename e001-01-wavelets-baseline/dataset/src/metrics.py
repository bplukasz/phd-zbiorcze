"""
Evaluation metrics: FID, KID, RPSE, WBED

Funkcje specyficzne dla eksperymentu wavelets-baseline (metryki spektralne).
Funkcje ogólne (FID, KID, utilities) przeniesione do shared.utils.metrics
"""

import os
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Import ogólnych funkcji z shared-lib
from utils import (
    export_real_images,
    generate_samples,
    compute_fid_kid,
    load_images_from_folder,
)

# Re-export dla zachowania kompatybilności
__all__ = [
    'export_real_images',
    'generate_samples',
    'compute_fid_kid',
    'load_images_from_folder',
    'compute_radial_power_spectrum',
    'compute_rpse',
    'compute_rpse_from_folders',
    'compute_wavelet_band_energies',
    'compute_wbed',
    'compute_wbed_from_folders',
    'compute_all_spectral_metrics',
    'compute_fft_radial_bin_energies_per_image',
]


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


def compute_fft_radial_bin_energies_per_image(
    imgs: torch.Tensor,
    num_bins: int = 16,
    downsample_to: Optional[int] = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-image radial bin energies z FFT power spectrum.

    Dla każdego obrazu liczy widmo mocy (|FFT|^2), uśrednia po kanałach,
    wykonuje radial binning i zwraca energie w binach dla każdego elementu batcha.

    Args:
        imgs: Tensor [B,C,H,W] (zakres [-1,1] lub [0,1])
        num_bins: liczba binów radialnych
        downsample_to: jeśli nie-None, to obraz jest bilinearnie zmniejszany do downsample_to x downsample_to
                       (żeby koszt FFT był porównywalny do wavelet reg)
        eps: mała stała stabilizująca (np. przy normalizacji)

    Returns:
        Tensor [B, num_bins] z energiami w binach (mean power w binie)
    """
    import torch.nn.functional as F

    B, C, H, W = imgs.shape
    device = imgs.device

    x = imgs
    if downsample_to is not None and (H != downsample_to or W != downsample_to):
        x = F.interpolate(x, size=(downsample_to, downsample_to), mode='bilinear', align_corners=False)

    # FFT per-image/per-channel
    fft = torch.fft.fft2(x, norm='ortho')
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = (fft_shifted.real ** 2 + fft_shifted.imag ** 2)  # [B,C,h,w]

    # uśrednij po kanałach -> [B,h,w]
    power = power.mean(dim=1)

    h, w = power.shape[-2:]
    cy, cx = h // 2, w // 2

    # radius map [h,w]
    yy = torch.arange(h, device=device, dtype=torch.float32) - cy
    xx = torch.arange(w, device=device, dtype=torch.float32) - cx
    Y, X = torch.meshgrid(yy, xx, indexing='ij')
    radius = torch.sqrt(X ** 2 + Y ** 2)  # [h,w]

    max_radius = min(cy, cx)
    bin_edges = torch.linspace(0, max_radius, num_bins + 1, device=device)

    # policz mean power per bin per image
    energies = torch.zeros((B, num_bins), device=device, dtype=power.dtype)

    # (opcjonalnie) normalizacja, żeby kara nie zależała od globalnej skali
    # tu: normalizujemy sumą energii per image
    power_flat = power.reshape(B, -1)
    norm = power_flat.sum(dim=1, keepdim=True).clamp_min(eps)
    power = power / norm.view(B, 1, 1)

    for i in range(num_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.any():
            vals = power[:, mask]  # [B, n_pix]
            energies[:, i] = vals.mean(dim=1)

    return energies

