"""
Loss functions and utilities
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def hinge_loss_d(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss dla Discriminatora."""
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def hinge_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss dla Generatora."""
    return -fake_logits.mean()


def r1_penalty(D: nn.Module, real_imgs: torch.Tensor) -> torch.Tensor:
    """
    R1 gradient penalty (Mescheder et al., 2018).
    Karze Discriminator za zbyt duże gradienty względem danych wejściowych.
    Stabilizuje trening i zapobiega mode collapse.

    Args:
        D: Discriminator model
        real_imgs: Prawdziwe obrazy

    Returns:
        Gradient penalty scalar
    """
    real_imgs = real_imgs.detach().requires_grad_(True)
    real_logits = D(real_imgs)

    # Backward przez sumę logits (nie mean, bo chcemy gradienty dla każdej próbki)
    grad_outputs = torch.ones_like(real_logits)
    gradients = torch.autograd.grad(
        outputs=real_logits,
        inputs=real_imgs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # L2 norm of gradients
    penalty = (gradients ** 2).sum([1, 2, 3]).mean()
    return penalty


def compute_grad_norm(model: nn.Module) -> float:
    """Oblicza normę gradientów modelu."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def wavereg_batch_stats(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zwraca (mean, std) po batchu dla tensora [B]."""
    mu = x.mean()
    # stabilniej użyć population variance (unbiased=False)
    sigma = x.var(unbiased=False).add(eps).sqrt()
    return mu, sigma


def wavelet_energy_matching_loss(
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    *,
    wavelet: str = 'haar',
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Regularizer: wavelet-energy matching.

    Dla real i fake liczymy per-image energie pasm (mean(coeff^2) dla LL/LH/HL/HH),
    następnie mean i std po batchu i karę L2:
        ||mu_real-mu_fake||^2 + ||std_real-std_fake||^2
    (suma po pasmach).

    Zwraca:
        (loss, logs)
        logs zawiera wavereg_loss oraz statystyki mu/std dla real/fake.

    Uwaga: zależność od DWT jest w `metrics.compute_wavelet_band_energies`, więc import jest lokalny.
    """
    from .metrics import compute_wavelet_band_energies

    bands = ['LL', 'LH', 'HL', 'HH']

    real_e = compute_wavelet_band_energies(real_imgs, wavelet=wavelet)
    fake_e = compute_wavelet_band_energies(fake_imgs, wavelet=wavelet)

    loss = real_imgs.new_tensor(0.0)
    logs: Dict[str, float] = {}

    for b in bands:
        mu_r, std_r = wavereg_batch_stats(real_e[b], eps=eps)
        mu_f, std_f = wavereg_batch_stats(fake_e[b], eps=eps)

        mu_diff = mu_r - mu_f
        std_diff = std_r - std_f

        loss = loss + (mu_diff ** 2) + (std_diff ** 2)

        logs[f'wavereg_mu_real_{b}'] = float(mu_r.detach().cpu())
        logs[f'wavereg_std_real_{b}'] = float(std_r.detach().cpu())
        logs[f'wavereg_mu_fake_{b}'] = float(mu_f.detach().cpu())
        logs[f'wavereg_std_fake_{b}'] = float(std_f.detach().cpu())
        logs[f'wavereg_mu_diff_{b}'] = float(mu_diff.detach().cpu())
        logs[f'wavereg_std_diff_{b}'] = float(std_diff.detach().cpu())

    logs['wavereg_loss'] = float(loss.detach().cpu())
    return loss, logs


def fft_energy_matching_loss(
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    *,
    num_bins: int = 16,
    downsample_to: int = 64,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Baseline regularization: Fourier (FFT) energy matching.

    Schemat analogiczny do wavelet reg, ale zamiast pasm DWT używa energii w radialnych binach
    widma mocy FFT (per-image). Następnie dla każdego binu liczy mean/std po batchu i karę L2:
        sum_i (mu_real[i]-mu_fake[i])^2 + (std_real[i]-std_fake[i])^2

    Args:
        real_imgs, fake_imgs: [B,C,H,W]
        num_bins: liczba binów radialnych
        downsample_to: rozmiar (H=W) po downsample przed FFT dla kontroli kosztu
        eps: stabilizacja

    Returns:
        (loss, logs) gdzie logs zawiera fftreg_loss + statystyki per bin
    """
    from .metrics import compute_fft_radial_bin_energies_per_image

    # [B,K]
    real_e = compute_fft_radial_bin_energies_per_image(
        real_imgs, num_bins=num_bins, downsample_to=downsample_to, eps=eps
    )
    fake_e = compute_fft_radial_bin_energies_per_image(
        fake_imgs, num_bins=num_bins, downsample_to=downsample_to, eps=eps
    )

    loss = real_imgs.new_tensor(0.0)
    logs: Dict[str, float] = {}

    for i in range(num_bins):
        mu_r, std_r = wavereg_batch_stats(real_e[:, i], eps=eps)
        mu_f, std_f = wavereg_batch_stats(fake_e[:, i], eps=eps)

        mu_diff = mu_r - mu_f
        std_diff = std_r - std_f

        loss = loss + (mu_diff ** 2) + (std_diff ** 2)

        # log (opcjonalnie) - per bin
        logs[f'fftreg_mu_real_bin{i:02d}'] = float(mu_r.detach().cpu())
        logs[f'fftreg_std_real_bin{i:02d}'] = float(std_r.detach().cpu())
        logs[f'fftreg_mu_fake_bin{i:02d}'] = float(mu_f.detach().cpu())
        logs[f'fftreg_std_fake_bin{i:02d}'] = float(std_f.detach().cpu())

    logs['fftreg_loss'] = float(loss.detach().cpu())
    return loss, logs

