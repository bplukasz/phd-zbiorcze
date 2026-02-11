"""Common GAN loss functions and utilities.

Standardowe funkcje strat używane w treningu GANów.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def hinge_loss_d(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss dla Discriminatora.

    Args:
        real_logits: Logity dla prawdziwych obrazów
        fake_logits: Logity dla wygenerowanych obrazów

    Returns:
        Wartość straty
    """
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def hinge_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss dla Generatora.

    Args:
        fake_logits: Logity dla wygenerowanych obrazów

    Returns:
        Wartość straty
    """
    return -fake_logits.mean()


def r1_penalty(D: nn.Module, real_imgs: torch.Tensor) -> torch.Tensor:
    """
    R1 gradient penalty (Mescheder et al., 2018).

    Karze Discriminator za zbyt duże gradienty względem danych wejściowych.
    Stabilizuje trening i zapobiega mode collapse.

    Args:
        D: Model Discriminatora
        real_imgs: Prawdziwe obrazy

    Returns:
        Gradient penalty (scalar)
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
    """
    Oblicza normę gradientów modelu.

    Przydatne do monitorowania treningu i diagnozowania problemów.

    Args:
        model: Model PyTorch

    Returns:
        L2 norma wszystkich gradientów
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def wasserstein_loss_d(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein loss dla Discriminatora (WGAN).

    Args:
        real_logits: Logity dla prawdziwych obrazów
        fake_logits: Logity dla wygenerowanych obrazów

    Returns:
        Wartość straty
    """
    return fake_logits.mean() - real_logits.mean()


def wasserstein_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein loss dla Generatora (WGAN).

    Args:
        fake_logits: Logity dla wygenerowanych obrazów

    Returns:
        Wartość straty
    """
    return -fake_logits.mean()


def non_saturating_loss_d(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating loss dla Discriminatora (standardowy GAN).

    Args:
        real_logits: Logity dla prawdziwych obrazów
        fake_logits: Logity dla wygenerowanych obrazów

    Returns:
        Wartość straty
    """
    loss_real = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    )
    loss_fake = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    return loss_real + loss_fake


def non_saturating_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating loss dla Generatora (standardowy GAN).

    Args:
        fake_logits: Logity dla wygenerowanych obrazów

    Returns:
        Wartość straty
    """
    return F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )

