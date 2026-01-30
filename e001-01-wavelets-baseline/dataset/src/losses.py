"""
Loss functions and utilities
"""

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
