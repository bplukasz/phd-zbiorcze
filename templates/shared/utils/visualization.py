"""Moduł wizualizacji dla eksperymentów."""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import torch

# Optional: notebook live display
try:
    from IPython.display import clear_output, display
    _HAS_IPYTHON = True
except Exception:
    _HAS_IPYTHON = False

# Optional: torchvision
try:
    from torchvision.utils import make_grid, save_image
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


def plot_losses(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[str] = None
) -> None:
    """
    Rysuje wykres strat.

    Args:
        losses: Lista wartości strat
        title: Tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisania wykresu
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[saved] {save_path}")

    plt.show()
    plt.close(fig)


def render_live(
    step: int,
    images: Optional[torch.Tensor] = None,
    losses_dict: Optional[dict] = None,
) -> None:
    """
    Renderuje live output w notebooku.

    Args:
        step: Aktualny krok
        images: Tensor obrazów do wyświetlenia (opcjonalny)
        losses_dict: Słownik z historiami strat {"name": [values]}
    """
    if not _HAS_IPYTHON:
        return

    clear_output(wait=True)

    n_plots = sum([images is not None, losses_dict is not None])
    if n_plots == 0:
        return

    fig = plt.figure(figsize=(5 * n_plots, 4))
    plot_idx = 1

    if images is not None:
        ax = fig.add_subplot(1, n_plots, plot_idx)
        if _HAS_TORCHVISION and images.dim() == 4:
            grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
            img = grid.permute(1, 2, 0).cpu().numpy()
        else:
            img = images.cpu().numpy()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Samples @ step {step}")
        plot_idx += 1

    if losses_dict:
        ax = fig.add_subplot(1, n_plots, plot_idx)
        for name, values in losses_dict.items():
            ax.plot(values, label=name)
        ax.set_title("Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)

    display(fig)
    plt.close(fig)


def save_artifacts(
    step: int,
    out_dir: str,
    images: Optional[torch.Tensor] = None,
    losses_dict: Optional[dict] = None,
) -> None:
    """
    Zapisuje artefakty (obrazy, wykresy) do folderu.

    Args:
        step: Aktualny krok
        out_dir: Folder wyjściowy
        images: Tensor obrazów
        losses_dict: Słownik z historiami strat
    """
    os.makedirs(out_dir, exist_ok=True)

    if images is not None:
        sample_path = os.path.join(out_dir, f"samples_step{step:06d}.png")
        latest_path = os.path.join(out_dir, "samples_latest.png")

        if _HAS_TORCHVISION and images.dim() == 4:
            grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
            save_image(grid, sample_path)
            save_image(grid, latest_path)
        else:
            img = images.cpu().numpy()
            plt.imsave(sample_path, img)
            plt.imsave(latest_path, img)

        print(f"[artifacts] {sample_path}")

    if losses_dict:
        fig = plt.figure(figsize=(8, 4))
        for name, values in losses_dict.items():
            plt.plot(values, label=name)
        plt.title("Losses")
        plt.legend()
        plt.grid(True, alpha=0.3)

        loss_path = os.path.join(out_dir, f"losses_step{step:06d}.png")
        latest_loss_path = os.path.join(out_dir, "losses_latest.png")
        fig.savefig(loss_path, bbox_inches="tight", dpi=150)
        fig.savefig(latest_loss_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        print(f"[artifacts] {loss_path}")

