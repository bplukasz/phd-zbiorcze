import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

# Optional: notebook live display
try:
    from IPython.display import clear_output, display  # type: ignore
    _HAS_IPYTHON = True
except Exception:
    _HAS_IPYTHON = False

import matplotlib.pyplot as plt

# Optional: torchvision grid helpers
try:
    from torchvision.utils import make_grid, save_image  # type: ignore
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


@dataclass
class RunConfig:
    # profile
    name: str = "preview"         # "preview" (notebook) lub "train" (script)
    steps: int = 120
    batch_size: int = 128
    z_dim: int = 64
    lr: float = 2e-4

    # logging / viz
    log_every: int = 10           # print loss co N kroków
    viz_every: int = 10           # generuj obrazki/wykres co N kroków
    live: bool = True             # notebook: True (inline), script: False (save to artifacts)

    # output
    out_dir: str = "/kaggle/working/artifacts"


def get_config(profile: str) -> RunConfig:
    profile = (profile or "").strip().lower()
    if profile in ("preview", "live", "notebook"):
        return RunConfig(
            name="preview",
            steps=120,
            batch_size=128,
            z_dim=64,
            lr=2e-4,
            log_every=10,
            viz_every=10,
            live=True,
        )
    if profile in ("train", "script", "long"):
        return RunConfig(
            name="train",
            steps=5000,
            batch_size=128,
            z_dim=64,
            lr=2e-4,
            log_every=100,
            viz_every=250,   # rzadziej, żeby nie zapchać outputu
            live=False,
        )
    # default fallback
    return RunConfig(name=profile or "custom")


class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _make_sample_grid(G: nn.Module, device: torch.device, z_dim: int) -> torch.Tensor:
    """
    Zwraca grid jako tensor CHW (3 kanały dla matplotlib, jeśli trzeba).
    """
    z = torch.randn(64, z_dim, device=device)
    samples = G(z).view(-1, 1, 28, 28).detach().cpu()

    if _HAS_TORCHVISION:
        grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))  # (C,H,W)
        # make_grid daje 1 kanał -> dla matplotlib zrobimy 3
        if grid.shape[0] == 1:
            grid = grid.repeat(3, 1, 1)
        return grid

    # fallback bez torchvision: prosty “tile” przez matplotlib ogarnie pojedyncze sample,
    # a grid zrobimy minimalnie: (8*28 x 8*28)
    # Zwracamy jako 3xH xW
    samples = (samples + 1.0) / 2.0  # 0..1
    tile = torch.zeros(1, 8 * 28, 8 * 28)
    idx = 0
    for r in range(8):
        for c in range(8):
            tile[:, r*28:(r+1)*28, c*28:(c+1)*28] = samples[idx]
            idx += 1
    return tile.repeat(3, 1, 1)


def _render_live(step: int, grid_chw: torch.Tensor, losses_g: List[float], losses_d: List[float]):
    if not (_HAS_IPYTHON):
        return

    clear_output(wait=True)
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    img = grid_chw.permute(1, 2, 0).numpy()
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(f"samples @ step {step}")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(losses_d, label="D loss")
    ax2.plot(losses_g, label="G loss")
    ax2.set_title("losses")
    ax2.legend()

    display(fig)
    plt.close(fig)


def _save_artifacts(step: int, grid_chw: torch.Tensor, losses_g: List[float], losses_d: List[float], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # samples
    sample_path = os.path.join(out_dir, f"samples_step{step:06d}.png")
    latest_path = os.path.join(out_dir, "samples_latest.png")

    if _HAS_TORCHVISION:
        # save_image oczekuje CHW 0..1, a grid jest już znormalizowany w make_grid
        save_image(grid_chw, sample_path)
        save_image(grid_chw, latest_path)
    else:
        img = grid_chw.permute(1, 2, 0).numpy()
        plt.imsave(sample_path, img)
        plt.imsave(latest_path, img)

    # losses plot
    fig = plt.figure(figsize=(8, 4))
    plt.plot(losses_d, label="D loss")
    plt.plot(losses_g, label="G loss")
    plt.title("losses")
    plt.legend()
    loss_path = os.path.join(out_dir, f"losses_step{step:06d}.png")
    latest_loss_path = os.path.join(out_dir, "losses_latest.png")
    fig.savefig(loss_path, bbox_inches="tight")
    fig.savefig(latest_loss_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[artifacts] {sample_path}")
    print(f"[artifacts] {loss_path}")


def train(profile: str = "preview") -> Tuple[nn.Module, nn.Module, List[float], List[float]]:
    cfg = get_config(profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("profile:", cfg.name)
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    G = Generator(z_dim=cfg.z_dim).to(device)
    D = Discriminator().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    losses_g: List[float] = []
    losses_d: List[float] = []

    t0 = time.time()

    # snapshot na starcie
    grid0 = _make_sample_grid(G, device, cfg.z_dim)
    if cfg.live:
        _render_live(0, grid0, losses_g, losses_d)
    else:
        _save_artifacts(0, grid0, losses_g, losses_d, cfg.out_dir)

    for step in range(1, cfg.steps + 1):
        # smoke: "real" = losowe wektory, bez datasetu/internetu
        real = torch.randn(cfg.batch_size, 784, device=device).clamp(-1, 1)

        # --- D ---
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake = G(z).detach()
        D_real = D(real)
        D_fake = D(fake)
        loss_d = bce(D_real, torch.ones_like(D_real)) + bce(D_fake, torch.zeros_like(D_fake))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # --- G ---
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake2 = G(z)
        D_fake2 = D(fake2)
        loss_g = bce(D_fake2, torch.ones_like(D_fake2))
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        losses_d.append(float(loss_d.item()))
        losses_g.append(float(loss_g.item()))

        if cfg.log_every > 0 and (step == 1 or step % cfg.log_every == 0):
            print(f"step {step:06d} | loss_d={losses_d[-1]:.4f} | loss_g={losses_g[-1]:.4f}")

        if cfg.viz_every > 0 and step % cfg.viz_every == 0:
            grid = _make_sample_grid(G, device, cfg.z_dim)
            if cfg.live:
                _render_live(step, grid, losses_g, losses_d)
            else:
                _save_artifacts(step, grid, losses_g, losses_d, cfg.out_dir)

    # checkpoint
    os.makedirs("/kaggle/working", exist_ok=True)
    ckpt_path = f"/kaggle/working/smoke_gan_{cfg.name}.pt"
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_path)
    print("saved:", ckpt_path)
    print("done in", round(time.time() - t0, 2), "s")
    print("out_dir:", cfg.out_dir)

    return G, D, losses_g, losses_d
