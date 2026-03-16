"""Artifact and CSV I/O helpers for the training pipeline."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Sequence

import torch
from torchvision.utils import save_image

from .r3gan_source import R3GANTrainer, parse_batch


LOG_FIELDNAMES: Sequence[str] = [
    "step",
    "kimg",
    "row_type",
    "d_loss",
    "d_adv",
    "r1",
    "r2",
    "g_loss",
    "g_adv",
    "g_reg",
    "real_score_mean",
    "fake_score_mean",
    "sec_per_iter",
    "vram_peak_mb",
    "eta_sec",
    "eta_human",
    "eta_finish_at",
    "fid",
    "fid_auc_vs_kimg",
    "kid_mean",
    "kid_std",
    "precision",
    "recall",
    "lpips_diversity",
    "metrics_elapsed_sec",
]


class CSVLogger:
    """Simple CSV logger with optional append mode for resume."""

    def __init__(self, filepath: str, fieldnames: Sequence[str], append: bool = False):
        self.filepath = filepath
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        mode = "a" if append else "w"
        needs_header = (not append) or (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0)
        with open(filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if needs_header:
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def make_csv_logger(out_dir: str, *, append: bool = False) -> CSVLogger:
    return CSVLogger(os.path.join(out_dir, "logs.csv"), LOG_FIELDNAMES, append=append)


@torch.no_grad()
def save_grid(trainer: R3GANTrainer, fixed_z: torch.Tensor, path: str, n_row: int = 8) -> None:
    trainer.G_ema.eval()
    imgs = trainer.G_ema(fixed_z)
    imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
    save_image(imgs, path, nrow=n_row)


def save_real_grid(real_imgs: torch.Tensor, path: str, n_row: int = 8) -> None:
    imgs = (real_imgs.clamp(-1, 1) + 1.0) / 2.0
    save_image(imgs[:64], path, nrow=n_row)


def export_real_samples(dataloader, n: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for batch in dataloader:
        imgs, _ = parse_batch(batch)
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
        for i in range(imgs.size(0)):
            if saved >= n:
                break
            save_image(imgs[i], os.path.join(out_dir, f"{saved:06d}.png"))
            saved += 1
        if saved >= n:
            break
    print(f"Exported {saved} real samples -> {out_dir}")


def export_samples(trainer: R3GANTrainer, n: int, out_dir: str, step: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    batch_size = 64
    saved = 0
    while saved < n:
        cur = min(batch_size, n - saved)
        imgs = trainer.sample(cur)
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
        for i in range(imgs.size(0)):
            save_image(imgs[i], os.path.join(out_dir, f"step{step:07d}_{saved:06d}.png"))
            saved += 1
    print(f"Saved {saved} fake samples -> {out_dir}")


def _count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_model_info(G: Any, D: Any, out_dir: str) -> None:
    g_info = _count_params(G)
    d_info = _count_params(D)
    info = {
        "generator": {
            "class": type(G).__name__,
            "params_total": g_info["total"],
            "params_trainable": g_info["trainable"],
            "params_total_M": round(g_info["total"] / 1e6, 4),
        },
        "discriminator": {
            "class": type(D).__name__,
            "params_total": d_info["total"],
            "params_trainable": d_info["trainable"],
            "params_total_M": round(d_info["total"] / 1e6, 4),
        },
        "total_params": g_info["total"] + d_info["total"],
        "total_params_M": round((g_info["total"] + d_info["total"]) / 1e6, 4),
    }
    path = os.path.join(out_dir, "model_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved -> {path}")
    print(f"  G ({info['generator']['class']}): {info['generator']['params_total_M']:.2f}M params")
    print(f"  D ({info['discriminator']['class']}): {info['discriminator']['params_total_M']:.2f}M params")

