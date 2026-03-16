"""Checkpoint save/load helpers with resume support."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional

import torch

from .config_loader import RunConfig
from .r3gan_source import R3GANTrainer


def save_training_checkpoint(
    path: str,
    *,
    step: int,
    trainer: R3GANTrainer,
    cfg: RunConfig,
    runtime_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "step": step,
        "G": trainer.G.state_dict(),
        "D": trainer.D.state_dict(),
        "G_ema": trainer.G_ema.state_dict(),
        "g_opt": trainer.g_opt.state_dict(),
        "d_opt": trainer.d_opt.state_dict(),
        "cfg": cfg.to_dict(),
    }
    if runtime_state is not None:
        payload["runtime_state"] = runtime_state
    torch.save(payload, path)


def load_training_checkpoint(
    path: str,
    *,
    trainer: R3GANTrainer,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    trainer.G.load_state_dict(ckpt["G"])
    trainer.D.load_state_dict(ckpt["D"])
    trainer.G_ema.load_state_dict(ckpt["G_ema"])
    trainer.g_opt.load_state_dict(ckpt["g_opt"])
    trainer.d_opt.load_state_dict(ckpt["d_opt"])
    return ckpt


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    pattern = os.path.join(ckpt_dir, "ckpt_*.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def resolve_resume_checkpoint(resume: str, ckpt_dir: str) -> str:
    """Resolve resume source from explicit file path, directory or 'latest'."""
    token = resume.strip()
    if token.lower() == "latest":
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        return latest

    if os.path.isdir(token):
        latest = find_latest_checkpoint(token)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {token}")
        return latest

    if not os.path.exists(token):
        raise FileNotFoundError(f"Resume checkpoint not found: {token}")
    return token


def validate_resume_compatibility(current_cfg: RunConfig, checkpoint_cfg: Dict[str, Any]) -> None:
    """Guard against resuming with an incompatible architecture."""
    critical_keys = [
        "img_resolution",
        "z_dim",
        "base_channels",
        "channel_max",
        "blocks_per_stage",
        "expansion_factor",
        "group_size",
        "in_channels",
        "out_channels",
    ]
    mismatches = []
    for key in critical_keys:
        current = getattr(current_cfg, key)
        previous = checkpoint_cfg.get(key)
        if previous is not None and previous != current:
            mismatches.append(f"{key} (resume={previous}, current={current})")

    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            "Resume checkpoint is incompatible with current config: "
            f"{mismatch_text}"
        )

