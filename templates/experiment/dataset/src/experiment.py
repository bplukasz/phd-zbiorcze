"""
Eksperyment: {{FULL_NAME}}
Opis: [UZUPEŁNIJ OPIS EKSPERYMENTU]
"""

import os
import time
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn

# Optional: notebook live display
try:
    from IPython.display import clear_output, display
    _HAS_IPYTHON = True
except Exception:
    _HAS_IPYTHON = False

import matplotlib.pyplot as plt

# Import configuration system
from .config_loader import RunConfig, get_config


# ============================================================================
# Training Loop
# ============================================================================

def train(profile: str = "preview", overrides: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, List[float]]:
    """
    Główna funkcja treningowa.

    Args:
        profile: "preview" dla notebooka, "train" dla pełnego treningu
        overrides: Dodatkowe nadpisania konfiguracji z CLI

    Returns:
        Tuple z modelem i historią strat
    """
    cfg = get_config(profile, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=" * 60)
    print(f"Eksperyment: {{FULL_NAME}}")
    print(f"Profile: {cfg.name}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"=" * 60)

    # Directories
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save actual config for reproducibility
    from .config_loader import ConfigLoader
    loader = ConfigLoader()
    loader.save_config(cfg, os.path.join(cfg.out_dir, "config_used.yaml"))

    # TODO: Zaimplementuj logikę eksperymentu

    t0 = time.time()
    losses: List[float] = []
    model = nn.Identity()  # Placeholder

    for step in range(1, cfg.steps + 1):
        # TODO: Główna pętla treningowa
        loss = 0.0
        losses.append(loss)

        if cfg.log_every > 0 and step % cfg.log_every == 0:
            print(f"[{step:06d}/{cfg.steps}] loss={losses[-1]:.4f}")

    print(f"=" * 60)
    print(f"Zakończono w {time.time() - t0:.2f}s")
    print(f"=" * 60)

    return model, losses

