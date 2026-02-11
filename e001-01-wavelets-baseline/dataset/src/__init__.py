# Setup sys.path for Kaggle environment BEFORE any imports
import sys
import os

# Dodaj shared-lib do sys.path jeśli jeszcze nie ma
SHARED_DIR = "/kaggle/input/shared-lib"
if SHARED_DIR not in sys.path and os.path.exists(SHARED_DIR):
    sys.path.insert(0, SHARED_DIR)

from .config_loader import (
    get_config,
    RunConfig,
    ConfigLoader,
)

# Import ogólnych funkcji z shared-lib
# Na Kaggle struktura to: /kaggle/input/shared-lib/utils/ (nie shared/utils/)
from utils import (
    DiffAugment,
    hinge_loss_d,
    hinge_loss_g,
    compute_grad_norm,
    compute_fid_kid,
    generate_samples,
    export_real_images,
    CSVLogger,
    set_seed,
)

# Import funkcji lokalnych (specyficznych dla eksperymentu)
from .data import get_dataloader

# Lazy import dla experiment (wymaga torch)
def __getattr__(name):
    if name in ["train", "Generator", "Discriminator", "EMA"]:
        from . import experiment
        return getattr(experiment, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "train",
    "get_config",
    "RunConfig",
    "ConfigLoader",
    "Generator",
    "Discriminator",
    "EMA",
    # Z shared utils
    "DiffAugment",
    "hinge_loss_d",
    "hinge_loss_g",
    "compute_grad_norm",
    "compute_fid_kid",
    "generate_samples",
    "export_real_images",
    "CSVLogger",
    "set_seed",
    "get_dataloader",
]

