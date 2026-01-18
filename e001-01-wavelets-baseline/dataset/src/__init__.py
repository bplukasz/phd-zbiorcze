from .config_loader import (
    get_config,
    RunConfig,
    ConfigLoader,
)

# Lazy import dla experiment (wymaga torch)
def __getattr__(name):
    if name in ["train", "Generator", "Discriminator", "EMA", "DiffAugment",
                "hinge_loss_d", "hinge_loss_g", "compute_grad_norm",
                "compute_fid_kid", "generate_samples", "CSVLogger"]:
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
    "DiffAugment",
    "hinge_loss_d",
    "hinge_loss_g",
    "compute_grad_norm",
    "compute_fid_kid",
    "generate_samples",
    "CSVLogger",
]

