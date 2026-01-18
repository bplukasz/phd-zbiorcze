from .config_loader import (
    get_config,
    RunConfig,
    ConfigLoader,
)

# Lazy import dla experiment (wymaga torch)
def __getattr__(name):
    if name == "train":
        from . import experiment
        return getattr(experiment, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "train",
    "get_config",
    "RunConfig",
    "ConfigLoader",
]

