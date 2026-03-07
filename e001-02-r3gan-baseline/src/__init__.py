"""e001-02-r3gan-baseline src package."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .experiment import train
from .config_loader import get_config, RunConfig, ConfigLoader

__all__ = ["train", "get_config", "RunConfig", "ConfigLoader"]
