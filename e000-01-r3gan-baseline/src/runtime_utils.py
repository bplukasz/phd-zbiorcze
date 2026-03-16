"""Runtime utilities local to e000-01 baseline."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set reproducible seeds for random, numpy and torch (if available)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

