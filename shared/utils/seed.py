"""Utilities for reproducibility (seeding).

Moduł do zarządzania seedami i zapewnienia reprodukowalności eksperymentów.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import random


@dataclass(frozen=True)
class SeedConfig:
    """Konfiguracja seeda dla eksperymentu."""
    seed: int = 42
    deterministic: bool = False


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Ustawia seed dla random/numpy/torch.

    Args:
        seed: Wartość seeda
        deterministic: Czy wymuszać deterministyczne operacje (może wpłynąć na wydajność)

    Uwaga:
        Deterministyczność na GPU nie zawsze jest 100% możliwa.
        Jeśli `deterministic=True`, PyTorch może rzucić wyjątek na niedeterministycznych operacjach.
    """

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy (opcjonalnie)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Best-effort determinism
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Starsze wersje torch albo brak wsparcia – ignoruj.
                pass
        else:
            # Domyślnie: wydajność > deterministyka
            torch.backends.cudnn.benchmark = True

    except Exception:
        # Torch nie jest dostępny w jakimś kontekście importu
        pass

