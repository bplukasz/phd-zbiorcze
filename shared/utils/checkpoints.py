"""Moduł zarządzania checkpointami."""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Zapisuje checkpoint modelu.
    
    Args:
        path: Ścieżka do pliku checkpoint
        model: Model do zapisania
        optimizer: Opcjonalny optymalizator
        step: Aktualny krok treningowy
        extra: Dodatkowe dane do zapisania
    
    Returns:
        Ścieżka do zapisanego checkpointu
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if extra:
        checkpoint.update(extra)
    
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved: {path}")
    
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Wczytuje checkpoint modelu.
    
    Args:
        path: Ścieżka do pliku checkpoint
        model: Model do wczytania wag
        optimizer: Opcjonalny optymalizator
        device: Device na który wczytać
    
    Returns:
        Słownik z danymi checkpointu
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    step = checkpoint.get("step", 0)
    print(f"[checkpoint] Loaded: {path} (step {step})")
    
    return checkpoint
