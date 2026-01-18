"""Moduł logowania dla eksperymentów."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "experiment",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Konfiguruje logger dla eksperymentu.

    Args:
        name: Nazwa loggera
        level: Poziom logowania
        log_file: Opcjonalna ścieżka do pliku logów

    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Formatter
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (opcjonalny)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger

