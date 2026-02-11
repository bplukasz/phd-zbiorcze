"""CSV logging utilities.

Prosty logger do zapisywania metryk eksperymentów do plików CSV.
"""

import os
import csv
from typing import List, Dict, Any


class CSVLogger:
    """Prosty logger do pliku CSV."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        """
        Inicjalizuje logger CSV.

        Args:
            filepath: Ścieżka do pliku CSV
            fieldnames: Lista nazw kolumn
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """
        Zapisuje wiersz do CSV.

        Args:
            row: Słownik z danymi (klucze muszą być w fieldnames)
        """
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

