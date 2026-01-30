"""
Utility functions: CSVLogger, gradient computation, etc.
"""

import os
import csv
from typing import List, Dict, Any


class CSVLogger:
    """Prosty logger do pliku CSV."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
