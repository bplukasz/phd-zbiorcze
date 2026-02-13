import csv, os, time
from .io import ensure_dir

class CSVLogger:
    def __init__(self, path: str, fieldnames):
        ensure_dir(os.path.dirname(path))
        self.path = path
        self.fieldnames = fieldnames
        self._init = not os.path.exists(path)
        if self._init:
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

    def log(self, row: dict):
        row = dict(row)
        row.setdefault("time", time.time())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)

