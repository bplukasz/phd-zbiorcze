import os, yaml, json
from dataclasses import dataclass

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_text(s: str, path: str):
    with open(path, "w") as f:
        f.write(s)

