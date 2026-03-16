"""ETA and runtime progress helpers for training loop."""

from __future__ import annotations

import csv
import os
import time
from typing import Optional


def step_to_kimg(step: int, batch_size: int) -> float:
    return (step * batch_size) / 1000.0


def update_ema(prev: Optional[float], value: float, alpha: float = 0.05) -> float:
    if prev is None:
        return float(value)
    return (1.0 - alpha) * prev + alpha * float(value)


def format_eta(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    if sec < 60:
        return f"{sec}s"
    minutes = sec // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h {minutes % 60:02d}m"
    days = hours // 24
    return f"{days}d {hours % 24:02d}h"


def format_eta_finish_ts(seconds: float) -> str:
    finish_ts = time.time() + max(0.0, seconds)
    fmt = "%H:%M" if seconds < 24 * 3600 else "%m-%d %H:%M"
    return time.strftime(fmt, time.localtime(finish_ts))


def count_remaining_metric_evals(step: int, total_steps: int, metrics_every: int) -> int:
    if metrics_every <= 0 or step >= total_steps:
        return 0
    if step % metrics_every == 0:
        next_eval_step = step
    else:
        next_eval_step = step + (metrics_every - (step % metrics_every))
    if next_eval_step > total_steps:
        return 0
    return 1 + (total_steps - next_eval_step) // metrics_every


def estimate_remaining_seconds(
    step: int,
    total_steps: int,
    sec_per_iter_ema: Optional[float],
    metrics_every: int,
    metrics_elapsed_ema: Optional[float],
) -> Optional[float]:
    if sec_per_iter_ema is None:
        return None
    remaining_steps = max(0, total_steps - step)
    eta_sec = remaining_steps * sec_per_iter_ema
    if metrics_every > 0 and metrics_elapsed_ema is not None:
        eta_sec += count_remaining_metric_evals(step, total_steps, metrics_every) * metrics_elapsed_ema
    return eta_sec


def load_metrics_elapsed_average(csv_path: str) -> Optional[float]:
    if not os.path.exists(csv_path):
        return None

    total = 0.0
    count = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "gan_metrics":
                continue
            raw_value = row.get("metrics_elapsed_sec", "")
            if raw_value in (None, ""):
                continue
            try:
                elapsed = float(raw_value)
            except ValueError:
                continue
            if elapsed <= 0:
                continue
            total += elapsed
            count += 1

    if count == 0:
        return None
    return total / count

