"""Runtime helpers for GAN metrics in training and resume."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional, Tuple, cast

import torch

from .config_loader import RunConfig
from .gan_metrics import DependencyError, GANMetricsConfig, GANMetricsSuite


def validate_metrics_dataset_size(cfg: RunConfig, dataset_size: Optional[int]) -> None:
    if cfg.metrics_every == 0:
        return
    if dataset_size is not None and dataset_size < cfg.metrics_kid_subset_size:
        raise ValueError(
            f"Dataset too small for KID subset_size={cfg.metrics_kid_subset_size}. "
            f"Available real samples: {dataset_size}."
        )


def build_metrics_suite(cfg: RunConfig, device: torch.device) -> Optional[GANMetricsSuite]:
    if cfg.metrics_every == 0:
        return None
    try:
        metrics_cfg = GANMetricsConfig(
            device=str(device),
            input_range="minus_one_to_one",
            fid_feature=cfg.metrics_fid_feature,
            kid_feature=cfg.metrics_kid_feature,
            kid_subsets=cfg.metrics_kid_subsets,
            kid_subset_size=cfg.metrics_kid_subset_size,
            max_real_images_fid_kid=cfg.metrics_max_real,
            reset_real_features=False,
            pr_num_samples=cfg.metrics_pr_num_samples,
            pr_k=cfg.metrics_pr_k,
            lpips_num_pairs=cfg.metrics_lpips_num_pairs,
            lpips_pool_size=cfg.metrics_lpips_pool_size,
            use_amp_for_feature_extractor=(device.type == "cuda"),
            amp_dtype=cast(Any, cfg.metrics_amp_dtype),
            use_channels_last=cfg.channels_last,
            seed=cfg.seed,
            verbose=True,
        )
        return GANMetricsSuite(metrics_cfg)
    except DependencyError as exc:
        raise RuntimeError(
            "GAN metrics are enabled, but required dependencies are missing. "
            "Install requirements from e000-01-r3gan-baseline/requirements.txt or disable metrics with metrics_every=0."
        ) from exc


def update_fid_auc(
    prev_point: Optional[Tuple[float, float]],
    current_kimg: float,
    current_fid: float,
    cumulative_auc: float,
) -> Tuple[Tuple[float, float], float]:
    if prev_point is None:
        return (current_kimg, current_fid), 0.0
    prev_kimg, prev_fid = prev_point
    delta_kimg = current_kimg - prev_kimg
    if delta_kimg <= 0:
        return (current_kimg, current_fid), cumulative_auc
    cumulative_auc += 0.5 * (prev_fid + current_fid) * delta_kimg
    return (current_kimg, current_fid), cumulative_auc


def load_metrics_state_from_logs(csv_path: str) -> Dict[str, Any]:
    """Recover metric runtime state from historical CSV rows for resume."""
    result: Dict[str, Any] = {
        "metrics_elapsed_ema": None,
        "prev_fid_point": None,
        "fid_auc_vs_kimg": 0.0,
    }
    if not os.path.exists(csv_path):
        return result

    elapsed_total = 0.0
    elapsed_count = 0
    last_kimg: Optional[float] = None
    last_fid: Optional[float] = None
    last_auc = 0.0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "gan_metrics":
                continue

            raw_elapsed = row.get("metrics_elapsed_sec", "")
            if raw_elapsed not in (None, ""):
                try:
                    elapsed = float(raw_elapsed)
                    if elapsed > 0:
                        elapsed_total += elapsed
                        elapsed_count += 1
                except ValueError:
                    pass

            raw_kimg = row.get("kimg", "")
            raw_fid = row.get("fid", "")
            raw_auc = row.get("fid_auc_vs_kimg", "")
            try:
                if raw_kimg not in (None, "") and raw_fid not in (None, ""):
                    last_kimg = float(raw_kimg)
                    last_fid = float(raw_fid)
                if raw_auc not in (None, ""):
                    last_auc = float(raw_auc)
            except ValueError:
                continue

    if elapsed_count > 0:
        result["metrics_elapsed_ema"] = elapsed_total / elapsed_count
    if last_kimg is not None and last_fid is not None:
        result["prev_fid_point"] = (last_kimg, last_fid)
    result["fid_auc_vs_kimg"] = last_auc
    return result

