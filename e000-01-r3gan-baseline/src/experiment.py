"""Experiment orchestration for e000-01 R3GAN baseline."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, cast

import torch

from .artifact_io import (
    export_real_samples,
    export_samples,
    make_csv_logger,
    save_grid,
    save_model_info,
    save_real_grid,
)
from .checkpointing import (
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_training_checkpoint,
    validate_resume_compatibility,
)
from .config_loader import ConfigLoader, RunConfig, get_config
from .data import get_dataloader
from .eta_logging import load_metrics_elapsed_average
from .metrics_runtime import (
    build_metrics_suite,
    load_metrics_state_from_logs,
    validate_metrics_dataset_size,
)
from .r3gan_source import (
    R3GANDiscriminator,
    R3GANGenerator,
    R3GANTrainer,
    TrainerConfig,
    build_stage_channels,
    parse_batch,
    setup_nvidia_performance,
)
from .runtime_utils import set_seed
from .training_loop import run_training_loop


def _build_models(cfg: RunConfig) -> Tuple[Any, Any]:
    """Buduje G i D na podstawie konfiguracji."""
    g_ch = build_stage_channels(cfg.img_resolution, cfg.base_channels, cfg.channel_max)
    d_ch = list(reversed(g_ch))
    G = R3GANGenerator(
        z_dim=cfg.z_dim,
        img_resolution=cfg.img_resolution,
        stage_channels=g_ch,
        blocks_per_stage=cfg.blocks_per_stage,
        expansion_factor=cfg.expansion_factor,
        group_size=cfg.group_size,
        resample_mode=cfg.resample_mode,
        out_channels=cfg.out_channels,
    )
    D = R3GANDiscriminator(
        img_resolution=cfg.img_resolution,
        stage_channels=d_ch,
        blocks_per_stage=cfg.blocks_per_stage,
        expansion_factor=cfg.expansion_factor,
        group_size=cfg.group_size,
        in_channels=cfg.in_channels,
        resample_mode=cfg.resample_mode,
    )
    return G, D


def _print_run_banner(cfg: RunConfig, device: torch.device, resume: Optional[str]) -> None:
    print("=" * 60)
    print("e000-01-r3gan-baseline")
    print(f"Profile : {cfg.name}")
    print(f"Device  : {device}" + (f"  [{torch.cuda.get_device_name(0)}]" if device.type == "cuda" else ""))
    print(f"Dataset : {cfg.dataset_name}  {cfg.img_resolution}x{cfg.img_resolution}")
    print(f"Steps   : {cfg.steps}")
    print(
        f"Metrics : {'enabled' if cfg.metrics_every > 0 else 'disabled'}"
        + (f"  (every {cfg.metrics_every} steps, fake={cfg.metrics_num_fake})" if cfg.metrics_every > 0 else "")
    )
    if resume:
        print(f"Resume  : {resume}")
    print("=" * 60)


def _prepare_output_dirs(out_dir: str) -> Dict[str, str]:
    paths = {
        "out_dir": out_dir,
        "grid_dir": os.path.join(out_dir, "grids"),
        "ckpt_dir": os.path.join(out_dir, "checkpoints"),
        "samp_dir": os.path.join(out_dir, "samples"),
        "real_dir": os.path.join(out_dir, "real_samples"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _initialize_data_artifacts(
    *,
    cfg: RunConfig,
    dataloader,
    resume: Optional[str],
    real_dir: str,
    out_dir: str,
    dataset_size: Optional[int],
) -> None:
    data_iter = iter(dataloader)
    first_batch, _ = parse_batch(next(data_iter))
    print(f"Real data  min={first_batch.min():.3f}  max={first_batch.max():.3f}  mean={first_batch.mean():.3f}")
    save_real_grid(first_batch, os.path.join(out_dir, "real_grid.png"))
    print("Saved real_grid.png — sprawdź czy dane wyglądają OK!")

    print("Eksportuję real_samples...")
    export_real_n = min(10_000, dataset_size) if dataset_size is not None else 10_000
    if not (resume and os.path.exists(real_dir) and os.listdir(real_dir)):
        export_real_samples(dataloader, n=export_real_n, out_dir=real_dir)
    else:
        print(f"real_samples already present in {real_dir}; skipping export.")


def train(
    profile: str = "base",
    overrides: Optional[Dict[str, Any]] = None,
    resume: Optional[str] = None,
):
    """Run training for selected profile with optional checkpoint resume."""
    cfg = get_config(profile, overrides)

    set_seed(cfg.seed, cfg.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_nvidia_performance(device)

    _print_run_banner(cfg, device, resume)

    dirs = _prepare_output_dirs(cfg.out_dir)
    out_dir = dirs["out_dir"]
    grid_dir = dirs["grid_dir"]
    ckpt_dir = dirs["ckpt_dir"]
    samp_dir = dirs["samp_dir"]
    real_dir = dirs["real_dir"]

    ConfigLoader().save_config(cfg, os.path.join(out_dir, "config_used.yaml"))

    G, D = _build_models(cfg)
    train_cfg = TrainerConfig(
        lr_g=cfg.lr_g,
        lr_d=cfg.lr_d,
        betas=tuple(cfg.betas),
        gamma=cfg.gamma,
        ema_beta=cfg.ema_beta,
        use_amp_for_g=cfg.use_amp_for_g,
        use_amp_for_d=cfg.use_amp_for_d,
        channels_last=cfg.channels_last,
        grad_clip=cfg.grad_clip,
    )
    trainer = R3GANTrainer(G, D, device=device, train_cfg=train_cfg)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"G params: {g_params/1e6:.2f}M   D params: {d_params/1e6:.2f}M")
    save_model_info(G, D, out_dir)

    dataloader = get_dataloader(
        cfg.data_dir,
        cfg.img_resolution,
        cfg.batch_size,
        dataset_name=cfg.dataset_name,
        img_channels=cfg.img_channels,
        seed=cfg.seed,
        num_workers=min(8, os.cpu_count() or 4),
    )

    dataset_size = int(len(cast(Any, dataloader.dataset))) if hasattr(dataloader, "dataset") else None
    validate_metrics_dataset_size(cfg, dataset_size)

    _initialize_data_artifacts(
        cfg=cfg,
        dataloader=dataloader,
        resume=resume,
        real_dir=real_dir,
        out_dir=out_dir,
        dataset_size=dataset_size,
    )

    metrics_suite = build_metrics_suite(cfg, device)
    if metrics_suite is not None:
        print(
            "Przygotowuję GAN metrics cache na real data "
            f"(max_real={cfg.metrics_max_real}, pr_samples={cfg.metrics_pr_num_samples})..."
        )
        metrics_suite.prepare_real(dataloader)
        print("GAN metrics cache ready.")

    logs_csv_path = os.path.join(out_dir, "logs.csv")
    metrics_state = load_metrics_state_from_logs(logs_csv_path)
    metrics_elapsed_ema = metrics_state.get("metrics_elapsed_ema")
    if metrics_elapsed_ema is None:
        metrics_elapsed_ema = load_metrics_elapsed_average(logs_csv_path)

    start_step = 1
    if resume:
        resume_path = resolve_resume_checkpoint(resume, ckpt_dir)
        ckpt = load_training_checkpoint(resume_path, trainer=trainer, device=device)
        validate_resume_compatibility(cfg, ckpt.get("cfg", {}))
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"Resumed from {resume_path} at step {start_step - 1}.")

        ckpt_runtime = ckpt.get("runtime_state", {})
        if isinstance(ckpt_runtime, dict):
            metrics_state["metrics_elapsed_ema"] = ckpt_runtime.get(
                "metrics_elapsed_ema",
                metrics_state.get("metrics_elapsed_ema"),
            )
            metrics_state["prev_fid_point"] = ckpt_runtime.get(
                "prev_fid_point",
                metrics_state.get("prev_fid_point"),
            )
            metrics_state["fid_auc_vs_kimg"] = ckpt_runtime.get(
                "fid_auc_vs_kimg",
                metrics_state.get("fid_auc_vs_kimg", 0.0),
            )

    csv_logger = make_csv_logger(out_dir, append=(resume is not None and os.path.exists(logs_csv_path)))

    fixed_z = torch.randn(cfg.save_n_samples, G.z_dim, device=device)
    runtime_state = run_training_loop(
        cfg=cfg,
        trainer=trainer,
        device=device,
        dataloader=dataloader,
        csv_logger=csv_logger,
        fixed_z=fixed_z,
        grid_dir=grid_dir,
        ckpt_dir=ckpt_dir,
        samp_dir=samp_dir,
        start_step=start_step,
        metrics_suite=metrics_suite,
        metrics_elapsed_ema=metrics_state.get("metrics_elapsed_ema", metrics_elapsed_ema),
        prev_fid_point=metrics_state.get("prev_fid_point"),
        fid_auc_vs_kimg=float(metrics_state.get("fid_auc_vs_kimg", 0.0)),
    )

    final_ckpt = os.path.join(ckpt_dir, "final.pt")
    save_training_checkpoint(
        final_ckpt,
        step=cfg.steps,
        trainer=trainer,
        cfg=cfg,
        runtime_state=runtime_state,
    )
    print(f"\nDone! Final checkpoint: {final_ckpt}")

    save_grid(trainer, fixed_z, os.path.join(grid_dir, "grid_final.png"))
    export_samples(trainer, n=1024, out_dir=samp_dir, step=cfg.steps)

    return trainer

