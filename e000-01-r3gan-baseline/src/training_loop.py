"""Core training loop extracted from experiment orchestrator."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .artifact_io import export_samples, save_grid
from .checkpointing import save_training_checkpoint
from .config_loader import RunConfig
from .eta_logging import (
    estimate_remaining_seconds,
    format_eta,
    format_eta_finish_ts,
    step_to_kimg,
    update_ema,
)
from .profiler import get_global_profiler
from .gan_metrics import format_metrics
from .metrics_runtime import update_fid_auc
from .r3gan_source import R3GANTrainer, parse_batch


@dataclass
class TrainingLoopIO:
    """IO hooks used by the training loop (logging, artifacts, console)."""

    row_logger: Any
    write_line: Any
    save_grid_at_step: Any
    save_checkpoint_at_step: Any
    export_samples_at_step: Any


def _build_default_io(
    *,
    cfg: RunConfig,
    trainer: R3GANTrainer,
    fixed_z: torch.Tensor,
    csv_logger,
    grid_dir: str,
    ckpt_dir: str,
    samp_dir: str,
) -> TrainingLoopIO:
    def _row_logger(row: Dict[str, Any]) -> None:
        csv_logger.log(row)

    def _write_line(message: str) -> None:
        print(message)

    def _save_grid(step: int) -> str:
        path = os.path.join(grid_dir, f"grid_{step:07d}.png")
        save_grid(trainer, fixed_z, path)
        return path

    def _save_checkpoint(step: int, runtime_state: Dict[str, Any]) -> str:
        path = os.path.join(ckpt_dir, f"ckpt_{step:07d}.pt")
        save_training_checkpoint(
            path,
            step=step,
            trainer=trainer,
            cfg=cfg,
            runtime_state=runtime_state,
        )
        return path

    def _export_samples(step: int, n: int = 256) -> None:
        export_samples(trainer, n=n, out_dir=samp_dir, step=step)

    return TrainingLoopIO(
        row_logger=_row_logger,
        write_line=_write_line,
        save_grid_at_step=_save_grid,
        save_checkpoint_at_step=_save_checkpoint,
        export_samples_at_step=_export_samples,
    )


def run_training_loop(
    *,
    cfg: RunConfig,
    trainer: R3GANTrainer,
    device: torch.device,
    dataloader,
    csv_logger,
    fixed_z: torch.Tensor,
    grid_dir: str,
    ckpt_dir: str,
    samp_dir: str,
    start_step: int = 1,
    metrics_suite=None,
    metrics_elapsed_ema: Optional[float] = None,
    prev_fid_point: Optional[Tuple[float, float]] = None,
    fid_auc_vs_kimg: float = 0.0,
    io: Optional[TrainingLoopIO] = None,
) -> Dict[str, Any]:
    sec_per_iter_ema: Optional[float] = None
    data_iter = iter(dataloader)
    io = io or _build_default_io(
        cfg=cfg,
        trainer=trainer,
        fixed_z=fixed_z,
        csv_logger=csv_logger,
        grid_dir=grid_dir,
        ckpt_dir=ckpt_dir,
        samp_dir=samp_dir,
    )

    if start_step <= cfg.steps:
        io.write_line(f"\nStart training: steps {start_step}..{cfg.steps}  batch={cfg.batch_size}")
    else:
        io.write_line(f"\nResume step ({start_step}) is already above target steps ({cfg.steps}).")
    io.write_line("-" * 60)

    profiler = get_global_profiler()
    
    for step in range(start_step, cfg.steps + 1):
        t_iter = time.time()
        kimg = step_to_kimg(step, cfg.batch_size)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Profiluj pobieranie danych
        with profiler.context("dataloader.fetch"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

        # Profiluj parsowanie batcha
        with profiler.context("batch.parse"):
            images, labels = parse_batch(batch)
        
        # Profiluj train_step
        with profiler.context("trainer.train_step"):
            metrics = trainer.train_step(images, labels)

        iter_time = time.time() - t_iter
        sec_per_iter_ema = update_ema(sec_per_iter_ema, iter_time)
        vram_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0

        if step % cfg.log_every == 0:
            eta_sec = estimate_remaining_seconds(
                step=step,
                total_steps=cfg.steps,
                sec_per_iter_ema=sec_per_iter_ema,
                metrics_every=cfg.metrics_every,
                metrics_elapsed_ema=metrics_elapsed_ema,
            )
            row = {
                "step": step,
                "kimg": round(kimg, 4),
                "row_type": "train",
                "d_loss": metrics.get("d_loss", 0.0),
                "d_adv": metrics.get("d_adv", 0.0),
                "r1": metrics.get("r1", 0.0),
                "r2": metrics.get("r2", 0.0),
                "g_loss": metrics.get("g_loss", 0.0),
                "g_adv": metrics.get("g_adv", 0.0),
                "g_reg": metrics.get("g_reg", 0.0),
                "real_score_mean": metrics.get("real_score_mean", 0.0),
                "fake_score_mean": metrics.get("fake_score_mean", 0.0),
                "sec_per_iter": round(iter_time, 4),
                "vram_peak_mb": round(vram_mb, 1),
                "eta_sec": round(eta_sec, 3) if eta_sec is not None else "",
                "eta_human": format_eta(eta_sec) if eta_sec is not None else "",
                "eta_finish_at": format_eta_finish_ts(eta_sec) if eta_sec is not None else "",
                "fid": "",
                "fid_auc_vs_kimg": "",
                "kid_mean": "",
                "kid_std": "",
                "precision": "",
                "recall": "",
                "lpips_diversity": "",
                "metrics_elapsed_sec": "",
            }
            io.row_logger(row)

            eta_txt = ""
            if eta_sec is not None:
                eta_txt = f"  eta={format_eta(eta_sec)} (to {row['eta_finish_at']})"

            io.write_line(
                f"[{step:>7d}/{cfg.steps}]  "
                f"d={row['d_loss']:.4f}  g={row['g_loss']:.4f}  "
                f"g_adv={row['g_adv']:.4f}  g_reg={row['g_reg']:.4f}  "
                f"r1={row['r1']:.4f}  r2={row['r2']:.4f}  "
                f"{iter_time * 1000:.0f}ms"
                + (f"  {vram_mb:.0f}MB" if device.type == "cuda" else "")
                + eta_txt
            )

        if step % cfg.grid_every == 0:
            grid_path = io.save_grid_at_step(step)
            io.write_line(f"  -> grid saved: {grid_path}")

        if step % cfg.ckpt_every == 0:
            ckpt_path = io.save_checkpoint_at_step(
                step,
                {
                    "metrics_elapsed_ema": metrics_elapsed_ema,
                    "prev_fid_point": prev_fid_point,
                    "fid_auc_vs_kimg": fid_auc_vs_kimg,
                },
            )
            io.write_line(f"  -> checkpoint saved: {ckpt_path}")
            io.export_samples_at_step(step, 256)

        if metrics_suite is not None and cfg.metrics_every > 0 and step % cfg.metrics_every == 0:
            io.write_line(f"\n[{step:>7d}/{cfg.steps}]  Obliczam metryki GAN ({cfg.metrics_num_fake} fake images)...")
            t_metrics = time.time()

            @torch.no_grad()
            def _sample_fn(n: int) -> torch.Tensor:
                return trainer.sample(n)

            gan_metrics = metrics_suite.evaluate_generator(
                _sample_fn,
                num_fake_images=cfg.metrics_num_fake,
                fake_batch_size=cfg.metrics_fake_batch_size,
            )

            elapsed_metrics = time.time() - t_metrics
            metrics_elapsed_ema = update_ema(metrics_elapsed_ema, elapsed_metrics)
            current_fid = float(gan_metrics["fid"])
            prev_fid_point, fid_auc_vs_kimg = update_fid_auc(
                prev_fid_point,
                kimg,
                current_fid,
                fid_auc_vs_kimg,
            )

            io.write_line(
                f"[{step:>7d}/{cfg.steps}]  METRICS  {format_metrics(gan_metrics)}"
                f"  fid_auc_vs_kimg={fid_auc_vs_kimg:.4f}"
                f"  ({elapsed_metrics:.1f}s)"
            )

            metrics_row = {
                "step": step,
                "kimg": round(kimg, 4),
                "row_type": "gan_metrics",
                "d_loss": "",
                "d_adv": "",
                "r1": "",
                "r2": "",
                "g_loss": "",
                "g_adv": "",
                "g_reg": "",
                "real_score_mean": "",
                "fake_score_mean": "",
                "sec_per_iter": "",
                "vram_peak_mb": "",
                "eta_sec": "",
                "eta_human": "",
                "eta_finish_at": "",
                "fid": round(current_fid, 4),
                "fid_auc_vs_kimg": round(fid_auc_vs_kimg, 4),
                "kid_mean": round(gan_metrics["kid_mean"], 6),
                "kid_std": round(gan_metrics["kid_std"], 6),
                "precision": round(gan_metrics["precision"], 4),
                "recall": round(gan_metrics["recall"], 4),
                "lpips_diversity": round(gan_metrics["lpips_diversity"], 4),
                "metrics_elapsed_sec": round(elapsed_metrics, 3),
            }
            io.row_logger(metrics_row)

    # Wydrukuj podsumowanie profilowania
    io.write_line(profiler.get_summary())
    
    return {
        "metrics_elapsed_ema": metrics_elapsed_ema,
        "prev_fid_point": prev_fid_point,
        "fid_auc_vs_kimg": fid_auc_vs_kimg,
    }

