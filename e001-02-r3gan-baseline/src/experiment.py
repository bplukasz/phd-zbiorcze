"""
e001-02-r3gan-baseline — experiment training loop.

Korzysta z architektury R3GAN (r3gan-source.py w korzeniu projektu).
Produkuje: gridy, checkpointy, real_samples, samples, logi CSV, config_used.yaml.
"""

import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple, cast
import importlib.util
from pathlib import Path

import torch
from torchvision.utils import save_image

# ---- Load local r3gan-source.py reliably ----------------------------------------
_R3GAN_PATH = Path(__file__).resolve().parents[1] / "r3gan-source.py"
_R3GAN_SPEC = importlib.util.spec_from_file_location("e001_02_r3gan_source", _R3GAN_PATH)
if _R3GAN_SPEC is None or _R3GAN_SPEC.loader is None:
    raise ImportError(f"Could not load R3GAN source module from {_R3GAN_PATH}")
_r3gan = importlib.util.module_from_spec(_R3GAN_SPEC)
sys.modules[_R3GAN_SPEC.name] = _r3gan
_R3GAN_SPEC.loader.exec_module(_r3gan)

R3GANGenerator = _r3gan.R3GANGenerator
R3GANDiscriminator = _r3gan.R3GANDiscriminator
WaveletR3GANDiscriminator = _r3gan.WaveletR3GANDiscriminator
MatchedCapacityR3GANDiscriminator = _r3gan.MatchedCapacityR3GANDiscriminator
R3GANTrainer = _r3gan.R3GANTrainer
TrainerConfig = _r3gan.TrainerConfig
build_stage_channels = _r3gan.build_stage_channels
setup_nvidia_performance = _r3gan.setup_nvidia_performance
parse_batch = _r3gan.parse_batch
WaveReg = _r3gan.WaveReg
FFTReg = _r3gan.FFTReg
from shared.utils import set_seed, CSVLogger

from .data import get_dataloader
from .config_loader import RunConfig, ConfigLoader, get_config
from .gan_metrics import DependencyError, GANMetricsConfig, GANMetricsSuite, format_metrics


# --------------------------------------------------------------------------

def _build_models(cfg: RunConfig) -> Tuple[Any, Any]:
    """Buduje G i D na podstawie konfiguracji."""
    if cfg.wavelet_enabled and cfg.matched_capacity_enabled:
        raise ValueError("wavelet_enabled and matched_capacity_enabled are mutually exclusive")

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
    d_kwargs = dict(
        img_resolution=cfg.img_resolution,
        stage_channels=d_ch,
        blocks_per_stage=cfg.blocks_per_stage,
        expansion_factor=cfg.expansion_factor,
        group_size=cfg.group_size,
        in_channels=cfg.in_channels,
        resample_mode=cfg.resample_mode,
    )
    if cfg.wavelet_enabled:
        D = WaveletR3GANDiscriminator(
            **d_kwargs,
            wavelet_type=cfg.wavelet_type,
            wavelet_level=cfg.wavelet_level,
            wavelet_hf_only=cfg.wavelet_hf_only,
            wavelet_fuse_after_stage=cfg.wavelet_fuse_after_stage,
            wavelet_branch_mid_scale=cfg.wavelet_branch_mid_scale,
            wavelet_init_gate=cfg.wavelet_init_gate,
        )
    elif cfg.matched_capacity_enabled:
        D = MatchedCapacityR3GANDiscriminator(
            **d_kwargs,
            wavelet_fuse_after_stage=cfg.wavelet_fuse_after_stage,
            wavelet_branch_mid_scale=cfg.wavelet_branch_mid_scale,
            wavelet_init_gate=cfg.wavelet_init_gate,
        )
    else:
        D = R3GANDiscriminator(**d_kwargs)
    return G, D


def _step_to_kimg(step: int, batch_size: int) -> float:
    return (step * batch_size) / 1000.0


def _update_fid_auc(
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


def _lerp(start: float, end: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return start + (end - start) * alpha


def _compute_aux_branch_gate(cfg: RunConfig, step: int) -> Optional[float]:
    if not cfg.aux_branch_gate_warmup_enabled:
        return None
    start_step = cfg.aux_branch_gate_warmup_start_step
    end_step = cfg.aux_branch_gate_warmup_end_step
    if end_step <= start_step:
        return float(cfg.aux_branch_gate_warmup_end_value)
    if step <= start_step:
        return float(cfg.aux_branch_gate_warmup_start_value)
    if step >= end_step:
        return float(cfg.aux_branch_gate_warmup_end_value)
    alpha = (step - start_step) / float(end_step - start_step)
    return _lerp(cfg.aux_branch_gate_warmup_start_value, cfg.aux_branch_gate_warmup_end_value, alpha)


def _compute_piecewise_weight(
    base_weight: float,
    schedule_enabled: bool,
    start_step: int,
    peak_step: int,
    end_step: int,
    start_weight: float,
    peak_weight: float,
    end_weight: float,
    step: int,
) -> float:
    if not schedule_enabled:
        return float(base_weight)
    if peak_step <= start_step:
        peak_step = start_step
    if end_step <= peak_step:
        end_step = peak_step
    if step <= start_step:
        return float(start_weight)
    if step <= peak_step:
        if peak_step == start_step:
            return float(peak_weight)
        alpha = (step - start_step) / float(peak_step - start_step)
        return _lerp(start_weight, peak_weight, alpha)
    if step <= end_step:
        if end_step == peak_step:
            return float(end_weight)
        alpha = (step - peak_step) / float(end_step - peak_step)
        return _lerp(peak_weight, end_weight, alpha)
    return float(end_weight)


def _resolve_fid_gated_activation(
    gate_enabled: bool,
    gate_threshold: float,
    gate_min_step: int,
    gate_latched: bool,
    current_step: int,
    last_fid: Optional[float],
    was_active: bool,
) -> Tuple[bool, bool]:
    if not gate_enabled:
        return True, was_active
    if gate_latched and was_active:
        return True, True
    fid_ready = last_fid is not None and last_fid <= gate_threshold
    active_now = current_step >= gate_min_step and fid_ready
    latched_state = was_active or active_now if gate_latched else active_now
    return active_now, latched_state


def _make_csv_logger(out_dir: str) -> CSVLogger:
    fieldnames = [
        "step",
        "kimg",
        "row_type",
        "aux_branch_gate",
        "last_fid_for_gates",
        "wave_reg_weight_eff",
        "wave_reg_active",
        "fft_reg_weight_eff",
        "fft_reg_active",
        "d_loss", "d_adv", "r1", "r2",
        "g_loss", "g_adv", "g_reg",
        "real_score_mean", "fake_score_mean",
        "sec_per_iter",
        "vram_peak_mb",
        # wavelet stat regularizer sub-metrics
        "wave_reg_total", "wave_mu_loss", "wave_std_loss",
        "wave_fake_mu_lh", "wave_fake_mu_hl", "wave_fake_mu_hh",
        "wave_real_mu_lh", "wave_real_mu_hl", "wave_real_mu_hh",
        # FFT stat regularizer sub-metrics
        "fft_reg_total", "fft_mu_loss", "fft_std_loss",
        # GAN metrics
        "fid", "fid_auc_vs_kimg", "kid_mean", "kid_std",
        "precision", "recall",
        "lpips_diversity",
        "rpse", "wbed",
        "metrics_elapsed_sec",
    ]
    return CSVLogger(os.path.join(out_dir, "logs.csv"), fieldnames)


@torch.no_grad()
def _save_grid(trainer: R3GANTrainer, fixed_z: torch.Tensor,
               path: str, n_row: int = 8) -> None:
    trainer.G_ema.eval()
    imgs = trainer.G_ema(fixed_z)             # [-1, 1]
    imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0   # [0, 1]
    save_image(imgs, path, nrow=n_row)


def _save_real_grid(real_imgs: torch.Tensor, path: str, n_row: int = 8) -> None:
    imgs = (real_imgs.clamp(-1, 1) + 1.0) / 2.0
    save_image(imgs[:64], path, nrow=n_row)


def _export_real_samples(dataloader, n: int, out_dir: str) -> None:
    """Zapisuje n prawdziwych próbek jako pojedyncze pliki PNG (do FID/KID)."""
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for batch in dataloader:
        imgs, _ = parse_batch(batch)
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
        for i in range(imgs.size(0)):
            if saved >= n:
                break
            save_image(imgs[i], os.path.join(out_dir, f"{saved:06d}.png"))
            saved += 1
        if saved >= n:
            break
    print(f"Exported {saved} real samples → {out_dir}")


def _export_samples(trainer: R3GANTrainer, n: int, out_dir: str, step: int) -> None:
    """Zapisuje n wygenerowanych próbek jako pojedyncze pliki PNG."""
    os.makedirs(out_dir, exist_ok=True)
    batch_size = 64
    saved = 0
    while saved < n:
        cur = min(batch_size, n - saved)
        imgs = trainer.sample(cur)
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
        for i in range(imgs.size(0)):
            save_image(imgs[i], os.path.join(out_dir, f"step{step:07d}_{saved:06d}.png"))
            saved += 1
    print(f"Saved {saved} fake samples → {out_dir}")


def _validate_metrics_config(cfg: RunConfig, dataset_size: Optional[int]) -> None:
    if cfg.wave_reg_fid_gate_enabled and cfg.metrics_every == 0:
        raise ValueError("wave_reg_fid_gate_enabled requires metrics_every > 0 (FID must be computed)")
    if cfg.fft_reg_fid_gate_enabled and cfg.metrics_every == 0:
        raise ValueError("fft_reg_fid_gate_enabled requires metrics_every > 0 (FID must be computed)")
    if cfg.aux_branch_gate_warmup_enabled and cfg.aux_branch_gate_warmup_end_step < cfg.aux_branch_gate_warmup_start_step:
        raise ValueError("aux_branch_gate_warmup_end_step must be >= aux_branch_gate_warmup_start_step")
    if cfg.wave_reg_schedule_enabled and not (
        cfg.wave_reg_schedule_start_step <= cfg.wave_reg_schedule_peak_step <= cfg.wave_reg_schedule_end_step
    ):
        raise ValueError("wave_reg schedule requires start_step <= peak_step <= end_step")
    if cfg.fft_reg_schedule_enabled and not (
        cfg.fft_reg_schedule_start_step <= cfg.fft_reg_schedule_peak_step <= cfg.fft_reg_schedule_end_step
    ):
        raise ValueError("fft_reg schedule requires start_step <= peak_step <= end_step")

    if cfg.metrics_every < 0:
        raise ValueError("metrics_every must be >= 0")
    if cfg.metrics_every == 0:
        return
    if cfg.metrics_num_fake <= 0:
        raise ValueError("metrics_num_fake must be > 0 when metrics are enabled")
    if cfg.metrics_fake_batch_size <= 0:
        raise ValueError("metrics_fake_batch_size must be > 0 when metrics are enabled")
    if cfg.metrics_pr_num_samples <= cfg.metrics_pr_k:
        raise ValueError("metrics_pr_num_samples must be greater than metrics_pr_k")
    if cfg.metrics_lpips_pool_size < 2:
        raise ValueError("metrics_lpips_pool_size must be >= 2")
    if cfg.metrics_lpips_num_pairs <= 0:
        raise ValueError("metrics_lpips_num_pairs must be > 0")
    if cfg.metrics_amp_dtype not in {"bf16", "fp16"}:
        raise ValueError("metrics_amp_dtype must be one of: bf16, fp16")
    if cfg.metrics_kid_subset_size > cfg.metrics_num_fake:
        raise ValueError("metrics_kid_subset_size cannot be larger than metrics_num_fake")
    if cfg.metrics_max_real < cfg.metrics_kid_subset_size:
        raise ValueError("metrics_max_real cannot be smaller than metrics_kid_subset_size")
    if dataset_size is not None and dataset_size < cfg.metrics_kid_subset_size:
        raise ValueError(
            f"Dataset too small for KID subset_size={cfg.metrics_kid_subset_size}. "
            f"Available real samples: {dataset_size}."
        )
    if cfg.metrics_spectral and cfg.metrics_spectral_num_images < 2:
        raise ValueError("metrics_spectral_num_images must be >= 2 when spectral metrics are enabled")


def _build_metrics_suite(cfg: RunConfig, device: torch.device) -> Optional[GANMetricsSuite]:
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
            spectral_enabled=cfg.metrics_spectral,
            spectral_num_images=cfg.metrics_spectral_num_images,
            spectral_rpse_num_bins=(
                cfg.metrics_spectral_rpse_num_bins if cfg.metrics_spectral_rpse_num_bins > 0 else None
            ),
        )
        return GANMetricsSuite(metrics_cfg)
    except DependencyError as exc:
        raise RuntimeError(
            "GAN metrics are enabled, but required dependencies are missing. "
            "Install requirements from e001-02-r3gan-baseline/requirements.txt or disable metrics with metrics_every=0."
        ) from exc


def _count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _save_model_info(G: Any, D: Any, out_dir: str) -> None:
    """Zapisuje informacje o architekturze (liczba parametrów) do model_info.json."""
    g_info = _count_params(G)
    d_info = _count_params(D)
    info = {
        "generator": {
            "class": type(G).__name__,
            "params_total": g_info["total"],
            "params_trainable": g_info["trainable"],
            "params_total_M": round(g_info["total"] / 1e6, 4),
        },
        "discriminator": {
            "class": type(D).__name__,
            "params_total": d_info["total"],
            "params_trainable": d_info["trainable"],
            "params_total_M": round(d_info["total"] / 1e6, 4),
        },
        "total_params": g_info["total"] + d_info["total"],
        "total_params_M": round((g_info["total"] + d_info["total"]) / 1e6, 4),
    }
    path = os.path.join(out_dir, "model_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved → {path}")
    print(f"  G ({info['generator']['class']}): {info['generator']['params_total_M']:.2f}M params")
    print(f"  D ({info['discriminator']['class']}): {info['discriminator']['params_total_M']:.2f}M params")


# --------------------------------------------------------------------------

def train(profile: str = "base", overrides: Optional[Dict[str, Any]] = None):
    """Główna funkcja treningowa."""
    cfg = get_config(profile, overrides)

    # Seed
    set_seed(cfg.seed, cfg.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_nvidia_performance(device)

    print("=" * 60)
    print("e001-02-r3gan-baseline")
    print(f"Profile : {cfg.name}")
    print(f"Device  : {device}" + (f"  [{torch.cuda.get_device_name(0)}]" if device.type == "cuda" else ""))
    print(f"Dataset : {cfg.dataset_name}  {cfg.img_resolution}×{cfg.img_resolution}")
    print(f"Steps   : {cfg.steps}")
    print(f"Metrics : {'enabled' if cfg.metrics_every > 0 else 'disabled'}"
          + (f"  (every {cfg.metrics_every} steps, fake={cfg.metrics_num_fake})" if cfg.metrics_every > 0 else ""))
    print("=" * 60)

    # Dirs
    out_dir   = cfg.out_dir
    grid_dir  = os.path.join(out_dir, "grids")
    ckpt_dir  = os.path.join(out_dir, "checkpoints")
    samp_dir  = os.path.join(out_dir, "samples")
    real_dir  = os.path.join(out_dir, "real_samples")
    for d in (out_dir, grid_dir, ckpt_dir, samp_dir, real_dir):
        os.makedirs(d, exist_ok=True)

    # Save used config immediately
    ConfigLoader().save_config(cfg, os.path.join(out_dir, "config_used.yaml"))

    # Build models + trainer
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
    trainer = R3GANTrainer(
        G, D,
        device=device,
        train_cfg=train_cfg,
        wave_reg=WaveReg(
            weight=cfg.wave_reg_weight,
            ema_beta=cfg.wave_reg_ema_beta,
            in_channels=cfg.out_channels,
        ) if cfg.wave_reg_enabled else None,
        fft_reg=FFTReg(
            weight=cfg.fft_reg_weight,
            ema_beta=cfg.fft_reg_ema_beta,
            num_bins=cfg.fft_reg_num_bins,
        ) if cfg.fft_reg_enabled else None,
    )

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"G params: {g_params/1e6:.2f}M   D params: {d_params/1e6:.2f}M")
    _save_model_info(G, D, out_dir)

    # Dataloader
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
    _validate_metrics_config(cfg, dataset_size)

    # Sanity check — real grid
    data_iter = iter(dataloader)
    first_batch, _ = parse_batch(next(data_iter))
    print(f"Real data  min={first_batch.min():.3f}  max={first_batch.max():.3f}  mean={first_batch.mean():.3f}")
    _save_real_grid(first_batch, os.path.join(out_dir, "real_grid.png"))
    print("Saved real_grid.png — sprawdź czy dane wyglądają OK!")

    # Export real samples (for metrics / FID)
    print("Eksportuję real_samples…")
    export_real_n = min(10_000, dataset_size) if dataset_size is not None else 10_000
    _export_real_samples(dataloader, n=export_real_n, out_dir=real_dir)

    # Prepare GAN metrics once at startup, so long runs fail fast instead of mid-training.
    metrics_suite = _build_metrics_suite(cfg, device)
    if metrics_suite is not None:
        print(f"Przygotowuję GAN metrics cache na real data (max_real={cfg.metrics_max_real}, pr_samples={cfg.metrics_pr_num_samples})…")
        metrics_suite.prepare_real(dataloader)
        print("GAN metrics cache ready.")

    # CSV logger
    csv_logger = _make_csv_logger(out_dir)

    # Fixed latents for grid visualization (always same images)
    fixed_z = torch.randn(cfg.save_n_samples, G.z_dim, device=device)

    print(f"\nStart training: {cfg.steps} steps  batch={cfg.batch_size}")
    print("-" * 60)

    prev_fid_point: Optional[Tuple[float, float]] = None
    fid_auc_vs_kimg = 0.0
    last_fid_for_gates: Optional[float] = None
    wave_reg_latched_active = False
    fft_reg_latched_active = False

    for step in range(1, cfg.steps + 1):
        t_iter = time.time()
        kimg = _step_to_kimg(step, cfg.batch_size)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Fetch data
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        images, labels = parse_batch(batch)

        # Runtime controls: aux branch gate, independent reg schedules, independent FID gates.
        aux_branch_gate = _compute_aux_branch_gate(cfg, step)
        if aux_branch_gate is not None:
            trainer.set_aux_branch_gate(aux_branch_gate)

        wave_reg_weight_eff = cfg.wave_reg_weight
        wave_reg_active_eff = cfg.wave_reg_enabled
        if cfg.wave_reg_enabled:
            wave_reg_weight_eff = _compute_piecewise_weight(
                base_weight=cfg.wave_reg_weight,
                schedule_enabled=cfg.wave_reg_schedule_enabled,
                start_step=cfg.wave_reg_schedule_start_step,
                peak_step=cfg.wave_reg_schedule_peak_step,
                end_step=cfg.wave_reg_schedule_end_step,
                start_weight=cfg.wave_reg_schedule_start_weight,
                peak_weight=cfg.wave_reg_schedule_peak_weight,
                end_weight=cfg.wave_reg_schedule_end_weight,
                step=step,
            )
            wave_reg_active_eff, wave_reg_latched_active = _resolve_fid_gated_activation(
                gate_enabled=cfg.wave_reg_fid_gate_enabled,
                gate_threshold=cfg.wave_reg_fid_gate_threshold,
                gate_min_step=cfg.wave_reg_fid_gate_min_step,
                gate_latched=cfg.wave_reg_fid_gate_latched,
                current_step=step,
                last_fid=last_fid_for_gates,
                was_active=wave_reg_latched_active,
            )
            trainer.set_wave_reg_weight(wave_reg_weight_eff)
            trainer.set_wave_reg_active(wave_reg_active_eff)

        fft_reg_weight_eff = cfg.fft_reg_weight
        fft_reg_active_eff = cfg.fft_reg_enabled
        if cfg.fft_reg_enabled:
            fft_reg_weight_eff = _compute_piecewise_weight(
                base_weight=cfg.fft_reg_weight,
                schedule_enabled=cfg.fft_reg_schedule_enabled,
                start_step=cfg.fft_reg_schedule_start_step,
                peak_step=cfg.fft_reg_schedule_peak_step,
                end_step=cfg.fft_reg_schedule_end_step,
                start_weight=cfg.fft_reg_schedule_start_weight,
                peak_weight=cfg.fft_reg_schedule_peak_weight,
                end_weight=cfg.fft_reg_schedule_end_weight,
                step=step,
            )
            fft_reg_active_eff, fft_reg_latched_active = _resolve_fid_gated_activation(
                gate_enabled=cfg.fft_reg_fid_gate_enabled,
                gate_threshold=cfg.fft_reg_fid_gate_threshold,
                gate_min_step=cfg.fft_reg_fid_gate_min_step,
                gate_latched=cfg.fft_reg_fid_gate_latched,
                current_step=step,
                last_fid=last_fid_for_gates,
                was_active=fft_reg_latched_active,
            )
            trainer.set_fft_reg_weight(fft_reg_weight_eff)
            trainer.set_fft_reg_active(fft_reg_active_eff)

        metrics = trainer.train_step(images, labels)

        iter_time = time.time() - t_iter
        vram_mb = (
            torch.cuda.max_memory_allocated() / 1e6
            if device.type == "cuda" else 0.0
        )

        # --- CSV log ---
        if step % cfg.log_every == 0:
            row = {
                "step": step,
                "kimg": round(kimg, 4),
                "row_type": "train",
                "aux_branch_gate": round(aux_branch_gate, 6) if aux_branch_gate is not None else "",
                "last_fid_for_gates": round(last_fid_for_gates, 6) if last_fid_for_gates is not None else "",
                "wave_reg_weight_eff": round(wave_reg_weight_eff, 6) if cfg.wave_reg_enabled else "",
                "wave_reg_active": int(bool(wave_reg_active_eff)) if cfg.wave_reg_enabled else "",
                "fft_reg_weight_eff": round(fft_reg_weight_eff, 6) if cfg.fft_reg_enabled else "",
                "fft_reg_active": int(bool(fft_reg_active_eff)) if cfg.fft_reg_enabled else "",
                "d_loss": metrics.get("d_loss", 0.0),
                "d_adv":  metrics.get("d_adv",  0.0),
                "r1":     metrics.get("r1",      0.0),
                "r2":     metrics.get("r2",      0.0),
                "g_loss": metrics.get("g_loss",  0.0),
                "g_adv":  metrics.get("g_adv",   0.0),
                "g_reg":  metrics.get("g_reg",   0.0),
                "real_score_mean": metrics.get("real_score_mean", 0.0),
                "fake_score_mean": metrics.get("fake_score_mean", 0.0),
                "sec_per_iter": round(iter_time, 4),
                "vram_peak_mb": round(vram_mb, 1),
                # wavelet regularizer sub-metrics (empty when disabled)
                "wave_reg_total":  metrics.get("wave_reg_total",  ""),
                "wave_mu_loss":    metrics.get("wave_mu_loss",    ""),
                "wave_std_loss":   metrics.get("wave_std_loss",   ""),
                "wave_fake_mu_lh": metrics.get("wave_fake_mu_lh", ""),
                "wave_fake_mu_hl": metrics.get("wave_fake_mu_hl", ""),
                "wave_fake_mu_hh": metrics.get("wave_fake_mu_hh", ""),
                "wave_real_mu_lh": metrics.get("wave_real_mu_lh", ""),
                "wave_real_mu_hl": metrics.get("wave_real_mu_hl", ""),
                "wave_real_mu_hh": metrics.get("wave_real_mu_hh", ""),
                # FFT regularizer sub-metrics (empty when disabled)
                "fft_reg_total":   metrics.get("fft_reg_total",   ""),
                "fft_mu_loss":     metrics.get("fft_mu_loss",     ""),
                "fft_std_loss":    metrics.get("fft_std_loss",    ""),
                "fid": "",
                "fid_auc_vs_kimg": "",
                "kid_mean": "",
                "kid_std": "",
                "precision": "",
                "recall": "",
                "lpips_diversity": "",
                "rpse": "",
                "wbed": "",
                "metrics_elapsed_sec": "",
            }
            csv_logger.log(row)
            print(
                f"[{step:>7d}/{cfg.steps}]  "
                f"d={row['d_loss']:.4f}  g={row['g_loss']:.4f}  "
                f"g_adv={row['g_adv']:.4f}  g_reg={row['g_reg']:.4f}  "
                f"r1={row['r1']:.4f}  r2={row['r2']:.4f}  "
                f"{iter_time*1000:.0f}ms"
                + (f"  {vram_mb:.0f}MB" if device.type == "cuda" else "")
            )

        # --- Grid ---
        if step % cfg.grid_every == 0:
            grid_path = os.path.join(grid_dir, f"grid_{step:07d}.png")
            _save_grid(trainer, fixed_z, grid_path)
            print(f"  → grid saved: {grid_path}")

        # --- Checkpoint ---
        if step % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{step:07d}.pt")
            torch.save({
                "step": step,
                "G": trainer.G.state_dict(),
                "D": trainer.D.state_dict(),
                "G_ema": trainer.G_ema.state_dict(),
                "g_opt": trainer.g_opt.state_dict(),
                "d_opt": trainer.d_opt.state_dict(),
                "cfg": cfg.to_dict(),
            }, ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path}")

            # Also save a few generated samples next to checkpoint
            _export_samples(trainer, n=256, out_dir=samp_dir, step=step)

        # --- GAN Metrics ---
        if metrics_suite is not None and cfg.metrics_every > 0 and step % cfg.metrics_every == 0:
            print(f"\n[{step:>7d}/{cfg.steps}]  Obliczam metryki GAN ({cfg.metrics_num_fake} fake images)…")
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
            current_fid = float(gan_metrics["fid"])
            last_fid_for_gates = current_fid
            prev_fid_point, fid_auc_vs_kimg = _update_fid_auc(
                prev_fid_point,
                kimg,
                current_fid,
                fid_auc_vs_kimg,
            )
            print(
                f"[{step:>7d}/{cfg.steps}]  METRICS  {format_metrics(gan_metrics)}"
                f"  fid_auc_vs_kimg={fid_auc_vs_kimg:.4f}"
                f"  ({elapsed_metrics:.1f}s)"
            )
            # Log metrics to CSV as a separate row
            metrics_row = {
                "step": step,
                "kimg": round(kimg, 4),
                "row_type": "gan_metrics",
                "aux_branch_gate": "",
                "last_fid_for_gates": round(last_fid_for_gates, 6) if last_fid_for_gates is not None else "",
                "wave_reg_weight_eff": "",
                "wave_reg_active": "",
                "fft_reg_weight_eff": "",
                "fft_reg_active": "",
                "d_loss": "", "d_adv": "", "r1": "", "r2": "", "g_loss": "", "g_adv": "", "g_reg": "",
                "real_score_mean": "", "fake_score_mean": "",
                "sec_per_iter": "", "vram_peak_mb": "",
                "wave_reg_total": "", "wave_mu_loss": "", "wave_std_loss": "",
                "wave_fake_mu_lh": "", "wave_fake_mu_hl": "", "wave_fake_mu_hh": "",
                "wave_real_mu_lh": "", "wave_real_mu_hl": "", "wave_real_mu_hh": "",
                "fft_reg_total": "", "fft_mu_loss": "", "fft_std_loss": "",
                "fid": round(current_fid, 4),
                "fid_auc_vs_kimg": round(fid_auc_vs_kimg, 4),
                "kid_mean": round(gan_metrics["kid_mean"], 6),
                "kid_std": round(gan_metrics["kid_std"], 6),
                "precision": round(gan_metrics["precision"], 4),
                "recall": round(gan_metrics["recall"], 4),
                "lpips_diversity": round(gan_metrics["lpips_diversity"], 4),
                "rpse": round(gan_metrics["rpse"], 6) if "rpse" in gan_metrics else "",
                "wbed": round(gan_metrics["wbed"], 6) if "wbed" in gan_metrics else "",
                "metrics_elapsed_sec": round(elapsed_metrics, 3),
            }
            csv_logger.log(metrics_row)

    # --- Final save ---
    final_ckpt = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "step": cfg.steps,
        "G": trainer.G.state_dict(),
        "D": trainer.D.state_dict(),
        "G_ema": trainer.G_ema.state_dict(),
        "g_opt": trainer.g_opt.state_dict(),
        "d_opt": trainer.d_opt.state_dict(),
        "cfg": cfg.to_dict(),
    }, final_ckpt)
    print(f"\nDone! Final checkpoint: {final_ckpt}")

    # Final grid
    _save_grid(trainer, fixed_z, os.path.join(grid_dir, "grid_final.png"))
    _export_samples(trainer, n=1024, out_dir=samp_dir, step=cfg.steps)

    return trainer

