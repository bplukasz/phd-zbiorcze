#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "e001-02-r3gan-baseline"
OUT_DIR = EXP_ROOT / "summary" / "r0-r11"
MAX_RUN_ID = 11

plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class RunData:
    artifact_dir: Path
    run_id: str
    run_num: int
    config_name: str
    config: dict[str, Any]
    train_df: pd.DataFrame
    metrics_df: pd.DataFrame


def parse_run_id(config_name: str) -> str | None:
    for part in config_name.split("_"):
        if part.startswith("r") and part[1:].isdigit():
            return part
    return None


def _run_num(run_id: str) -> int | None:
    if run_id.startswith("r") and run_id[1:].isdigit():
        return int(run_id[1:])
    return None


def load_runs(max_run_id: int = MAX_RUN_ID) -> list[RunData]:
    runs: list[RunData] = []
    for artifact_dir in sorted(EXP_ROOT.glob("artifacts-03-*-phase_*_32")):
        log_path = artifact_dir / "logs.csv"
        cfg_path = artifact_dir / "config_used.yaml"
        if not log_path.exists() or not cfg_path.exists():
            continue

        with cfg_path.open("r", encoding="utf-8") as handle:
            config = yaml.full_load(handle) or {}
        config_name = str(config.get("name", artifact_dir.name))

        run_id = parse_run_id(config_name)
        if run_id is None:
            continue
        run_num = _run_num(run_id)
        if run_num is None or run_num > max_run_id:
            continue

        df = pd.read_csv(log_path)
        if "row_type" not in df.columns:
            continue

        train_df = df.loc[df["row_type"] == "train"].copy().reset_index(drop=True)
        metrics_df = df.loc[df["row_type"] != "train"].copy().reset_index(drop=True)

        for col in train_df.columns:
            if col != "row_type":
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        for col in metrics_df.columns:
            if col != "row_type":
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        batch_size = int(config.get("batch_size", 128))
        if "step" in metrics_df.columns:
            metrics_df["kimg"] = pd.to_numeric(metrics_df["step"], errors="coerce") * batch_size / 1000.0

        runs.append(
            RunData(
                artifact_dir=artifact_dir,
                run_id=run_id,
                run_num=run_num,
                config_name=config_name,
                config=config,
                train_df=train_df,
                metrics_df=metrics_df,
            )
        )

    runs.sort(key=lambda r: r.run_num)
    return runs


def _final_value(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _best_min(df: pd.DataFrame, col: str) -> tuple[float | None, int | None]:
    if col not in df.columns:
        return None, None
    series = pd.to_numeric(df[col], errors="coerce")
    if series.dropna().empty:
        return None, None
    idx = int(series.idxmin())
    step = int(pd.to_numeric(df.loc[idx, "step"], errors="coerce")) if "step" in df.columns else None
    return float(series.loc[idx]), step


def _auc_fid(metrics_df: pd.DataFrame) -> float | None:
    if "fid" not in metrics_df.columns or "kimg" not in metrics_df.columns:
        return None
    sub = metrics_df[["kimg", "fid"]].copy()
    sub["kimg"] = pd.to_numeric(sub["kimg"], errors="coerce")
    sub["fid"] = pd.to_numeric(sub["fid"], errors="coerce")
    sub = sub.dropna().sort_values("kimg")
    if len(sub) < 2:
        return None
    width = float(sub["kimg"].iloc[-1] - sub["kimg"].iloc[0])
    if width <= 0:
        return None
    area = float(np.trapezoid(sub["fid"].to_numpy(), sub["kimg"].to_numpy()))
    return area / width


def build_summary_table(runs: list[RunData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        train = run.train_df
        metrics = run.metrics_df

        best_wbed, best_wbed_step = _best_min(metrics, "wbed")
        best_rpse, best_rpse_step = _best_min(metrics, "rpse")

        row = {
            "run_id": run.run_id,
            "config_name": run.config_name,
            "final_fid": _final_value(metrics, "fid"),
            "final_kid": _final_value(metrics, "kid_mean"),
            "final_precision": _final_value(metrics, "precision"),
            "final_recall": _final_value(metrics, "recall"),
            "final_lpips": _final_value(metrics, "lpips_diversity"),
            "final_rpse": _final_value(metrics, "rpse"),
            "final_wbed": _final_value(metrics, "wbed"),
            "best_wbed": best_wbed,
            "best_wbed_step": best_wbed_step,
            "best_rpse": best_rpse,
            "best_rpse_step": best_rpse_step,
            "fid_auc_vs_kimg": _auc_fid(metrics),
            "avg_sec_per_iter": None,
            "avg_vram_peak_mb": None,
        }

        if "sec_per_iter" in train.columns:
            sec = pd.to_numeric(train["sec_per_iter"], errors="coerce").dropna()
            row["avg_sec_per_iter"] = float(sec.mean()) if not sec.empty else None
        if "vram_peak_mb" in train.columns:
            vr = pd.to_numeric(train["vram_peak_mb"], errors="coerce").dropna()
            row["avg_vram_peak_mb"] = float(vr.mean()) if not vr.empty else None

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("run_id", key=lambda s: s.str[1:].astype(int))
    if "r0" in set(summary["run_id"]):
        baseline_fid = float(summary.loc[summary["run_id"] == "r0", "final_fid"].iloc[0])
        summary["delta_fid_vs_r0"] = summary["final_fid"] - baseline_fid
    return summary.reset_index(drop=True)


def _save(fig: plt.Figure, filename: str) -> str:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_fid_kid_trajectories(runs: list[RunData]) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for run in runs:
        m = run.metrics_df
        if "kimg" not in m.columns:
            continue
        if "fid" in m.columns and pd.to_numeric(m["fid"], errors="coerce").notna().any():
            axes[0].plot(m["kimg"], m["fid"], marker="o", linewidth=1.6, label=run.run_id)
        if "kid_mean" in m.columns and pd.to_numeric(m["kid_mean"], errors="coerce").notna().any():
            axes[1].plot(m["kimg"], m["kid_mean"], marker="o", linewidth=1.6, label=run.run_id)

    axes[0].set_title("Trajektorie FID (r0-r11)")
    axes[0].set_ylabel("FID (mniej = lepiej)")
    axes[1].set_title("Trajektorie KID mean (r0-r11)")
    axes[1].set_ylabel("KID mean (mniej = lepiej)")
    axes[1].set_xlabel("kimg")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].legend(ncol=4, fontsize=8)
    return _save(fig, "fid_kid_trajectories.png")


def plot_quality_tradeoff(summary: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x1 = pd.to_numeric(summary["avg_sec_per_iter"], errors="coerce")
    x2 = pd.to_numeric(summary["avg_vram_peak_mb"], errors="coerce")
    y = pd.to_numeric(summary["final_fid"], errors="coerce")

    axes[0].scatter(x1, y, s=80)
    axes[1].scatter(x2, y, s=80)

    for _, row in summary.iterrows():
        axes[0].annotate(row["run_id"], (row["avg_sec_per_iter"], row["final_fid"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        axes[1].annotate(row["run_id"], (row["avg_vram_peak_mb"], row["final_fid"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    axes[0].set_title("Quality-cost: FID vs sec/iter")
    axes[0].set_xlabel("avg sec/iter")
    axes[0].set_ylabel("final FID")
    axes[1].set_title("Quality-cost: FID vs VRAM")
    axes[1].set_xlabel("avg peak VRAM [MB]")
    axes[1].set_ylabel("final FID")
    return _save(fig, "quality_cost_tradeoff.png")


def plot_final_pr_div(summary: pd.DataFrame) -> str:
    runs = summary["run_id"].tolist()
    x = np.arange(len(runs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, summary["final_precision"], width=width, label="precision")
    ax.bar(x, summary["final_recall"], width=width, label="recall")
    ax.bar(x + width, summary["final_lpips"], width=width, label="lpips_diversity")
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_title("Final precision / recall / LPIPS diversity")
    ax.set_ylabel("value")
    ax.legend()
    return _save(fig, "final_pr_lpips.png")


def plot_spectral(summary: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(summary["final_wbed"], summary["final_fid"], s=85)
    axes[1].scatter(summary["final_rpse"], summary["final_fid"], s=85)

    for _, row in summary.iterrows():
        axes[0].annotate(row["run_id"], (row["final_wbed"], row["final_fid"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        axes[1].annotate(row["run_id"], (row["final_rpse"], row["final_fid"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    axes[0].set_title("Final WBED vs final FID")
    axes[0].set_xlabel("WBED (mniej = lepiej)")
    axes[0].set_ylabel("FID (mniej = lepiej)")
    axes[1].set_title("Final RPSE vs final FID")
    axes[1].set_xlabel("RPSE (mniej = lepiej)")
    axes[1].set_ylabel("FID (mniej = lepiej)")
    return _save(fig, "spectral_vs_fid.png")


def plot_regularizer_dynamics(runs: list[RunData]) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for run in runs:
        t = run.train_df
        if "wave_reg_total" in t.columns and pd.to_numeric(t["wave_reg_total"], errors="coerce").notna().any():
            axes[0].plot(t["step"], t["wave_reg_total"], linewidth=1.6, label=run.run_id)
        if "fft_reg_total" in t.columns and pd.to_numeric(t["fft_reg_total"], errors="coerce").notna().any():
            axes[1].plot(t["step"], t["fft_reg_total"], linewidth=1.6, label=run.run_id)

    axes[0].set_title("Wave regularization dynamics")
    axes[0].set_ylabel("wave_reg_total")
    axes[1].set_title("FFT regularization dynamics")
    axes[1].set_ylabel("fft_reg_total")
    axes[1].set_xlabel("step")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    return _save(fig, "regularizer_dynamics.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs(MAX_RUN_ID)
    if not runs:
        raise SystemExit("Nie znaleziono runow r0-r11.")

    summary = build_summary_table(runs)
    summary_path = OUT_DIR / "run_summary_r0_r11.csv"
    summary.to_csv(summary_path, index=False)

    plot_files = [
        plot_fid_kid_trajectories(runs),
        plot_quality_tradeoff(summary),
        plot_final_pr_div(summary),
        plot_spectral(summary),
        plot_regularizer_dynamics(runs),
    ]

    print(f"Saved: {summary_path.relative_to(ROOT)}")
    for item in plot_files:
        print(f"Saved: {(OUT_DIR / item).relative_to(ROOT)}")


if __name__ == "__main__":
    main()

