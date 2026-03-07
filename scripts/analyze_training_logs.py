#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class RunSummary:
    log_path: Path
    run_dir: Path
    run_name: str
    experiment_name: str
    config: dict[str, Any]
    train_df: pd.DataFrame
    metrics_df: pd.DataFrame
    average_sec_per_iter: float | None
    average_vram_peak_mb: float | None
    best_fid: float | None
    best_fid_step: int | None
    final_fid: float | None
    best_kid: float | None
    best_kid_step: int | None
    final_kid: float | None
    final_step: int | None
    output_dir: Path


def discover_logs(root: Path) -> list[Path]:
    return sorted(root.rglob('logs.csv'))


def read_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / 'config_used.yaml'
    if not config_path.exists():
        return {}
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.full_load(handle) or {}
    return data if isinstance(data, dict) else {}


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors='coerce')
    return out


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_candidates = [column for column in df.columns if column != 'row_type']
    df = ensure_numeric(df, numeric_candidates)
    if 'row_type' not in df.columns:
        metrics_cols = [
            c for c in [
                'fid', 'kid', 'kid_mean', 'kid_std', 'precision', 'recall', 'lpips_diversity',
                'rpse', 'wbed_total', 'wavereg_loss', 'fftreg_loss', 'fftreg_time_ms'
            ]
            if c in df.columns
        ]
        metrics_df = df[['step', *metrics_cols]].copy() if metrics_cols else pd.DataFrame(columns=['step'])
        if not metrics_df.empty:
            has_metrics = metrics_df.drop(columns=['step']).notna().any(axis=1)
            metrics_df = metrics_df.loc[has_metrics].reset_index(drop=True)
        return df.reset_index(drop=True), metrics_df.reset_index(drop=True)

    row_type = df['row_type'].fillna('train')
    train_df = df.loc[row_type == 'train'].copy().reset_index(drop=True)
    metrics_df = df.loc[row_type != 'train'].copy().reset_index(drop=True)
    if not metrics_df.empty:
        keep_cols = [
            c for c in [
                'step', 'fid', 'kid', 'kid_mean', 'kid_std', 'precision', 'recall',
                'lpips_diversity', 'metrics_elapsed_sec', 'rpse', 'wbed_total',
                'wavereg_loss', 'fftreg_loss', 'fftreg_time_ms'
            ]
            if c in metrics_df.columns
        ]
        metrics_df = metrics_df[keep_cols]
    return train_df, metrics_df


def rolling(series: pd.Series, frac: float = 0.08, min_window: int = 5, max_window: int = 51) -> pd.Series:
    valid_len = len(series.dropna())
    if valid_len <= 2:
        return series
    window = max(min_window, min(max_window, int(math.ceil(valid_len * frac))))
    if window % 2 == 0:
        window += 1
    return series.rolling(window=window, min_periods=max(2, window // 3)).mean()


def metric_series(metrics_df: pd.DataFrame, preferred: str, fallbacks: list[str] | None = None) -> tuple[str | None, pd.Series | None]:
    for name in [preferred, *(fallbacks or [])]:
        if name in metrics_df.columns:
            return name, metrics_df[name]
    return None, None


def fmt_number(value: float | int | None, digits: int = 4, suffix: str = '') -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 'n/a'
    return f'{value:.{digits}f}{suffix}' if isinstance(value, float) else f'{value}{suffix}'


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def relative_markdown_path(target: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(target, start=base_dir)).as_posix()


def markdown_figure_block(title: str, image_path: Path, report_dir: Path) -> list[str]:
    rel_path = relative_markdown_path(image_path, report_dir)
    return [
        f'### {title}',
        '',
        f'[Otw\u00f3rz obraz]({rel_path})',
        '',
        f'<img src="{rel_path}" alt="{title}" style="max-width: 100%;" />',
        '',
    ]


def plot_losses(train_df: pd.DataFrame, output_dir: Path, run_type: str) -> list[str]:
    files: list[str] = []
    if run_type == 'wavelets':
        pairs = [('loss_D', 'Discriminator'), ('loss_G', 'Generator')]
    else:
        pairs = [('d_loss', 'Discriminator'), ('g_loss', 'Generator')]
    existing = [(column, label) for column, label in pairs if column in train_df.columns]
    if not existing:
        return files
    fig, ax = plt.subplots(figsize=(10, 5))
    for column, label in existing:
        series = pd.to_numeric(train_df[column], errors='coerce')
        ax.plot(train_df['step'], series, alpha=0.25, linewidth=1.0, label=f'{label} (raw)')
        ax.plot(train_df['step'], rolling(series), linewidth=2.0, label=f'{label} (smooth)')
    ax.set_title('Loss trajectories')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend(ncol=2, fontsize=9)
    path = output_dir / 'losses.png'
    save_figure(fig, path)
    files.append(path.name)
    return files


def plot_efficiency(train_df: pd.DataFrame, output_dir: Path) -> list[str]:
    metrics = [metric for metric in ['sec_per_iter', 'vram_peak_mb'] if metric in train_df.columns]
    if not metrics:
        return []
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.6 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    labels = {
        'sec_per_iter': 'Seconds / iteration',
        'vram_peak_mb': 'Peak VRAM [MB]',
    }
    for ax, metric in zip(axes, metrics):
        series = pd.to_numeric(train_df[metric], errors='coerce')
        ax.plot(train_df['step'], series, alpha=0.3, label='raw')
        ax.plot(train_df['step'], rolling(series), linewidth=2.0, label='smooth')
        ax.set_ylabel(labels.get(metric, metric))
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('Step')
    axes[0].set_title('Efficiency profile')
    path = output_dir / 'efficiency.png'
    save_figure(fig, path)
    return [path.name]


def plot_wavelet_specific(train_df: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: Path) -> list[str]:
    files: list[str] = []
    regularizers = [metric for metric in ['rpse', 'wbed_total', 'wavereg_loss', 'fftreg_loss'] if metric in train_df.columns or metric in metrics_df.columns]
    if regularizers:
        fig, ax = plt.subplots(figsize=(10, 5))
        base_df = train_df if any(metric in train_df.columns for metric in regularizers) else metrics_df
        for metric in regularizers:
            if metric not in base_df.columns:
                continue
            series = pd.to_numeric(base_df[metric], errors='coerce')
            if series.notna().sum() == 0:
                continue
            ax.plot(base_df['step'], series, marker='o', linewidth=1.8, label=metric)
        ax.set_title('Wavelet / spectral metrics')
        ax.set_xlabel('Step')
        ax.set_ylabel('Metric value')
        ax.legend(fontsize=9)
        path = output_dir / 'wavelet_metrics.png'
        save_figure(fig, path)
        files.append(path.name)

    band_columns = [c for c in train_df.columns if c.startswith('wavereg_mu_diff_') or c.startswith('wavereg_std_diff_')]
    if band_columns:
        available = [c for c in band_columns if train_df[c].notna().sum() > 0]
        if available:
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            for prefix, ax in [('wavereg_mu_diff_', axes[0]), ('wavereg_std_diff_', axes[1])]:
                subset = [c for c in available if c.startswith(prefix)]
                for column in subset:
                    band = column.replace(prefix, '')
                    ax.plot(train_df['step'], train_df[column], marker='o', linewidth=1.6, label=band)
                ax.set_ylabel(prefix.replace('wavereg_', '').replace('_', ' '))
                ax.legend(fontsize=9)
            axes[0].set_title('Wavelet band mismatch')
            axes[-1].set_xlabel('Step')
            path = output_dir / 'wavelet_band_diffs.png'
            save_figure(fig, path)
            files.append(path.name)
    return files


def plot_r3gan_specific(train_df: pd.DataFrame, output_dir: Path) -> list[str]:
    files: list[str] = []
    score_cols = [c for c in ['real_score_mean', 'fake_score_mean'] if c in train_df.columns]
    reg_cols = [c for c in ['d_adv', 'r1', 'r2'] if c in train_df.columns]
    if score_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for column in score_cols:
            series = pd.to_numeric(train_df[column], errors='coerce')
            ax.plot(train_df['step'], series, alpha=0.25, linewidth=1.0, label=f'{column} (raw)')
            ax.plot(train_df['step'], rolling(series), linewidth=2.0, label=f'{column} (smooth)')
        ax.set_title('Discriminator score dynamics')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        ax.legend(fontsize=9, ncol=2)
        path = output_dir / 'score_dynamics.png'
        save_figure(fig, path)
        files.append(path.name)
    if reg_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for column in reg_cols:
            series = pd.to_numeric(train_df[column], errors='coerce')
            ax.plot(train_df['step'], series, alpha=0.2, linewidth=1.0, label=f'{column} (raw)')
            ax.plot(train_df['step'], rolling(series), linewidth=2.0, label=f'{column} (smooth)')
        ax.set_title('Regularization and adversarial components')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=9, ncol=2)
        path = output_dir / 'regularization.png'
        save_figure(fig, path)
        files.append(path.name)
    return files


def plot_quality_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> list[str]:
    if metrics_df.empty:
        return []
    chart_specs = [
        ('fid', ['fid'], 'FID'),
        ('kid', ['kid', 'kid_mean'], 'KID'),
        ('pr_diversity', ['precision', 'recall', 'lpips_diversity'], 'Precision / recall / diversity'),
    ]
    files: list[str] = []
    for stem, candidates, title in chart_specs:
        available = [c for c in candidates if c in metrics_df.columns and metrics_df[c].notna().sum() > 0]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for column in available:
            ax.plot(metrics_df['step'], metrics_df[column], marker='o', linewidth=2.0, label=column)
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=9)
        path = output_dir / f'{stem}.png'
        save_figure(fig, path)
        files.append(path.name)
    return files


def summarize_run(log_path: Path) -> RunSummary:
    run_dir = log_path.parent
    with log_path.open('r', encoding='utf-8') as handle:
        df = cast(pd.DataFrame, pd.read_csv(handle))  # type: ignore[call-overload]
    train_df, metrics_df = split_frames(df)
    config = read_config(run_dir)
    experiment_name = run_dir.parent.name
    run_name = run_dir.name if run_dir.name != 'artifacts' else f'{experiment_name}/{run_dir.name}'
    output_dir = run_dir / 'summary'
    output_dir.mkdir(exist_ok=True)

    kid_name, kid_series = metric_series(metrics_df, 'kid', ['kid_mean'])
    fid_series = metrics_df['fid'] if 'fid' in metrics_df.columns else None

    def best_metric(series: pd.Series | None) -> tuple[float | None, int | None, float | None]:
        if series is None or series.notna().sum() == 0:
            return None, None, None
        valid = metrics_df.loc[series.notna(), ['step']].copy()
        valid['metric'] = series.loc[series.notna()].to_numpy()
        best_row = valid.sort_values('metric', ascending=True).iloc[0]
        return float(best_row['metric']), int(best_row['step']), float(valid.iloc[-1]['metric'])

    best_fid, best_fid_step, final_fid = best_metric(fid_series)
    best_kid, best_kid_step, final_kid = best_metric(kid_series)

    avg_sec = float(train_df['sec_per_iter'].dropna().mean()) if 'sec_per_iter' in train_df.columns and train_df['sec_per_iter'].notna().any() else None
    avg_vram = float(train_df['vram_peak_mb'].dropna().mean()) if 'vram_peak_mb' in train_df.columns and train_df['vram_peak_mb'].notna().any() else None
    final_step = int(train_df['step'].dropna().max()) if 'step' in train_df.columns and train_df['step'].notna().any() else None

    summary = RunSummary(
        log_path=log_path,
        run_dir=run_dir,
        run_name=run_name,
        experiment_name=experiment_name,
        config=config,
        train_df=train_df,
        metrics_df=metrics_df,
        average_sec_per_iter=avg_sec,
        average_vram_peak_mb=avg_vram,
        best_fid=best_fid,
        best_fid_step=best_fid_step,
        final_fid=final_fid,
        best_kid=best_kid,
        best_kid_step=best_kid_step,
        final_kid=final_kid,
        final_step=final_step,
        output_dir=output_dir,
    )
    return summary


def write_key_stats(summary: RunSummary) -> None:
    train_df = summary.train_df
    metrics_df = summary.metrics_df
    stats: list[dict[str, Any]] = [
        {'metric': 'final_step', 'value': summary.final_step},
        {'metric': 'train_rows', 'value': len(train_df)},
        {'metric': 'metrics_rows', 'value': len(metrics_df)},
        {'metric': 'avg_sec_per_iter', 'value': summary.average_sec_per_iter},
        {'metric': 'avg_vram_peak_mb', 'value': summary.average_vram_peak_mb},
        {'metric': 'best_fid', 'value': summary.best_fid},
        {'metric': 'best_fid_step', 'value': summary.best_fid_step},
        {'metric': 'final_fid', 'value': summary.final_fid},
        {'metric': 'best_kid', 'value': summary.best_kid},
        {'metric': 'best_kid_step', 'value': summary.best_kid_step},
        {'metric': 'final_kid', 'value': summary.final_kid},
    ]
    if 'metrics_elapsed_sec' in metrics_df.columns and metrics_df['metrics_elapsed_sec'].notna().any():
        stats.append({'metric': 'avg_metrics_elapsed_sec', 'value': float(metrics_df['metrics_elapsed_sec'].dropna().mean())})
    pd.DataFrame(stats).to_csv(summary.output_dir / 'key_stats.csv', index=False)


def write_report(summary: RunSummary, generated_files: list[str]) -> None:
    cfg = summary.config
    train_df = summary.train_df
    metrics_df = summary.metrics_df
    run_type = 'wavelets' if 'loss_D' in train_df.columns else 'r3gan'

    config_fields = []
    for key in ['name', 'dataset_name', 'steps', 'batch_size', 'img_size', 'img_resolution', 'z_dim', 'log_every', 'eval_every', 'metrics_every']:
        if key in cfg:
            config_fields.append((key, cfg[key]))

    insights: list[str] = []
    if summary.best_fid is not None:
        insights.append(
            f'- Najlepszy FID: **{summary.best_fid:.4f}** przy kroku **{summary.best_fid_step}**; końcowy FID z logu: **{summary.final_fid:.4f}**.'
        )
    if summary.best_kid is not None:
        insights.append(
            f'- Najlepszy KID: **{summary.best_kid:.6f}** przy kroku **{summary.best_kid_step}**; końcowy KID z logu: **{summary.final_kid:.6f}**.'
        )
    if summary.average_sec_per_iter is not None:
        insights.append(f'- Średni czas iteracji: **{summary.average_sec_per_iter:.4f} s/iter**.')
    if summary.average_vram_peak_mb is not None:
        insights.append(f'- Średni peak VRAM: **{summary.average_vram_peak_mb:.1f} MB**.')

    if run_type == 'wavelets':
        if 'loss_G' in train_df.columns and train_df['loss_G'].notna().any():
            start_g = float(train_df['loss_G'].dropna().iloc[0])
            end_g = float(train_df['loss_G'].dropna().iloc[-1])
            insights.append(f'- `loss_G` zmienił się z **{start_g:.4f}** na **{end_g:.4f}**, co dobrze pokazuje przejście z wczesnej niestabilności do późniejszej stabilizacji.')
        if 'rpse' in metrics_df.columns and metrics_df['rpse'].notna().any():
            best_rpse_idx = metrics_df['rpse'].idxmin()
            best_rpse = float(metrics_df.loc[best_rpse_idx, 'rpse'])
            best_rpse_step = int(metrics_df.loc[best_rpse_idx, 'step'])
            insights.append(f'- Najniższe `rpse` wynosi **{best_rpse:.6f}** przy kroku **{best_rpse_step}**.')
    else:
        if 'g_loss' in train_df.columns and train_df['g_loss'].notna().any():
            start_g = float(train_df['g_loss'].dropna().iloc[0])
            end_g = float(train_df['g_loss'].dropna().iloc[-1])
            insights.append(f'- `g_loss` przeszedł z **{start_g:.4f}** na **{end_g:.4f}**; warto czytać to razem z dynamiką `real_score_mean` / `fake_score_mean`.')
        if 'precision' in metrics_df.columns and metrics_df['precision'].notna().any():
            best_prec_idx = metrics_df['precision'].idxmax()
            best_prec = float(metrics_df.loc[best_prec_idx, 'precision'])
            best_prec_step = int(metrics_df.loc[best_prec_idx, 'step'])
            insights.append(f'- Najwyższa precyzja generatora (`precision`) to **{best_prec:.4f}** przy kroku **{best_prec_step}**.')
        if 'lpips_diversity' in metrics_df.columns and metrics_df['lpips_diversity'].notna().any():
            best_div_idx = metrics_df['lpips_diversity'].idxmax()
            best_div = float(metrics_df.loc[best_div_idx, 'lpips_diversity'])
            best_div_step = int(metrics_df.loc[best_div_idx, 'step'])
            insights.append(f'- Największa r\u00f3\u017cnorodno\u015b\u0107 (`lpips_diversity`) pojawia si\u0119 przy kroku **{best_div_step}** i wynosi **{best_div:.4f}**.')

    sections = [
        f'# Summary for `{summary.run_name}`',
        '',
        f'- Eksperyment: `{summary.experiment_name}`',
        f'- \u0179r\u00f3d\u0142owy log: `{summary.log_path.relative_to(ROOT)}`',
        f'- Typ runu: `{run_type}`',
        f'- Liczba wierszy treningowych: **{len(train_df)}**',
        f'- Liczba snapshot\u00f3w metryk: **{len(metrics_df)}**',
        '',
        '## Konfiguracja (wybrane pola)',
        '',
    ]
    if config_fields:
        sections.extend([f'- `{key}`: `{value}`' for key, value in config_fields])
    else:
        sections.append('- Brak `config_used.yaml` lub brak odczytywalnych pól.')
    sections.extend(['', '## Najważniejsze obserwacje', ''])
    sections.extend(insights or ['- Brak wystarczających metryk do automatycznego wniosku.'])
    sections.extend(['', '## Wygenerowane wykresy', ''])
    for filename in generated_files:
        title = filename.replace('.png', '').replace('_', ' ').title()
        sections.extend(markdown_figure_block(title, summary.output_dir / filename, summary.output_dir))
    report_path = summary.output_dir / 'report.md'
    report_path.write_text('\n'.join(sections).strip() + '\n', encoding='utf-8')


def generate_run_outputs(summary: RunSummary) -> None:
    run_type = 'wavelets' if 'loss_D' in summary.train_df.columns else 'r3gan'
    generated_files: list[str] = []
    generated_files.extend(plot_losses(summary.train_df, summary.output_dir, run_type))
    generated_files.extend(plot_efficiency(summary.train_df, summary.output_dir))
    generated_files.extend(plot_quality_metrics(summary.metrics_df, summary.output_dir))
    if run_type == 'wavelets':
        generated_files.extend(plot_wavelet_specific(summary.train_df, summary.metrics_df, summary.output_dir))
    else:
        generated_files.extend(plot_r3gan_specific(summary.train_df, summary.output_dir))
    write_key_stats(summary)
    write_report(summary, generated_files)


def compare_r3gan_runs(run_summaries: list[RunSummary]) -> None:
    r3_runs = [summary for summary in run_summaries if summary.experiment_name == 'e001-02-r3gan-baseline']
    if len(r3_runs) < 2:
        return
    base_dir = ROOT / 'e001-02-r3gan-baseline' / 'summary'
    base_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for summary in r3_runs:
        df = summary.train_df
        if 'g_loss' not in df.columns:
            continue
        ax.plot(df['step'], rolling(df['g_loss']), linewidth=2.0, label=f'{summary.run_name} - g_loss')
    ax.set_title('R3GAN run comparison: smoothed generator loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('g_loss')
    ax.legend(fontsize=8)
    save_figure(fig, base_dir / 'compare_g_loss.png')

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for summary in r3_runs:
        df = summary.train_df
        if 'sec_per_iter' in df.columns:
            axes[0].plot(df['step'], rolling(df['sec_per_iter']), linewidth=2.0, label=summary.run_name)
        if 'vram_peak_mb' in df.columns:
            axes[1].plot(df['step'], rolling(df['vram_peak_mb']), linewidth=2.0, label=summary.run_name)
    axes[0].set_title('R3GAN efficiency comparison')
    axes[0].set_ylabel('sec_per_iter')
    axes[1].set_ylabel('vram_peak_mb')
    axes[1].set_xlabel('Step')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    save_figure(fig, base_dir / 'compare_efficiency.png')

    metric_runs = [summary for summary in r3_runs if not summary.metrics_df.empty and ('fid' in summary.metrics_df.columns and summary.metrics_df['fid'].notna().any())]
    metric_plot_files = ['compare_g_loss.png', 'compare_efficiency.png']
    if metric_runs:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for summary in metric_runs:
            mdf = summary.metrics_df
            if 'fid' in mdf.columns and mdf['fid'].notna().any():
                axes[0].plot(mdf['step'], mdf['fid'], marker='o', linewidth=2.0, label=summary.run_name)
            kid_col = 'kid_mean' if 'kid_mean' in mdf.columns else 'kid'
            if kid_col in mdf.columns and mdf[kid_col].notna().any():
                axes[1].plot(mdf['step'], mdf[kid_col], marker='o', linewidth=2.0, label=summary.run_name)
        axes[0].set_title('R3GAN quality metrics comparison')
        axes[0].set_ylabel('FID')
        axes[1].set_ylabel('KID')
        axes[1].set_xlabel('Step')
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        save_figure(fig, base_dir / 'compare_quality.png')
        metric_plot_files.append('compare_quality.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        for summary in metric_runs:
            mdf = summary.metrics_df
            for column in ['precision', 'recall', 'lpips_diversity']:
                if column in mdf.columns and mdf[column].notna().any():
                    ax.plot(mdf['step'], mdf[column], marker='o', linewidth=1.8, label=f'{summary.run_name} - {column}')
        ax.set_title('R3GAN precision / recall / diversity comparison')
        ax.set_xlabel('Step')
        ax.set_ylabel('Metric value')
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, base_dir / 'compare_pr_diversity.png')
        metric_plot_files.append('compare_pr_diversity.png')

    rows: list[dict[str, Any]] = []
    for summary in r3_runs:
        rows.append(
            {
                'run_name': summary.run_name,
                'config_name': summary.config.get('name'),
                'steps': summary.config.get('steps'),
                'batch_size': summary.config.get('batch_size'),
                'img_size': summary.config.get('img_size', summary.config.get('img_resolution')),
                'avg_sec_per_iter': summary.average_sec_per_iter,
                'avg_vram_peak_mb': summary.average_vram_peak_mb,
                'best_fid': summary.best_fid,
                'best_fid_step': summary.best_fid_step,
                'best_kid': summary.best_kid,
                'best_kid_step': summary.best_kid_step,
                'final_step_logged': summary.final_step,
            }
        )
    overview = pd.DataFrame(rows).sort_values(['best_fid', 'best_kid', 'avg_sec_per_iter'], na_position='last')
    overview.to_csv(base_dir / 'run_overview.csv', index=False)

    bullets = []
    if metric_runs:
        best_fid_run = min((r for r in metric_runs if r.best_fid is not None), key=lambda r: r.best_fid, default=None)
        if best_fid_run is not None:
            bullets.append(f'- Najlepszy FID wśród runów R3GAN ma `{best_fid_run.run_name}`: **{best_fid_run.best_fid:.4f}** przy kroku **{best_fid_run.best_fid_step}**.')
        best_kid_run = min((r for r in metric_runs if r.best_kid is not None), key=lambda r: r.best_kid, default=None)
        if best_kid_run is not None:
            bullets.append(f'- Najlepszy KID ma `{best_kid_run.run_name}`: **{best_kid_run.best_kid:.6f}** przy kroku **{best_kid_run.best_kid_step}**.')
    fastest_run = min((r for r in r3_runs if r.average_sec_per_iter is not None), key=lambda r: r.average_sec_per_iter, default=None)
    if fastest_run is not None:
        bullets.append(f'- Najszybszy run wg średniego czasu iteracji to `{fastest_run.run_name}`: **{fastest_run.average_sec_per_iter:.4f} s/iter**.')
    lowest_vram_run = min((r for r in r3_runs if r.average_vram_peak_mb is not None), key=lambda r: r.average_vram_peak_mb, default=None)
    if lowest_vram_run is not None:
        bullets.append(f'- Najmniejsze średnie użycie VRAM ma `{lowest_vram_run.run_name}`: **{lowest_vram_run.average_vram_peak_mb:.1f} MB**.')
    bullets.append('- Uwaga interpretacyjna: runy różnią się rozdzielczością, batch size i harmonogramem metryk, więc porównanie jakości jest najbardziej uczciwe dla runów z podobnym setupem, a porównanie wydajności należy czytać razem z konfiguracją.')

    report_lines = [
        '# R3GAN cross-run summary',
        '',
        '## Najważniejsze obserwacje',
        '',
        *(bullets or ['- Brak wystarczających danych do automatycznych wniosków.']),
        '',
        '## Zestawienie runów',
        '',
        '- Plik tabelaryczny: `./run_overview.csv`',
        '',
    ]
    for filename in metric_plot_files:
        title = filename.replace('.png', '').replace('_', ' ').title()
        report_lines.extend(markdown_figure_block(title, base_dir / filename, base_dir))
    (base_dir / 'report.md').write_text('\n'.join(report_lines).strip() + '\n', encoding='utf-8')


def compare_shared_metrics(run_summaries: list[RunSummary]) -> None:
    comparable = [summary for summary in run_summaries if summary.best_fid is not None or summary.best_kid is not None]
    if len(comparable) < 2:
        return
    base_dir = ROOT / 'summary'
    base_dir.mkdir(exist_ok=True)

    rows = []
    for summary in comparable:
        rows.append(
            {
                'experiment_name': summary.experiment_name,
                'run_name': summary.run_name,
                'config_name': summary.config.get('name'),
                'img_size': summary.config.get('img_size', summary.config.get('img_resolution')),
                'steps': summary.config.get('steps'),
                'best_fid': summary.best_fid,
                'best_fid_step': summary.best_fid_step,
                'best_kid': summary.best_kid,
                'best_kid_step': summary.best_kid_step,
                'avg_sec_per_iter': summary.average_sec_per_iter,
                'avg_vram_peak_mb': summary.average_vram_peak_mb,
            }
        )
    overview = pd.DataFrame(rows).sort_values(['best_fid', 'best_kid'], na_position='last')
    overview.to_csv(base_dir / 'metric_overview.csv', index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    ranked_fid = overview.dropna(subset=['best_fid']).sort_values('best_fid')
    ranked_kid = overview.dropna(subset=['best_kid']).sort_values('best_kid')
    if not ranked_fid.empty:
        axes[0].barh(ranked_fid['run_name'], ranked_fid['best_fid'], color='tab:blue')
        axes[0].set_title('Best FID by run')
        axes[0].set_xlabel('FID (lower is better)')
    if not ranked_kid.empty:
        axes[1].barh(ranked_kid['run_name'], ranked_kid['best_kid'], color='tab:green')
        axes[1].set_title('Best KID by run')
        axes[1].set_xlabel('KID (lower is better)')
    save_figure(fig, base_dir / 'best_metric_ranking.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    for summary in comparable:
        if summary.metrics_df.empty or 'fid' not in summary.metrics_df.columns or summary.metrics_df['fid'].notna().sum() == 0:
            continue
        ax.plot(summary.metrics_df['step'], summary.metrics_df['fid'], marker='o', linewidth=2.0, label=summary.run_name)
    ax.set_title('Shared FID trajectories across runs')
    ax.set_xlabel('Step')
    ax.set_ylabel('FID')
    ax.legend(fontsize=8)
    save_figure(fig, base_dir / 'fid_trajectories.png')

    best_fid_run = min((r for r in comparable if r.best_fid is not None), key=lambda r: r.best_fid, default=None)
    best_kid_run = min((r for r in comparable if r.best_kid is not None), key=lambda r: r.best_kid, default=None)
    report_lines = [
        '# Global metric summary',
        '',
        'To zestawienie porównuje tylko wspólne metryki jakości (`FID`, `KID`). Traktuj je ostrożnie, bo runy różnią się architekturą, rozdzielczością i budżetem ewaluacji.',
        '',
        '## Najważniejsze obserwacje',
        '',
    ]
    if best_fid_run is not None:
        report_lines.append(f'- Najlepszy zaobserwowany FID ma `{best_fid_run.run_name}` z eksperymentu `{best_fid_run.experiment_name}`: **{best_fid_run.best_fid:.4f}**.')
    if best_kid_run is not None:
        report_lines.append(f'- Najlepszy zaobserwowany KID ma `{best_kid_run.run_name}` z eksperymentu `{best_kid_run.experiment_name}`: **{best_kid_run.best_kid:.6f}**.')
    report_lines.extend([
        '- Plik tabelaryczny z rankingiem: `./metric_overview.csv`',
        '',
        '## Wykresy',
        '',
    ])
    report_lines.extend(markdown_figure_block('Best Metric Ranking', base_dir / 'best_metric_ranking.png', base_dir))
    report_lines.extend(markdown_figure_block('FID Trajectories', base_dir / 'fid_trajectories.png', base_dir))
    (base_dir / 'report.md').write_text('\n'.join(report_lines), encoding='utf-8')


def main() -> None:
    log_paths = discover_logs(ROOT)
    if not log_paths:
        raise SystemExit('No logs.csv files found.')
    run_summaries = [summarize_run(log_path) for log_path in log_paths]
    for summary in run_summaries:
        generate_run_outputs(summary)
    compare_r3gan_runs(run_summaries)
    compare_shared_metrics(run_summaries)
    print(f'Generated summaries for {len(run_summaries)} runs.')
    for summary in run_summaries:
        print(summary.output_dir)
    print(ROOT / 'e001-02-r3gan-baseline' / 'summary')
    print(ROOT / 'summary')


if __name__ == '__main__':
    main()

