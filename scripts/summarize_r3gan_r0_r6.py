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
OUT_DIR = EXP_ROOT / "summary" / "r0-r6"

plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class RunData:
    artifact_dir: Path
    run_id: str
    config_name: str
    config: dict[str, Any]
    train_df: pd.DataFrame
    metrics_df: pd.DataFrame


def parse_run_id(config_name: str) -> str:
    # Example: phase_c_r6_fftreg_32 -> r6
    parts = config_name.split("_")
    for part in parts:
        if part.startswith("r") and part[1:].isdigit():
            return part
    return config_name


def load_runs() -> list[RunData]:
    runs: list[RunData] = []
    for artifact_dir in sorted(EXP_ROOT.glob("artifacts-03-*-phase_*_32")):
        log_path = artifact_dir / "logs.csv"
        cfg_path = artifact_dir / "config_used.yaml"
        if not log_path.exists() or not cfg_path.exists():
            continue

        with cfg_path.open("r", encoding="utf-8") as handle:
            config = yaml.full_load(handle) or {}
        config_name = str(config.get("name", artifact_dir.name))
        if "_r" not in config_name:
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
                run_id=parse_run_id(config_name),
                config_name=config_name,
                config=config,
                train_df=train_df,
                metrics_df=metrics_df,
            )
        )

    runs.sort(key=lambda r: int(r.run_id[1:]) if r.run_id.startswith("r") and r.run_id[1:].isdigit() else 999)
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
            "wavelet_enabled": bool(run.config.get("wavelet_enabled", False)),
            "matched_capacity_enabled": bool(run.config.get("matched_capacity_enabled", False)),
            "wave_reg_enabled": bool(run.config.get("wave_reg_enabled", False)),
            "fft_reg_enabled": bool(run.config.get("fft_reg_enabled", False)),
            "wave_reg_weight": float(run.config.get("wave_reg_weight", 0.0)),
            "fft_reg_weight": float(run.config.get("fft_reg_weight", 0.0)),
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
            "avg_sec_per_iter": _final_value(train.assign(sec_per_iter=pd.to_numeric(train.get("sec_per_iter"), errors="coerce")), "sec_per_iter"),
            "avg_vram_peak_mb": _final_value(train.assign(vram_peak_mb=pd.to_numeric(train.get("vram_peak_mb"), errors="coerce")), "vram_peak_mb"),
        }

        # Replace the two fields above with true means if available.
        if "sec_per_iter" in train.columns:
            sec = pd.to_numeric(train["sec_per_iter"], errors="coerce").dropna()
            row["avg_sec_per_iter"] = float(sec.mean()) if not sec.empty else None
        if "vram_peak_mb" in train.columns:
            vr = pd.to_numeric(train["vram_peak_mb"], errors="coerce").dropna()
            row["avg_vram_peak_mb"] = float(vr.mean()) if not vr.empty else None

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("run_id")
    if "r0" in set(summary["run_id"]):
        baseline_fid = float(summary.loc[summary["run_id"] == "r0", "final_fid"].iloc[0])
        summary["delta_fid_vs_r0"] = summary["final_fid"] - baseline_fid
    return summary.reset_index(drop=True)


def _save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fid_kid_trajectories(runs: list[RunData]) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for run in runs:
        m = run.metrics_df
        if "kimg" not in m.columns:
            continue
        if "fid" in m.columns and pd.to_numeric(m["fid"], errors="coerce").notna().any():
            axes[0].plot(m["kimg"], m["fid"], marker="o", linewidth=1.8, label=run.run_id)
        if "kid_mean" in m.columns and pd.to_numeric(m["kid_mean"], errors="coerce").notna().any():
            axes[1].plot(m["kimg"], m["kid_mean"], marker="o", linewidth=1.8, label=run.run_id)

    axes[0].set_title("Trajektorie FID (r0-r6)")
    axes[0].set_ylabel("FID (mniej = lepiej)")
    axes[1].set_title("Trajektorie KID mean (r0-r6)")
    axes[1].set_ylabel("KID mean (mniej = lepiej)")
    axes[1].set_xlabel("kimg")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].legend(ncol=4, fontsize=8)
    name = "fid_kid_trajectories.png"
    _save(fig, name)
    return name


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

    name = "quality_cost_tradeoff.png"
    _save(fig, name)
    return name


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

    name = "final_pr_lpips.png"
    _save(fig, name)
    return name


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

    name = "spectral_vs_fid.png"
    _save(fig, name)
    return name


def plot_regularizer_dynamics(runs: list[RunData]) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for run in runs:
        t = run.train_df
        if "wave_reg_total" in t.columns and pd.to_numeric(t["wave_reg_total"], errors="coerce").notna().any():
            axes[0].plot(t["step"], t["wave_reg_total"], linewidth=1.8, label=run.run_id)
        if "fft_reg_total" in t.columns and pd.to_numeric(t["fft_reg_total"], errors="coerce").notna().any():
            axes[1].plot(t["step"], t["fft_reg_total"], linewidth=1.8, label=run.run_id)

    axes[0].set_title("Wave regularization dynamics")
    axes[0].set_ylabel("wave_reg_total")
    axes[1].set_title("FFT regularization dynamics")
    axes[1].set_ylabel("fft_reg_total")
    axes[1].set_xlabel("step")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    name = "regularizer_dynamics.png"
    _save(fig, name)
    return name


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    subset = df[columns].copy()
    for col in subset.columns:
        if subset[col].dtype.kind in {"f", "i"}:
            subset[col] = subset[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    header = "| " + " | ".join(subset.columns) + " |"
    sep = "|" + "|".join(["---"] * len(subset.columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in subset.to_numpy()]
    return "\n".join([header, sep, *rows])


def make_report(summary: pd.DataFrame, plot_files: list[str]) -> None:
    best_fid = summary.sort_values("final_fid").iloc[0]
    best_auc = summary.sort_values("fid_auc_vs_kimg").iloc[0]
    best_recall = summary.sort_values("final_recall", ascending=False).iloc[0]
    best_wbed = summary.sort_values("final_wbed").iloc[0]

    lines: list[str] = [
        "# R3GAN r0-r6: podsumowanie zbiorcze",
        "",
        "Raport obejmuje runy `r0`-`r6` z 30k krokami i wspolnym settingiem CIFAR-10 32x32.",
        "",
        "## Najwazniejsze wnioski",
        "",
        f"- Najlepszy koncowy FID ma `{best_fid['run_id']}`: **{best_fid['final_fid']:.4f}**.",
        f"- Najlepsza srednia trajektoria (AUC FID vs kimg) ma `{best_auc['run_id']}`: **{best_auc['fid_auc_vs_kimg']:.4f}**.",
        f"- Najwyzszy koncowy recall ma `{best_recall['run_id']}`: **{best_recall['final_recall']:.4f}**.",
        f"- Najlepszy koncowy WBED ma `{best_wbed['run_id']}`: **{best_wbed['final_wbed']:.4f}**.",
        "- `wave_reg` (r3, r5) poprawia metryki pasmowe, ale w tej konfiguracji pogarsza FID/KID.",
        "- `fft_reg` bez galezi waveletowej (r6) daje najlepszy kompromis jakosc/czas sposrod wariantow regularizowanych.",
        "",
        "## Tabela porownawcza (final)",
        "",
        markdown_table(
            summary,
            [
                "run_id",
                "final_fid",
                "final_kid",
                "final_precision",
                "final_recall",
                "final_lpips",
                "final_rpse",
                "final_wbed",
                "fid_auc_vs_kimg",
                "avg_sec_per_iter",
                "avg_vram_peak_mb",
                "delta_fid_vs_r0",
            ],
        ),
        "",
        "## Co nie wyszlo (i dlaczego to wazne)",
        "",
        "- `r3` (wavelet D + wave_reg) i `r5` (wave_reg only) koncza z FID ~24.7-24.9, czyli ~+3.4 do +3.5 pkt vs `r0`.",
        "- W `r3` i `r6` finalne RPSE/WBED sa wyraznie gorsze niz ich minima w trakcie treningu, co wskazuje na rozjazd pod koniec (problem harmonogramu, nie samej idei).",
        "- Sama galaz waveletowa w D (`r2`) nie poprawia FID wzgledem baseline: wynik bliski `r1`, ale dalej slabszy od `r0`.",
        "",
        "## Rekomendacja pod publikacje",
        "",
        "- **Glowny punkt odniesienia:** utrzymac `r0` jako quality anchor (najlepszy FID/AUC).",
        "- **Glowny kandydat nowosci:** `r6` (FFT reg bez wavelet D) jako wariant kompromisowy: FID blisko r0, najwyzszy recall, praktycznie koszt baseline.",
        "- **Material dodatkowy (ablation):** `r4` i `r5` jako dowod, ze da sie mocno poprawic metryki spektralne, ale kosztem FID lub stabilnosci koncowej.",
        "",
        "## Jak realnie podniesc `waved` i `wavereg`, zeby walczyly z baseline",
        "",
        "Punkty ponizej sa wysokiej pewnosci, bo wynikaja bezposrednio z logow r2/r3/r5:",
        "- `wavereg` potrafi mocno poprawiac metryki pasmowe (WBED), ale przy stalej wadze 0.02 konczy z gorszym FID.",
        "- Najwiekszy problem jest na koncowce treningu: najlepsze WBED/RPSE pojawia sie wczesniej niz final, a potem czesto jest dryf.",
        "",
        "1. **Najwazniejsza zmiana (P0): harmonogram `wave_reg_weight` zamiast stalej wartosci.**",
        "   - Proponowany schedule: `0.00` (0-5k) -> liniowo do `0.02` (5k-15k) -> liniowo do `0.005` (15k-30k).",
        "   - Dlaczego: w r3/r5 kara jest najsilniejsza wtedy, gdy model potrzebuje juz fine-tuningu pod FID; to typowy over-regularization tail.",
        "   - Oczekiwany efekt: zachowac zysk WBED z polowy treningu, a jednoczesnie odzyskac 1-2 pkt FID na koncu.",
        "",
        "2. **P0 dla `waved`: delayed activation galezi waveletowej (gate warmup).**",
        "   - Obecnie gate startuje z 0 i model od razu uczy sie wszystkiego naraz; to zwieksza ryzyko konfliktu celow na starcie.",
        "   - Proponowany warmup gate: 0-3k utrzymac blisko 0, potem ramp do ~0.3-0.5 do 12k.",
        "   - Oczekiwany efekt: stabilniejszy poczatek, mniej kary dla FID przy zachowaniu korzysci spektralnych.",
        "",
        "3. **P1: `wavereg` wlaczac warunkowo, nie od kroku 0.**",
        "   - Trigger: wlaczenie dopiero po osiagnieciu `FID < 60` albo po 7.5k krokow (co nastapi pozniej).",
        "   - Dlaczego: wczesny etap powinien budowac semantyke/ksztalt, a nie byc silnie domykany przez kryterium czestotliwosciowe.",
        "",
        "4. **P1: checkpoint selection pod paper nie z finalu 30k, tylko wielokryterialnie.**",
        "   - Regula praktyczna: wybieraj checkpoint o najnizszym FID przy constraintach `WBED <= 1.3 * min_WBED` i `RPSE <= 1.3 * min_RPSE`.",
        "   - Dlaczego: Twoje logi pokazuja, ze najlepsze punkty spektralne i percepcyjne nie zawsze sa w tym samym kroku.",
        "",
        "5. **P2: mala siatka, ktora ma najwieksza szanse przebic r0 bez duzego budzetu.**",
        "   - `r2 + wavereg_schedule` (bez zmian architektury, tylko schedule).",
        "   - `r5 + wavereg_schedule` (najtanszy wariant, juz teraz ma mocny recall i WBED).",
        "   - Dla obu: 3 seedy, ten sam budzet 30k, ta sama ewaluacja co baseline.",
        "",
        "Kryterium sukcesu pod publikacje: medianowy FID (3 seedy) <= FID `r0` + 0.5 przy jednoczesnie lepszym WBED lub recall.",
        "",
        "## Co odpalac dalej (priorytet)",
        "",
        "1. `r6` x 3 seedy (obowiazkowo) + `r0` x 3 seedy dla testu istotnosci roznic FID/recall.",
        "2. `r6` z harmonogramem `fft_reg_weight`: 0.02 do 20k, potem liniowo do 0.005 na 30k (cel: zatrzymac koncowy dryf RPSE/WBED).",
        "3. `r6` weight sweep: 0.01 / 0.02 / 0.03, z tym samym seedem i 30k krokami (ablation pod paper).",
        "4. Krotki run 40k dla `r0` i najlepszego wariantu `r6`, ale z early-stop na minimum WBED/RPSE, nie tylko na koncu 30k.",
        "5. Dla `wave_reg` test annealingu (0.02 -> 0.005 po 15k), bo obecnie sygnal sugeruje over-regularization pod koniec.",
        "",
        "## Wykresy",
        "",
    ]

    for filename in plot_files:
        lines.extend(
            [
                f"### {filename}",
                "",
                f"![{filename}](./{filename})",
                "",
            ]
        )

    (OUT_DIR / "report_r0_r6.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("Nie znaleziono runow r0-r6.")

    summary = build_summary_table(runs)
    summary.to_csv(OUT_DIR / "run_summary_r0_r6.csv", index=False)

    plot_files = [
        plot_fid_kid_trajectories(runs),
        plot_quality_tradeoff(summary),
        plot_final_pr_div(summary),
        plot_spectral(summary),
        plot_regularizer_dynamics(runs),
    ]
    make_report(summary, plot_files)

    print(f"Saved: {(OUT_DIR / 'run_summary_r0_r6.csv').relative_to(ROOT)}")
    print(f"Saved: {(OUT_DIR / 'report_r0_r6.md').relative_to(ROOT)}")
    for item in plot_files:
        print(f"Saved: {(OUT_DIR / item).relative_to(ROOT)}")


if __name__ == "__main__":
    main()


