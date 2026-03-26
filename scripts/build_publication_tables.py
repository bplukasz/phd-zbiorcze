#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = ROOT / "e001-02-r3gan-baseline" / "artifacts"
DEFAULT_OUT_DIR = ROOT / "e001-02-r3gan-baseline" / "summary" / "publication"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build publication tables from artifact logs. "
            "Filters runs with max(step) >= min_step and reports fair metrics at eval_step."
        )
    )
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-step", type=int, default=30000)
    parser.add_argument("--eval-step", type=int, default=30000)
    return parser.parse_args()


def parse_recipe(artifact_name: str) -> str:
    m = re.match(r"artifacts-\d{2}-\d{2}-\d{2}-(.+)$", artifact_name)
    if m:
        return m.group(1)
    m = re.match(r"artifacts-\d{2}-\d{2}-(.+)$", artifact_name)
    if m:
        return m.group(1)
    return artifact_name


def read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            data = yaml.full_load(handle) or {}
        except yaml.YAMLError:
            handle.seek(0)
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def extract_metrics_row(metrics_df: pd.DataFrame, eval_step: int) -> pd.Series:
    sub = metrics_df.loc[numeric(metrics_df, "step") <= eval_step].copy()
    if sub.empty:
        return metrics_df.iloc[-1]
    exact = sub.loc[numeric(sub, "step") == eval_step]
    if not exact.empty:
        return exact.iloc[-1]
    return sub.iloc[-1]


def summarize_artifact(artifact_dir: Path, min_step: int, eval_step: int) -> dict[str, Any] | None:
    logs_path = artifact_dir / "logs.csv"
    if not logs_path.exists():
        return None

    df = cast(pd.DataFrame, pd.read_csv(logs_path))
    if "step" not in df.columns:
        return None

    step = numeric(df, "step")
    max_step = float(step.max()) if step.notna().any() else np.nan
    if np.isnan(max_step) or max_step < min_step:
        return None

    row_type = df["row_type"].astype(str) if "row_type" in df.columns else pd.Series(["" for _ in range(len(df))])
    metrics_mask = row_type.str.contains("gan_metrics", na=False)
    if "fid" in df.columns:
        metrics_mask = metrics_mask | numeric(df, "fid").notna()
    metrics_df = df.loc[metrics_mask].copy()
    if metrics_df.empty:
        return None

    metrics_df = metrics_df.assign(step_num=numeric(metrics_df, "step"))
    metrics_df = metrics_df.sort_values("step_num")

    eval_row = extract_metrics_row(metrics_df, eval_step)
    final_row = metrics_df.iloc[-1]

    fid_series = numeric(metrics_df, "fid")
    fid_valid = fid_series.dropna()
    if fid_valid.empty:
        best_fid = np.nan
        best_fid_step = np.nan
    else:
        best_idx = fid_valid.idxmin()
        best_fid = float(fid_series.loc[best_idx])
        best_fid_step = float(numeric(metrics_df, "step").loc[best_idx])

    upto_eval = metrics_df.loc[numeric(metrics_df, "step") <= eval_step].copy()
    fid_upto_eval = numeric(upto_eval, "fid") if not upto_eval.empty else pd.Series(dtype=float)
    if fid_upto_eval.dropna().empty:
        best_fid_upto_eval = np.nan
        best_fid_upto_eval_step = np.nan
    else:
        best_idx_eval = fid_upto_eval.idxmin()
        best_fid_upto_eval = float(fid_upto_eval.loc[best_idx_eval])
        best_fid_upto_eval_step = float(numeric(upto_eval, "step").loc[best_idx_eval])

    train_df = df.loc[row_type.str.contains("train", na=False)].copy()
    sec_tail = np.nan
    vram_peak = np.nan
    if not train_df.empty:
        sec = numeric(train_df, "sec_per_iter").dropna()
        if not sec.empty:
            sec_tail = float(sec.tail(20).mean())
        vram = numeric(train_df, "vram_peak_mb").dropna()
        if not vram.empty:
            vram_peak = float(vram.max())

    config = read_config(artifact_dir / "config_used.yaml")

    def val(row: pd.Series, col: str) -> float:
        return float(pd.to_numeric(row.get(col), errors="coerce"))

    return {
        "artifact_dir": artifact_dir.name,
        "recipe": parse_recipe(artifact_dir.name),
        "seed": config.get("seed"),
        "max_step": int(max_step),
        "eval_step": int(eval_step),
        "fid_at_eval": val(eval_row, "fid"),
        "kid_at_eval": val(eval_row, "kid_mean"),
        "precision_at_eval": val(eval_row, "precision"),
        "recall_at_eval": val(eval_row, "recall"),
        "lpips_at_eval": val(eval_row, "lpips_diversity"),
        "rpse_at_eval": val(eval_row, "rpse"),
        "wbed_at_eval": val(eval_row, "wbed"),
        "fid_auc_at_eval": val(eval_row, "fid_auc_vs_kimg"),
        "fid_final": val(final_row, "fid"),
        "kid_final": val(final_row, "kid_mean"),
        "best_fid": best_fid,
        "best_fid_step": best_fid_step,
        "best_fid_upto_eval": best_fid_upto_eval,
        "best_fid_upto_eval_step": best_fid_upto_eval_step,
        "sec_per_iter_tail20": sec_tail,
        "vram_peak_mb": vram_peak,
    }


def build_tables(artifacts_root: Path, min_step: int, eval_step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for artifact_dir in sorted(artifacts_root.glob("artifacts-*")):
        if not artifact_dir.is_dir():
            continue
        row = summarize_artifact(artifact_dir, min_step=min_step, eval_step=eval_step)
        if row is not None:
            rows.append(row)

    run_df = pd.DataFrame(rows)
    if run_df.empty:
        return run_df, pd.DataFrame()

    run_df = run_df.sort_values(["fid_at_eval", "kid_at_eval", "artifact_dir"], na_position="last").reset_index(drop=True)

    ranking_df = run_df.copy()
    ranking_df["rank_fid"] = ranking_df["fid_at_eval"].rank(method="min", ascending=True)
    ranking_df["rank_kid"] = ranking_df["kid_at_eval"].rank(method="min", ascending=True)
    ranking_df["rank_precision"] = ranking_df["precision_at_eval"].rank(method="min", ascending=False)
    ranking_df["rank_recall"] = ranking_df["recall_at_eval"].rank(method="min", ascending=False)
    ranking_df["rank_rpse"] = ranking_df["rpse_at_eval"].rank(method="min", ascending=True)
    ranking_df["rank_wbed"] = ranking_df["wbed_at_eval"].rank(method="min", ascending=True)

    rank_cols = [
        "rank_fid",
        "rank_kid",
        "rank_precision",
        "rank_recall",
        "rank_rpse",
        "rank_wbed",
    ]
    ranking_df["rank_mean"] = ranking_df[rank_cols].mean(axis=1)
    ranking_df = ranking_df.sort_values(["rank_mean", "rank_fid", "artifact_dir"]).reset_index(drop=True)

    return run_df, ranking_df


def build_claim_readiness(run_df: pd.DataFrame, eval_step: int) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    by_recipe = (
        run_df.groupby("recipe", as_index=False)
        .agg(
            n_runs=("artifact_dir", "count"),
            n_unique_seeds=("seed", lambda s: len(set(x for x in s.dropna()))),
            median_fid=("fid_at_eval", "median"),
            best_fid=("fid_at_eval", "min"),
            median_kid=("kid_at_eval", "median"),
            median_precision=("precision_at_eval", "median"),
            median_recall=("recall_at_eval", "median"),
        )
        .sort_values("median_fid", na_position="last")
        .reset_index(drop=True)
    )

    baseline_key = "phase_b_r0_baseline_32"
    baseline = by_recipe.loc[by_recipe["recipe"] == baseline_key]
    baseline_median = float(baseline["median_fid"].iloc[0]) if not baseline.empty else np.nan
    baseline_runs = int(baseline["n_runs"].iloc[0]) if not baseline.empty else 0

    statuses: list[str] = []
    notes: list[str] = []

    for _, row in by_recipe.iterrows():
        recipe = str(row["recipe"])
        n_runs = int(row["n_runs"])
        median_fid = float(row["median_fid"])
        best_fid = float(row["best_fid"])

        if recipe == baseline_key:
            statuses.append("baseline_anchor")
            notes.append("Referencja porownawcza.")
            continue

        if n_runs >= 3 and baseline_runs >= 3 and np.isfinite(baseline_median):
            if median_fid <= baseline_median:
                statuses.append("ready_main_claim")
                notes.append(f"Mediana FID@{eval_step} nie gorsza niz baseline i >=3 runy.")
            elif median_fid <= baseline_median + 0.5:
                statuses.append("near_claim_need_tuning")
                notes.append(f"Blisko baseline (<= +0.5 FID mediana), potrzebne dostrojenie.")
            else:
                statuses.append("ablation_only")
                notes.append("Reprodukowalnie gorszy od baseline.")
        else:
            if np.isfinite(baseline_median) and best_fid <= baseline_median + 1.0:
                statuses.append("promising_single_seed")
                notes.append("Blisko baseline, ale brak mocy statystycznej (za malo seedow).")
            else:
                statuses.append("ablation_only")
                notes.append("Brak przewagi i/lub za malo seedow.")

    by_recipe["claim_status"] = statuses
    by_recipe["claim_note"] = notes
    return by_recipe


def frame_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "(empty)"

    def fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{float_digits}f}"
        return str(v)

    headers = list(df.columns)
    rows = [[fmt(v) for v in row] for row in df.to_numpy().tolist()]
    sep = ["---"] * len(headers)

    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(sep) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def to_markdown(run_df: pd.DataFrame, ranking_df: pd.DataFrame, claim_df: pd.DataFrame, min_step: int, eval_step: int) -> str:
    lines: list[str] = []
    lines.append("# Publication summary (step >= 30000)")
    lines.append("")
    lines.append(f"- Scope: `artifacts-*/logs.csv` z filtrem `max(step) >= {min_step}`.")
    lines.append(f"- Fair comparison point: metryki przy `step <= {eval_step}` (preferowane dokladnie `{eval_step}`).")
    lines.append(f"- Qualified runs: **{len(run_df)}**.")
    lines.append("")

    if run_df.empty:
        lines.append("Brak runow spelniajacych warunki.")
        return "\n".join(lines) + "\n"

    top_cols = [
        "artifact_dir",
        "recipe",
        "max_step",
        "fid_at_eval",
        "kid_at_eval",
        "precision_at_eval",
        "recall_at_eval",
        "rpse_at_eval",
        "wbed_at_eval",
        "sec_per_iter_tail20",
    ]
    lines.append("## Top runs by FID@eval")
    lines.append("")
    lines.append(frame_to_markdown(run_df[top_cols].head(10), float_digits=4))
    lines.append("")

    rank_cols = [
        "artifact_dir",
        "recipe",
        "rank_mean",
        "rank_fid",
        "rank_kid",
        "rank_precision",
        "rank_recall",
        "rank_rpse",
        "rank_wbed",
    ]
    lines.append("## Composite ranking (quality balance)")
    lines.append("")
    lines.append(frame_to_markdown(ranking_df[rank_cols].head(10), float_digits=3))
    lines.append("")

    lines.append("## Claim readiness by recipe")
    lines.append("")
    if claim_df.empty:
        lines.append("Brak danych do oceny claim readiness.")
    else:
        lines.append(
            frame_to_markdown(
                claim_df[
                    [
                        "recipe",
                        "n_runs",
                        "n_unique_seeds",
                        "median_fid",
                        "best_fid",
                        "median_kid",
                        "claim_status",
                        "claim_note",
                    ]
                ],
                float_digits=4,
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    best_row = run_df.iloc[0]
    lines.append(
        f"- Best run at eval step: `{best_row['artifact_dir']}` with FID `{best_row['fid_at_eval']:.4f}` and KID `{best_row['kid_at_eval']:.6f}`."
    )

    baseline_rows = run_df.loc[run_df["recipe"] == "phase_b_r0_baseline_32"]
    if not baseline_rows.empty:
        baseline_best = float(baseline_rows["fid_at_eval"].min())
        non_baseline = run_df.loc[run_df["recipe"] != "phase_b_r0_baseline_32"]
        if not non_baseline.empty:
            best_non_base = non_baseline.iloc[0]
            delta = float(best_non_base["fid_at_eval"] - baseline_best)
            lines.append(
                f"- Best non-baseline run: `{best_non_base['artifact_dir']}`; gap vs best baseline: `{delta:+.4f}` FID."
            )

    unstable = run_df.loc[(run_df["max_step"] > eval_step) & (run_df["fid_final"] > run_df["fid_at_eval"] + 10.0)]
    if not unstable.empty:
        bad = unstable.iloc[0]
        lines.append(
            f"- Long-run instability detected in `{bad['artifact_dir']}` (FID@{eval_step} `{bad['fid_at_eval']:.4f}` -> final `{bad['fid_final']:.4f}`)."
        )

    lines.append("- Publication stance: ablation-ready now; main-claim readiness requires >=3 seeds for top candidate recipes.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_df, ranking_df = build_tables(args.artifacts_root, min_step=args.min_step, eval_step=args.eval_step)
    claim_df = build_claim_readiness(run_df, eval_step=args.eval_step)

    runs_csv = args.out_dir / f"run_metrics_step_ge_{args.min_step}.csv"
    ranking_csv = args.out_dir / f"run_rankings_step_ge_{args.min_step}.csv"
    claim_csv = args.out_dir / f"claim_readiness_step_ge_{args.min_step}.csv"
    report_md = args.out_dir / f"report_step_ge_{args.min_step}.md"

    run_df.to_csv(runs_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    claim_df.to_csv(claim_csv, index=False)
    report_md.write_text(
        to_markdown(run_df, ranking_df, claim_df, min_step=args.min_step, eval_step=args.eval_step),
        encoding="utf-8",
    )

    print(f"Saved: {runs_csv}")
    print(f"Saved: {ranking_csv}")
    print(f"Saved: {claim_csv}")
    print(f"Saved: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




