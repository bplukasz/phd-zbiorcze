# AGENTS.md

## Scope and intent
- This repo is an experiment monorepo for GAN research; each `eXXX-*` folder is an independent experiment package with its own `run.py`, `src/`, `requirements.txt`, and often `tests/`.
- Root-level orchestration lives in `Makefile` and `scripts/`; shared reusable code is in `shared/utils`.

## Big picture architecture
- Main active baselines are `e000-01-r3gan-baseline` and `e001-02-r3gan-baseline`; both expose `from src import train, get_config` via `src/__init__.py`.
- Config flow is hierarchical: `src/configs/base.yaml` + `<profile>.yaml` + CLI overrides (`--override key=value ...`) loaded by `src/config_loader.py`.
- Both R3GAN baselines auto-generate `out_dir` under `artifacts/artifacts-MM-DD-XX-<profile>` when `out_dir` is not explicitly overridden.
- Training writes reproducibility artifacts to the run output: `config_used.yaml`, `logs.csv`, `grids/`, `checkpoints/`, `samples/`, and often `real_samples/`.
- `e001-02` uses wavelet/frequency variants (R0/R1/R2/R3/R4 and later) documented in `e001-02-r3gan-baseline/src/configs/README.md` and analyzed by scripts under `scripts/`.

## Critical workflows (use these first)
- Bootstrap environment: `make venv` (runs `scripts/bootstrap_env.sh`; depends on `${HOME}/Projekty/common-env/edgexpert-ml.txt`).
- Standard training shortcuts: `make train-e001-02-fast`, `make train-e001-02-smoke`, `make train-e001-02-overnight`, `make run-e001-02 PROFILE=<profile>`.
- e000 runner shortcut: `make run-e000-01 PROFILE=<profile>`.
- Queueing for long runs (task-spooler): `make queue-add-e001-02 PROFILE=<profile>`, `make queue-enqueue-e001-02`, `make queue-sync-e001-02`, `make queue-status-e001-02`.
- Quality checks: `make lint`, `make format`, `make test` (note: subprojects also have local `pytest.ini` with `testpaths=tests`).

## Repo-specific conventions to preserve
- Keep experiment boundaries: do not casually import between `e000-*`, `e001-*`, `e002-*`; shared code should go through `shared/utils`.
- Preserve profile-driven execution; add new behavior behind new YAML profiles in `src/configs/` rather than hardcoding in `run.py`.
- Keep CLI/data precedence consistent: CLI args > env vars (`DATA_DIR`) > YAML defaults.
- When adding logs/metrics in training loops, append to CSV schema intentionally (see `e000-01-r3gan-baseline/src/artifact_io.py` and `e001-02-r3gan-baseline/src/experiment.py`).
- Keep run reproducibility pattern: copy effective config to run output (`config_used.yaml`) and keep per-run logs under `runs/` or experiment `artifacts/`.

## Integration points and dependencies
- Runtime environment setup is centralized in `scripts/env_paths.sh` (`PYTHONPATH`, cache dirs, default `DATA_DIR`).
- `e001-02` queue integration is JSON-backed (`queues/e001_02_queue.json`) and controlled by `scripts/experiment_queue.py` + `tsp`.
- Publication/reporting pipeline consumes `artifacts/*/logs.csv`: see `scripts/build_publication_tables.py` and `e001-02-r3gan-baseline/summary/publication/README.md`.
- Dataset bootstrap/validation is handled via `scripts/download_datasets.py` (usage documented in `scripts/README-datasets.md`).
- `e002-01-gan-cluster-metrics` is partially scaffolded; `dataset/src/experiment.py` is template-like, while `dataset/src/train.py` is the concrete pipeline used by `run_pipeline.sh`.

## Existing AI/project guidance discovered
- `queues/README.md` documents queue flow and corresponding `make` targets.
- `e000-01-r3gan-baseline/src/configs/README.md` defines allowed baseline profiles (`base`, `smoke`, `fast`, `overnight`).
- `e001-02-r3gan-baseline/src/configs/README.md` defines the experiment matrix/phases and frozen hyperparameters for fair comparisons.
- `e001-02-r3gan-baseline/summary/publication/README.md` defines publication-table filtering rules (e.g., compare at `step <= 30000`).
