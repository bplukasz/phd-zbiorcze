.PHONY: help venv deps gpu train tmux logs lint format test \
        train-e001 train-e001-full \
        train-e001-02 train-e001-02-fast train-e001-02-smoke train-e001-02-overnight train-e001-02-full

SHELL := /bin/bash

# Domyślny profil: fast = CIFAR-10, pobiera się automatycznie, nie wymaga DATA_DIR
# Zmień przez: make train-e001 PROFILE=train  (+ ustaw DATA_DIR do CelebA)
PROFILE ?= fast

help:
	@echo "Targets:"
	@echo "  make venv                   - create venv + install deps"
	@echo "  make gpu                    - check torch/cuda"
	@echo ""
	@echo "  === e001-01-wavelets-baseline ==="
	@echo "  make train-e001             - CIFAR-10 auto-download (PROFILE=fast)"
	@echo "  make train-e001-full        - CelebA full run (wymaga DATA_DIR)"
	@echo ""
	@echo "  === e001-02-r3gan-baseline ==="
	@echo "  make train-e001-02             - CIFAR-10 32×32, 10k kroków  (profile: fast)"
	@echo "  make train-e001-02-fast        - alias dla train-e001-02"
	@echo "  make train-e001-02-smoke       - smoke test, ~2 min"
	@echo "  make train-e001-02-overnight   - CIFAR-10 64×64, 200k kroków (zostaw na noc)"
	@echo "  make train-e001-02-full        - CelebA 64×64, 200k kroków (wymaga DATA_DIR)"
	@echo "  make train-e001-02 PROFILE=overnight DATA_DIR=/path/to/celeba  - custom"
	@echo ""
	@echo "  make tmux              - run training in tmux (background)"
	@echo "  make logs              - tail latest train.log"
	@echo "  make lint              - ruff"
	@echo "  make format            - black"
	@echo "  make test              - pytest"

venv:
	@bash scripts/bootstrap_env.sh

gpu:
	@. .venv/bin/activate && python scripts/gpu_check.py

train-e001:
	@. .venv/bin/activate && . scripts/env_paths.sh && \
	 cd e001-01-wavelets-baseline && python run.py --profile $(PROFILE) $(if $(DATA_DIR),--data-dir $(DATA_DIR),)

# Pełny trening na CelebA — wymaga ustawionego DATA_DIR
# Użycie: make train-e001-full DATA_DIR=/sciezka/do/celeba/img_align_celeba
train-e001-full:
	@. .venv/bin/activate && . scripts/env_paths.sh && \
	 cd e001-01-wavelets-baseline && python run.py --profile train $(if $(DATA_DIR),--data-dir $(DATA_DIR),)

# Legacy alias
train: train-e001

tmux:
	@bash scripts/run_tmux.sh e001 e001-01-wavelets-baseline/src/configs/train.yaml \
	 "cd e001-01-wavelets-baseline && python run.py --profile fast"

logs:
	@ls -dt runs/*/train.log 2>/dev/null | head -n 1 | xargs -r tail -f

lint:
	@. .venv/bin/activate && ruff check .

format:
	@. .venv/bin/activate && black .

test:
	@. .venv/bin/activate && pytest -q

# ─── e001-02-r3gan-baseline ───────────────────────────────────────────────────

# Ogólny runner: make train-e001-02 PROFILE=fast [DATA_DIR=/path]
train-e001-02:
	@. .venv/bin/activate && . scripts/env_paths.sh && \
	 cd e001-02-r3gan-baseline && python run.py --profile $(PROFILE) \
	 $(if $(DATA_DIR),--data-dir $(DATA_DIR),)

train-e001-02-fast:
	@. .venv/bin/activate && . scripts/env_paths.sh && \
	 cd e001-02-r3gan-baseline && python run.py --profile fast

train-e001-02-smoke:
	@. .venv/bin/activate && . scripts/env_paths.sh && \
	 cd e001-02-r3gan-baseline && python run.py --profile smoke

# Zostaw na noc — CIFAR-10 64×64, 200k kroków, nie wymaga DATA_DIR
train-e001-02-overnight:
	@. .venv/bin/activate && . scripts/env_paths.sh && \
	 cd e001-02-r3gan-baseline && python run.py --profile overnight \
	 $(if $(DATA_DIR),--data-dir $(DATA_DIR),)

# Pełny trening CelebA — wymaga DATA_DIR
# Użycie: make train-e001-02-full DATA_DIR=/sciezka/do/celeba/img_align_celeba
train-e001-02-full:
	@. .venv/bin/activate && . scripts/env_paths.sh && \
	 cd e001-02-r3gan-baseline && python run.py --profile overnight \
	 $(if $(DATA_DIR),--data-dir $(DATA_DIR),--data-dir $(error DATA_DIR is required for CelebA))

