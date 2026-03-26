# Experiment Queue (tsp)

Queue file for e001-02 experiments:

- `queues/e001_02_queue.json`

Managed by:

- `python scripts/experiment_queue.py`

Quick flow:

```bash
. .venv/bin/activate
python scripts/experiment_queue.py add --profile phase_b_r0_baseline_64
python scripts/experiment_queue.py add --profile phase_b_r2_waved_64 --override "steps=20000 batch_size=32"
python scripts/experiment_queue.py enqueue
python scripts/experiment_queue.py sync --show
```

```bash
make queue-add-e001-02 PROFILE=phase_b_r0_baseline_64
make queue-enqueue-e001-02
make queue-sync-e001-02
make queue-status-e001-02
```