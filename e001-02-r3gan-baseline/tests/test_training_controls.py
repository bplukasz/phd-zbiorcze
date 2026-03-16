from __future__ import annotations

from src.config_loader import RunConfig
import src.experiment as experiment_module


compute_aux_branch_gate = getattr(experiment_module, "_compute_aux_branch_gate")
compute_piecewise_weight = getattr(experiment_module, "_compute_piecewise_weight")
resolve_fid_gated_activation = getattr(experiment_module, "_resolve_fid_gated_activation")
validate_metrics_config = getattr(experiment_module, "_validate_metrics_config")
format_eta = getattr(experiment_module, "_format_eta")
count_remaining_metric_evals = getattr(experiment_module, "_count_remaining_metric_evals")
estimate_remaining_seconds = getattr(experiment_module, "_estimate_remaining_seconds")


def test_aux_branch_gate_warmup_is_linear_between_steps() -> None:
    cfg = RunConfig()
    cfg.aux_branch_gate_warmup_enabled = True
    cfg.aux_branch_gate_warmup_start_step = 10
    cfg.aux_branch_gate_warmup_end_step = 20
    cfg.aux_branch_gate_warmup_start_value = 0.0
    cfg.aux_branch_gate_warmup_end_value = 0.5

    assert compute_aux_branch_gate(cfg, 1) == 0.0
    assert compute_aux_branch_gate(cfg, 10) == 0.0
    assert compute_aux_branch_gate(cfg, 15) == 0.25
    assert compute_aux_branch_gate(cfg, 20) == 0.5
    assert compute_aux_branch_gate(cfg, 100) == 0.5


def test_aux_branch_gate_warmup_disabled_returns_none() -> None:
    cfg = RunConfig()
    cfg.aux_branch_gate_warmup_enabled = False
    assert compute_aux_branch_gate(cfg, 123) is None


def test_piecewise_weight_schedule_has_rise_and_decay() -> None:
    w_start = compute_piecewise_weight(0.02, True, 5, 15, 30, 0.0, 0.02, 0.005, 1)
    w_mid = compute_piecewise_weight(0.02, True, 5, 15, 30, 0.0, 0.02, 0.005, 10)
    w_peak = compute_piecewise_weight(0.02, True, 5, 15, 30, 0.0, 0.02, 0.005, 15)
    w_tail = compute_piecewise_weight(0.02, True, 5, 15, 30, 0.0, 0.02, 0.005, 25)
    w_end = compute_piecewise_weight(0.02, True, 5, 15, 30, 0.0, 0.02, 0.005, 40)

    assert w_start == 0.0
    assert 0.0 < w_mid < 0.02
    assert w_peak == 0.02
    assert 0.005 < w_tail < 0.02
    assert w_end == 0.005


def test_piecewise_weight_schedule_disabled_uses_base_weight() -> None:
    w = compute_piecewise_weight(0.02, False, 5, 15, 30, 0.0, 0.02, 0.005, 17)
    assert w == 0.02


def test_fid_gate_turns_on_only_when_step_and_fid_conditions_are_met() -> None:
    active, latched = resolve_fid_gated_activation(True, 60.0, 7500, True, 7000, 55.0, False)
    assert active is False and latched is False

    active, latched = resolve_fid_gated_activation(True, 60.0, 7500, True, 8000, 65.0, False)
    assert active is False and latched is False

    active, latched = resolve_fid_gated_activation(True, 60.0, 7500, True, 8000, 55.0, False)
    assert active is True and latched is True


def test_fid_gate_latched_mode_stays_active_after_trigger() -> None:
    active, latched = resolve_fid_gated_activation(True, 60.0, 0, True, 1000, 55.0, False)
    assert active is True and latched is True

    active, latched = resolve_fid_gated_activation(True, 60.0, 0, True, 2000, 90.0, latched)
    assert active is True and latched is True


def test_fid_gate_non_latched_can_turn_off() -> None:
    active, latched = resolve_fid_gated_activation(True, 60.0, 0, False, 1000, 55.0, False)
    assert active is True and latched is True

    active, latched = resolve_fid_gated_activation(True, 60.0, 0, False, 2000, 90.0, latched)
    assert active is False and latched is False


def test_fid_gate_requires_metrics_enabled() -> None:
    cfg = RunConfig()
    cfg.metrics_every = 0
    cfg.wave_reg_fid_gate_enabled = True

    try:
        validate_metrics_config(cfg, dataset_size=None)
    except ValueError as exc:
        assert "wave_reg_fid_gate_enabled requires metrics_every > 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError when FID-gated regularizer is enabled without metrics")


def test_format_eta_adapts_units() -> None:
    assert format_eta(45) == "45s"
    assert format_eta(120) == "2m"
    assert format_eta(3660) == "1h 01m"


def test_count_remaining_metric_evals_includes_current_due_eval() -> None:
    assert count_remaining_metric_evals(step=4000, total_steps=10000, metrics_every=2000) == 4
    assert count_remaining_metric_evals(step=4100, total_steps=10000, metrics_every=2000) == 3


def test_estimate_remaining_seconds_adds_metrics_overhead() -> None:
    eta = estimate_remaining_seconds(
        step=4000,
        total_steps=10000,
        sec_per_iter_ema=1.0,
        metrics_every=2000,
        metrics_elapsed_ema=10.0,
    )
    # 6000 training iterations + 4 metric evaluations * 10s
    assert eta == 6040.0


