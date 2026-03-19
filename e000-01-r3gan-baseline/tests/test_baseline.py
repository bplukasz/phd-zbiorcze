from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path

import pytest
import torch

import src.config_loader as config_loader
from src.config_loader import ConfigLoader, RunConfig, get_config
from src.artifact_io import LOG_FIELDNAMES, make_csv_logger
from src.checkpointing import load_training_checkpoint, save_training_checkpoint, validate_resume_compatibility
from src.data import get_dataloader
from src.eta_logging import count_remaining_metric_evals, estimate_remaining_seconds
from src.metrics_runtime import update_fid_auc
from src.r3gan_source import (
    R3GANDiscriminator,
    R3GANGenerator,
    R3GANTrainer,
    TrainerConfig,
    build_stage_channels,
)
import src.data as data_module


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = EXPERIMENT_ROOT / "run.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("e000_01_runner", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise ImportError(f"Could not load run.py from {RUNNER_PATH}")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)
auto_out_dir = getattr(config_loader, "_auto_out_dir")


def _make_models(img_resolution: int = 8, z_dim: int = 16, base_channels: int = 16):
    g_channels = build_stage_channels(img_resolution, base_channels=base_channels, channel_max=128)
    d_channels = list(reversed(g_channels))
    G = R3GANGenerator(
        z_dim=z_dim,
        img_resolution=img_resolution,
        stage_channels=g_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=8,
        out_channels=3,
    )
    D = R3GANDiscriminator(
        img_resolution=img_resolution,
        stage_channels=d_channels,
        blocks_per_stage=1,
        expansion_factor=2,
        group_size=8,
        in_channels=3,
    )
    return G, D


def test_build_stage_channels_returns_correct_length() -> None:
    assert len(build_stage_channels(8)) == 2
    assert len(build_stage_channels(16)) == 3
    assert len(build_stage_channels(32)) == 4
    assert len(build_stage_channels(64)) == 5


def test_generator_forward_shape() -> None:
    G, _ = _make_models(img_resolution=8)
    z = torch.randn(4, G.z_dim)
    out = G(z)
    assert out.shape == (4, 3, 8, 8)


def test_discriminator_forward_shape() -> None:
    _, D = _make_models(img_resolution=8)
    x = torch.randn(4, 3, 8, 8)
    out = D(x)
    assert out.shape == (4,)


def test_trainer_train_step_returns_expected_keys() -> None:
    G, D = _make_models(img_resolution=8)
    trainer = R3GANTrainer(
        G,
        D,
        device=torch.device("cpu"),
        train_cfg=TrainerConfig(use_amp_for_g=False, use_amp_for_d=False, channels_last=False),
    )
    images = torch.randn(4, 3, 8, 8)
    metrics = trainer.train_step(images)
    expected = {
        "d_loss", "g_loss", "d_adv", "g_adv", "r1", "r2", "real_score_mean", "fake_score_mean",
    }
    assert expected.issubset(metrics.keys())


def test_trainer_ema_diverges_from_live_after_training() -> None:
    G, D = _make_models(img_resolution=8)
    trainer = R3GANTrainer(
        G,
        D,
        device=torch.device("cpu"),
        train_cfg=TrainerConfig(use_amp_for_g=False, use_amp_for_d=False, channels_last=False),
    )
    before = [p.detach().clone() for p in trainer.G_ema.parameters()]
    for _ in range(3):
        images = torch.randn(4, 3, 8, 8)
        trainer.train_step(images)
    assert any(not torch.allclose(p_live, p_ema) for p_live, p_ema in zip(trainer.G.parameters(), trainer.G_ema.parameters()))
    assert any(not torch.allclose(p_before, p_after) for p_before, p_after in zip(before, trainer.G_ema.parameters()))


def test_config_loader_roundtrip(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="roundtrip",
        steps=123,
        batch_size=7,
        out_dir=str(tmp_path / "artifacts-fixed"),
        data_dir="/tmp/data",
    )
    cfg.metrics_every = 0
    loader = ConfigLoader(config_dir=str(tmp_path))
    loader.save_config(cfg, str(tmp_path / "base.yaml"))
    loaded = loader.get_config("base")
    assert loaded.to_dict() == cfg.to_dict()


def test_get_config_smoke_profile_overrides_steps() -> None:
    base = get_config("base", overrides={"out_dir": "/tmp/e000-base"})
    smoke = get_config("smoke", overrides={"out_dir": "/tmp/e000-smoke"})
    assert smoke.steps < base.steps
    assert smoke.name == "smoke"


def test_auto_out_dir_is_nested_under_artifacts(tmp_path: Path) -> None:
    out_dir = Path(auto_out_dir("phase_b", base_dir=tmp_path))
    assert out_dir.parent == tmp_path / "artifacts"
    assert out_dir.parent.exists()
    assert re.match(r"^artifacts-\d{2}-\d{2}-\d{2}-phase_b$", out_dir.name)


def test_auto_out_dir_increments_index_inside_artifacts_dir(tmp_path: Path) -> None:
    first = Path(auto_out_dir("phase_b", base_dir=tmp_path))
    first.mkdir(parents=True, exist_ok=True)
    second = Path(auto_out_dir("phase_b", base_dir=tmp_path))

    assert first.parent == second.parent == tmp_path / "artifacts"
    assert first.name.endswith("-phase_b")
    assert second.name.endswith("-phase_b")
    assert first.name != second.name


def test_config_loader_unknown_keys_report_source_base(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("steps: 10\nunknown_base: 1\n", encoding="utf-8")
    loader = ConfigLoader(config_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Unknown config keys in base: unknown_base"):
        loader.get_config("base")


def test_config_loader_unknown_keys_report_source_profile(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("steps: 10\n", encoding="utf-8")
    (tmp_path / "fast.yaml").write_text("unknown_profile: 1\n", encoding="utf-8")
    loader = ConfigLoader(config_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Unknown config keys in profile: unknown_profile"):
        loader.get_config("fast")


def test_config_loader_unknown_keys_report_source_override() -> None:
    with pytest.raises(ValueError, match="Unknown config keys in override: img_size"):
        get_config("base", overrides={"img_size": 32, "out_dir": "/tmp/e000-override"})


def test_config_loader_validates_merged_metrics_constraints() -> None:
    with pytest.raises(ValueError, match="metrics_pr_num_samples must be greater than metrics_pr_k"):
        get_config(
            "base",
            overrides={
                "out_dir": "/tmp/e000-invalid-metrics",
                "metrics_pr_k": 8,
                "metrics_pr_num_samples": 8,
            },
        )


def test_parse_overrides_typed_values() -> None:
    type_hints = runner.get_type_hints(RunConfig)
    valid_keys = {f.name for f in runner.fields(RunConfig)}
    parsed = runner._parse_overrides(
        ["steps=500", "deterministic=true", "betas=[0.1, 0.9]", "grad_clip=null"],
        valid_keys=valid_keys,
        type_hints=type_hints,
    )
    assert parsed["steps"] == 500
    assert parsed["deterministic"] is True
    assert parsed["betas"] == (0.1, 0.9)
    assert parsed["grad_clip"] is None


def test_parse_overrides_rejects_unknown_key() -> None:
    type_hints = runner.get_type_hints(RunConfig)
    valid_keys = {f.name for f in runner.fields(RunConfig)}
    with pytest.raises(ValueError, match="Unknown override key: img_size"):
        runner._parse_overrides(["img_size=32"], valid_keys=valid_keys, type_hints=type_hints)


def test_parse_overrides_rejects_wrong_type() -> None:
    type_hints = runner.get_type_hints(RunConfig)
    valid_keys = {f.name for f in runner.fields(RunConfig)}
    with pytest.raises(ValueError, match="Invalid override value for 'steps'"):
        runner._parse_overrides(["steps=abc"], valid_keys=valid_keys, type_hints=type_hints)


def test_checkpoint_save_and_load_restores_model_and_step(tmp_path: Path) -> None:
    G, D = _make_models(img_resolution=8)
    trainer_a = R3GANTrainer(
        G,
        D,
        device=torch.device("cpu"),
        train_cfg=TrainerConfig(use_amp_for_g=False, use_amp_for_d=False, channels_last=False),
    )
    trainer_a.train_step(torch.randn(4, 3, 8, 8))

    cfg = RunConfig(metrics_every=0, out_dir=str(tmp_path))
    ckpt_path = tmp_path / "ckpt.pt"
    save_training_checkpoint(
        str(ckpt_path),
        step=7,
        trainer=trainer_a,
        cfg=cfg,
        runtime_state={"fid_auc_vs_kimg": 1.5},
    )

    G2, D2 = _make_models(img_resolution=8)
    trainer_b = R3GANTrainer(
        G2,
        D2,
        device=torch.device("cpu"),
        train_cfg=TrainerConfig(use_amp_for_g=False, use_amp_for_d=False, channels_last=False),
    )
    ckpt = load_training_checkpoint(str(ckpt_path), trainer=trainer_b, device=torch.device("cpu"))

    assert ckpt["step"] == 7
    assert ckpt["runtime_state"]["fid_auc_vs_kimg"] == 1.5
    key = next(iter(trainer_a.G.state_dict().keys()))
    assert torch.allclose(trainer_a.G.state_dict()[key], trainer_b.G.state_dict()[key])


def test_validate_resume_compatibility_rejects_arch_mismatch() -> None:
    cfg = RunConfig(img_resolution=64)
    with pytest.raises(ValueError, match="Resume checkpoint is incompatible"):
        validate_resume_compatibility(cfg, {"img_resolution": 32})


def test_csv_logger_append_mode_keeps_existing_rows(tmp_path: Path) -> None:
    logger_a = make_csv_logger(str(tmp_path), append=False)
    logger_a.log({"step": 1, "row_type": "train"})

    logger_b = make_csv_logger(str(tmp_path), append=True)
    logger_b.log({"step": 2, "row_type": "train"})

    with open(tmp_path / "logs.csv", "r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(LOG_FIELDNAMES) > 0
    assert [row["step"] for row in rows] == ["1", "2"]


@pytest.mark.parametrize(
    ("step", "total_steps", "metrics_every", "expected"),
    [
        (10, 100, 10, 10),
        (11, 100, 10, 9),
        (95, 100, 10, 1),
        (100, 100, 10, 0),
        (50, 100, 0, 0),
    ],
)
def test_count_remaining_metric_evals_cases(step: int, total_steps: int, metrics_every: int, expected: int) -> None:
    assert count_remaining_metric_evals(step, total_steps, metrics_every) == expected


def test_estimate_remaining_seconds_includes_metrics_cost() -> None:
    eta = estimate_remaining_seconds(
        step=50,
        total_steps=100,
        sec_per_iter_ema=2.0,
        metrics_every=20,
        metrics_elapsed_ema=10.0,
    )
    # 50 iter * 2s + evaluations at steps 60/80/100 => 3 * 10s
    assert eta == pytest.approx(130.0)


def test_estimate_remaining_seconds_returns_none_without_iter_ema() -> None:
    assert estimate_remaining_seconds(10, 100, None, 10, 5.0) is None


def test_update_fid_auc_trapezoid_and_non_increasing_kimg() -> None:
    p1, auc1 = update_fid_auc(None, current_kimg=0.64, current_fid=40.0, cumulative_auc=0.0)
    assert p1 == (0.64, 40.0)
    assert auc1 == 0.0

    p2, auc2 = update_fid_auc(p1, current_kimg=1.28, current_fid=20.0, cumulative_auc=auc1)
    # Trapezoid area: ((40 + 20) / 2) * 0.64 = 19.2
    assert p2 == (1.28, 20.0)
    assert auc2 == pytest.approx(19.2)

    p3, auc3 = update_fid_auc(p2, current_kimg=1.0, current_fid=10.0, cumulative_auc=auc2)
    assert p3 == (1.0, 10.0)
    assert auc3 == pytest.approx(19.2)


def test_get_dataloader_supported_dataset_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyDataset:
        def __len__(self):
            return 12

        def __getitem__(self, idx):
            return idx

    captured = []

    def _factory(name: str):
        def _ctor(*args, **kwargs):
            captured.append((name, kwargs.get("root")))
            return DummyDataset()

        return _ctor

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    monkeypatch.setattr(data_module.datasets, "ImageFolder", _factory("celeba"))
    monkeypatch.setattr(data_module.datasets, "CIFAR10", _factory("cifar10"))
    monkeypatch.setattr(data_module.datasets, "CIFAR100", _factory("cifar100"))
    monkeypatch.setattr(data_module.datasets, "MNIST", _factory("mnist"))
    monkeypatch.setattr(data_module.datasets, "FashionMNIST", _factory("fashion_mnist"))
    monkeypatch.setattr(data_module, "DataLoader", DummyLoader)

    celeba_root = tmp_path / "celeba" / "img_align_celeba"
    celeba_root.mkdir(parents=True)
    get_dataloader(str(celeba_root), 32, 4, num_workers=0, dataset_name="celeba")
    get_dataloader(str(tmp_path / "missing-c10"), 32, 4, num_workers=0, dataset_name="cifar10")
    get_dataloader(str(tmp_path / "missing-c100"), 32, 4, num_workers=0, dataset_name="cifar100")
    get_dataloader(str(tmp_path / "missing-mnist"), 32, 4, num_workers=0, dataset_name="mnist")
    get_dataloader(str(tmp_path / "missing-fmnist"), 32, 4, num_workers=0, dataset_name="fashion_mnist")

    assert [name for name, _ in captured] == ["celeba", "cifar10", "cifar100", "mnist", "fashion_mnist"]


def test_get_dataloader_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataloader(str(tmp_path), 32, 4, num_workers=0, dataset_name="not-a-dataset")


