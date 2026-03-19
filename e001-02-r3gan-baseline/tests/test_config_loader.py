from __future__ import annotations

import re
from pathlib import Path

import src.config_loader as config_loader
from src.config_loader import get_config
auto_out_dir = getattr(config_loader, "_auto_out_dir")


def test_auto_out_dir_is_nested_under_artifacts(tmp_path: Path) -> None:
    out_dir = Path(auto_out_dir("phase_b_r0_baseline_32", base_dir=tmp_path))
    assert out_dir.parent == tmp_path / "artifacts"
    assert out_dir.parent.exists()
    assert re.match(r"^artifacts-\d{2}-\d{2}-\d{2}-phase_b_r0_baseline_32$", out_dir.name)


def test_auto_out_dir_increments_index_inside_artifacts_dir(tmp_path: Path) -> None:
    first = Path(auto_out_dir("phase_b", base_dir=tmp_path))
    first.mkdir(parents=True, exist_ok=True)
    second = Path(auto_out_dir("phase_b", base_dir=tmp_path))

    assert first.parent == second.parent == tmp_path / "artifacts"
    assert first.name != second.name


def test_get_config_keeps_explicit_out_dir_override(tmp_path: Path) -> None:
    explicit_out_dir = str(tmp_path / "manual-output")
    cfg = get_config(
        "base",
        overrides={"out_dir": explicit_out_dir},
        config_dir=str(tmp_path),
    )
    assert cfg.out_dir == explicit_out_dir



