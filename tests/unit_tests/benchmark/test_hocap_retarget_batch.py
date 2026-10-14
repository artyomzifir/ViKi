"""HO-Cap batch profile and stale-plan preflight checks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_hocap_retarget_batch import (
    _hand_to_ee_rpy_deg,
    _plan_metrics,
    _process_episode,
    _validate_smoke,
)
from viki.retarget.archive import write_hdf5_archive


def _episode(root: Path, side: str = "left") -> Path:
    root.mkdir()
    (root / "meta.json").write_text(json.dumps({
        "source": "hocap",
        "hocap_world_frame": "tag_1",
        "hocap_mano_sides_meta": [side],
    }))
    return root


def _plan(episode: Path, hand_rpy: list[float], scale: float = 1.0) -> None:
    write_hdf5_archive(episode / "plan.h5", {
        "coordinate_frame": "hocap_tag_1",
        "config_json": json.dumps({
            "robot": "ur10",
            "gripper": "robotiq_2f85",
            "base_position": [0.0, -1.0, 0.0],
            "table_plane_z_m": 0.0,
            "approach_sec": 0.0,
            "reference_policy": "zero",
            "object_grasp": False,
            "solver": {"collision_enabled": True},
            "collision_pairs": 8,
            "hand_to_ee_rpy_deg": hand_rpy,
            "velocity_limit_scale": scale,
        }),
        "q": np.zeros((2, 6)),
        "timestamps": np.arange(2),
        "metrics_json": "{}",
        "solver_status": "converged",
    })


def test_hocap_batch_requires_single_known_hand_side(tmp_path):
    episode = _episode(tmp_path / "left")
    assert _hand_to_ee_rpy_deg(episode) == [-175.675, 15.45, -47.922]
    right = _episode(tmp_path / "right", "right")
    assert _hand_to_ee_rpy_deg(right) == [4.325, 15.45, -47.922]
    metadata = json.loads((episode / "meta.json").read_text())
    metadata["hocap_mano_sides_meta"] = ["left", "right"]
    (episode / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="exactly one"):
        _hand_to_ee_rpy_deg(episode)


def test_hocap_batch_rejects_old_side_mapping_and_scaled_speed(tmp_path):
    episode = _episode(tmp_path / "left")
    _plan(episode, [4.325, 15.45, -47.922])
    with pytest.raises(ValueError, match="unexpected plan configuration"):
        _plan_metrics(episode / "plan.h5")
    _plan(episode, [-175.675, 15.45, -47.922], scale=2.0)
    with pytest.raises(ValueError, match="unexpected plan configuration"):
        _plan_metrics(episode / "plan.h5")
    _plan(episode, [-175.675, 15.45, -47.922])
    assert _plan_metrics(episode / "plan.h5")["frames"] == 2


@pytest.mark.parametrize("side,expected_roll", [("left", "-175.675"), ("right", "4.325")])
def test_hocap_batch_passes_explicit_mapping_to_cli(tmp_path, monkeypatch, side, expected_roll):
    episode = _episode(tmp_path / side, side)
    (episode / "cln.npz").write_bytes(b"placeholder")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("scripts.run_hocap_retarget_batch.subprocess.run", run)
    result = _process_episode(tmp_path, episode)
    assert result["status"] == "failed"
    command = commands[0]
    start = command.index("--hand-to-ee-rpy-deg")
    assert command[start + 1:start + 4] == [expected_roll, "15.45", "-47.922"]
    assert "--velocity-limit-scale" not in command
    assert command[command.index("--reference-policy") + 1] == "zero"


def test_hocap_smoke_gate_requires_convergence_and_finite_safe_margins():
    smoke = {
        "frames": 676,
        "solver_status": "converged",
        "metrics": {
            "position_rmse_mm": 20.0,
            "position_p95_mm": 40.0,
            "orientation_rmse_deg": 30.0,
            "velocity_repair_max_rad": 0.0,
            "min_velocity_limit_margin_rad_s": 0.1,
            "min_urdf_velocity_limit_margin_rad_s": 0.1,
            "min_floor_margin_mm": 1.0,
            "min_collision_margin": 0.01,
        },
    }
    _validate_smoke(smoke)
    smoke["solver_status"] = "max_iterations"
    with pytest.raises(ValueError, match="convergence"):
        _validate_smoke(smoke)
    smoke["solver_status"] = "converged"
    smoke["metrics"]["min_collision_margin"] = float("nan")
    with pytest.raises(ValueError, match="min_collision_margin"):
        _validate_smoke(smoke)


def test_hocap_preflight_never_starts_work_or_writes_status(tmp_path, monkeypatch):
    from scripts.run_hocap_retarget_batch import main

    for index in range(28):
        episode = _episode(tmp_path / f"episode_{index:02d}", "right")
        (episode / "cln.npz").write_bytes(b"placeholder")
    monkeypatch.setattr(sys, "argv", ["batch", str(tmp_path), "--preflight-only"])
    monkeypatch.setattr(
        "scripts.run_hocap_retarget_batch._process_episode",
        lambda *_args: pytest.fail("preflight started retarget"),
    )
    main()
    assert not (tmp_path / "retarget_batch_status.json").exists()
