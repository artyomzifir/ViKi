"""The dependency-free trajectory bundle: what it writes, and what it admits to."""

import json

import numpy as np
import pytest

from viki.contracts import Episode
from viki.episode import mark_stage, new_episode
from viki.export import export_trajectories
from viki.retarget.archive import write_hdf5_archive


def _planned_episode(tmp_path, n=12, task="pick up the block"):
    ep = new_episode(tmp_path)
    meta = json.loads(ep.meta_path.read_text())
    meta["task"] = task
    meta["calibration_preset"] = "unit-test"
    ep.meta_path.write_text(json.dumps(meta))
    write_hdf5_archive(
        ep.plan_h5,
        {
            "q": np.arange(n * 6, dtype=np.float64).reshape(n, 6),
            "joint_velocity": np.zeros((n, 6)),
            "gripper_opening": np.linspace(0.0, 1.0, n),
            "timestamps": np.arange(n, dtype=np.int64) * 33333,
            "target_position_calibration": np.zeros((n, 3), np.float32),
            "achieved_position_calibration": np.ones((n, 3), np.float32),
            "position_error_m": np.full(n, 0.01, np.float32),
            "dt": 1 / 30.0,
            "fps": 30.0,
            "robot": "ur10_official_description",
            "robot_key": "ur10",
            "solver_status": "converged",
            "joint_names_json": json.dumps([f"j{i}" for i in range(6)]),
            "config_json": "{}",
            "metrics_json": json.dumps({"position_rmse_mm": 12.5}),
        },
    )
    mark_stage(ep, "retarget", robot="ur10_official_description")
    return ep, n


def test_bundle_round_trips_the_plan(tmp_path):
    ep, n = _planned_episode(tmp_path / "ep")
    out = export_trajectories([str(ep.root)], tmp_path / "bundle", name="v0-test")

    manifest = json.loads((tmp_path / "bundle" / "dataset.json").read_text())
    assert manifest["schema"] == "viki_trajectory_bundle"
    assert manifest["name"] == "v0-test"
    assert manifest["robots"] == ["ur10"]
    assert manifest["episode_count"] == 1 and manifest["total_frames"] == n
    assert manifest["code_sha256"]
    # Units and frames are stated rather than assumed by the consumer.
    assert manifest["units"]["position"] == "m"
    assert "calibration frame" in manifest["frames"]["poses"]

    arrays = np.load(
        tmp_path / "bundle" / "episodes" / ep.id / "trajectory.npz", allow_pickle=False
    )
    assert arrays["q"].shape == (n, 6)
    np.testing.assert_allclose(arrays["q"], np.arange(n * 6).reshape(n, 6))
    assert arrays["gripper_opening"].shape == (n,)
    assert str(out).endswith("bundle")

    record = json.loads((tmp_path / "bundle" / "episodes" / ep.id / "meta.json").read_text())
    assert record["task"] == "pick up the block"
    assert record["robot_key"] == "ur10"
    assert record["frames"] == n
    assert record["retarget_metrics"]["position_rmse_mm"] == 12.5


def test_manifest_admits_that_nothing_was_screened(tmp_path):
    ep, _ = _planned_episode(tmp_path / "ep")
    export_trajectories([str(ep.root)], tmp_path / "bundle")

    manifest = json.loads((tmp_path / "bundle" / "dataset.json").read_text())
    # A consumer must not be able to mistake this for a screened dataset.
    assert "NONE" in manifest["screening"]
    screening = manifest["episodes"][0]["screening"]
    assert screening["replay_run"] is False
    assert screening["replay_verdict"] is None
    assert screening["outcome_rated"] is False


def test_unretargeted_episode_is_skipped_not_fatal(tmp_path):
    good, n = _planned_episode(tmp_path / "good")
    # new_episode names by timestamp, so two made in the same second collide.
    # The ids have to differ for this test to say anything.
    made = new_episode(tmp_path / "bad")
    bad_root = made.root.parent / f"{made.root.name}-second"
    made.root.rename(bad_root)
    bad = Episode(root=bad_root)
    assert bad.id != good.id

    export_trajectories([str(good.root), str(bad.root)], tmp_path / "bundle")

    manifest = json.loads((tmp_path / "bundle" / "dataset.json").read_text())
    assert manifest["episode_count"] == 1
    assert [s["episode_id"] for s in manifest["skipped"]] == [bad.id]
    assert "retarget" in manifest["skipped"][0]["why"]
    assert not (tmp_path / "bundle" / "episodes" / bad.id).exists()
    assert [d.name for d in (tmp_path / "bundle" / "episodes").iterdir()] == [good.id]


def test_nothing_exportable_fails_loudly(tmp_path):
    ep = new_episode(tmp_path / "ep")
    with pytest.raises(RuntimeError, match="no episodes could be exported"):
        export_trajectories([str(ep.root)], tmp_path / "bundle")


def test_missing_cln_does_not_prevent_export(tmp_path):
    """The plan is the deliverable; the human trajectory is carried when present."""
    ep, n = _planned_episode(tmp_path / "ep")
    assert not ep.cln_npz.exists()
    export_trajectories([str(ep.root)], tmp_path / "bundle")
    arrays = np.load(
        tmp_path / "bundle" / "episodes" / ep.id / "trajectory.npz", allow_pickle=False
    )
    assert "q" in arrays.files
    assert "hand_position_rig" not in arrays.files
