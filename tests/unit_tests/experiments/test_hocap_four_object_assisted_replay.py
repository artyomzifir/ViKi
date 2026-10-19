"""Checks for assembling a full trajectory without a hand-only fallback."""

import json

import h5py
import numpy as np
import pytest

pytest.importorskip("mujoco")

from experiments.hocap_four_object_assisted_replay import (
    WINDOWS,
    assemble_object_trajectory,
)


def _plans(tmp_path, timestamps):
    paths = {}
    for identifier, first, last in WINDOWS:
        path = tmp_path / f"object_{identifier}.h5"
        with h5py.File(path, "w") as plan:
            plan["source_pose"] = "hand_fit"
            plan["config_json"] = json.dumps({"pose_source": "hand_fit", "target_position_anchor": "wrist"})
            plan["q"] = np.full((last - first, 6), identifier, dtype=float)
            plan["gripper_joint_position"] = np.full(last - first, 0.2)
            plan["timestamps"] = timestamps[first:last]
            plan["metrics_json"] = json.dumps({
                "object_grasp": {
                    "object_id": identifier,
                    "prealign_start": 5,
                    "phase_bounds": [25, 45, last - first - 50, last - first - 30],
                },
            })
        paths[identifier] = path
    return paths


def test_four_object_trajectory_covers_all_frames_without_hand_only(tmp_path):
    timestamps = np.arange(831, dtype=np.int64) * 66667
    paths = _plans(tmp_path, timestamps)
    joints, drive, active, segments = assemble_object_trajectory(timestamps, paths)
    assert joints.shape == (831, 6)
    assert np.isfinite(joints).all() and np.isfinite(drive).all()
    assert [segment["object_id"] for segment in segments] == [2, 3, 4, 1]
    assert set(active) == {0, 1, 2, 3, 4}
    assert np.all(active[segments[0]["grasp_window"][0]:segments[0]["grasp_window"][1]] == 2)
    assert 2 < joints[185, 0] < 3


def test_four_object_trajectory_rejects_non_object_plan(tmp_path):
    timestamps = np.arange(831, dtype=np.int64) * 66667
    paths = _plans(tmp_path, timestamps)
    with h5py.File(paths[3], "r+") as plan:
        plan["metrics_json"][...] = json.dumps({"object_grasp": None})
    with pytest.raises(ValueError, match="lacks an aligned object-aware plan"):
        assemble_object_trajectory(timestamps, paths)
