"""Calibration-frame transforms and robot registry."""

import json

import numpy as np
import pytest

from viki.retarget.frames import (
    calibration_to_robot,
    load_rig_to_calibration,
    require_rig_frame,
    rig_pose_to_calibration,
    robot_to_calibration,
)
from viki.retarget.robots import ROBOT_CONFIGS, normalize_robot


def test_translation_only_frame_round_trip():
    points = np.array([[0.4, -0.1, 0.2], [0.5, 0.0, 0.3]])
    base = [0.2, -0.3, 0.1]
    robot = calibration_to_robot(points, base)
    np.testing.assert_allclose(robot, points - base)
    np.testing.assert_allclose(robot_to_calibration(robot, base), points)


def test_rig_and_known_mislabeled_rig_artifacts_are_accepted():
    assert require_rig_frame(np.asarray("rig")) == "rig"
    assert require_rig_frame("robot_base") == "rig"
    assert require_rig_frame("calibration_base") == "rig"
    with pytest.raises(ValueError, match="rerun Extract/Prepare"):
        require_rig_frame("camera")


def test_rig_pose_is_rotated_and_translated_once():
    # +90 deg about Z, followed by a calibration-frame translation.
    anchor = np.array([
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    position, rotation = rig_pose_to_calibration(
        np.array([[1.0, 0.0, 0.0]]), np.eye(3)[None], anchor
    )
    np.testing.assert_allclose(position, [[1.0, 3.0, 3.0]])
    np.testing.assert_allclose(rotation[0], anchor[:3, :3])


def test_episode_anchor_loads_or_defaults_to_identity(tmp_path):
    np.testing.assert_allclose(load_rig_to_calibration(tmp_path), np.eye(4))
    anchor = np.eye(4)
    anchor[:3, 3] = [0.1, -0.2, 0.3]
    (tmp_path / "world_anchor.json").write_text(json.dumps({
        "T_world_display": anchor.tolist(),
    }))
    np.testing.assert_allclose(load_rig_to_calibration(tmp_path), anchor)


def test_invalid_finite_position_is_excluded_from_interpolation(tmp_path):
    from viki.retarget.run import config_from_options, load_targets

    path = tmp_path / "cln.npz"
    np.savez_compressed(
        path,
        timestamps=np.arange(3),
        positions=np.array([[99.0, 99.0, 99.0], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]),
        rotations=np.tile(np.eye(3), (3, 1, 1)),
        valid=np.array([False, True, True]),
        omega=np.ones(3),
        gripper=np.zeros(3, dtype=bool),
        coordinate_frame="robot_base",
    )
    targets = load_targets(path, config_from_options("ur10"))
    np.testing.assert_allclose(targets.position_rig[0], [0.2, 0.3, 0.4])
    assert targets.confidence.tolist() == [0.0, 1.0, 1.0]


def test_robot_names_resolve():
    assert normalize_robot("ur10") is ROBOT_CONFIGS["ur10"]
    assert normalize_robot("ur10_description") is ROBOT_CONFIGS["ur10"]
    with pytest.raises(ValueError):
        normalize_robot("panda")


def test_default_retarget_config_tracks_orientation_with_explicit_offset():
    from viki.retarget.run import config_from_options

    cfg = config_from_options("ur10")
    assert cfg.weights.orientation == pytest.approx(0.01)
    np.testing.assert_allclose(
        cfg.hand_to_ee_rpy_deg,
        [-175.675, 15.45, -47.922],
    )
