"""Calibration-frame transforms and robot registry."""

import json

import numpy as np
import pytest

from viki.retarget.frames import (
    base_rotation,
    base_transform,
    calibration_to_robot,
    calibration_rotation_to_robot,
    load_rig_to_calibration,
    require_rig_frame,
    rig_pose_to_calibration,
    robot_rotation_to_calibration,
    robot_to_calibration,
)
from viki.retarget.robots import ROBOT_CONFIGS, normalize_robot


def test_base_pose_frame_round_trip():
    points = np.array([[0.4, -0.1, 0.2], [0.5, 0.0, 0.3]])
    base = [0.2, -0.3, 0.1]
    rpy = [10.0, -20.0, 90.0]
    robot = calibration_to_robot(points, base, rpy)
    np.testing.assert_allclose(
        robot_to_calibration(robot, base, rpy), points, atol=1e-12
    )

    rotations = np.stack((np.eye(3), base_rotation([5.0, 6.0, 7.0])))
    robot_rotations = calibration_rotation_to_robot(rotations, rpy)
    np.testing.assert_allclose(
        robot_rotation_to_calibration(robot_rotations, rpy), rotations, atol=1e-12
    )
    pose = base_transform(base, rpy)
    np.testing.assert_allclose(pose[:3, :3], base_rotation(rpy))
    np.testing.assert_allclose(pose[:3, 3], base)


def test_yaw_turns_robot_x_into_calibration_y():
    base = [0.7, 0.0, 0.0]
    calibration_point = np.array([[0.7, 1.0, 0.0]])
    np.testing.assert_allclose(
        calibration_to_robot(calibration_point, base, [0.0, 0.0, 90.0]),
        [[1.0, 0.0, 0.0]],
        atol=1e-12,
    )


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
    targets = load_targets(path, config_from_options("ur10", {
        "target_position_anchor": "wrist",
    }))
    np.testing.assert_allclose(targets.position_rig[0], [0.2, 0.3, 0.4])
    assert targets.confidence.tolist() == [0.0, 1.0, 1.0]
    # Protected V1 artifacts stored ``closed`` as a boolean.  The current
    # semantic contract is the inverse: 0 closed, 1 fully open.
    np.testing.assert_allclose(targets.gripper_opening, np.ones(3))


def test_current_float_gripper_opening_is_preserved(tmp_path):
    from viki.retarget.run import config_from_options, load_targets

    path = tmp_path / "cln.npz"
    np.savez_compressed(
        path,
        timestamps=np.arange(3),
        positions=np.tile([0.2, 0.3, 0.4], (3, 1)),
        rotations=np.tile(np.eye(3), (3, 1, 1)),
        valid=np.ones(3, dtype=bool),
        gripper=np.array([1.0, 0.45, 0.0], dtype=np.float32),
        coordinate_frame="rig",
    )
    targets = load_targets(path, config_from_options("ur10", {
        "target_position_anchor": "wrist",
    }))
    np.testing.assert_allclose(targets.gripper_opening, [1.0, 0.45, 0.0])


def test_default_target_is_thumb_index_midpoint_with_palm_orientation(tmp_path):
    from viki.retarget.run import config_from_options, load_targets

    points = np.zeros((3, 21, 3), dtype=np.float32)
    points[:, 4] = [[0.20, 0.10, 0.40], [0.22, 0.10, 0.40], [0.24, 0.10, 0.40]]
    points[:, 8] = [[0.30, 0.10, 0.40], [0.32, 0.10, 0.40], [0.34, 0.10, 0.40]]
    rotations = np.tile(np.eye(3), (3, 1, 1))
    landmark_confidence = np.ones((3, 21), dtype=np.float32)
    landmark_confidence[1, 8] = 0.2
    path = tmp_path / "cln.npz"
    np.savez_compressed(
        path,
        timestamps=np.arange(3),
        positions=np.tile([9.0, 9.0, 9.0], (3, 1)),
        rotations=rotations,
        valid=np.ones(3, dtype=bool),
        gripper=np.ones(3, dtype=np.float32),
        smoothed_points=points,
        landmark_ids=np.arange(21, dtype=np.int32),
        landmark_confidence=landmark_confidence,
        coordinate_frame="rig",
    )
    targets = load_targets(path, config_from_options("ur10"))
    np.testing.assert_allclose(
        targets.position_rig,
        [[0.25, 0.10, 0.40], [0.27, 0.10, 0.40], [0.29, 0.10, 0.40]],
    )
    np.testing.assert_allclose(targets.rotation_rig, rotations)
    np.testing.assert_allclose(targets.confidence, [1.0, 0.2, 1.0])
    assert targets.position_anchor == "pinch_center"


def test_robot_names_resolve():
    assert normalize_robot("ur10") is ROBOT_CONFIGS["ur10"]
    assert normalize_robot("ur10_description") is ROBOT_CONFIGS["ur10"]
    with pytest.raises(ValueError):
        normalize_robot("panda")


def test_default_retarget_config_tracks_orientation_with_explicit_offset():
    from viki.retarget.run import config_from_options

    cfg = config_from_options("ur10")
    assert cfg.weights.orientation == pytest.approx(0.01)
    assert cfg.target_position_anchor == "pinch_center"
    np.testing.assert_allclose(cfg.base_rpy_deg, [0.0, 0.0, 0.0])
    assert cfg.adapter.kind == "user_cylinder"
    assert cfg.adapter.length_m == pytest.approx(0.011)
    np.testing.assert_allclose(
        cfg.hand_to_ee_rpy_deg,
        [-175.675, 15.45, -47.922],
    )


def test_adapter_options_are_user_facing_millimetres():
    from viki.retarget.run import config_from_options

    cfg = config_from_options("ur10", {
        "adapter_translation_mm": [10.0, -20.0, 30.0],
        "adapter_rpy_deg": [1.0, 2.0, 3.0],
        "adapter_radius_mm": 18.0,
    })
    np.testing.assert_allclose(cfg.adapter.translation_m, [0.01, -0.02, 0.03])
    np.testing.assert_allclose(cfg.adapter.rpy_deg, [1.0, 2.0, 3.0])
    assert cfg.adapter.radius_m == pytest.approx(0.018)


def test_base_pose_options_are_preserved():
    from viki.retarget.run import config_from_options

    cfg = config_from_options("ur10", {
        "base_position": [0.7, -0.1, 0.2],
        "base_rpy_deg": [5.0, -10.0, 180.0],
    })
    np.testing.assert_allclose(cfg.base_position, [0.7, -0.1, 0.2])
    np.testing.assert_allclose(cfg.base_rpy_deg, [5.0, -10.0, 180.0])


def test_invalid_target_anchor_and_adapter_are_rejected():
    from viki.retarget.run import config_from_options

    with pytest.raises(ValueError, match="target_position_anchor"):
        config_from_options("ur10", {"target_position_anchor": "thumb_tip"})
    with pytest.raises(ValueError, match="adapter radius"):
        config_from_options("ur10", {"adapter_radius_mm": 0.0})
