"""Tracked-object collision barriers and their input contract."""

import numpy as np
import pytest

from viki.perception import OBJECT_MODEL_SCHEMA
from viki.retarget.obstacles import SceneCollisionKinematics, load_object_boxes


def _track(path, confidence=1.0):
    axis = np.linspace(-0.02, 0.02, 5)
    surface = np.array(np.meshgrid(axis, axis, axis)).reshape(3, -1).T
    np.savez_compressed(
        path, schema=OBJECT_MODEL_SCHEMA, object_ids=np.array([3, 4]),
        model_offsets=np.array([0, len(surface), 2 * len(surface)]),
        model_xyz_object=np.concatenate((surface, surface)),
        translation_world=np.tile([[0.3, 0.0, 0.2], [0.8, 0.2, 0.2]], (3, 1, 1)),
        rotation_world_object=np.tile(np.eye(3), (3, 2, 1, 1)),
        track_confidence=np.full((3, 2), confidence),
    )


def test_object_boxes_are_aligned_and_enclose_the_cloud(tmp_path):
    path = tmp_path / "models.npz"
    _track(path)
    selected, obstacles = load_object_boxes(path, 3, 3, (0, -1, 0), (0, 0, 0))
    assert selected.identifier == 3
    assert [item.identifier for item in obstacles] == [4]
    np.testing.assert_allclose(selected.positions[0], [0.3, 1.0, 0.2])
    np.testing.assert_allclose(selected.size, [0.04, 0.04, 0.04])


def test_object_boxes_fail_closed_on_missing_or_weak_tracks(tmp_path):
    path = tmp_path / "models.npz"
    _track(path, confidence=0.2)
    with pytest.raises(ValueError, match="complete, confident"):
        load_object_boxes(path, 3, 3, (0, 0, 0), (0, 0, 0))
    _track(path)
    with pytest.raises(ValueError, match="complete, confident"):
        load_object_boxes(path, 3, 4, (0, 0, 0), (0, 0, 0))


def test_tracked_grasp_cannot_disable_collision_barriers():
    from viki.retarget.run import config_from_options

    with pytest.raises(ValueError, match="requires collision constraints"):
        config_from_options(options={
            "object_grasp": True, "object_track_path": "models.npz",
            "object_id": 3, "collision_enabled": False,
        })


def test_object_barrier_jacobian_and_contact_exemption(tmp_path):
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")
    from viki.retarget.grippers import attach_gripper, normalize_gripper
    from viki.retarget.robots import normalize_robot
    from viki.retarget.run import _load_robot_description
    from viki.retarget.solver import PinocchioKinematics

    path = tmp_path / "models.npz"
    _track(path)
    selected, obstacles = load_object_boxes(path, 3, 3, (0, 0, 0), (0, 0, 0))
    robot = normalize_robot("ur10")
    gripper = normalize_gripper("robotiq_2f85")
    try:
        arm = _load_robot_description(robot.description)
    except Exception as exc:
        pytest.skip(f"robot description unavailable: {exc}")
    assembly = attach_gripper(arm, robot, gripper)
    base = PinocchioKinematics(
        assembly.robot, assembly.orientation_frame,
        collision_pairs=8, collision_min_distance_m=0.02,
        actuated_joint_names=robot.joint_names,
        actuated_position_limits=robot.position_limits,
        reference_q=np.asarray(robot.home_q),
        passive_joint_positions={assembly.drive_joint: np.zeros(3)},
        gripper_prefix=assembly.gripper_prefix,
        position_points=assembly.position_points,
    )
    kin = SceneCollisionKinematics(
        base, selected, obstacles, np.array([False, True, False]),
        np.array([False, True, False]), np.zeros((3, 3)),
        np.tile(np.eye(3), (3, 1, 1)),
    )
    q = np.asarray(robot.home_q)
    kin.set_frame(0)
    margins, jacobian = kin._object_constraints(q, linearize=True)
    np.testing.assert_allclose(margins, kin.object_margins(q))
    assert len(margins) == jacobian.shape[0] and jacobian.shape[1] == 6
    assert np.isfinite(margins).all() and np.isfinite(jacobian).all()
    candidate = next(index for index in range(len(margins))
                     if margins[index] > 0.01 and abs(jacobian[index, 0]) > 0.01)
    delta = np.zeros(6)
    delta[0] = 1e-5
    plus = kin.object_margins(q + delta)[candidate]
    minus = kin.object_margins(q - delta)[candidate]
    assert jacobian[candidate, 0] == pytest.approx((plus - minus) / 2e-5, abs=1e-3)
    kin.set_frame(1)
    contact_margins, _ = kin._object_constraints(q, linearize=True)
    assert len(contact_margins) < len(margins) + len(obstacles)
