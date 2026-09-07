"""Physical tool registry and robot/gripper composition."""

import numpy as np
import pytest

from viki.retarget.grippers import (
    DEFAULT_GRIPPER,
    GRIPPER_CONFIGS,
    gripper_catalog,
    normalize_gripper,
)


def test_catalog_exposes_requested_two_finger_family_and_safe_default():
    assert DEFAULT_GRIPPER == "robotiq_2f85"
    assert set(GRIPPER_CONFIGS) == {
        "robotiq_2f85",
        "robotiq_2f140",
        "schunk_wsg50",
        "smc_mhz2_16d",
        "smc_mhz2_20d",
    }
    rows = {row["key"]: row for row in gripper_catalog()}
    assert rows[DEFAULT_GRIPPER]["available"] is True
    assert all(row["max_width_m"] > 0.0 for row in rows.values())


def test_registry_maps_binary_state_to_physical_joint_and_opening():
    cfg = normalize_gripper("2f-85")
    closed = np.array([False, True, False])
    np.testing.assert_allclose(cfg.joint_positions(closed), [0.0, 0.8, 0.0])
    np.testing.assert_allclose(cfg.opening_widths(closed), [0.085, 0.0, 0.085])


def test_unavailable_and_unknown_models_fail_before_ik():
    with pytest.raises(ValueError, match="SCHUNK WSG-50 is not available"):
        normalize_gripper("wsg-50")
    with pytest.raises(ValueError, match="unknown physical gripper"):
        normalize_gripper("imaginary")


@pytest.mark.slow
def test_robotiq_is_attached_at_tool0_and_moves_as_passive_geometry():
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")

    from viki.retarget.grippers import attach_gripper
    from viki.retarget.run import _load_robot_description
    from viki.retarget.robots import normalize_robot
    from viki.retarget.solver import PinocchioKinematics

    robot_cfg = normalize_robot("ur3")
    gripper_cfg = normalize_gripper("robotiq_2f85")
    try:
        arm = _load_robot_description(robot_cfg.description)
        assembly = attach_gripper(arm, robot_cfg, gripper_cfg)
    except Exception as exc:  # noqa: BLE001 - first run may need a model download
        pytest.skip(f"robot/gripper descriptions unavailable: {exc}")

    joint_position = gripper_cfg.joint_positions(np.array([False, True]))
    kinematics = PinocchioKinematics(
        assembly.robot,
        assembly.tcp_frame,
        collision_pairs=0,
        collision_min_distance_m=0.0,
        actuated_joint_names=robot_cfg.joint_names,
        passive_joint_positions={assembly.drive_joint: joint_position},
        gripper_prefix=assembly.gripper_prefix,
    )

    # The tool adds a physical DoF internally, but never leaks it into the
    # six-dimensional arm command solved by retargeting.
    assert kinematics.nq == 6
    assert kinematics.model.nv == 7
    assert kinematics.joint_names == list(robot_cfg.joint_names)
    assert assembly.tcp_frame.endswith("robotiq_arg2f_tcp")

    kinematics.set_frame(0)
    open_tcp, _ = kinematics.pose(kinematics.q_reference)
    open_points = kinematics.joint_points(kinematics.q_reference)
    kinematics.set_frame(1)
    closed_tcp, _ = kinematics.pose(kinematics.q_reference)
    closed_points = kinematics.joint_points(kinematics.q_reference)

    # Opening does not move the task frame; it only articulates the fingers.
    np.testing.assert_allclose(open_tcp, closed_tcp, atol=1e-10)
    gripper_mask = np.asarray(kinematics.point_groups) == "gripper"
    robot_mask = ~gripper_mask
    assert int(gripper_mask.sum()) >= 6
    np.testing.assert_allclose(open_points[robot_mask], closed_points[robot_mask], atol=1e-10)
    assert float(np.max(np.linalg.norm(
        open_points[gripper_mask] - closed_points[gripper_mask], axis=1
    ))) > 0.01

    # At neutral, TCP must lie beyond tool0 by adapter + gripper length. This
    # catches accidentally solving for the wrist or mounting the URDF backwards.
    tool0_id = kinematics.model.getFrameId(robot_cfg.mount_frame)
    tool0 = np.asarray(kinematics.data.oMf[tool0_id].translation)
    assert np.linalg.norm(open_tcp - tool0) == pytest.approx(0.185, abs=0.003)
    assert "gripper" in kinematics.link_groups
