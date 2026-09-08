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


def test_registry_maps_continuous_opening_to_physical_joint_and_width():
    cfg = normalize_gripper("2f-85")
    opening = np.array([1.0, 0.5, 0.0])
    np.testing.assert_allclose(cfg.joint_positions(opening), [0.0, 0.4, 0.8])
    np.testing.assert_allclose(cfg.opening_widths(opening), [0.085, 0.0425, 0.0])


def test_physical_speed_profile_turns_binary_jump_into_a_ramp():
    cfg = normalize_gripper("2f-85")
    target = np.array([1.0, 0.0, 0.0, 0.0])
    profiled = cfg.profile_opening(target, dt=1 / 30)
    max_step = 0.150 / 30 / 0.085
    np.testing.assert_allclose(np.diff(profiled), [-max_step] * 3)


def test_unavailable_and_unknown_models_fail_before_ik():
    with pytest.raises(ValueError, match="SCHUNK WSG-50 is not available"):
        normalize_gripper("wsg-50")
    with pytest.raises(ValueError, match="unknown physical gripper"):
        normalize_gripper("imaginary")


def test_cylinder_adapter_has_user_pose_and_vector_length():
    from viki.retarget.adapters import CylinderGripperAdapter

    adapter = CylinderGripperAdapter(
        translation_m=(0.01, -0.02, 0.03),
        rpy_deg=(10.0, 20.0, 30.0),
        radius_m=0.018,
    )
    assert adapter.kind == "user_cylinder"
    assert adapter.length_m == pytest.approx(np.linalg.norm([0.01, -0.02, 0.03]))
    np.testing.assert_allclose(
        adapter.rotation_matrix().T @ adapter.rotation_matrix(), np.eye(3), atol=1e-12
    )


@pytest.mark.slow
def test_robotiq_is_attached_at_tool0_and_moves_as_passive_geometry():
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")

    from viki.retarget.grippers import attach_gripper
    from viki.retarget.adapters import CylinderGripperAdapter
    from viki.retarget.run import _load_robot_description
    from viki.retarget.robots import normalize_robot
    from viki.retarget.solver import PinocchioKinematics

    robot_cfg = normalize_robot("ur3")
    gripper_cfg = normalize_gripper("robotiq_2f85")
    try:
        arm = _load_robot_description(robot_cfg.description)
        assembly = attach_gripper(
            arm,
            robot_cfg,
            gripper_cfg,
            CylinderGripperAdapter(translation_m=(0.0, 0.0, 0.020)),
        )
    except Exception as exc:  # noqa: BLE001 - first run may need a model download
        pytest.skip(f"robot/gripper descriptions unavailable: {exc}")

    joint_position = gripper_cfg.joint_positions(np.array([1.0, 0.0]))
    kinematics = PinocchioKinematics(
        assembly.robot,
        assembly.orientation_frame,
        collision_pairs=0,
        collision_min_distance_m=0.0,
        actuated_joint_names=robot_cfg.joint_names,
        passive_joint_positions={assembly.drive_joint: joint_position},
        gripper_prefix=assembly.gripper_prefix,
        position_points=assembly.position_points,
        floor_z_m=0.0,
    )

    # The tool adds a physical DoF internally, but never leaks it into the
    # six-dimensional arm command solved by retargeting.
    assert kinematics.nq == 6
    assert kinematics.model.nv == 7
    assert kinematics.joint_names == list(robot_cfg.joint_names)
    assert assembly.tcp_frame.endswith("grasp_center")
    assert assembly.orientation_frame.endswith("robotiq_arg2f_tcp")

    kinematics.set_frame(0)
    open_tcp, open_rotation = kinematics.pose(kinematics.q_reference)
    open_points = kinematics.joint_points(kinematics.q_reference)
    kinematics.set_frame(1)
    closed_tcp, closed_rotation = kinematics.pose(kinematics.q_reference)
    closed_points = kinematics.joint_points(kinematics.q_reference)

    # The task point is the midpoint of the moving jaw panels, not the old
    # fixed axial TCP beyond the fingertips. The four-bar linkage shifts that
    # midpoint distally as it closes while keeping it on the symmetry axis.
    assert np.linalg.norm(closed_tcp - open_tcp) > 0.01
    np.testing.assert_allclose(open_rotation, closed_rotation, atol=1e-10)
    axial_shift = open_rotation[:, 2] @ (closed_tcp - open_tcp)
    np.testing.assert_allclose(
        closed_tcp - open_tcp,
        axial_shift * open_rotation[:, 2],
        atol=1e-9,
    )
    gripper_mask = np.asarray(kinematics.point_groups) == "gripper"
    robot_mask = ~gripper_mask
    assert int(gripper_mask.sum()) >= 6
    np.testing.assert_allclose(open_points[robot_mask], closed_points[robot_mask], atol=1e-10)
    assert float(np.max(np.linalg.norm(
        open_points[gripper_mask] - closed_points[gripper_mask], axis=1
    ))) > 0.01

    # At neutral, the tracked point is adapter + panel-centre depth, roughly
    # 20 + 130 mm from tool0 (the retired vendor TCP was at 20 + 174 mm).
    tool0_id = kinematics.model.getFrameId(robot_cfg.mount_frame)
    tool0 = np.asarray(kinematics.data.oMf[tool0_id].translation)
    assert np.linalg.norm(open_tcp - tool0) == pytest.approx(0.150, abs=0.003)
    # The floor owns one exact support point per movable collision body, plus
    # the two explicit jaw contact centres.  Evaluate derivatives away from a
    # mesh face tie, where the minimum-height support vertex is unique.
    assert len(kinematics._floor_geometry_specs) == sum(
        int(item.parentJoint) != 0
        for item in assembly.robot.collision_model.geometryObjects
    )
    floor_q = np.array([0.12, -0.31, 0.27, -0.19, 0.23, -0.14])
    kinematics.set_frame(0)
    margin, jacobian = kinematics.floor_linearization(floor_q)
    assert len(margin) == len(jacobian) and jacobian.shape[1] == 6
    # The old body-origin constraint considered this neutral pose legal.  The
    # actual open-finger mesh protrudes through z=0 and must now be detected.
    assert float(np.min(kinematics.floor_margins(kinematics.q_reference))) < -0.005
    eps = 1e-7
    for joint in range(kinematics.nq):
        step = np.zeros(kinematics.nq)
        step[joint] = eps
        finite_difference = (
            kinematics.floor_margins(floor_q + step) - margin
        ) / eps
        np.testing.assert_allclose(jacobian[:, joint], finite_difference, atol=2e-6)

    # A tilted robot base makes calibration z=0 an oblique plane in robot
    # coordinates. Its analytical point Jacobian must remain exact.
    kinematics.floor_normal = np.asarray([0.2, -0.3, 0.9327379])
    kinematics.floor_normal /= np.linalg.norm(kinematics.floor_normal)
    kinematics.floor_offset_m = -1.0
    tilted_margin, tilted_jacobian = kinematics.floor_linearization(
        floor_q
    )
    for joint in range(kinematics.nq):
        step = np.zeros(kinematics.nq)
        step[joint] = eps
        finite_difference = (
            kinematics.floor_margins(floor_q + step) - tilted_margin
        ) / eps
        np.testing.assert_allclose(
            tilted_jacobian[:, joint], finite_difference, atol=2e-6
        )
    assert "gripper" in kinematics.link_groups
    adapter_visual = next(
        item for item in kinematics.visual_geometries()
        if item["name"] == "adapter__cylinder_visual"
    )
    assert adapter_visual["primitive"]["type"] == "cylinder"
    assert adapter_visual["primitive"]["radius"] == pytest.approx(0.035)
    assert adapter_visual["primitive"]["length"] == pytest.approx(0.020)
    mount_frame = kinematics.model.frames[tool0_id]
    expected_center = (
        np.asarray(mount_frame.placement.translation)
        + np.asarray(mount_frame.placement.rotation) @ np.array([0.0, 0.0, 0.010])
    )
    np.testing.assert_allclose(
        np.asarray(adapter_visual["placement"])[:3, 3],
        expected_center,
        atol=1e-10,
    )
    gripper_base_visual = next(
        item for item in kinematics.visual_geometries()
        if item["name"].endswith("robotiq_arg2f_base_link_0")
    )
    np.testing.assert_allclose(
        np.asarray(adapter_visual["placement"])[:3, 3],
        0.5 * (
            np.asarray(mount_frame.placement.translation)
            + np.asarray(gripper_base_visual["placement"])[:3, 3]
        ),
        atol=1e-10,
    )
    assert any(
        str(item.name) == "adapter__cylinder_collision"
        for item in assembly.robot.collision_model.geometryObjects
    )
