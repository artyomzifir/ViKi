import numpy as np
import pytest

from experiments.grasp_gate import BilateralForceGate, insert_grasp_dwell


def test_dwell_holds_arm_and_gripper_before_lift():
    times = np.array([0.0, 0.3, 0.4, 0.5, 0.6])
    joints = np.arange(5)[:, None]
    drive = np.arange(5, dtype=float)
    paused, paused_joints, paused_drive, start, end = insert_grasp_dwell(
        times, joints, drive, close_frame=2, frame_start=1, dwell_s=0.5,
    )
    np.testing.assert_allclose(paused, [0.0, 0.3, 0.4, 0.9, 1.0, 1.1])
    np.testing.assert_array_equal(paused_joints[:, 0], [0, 1, 2, 2, 3, 4])
    np.testing.assert_array_equal(paused_drive, [0, 1, 2, 2, 3, 4])
    assert (start, end) == pytest.approx((0.4, 0.9))


def test_dwell_can_hold_full_grasp_command_before_lift():
    times = np.array([0.0, 0.3, 0.4, 0.5, 0.6])
    joints = np.arange(5)[:, None]
    drive = np.array([0.0, 0.0, 0.1, 0.2, 0.5])
    _, held_joints, held_drive, _, _ = insert_grasp_dwell(
        times, joints, drive, close_frame=2, frame_start=1, dwell_s=0.5,
        hold_drive=0.5,
    )
    np.testing.assert_array_equal(held_joints[:, 0], [0, 1, 2, 2, 3, 4])
    np.testing.assert_allclose(held_drive, [0, 0, 0.5, 0.5, 0.2, 0.5])


def test_bilateral_gate_requires_continuous_force():
    gate = BilateralForceGate(1.0, 0.1)
    assert not gate.observe(1.1, 0.9, 0.05)
    assert not gate.observe(1.1, 1.2, 0.05)
    assert not gate.observe(0.0, 1.2, 0.05)
    assert not gate.observe(1.1, 1.2, 0.05)
    assert gate.observe(1.1, 1.2, 0.05)
    assert not gate.observe(0.0, 1.2, 0.05)


def test_invalid_gate_or_dwell_is_rejected():
    with pytest.raises(ValueError, match="dwell"):
        insert_grasp_dwell(np.arange(4), np.zeros((4, 6)), np.zeros(4), 1, 0, 0)
    with pytest.raises(ValueError, match="force"):
        BilateralForceGate(float("nan"), 0.1)
