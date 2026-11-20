import numpy as np
import pytest

from experiments.gripper_control import pressure_commands


def test_pressure_is_limited_to_grasp_phase():
    command = np.full(40, 0.4)
    phases = {
        "mode": "experimental_cube_face_transfer_v2",
        "grasp_start": 5,
        "close_complete": 10,
        "phase_bounds": [2, 12, 25, 35],
    }
    result = pressure_commands(command, phases, 0.1)
    np.testing.assert_allclose(result[:6], command[:6])
    np.testing.assert_allclose(result[10:26], 0.5)
    np.testing.assert_allclose(result[35:], command[35:])


def test_pressure_requires_current_grasp_plan():
    with pytest.raises(ValueError, match="supported phased grasp"):
        pressure_commands(np.zeros(40), {"mode": "experimental_cube_face_transfer"}, 0.1)


def test_pressure_supports_tracked_object_plan():
    command = np.full(40, 0.4)
    phases = {
        "mode": "experimental_tracked_side_grasp_v1",
        "grasp_start": 5,
        "close_complete": 10,
        "phase_bounds": [2, 12, 25, 35],
    }
    result = pressure_commands(command, phases, 0.1)
    np.testing.assert_allclose(result[10:26], 0.5)


def test_full_close_holds_until_release_then_reopens():
    command = np.full(40, 0.4)
    phases = {
        "mode": "experimental_tracked_side_grasp_v1",
        "grasp_start": 5,
        "close_complete": 10,
        "phase_bounds": [2, 12, 25, 35],
    }
    result = pressure_commands(command, phases, 0.0, fully_close=True)
    np.testing.assert_allclose(result[:6], command[:6])
    np.testing.assert_allclose(result[10:26], 0.8)
    np.testing.assert_allclose(result[35:], command[35:])
    assert result[26] < 0.8
    with pytest.raises(ValueError, match="mutually exclusive"):
        pressure_commands(command, phases, 0.1, fully_close=True)


def test_zero_pressure_preserves_geometric_plan():
    command = np.array([0.0, 0.3, 0.8])
    np.testing.assert_allclose(pressure_commands(command, None, 0.0), command)
