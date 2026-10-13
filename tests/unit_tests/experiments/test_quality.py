"""Cube centering and orientation are measured modulo cube symmetry."""

import numpy as np
import pytest

from experiments.quality import CUBE_SYMMETRIES, cube_pose_errors, quality_penalty


def test_cube_pose_errors_include_center_and_uprightness():
    tilt = np.deg2rad(10)
    rotation = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                         [0, np.sin(tilt), np.cos(tilt)]])
    result = cube_pose_errors(np.array([0.006, -0.008, 0.06]),
                              np.array([0.0, 0.0, 0.02]),
                              rotation, np.eye(3))
    assert result["final_blue_green_center_error_m"] == pytest.approx(0.01)
    assert result["final_blue_green_height_error_m"] == pytest.approx(0.0)
    assert result["final_blue_upright_error_deg"] == pytest.approx(10.0)
    assert result["final_blue_green_rotation_error_deg"] == pytest.approx(10.0)


def test_cube_symmetry_ignores_quarter_turns():
    assert len(CUBE_SYMMETRIES) == 24
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = cube_pose_errors(np.array([0, 0, 0.06]), np.array([0, 0, 0.02]),
                              quarter_turn, np.eye(3))
    assert result["final_blue_green_rotation_error_deg"] == pytest.approx(0.0)


def test_quality_penalty_rejects_failed_rollout_and_rewards_clean_centered_stack():
    result = {
        "released_stack_success": True,
        "support_robot_normal_impulse_Ns": {"red": 1.0, "green": 0.0},
        "final_blue_green_center_error_m": 0.01,
        "final_blue_upright_error_deg": 0.0,
        "final_blue_green_rotation_error_deg": 20.0,
    }
    assert quality_penalty(result) == pytest.approx(6.0)
    result["support_robot_normal_impulse_Ns"]["red"] = 0.0
    result["final_blue_green_center_error_m"] = 0.0
    result["final_blue_green_rotation_error_deg"] = 0.0
    assert quality_penalty(result) == pytest.approx(0.0)
    result["released_stack_success"] = False
    assert quality_penalty(result) is None
