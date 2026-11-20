import numpy as np
import pytest

from viki.retarget.solver import _repair_velocity_roundoff


def test_tiny_qp_velocity_residual_is_projected_inside_hard_limit():
    trajectory = np.array([[0.0], [0.100004927], [0.200004927]])
    repaired, correction = _repair_velocity_roundoff(
        trajectory, np.array([-1.0]), np.array([1.0]), np.array([1.0]), 0.1
    )
    assert 0.0 < correction < 1e-3
    assert np.max(np.abs(np.diff(repaired, axis=0) / 0.1)) <= 1.0 + 1e-7
    assert np.min(repaired) >= -1.0
    assert np.max(repaired) <= 1.0


def test_velocity_feasible_solution_is_unchanged():
    trajectory = np.array([[0.0], [0.1], [0.2]])
    repaired, correction = _repair_velocity_roundoff(
        trajectory, np.array([-1.0]), np.array([1.0]), np.array([1.0]), 0.1
    )
    np.testing.assert_array_equal(repaired, trajectory)
    assert correction == 0.0


def test_non_roundoff_violation_is_not_hidden():
    trajectory = np.array([[0.0], [0.101]])
    with pytest.raises(RuntimeError, match="exceeds joint velocity limits"):
        _repair_velocity_roundoff(
            trajectory, np.array([-1.0]), np.array([1.0]), np.array([1.0]), 0.1
        )
