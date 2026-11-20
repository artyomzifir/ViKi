import numpy as np
import pytest

from viki.retarget.run import _velocity_limited_approach


def test_high_initial_curvature_uses_velocity_feasible_fallback():
    dt = 1.0 / 30.0
    trajectory = np.array([[1.0], [1.01], [1.04], [1.06]])
    approach = _velocity_limited_approach(
        np.zeros(1), trajectory, 2.0, dt, np.ones(1)
    )
    full = np.concatenate((approach, trajectory), axis=0)
    assert len(approach) >= 60
    assert np.max(np.abs(np.diff(full, axis=0) / dt)) <= 1.0 + 1e-9


def test_approach_rejects_overspeed_source_trajectory():
    trajectory = np.array([[0.0], [0.2], [0.4]])
    with pytest.raises(RuntimeError, match="trajectory itself exceeds"):
        _velocity_limited_approach(
            np.zeros(1), trajectory, 2.0, 1.0 / 30.0, np.ones(1)
        )


def test_approach_uses_solver_numeric_velocity_tolerance():
    dt = 1.0 / 30.0
    trajectory = np.arange(4, dtype=np.float64)[:, None] * dt * (1.0 + 5e-8)
    approach = _velocity_limited_approach(
        np.zeros(1), trajectory, 2.0, dt, np.ones(1)
    )
    full = np.concatenate((approach, trajectory), axis=0)
    assert np.max(np.abs(np.diff(full, axis=0) / dt) - 1.0) <= 1e-7
