import numpy as np

from viki.retarget.run import _approach


def test_approach_removes_linear_join_acceleration_impulse():
    dt = 0.01
    seconds = 1.0
    reference = np.zeros(2)
    trajectory = np.tile(np.array([1.0, -0.5]), (3, 1))

    smooth = _approach(reference, trajectory, seconds, dt)
    count = len(smooth)
    linear_phase = np.arange(count, dtype=np.float64)[:, None] / count
    linear = reference[None, :] + linear_phase * (trajectory[0] - reference)[None, :]

    smooth_full = np.concatenate((smooth, trajectory), axis=0)
    linear_full = np.concatenate((linear, trajectory), axis=0)
    smooth_acceleration = np.diff(smooth_full, n=2, axis=0) / (dt * dt)
    linear_acceleration = np.diff(linear_full, n=2, axis=0) / (dt * dt)

    assert np.allclose(smooth[0], reference)
    assert np.max(np.abs(smooth_acceleration)) < 0.1 * np.max(
        np.abs(linear_acceleration)
    )
    assert np.max(np.abs(smooth_acceleration[count - 1])) < 0.1


def test_approach_matches_nonzero_endpoint_derivatives():
    dt = 0.002
    seconds = 1.0
    reference = np.array([0.0, 0.2])
    position = np.array([0.8, -0.4])
    velocity = np.array([0.15, -0.08])
    acceleration = np.array([0.04, 0.02])
    times = np.arange(3, dtype=np.float64)[:, None] * dt
    trajectory = position + times * velocity + 0.5 * times**2 * acceleration

    approach = _approach(reference, trajectory, seconds, dt)
    joined = np.concatenate((approach, trajectory), axis=0)
    endpoint = len(approach)
    estimated_velocity = (
        3.0 * joined[endpoint]
        - 4.0 * joined[endpoint - 1]
        + joined[endpoint - 2]
    ) / (2.0 * dt)
    estimated_acceleration = (
        2.0 * joined[endpoint]
        - 5.0 * joined[endpoint - 1]
        + 4.0 * joined[endpoint - 2]
        - joined[endpoint - 3]
    ) / (dt * dt)

    assert np.allclose(estimated_velocity, velocity, atol=2e-4)
    assert np.allclose(estimated_acceleration, acceleration, atol=2e-2)


def test_approach_can_be_disabled():
    trajectory = np.ones((2, 3))
    assert _approach(np.zeros(3), trajectory, 0.0, 0.01).shape == (0, 3)
