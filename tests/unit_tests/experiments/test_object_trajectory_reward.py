import numpy as np
import pytest

from experiments.object_trajectory_reward import ObjectTrajectoryReference


def test_target_uses_plan_timestamps_and_replay_time_scale():
    reference = ObjectTrajectoryReference(
        np.array([0, 1_000_000, 2_000_000]),
        np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]),
        reset_frame=0,
    )
    assert reference.duration_s == pytest.approx(4.0)
    np.testing.assert_allclose(reference.target(1.0), [0.05, 0, 0])
    np.testing.assert_allclose(reference.target(10.0), [0.2, 0, 0])


def test_corridor_allows_small_lag_but_not_stationary_holding():
    reference = ObjectTrajectoryReference(
        np.array([0, 1_000_000, 2_000_000]),
        np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]),
        reset_frame=0,
    )
    assert reference.outside_m(np.array([0, 0, 0]), 0.0) == 0.0
    assert reference.outside_m(np.array([0.075, 0.015, 0]), 2.0) == 0.0
    assert reference.outside_m(np.array([0, 0, 0]), 2.0) == pytest.approx(0.05)
    assert reference.outside_m(np.array([0, 0, 0]), 4.0) > 0.14


def test_reference_rejects_misaligned_or_nonmonotone_tracks():
    positions = np.zeros((3, 3))
    with pytest.raises(ValueError, match="trajectory"):
        ObjectTrajectoryReference(np.array([0, 1, 1]), positions, 0)
    with pytest.raises(ValueError, match="trajectory"):
        ObjectTrajectoryReference(np.array([0, 1, 2]), positions[:2], 0)
    with pytest.raises(ValueError, match="trajectory"):
        ObjectTrajectoryReference(np.array([0, 1, 2]), positions, 2)
