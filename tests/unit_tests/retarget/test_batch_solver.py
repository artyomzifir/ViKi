"""Paper objective helpers and a small whole-trajectory solve."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viki.retarget.cost import difference_matrix, huber_irls_weights, huber_loss
from viki.retarget.solver import (
    BatchOptions,
    BatchWeights,
    _local_floor_support,
    solve_trajectory,
)


class LinearXY:
    nq = 2
    q_min = np.array([-2.0, -2.0])
    q_max = np.array([2.0, 2.0])
    velocity_limit = np.array([20.0, 20.0])
    q_reference = np.zeros(2)

    def pose(self, q):
        return np.array([q[0], q[1], 0.0]), np.eye(3)

    def integrate(self, q, dq):
        return np.asarray(q) + np.asarray(dq)

    def collision_linearization(self, q):
        return np.empty(0), np.empty((0, 2))


class RevoluteZ:
    nq = 1
    q_min = np.array([-np.pi])
    q_max = np.array([np.pi])
    velocity_limit = np.array([20.0])
    q_reference = np.zeros(1)

    def pose(self, q):
        return np.zeros(3), Rotation.from_rotvec([0.0, 0.0, q[0]]).as_matrix()

    def integrate(self, q, dq):
        return np.asarray(q) + np.asarray(dq)

    def collision_linearization(self, q):
        return np.empty(0), np.empty((0, 1))


class LinearZFloor:
    nq = 1
    q_min = np.array([-1.0])
    q_max = np.array([1.0])
    velocity_limit = np.array([20.0])
    q_reference = np.array([0.2])

    def pose(self, q):
        return np.array([0.0, 0.0, q[0]]), np.eye(3)

    def integrate(self, q, dq):
        return np.asarray(q) + np.asarray(dq)

    def collision_linearization(self, q):
        return np.empty(0), np.empty((0, 1))

    def floor_margins(self, q):
        return np.array([q[0]])

    def floor_linearization(self, q):
        return self.floor_margins(q), np.ones((1, 1))


def test_difference_operators_are_velocity_and_acceleration():
    x = np.array([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(difference_matrix(4, 1, 0.5) @ x, [1, 1, 1])
    np.testing.assert_allclose(difference_matrix(4, 2, 0.5) @ x, [0, 0])


def test_huber_is_quadratic_then_linear():
    np.testing.assert_allclose(huber_loss(np.array([0.5, 2.0]), 1.0), [0.125, 1.5])
    np.testing.assert_allclose(huber_irls_weights(np.array([0.5, 2.0]), 1.0), [1, np.sqrt(0.5)])


def test_floor_support_points_cover_mesh_and_common_primitives():
    vertices = np.array([[1.0, 0.0, -0.2], [-1.0, 0.0, 0.4], [0.0, 1.0, 0.1]])
    np.testing.assert_allclose(
        _local_floor_support(("vertices", vertices), np.array([0.0, 0.0, 1.0])),
        vertices[0],
    )
    np.testing.assert_allclose(
        _local_floor_support(("box", np.array([1.0, 2.0, 3.0])), np.ones(3)),
        [-1.0, -2.0, -3.0],
    )
    np.testing.assert_allclose(
        _local_floor_support(("cylinder", (0.3, 0.7)), np.array([0.0, 0.0, 1.0])),
        [0.0, 0.0, -0.7],
    )
    np.testing.assert_allclose(
        _local_floor_support(("cylinder", (0.3, 0.7)), np.array([1.0, 0.0, 0.0])),
        [-0.3, 0.0, 0.0],
    )
    direction = np.array([0.0, 0.6, 0.8])
    np.testing.assert_allclose(
        _local_floor_support(("capsule", (0.3, 0.7)), direction),
        -0.3 * direction + np.array([0.0, 0.0, -0.7]),
    )


def test_batch_solver_tracks_all_frames_in_one_problem():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    target = np.array([[0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [0.2, 0.1, 0.0]])
    result = solve_trajectory(
        LinearXY(), target, np.tile(np.eye(3), (3, 1, 1)), np.ones(3), 0.1,
        BatchWeights(position=1, orientation=0, velocity=1e-5, acceleration=0, posture=0),
        BatchOptions(max_iterations=4, max_step_rad=0.2, collision_enabled=False),
    )
    assert result.q.shape == (3, 2)
    np.testing.assert_allclose(result.achieved_position[:, :2], target[:, :2], atol=2e-3)
    assert result.position_error_m.max() < 0.003


def test_batch_solver_tracks_so3_orientation_term():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    angles = np.array([0.0, 0.2, 0.4])
    target_rotations = Rotation.from_rotvec(
        np.column_stack((np.zeros(3), np.zeros(3), angles))
    ).as_matrix()
    result = solve_trajectory(
        RevoluteZ(), np.zeros((3, 3)), target_rotations, np.ones(3), 0.1,
        BatchWeights(position=0, orientation=1, velocity=1e-6, acceleration=0, posture=0),
        BatchOptions(max_iterations=5, max_step_rad=0.2, collision_enabled=False),
    )
    np.testing.assert_allclose(result.q[:, 0], angles, atol=2e-3)
    assert np.rad2deg(result.orientation_error_rad.max()) < 0.2


def test_floor_is_a_hard_constraint_when_target_is_below_zero():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    result = solve_trajectory(
        LinearZFloor(), np.array([[0.0, 0.0, -0.3]]), np.eye(3)[None],
        np.ones(1), 0.1,
        BatchWeights(position=1, orientation=0, velocity=0, acceleration=0, posture=0),
        BatchOptions(max_iterations=5, max_step_rad=0.2, collision_enabled=False),
    )
    assert result.achieved_position[0, 2] >= -1e-9
    assert result.min_floor_margin >= -1e-9
    assert 0.3 <= result.position_error_m[0] < 0.31
