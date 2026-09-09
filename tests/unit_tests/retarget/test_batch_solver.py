"""Paper objective helpers and a small whole-trajectory solve."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viki.retarget.cost import (
    difference_matrix,
    huber_irls_weights,
    huber_loss,
    joint_difference_matrix,
)
from viki.retarget.solver import (
    BatchOptions,
    BatchWeights,
    _local_floor_support,
    solve_sequential_baseline,
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


class NonlinearCollision:
    nq = 1
    q_min = np.array([-1.0])
    q_max = np.array([1.0])
    velocity_limit = np.array([20.0])
    q_reference = np.array([0.4])

    def pose(self, q):
        return np.array([q[0], 0.0, 0.0]), np.eye(3)

    def integrate(self, q, dq):
        return np.asarray(q) + np.asarray(dq)

    def collision_margins(self, q):
        return np.array([0.25 - q[0] ** 2])

    def collision_linearization(self, q):
        return self.collision_margins(q), np.array([[-2.0 * q[0]]])


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
    totals = np.asarray([item["total"] for item in result.objective_history])
    assert np.all(np.diff(totals) <= 1e-10)


def test_batch_solver_enforces_inter_frame_velocity_limits():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    kinematics = LinearXY()
    kinematics.velocity_limit = np.array([0.25, 0.25])
    result = solve_trajectory(
        kinematics,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.tile(np.eye(3), (3, 1, 1)),
        np.ones(3),
        0.1,
        BatchWeights(
            position=1.0,
            orientation=0.0,
            velocity=0.0,
            acceleration=0.0,
            posture=0.0,
        ),
        BatchOptions(max_iterations=5, collision_enabled=False),
    )
    assert float(np.max(np.abs(np.diff(result.q, axis=0) / 0.1))) <= 0.25 + 1e-7


def test_reported_objective_uses_the_final_trajectory_consistently():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    target = np.array([[0.8, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dt = 0.1
    weights = BatchWeights(
        position=1.0,
        orientation=0.0,
        velocity=0.2,
        acceleration=0.0,
        posture=0.1,
    )
    options = BatchOptions(
        huber_delta=2.0,
        max_iterations=1,
        max_step_rad=0.2,
        collision_enabled=False,
    )
    result = solve_trajectory(
        LinearXY(),
        target,
        np.tile(np.eye(3), (2, 1, 1)),
        np.ones(2),
        dt,
        weights,
        options,
    )
    q_flat = result.q.reshape(-1)
    d1 = joint_difference_matrix(2, 2, 1, dt)
    expected_data = float(
        np.sum(huber_loss(result.position_error_m, options.huber_delta))
    )
    expected_velocity = float(weights.velocity * np.dot(d1 @ q_flat, d1 @ q_flat))
    expected_posture = float(weights.posture * np.dot(q_flat, q_flat))
    assert result.objective_terms == pytest.approx({
        "data": expected_data,
        "velocity": expected_velocity,
        "acceleration": 0.0,
        "posture": expected_posture,
        "total": expected_data + expected_velocity + expected_posture,
    })
    assert result.objective == pytest.approx(result.objective_terms["total"])
    assert result.objective_history[-1] == result.objective_terms


def test_zero_confidence_target_has_no_data_term_even_with_a_floor():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    target = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    result = solve_trajectory(
        LinearXY(),
        target,
        np.tile(np.eye(3), (3, 1, 1)),
        np.array([1.0, 0.0, 1.0]),
        0.1,
        BatchWeights(
            position=1.0,
            orientation=0.0,
            velocity=0.1,
            acceleration=0.0,
            posture=0.0,
        ),
        BatchOptions(
            huber_delta=2.0,
            confidence_floor=0.5,
            max_iterations=3,
            collision_enabled=False,
        ),
    )
    np.testing.assert_allclose(result.q, 0.0, atol=1e-9)
    assert result.objective_terms["data"] == pytest.approx(0.0)


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


def test_collision_is_rechecked_after_the_nonlinear_step():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    result = solve_trajectory(
        NonlinearCollision(),
        np.array([[1.0, 0.0, 0.0]]),
        np.eye(3)[None],
        np.ones(1),
        0.1,
        BatchWeights(position=1, orientation=0, velocity=0, acceleration=0, posture=0),
        BatchOptions(max_iterations=4, max_step_rad=0.2, collision_enabled=True),
    )
    assert result.q[0, 0] <= 0.5 + 1e-9
    assert result.min_collision_margin >= -1e-9


def test_sequential_baseline_reports_post_smoothing_constraint_violations():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    kinematics = LinearXY()
    kinematics.velocity_limit = np.array([0.5, 0.5])
    target = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    result = solve_sequential_baseline(
        kinematics,
        target,
        np.tile(np.eye(3), (3, 1, 1)),
        np.ones(3),
        0.1,
        BatchWeights(
            position=1.0,
            orientation=0.0,
            velocity=0.1,
            acceleration=0.1,
            posture=0.0,
        ),
        BatchOptions(
            max_iterations=6,
            max_step_rad=0.2,
            collision_enabled=False,
        ),
        smoothing_window=1,
    )
    assert result.q.shape == (3, 2)
    assert result.position_error_m.max() < 1e-3
    assert result.constraint_margins["velocity"] < 0.0
    assert result.objective_terms["velocity"] > 0.0
    assert result.objective_terms["total"] == pytest.approx(
        sum(
            result.objective_terms[key]
            for key in ("data", "velocity", "acceleration", "posture")
        )
    )


def test_sequential_baseline_holds_last_command_without_observation():
    pytest.importorskip("qpsolvers")
    if "osqp" not in pytest.importorskip("qpsolvers").available_solvers:
        pytest.skip("OSQP backend unavailable")
    result = solve_sequential_baseline(
        LinearXY(),
        np.array([[0.8, 0.0, 0.0], [-1.5, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        np.tile(np.eye(3), (3, 1, 1)),
        np.array([1.0, 0.0, 1.0]),
        0.1,
        BatchWeights(
            position=1.0,
            orientation=0.0,
            velocity=0.0,
            acceleration=0.0,
            posture=0.1,
        ),
        BatchOptions(max_iterations=6, collision_enabled=False),
        smoothing_window=1,
    )
    np.testing.assert_allclose(result.q[1], result.q[0])
