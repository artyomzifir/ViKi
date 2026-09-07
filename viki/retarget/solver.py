"""Whole-trajectory robust inverse kinematics.

Unlike the retired causal PINK loop, this module optimises ``q[0:T]`` in one
sparse QP at every Gauss--Newton iteration.  Data, velocity, acceleration and
posture terms therefore act on the same trajectory.  Joint/velocity limits and
linearised self-collision barriers are hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from viki.retarget.cost import (
    huber_irls_weights,
    huber_loss,
    joint_difference_matrix,
)


class Kinematics(Protocol):
    nq: int
    q_min: np.ndarray
    q_max: np.ndarray
    velocity_limit: np.ndarray
    q_reference: np.ndarray

    def pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def integrate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray: ...
    def collision_linearization(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class BatchWeights:
    position: float = 1.0
    orientation: float = 0.01
    velocity: float = 2e-3
    acceleration: float = 2e-5
    posture: float = 1e-4

    def validate(self) -> None:
        values = vars(self)
        if any(not np.isfinite(v) or v < 0.0 for v in values.values()):
            raise ValueError("retarget weights must be finite and non-negative")
        if self.position == 0.0 and self.orientation == 0.0:
            raise ValueError("position and orientation weights cannot both be zero")


@dataclass(frozen=True)
class BatchOptions:
    huber_delta: float = 0.05
    confidence_floor: float = 0.05
    lm_damping: float = 1e-5
    max_iterations: int = 40
    max_step_rad: float = 0.20
    convergence_rad: float = 1e-3
    qp_solver: str = "osqp"
    collision_enabled: bool = True

    def validate(self) -> None:
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")
        if self.lm_damping <= 0.0:
            raise ValueError("lm_damping must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.max_step_rad <= 0.0 or self.convergence_rad <= 0.0:
            raise ValueError("step and convergence limits must be positive")


@dataclass(frozen=True)
class BatchResult:
    q: np.ndarray
    achieved_position: np.ndarray
    achieved_rotation: np.ndarray
    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray
    iterations: int
    converged: bool
    objective: float
    min_collision_margin: float


def _pose_error(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    weights: BatchWeights,
) -> np.ndarray:
    pos = np.sqrt(weights.position) * (current_position - target_position)
    if weights.orientation == 0.0:
        ori = np.zeros(3, dtype=np.float64)
    else:
        ori = np.sqrt(weights.orientation) * Rotation.from_matrix(
            target_rotation.T @ current_rotation
        ).as_rotvec()
    return np.concatenate((pos, ori))


def _linearize_pose(
    kinematics: Kinematics,
    q: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    weights: BatchWeights,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position, rotation = kinematics.pose(q)
    error = _pose_error(position, rotation, target_position, target_rotation, weights)
    jacobian = np.empty((6, kinematics.nq), dtype=np.float64)
    for j in range(kinematics.nq):
        step = np.zeros(kinematics.nq, dtype=np.float64)
        step[j] = epsilon
        p1, r1 = kinematics.pose(kinematics.integrate(q, step))
        e1 = _pose_error(p1, r1, target_position, target_rotation, weights)
        jacobian[:, j] = (e1 - error) / epsilon
    return error, jacobian, position, rotation


def _block_diagonal_jacobian(blocks: list[np.ndarray]) -> sparse.csr_matrix:
    return sparse.block_diag(blocks, format="csr")


def _solve_qp(
    hessian: sparse.csc_matrix,
    gradient: np.ndarray,
    *,
    inequalities: sparse.csc_matrix | None,
    inequality_bound: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
    solver: str,
) -> np.ndarray:
    try:
        from qpsolvers import available_solvers, solve_qp
    except ImportError as exc:
        raise RuntimeError("retarget requires qpsolvers and a sparse QP backend") from exc
    if solver not in available_solvers:
        raise RuntimeError(
            f"QP solver {solver!r} is unavailable; installed backends: {available_solvers}"
        )
    answer = solve_qp(
        hessian,
        gradient,
        G=inequalities,
        h=inequality_bound,
        lb=lower,
        ub=upper,
        solver=solver,
        verbose=False,
    )
    if answer is None or not np.isfinite(answer).all():
        raise RuntimeError("whole-trajectory retarget QP is infeasible or failed")
    return np.asarray(answer, dtype=np.float64)


def solve_trajectory(
    kinematics: Kinematics,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
    confidence: np.ndarray,
    dt: float,
    weights: BatchWeights,
    options: BatchOptions,
    *,
    initial_q: np.ndarray | None = None,
    report: Callable[..., None] | None = None,
) -> BatchResult:
    """Optimise all robot configurations jointly with robust SE(3) tracking."""
    weights.validate()
    options.validate()
    positions = np.asarray(target_positions, dtype=np.float64)
    rotations = np.asarray(target_rotations, dtype=np.float64)
    omega = np.asarray(confidence, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"target positions must have shape (T, 3), got {positions.shape}")
    n_frames = len(positions)
    if n_frames == 0 or rotations.shape != (n_frames, 3, 3) or omega.shape != (n_frames,):
        raise ValueError("target position, rotation and confidence lengths must agree")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("retarget targets must be finite after interpolation")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")

    nq = int(kinematics.nq)
    if nq < 1 or np.asarray(kinematics.q_reference).shape != (nq,):
        raise ValueError("kinematics adapter has an invalid reference configuration")
    if initial_q is None:
        q = np.tile(np.asarray(kinematics.q_reference, dtype=np.float64), (n_frames, 1))
    else:
        q = np.asarray(initial_q, dtype=np.float64).copy()
        if q.shape != (n_frames, nq):
            raise ValueError(f"initial_q must have shape {(n_frames, nq)}, got {q.shape}")

    d1 = joint_difference_matrix(n_frames, nq, 1, dt)
    d2 = joint_difference_matrix(n_frames, nq, 2, dt)
    identity = sparse.eye(n_frames * nq, format="csr")
    q_ref = np.tile(np.asarray(kinematics.q_reference, dtype=np.float64), n_frames)
    q_min = np.tile(np.asarray(kinematics.q_min, dtype=np.float64), n_frames)
    q_max = np.tile(np.asarray(kinematics.q_max, dtype=np.float64), n_frames)
    velocity_limit = np.tile(
        np.asarray(kinematics.velocity_limit, dtype=np.float64), max(0, n_frames - 1)
    )
    omega = np.clip(np.nan_to_num(omega, nan=0.0), 0.0, 1.0)
    omega = np.maximum(omega, options.confidence_floor)

    converged = False
    objective = np.inf
    min_collision = np.inf
    achieved_p = np.empty((n_frames, 3), dtype=np.float64)
    achieved_r = np.empty((n_frames, 3, 3), dtype=np.float64)
    errors = np.empty((n_frames, 6), dtype=np.float64)

    for iteration in range(options.max_iterations):
        if report is not None:
            report(stage="retarget_linearize", frame=iteration + 1, total=options.max_iterations)
        jacobians: list[np.ndarray] = []
        for t in range(n_frames):
            err, jac, pos, rot = _linearize_pose(
                kinematics, q[t], positions[t], rotations[t], weights
            )
            errors[t] = err
            jacobians.append(jac)
            achieved_p[t] = pos
            achieved_r[t] = rot

        robust = huber_irls_weights(np.linalg.norm(errors, axis=1), options.huber_delta)
        data_scale = np.repeat(np.sqrt(omega) * robust, 6)
        data_jac = sparse.diags(data_scale) @ _block_diagonal_jacobian(jacobians)
        data_rhs = data_scale * errors.reshape(-1)

        q_flat = q.reshape(-1)
        matrices = [data_jac]
        rhs = [data_rhs]
        if weights.velocity > 0.0 and d1.shape[0]:
            scale = np.sqrt(weights.velocity)
            matrices.append(scale * d1)
            rhs.append(scale * np.asarray(d1 @ q_flat).reshape(-1))
        if weights.acceleration > 0.0 and d2.shape[0]:
            scale = np.sqrt(weights.acceleration)
            matrices.append(scale * d2)
            rhs.append(scale * np.asarray(d2 @ q_flat).reshape(-1))
        if weights.posture > 0.0:
            scale = np.sqrt(weights.posture)
            matrices.append(scale * identity)
            rhs.append(scale * (q_flat - q_ref))
        matrices.append(np.sqrt(options.lm_damping) * identity)
        rhs.append(np.zeros(n_frames * nq, dtype=np.float64))
        design = sparse.vstack(matrices, format="csr")
        residual = np.concatenate(rhs)
        hessian = (design.T @ design).tocsc()
        gradient = np.asarray(design.T @ residual).reshape(-1)

        lower = np.maximum(q_min - q_flat, -options.max_step_rad)
        upper = np.minimum(q_max - q_flat, options.max_step_rad)
        g_parts: list[sparse.spmatrix] = []
        h_parts: list[np.ndarray] = []
        if d1.shape[0]:
            current_velocity = np.asarray(d1 @ q_flat).reshape(-1)
            g_parts.extend((d1, -d1))
            h_parts.extend((velocity_limit - current_velocity, velocity_limit + current_velocity))

        min_collision = np.inf
        if options.collision_enabled:
            rows: list[sparse.spmatrix] = []
            bounds: list[np.ndarray] = []
            for t in range(n_frames):
                margin, jac = kinematics.collision_linearization(q[t])
                margin = np.asarray(margin, dtype=np.float64).reshape(-1)
                jac = np.asarray(jac, dtype=np.float64)
                if jac.shape != (len(margin), nq):
                    raise RuntimeError("collision barrier returned an invalid Jacobian")
                if len(margin):
                    min_collision = min(min_collision, float(np.min(margin)))
                    left = sparse.csr_matrix((len(margin), t * nq))
                    right = sparse.csr_matrix((len(margin), (n_frames - t - 1) * nq))
                    rows.append(sparse.hstack((left, -sparse.csr_matrix(jac), right)))
                    bounds.append(margin)
            if rows:
                g_parts.append(sparse.vstack(rows, format="csr"))
                h_parts.append(np.concatenate(bounds))

        inequalities = sparse.vstack(g_parts, format="csc") if g_parts else None
        inequality_bound = np.concatenate(h_parts) if h_parts else None
        delta = _solve_qp(
            hessian,
            gradient,
            inequalities=inequalities,
            inequality_bound=inequality_bound,
            lower=lower,
            upper=upper,
            solver=options.qp_solver,
        ).reshape(n_frames, nq)
        for t in range(n_frames):
            q[t] = kinematics.integrate(q[t], delta[t])
        max_step = float(np.max(np.abs(delta)))
        objective = float(
            np.sum(omega * huber_loss(np.linalg.norm(errors, axis=1), options.huber_delta))
            + weights.velocity * np.dot(d1 @ q_flat, d1 @ q_flat)
            + weights.acceleration * np.dot(d2 @ q_flat, d2 @ q_flat)
            + weights.posture * np.dot(q_flat - q_ref, q_flat - q_ref)
        )
        if report is not None:
            report(
                stage="retarget_solve", frame=iteration + 1,
                total=options.max_iterations, max_step=max_step, objective=objective,
            )
        if max_step < options.convergence_rad:
            converged = True
            break

    for t in range(n_frames):
        achieved_p[t], achieved_r[t] = kinematics.pose(q[t])
    position_error = np.linalg.norm(achieved_p - positions, axis=1)
    orientation_error = np.array([
        np.linalg.norm(Rotation.from_matrix(rotations[t].T @ achieved_r[t]).as_rotvec())
        for t in range(n_frames)
    ])
    return BatchResult(
        q=q,
        achieved_position=achieved_p,
        achieved_rotation=achieved_r,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        iterations=iteration + 1,
        converged=converged,
        objective=objective,
        min_collision_margin=min_collision,
    )


class PinocchioKinematics:
    """Fixed-base adapter exposing one physical angle per actuated joint.

    Pinocchio represents unbounded revolute joints as ``[cos(q), sin(q)]`` so
    UR models have ``model.nq != model.nv``.  The batch problem and robot
    command archive must contain six actual angles, not twelve embedding
    coordinates.  We therefore optimise a tangent chart around neutral and
    convert it to Pinocchio configuration vectors only for FK/collision.
    """

    def __init__(
        self,
        robot: object,
        ee_frame: str,
        *,
        collision_pairs: int,
        collision_min_distance_m: float,
    ) -> None:
        try:
            import pinocchio as pin
            from pink import Configuration
            from pink.barriers import SelfCollisionBarrier
        except ImportError as exc:
            raise RuntimeError("retarget requires robotics Pinocchio and pin-pink") from exc
        self.pin = pin
        self.Configuration = Configuration
        self.robot = robot
        self.model = robot.model
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(ee_frame)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"robot model has no end-effector frame {ee_frame!r}")
        self.configuration_reference = np.asarray(pin.neutral(self.model), dtype=np.float64)
        self.nq = int(self.model.nv)
        self.q_reference = np.zeros(self.nq, dtype=np.float64)
        self.q_min = np.full(self.nq, -np.inf, dtype=np.float64)
        self.q_max = np.full(self.nq, np.inf, dtype=np.float64)
        for joint_id in range(1, self.model.njoints):
            joint_nq = int(self.model.nqs[joint_id])
            joint_nv = int(self.model.nvs[joint_id])
            if joint_nv == 0:
                continue
            if joint_nv != 1 or joint_nq not in (1, 2):
                raise ValueError(
                    "PoC batch solver supports fixed-base chains with 1-DoF joints"
                )
            v_index = int(self.model.idx_vs[joint_id])
            if joint_nq == 1:
                q_index = int(self.model.idx_qs[joint_id])
                self.q_min[v_index] = float(self.model.lowerPositionLimit[q_index])
                self.q_max[v_index] = float(self.model.upperPositionLimit[q_index])
        self.q_min[~np.isfinite(self.q_min) | (self.q_min < -1e10)] = -np.inf
        self.q_max[~np.isfinite(self.q_max) | (self.q_max > 1e10)] = np.inf
        self.velocity_limit = np.asarray(self.model.velocityLimit, dtype=np.float64).copy()
        self.velocity_limit[~np.isfinite(self.velocity_limit) | (self.velocity_limit <= 0)] = np.inf

        self.collision_model = getattr(robot, "collision_model", None)
        self.collision_data = None
        self.collision_barrier = None
        if collision_pairs > 0:
            if self.collision_model is None:
                raise RuntimeError("collision constraints requested but URDF has no collision model")
            if len(self.collision_model.collisionPairs) == 0:
                self.collision_model.addAllCollisionPairs()
            # Evaluate the original pair list once at neutral, then rebuild it
            # instead of mutating Pinocchio's C++ vector while iterating Python
            # wrappers into that vector (those wrappers can be invalidated by a
            # removal). Apply this to generated and vendor/SRDF-provided lists.
            collision_data = self.collision_model.createData()
            neutral_data = self.model.createData()
            self.pin.forwardKinematics(
                self.model, neutral_data, self.configuration_reference
            )
            self.pin.updateGeometryPlacements(
                self.model,
                neutral_data,
                self.collision_model,
                collision_data,
                self.configuration_reference,
            )
            self.pin.computeDistances(
                self.model,
                neutral_data,
                self.collision_model,
                collision_data,
                self.configuration_reference,
            )
            keep_pairs = []
            for pair, distance in zip(
                list(self.collision_model.collisionPairs),
                list(collision_data.distanceResults),
            ):
                first = int(self.collision_model.geometryObjects[pair.first].parentJoint)
                second = int(self.collision_model.geometryObjects[pair.second].parentJoint)
                adjacent = (
                    first == second
                    or int(self.model.parents[first]) == second
                    or int(self.model.parents[second]) == first
                )
                nominal_contact = (
                    float(distance.min_distance)
                    <= float(collision_min_distance_m) + 1e-6
                )
                if not adjacent and not nominal_contact:
                    keep_pairs.append(self.pin.CollisionPair(pair.first, pair.second))
            self.collision_model.removeAllCollisionPairs()
            for pair in keep_pairs:
                self.collision_model.addCollisionPair(pair)
            total = len(self.collision_model.collisionPairs)
            if total == 0:
                raise RuntimeError("collision constraints requested but URDF has no collision pairs")
            self.collision_data = self.collision_model.createData()
            self.collision_barrier = SelfCollisionBarrier(
                min(int(collision_pairs), total), d_min=float(collision_min_distance_m)
            )

    def pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        configuration = self._configuration(q)
        self.pin.forwardKinematics(self.model, self.data, configuration)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame_id]
        return np.asarray(pose.translation).copy(), np.asarray(pose.rotation).copy()

    def integrate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float64) + np.asarray(dq, dtype=np.float64)

    def _configuration(self, q: np.ndarray) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (self.nq,):
            raise ValueError(f"expected {self.nq} joint angles, got {values.shape}")
        return np.asarray(
            self.pin.integrate(self.model, self.configuration_reference, values),
            dtype=np.float64,
        )

    def collision_linearization(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.collision_barrier is None:
            return np.empty(0), np.empty((0, self.nq))
        configuration = self.Configuration(
            self.model,
            self.model.createData(),
            self._configuration(q),
            collision_model=self.collision_model,
            collision_data=self.collision_model.createData(),
        )
        return (
            np.asarray(self.collision_barrier.compute_barrier(configuration)),
            np.asarray(self.collision_barrier.compute_jacobian(configuration)),
        )

    def joint_points(self, q: np.ndarray) -> np.ndarray:
        """URDF joint origins, including the robot-base origin at index zero."""
        self.pin.forwardKinematics(self.model, self.data, self._configuration(q))
        points = np.zeros((self.model.njoints, 3), dtype=np.float64)
        for joint_id in range(1, self.model.njoints):
            points[joint_id] = np.asarray(self.data.oMi[joint_id].translation)
        return points

    @property
    def link_edges(self) -> np.ndarray:
        return np.asarray(
            [[int(self.model.parents[j]), j] for j in range(1, self.model.njoints)],
            dtype=np.int32,
        )

    @property
    def joint_names(self) -> list[str]:
        return [str(name) for name in self.model.names]
