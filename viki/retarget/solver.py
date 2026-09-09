"""Whole-trajectory robust inverse kinematics.

Unlike the retired causal PINK loop, this module optimises ``q[0:T]`` in one
sparse QP at every Gauss--Newton iteration.  Data, velocity, acceleration and
posture terms therefore act on the same trajectory. Joint/velocity limits,
the calibration-floor half-space and linearised self-collision barriers are
hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
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
    def pose_jacobian(
        self, q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def integrate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray: ...
    def collision_linearization(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def collision_margins(self, q: np.ndarray) -> np.ndarray: ...
    def floor_linearization(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def _floor_support_descriptor(shape: object) -> tuple[str, object]:
    """Cache enough local geometry to query its support point cheaply.

    Meshes dominate the current URDFs.  Their lowest point in any direction is
    exactly one of their vertices, so keeping the vertex array avoids crossing
    the Python/C++ boundary for every frame and Gauss--Newton iteration.
    """
    vertices_attr = getattr(shape, "vertices", None)
    if vertices_attr is not None:
        vertices = vertices_attr() if callable(vertices_attr) else vertices_attr
        values = np.asarray(vertices, dtype=np.float64)
        if values.ndim == 2 and values.shape[1] == 3 and len(values):
            return "vertices", values.copy()

    kind = type(shape).__name__.lower()
    if kind == "sphere":
        return "sphere", float(shape.radius)
    if kind == "box":
        return "box", np.asarray(shape.halfSide, dtype=np.float64).copy()
    if kind == "cylinder":
        return "cylinder", (float(shape.radius), float(shape.halfLength))
    if kind == "capsule":
        return "capsule", (float(shape.radius), float(shape.halfLength))

    # Unknown FCL primitives still get a safe, conservative constraint.  The
    # local AABB contains the complete shape, so keeping its support above the
    # plane also keeps the shape above it (at the cost of some extra clearance).
    compute_aabb = getattr(shape, "computeLocalAABB", None)
    if callable(compute_aabb):
        compute_aabb()
    aabb = getattr(shape, "aabb_local", None)
    if aabb is None:
        raise TypeError(f"unsupported floor collision geometry {type(shape).__name__}")
    lower = np.asarray(aabb.min_, dtype=np.float64)
    upper = np.asarray(aabb.max_, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise TypeError(f"invalid local AABB for {type(shape).__name__}")
    return "aabb", (lower.copy(), upper.copy())


def _local_floor_support(
    descriptor: tuple[str, object], direction: np.ndarray
) -> np.ndarray:
    """Point minimising ``direction @ point`` in geometry-local coordinates."""
    kind, payload = descriptor
    axis = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.isfinite(axis).all() or norm <= 1e-12:
        raise ValueError("floor support direction must be a finite non-zero XYZ vector")
    axis = axis / norm

    if kind == "vertices":
        vertices = np.asarray(payload, dtype=np.float64)
        return vertices[int(np.argmin(vertices @ axis))].copy()
    if kind == "sphere":
        return -float(payload) * axis
    if kind == "box":
        half_side = np.asarray(payload, dtype=np.float64)
        return np.where(axis > 0.0, -half_side, np.where(axis < 0.0, half_side, 0.0))
    if kind in {"cylinder", "capsule"}:
        radius, half_length = payload
        if kind == "capsule":
            point = -float(radius) * axis
        else:
            point = np.zeros(3, dtype=np.float64)
            radial_norm = float(np.linalg.norm(axis[:2]))
            if radial_norm > 1e-12:
                point[:2] = -float(radius) * axis[:2] / radial_norm
        if axis[2] > 0.0:
            point[2] -= float(half_length)
        elif axis[2] < 0.0:
            point[2] += float(half_length)
        return point
    if kind == "aabb":
        lower, upper = payload
        return np.where(axis > 0.0, lower, np.where(axis < 0.0, upper, 0.5 * (lower + upper)))
    raise RuntimeError(f"unknown floor support descriptor {kind!r}")


def _set_frame(kinematics: Kinematics, frame: int) -> None:
    """Select known per-frame passive joints when the adapter provides them."""
    setter = getattr(kinematics, "set_frame", None)
    if setter is not None:
        setter(frame)


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
    objective_terms: dict[str, float]
    objective_history: tuple[dict[str, float], ...]
    min_collision_margin: float
    min_floor_margin: float


@dataclass(frozen=True)
class SequentialBaselineResult:
    """Frame-wise IK followed by the conventional post-hoc SG smoothing."""

    q: np.ndarray
    achieved_position: np.ndarray
    achieved_rotation: np.ndarray
    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray
    objective_terms: dict[str, float]
    constraint_margins: dict[str, float]
    total_iterations: int
    solved_frames: int
    held_frames: int


def _objective_terms(
    kinematics: Kinematics,
    q: np.ndarray,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
    omega: np.ndarray,
    weights: BatchWeights,
    options: BatchOptions,
    d1: sparse.csr_matrix,
    d2: sparse.csr_matrix,
    q_ref: np.ndarray,
) -> dict[str, float]:
    """Evaluate every energy term at one internally consistent trajectory."""
    errors = np.empty((len(q), 6), dtype=np.float64)
    for frame, configuration in enumerate(q):
        _set_frame(kinematics, frame)
        position, rotation = kinematics.pose(configuration)
        errors[frame] = _pose_error(
            position,
            rotation,
            target_positions[frame],
            target_rotations[frame],
            weights,
        )
    q_flat = np.asarray(q, dtype=np.float64).reshape(-1)
    terms = {
        "data": float(
            np.sum(
                omega
                * huber_loss(np.linalg.norm(errors, axis=1), options.huber_delta)
            )
        ),
        "velocity": float(weights.velocity * np.dot(d1 @ q_flat, d1 @ q_flat)),
        "acceleration": float(
            weights.acceleration * np.dot(d2 @ q_flat, d2 @ q_flat)
        ),
        "posture": float(
            weights.posture * np.dot(q_flat - q_ref, q_flat - q_ref)
        ),
    }
    terms["total"] = float(sum(terms.values()))
    return terms


def _floor_linearization(
    kinematics: Kinematics, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return optional hard floor margins without burdening simple adapters."""
    linearize = getattr(kinematics, "floor_linearization", None)
    if linearize is None:
        return np.empty(0), np.empty((0, int(kinematics.nq)))
    margin, jacobian = linearize(q)
    return (
        np.asarray(margin, dtype=np.float64).reshape(-1),
        np.asarray(jacobian, dtype=np.float64),
    )


def _minimum_floor_margin(kinematics: Kinematics, q: np.ndarray) -> float:
    """Evaluate the exact (non-linear) floor margin for one configuration."""
    margins = getattr(kinematics, "floor_margins", None)
    values = margins(q) if margins is not None else _floor_linearization(kinematics, q)[0]
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.min(values)) if len(values) else np.inf


def _trajectory_min_floor_margin(
    kinematics: Kinematics,
    trajectory: np.ndarray,
    frame_indices: np.ndarray | None = None,
) -> float:
    minimum = np.inf
    for frame, configuration in enumerate(trajectory):
        passive_frame = frame if frame_indices is None else int(frame_indices[frame])
        _set_frame(kinematics, passive_frame)
        minimum = min(minimum, _minimum_floor_margin(kinematics, configuration))
    return minimum


def _collision_linearization(
    kinematics: Kinematics, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    linearize = getattr(kinematics, "collision_linearization", None)
    if linearize is None:
        return np.empty(0), np.empty((0, int(kinematics.nq)))
    margin, jacobian = linearize(q)
    return (
        np.asarray(margin, dtype=np.float64).reshape(-1),
        np.asarray(jacobian, dtype=np.float64),
    )


def _minimum_collision_margin(kinematics: Kinematics, q: np.ndarray) -> float:
    margins = getattr(kinematics, "collision_margins", None)
    values = margins(q) if margins is not None else _collision_linearization(kinematics, q)[0]
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.min(values)) if len(values) else np.inf


def _trajectory_min_collision_margin(
    kinematics: Kinematics,
    trajectory: np.ndarray,
    frame_indices: np.ndarray | None = None,
) -> float:
    minimum = np.inf
    for frame, configuration in enumerate(trajectory):
        passive_frame = frame if frame_indices is None else int(frame_indices[frame])
        _set_frame(kinematics, passive_frame)
        minimum = min(minimum, _minimum_collision_margin(kinematics, configuration))
    return minimum


def trajectory_constraint_margins(
    kinematics: Kinematics,
    trajectory: np.ndarray,
    dt: float,
    *,
    frame_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate exact joint, velocity, floor and collision margins."""
    q = np.asarray(trajectory, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != int(kinematics.nq) or len(q) == 0:
        raise ValueError("trajectory must be a non-empty (T, nq) array")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    if frame_indices is not None and np.asarray(frame_indices).shape != (len(q),):
        raise ValueError("frame_indices must have one entry per trajectory frame")
    q_min = np.asarray(kinematics.q_min, dtype=np.float64)
    q_max = np.asarray(kinematics.q_max, dtype=np.float64)
    joint_margin = float(np.min(np.minimum(q - q_min, q_max - q)))
    if len(q) > 1:
        velocity_margin = float(np.min(
            np.asarray(kinematics.velocity_limit, dtype=np.float64)
            - np.abs(np.diff(q, axis=0) / dt)
        ))
    else:
        velocity_margin = np.inf
    return {
        "joint": joint_margin,
        "velocity": velocity_margin,
        "floor": _trajectory_min_floor_margin(kinematics, q, frame_indices),
        "collision": _trajectory_min_collision_margin(
            kinematics, q, frame_indices
        ),
    }


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


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def _so3_left_jacobian_inverse(rotvec: np.ndarray) -> np.ndarray:
    """Map a left SO(3) perturbation to a log-coordinate perturbation."""
    phi = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(phi))
    cross = _skew(phi)
    if theta < 1e-5:
        coefficient = 1.0 / 12.0 + theta * theta / 720.0
    else:
        coefficient = (
            1.0 - 0.5 * theta / np.tan(0.5 * theta)
        ) / (theta * theta)
    return np.eye(3) - 0.5 * cross + coefficient * (cross @ cross)


def _linearize_pose(
    kinematics: Kinematics,
    q: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    weights: BatchWeights,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    analytical = getattr(kinematics, "pose_jacobian", None)
    if analytical is not None:
        position, rotation, position_jacobian, angular_jacobian = analytical(q)
        position = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        position_jacobian = np.asarray(position_jacobian, dtype=np.float64)
        angular_jacobian = np.asarray(angular_jacobian, dtype=np.float64)
        expected = (3, int(kinematics.nq))
        if position_jacobian.shape != expected or angular_jacobian.shape != expected:
            raise RuntimeError("pose adapter returned an invalid analytical Jacobian")
        error = _pose_error(
            position, rotation, target_position, target_rotation, weights
        )
        jacobian = np.empty((6, kinematics.nq), dtype=np.float64)
        jacobian[:3] = np.sqrt(weights.position) * position_jacobian
        if weights.orientation == 0.0:
            jacobian[3:] = 0.0
        else:
            orientation_error = error[3:] / np.sqrt(weights.orientation)
            # LOCAL_WORLD_ALIGNED supplies angular velocity in world axes.
            # Express its left perturbation in target axes before applying the
            # differential of Log(R_target.T @ R_current).
            jacobian[3:] = (
                np.sqrt(weights.orientation)
                * _so3_left_jacobian_inverse(orientation_error)
                @ target_rotation.T
                @ angular_jacobian
            )
        return error, jacobian, position, rotation

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
    # Missing/invalid frames are deliberately absent from the data term.  The
    # confidence floor only prevents a *real but weak* observation from becoming
    # numerically irrelevant; it must not turn an interpolated target into data.
    omega = np.where(
        omega > 0.0,
        np.maximum(omega, options.confidence_floor),
        0.0,
    )

    converged = False
    objective = np.inf
    min_collision = np.inf
    min_floor = _trajectory_min_floor_margin(kinematics, q)
    objective_terms = _objective_terms(
        kinematics,
        q,
        positions,
        rotations,
        omega,
        weights,
        options,
        d1,
        d2,
        q_ref,
    )
    objective = objective_terms["total"]
    objective_history: list[dict[str, float]] = []
    achieved_p = np.empty((n_frames, 3), dtype=np.float64)
    achieved_r = np.empty((n_frames, 3, 3), dtype=np.float64)
    errors = np.empty((n_frames, 6), dtype=np.float64)

    for iteration in range(options.max_iterations):
        if report is not None:
            report(stage="retarget_linearize", frame=iteration + 1, total=options.max_iterations)
        jacobians: list[np.ndarray] = []
        for t in range(n_frames):
            _set_frame(kinematics, t)
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

        # A copy is intentional. ``q`` is updated in-place below; retaining a
        # view here used to mix old pose residuals with new regularisation terms
        # in the reported objective.
        q_flat = q.reshape(-1).copy()
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
        collision_rows_present = False
        if options.collision_enabled:
            rows: list[sparse.spmatrix] = []
            bounds: list[np.ndarray] = []
            for t in range(n_frames):
                _set_frame(kinematics, t)
                margin, jac = _collision_linearization(kinematics, q[t])
                if jac.shape != (len(margin), nq):
                    raise RuntimeError("collision barrier returned an invalid Jacobian")
                if len(margin):
                    min_collision = min(min_collision, float(np.min(margin)))
                    left = sparse.csr_matrix((len(margin), t * nq))
                    right = sparse.csr_matrix((len(margin), (n_frames - t - 1) * nq))
                    rows.append(sparse.hstack((left, -sparse.csr_matrix(jac), right)))
                    bounds.append(margin)
            if rows:
                collision_rows_present = True
                g_parts.append(sparse.vstack(rows, format="csr"))
                h_parts.append(np.concatenate(bounds))

        floor_rows: list[sparse.spmatrix] = []
        floor_bounds: list[np.ndarray] = []
        min_floor = np.inf
        for t in range(n_frames):
            _set_frame(kinematics, t)
            margin, jac = _floor_linearization(kinematics, q[t])
            if jac.shape != (len(margin), nq):
                raise RuntimeError("floor barrier returned an invalid Jacobian")
            if len(margin):
                min_floor = min(min_floor, float(np.min(margin)))
                left = sparse.csr_matrix((len(margin), t * nq))
                right = sparse.csr_matrix((len(margin), (n_frames - t - 1) * nq))
                floor_rows.append(
                    sparse.hstack((left, -sparse.csr_matrix(jac), right))
                )
                floor_bounds.append(margin)
        if floor_rows:
            # margin(q + dq) ~= margin(q) + J dq >= 0
            g_parts.append(sparse.vstack(floor_rows, format="csr"))
            h_parts.append(np.concatenate(floor_bounds))

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

        # The pose and both barriers are non-linear. Shorten the global step
        # until the exact robust objective does not increase and already
        # feasible geometric invariants remain feasible. A shared scale also
        # preserves the convex inter-frame velocity constraints. An initially
        # infeasible barrier is left to the QP to repair; once feasible, it can
        # never regress below zero again.
        current_floor = (
            _trajectory_min_floor_margin(kinematics, q) if floor_rows else np.inf
        )
        current_collision = (
            _trajectory_min_collision_margin(kinematics, q)
            if collision_rows_present else np.inf
        )
        preserve_floor = bool(floor_rows) and current_floor >= -1e-9
        preserve_collision = collision_rows_present and current_collision >= -1e-9
        preserve_objective = (
            (not floor_rows or current_floor >= -1e-9)
            and (not collision_rows_present or current_collision >= -1e-9)
        )
        accepted_terms: dict[str, float] | None = None
        if preserve_floor or preserve_collision or preserve_objective:
            scale = 1.0
            feasible_step_found = False
            while scale > 1e-6:
                candidate = np.stack([
                    kinematics.integrate(q[t], scale * delta[t])
                    for t in range(n_frames)
                ])
                floor_ok = (
                    not preserve_floor
                    or _trajectory_min_floor_margin(kinematics, candidate) >= -1e-9
                )
                collision_ok = (
                    not preserve_collision
                    or _trajectory_min_collision_margin(kinematics, candidate) >= -1e-9
                )
                candidate_terms = _objective_terms(
                    kinematics,
                    candidate,
                    positions,
                    rotations,
                    omega,
                    weights,
                    options,
                    d1,
                    d2,
                    q_ref,
                )
                objective_ok = (
                    not preserve_objective
                    or candidate_terms["total"]
                    <= objective + max(1e-12, 1e-10 * abs(objective))
                )
                if floor_ok and collision_ok and objective_ok:
                    feasible_step_found = True
                    accepted_terms = candidate_terms
                    break
                scale *= 0.5
            if not feasible_step_found:
                scale = 0.0
            delta *= scale
        for t in range(n_frames):
            q[t] = kinematics.integrate(q[t], delta[t])
        max_step = float(np.max(np.abs(delta)))
        objective_terms = accepted_terms or _objective_terms(
            kinematics, q, positions, rotations, omega, weights, options,
            d1, d2, q_ref,
        )
        objective = objective_terms["total"]
        objective_history.append(objective_terms.copy())
        if report is not None:
            report(
                stage="retarget_solve", frame=iteration + 1,
                total=options.max_iterations, max_step=max_step, objective=objective,
            )
        if max_step < options.convergence_rad:
            converged = True
            break

    for t in range(n_frames):
        _set_frame(kinematics, t)
        achieved_p[t], achieved_r[t] = kinematics.pose(q[t])
    position_error = np.linalg.norm(achieved_p - positions, axis=1)
    orientation_error = np.array([
        np.linalg.norm(Rotation.from_matrix(rotations[t].T @ achieved_r[t]).as_rotvec())
        for t in range(n_frames)
    ])
    min_floor = _trajectory_min_floor_margin(kinematics, q)
    if min_floor < -1e-7:
        raise RuntimeError(
            "retarget could not find a floor-feasible trajectory; remaining "
            f"penetration is {-min_floor * 1000.0:.1f} mm"
        )
    min_collision = (
        _trajectory_min_collision_margin(kinematics, q)
        if options.collision_enabled else np.inf
    )
    if min_collision < -1e-7:
        raise RuntimeError(
            "retarget could not find a collision-feasible trajectory; remaining "
            f"barrier violation is {-min_collision * 1000.0:.1f} mm"
        )
    return BatchResult(
        q=q,
        achieved_position=achieved_p,
        achieved_rotation=achieved_r,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        iterations=iteration + 1,
        converged=converged,
        objective=objective,
        objective_terms=objective_terms,
        objective_history=tuple(objective_history),
        min_collision_margin=min_collision,
        min_floor_margin=min_floor,
    )


class _FixedPassiveFrameKinematics:
    """Present one gripper state as a one-frame kinematics problem."""

    def __init__(self, base: Kinematics, frame: int):
        self._base = base
        self._frame = int(frame)

    def set_frame(self, _frame: int) -> None:
        _set_frame(self._base, self._frame)

    def __getattr__(self, name: str):
        self.set_frame(0)
        return getattr(self._base, name)


def solve_sequential_baseline(
    kinematics: Kinematics,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
    confidence: np.ndarray,
    dt: float,
    weights: BatchWeights,
    options: BatchOptions,
    *,
    smoothing_window: int = 7,
    smoothing_polyorder: int = 2,
    report: Callable[..., None] | None = None,
) -> SequentialBaselineResult:
    """Solve independent causal IK frames, then smooth the joint sequence.

    This is deliberately a comparison baseline, not an executable alternative:
    post-hoc smoothing can reintroduce velocity, floor or collision violations.
    Those exact margins are therefore measured and archived instead of hidden.
    """
    positions = np.asarray(target_positions, dtype=np.float64)
    rotations = np.asarray(target_rotations, dtype=np.float64)
    omega = np.asarray(confidence, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("target positions must have shape (T, 3)")
    n_frames = len(positions)
    if n_frames == 0 or rotations.shape != (n_frames, 3, 3) or omega.shape != (n_frames,):
        raise ValueError("target position, rotation and confidence lengths must agree")

    q = np.empty((n_frames, int(kinematics.nq)), dtype=np.float64)
    previous = np.asarray(kinematics.q_reference, dtype=np.float64).copy()
    total_iterations = 0
    solved_frames = 0
    for frame in range(n_frames):
        if report is not None:
            report(stage="retarget_sequential_baseline", frame=frame + 1, total=n_frames)
        # A causal per-frame method has no observation and no temporal problem
        # to solve on a missing frame. Holding the last command is the standard
        # neutral policy; letting the posture prior walk toward q_reference
        # would manufacture motion and make the comparison artificially bad.
        if not np.isfinite(omega[frame]) or omega[frame] <= 0.0:
            q[frame] = previous
            continue
        fixed = _FixedPassiveFrameKinematics(kinematics, frame)
        result = solve_trajectory(
            fixed,
            positions[frame : frame + 1],
            rotations[frame : frame + 1],
            omega[frame : frame + 1],
            dt,
            weights,
            options,
            initial_q=previous[None],
        )
        previous = result.q[0].copy()
        q[frame] = previous
        total_iterations += result.iterations
        solved_frames += 1

    window = min(int(smoothing_window), n_frames)
    if window % 2 == 0:
        window -= 1
    if window >= 3 and int(smoothing_polyorder) < window:
        q = savgol_filter(
            q,
            window_length=window,
            polyorder=int(smoothing_polyorder),
            axis=0,
            mode="interp",
        )
    q = np.minimum(
        np.maximum(q, np.asarray(kinematics.q_min, dtype=np.float64)),
        np.asarray(kinematics.q_max, dtype=np.float64),
    )

    achieved_position = np.empty((n_frames, 3), dtype=np.float64)
    achieved_rotation = np.empty((n_frames, 3, 3), dtype=np.float64)
    for frame, configuration in enumerate(q):
        _set_frame(kinematics, frame)
        achieved_position[frame], achieved_rotation[frame] = kinematics.pose(configuration)
    position_error = np.linalg.norm(achieved_position - positions, axis=1)
    orientation_error = np.asarray([
        np.linalg.norm(
            Rotation.from_matrix(rotations[frame].T @ achieved_rotation[frame]).as_rotvec()
        )
        for frame in range(n_frames)
    ])

    effective_omega = np.clip(np.nan_to_num(omega, nan=0.0), 0.0, 1.0)
    effective_omega = np.where(
        effective_omega > 0.0,
        np.maximum(effective_omega, options.confidence_floor),
        0.0,
    )
    d1 = joint_difference_matrix(n_frames, int(kinematics.nq), 1, dt)
    d2 = joint_difference_matrix(n_frames, int(kinematics.nq), 2, dt)
    objective_terms = _objective_terms(
        kinematics,
        q,
        positions,
        rotations,
        effective_omega,
        weights,
        options,
        d1,
        d2,
        np.tile(np.asarray(kinematics.q_reference, dtype=np.float64), n_frames),
    )
    return SequentialBaselineResult(
        q=q,
        achieved_position=achieved_position,
        achieved_rotation=achieved_rotation,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        objective_terms=objective_terms,
        constraint_margins=trajectory_constraint_margins(kinematics, q, dt),
        total_iterations=total_iterations,
        solved_frames=solved_frames,
        held_frames=n_frames - solved_frames,
    )


def _primitive_descriptor(shape: object) -> dict | None:
    """hpp-fcl primitive → a small dict the viewer can rebuild, or None."""
    if shape is None:
        return None
    name = type(shape).__name__.lower()
    try:
        if "box" in name:
            hs = np.asarray(shape.halfSide, dtype=float).reshape(-1)
            return {"type": "box", "size": [float(2 * v) for v in hs[:3]]}
        if "sphere" in name:
            return {"type": "sphere", "radius": float(shape.radius)}
        if "cylinder" in name:
            return {"type": "cylinder", "radius": float(shape.radius),
                    "length": float(2 * shape.halfLength)}
    except Exception:  # noqa: BLE001
        return None
    return None


class PinocchioKinematics:
    """Fixed-base adapter exposing only commanded arm joints to the solver.

    Pinocchio represents unbounded revolute joints as ``[cos(q), sin(q)]`` so
    some models have ``model.nq != model.nv``. The batch problem and robot
    command archive contain physical arm angles, while attached-tool joints are
    known inputs selected per frame. We optimise the arm in a tangent chart and
    inject passive gripper positions only for FK and collision evaluation.
    """

    def __init__(
        self,
        robot: object,
        ee_frame: str,
        *,
        collision_pairs: int,
        collision_min_distance_m: float,
        actuated_joint_names: tuple[str, ...] | None = None,
        actuated_position_limits: tuple[tuple[float, float], ...] | None = None,
        passive_joint_positions: dict[str, np.ndarray] | None = None,
        gripper_prefix: str = "",
        position_points: tuple[
            tuple[str, tuple[float, float, float]], ...
        ] = (),
        floor_z_m: float | None = None,
        floor_normal: tuple[float, float, float] | np.ndarray | None = None,
        floor_offset_m: float = 0.0,
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
        self._task_position_points: tuple[tuple[int, np.ndarray], ...] = tuple(
            (
                int(self.model.getFrameId(frame_name)),
                np.asarray(offset, dtype=np.float64),
            )
            for frame_name, offset in position_points
        )
        for (frame_name, _offset), (frame_id, offset) in zip(
            position_points, self._task_position_points
        ):
            if frame_id >= len(self.model.frames):
                raise ValueError(f"robot model has no task-point frame {frame_name!r}")
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError(f"task-point offset for {frame_name!r} must be finite XYZ")
        if floor_z_m is not None and floor_normal is not None:
            raise ValueError("use either floor_z_m or floor_normal, not both")
        if floor_z_m is not None:
            if not np.isfinite(floor_z_m):
                raise ValueError("floor height must be finite")
            floor_normal = (0.0, 0.0, 1.0)
            floor_offset_m = float(floor_z_m)
        if floor_normal is None:
            self.floor_normal = None
            self.floor_offset_m = 0.0
        else:
            normal = np.asarray(floor_normal, dtype=np.float64)
            if normal.shape != (3,) or not np.isfinite(normal).all():
                raise ValueError("floor normal must be a finite XYZ vector")
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                raise ValueError("floor normal must be non-zero")
            if not np.isfinite(floor_offset_m):
                raise ValueError("floor offset must be finite")
            self.floor_normal = normal / norm
            self.floor_offset_m = float(floor_offset_m) / norm
        self.configuration_reference = np.asarray(pin.neutral(self.model), dtype=np.float64)

        if actuated_joint_names is None:
            active_joint_ids = [
                joint_id
                for joint_id in range(1, self.model.njoints)
                if int(self.model.nvs[joint_id]) > 0
            ]
        else:
            active_joint_ids = []
            for name in actuated_joint_names:
                joint_id = int(self.model.getJointId(name))
                if joint_id >= self.model.njoints:
                    raise ValueError(f"robot model has no actuated joint {name!r}")
                active_joint_ids.append(joint_id)
        if not active_joint_ids:
            raise ValueError("robot model exposes no actuated joints")

        active_v_indices: list[int] = []
        for joint_id in active_joint_ids:
            joint_nv = int(self.model.nvs[joint_id])
            if joint_nv != 1:
                raise ValueError(
                    "PoC batch solver supports fixed-base chains with 1-DoF arm joints"
                )
            active_v_indices.append(int(self.model.idx_vs[joint_id]))
        if len(set(active_v_indices)) != len(active_v_indices):
            raise ValueError("actuated joint list contains duplicate velocity coordinates")
        self._active_joint_ids = tuple(active_joint_ids)
        self._active_v_indices = np.asarray(active_v_indices, dtype=np.int32)
        self.nq = len(active_joint_ids)
        self.q_reference = np.zeros(self.nq, dtype=np.float64)
        self.q_min = np.full(self.nq, -np.inf, dtype=np.float64)
        self.q_max = np.full(self.nq, np.inf, dtype=np.float64)
        for output_index, joint_id in enumerate(active_joint_ids):
            joint_nq = int(self.model.nqs[joint_id])
            joint_nv = int(self.model.nvs[joint_id])
            if joint_nv != 1 or joint_nq not in (1, 2):
                raise ValueError(
                    "PoC batch solver supports fixed-base chains with 1-DoF joints"
                )
            if joint_nq == 1:
                q_index = int(self.model.idx_qs[joint_id])
                self.q_min[output_index] = float(self.model.lowerPositionLimit[q_index])
                self.q_max[output_index] = float(self.model.upperPositionLimit[q_index])
        self.q_min[~np.isfinite(self.q_min) | (self.q_min < -1e10)] = -np.inf
        self.q_max[~np.isfinite(self.q_max) | (self.q_max > 1e10)] = np.inf
        if actuated_position_limits is not None:
            limits = np.asarray(actuated_position_limits, dtype=np.float64)
            if limits.shape != (self.nq, 2):
                raise ValueError(
                    f"actuated position limits must have shape {(self.nq, 2)}"
                )
            if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
                raise ValueError("actuated position limits must be finite lower/upper pairs")
            self.q_min = limits[:, 0].copy()
            self.q_max = limits[:, 1].copy()
        self.velocity_limit = np.asarray(
            self.model.velocityLimit, dtype=np.float64
        )[self._active_v_indices].copy()
        self.velocity_limit[~np.isfinite(self.velocity_limit) | (self.velocity_limit <= 0)] = np.inf

        self._passive_positions: list[tuple[int, np.ndarray]] = []
        passive_lengths: set[int] = set()
        for name, trajectory in (passive_joint_positions or {}).items():
            joint_id = int(self.model.getJointId(name))
            if joint_id >= self.model.njoints:
                raise ValueError(f"robot model has no passive joint {name!r}")
            if joint_id in active_joint_ids:
                raise ValueError(f"joint {name!r} cannot be both actuated and passive")
            if int(self.model.nqs[joint_id]) != 1 or int(self.model.nvs[joint_id]) != 1:
                raise ValueError(f"passive joint {name!r} must own one configuration coordinate")
            values = np.asarray(trajectory, dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError(f"passive joint {name!r} needs a finite 1-D trajectory")
            q_index = int(self.model.idx_qs[joint_id])
            lower = float(self.model.lowerPositionLimit[q_index])
            upper = float(self.model.upperPositionLimit[q_index])
            if np.any(values < lower - 1e-9) or np.any(values > upper + 1e-9):
                raise ValueError(
                    f"passive joint {name!r} exceeds URDF limits [{lower}, {upper}]"
                )
            self._passive_positions.append((q_index, values.copy()))
            passive_lengths.add(len(values))
        if len(passive_lengths) > 1:
            raise ValueError("passive joint trajectories have different frame counts")
        self._passive_frame_count = next(iter(passive_lengths), 0)
        self._frame = 0

        self._scene_frame_ids = [0] + [
            frame_id
            for frame_id, frame in enumerate(self.model.frames)
            if frame_id > 0 and frame.type == pin.FrameType.BODY
        ]
        scene_index = {
            frame_id: index for index, frame_id in enumerate(self._scene_frame_ids)
        }
        scene_edges: list[tuple[int, int]] = []
        scene_edge_groups: list[str] = []
        for child_index, frame_id in enumerate(self._scene_frame_ids[1:], start=1):
            parent_frame = int(self.model.frames[frame_id].parentFrame)
            seen: set[int] = set()
            while parent_frame not in scene_index and parent_frame not in seen:
                seen.add(parent_frame)
                if parent_frame <= 0:
                    parent_frame = 0
                    break
                parent_frame = int(self.model.frames[parent_frame].parentFrame)
            parent_index = scene_index.get(parent_frame, 0)
            scene_edges.append((parent_index, child_index))
            child_name = str(self.model.frames[frame_id].name)
            scene_edge_groups.append(
                "gripper" if gripper_prefix and child_name.startswith(gripper_prefix)
                else "robot"
            )
        self._link_edges = np.asarray(scene_edges, dtype=np.int32).reshape(-1, 2)
        self._link_groups = tuple(scene_edge_groups)
        self._point_groups = tuple(
            "gripper"
            if gripper_prefix
            and str(self.model.frames[frame_id].name).startswith(gripper_prefix)
            else "robot"
            for frame_id in self._scene_frame_ids
        )

        self.collision_model = getattr(robot, "collision_model", None)
        self.collision_data = None
        self.collision_barrier = None
        self._floor_geometry_specs: tuple[
            tuple[int, np.ndarray, np.ndarray, tuple[str, object]], ...
        ] = ()
        if self.floor_normal is not None:
            if self.collision_model is None:
                raise RuntimeError(
                    "floor constraints requested but URDF has no collision model"
                )
            self._floor_geometry_specs = tuple(
                (
                    int(geometry.parentJoint),
                    np.asarray(geometry.placement.rotation, dtype=np.float64).copy(),
                    np.asarray(geometry.placement.translation, dtype=np.float64).copy(),
                    _floor_support_descriptor(geometry.geometry),
                )
                for geometry in self.collision_model.geometryObjects
                # The fixed pedestal is the definition of robot-base height,
                # not an IK-controlled body. Vendor meshes can also extend a
                # fraction of a millimetre below that mounting plane. All
                # movable arm, adapter and gripper collision bodies are kept.
                if int(geometry.parentJoint) != 0
            )
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
            neutral_configuration = self._configuration(self.q_reference)
            self.pin.forwardKinematics(self.model, neutral_data, neutral_configuration)
            self.pin.updateGeometryPlacements(
                self.model,
                neutral_data,
                self.collision_model,
                collision_data,
                neutral_configuration,
            )
            self.pin.computeDistances(
                self.model,
                neutral_data,
                self.collision_model,
                collision_data,
                neutral_configuration,
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

    def set_frame(self, frame: int) -> None:
        if self._passive_frame_count == 0:
            self._frame = 0
            return
        if not 0 <= int(frame) < self._passive_frame_count:
            raise IndexError(
                f"passive-joint frame {frame} outside [0, {self._passive_frame_count})"
            )
        self._frame = int(frame)

    def pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        configuration = self._configuration(q)
        self.pin.forwardKinematics(self.model, self.data, configuration)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame_id]
        if self._task_position_points:
            position = np.mean([
                np.asarray(self.data.oMf[frame_id].translation)
                + np.asarray(self.data.oMf[frame_id].rotation) @ offset
                for frame_id, offset in self._task_position_points
            ], axis=0)
        else:
            position = np.asarray(pose.translation)
        return np.asarray(position).copy(), np.asarray(pose.rotation).copy()

    def pose_jacobian(
        self, q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pose plus analytical linear/world-angular Jacobians of the task."""
        configuration = self._configuration(q)
        self.pin.computeJointJacobians(self.model, self.data, configuration)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame_id]
        orientation_jacobian = np.asarray(self.pin.getFrameJacobian(
            self.model,
            self.data,
            self.frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ))
        if self._task_position_points:
            positions: list[np.ndarray] = []
            position_jacobians: list[np.ndarray] = []
            for frame_id, offset in self._task_position_points:
                placement = self.data.oMf[frame_id]
                world_offset = np.asarray(placement.rotation) @ offset
                positions.append(np.asarray(placement.translation) + world_offset)
                frame_jacobian = np.asarray(self.pin.getFrameJacobian(
                    self.model,
                    self.data,
                    frame_id,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                ))
                position_jacobians.append(
                    frame_jacobian[:3] - _skew(world_offset) @ frame_jacobian[3:]
                )
            position = np.mean(positions, axis=0)
            position_jacobian = np.mean(position_jacobians, axis=0)
        else:
            position = np.asarray(pose.translation)
            position_jacobian = orientation_jacobian[:3]
        active = self._active_v_indices
        return (
            np.asarray(position, dtype=np.float64).copy(),
            np.asarray(pose.rotation, dtype=np.float64).copy(),
            np.asarray(position_jacobian[:, active], dtype=np.float64).copy(),
            np.asarray(orientation_jacobian[3:, active], dtype=np.float64).copy(),
        )

    def integrate(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float64) + np.asarray(dq, dtype=np.float64)

    def _configuration(self, q: np.ndarray) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (self.nq,):
            raise ValueError(f"expected {self.nq} joint angles, got {values.shape}")
        reference = self.configuration_reference.copy()
        for q_index, trajectory in self._passive_positions:
            reference[q_index] = trajectory[self._frame]
        tangent = np.zeros(self.model.nv, dtype=np.float64)
        tangent[self._active_v_indices] = values
        return np.asarray(
            self.pin.integrate(self.model, reference, tangent),
            dtype=np.float64,
        )

    def _collision_configuration(self, q: np.ndarray):
        return self.Configuration(
            self.model,
            self.model.createData(),
            self._configuration(q),
            collision_model=self.collision_model,
            collision_data=self.collision_model.createData(),
        )

    def collision_margins(self, q: np.ndarray) -> np.ndarray:
        if self.collision_barrier is None:
            return np.empty(0)
        return np.asarray(
            self.collision_barrier.compute_barrier(
                self._collision_configuration(q)
            ),
            dtype=np.float64,
        )

    def collision_linearization(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.collision_barrier is None:
            return np.empty(0), np.empty((0, self.nq))
        configuration = self._collision_configuration(q)
        jacobian = np.asarray(self.collision_barrier.compute_jacobian(configuration))
        return (
            np.asarray(self.collision_barrier.compute_barrier(configuration)),
            jacobian[:, self._active_v_indices],
        )

    def _floor_geometry_supports(self) -> tuple[tuple[int, np.ndarray], ...]:
        """Current world support point and parent joint for every collision body."""
        if self.floor_normal is None:
            return ()
        supports: list[tuple[int, np.ndarray]] = []
        for parent_joint, local_rotation, local_translation, descriptor in (
            self._floor_geometry_specs
        ):
            joint_placement = self.data.oMi[parent_joint]
            joint_rotation = np.asarray(joint_placement.rotation, dtype=np.float64)
            joint_translation = np.asarray(joint_placement.translation, dtype=np.float64)
            rotation = joint_rotation @ local_rotation
            translation = joint_translation + joint_rotation @ local_translation
            local_direction = rotation.T @ self.floor_normal
            local_point = _local_floor_support(descriptor, local_direction)
            supports.append((parent_joint, translation + rotation @ local_point))
        return tuple(supports)

    def floor_margins(self, q: np.ndarray) -> np.ndarray:
        """Signed plane margins of complete collision bodies and task points."""
        if self.floor_normal is None:
            return np.empty(0)
        configuration = self._configuration(q)
        self.pin.forwardKinematics(self.model, self.data, configuration)
        self.pin.updateFramePlacements(self.model, self.data)
        geometry_margins = [
            float(self.floor_normal @ point - self.floor_offset_m)
            for _parent_joint, point in self._floor_geometry_supports()
        ]
        task_margins = [
            float(self.floor_normal @ (
                np.asarray(self.data.oMf[frame_id].translation)
                + np.asarray(self.data.oMf[frame_id].rotation) @ offset
            ) - self.floor_offset_m)
            for frame_id, offset in self._task_position_points
        ]
        return np.asarray(geometry_margins + task_margins, dtype=np.float64)

    def floor_linearization(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearise the floor at each collision body's lowest support point."""
        if self.floor_normal is None:
            return np.empty(0), np.empty((0, self.nq))
        configuration = self._configuration(q)
        self.pin.computeJointJacobians(self.model, self.data, configuration)
        self.pin.updateFramePlacements(self.model, self.data)
        margins: list[float] = []
        rows: list[np.ndarray] = []
        for parent_joint, point in self._floor_geometry_supports():
            joint_origin = np.asarray(self.data.oMi[parent_joint].translation)
            world_offset = point - joint_origin
            if parent_joint == 0:
                joint_jacobian = np.zeros((6, self.model.nv), dtype=np.float64)
            else:
                joint_jacobian = np.asarray(self.pin.getJointJacobian(
                    self.model,
                    self.data,
                    parent_joint,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                ))
            skew = np.array([
                [0.0, -world_offset[2], world_offset[1]],
                [world_offset[2], 0.0, -world_offset[0]],
                [-world_offset[1], world_offset[0], 0.0],
            ])
            point_jacobian = joint_jacobian[:3] - skew @ joint_jacobian[3:]
            row = self.floor_normal @ point_jacobian[:, self._active_v_indices]
            margins.append(float(self.floor_normal @ point - self.floor_offset_m))
            rows.append(np.asarray(row, dtype=np.float64))
        for frame_id, offset in self._task_position_points:
            placement = self.data.oMf[frame_id]
            world_offset = np.asarray(placement.rotation) @ offset
            point = np.asarray(placement.translation) + world_offset
            frame_jacobian = np.asarray(self.pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ))
            skew = np.array([
                [0.0, -world_offset[2], world_offset[1]],
                [world_offset[2], 0.0, -world_offset[0]],
                [-world_offset[1], world_offset[0], 0.0],
            ])
            point_jacobian = frame_jacobian[:3] - skew @ frame_jacobian[3:]
            row = self.floor_normal @ point_jacobian[:, self._active_v_indices]
            margins.append(float(self.floor_normal @ point - self.floor_offset_m))
            rows.append(np.asarray(row, dtype=np.float64))
        if not rows:
            return np.empty(0), np.empty((0, self.nq))
        return np.asarray(margins), np.stack(rows)

    def joint_points(self, q: np.ndarray) -> np.ndarray:
        """URDF body-frame origins, including fixed tool/TCP frames."""
        self.pin.forwardKinematics(self.model, self.data, self._configuration(q))
        self.pin.updateFramePlacements(self.model, self.data)
        points = np.zeros((len(self._scene_frame_ids), 3), dtype=np.float64)
        for index, frame_id in enumerate(self._scene_frame_ids[1:], start=1):
            points[index] = np.asarray(self.data.oMf[frame_id].translation)
        return points

    def joint_placements(self, q: np.ndarray) -> np.ndarray:
        """``(model.njoints, 4, 4)`` world←joint homogeneous transforms for the
        current passive-joint frame. A visual mesh's world pose is
        ``joint_placements[g.parent_joint] @ g.placement`` (see
        :meth:`visual_geometries`)."""
        self.pin.forwardKinematics(self.model, self.data, self._configuration(q))
        out = np.zeros((self.model.njoints, 4, 4), dtype=np.float64)
        for joint_id in range(self.model.njoints):
            out[joint_id] = np.asarray(self.data.oMi[joint_id].homogeneous)
        return out

    def visual_geometries(self) -> list[dict]:
        """Static descriptors for the URDF *visual* meshes (arm + attached
        gripper), for the 3-D viewer. Each entry:

        ``mesh_path``      path relative to ``<MODELS_DIR>/robot_descriptions/``
                           (or absent for a primitive shape)
        ``primitive``      ``{"type": "box|sphere|cylinder", ...}`` when there is
                           no mesh file
        ``parent_joint``   index into :meth:`joint_placements`
        ``placement``      4×4 joint←geometry offset
        ``scale``          mesh scale (3,)
        ``color``          RGBA (4,), 0..1
        """
        import os

        model = getattr(self.robot, "visual_model", None)
        if model is None:
            return []
        try:
            from viki import config as _cfg

            root = os.path.join(
                str(getattr(_cfg, "MODELS_DIR", "models/")), "robot_descriptions"
            )
            root = os.path.realpath(root)
        except Exception:  # noqa: BLE001
            root = None

        out: list[dict] = []
        for g in model.geometryObjects:
            item: dict = {
                "name": str(g.name),
                "parent_joint": int(g.parentJoint),
                "placement": np.asarray(g.placement.homogeneous, dtype=float).tolist(),
                "scale": [float(x) for x in np.asarray(g.meshScale).reshape(-1)[:3]],
                "color": [float(x) for x in np.asarray(g.meshColor).reshape(-1)[:4]],
            }
            mesh_path = str(getattr(g, "meshPath", "") or "")
            if mesh_path and os.path.isfile(mesh_path):
                real = os.path.realpath(mesh_path)
                if root and real.startswith(root + os.sep):
                    item["mesh_path"] = os.path.relpath(real, root)
                else:
                    item["mesh_path_abs"] = real  # served only if under a safe root
            else:
                prim = _primitive_descriptor(getattr(g, "geometry", None))
                if prim is None:
                    continue  # nothing drawable
                item["primitive"] = prim
            out.append(item)
        return out

    @property
    def link_edges(self) -> np.ndarray:
        return self._link_edges.copy()

    @property
    def link_groups(self) -> tuple[str, ...]:
        return self._link_groups

    @property
    def point_groups(self) -> tuple[str, ...]:
        return self._point_groups

    @property
    def joint_names(self) -> list[str]:
        return [str(self.model.names[joint_id]) for joint_id in self._active_joint_ids]
