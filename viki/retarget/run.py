"""Episode retargeting: rig-frame hand poses to a calibrated robot plan.

The implementation intentionally has no object-centric or implicit legacy
coordinate path. Input poses are explicitly transformed by the episode's
calibration anchor; robot axes are parallel to that calibrated frame and
``base_position=[x,y,z]`` is the sole robot registration parameter.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import viki.config as app_config
from viki.contracts import Episode, GripperState, PLAN_KEYS, cln_pose_keys
from viki.episode import mark_stage
from viki.gripper import load_gripper
from viki.retarget.archive import write_hdf5_archive
from viki.retarget.frames import (
    CALIBRATION_FRAME,
    calibration_to_robot,
    load_rig_to_calibration,
    require_rig_frame,
    rig_pose_to_calibration,
    robot_to_calibration,
    validate_base_position,
)
from viki.retarget.grippers import (
    DEFAULT_GRIPPER,
    attach_gripper,
    normalize_gripper,
)
from viki.retarget.robots import RobotConfig, normalize_robot
from viki.retarget.solver import (
    BatchOptions,
    BatchWeights,
    PinocchioKinematics,
    solve_trajectory,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetargetConfig:
    robot: str
    base_position: tuple[float, float, float]
    hand_to_ee_translation: tuple[float, float, float]
    hand_to_ee_rpy_deg: tuple[float, float, float]
    gripper: str
    pose_source: str
    approach_sec: float
    collision_pairs: int
    collision_min_distance_m: float
    weights: BatchWeights
    solver: BatchOptions


@dataclass(frozen=True)
class RetargetTargets:
    timestamps: np.ndarray
    position_rig: np.ndarray
    rotation_rig: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    gripper_closed: np.ndarray
    fps: float
    pose_source: str


def _setting(options: Mapping[str, Any], key: str, config_key: str, default: Any) -> Any:
    return options[key] if key in options else getattr(app_config, config_key, default)


def config_from_options(
    robot: str | None = None, options: Mapping[str, Any] | None = None
) -> RetargetConfig:
    """Merge one request over the restart-persisted retarget defaults."""
    values = dict(options or {})
    base = validate_base_position(
        _setting(values, "base_position", "RETARGET_ROBOT_BASE_POSITION", [0, 0, 0])
    )
    hand_t = validate_base_position(
        _setting(values, "hand_to_ee_translation", "RETARGET_HAND_TO_EE_TRANSLATION", [0, 0, 0])
    )
    hand_rpy = validate_base_position(
        _setting(
            values,
            "hand_to_ee_rpy_deg",
            "RETARGET_HAND_TO_EE_RPY_DEG",
            [-175.675, 15.45, -47.922],
        )
    )
    weights = BatchWeights(
        position=float(_setting(values, "w_position", "RETARGET_W_POSITION", 1.0)),
        orientation=float(
            _setting(values, "w_orientation", "RETARGET_W_ORIENTATION", 0.01)
        ),
        velocity=float(_setting(values, "w_velocity", "RETARGET_W_VELOCITY", 2e-3)),
        acceleration=float(_setting(values, "w_acceleration", "RETARGET_W_ACCELERATION", 2e-5)),
        posture=float(_setting(values, "w_posture", "RETARGET_W_POSTURE", 1e-4)),
    )
    collision_enabled = bool(
        _setting(values, "collision_enabled", "RETARGET_COLLISION_ENABLED", True)
    )
    solver = BatchOptions(
        huber_delta=float(_setting(values, "huber_delta", "RETARGET_HUBER_DELTA", 0.05)),
        confidence_floor=float(
            _setting(values, "confidence_floor", "RETARGET_CONFIDENCE_FLOOR", 0.05)
        ),
        lm_damping=float(_setting(values, "lm_damping", "RETARGET_LM_DAMPING", 1e-5)),
        max_iterations=int(
            _setting(values, "max_iterations", "RETARGET_MAX_ITERATIONS", 40)
        ),
        max_step_rad=float(
            _setting(values, "max_step_rad", "RETARGET_MAX_STEP_RAD", 0.20)
        ),
        convergence_rad=float(
            _setting(values, "convergence_rad", "RETARGET_CONVERGENCE_RAD", 1e-3)
        ),
        qp_solver=str(_setting(values, "qp_solver", "RETARGET_QP_SOLVER", "osqp")),
        collision_enabled=collision_enabled,
    )
    gripper_name = str(
        _setting(
            values,
            "gripper",
            "RETARGET_DEFAULT_GRIPPER_MODEL",
            DEFAULT_GRIPPER,
        )
    )
    physical_gripper = normalize_gripper(gripper_name)
    out = RetargetConfig(
        robot=str(robot or _setting(values, "robot", "RETARGET_DEFAULT_ROBOT", "ur10")),
        base_position=tuple(float(x) for x in base),
        hand_to_ee_translation=tuple(float(x) for x in hand_t),
        hand_to_ee_rpy_deg=tuple(float(x) for x in hand_rpy),
        gripper=physical_gripper.key,
        pose_source=str(
            _setting(values, "pose_source", "PERCEPTION_HAND_POSE_SOURCE", "landmarks")
        ),
        approach_sec=float(
            _setting(values, "approach_sec", "RETARGET_APPROACH_SEC", 2.0)
        ),
        collision_pairs=(
            int(_setting(values, "collision_pairs", "RETARGET_COLLISION_PAIRS", 8))
            if collision_enabled else 0
        ),
        collision_min_distance_m=float(
            _setting(
                values,
                "collision_min_distance_m",
                "RETARGET_COLLISION_MIN_DISTANCE_M",
                0.02,
            )
        ),
        weights=weights,
        solver=solver,
    )
    out.weights.validate()
    out.solver.validate()
    if out.approach_sec < 0.0:
        raise ValueError("approach_sec must be non-negative")
    if out.collision_pairs < 0 or out.collision_min_distance_m < 0.0:
        raise ValueError("collision settings must be non-negative")
    normalize_robot(out.robot)
    return out


def estimate_fps(timestamps: np.ndarray) -> float:
    values = np.asarray(timestamps, dtype=np.float64)
    if len(values) < 2:
        return 30.0
    delta = np.diff(values)
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if not len(delta):
        return 30.0
    median = float(np.median(delta))
    seconds = median / 1_000_000.0 if median > 1000.0 else median
    return 1.0 / seconds


def _interpolate_positions(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"positions must have shape (T, 3), got {out.shape}")
    time = np.arange(len(out), dtype=np.float64)
    for axis in range(3):
        finite = np.isfinite(out[:, axis])
        if not finite.any():
            raise ValueError("target positions contain no finite samples")
        out[:, axis] = np.interp(time, time[finite], out[finite, axis])
    return out


def _interpolate_rotations(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rotations = np.asarray(values, dtype=np.float64)
    if rotations.shape != (len(valid), 3, 3):
        raise ValueError(f"rotations must have shape ({len(valid)}, 3, 3)")
    good = np.asarray(valid, dtype=bool) & np.isfinite(rotations).all(axis=(1, 2))
    indices = np.flatnonzero(good)
    if not len(indices):
        return np.tile(np.eye(3), (len(valid), 1, 1))
    # Project detector noise back onto SO(3) before Slerp.
    clean = Rotation.from_matrix(rotations[indices]).as_matrix()
    if len(indices) == 1:
        return np.tile(clean[0], (len(valid), 1, 1))
    query = np.arange(len(valid), dtype=np.float64)
    query = np.clip(query, indices[0], indices[-1])
    return Slerp(indices.astype(np.float64), Rotation.from_matrix(clean))(query).as_matrix()


def load_targets(path: Path, cfg: RetargetConfig) -> RetargetTargets:
    """Read the canonical cln artifact and make finite batch-solver targets."""
    with np.load(path, allow_pickle=False) as data:
        required = {"timestamps", "positions", "rotations", "valid", "coordinate_frame"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing cln keys: {', '.join(missing)}")
        require_rig_frame(data["coordinate_frame"])
        p_key, r_key = cln_pose_keys(data.files, cfg.pose_source)
        timestamps = np.asarray(data["timestamps"]).copy()
        positions = np.asarray(data[p_key], dtype=np.float64)
        rotations = np.asarray(data[r_key], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        confidence = np.asarray(
            data["omega"] if "omega" in data.files else valid.astype(np.float64),
            dtype=np.float64,
        )
        gripper = np.asarray(
            data["gripper"] if "gripper" in data.files else np.zeros(len(valid)),
            dtype=bool,
        )
    n = len(timestamps)
    if any(len(value) != n for value in (positions, rotations, valid, confidence, gripper)):
        raise ValueError("cln target arrays have different frame counts")
    # Some historical CLNs retain finite detector/fusion garbage in frames
    # explicitly marked invalid. Finite-ness alone is therefore not evidence:
    # remove those samples before interpolation so endpoint fill uses the first
    # or last valid pose instead of metre-scale outliers.
    positions = positions.copy()
    positions[~valid] = np.nan
    return RetargetTargets(
        timestamps=timestamps,
        position_rig=_interpolate_positions(positions),
        rotation_rig=_interpolate_rotations(rotations, valid),
        confidence=np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0) * valid,
        valid=valid,
        gripper_closed=gripper,
        fps=estimate_fps(timestamps),
        pose_source=("hand_fit" if p_key.startswith("hand_fit") else "landmarks"),
    )


def _apply_hand_to_ee(
    positions: np.ndarray,
    rotations: np.ndarray,
    translation: tuple[float, float, float],
    rpy_deg: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    local_t = np.asarray(translation, dtype=np.float64)
    local_r = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return (
        positions + np.einsum("tij,j->ti", rotations, local_t),
        np.einsum("tij,jk->tik", rotations, local_r),
    )


def _load_robot_description(description: str):
    os.environ.setdefault(
        "ROBOT_DESCRIPTIONS_CACHE",
        os.path.join(str(getattr(app_config, "MODELS_DIR", "models/")), "robot_descriptions"),
    )
    try:
        from robot_descriptions.loaders.pinocchio import load_robot_description
    except ImportError as exc:
        raise RuntimeError("retarget requires robot_descriptions") from exc
    return load_robot_description(description)


def _gripper_commands(
    closed: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    # Perception still emits one semantic binary state. The physical gripper
    # registry separately maps it to the URDF's master joint for geometry.
    model = load_gripper("binary")
    states = [
        GripperState(
            closed=bool(is_closed),
            width=0.0 if is_closed else 1.0,
            confidence=float(confidence[t]),
        )
        for t, is_closed in enumerate(closed)
    ]
    return model.encode(states), model.command_names


def _approach(
    reference: np.ndarray,
    trajectory: np.ndarray,
    seconds: float,
    dt: float,
) -> np.ndarray:
    """Blend neutral into the first trajectory sample with matched derivatives.

    The endpoint itself is omitted because it is already ``trajectory[0]``.
    Matching its forward velocity and acceleration avoids the large acceleration
    impulse produced by the retired linear interpolation at that join.
    """
    count = int(round(seconds / dt))
    if count <= 0:
        return np.empty((0, len(reference)), dtype=np.float64)
    q = np.asarray(trajectory, dtype=np.float64)
    start = np.asarray(reference, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1:] != start.shape:
        raise ValueError("trajectory must have shape (T, len(reference)) with T > 0")

    if len(q) >= 3:
        end_velocity = (-3.0 * q[0] + 4.0 * q[1] - q[2]) / (2.0 * dt)
        end_acceleration = (q[0] - 2.0 * q[1] + q[2]) / (dt * dt)
    elif len(q) == 2:
        end_velocity = (q[1] - q[0]) / dt
        end_acceleration = np.zeros_like(start)
    else:
        end_velocity = np.zeros_like(start)
        end_acceleration = np.zeros_like(start)

    duration = count * dt
    displacement = q[0] - start
    scaled_velocity = duration * end_velocity
    scaled_acceleration = duration * duration * end_acceleration
    # Quintic coefficients after fixing p(0), v(0), a(0) to the neutral pose
    # and p(1), v(1), a(1) to the start of the optimised trajectory.
    c3 = 10.0 * displacement - 4.0 * scaled_velocity + 0.5 * scaled_acceleration
    c4 = -15.0 * displacement + 7.0 * scaled_velocity - scaled_acceleration
    c5 = 6.0 * displacement - 3.0 * scaled_velocity + 0.5 * scaled_acceleration
    phase = (np.arange(count, dtype=np.float64) * dt / duration)[:, None]
    return start[None, :] + phase**3 * c3 + phase**4 * c4 + phase**5 * c5


def _config_json(cfg: RetargetConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))


def retarget_episode(
    ep: Episode,
    robot: str | None = None,
    *,
    options: Mapping[str, Any] | None = None,
    report: Callable[..., None] | None = None,
    log: Callable[[str], None] | None = None,
) -> str:
    """Optimise ``ep.cln_npz`` into the canonical, self-contained ``plan.h5``."""
    if not ep.cln_npz.exists():
        raise FileNotFoundError(f"no cln.npz for episode {ep.id}; run Extract first")
    cfg = config_from_options(robot, options)
    robot_cfg: RobotConfig = normalize_robot(cfg.robot)
    gripper_cfg = normalize_gripper(cfg.gripper)
    targets = load_targets(ep.cln_npz, cfg)
    rig_to_calibration = load_rig_to_calibration(ep.raw_dir)
    hand_calib_p, hand_calib_r = rig_pose_to_calibration(
        targets.position_rig,
        targets.rotation_rig,
        rig_to_calibration,
    )
    target_calib_p, target_calib_r = _apply_hand_to_ee(
        hand_calib_p,
        hand_calib_r,
        cfg.hand_to_ee_translation,
        cfg.hand_to_ee_rpy_deg,
    )
    target_robot_p = calibration_to_robot(target_calib_p, cfg.base_position)
    if log is not None:
        log(
            f"{robot_cfg.description}: {len(target_robot_p)} frames in {CALIBRATION_FRAME}, "
            f"base={list(cfg.base_position)}"
        )
    arm_model = _load_robot_description(robot_cfg.description)
    assembly = attach_gripper(arm_model, robot_cfg, gripper_cfg)
    gripper_joint_position = gripper_cfg.joint_positions(targets.gripper_closed)
    kinematics = PinocchioKinematics(
        assembly.robot,
        assembly.tcp_frame,
        collision_pairs=cfg.collision_pairs,
        collision_min_distance_m=cfg.collision_min_distance_m,
        actuated_joint_names=robot_cfg.joint_names,
        passive_joint_positions={
            assembly.drive_joint: gripper_joint_position,
        },
        gripper_prefix=assembly.gripper_prefix,
    )
    result = solve_trajectory(
        kinematics,
        target_robot_p,
        target_calib_r,
        targets.confidence,
        1.0 / targets.fps,
        cfg.weights,
        cfg.solver,
        report=report,
    )
    achieved_calib = robot_to_calibration(result.achieved_position, cfg.base_position)
    link_positions = []
    for frame, q in enumerate(result.q):
        kinematics.set_frame(frame)
        link_positions.append(
            robot_to_calibration(kinematics.joint_points(q), cfg.base_position)
        )
    link_positions = np.stack(link_positions)
    gripper_command, command_names = _gripper_commands(
        targets.gripper_closed, targets.confidence
    )
    dt = 1.0 / targets.fps
    velocity = np.gradient(result.q, dt, axis=0) if len(result.q) > 1 else np.zeros_like(result.q)
    acceleration = np.gradient(velocity, dt, axis=0) if len(result.q) > 2 else np.zeros_like(result.q)
    q_approach = _approach(
        kinematics.q_reference, result.q, cfg.approach_sec, dt
    )
    full_q = np.concatenate((q_approach, result.q), axis=0)
    full_velocity = (
        np.gradient(full_q, dt, axis=0) if len(full_q) > 1 else np.zeros_like(full_q)
    )
    full_acceleration = (
        np.gradient(full_velocity, dt, axis=0)
        if len(full_velocity) > 2 else np.zeros_like(full_velocity)
    )
    metrics = {
        "position_rmse_mm": float(np.sqrt(np.mean(result.position_error_m ** 2)) * 1000.0),
        "position_p95_mm": float(np.percentile(result.position_error_m, 95) * 1000.0),
        "orientation_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(result.orientation_error_rad ** 2)))
        ),
        "max_joint_velocity_rad_s": float(np.max(np.abs(full_velocity))),
        "max_joint_acceleration_rad_s2": float(np.max(np.abs(full_acceleration))),
        "iterations": result.iterations,
        "converged": result.converged,
        "objective": result.objective,
        "min_collision_margin": (
            result.min_collision_margin if np.isfinite(result.min_collision_margin) else None
        ),
    }
    plan = {
        "schema_version": 3,
        "coordinate_frame": CALIBRATION_FRAME,
        "timestamps": targets.timestamps,
        "fps": targets.fps,
        "dt": dt,
        "robot": robot_cfg.description,
        "robot_key": cfg.robot,
        "ee_frame": assembly.tcp_frame,
        "joint_names_json": json.dumps(kinematics.joint_names),
        "link_edges": kinematics.link_edges,
        "link_groups_json": json.dumps(kinematics.link_groups),
        "point_groups_json": json.dumps(kinematics.point_groups),
        "T_calibration_rig": rig_to_calibration,
        "base_position_calibration": np.asarray(cfg.base_position),
        "q": result.q.astype(np.float32),
        "q_approach": q_approach.astype(np.float32),
        "joint_velocity": velocity.astype(np.float32),
        "joint_acceleration": acceleration.astype(np.float32),
        "gripper_model": cfg.gripper,
        "gripper_description": str(gripper_cfg.description),
        "gripper_tcp_frame": assembly.tcp_frame,
        "gripper_drive_joint": assembly.drive_joint,
        "gripper_command_names_json": json.dumps(command_names),
        "gripper_command": gripper_command,
        "gripper_closed": targets.gripper_closed,
        "gripper_joint_position": gripper_joint_position.astype(np.float32),
        "gripper_opening_m": gripper_cfg.opening_widths(
            targets.gripper_closed
        ).astype(np.float32),
        "omega": targets.confidence.astype(np.float32),
        "target_position_calibration": target_calib_p.astype(np.float32),
        "target_rotation_calibration": target_calib_r.astype(np.float32),
        "target_position_robot": target_robot_p.astype(np.float32),
        "achieved_position_calibration": achieved_calib.astype(np.float32),
        "achieved_rotation_calibration": result.achieved_rotation.astype(np.float32),
        "position_error_m": result.position_error_m.astype(np.float32),
        "orientation_error_rad": result.orientation_error_rad.astype(np.float32),
        "link_positions_calibration": link_positions.astype(np.float32),
        "solver_status": "converged" if result.converged else "max_iterations",
        "config_json": _config_json(cfg),
        "metrics_json": json.dumps(metrics, sort_keys=True),
        "source_pose": targets.pose_source,
    }
    assert set(plan) == set(PLAN_KEYS)
    write_hdf5_archive(ep.plan_h5, plan, schema="viki_plan_hdf5_v3")
    mark_stage(
        ep,
        "retarget",
        robot=robot_cfg.description,
        coordinate_frame=CALIBRATION_FRAME,
        position_rmse_mm=metrics["position_rmse_mm"],
        solver_status=plan["solver_status"],
    )
    logger.info(
        "retarget %s: %s, %.1f mm RMSE",
        ep.id,
        plan["solver_status"],
        metrics["position_rmse_mm"],
    )
    return str(ep.plan_h5)
