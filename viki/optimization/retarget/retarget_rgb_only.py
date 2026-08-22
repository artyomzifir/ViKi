"""RGB-only end-effector retargeting for ViKi.

Consumes pre-computed end-effector trajectories (wrist positions and palm
rotations, in robot-frame metres) produced by the smooth stage
(``viki.optimization.preparation.processor``) and runs PINK IK against a Pinocchio robot
description, writing a trajectory archive.

This module is IK-only: deriving end-effector poses from raw landmarks happens
once, at the smooth stage, not here.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from viki.config import MODELS_DIR
import viki.config as viki_config
from .archive_io import write_hdf5_archive
from .smoothing import adjusted_savgol_window, smooth_savgol


RIGHT_BODY_WRIST = 16
LEFT_BODY_WRIST = 15
SMOOTHED_TARGET_KEYS = {"positions", "rotations", "valid", "timestamps"}

# Same transform used in the exploration notebook: MediaPipe RGB coordinates
# into the robot-facing convention used by the saved trajectory archives.
R_DEFAULT = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
T_DEFAULT = np.zeros(3, dtype=np.float64)
R_REFLECTION_FIX = np.diag([1.0, 1.0, -1.0])
ROBOT_BASE_OFFSET = np.array(
    getattr(viki_config, "ROBOT_BASE_OFFSET", [0.0, 0.0, 0.0]), dtype=np.float64,
)
TARGET_OFFSET = np.array(
    getattr(viki_config, "TARGET_OFFSET", [0.0, 0.0, 0.0]), dtype=np.float64,
)


@dataclass(frozen=True)
class RobotConfig:
    description: str
    ee_frame: str
    joint_names: tuple[str, ...]


ROBOT_CONFIGS: dict[str, RobotConfig] = {
    "ur10": RobotConfig(
        "ur10_official_description",
        "wrist_3_link",
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
    ),
    "ur10_description": RobotConfig(
        "ur10_official_description",
        "wrist_3_link",
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
    ),
    "iiwa14": RobotConfig(
        "iiwa14_description",
        "iiwa_link_ee",
        (
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ),
    ),
    "iiwa14_description": RobotConfig(
        "iiwa14_description",
        "iiwa_link_ee",
        (
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ),
    ),
}


@dataclass(frozen=True)
class RunConfig:
    robot: RobotConfig
    working_hand: str
    landmark_sg_window: int
    landmark_sg_polyorder: int
    ik_position_cost: float
    ik_orientation_cost: float
    ik_posture_cost: float
    target_mode: str
    ik_substeps: int
    ik_solver: str
    approach_sec: float
    joint_sg_window: int
    joint_sg_polyorder: int
    limit_frames: int | None
    recenter_to_neutral: bool
    trajectory_scale: float
    align_initial_orientation: bool
    trajectory_scale_origin: str = "auto"
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RetargetInput:
    body: np.ndarray
    hand: np.ndarray | None
    fps: float
    orientation_valid: np.ndarray | None = None
    target_rotations: np.ndarray | None = None
    timestamps_us: np.ndarray | None = None
    source_format: str = "legacy_sample"
    coordinate_frame: str = "viki_world_or_camera"


def _load_robot_description(description: str):
    """Load robot description, caching git clones in MODELS_DIR."""
    os.environ.setdefault(
        "ROBOT_DESCRIPTIONS_CACHE",
        os.path.join(MODELS_DIR, "robot_descriptions"),
    )
    from robot_descriptions.loaders.pinocchio import load_robot_description as _load

    return _load(description)


def require_ik_dependencies():
    """Import PINK/Pinocchio dependencies with a direct runtime message."""
    try:
        import pinocchio as pin
        import pink
        from pink import Configuration, solve_ik
        from pink.tasks import FrameTask, PostureTask
    except ImportError as exc:
        raise RuntimeError(
            "RGB-only retargeting requires robotics Pinocchio, PINK "
            "(pin-pink), robot_descriptions, and a QP solver such as quadprog. "
            "In the conda env used here: `python -m pip install pin-pink "
            "robot_descriptions qpsolvers quadprog typing_extensions`."
        ) from exc

    missing = [name for name in ("SE3", "neutral") if not hasattr(pin, name)]
    if missing:
        raise RuntimeError(
            "The imported 'pinocchio' module is not the robotics Pinocchio "
            f"runtime. Missing attributes: {', '.join(missing)}."
        )
    return pin, pink, Configuration, solve_ik, FrameTask, PostureTask, _load_robot_description


def npz_scalar(value: Any, default: Any = None) -> Any:
    """Return a Python scalar from a 0-D npz value."""
    if value is None:
        return default
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def should_apply_legacy_transform(coordinate_frame: Any) -> bool:
    """Legacy samples need the MediaPipe-to-robot convention transform."""
    frame = str(npz_scalar(coordinate_frame, "") or "").strip().lower()
    return frame not in {"robot_base", "robot-base", "robot base"}


def normalize_coordinate_frame(coordinate_frame: Any) -> str:
    frame = str(npz_scalar(coordinate_frame, "") or "").strip().lower()
    if frame in {"robot_base", "robot-base", "robot base"}:
        return "robot_base"
    return "viki_world_or_camera"


def resolve_trajectory_scale_origin(requested: str, coordinate_frame: Any) -> str:
    """Resolve automatic scaling to the calibrated frame's natural origin."""
    if requested == "auto":
        return (
            "robot_base"
            if normalize_coordinate_frame(coordinate_frame) == "robot_base"
            else "initial_wrist"
        )
    if requested not in {"initial_wrist", "robot_base"}:
        raise ValueError(
            "trajectory_scale_origin must be auto, initial_wrist, or robot_base."
        )
    return requested


def normalize_robot(robot: str) -> RobotConfig:
    key = robot.strip()
    if key not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot '{robot}'. Expected one of: {', '.join(sorted(ROBOT_CONFIGS))}.")
    return ROBOT_CONFIGS[key]


def output_traj_path(out: Path, sample_path: Path, robot: RobotConfig) -> Path:
    """Resolve --out into a trajectory archive path."""
    if out.suffix.lower() in {".h5", ".hdf5"}:
        return out
    if out.suffix.lower() == ".npz":
        return out.with_suffix(".h5")
    robot_alias = robot.description.replace("_description", "")
    name = out.name
    if not name.endswith("_traj"):
        name = f"{name}_traj"
    if out.name in {"", "."}:
        name = f"{sample_path.stem}_{robot_alias}_traj"
    return out.with_name(name + ".h5")


def transform_points(points: np.ndarray, rotation: np.ndarray = R_DEFAULT, translation: np.ndarray = T_DEFAULT) -> np.ndarray:  # noqa: F821
    arr = np.asarray(points, dtype=np.float64)
    rot = np.asarray(rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected points with trailing xyz dimension, got {arr.shape}.")
    if rot.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {rot.shape}.")
    if trans.shape != (3,):
        raise ValueError(f"Expected translation shape (3,), got {trans.shape}.")

    out = np.empty_like(arr, dtype=np.float64)
    out[..., 0] = arr[..., 0] * rot[0, 0] + arr[..., 1] * rot[0, 1] + arr[..., 2] * rot[0, 2] + trans[0]
    out[..., 1] = arr[..., 0] * rot[1, 0] + arr[..., 1] * rot[1, 1] + arr[..., 2] * rot[1, 2] + trans[1]
    out[..., 2] = arr[..., 0] * rot[2, 0] + arr[..., 1] * rot[2, 1] + arr[..., 2] * rot[2, 2] + trans[2]
    return out


def transform_rotations_to_robot(rotations: np.ndarray) -> np.ndarray:
    """Apply the legacy ViKi/world-to-robot rotation convention."""
    arr = np.asarray(rotations, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotations shape (T, 3, 3), got {arr.shape}.")
    out = np.full_like(arr, np.nan, dtype=np.float64)
    finite = np.isfinite(arr).all(axis=(1, 2))
    out[finite] = np.einsum("ij,tjk,kl->til", R_DEFAULT, arr[finite], R_REFLECTION_FIX)
    return out


def estimate_fps_from_timestamps(timestamps_us: np.ndarray) -> float:
    timestamps = np.asarray(timestamps_us, dtype=np.float64)
    if len(timestamps) < 2:
        return 30.0
    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    # Current smoothed files store microseconds. Keep a small fallback for
    # already-second timestamps to make tests and hand-authored files readable.
    scale = 1_000_000.0 if float(np.median(dt)) > 1_000.0 else 1.0
    return float(1.0 / np.median(dt / scale))


def interpolate_nan_positions(positions: np.ndarray) -> np.ndarray:
    """Linearly fill missing xyz target positions over time."""
    out = np.asarray(positions, dtype=np.float64).copy()
    frames = np.arange(len(out), dtype=np.float64)
    for dim in range(3):
        series = out[:, dim]
        valid = np.isfinite(series)
        if valid.all():
            continue
        if not valid.any():
            raise ValueError("Smoothed target positions contain no finite samples.")
        if valid.sum() == 1:
            series[~valid] = series[valid][0]
        else:
            series[~valid] = np.interp(frames[~valid], frames[valid], series[valid])
        out[:, dim] = series
    return out


def load_smoothed_targets(
    sample_path: Path,
    working_hand: str,
    limit_frames: int | None,
) -> RetargetInput:
    """Load already-smoothed wrist positions and palm rotations."""
    with np.load(sample_path, allow_pickle=True) as data:
        missing = sorted(SMOOTHED_TARGET_KEYS.difference(data.files))
        if missing:
            raise KeyError(f"{sample_path} is missing smoothed target keys: {', '.join(missing)}.")
        positions = np.asarray(data["positions"], dtype=np.float64)
        rotations = np.asarray(data["rotations"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        timestamps_us = np.asarray(data["timestamps"], dtype=np.int64)
        coordinate_frame = (
            data["coordinate_frame"]
            if "coordinate_frame" in data.files
            else "viki_world_or_camera"
        )

    if limit_frames is not None:
        if limit_frames <= 0:
            raise ValueError("--limit-frames must be positive when set.")
        positions = positions[:limit_frames]
        rotations = rotations[:limit_frames]
        valid = valid[:limit_frames]
        timestamps_us = timestamps_us[:limit_frames]

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions shape (T, 3), got {positions.shape}.")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotations shape (T, 3, 3), got {rotations.shape}.")
    if valid.ndim != 1:
        raise ValueError(f"Expected valid shape (T,), got {valid.shape}.")
    if timestamps_us.ndim != 1:
        raise ValueError(f"Expected timestamps shape (T,), got {timestamps_us.shape}.")
    if not (len(positions) == len(rotations) == len(valid) == len(timestamps_us)):
        raise ValueError(
            "Smoothed target frame counts differ: "
            f"positions={len(positions)}, rotations={len(rotations)}, "
            f"valid={len(valid)}, timestamps={len(timestamps_us)}."
        )

    positions = interpolate_nan_positions(positions)
    if should_apply_legacy_transform(coordinate_frame):
        positions = transform_points(positions)
        rotations = transform_rotations_to_robot(rotations)
    else:
        positions = positions + TARGET_OFFSET - ROBOT_BASE_OFFSET
    wrist_idx = body_wrist_index(working_hand)
    body = np.broadcast_to(positions[:, None, :], (len(positions), 33, 3)).copy()
    body[:, wrist_idx, :] = positions
    finite_rotations = np.isfinite(rotations).all(axis=(1, 2))
    orientation_valid = valid & finite_rotations
    return RetargetInput(
        body=body,
        hand=None,
        fps=estimate_fps_from_timestamps(timestamps_us),
        orientation_valid=orientation_valid,
        target_rotations=rotations,
        timestamps_us=timestamps_us,
        source_format="smoothed_targets",
        coordinate_frame=normalize_coordinate_frame(coordinate_frame),
    )


def load_retarget_input(
    sample_path: Path,
    working_hand: str,
    landmark_sg_window: int,
    landmark_sg_polyorder: int,
    limit_frames: int | None,
) -> RetargetInput:
    """Load pre-computed smoothed end-effector targets.

    The retarget stage only consumes end-effector trajectories; deriving poses
    from raw landmarks happens earlier, at the smooth stage.
    """
    return load_smoothed_targets(sample_path, working_hand, limit_frames)


def body_wrist_index(working_hand: str) -> int:
    return RIGHT_BODY_WRIST if working_hand == "right" else LEFT_BODY_WRIST


def fill_invalid_rotations(rotations: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.array([rotation is not None for rotation in rotations], dtype=bool)
    if not valid.any():
        raise ValueError(
            "target_mode=hand_se3 requires at least one valid hand orientation "
            "from landmarks 0 (wrist), 1 (thumb CMC), and 9 (middle MCP)."
        )

    from scipy.spatial.transform import Rotation, Slerp

    valid_indices = np.flatnonzero(valid)
    valid_rotations = np.stack(
        [np.asarray(rotations[int(idx)], dtype=np.float64) for idx in valid_indices]
    )
    filled = np.empty((len(rotations), 3, 3), dtype=np.float64)
    if len(valid_indices) == 1:
        filled[:] = valid_rotations[0]
        return filled, valid

    first = int(valid_indices[0])
    last = int(valid_indices[-1])
    filled[:first] = valid_rotations[0]
    filled[last + 1 :] = valid_rotations[-1]
    interpolation_frames = np.arange(first, last + 1, dtype=np.float64)
    filled[first : last + 1] = Slerp(
        valid_indices.astype(np.float64),
        Rotation.from_matrix(valid_rotations),
    )(interpolation_frames).as_matrix()
    return filled, valid


def extract_se3(pin: Any, body_frame: np.ndarray, rotation: np.ndarray, wrist_body_idx: int) -> Any:
    """Build a hand target SE3 from body wrist translation and hand orientation."""
    p = np.asarray(body_frame[wrist_body_idx], dtype=np.float64)
    return pin.SE3(np.asarray(rotation, dtype=np.float64), p)


def align_rotations_to_initial(rotations: np.ndarray, initial_target_rotation: np.ndarray) -> np.ndarray:
    """Map frame-0 hand rotation onto a desired robot tool rotation."""
    arr = np.asarray(rotations, dtype=np.float64)
    initial = np.asarray(initial_target_rotation, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotations shape (T, 3, 3), got {arr.shape}.")
    if initial.shape != (3, 3):
        raise ValueError(f"Expected initial target rotation shape (3, 3), got {initial.shape}.")
    offset = arr[0].T @ initial
    return np.stack([rotation @ offset for rotation in arr], axis=0)


def build_direct_rotation_targets(
    pin: Any,
    body: np.ndarray,
    rotations: np.ndarray,
    working_hand: str,
    orientation_valid_hint: np.ndarray | None = None,
    initial_target_rotation: np.ndarray | None = None,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    wrist_idx = body_wrist_index(working_hand)
    arr = np.asarray(rotations, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected direct target rotations shape (T, 3, 3), got {arr.shape}.")
    if len(arr) != len(body):
        raise ValueError(f"Body/rotation frame counts differ: body={len(body)}, rotations={len(arr)}.")
    valid_hint = np.isfinite(arr).all(axis=(1, 2))
    if orientation_valid_hint is not None:
        hint = np.asarray(orientation_valid_hint, dtype=bool)
        if len(hint) != len(arr):
            raise ValueError(
                "orientation_valid length does not match rotations: "
                f"{len(hint)} != {len(arr)}."
            )
        valid_hint &= hint
    rotation_items = [arr[t] if valid_hint[t] else None for t in range(len(arr))]
    filled, valid = fill_invalid_rotations(rotation_items)
    if initial_target_rotation is not None:
        filled = align_rotations_to_initial(filled, initial_target_rotation)
    targets = [extract_se3(pin, body[t], filled[t], wrist_idx) for t in range(len(body))]
    return targets, valid, filled


def build_wrist_position_targets(pin: Any, body: np.ndarray, working_hand: str) -> list[Any]:
    wrist_idx = body_wrist_index(working_hand)
    identity = np.eye(3, dtype=np.float64)
    return [pin.SE3(identity, np.asarray(body[t, wrist_idx], dtype=np.float64)) for t in range(len(body))]


def effective_orientation_cost(cfg: RunConfig) -> float:
    """Orientation is intentionally disabled for wrist-position-only targets."""
    return 0.0 if cfg.target_mode == "wrist_position" else cfg.ik_orientation_cost


def neutral_ee_position(pin: Any, robot: Any, ee_frame: str) -> np.ndarray:
    """Return the end-effector position at the robot neutral configuration."""
    frame_id = robot.model.getFrameId(ee_frame)
    q0 = pin.neutral(robot.model)
    pin.forwardKinematics(robot.model, robot.data, q0)
    pin.updateFramePlacements(robot.model, robot.data)
    return np.asarray(robot.data.oMf[frame_id].translation, dtype=np.float64)


def neutral_ee_rotation(pin: Any, robot: Any, ee_frame: str) -> np.ndarray:
    """Return the end-effector rotation at the robot neutral configuration."""
    frame_id = robot.model.getFrameId(ee_frame)
    q0 = pin.neutral(robot.model)
    pin.forwardKinematics(robot.model, robot.data, q0)
    pin.updateFramePlacements(robot.model, robot.data)
    return np.asarray(robot.data.oMf[frame_id].rotation, dtype=np.float64)


def run_approach(
    pin: Any,
    Configuration: Any,
    solve_ik: Any,
    FrameTask: Any,
    PostureTask: Any,
    robot: Any,
    ee_frame: str,
    first_target: Any,
    cfg: RunConfig,
    fps: float,
) -> np.ndarray:
    """Move from neutral to the first target before scene tracking."""
    q0 = pin.neutral(robot.model)
    configuration = Configuration(robot.model, robot.data, q0)
    frame_task = FrameTask(ee_frame, position_cost=cfg.ik_position_cost, orientation_cost=effective_orientation_cost(cfg))
    posture_task = PostureTask(cost=cfg.ik_posture_cost)
    posture_task.set_target_from_configuration(configuration)

    frames = max(1, int(round(cfg.approach_sec * fps)))
    dt = 1.0 / max(fps, 1e-9) / max(cfg.ik_substeps, 1)
    q_traj = np.zeros((frames, robot.model.nq), dtype=np.float64)
    for i in range(frames):
        alpha = (i + 1) / frames
        target = pin.SE3(first_target.rotation, alpha * first_target.translation)
        frame_task.set_target(target)
        for _ in range(cfg.ik_substeps):
            velocity = solve_ik(configuration, [frame_task, posture_task], dt, solver=cfg.ik_solver)
            configuration.integrate_inplace(velocity, dt)
        q_traj[i] = configuration.q
    return q_traj


def run_scene_ik(
    Configuration: Any,
    solve_ik: Any,
    FrameTask: Any,
    PostureTask: Any,
    robot: Any,
    ee_frame: str,
    targets: list[Any],
    q_start: np.ndarray,
    cfg: RunConfig,
    fps: float,
) -> np.ndarray:
    """Track all scene targets with differential IK."""
    configuration = Configuration(robot.model, robot.data, np.asarray(q_start, dtype=np.float64).copy())
    frame_task = FrameTask(ee_frame, position_cost=cfg.ik_position_cost, orientation_cost=effective_orientation_cost(cfg))
    posture_task = PostureTask(cost=cfg.ik_posture_cost)
    posture_task.set_target_from_configuration(configuration)

    dt = 1.0 / max(fps, 1e-9) / max(cfg.ik_substeps, 1)
    q_traj = np.zeros((len(targets), robot.model.nq), dtype=np.float64)
    for i, target in enumerate(targets):
        frame_task.set_target(target)
        for _ in range(cfg.ik_substeps):
            velocity = solve_ik(configuration, [frame_task, posture_task], dt, solver=cfg.ik_solver)
            configuration.integrate_inplace(velocity, dt)
        q_traj[i] = configuration.q
    return q_traj


def smooth_joint_trajectory(q_scene: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if window <= 0:
        return np.asarray(q_scene, dtype=np.float64).copy()
    adjusted = adjusted_savgol_window(len(q_scene), window, polyorder)
    return smooth_savgol(q_scene, window=adjusted, polyorder=polyorder, axis=0)


def compute_tracking_error(pin: Any, robot: Any, ee_frame: str, q_traj: np.ndarray, targets: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    frame_id = robot.model.getFrameId(ee_frame)
    pos_err = np.zeros(len(q_traj), dtype=np.float64)
    ori_err = np.zeros(len(q_traj), dtype=np.float64)
    for i, q in enumerate(q_traj):
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        ee_pose = robot.data.oMf[frame_id]
        delta = targets[i].actInv(ee_pose)
        pos_err[i] = float(np.linalg.norm(ee_pose.translation - targets[i].translation))
        ori_err[i] = float(np.linalg.norm(pin.log3(delta.rotation)))
    return pos_err, ori_err


def _build_targets(
    pin: Any,
    robot: Any,
    positions: np.ndarray,
    rotations: np.ndarray | None,
    orientation_valid: np.ndarray | None,
    cfg: RunConfig,
) -> tuple[list[Any], np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    """Convert end-effector positions/rotations into PINK SE3 targets."""
    T = len(positions)
    wrist_idx = body_wrist_index(cfg.working_hand)
    body = np.broadcast_to(positions[:, None, :], (T, 33, 3)).copy()
    body[:, wrist_idx, :] = positions

    if rotations is None or cfg.target_mode == "wrist_position":
        targets = build_wrist_position_targets(pin, body, cfg.working_hand)
        target_rot = None
        orientation_valid_out = None
        target_mode = "wrist_position"
    else:
        initial_target_rotation = None
        if cfg.align_initial_orientation:
            initial_target_rotation = neutral_ee_rotation(pin, robot, cfg.robot.ee_frame)
        targets, orientation_valid_out, target_rot = build_direct_rotation_targets(
            pin,
            body,
            rotations,
            cfg.working_hand,
            orientation_valid,
            initial_target_rotation,
        )
        target_mode = "hand_se3"

    target_pos = np.vstack([target.translation for target in targets])
    return targets, target_pos, target_rot, orientation_valid_out, target_mode


def _run_ik_and_write(
    deps: tuple,
    targets: list[Any],
    target_pos: np.ndarray,
    target_rot: np.ndarray | None,
    orientation_valid: np.ndarray | None,
    fps: float,
    out_path: Path,
    cfg: RunConfig,
    target_mode: str,
    scale_origin: str,
    recenter_offset: np.ndarray,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    """Run approach + scene IK, smooth, score, and write the trajectory archive."""
    pin, _pink, Configuration, solve_ik, FrameTask, PostureTask, load_robot_description = deps
    robot = load_robot_description(cfg.robot.description)
    if robot.model.getFrameId(cfg.robot.ee_frame) >= len(robot.model.frames):
        raise ValueError(f"End-effector frame '{cfg.robot.ee_frame}' not found in {cfg.robot.description}.")

    q_approach = run_approach(
        pin, Configuration, solve_ik, FrameTask, PostureTask, robot, cfg.robot.ee_frame, targets[0], cfg, fps,
    )
    q_scene_raw = run_scene_ik(
        Configuration, solve_ik, FrameTask, PostureTask, robot, cfg.robot.ee_frame, targets, q_approach[-1], cfg, fps,
    )
    q_scene_smooth = smooth_joint_trajectory(q_scene_raw, cfg.joint_sg_window, cfg.joint_sg_polyorder)
    pos_err_smooth, ori_err_smooth = compute_tracking_error(pin, robot, cfg.robot.ee_frame, q_scene_smooth, targets)
    ori_err_smooth_deg = np.degrees(ori_err_smooth)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    archive = {
        "q_approach": q_approach,
        "q_scene_raw": q_scene_raw,
        "q_scene_smooth": q_scene_smooth,
        "ee_target_pos": target_pos,
        "pos_err_smooth": pos_err_smooth,
        "ori_err_smooth": ori_err_smooth,
        "ori_err_smooth_deg": ori_err_smooth_deg,
        "fps": float(fps),
        "dt": float(1.0 / max(fps, 1e-9)),
        "robot": cfg.robot.description,
        "ee_frame": cfg.robot.ee_frame,
        "working_hand": cfg.working_hand,
        "target_mode": target_mode,
        "joint_sg_window": int(cfg.joint_sg_window),
        "joint_sg_polyorder": int(cfg.joint_sg_polyorder),
        "ik_position_cost": float(cfg.ik_position_cost),
        "ik_orientation_cost": float(cfg.ik_orientation_cost),
        "effective_orientation_cost": float(effective_orientation_cost(cfg)),
        "ik_posture_cost": float(cfg.ik_posture_cost),
        "ik_substeps": int(cfg.ik_substeps),
        "ik_solver": cfg.ik_solver,
        "recenter_to_neutral": bool(cfg.recenter_to_neutral),
        "recenter_offset": recenter_offset,
        "trajectory_scale": float(cfg.trajectory_scale),
        "trajectory_scale_origin": scale_origin,
        "base_offset": np.array(cfg.base_offset, dtype=np.float64),
        "target_offset": np.array(cfg.target_offset, dtype=np.float64),
        **source_meta,
    }
    if target_rot is not None:
        archive["ee_target_rot"] = target_rot
    if orientation_valid is not None:
        archive["orientation_valid"] = orientation_valid
    write_hdf5_archive(out_path, archive)

    summary = {
        "traj_path": str(out_path),
        "robot": cfg.robot.description,
        "ee_frame": cfg.robot.ee_frame,
        "frames": int(len(q_scene_smooth)),
        "fps": float(fps),
        "working_hand": cfg.working_hand,
        "ik_position_cost": float(cfg.ik_position_cost),
        "ik_orientation_cost": float(cfg.ik_orientation_cost),
        "effective_orientation_cost": float(effective_orientation_cost(cfg)),
        "ik_posture_cost": float(cfg.ik_posture_cost),
        "ik_substeps": int(cfg.ik_substeps),
        "ik_solver": cfg.ik_solver,
        "target_mode": target_mode,
        "joint_sg_window": int(cfg.joint_sg_window),
        "recenter_to_neutral": bool(cfg.recenter_to_neutral),
        "recenter_offset": recenter_offset.tolist(),
        "trajectory_scale": float(cfg.trajectory_scale),
        "trajectory_scale_origin": scale_origin,
        "mean_not_aligned_pos_error_mm": float(1000.0 * np.mean(pos_err_smooth)),
        "median_not_aligned_pos_error_mm": float(1000.0 * np.median(pos_err_smooth)),
        "mean_not_aligned_orientation_error_deg": float(np.mean(ori_err_smooth_deg)),
        "median_not_aligned_orientation_error_deg": float(np.median(ori_err_smooth_deg)),
        "p95_not_aligned_orientation_error_deg": float(np.percentile(ori_err_smooth_deg, 95)),
        "max_not_aligned_orientation_error_deg": float(np.max(ori_err_smooth_deg)),
        **source_meta,
    }
    if orientation_valid is not None:
        summary["orientation_valid_frames"] = int(orientation_valid.sum())
        summary["orientation_total_frames"] = int(len(orientation_valid))
    print(
        f"Saved trajectory: {out_path} "
        f"(mean error={summary['mean_not_aligned_pos_error_mm']:.1f} mm, "
        f"orientation={summary['mean_not_aligned_orientation_error_deg']:.1f} deg)"
    )
    return summary


def _prepare_positions(positions: np.ndarray, cfg: RunConfig, scale_origin: str) -> np.ndarray:
    """Apply trajectory scaling about the resolved origin."""
    if abs(cfg.trajectory_scale - 1.0) > 1e-12:
        if cfg.trajectory_scale <= 0.0:
            raise ValueError("trajectory_scale must be positive.")
        anchor = (
            np.zeros(3, dtype=np.float64)
            if scale_origin == "robot_base"
            else positions[0].copy()
        )
        positions = anchor + (positions - anchor) * cfg.trajectory_scale
    return positions


def retarget(sample_path: Path, out_path: Path, cfg: RunConfig) -> dict[str, Any]:
    """Run one retargeting job from a pre-computed smoothed target archive."""
    deps = require_ik_dependencies()
    pin, _pink, Configuration, solve_ik, FrameTask, PostureTask, load_robot_description = deps

    retarget_input = load_smoothed_targets(sample_path, cfg.working_hand, cfg.limit_frames)
    positions = retarget_input.body[:, body_wrist_index(cfg.working_hand)]
    rotations = retarget_input.target_rotations
    orientation_valid = retarget_input.orientation_valid
    fps = retarget_input.fps

    scale_origin = resolve_trajectory_scale_origin(cfg.trajectory_scale_origin, retarget_input.coordinate_frame)
    positions = _prepare_positions(positions, cfg, scale_origin)

    recenter_offset = np.zeros(3, dtype=np.float64)
    robot = load_robot_description(cfg.robot.description)
    if cfg.recenter_to_neutral:
        offset = neutral_ee_position(pin, robot, cfg.robot.ee_frame) - positions[0]
        positions = positions + offset
        recenter_offset = offset

    targets, target_pos, target_rot, orientation_valid_out, target_mode = _build_targets(
        pin, robot, positions, rotations, orientation_valid, cfg,
    )

    source_meta: dict[str, Any] = {
        "source_format": retarget_input.source_format,
        "source_coordinate_frame": retarget_input.coordinate_frame,
        "source_npz": str(sample_path),
    }
    if retarget_input.timestamps_us is not None:
        source_meta["timestamps_us"] = retarget_input.timestamps_us

    return _run_ik_and_write(
        deps, targets, target_pos, target_rot, orientation_valid_out, fps, out_path, cfg,
        target_mode, scale_origin, recenter_offset, source_meta,
    )


def retarget_from_poses(
    positions: np.ndarray,
    rotations: np.ndarray | None,
    validity: np.ndarray | None,
    fps: float,
    out_path: Path,
    cfg: RunConfig,
) -> dict[str, Any]:
    """Retarget pre-computed wrist positions and palm rotations to a robot trajectory.

    Parameters
    ----------
    positions : (T, 3) wrist positions in world-frame metres.
    rotations : (T, 3, 3) palm rotation matrices, or None for position-only.
    validity  : (T,) bool mask of frames with valid rotations, or None (all valid).
    fps       : video frame rate.
    out_path  : output .h5 path.
    cfg       : run configuration (IK costs, smoothing, recenter, scale, etc.).

    Returns summary dict (same schema as retarget()).
    """
    deps = require_ik_dependencies()
    pin, _pink, Configuration, solve_ik, FrameTask, PostureTask, load_robot_description = deps

    T = len(positions)
    positions = np.asarray(positions, dtype=np.float64)
    if rotations is not None:
        rotations = np.asarray(rotations, dtype=np.float64)
        if rotations.shape != (T, 3, 3):
            raise ValueError(f"Expected rotations shape ({T}, 3, 3), got {rotations.shape}.")

    to = np.array(cfg.target_offset, dtype=np.float64)
    positions = positions + to - ROBOT_BASE_OFFSET

    robot = load_robot_description(cfg.robot.description)
    if robot.model.getFrameId(cfg.robot.ee_frame) >= len(robot.model.frames):
        raise ValueError(f"End-effector frame '{cfg.robot.ee_frame}' not found in {cfg.robot.description}.")

    scale_origin = resolve_trajectory_scale_origin(cfg.trajectory_scale_origin, "robot_base")
    positions = _prepare_positions(positions, cfg, scale_origin)

    recenter_offset = np.zeros(3, dtype=np.float64)
    if cfg.recenter_to_neutral:
        offset = neutral_ee_position(pin, robot, cfg.robot.ee_frame) - positions[0]
        positions = positions + offset
        recenter_offset = offset

    targets, target_pos, target_rot, orientation_valid, target_mode = _build_targets(
        pin, robot, positions, rotations, validity, cfg,
    )

    source_meta: dict[str, Any] = {"source_cln": str(out_path)}
    return _run_ik_and_write(
        deps, targets, target_pos, target_rot, orientation_valid, fps, out_path, cfg,
        target_mode, scale_origin, recenter_offset, source_meta,
    )


def evaluate_saved_traj(sample_path: Path, traj_path: Path, robot: RobotConfig, align: str, out_prefix: Path) -> dict[str, Any]:
    """Run eval_tracking_error.py's evaluator for a saved trajectory."""
    from .eval_tracking_error import evaluate

    args = argparse.Namespace(
        human=str(sample_path),
        robot_traj=str(traj_path),
        robot=robot.description,
        ee_frame=robot.ee_frame,
        q_key="auto",
        target_source="auto",
        hand=None,
        smoothing="none",
        smooth_window=15,
        smooth_polyorder=3,
        align=align,
        threshold_mm=50.0,
        out=str(out_prefix),
    )
    return evaluate(args)
