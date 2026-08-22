"""Offline FK tracking-error evaluation for ViKi experiment trajectories.

This script evaluates saved robot joint trajectories against the human target
trajectory used for retargeting. It is intentionally RGB-only for now: the
default target source is ``ee_target_pos`` from the retargeting archive, while
``body_wrist`` is available for later depth-fused or calibrated landmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .archive_io import load_archive, write_hdf5_archive
from .smoothing import smooth_trajectory


RIGHT_WRIST = 16
LEFT_WRIST = 15
Q_KEY_PRIORITY = ("q_scene_smooth", "q_scene_raw", "q_approach")
TARGET_ROTATION_KEYS = (
    "ee_target_rot",
    "ee_target_rotation",
    "ee_target_rotmat",
    "target_rot",
    "target_rotation",
)
REQUIRED_PINOCCHIO_ATTRS = (
    "SE3",
    "JointModelRX",
    "forwardKinematics",
    "updateFramePlacements",
)


@dataclass(frozen=True)
class RobotDefaults:
    description: str
    ee_frame: str


ROBOT_DEFAULTS = {
    "iiwa": RobotDefaults("iiwa14_description", "iiwa_link_ee"),
    "iiwa14": RobotDefaults("iiwa14_description", "iiwa_link_ee"),
    "iiwa14_description": RobotDefaults("iiwa14_description", "iiwa_link_ee"),
    "ur10": RobotDefaults("ur10_official_description", "tool0"),
    "ur10_description": RobotDefaults("ur10_official_description", "tool0"),
    "ur10_official": RobotDefaults("ur10_official_description", "tool0"),
    "ur10_official_description": RobotDefaults("ur10_official_description", "tool0"),
}


def npz_scalar(value: Any, default: Any = None) -> Any:
    """Return a Python scalar from a 0-D npz value."""
    if value is None:
        return default
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def json_safe_npz_value(value: Any) -> Any:
    """Convert npz metadata values to JSON-safe Python values."""
    value = npz_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_robot_name(robot: str | None, q_dim: int | None = None) -> RobotDefaults:
    """Resolve CLI or archive robot names to robot_descriptions metadata."""
    if robot:
        key = str(robot).strip()
        if key in ROBOT_DEFAULTS:
            return ROBOT_DEFAULTS[key]
        raise ValueError(
            f"Unknown robot '{robot}'. Expected one of: "
            f"{', '.join(sorted(ROBOT_DEFAULTS))}."
        )

    if q_dim == 7:
        return ROBOT_DEFAULTS["iiwa14"]
    if q_dim == 6:
        return ROBOT_DEFAULTS["ur10"]
    raise ValueError("Cannot infer robot from trajectory shape; pass --robot.")


def select_q_key(robot_npz: np.lib.npyio.NpzFile, requested: str = "auto") -> str:
    """Select a joint trajectory key from a robot archive."""
    if requested != "auto":
        if requested not in robot_npz.files:
            raise KeyError(
                f"Robot trajectory archive does not contain q-key '{requested}'. "
                f"Available keys: {', '.join(robot_npz.files)}"
            )
        return requested

    for key in Q_KEY_PRIORITY:
        if key in robot_npz.files:
            return key
    raise KeyError(
        "Robot trajectory archive does not contain any supported q trajectory key. "
        f"Expected one of: {', '.join(Q_KEY_PRIORITY)}. "
        f"Available keys: {', '.join(robot_npz.files)}"
    )


def select_target_source(robot_npz: np.lib.npyio.NpzFile, requested: str = "auto") -> str:
    """Select the human target source."""
    if requested != "auto":
        return requested
    return "ee_target_pos" if "ee_target_pos" in robot_npz.files else "body_wrist"


def resample_trajectory(points: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (T, D) trajectory to target_len frames."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D trajectory, got shape {arr.shape}.")
    if target_len <= 0:
        raise ValueError("target_len must be positive.")
    if len(arr) == target_len:
        return arr.copy()
    if len(arr) == 0:
        raise ValueError("Cannot resample an empty trajectory.")
    if len(arr) == 1:
        return np.repeat(arr, target_len, axis=0)

    src_t = np.linspace(0.0, 1.0, len(arr))
    dst_t = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, arr.shape[1]), dtype=np.float64)
    for dim in range(arr.shape[1]):
        out[:, dim] = np.interp(dst_t, src_t, arr[:, dim])
    return out


def rigid_align_points(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align source points to target points with Kabsch rotation + translation.

    No scale is estimated or applied.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"Alignment shapes differ: source={src.shape}, target={dst.shape}.")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected (T, 3) point arrays, got {src.shape}.")
    if len(src) < 3:
        raise ValueError("Rigid alignment needs at least 3 points.")

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    covariance = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation = dst_centroid - src_centroid @ rotation.T
    aligned = src @ rotation.T + translation
    return aligned, rotation, translation


def compute_error_mm(robot_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    """Return per-frame Euclidean position error in millimetres."""
    robot = np.asarray(robot_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if robot.shape != target.shape:
        raise ValueError(f"Error shapes differ: robot={robot.shape}, target={target.shape}.")
    return 1000.0 * np.linalg.norm(robot - target, axis=1)


def compute_error_metrics(error_mm: np.ndarray, threshold_mm: float = 50.0) -> dict[str, Any]:
    """Compute summary metrics for a position-error vector."""
    error = np.asarray(error_mm, dtype=np.float64)
    if error.ndim != 1 or len(error) == 0:
        raise ValueError("error_mm must be a non-empty 1-D array.")
    return {
        "mean_error_mm": float(np.mean(error)),
        "median_error_mm": float(np.median(error)),
        "p95_error_mm": float(np.percentile(error, 95)),
        "max_error_mm": float(np.max(error)),
        "frames_under_50mm_pct": float(np.mean(error < threshold_mm) * 100.0),
        "num_frames": int(len(error)),
        "threshold_mm": float(threshold_mm),
    }


def project_rotation_matrices(rotations: np.ndarray) -> np.ndarray:
    """Project matrices onto SO(3) with SVD for numeric robustness."""
    arr = np.asarray(rotations, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotation matrices shape (T, 3, 3), got {arr.shape}.")

    projected = np.empty_like(arr)
    for i, rotation in enumerate(arr):
        u, _, vt = np.linalg.svd(rotation)
        clean = u @ vt
        if np.linalg.det(clean) < 0:
            u[:, -1] *= -1.0
            clean = u @ vt
        projected[i] = clean
    return projected


def resample_rotations_nearest(rotations: np.ndarray, target_len: int) -> np.ndarray:
    """Resample rotation matrices with nearest-neighbour frame selection."""
    arr = project_rotation_matrices(rotations)
    if target_len <= 0:
        raise ValueError("target_len must be positive.")
    if len(arr) == target_len:
        return arr.copy()
    if len(arr) == 0:
        raise ValueError("Cannot resample empty rotations.")
    indices = np.rint(np.linspace(0, len(arr) - 1, target_len)).astype(np.int64)
    return arr[indices].copy()


def apply_alignment_to_rotations(rotations: np.ndarray, alignment_rotation: np.ndarray) -> np.ndarray:
    """Apply the same rigid alignment rotation used for target positions."""
    clean = project_rotation_matrices(rotations)
    aligned = np.einsum("ij,tjk->tik", alignment_rotation, clean)
    return project_rotation_matrices(aligned)


def compute_orientation_error_deg(robot_rotations: np.ndarray, target_rotations: np.ndarray) -> np.ndarray:
    """Return per-frame angular orientation error in degrees."""
    robot = project_rotation_matrices(robot_rotations)
    target = project_rotation_matrices(target_rotations)
    if robot.shape != target.shape:
        raise ValueError(f"Orientation shapes differ: robot={robot.shape}, target={target.shape}.")

    relative = np.einsum("tji,tjk->tik", target, robot)
    traces = np.trace(relative, axis1=1, axis2=2)
    cos_theta = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_orientation_metrics(error_deg: np.ndarray) -> dict[str, Any]:
    """Compute summary metrics for an angular orientation-error vector."""
    error = np.asarray(error_deg, dtype=np.float64)
    if error.ndim != 1 or len(error) == 0:
        raise ValueError("orientation error must be a non-empty 1-D array.")
    return {
        "mean_orientation_error_deg": float(np.mean(error)),
        "median_orientation_error_deg": float(np.median(error)),
        "p95_orientation_error_deg": float(np.percentile(error, 95)),
        "max_orientation_error_deg": float(np.max(error)),
    }


def pinocchio_runtime_error(pin: Any, missing: list[str]) -> RuntimeError:
    """Build a diagnostic error for wrong or incompatible Pinocchio imports."""
    pin_file = getattr(pin, "__file__", "<unknown>")
    pin_version = getattr(pin, "__version__", "<unknown>")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return RuntimeError(
        "The imported 'pinocchio' module is not the robotics Pinocchio runtime "
        "needed by robot_descriptions. "
        f"Python: {py_version}; imported from: {pin_file}; version: {pin_version}; "
        f"missing attributes: {', '.join(missing)}. "
        "This commonly means the unrelated PyPI package named 'pinocchio' is "
        "shadowing the robotics package. In the same Python environment, remove "
        "the wrong package. On Windows, use conda-forge/Pixi, WSL, or a Linux "
        "Docker container for robotics Pinocchio; the PyPI `pin` package is "
        "Linux-only for pip wheels and Windows pip may try to compile "
        "Boost/Pinocchio from source."
    )


def import_fk_dependencies():
    """Import and validate the FK dependency stack."""
    try:
        import pinocchio as pin
        from robot_descriptions.loaders.pinocchio import load_robot_description as _load
    except ImportError as exc:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "Pinocchio FK dependencies are not installed. Install robotics "
            "Pinocchio and robot_descriptions in the Python environment used to "
            "run this script, or run inside the retargeting research environment. "
            f"Current Python is {py_version}. On Windows, use conda-forge/Pixi, "
            "WSL, or a Linux Docker container; the PyPI `pin` package is "
            "Linux-only for pip wheels and Windows pip may try to compile "
            "Boost/Pinocchio from source."
        ) from exc

    def _cached_load(description: str):
        os.environ.setdefault(
            "ROBOT_DESCRIPTIONS_CACHE",
            "/app/models/robot_descriptions",
        )
        return _load(description)

    missing = [name for name in REQUIRED_PINOCCHIO_ATTRS if not hasattr(pin, name)]
    if missing:
        raise pinocchio_runtime_error(pin, missing)
    return pin, _cached_load


def load_robot_poses(
    robot_description: str,
    q_traj: np.ndarray,
    ee_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FK end-effector positions and rotations from a joint trajectory."""
    pin, load_robot_description = import_fk_dependencies()
    try:
        robot = load_robot_description(robot_description)
    except AttributeError as exc:
        missing = []
        attr = str(exc).split("has no attribute")[-1].strip().strip("'\"")
        if attr:
            missing.append(attr)
        raise pinocchio_runtime_error(pin, missing or ["<unknown>"]) from exc

    q = np.asarray(q_traj, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"Expected q trajectory shape (T, nq), got {q.shape}.")
    if q.shape[1] != robot.model.nq:
        raise ValueError(
            f"Joint dimension mismatch for {robot_description}: "
            f"trajectory nq={q.shape[1]}, model nq={robot.model.nq}."
        )

    frame_id = robot.model.getFrameId(ee_frame)
    if frame_id >= len(robot.model.frames):
        raise ValueError(f"End-effector frame '{ee_frame}' not found in {robot_description}.")

    positions = np.zeros((len(q), 3), dtype=np.float64)
    rotations = np.zeros((len(q), 3, 3), dtype=np.float64)
    for i, q_row in enumerate(q):
        pin.forwardKinematics(robot.model, robot.data, q_row)
        pin.updateFramePlacements(robot.model, robot.data)
        pose = robot.data.oMf[frame_id]
        positions[i] = pose.translation
        rotations[i] = pose.rotation
    return positions, rotations


def load_robot_positions(
    robot_description: str,
    q_traj: np.ndarray,
    ee_frame: str,
) -> np.ndarray:
    """Compute FK end-effector positions from a joint trajectory."""
    positions, _ = load_robot_poses(robot_description, q_traj, ee_frame)
    return positions


def load_human_target(
    human_npz: np.lib.npyio.NpzFile,
    robot_npz: np.lib.npyio.NpzFile,
    target_source: str,
    hand: str,
) -> np.ndarray:
    """Load the human target trajectory in metres."""
    if target_source == "ee_target_pos":
        if "ee_target_pos" not in robot_npz.files:
            raise KeyError("Robot trajectory archive does not contain 'ee_target_pos'.")
        target = robot_npz["ee_target_pos"]
    elif target_source == "body_wrist":
        if "body" not in human_npz.files:
            raise KeyError("Human archive does not contain 'body'.")
        wrist_idx = RIGHT_WRIST if hand == "right" else LEFT_WRIST
        target = human_npz["body"][:, wrist_idx, :]
    else:
        raise ValueError(f"Unknown target source: {target_source}")

    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"Expected target shape (T, 3), got {target.shape}.")
    return target


def load_target_rotations(robot_npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray | None, str | None]:
    """Load optional target end-effector rotation matrices from a robot archive."""
    for key in TARGET_ROTATION_KEYS:
        if key not in robot_npz.files:
            continue
        rotations = np.asarray(robot_npz[key], dtype=np.float64)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(f"Expected '{key}' shape (T, 3, 3), got {rotations.shape}.")
        return project_rotation_matrices(rotations), key
    return None, None


def make_time_axis(num_frames: int, fps: float | None) -> np.ndarray:
    """Return seconds if fps is valid, otherwise frame indices."""
    if fps and fps > 0:
        return np.arange(num_frames, dtype=np.float64) / float(fps)
    return np.arange(num_frames, dtype=np.float64)


def require_matplotlib():
    """Import matplotlib lazily with a direct error message."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to save evaluation plots. Install matplotlib "
            "or run in the retargeting research environment."
        ) from exc
    return plt


def set_axes_equal(ax: Any, points: np.ndarray) -> None:
    """Set equal scale on all 3D axes."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 0.1
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_error_plot(
    out_path: Path,
    time_axis: np.ndarray,
    error_mm: np.ndarray,
    metrics: dict[str, Any],
    title: str,
    x_label: str,
) -> None:
    """Save the tracking-error plot."""
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_axis, error_mm, color="#e74c3c", lw=1.5, label="position error")
    ax.axhline(
        metrics["threshold_mm"],
        color="#c0392b",
        lw=1.2,
        ls="--",
        label=f"{metrics['threshold_mm']:g} mm threshold",
    )
    ax.axhline(metrics["mean_error_mm"], color="#7f8c8d", lw=1.0, ls=":", label="mean")
    ax.set_title(
        f"{title}\n"
        f"mean={metrics['mean_error_mm']:.1f} mm, "
        f"median={metrics['median_error_mm']:.1f} mm, "
        f"p95={metrics['p95_error_mm']:.1f} mm"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Position error [mm]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_orientation_error_plot(
    out_path: Path,
    time_axis: np.ndarray,
    error_deg: np.ndarray,
    metrics: dict[str, Any],
    title: str,
    x_label: str,
) -> None:
    """Save the angular orientation-error plot."""
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_axis, error_deg, color="#8e44ad", lw=1.5, label="orientation error")
    ax.axhline(
        metrics["mean_orientation_error_deg"],
        color="#7f8c8d",
        lw=1.0,
        ls=":",
        label="mean",
    )
    ax.set_title(
        f"{title}\n"
        f"mean={metrics['mean_orientation_error_deg']:.1f} deg, "
        f"median={metrics['median_orientation_error_deg']:.1f} deg, "
        f"p95={metrics['p95_orientation_error_deg']:.1f} deg"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Orientation error [deg]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_plot(
    out_path: Path,
    robot_positions: np.ndarray,
    target_positions: np.ndarray,
    title: str,
) -> None:
    """Save a 3D robot-vs-target trajectory plot."""
    plt = require_matplotlib()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        target_positions[:, 0],
        target_positions[:, 1],
        target_positions[:, 2],
        color="#2ecc71",
        lw=1.8,
        label="human target (aligned)",
    )
    ax.plot(
        robot_positions[:, 0],
        robot_positions[:, 1],
        robot_positions[:, 2],
        color="#3498db",
        lw=1.5,
        label="robot EE FK",
    )
    ax.scatter(*target_positions[0], color="#27ae60", s=35, label="target start")
    ax.scatter(*target_positions[-1], color="#145a32", s=35, label="target end")
    ax.scatter(*robot_positions[0], color="#2980b9", s=25, label="robot start")
    ax.scatter(*robot_positions[-1], color="#1f618d", s=25, label="robot end")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    all_points = np.vstack([target_positions, robot_positions])
    set_axes_equal(ax, all_points)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def output_paths(prefix: Path) -> dict[str, Path]:
    """Return all output paths from an output prefix."""
    return {
        "error_plot": prefix.with_name(prefix.name + "_error.png"),
        "orientation_error_plot": prefix.with_name(prefix.name + "_orientation_error.png"),
        "trajectory_plot": prefix.with_name(prefix.name + "_trajectory_3d.png"),
        "metrics": prefix.with_name(prefix.name + "_metrics.json"),
        "aligned": prefix.with_name(prefix.name + "_aligned.h5"),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run evaluation and write output artifacts."""
    human_path = Path(args.human)
    robot_path = Path(args.robot_traj)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with load_archive(human_path) as human_npz, load_archive(robot_path) as robot_npz:
        q_key = select_q_key(robot_npz, args.q_key)
        target_source = select_target_source(robot_npz, args.target_source)

        q_traj = np.asarray(robot_npz[q_key], dtype=np.float64)
        robot_from_archive = npz_scalar(robot_npz["robot"], None) if "robot" in robot_npz.files else None
        robot_defaults = normalize_robot_name(args.robot or robot_from_archive, q_dim=q_traj.shape[1])
        ee_frame = args.ee_frame or (
            npz_scalar(robot_npz["ee_frame"], None) if "ee_frame" in robot_npz.files else None
        )
        ee_frame = ee_frame or robot_defaults.ee_frame
        hand = args.hand or (
            npz_scalar(robot_npz["working_hand"], "right") if "working_hand" in robot_npz.files else "right"
        )
        if hand not in {"right", "left"}:
            raise ValueError(f"Unknown hand '{hand}'. Expected 'right' or 'left'.")

        target_raw = load_human_target(human_npz, robot_npz, target_source, hand=hand)
        target_smoothed = smooth_trajectory(
            target_raw,
            method=args.smoothing,
            window=args.smooth_window,
            polyorder=args.smooth_polyorder,
        )
        target_resampled = resample_trajectory(target_smoothed, len(q_traj))
        target_rot_raw, target_rot_key = load_target_rotations(robot_npz)
        target_rot_resampled = (
            resample_rotations_nearest(target_rot_raw, len(q_traj))
            if target_rot_raw is not None
            else None
        )

        if "fps" in robot_npz.files:
            fps = float(npz_scalar(robot_npz["fps"], 0.0))
        elif "fps" in human_npz.files:
            fps = float(npz_scalar(human_npz["fps"], 0.0))
        else:
            fps = 0.0

        trajectory_metadata = {}
        for key in (
            "ik_position_cost",
            "ik_orientation_cost",
            "ik_posture_cost",
            "ik_substeps",
            "ik_solver",
            "joint_sg_window",
            "joint_sg_polyorder",
            "sg_window",
            "sg_polyorder",
            "recenter_to_neutral",
            "recenter_offset",
            "trajectory_scale",
        ):
            if key in robot_npz.files:
                trajectory_metadata[key] = json_safe_npz_value(robot_npz[key])

    robot_positions, robot_rotations = load_robot_poses(robot_defaults.description, q_traj, ee_frame)

    if args.align == "rigid":
        target_aligned, alignment_r, alignment_t = rigid_align_points(target_resampled, robot_positions)
    elif args.align == "none":
        target_aligned = target_resampled.copy()
        alignment_r = np.eye(3, dtype=np.float64)
        alignment_t = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError(f"Unknown alignment mode: {args.align}")

    error_mm = compute_error_mm(robot_positions, target_aligned)
    metrics = compute_error_metrics(error_mm, threshold_mm=args.threshold_mm)
    if target_rot_resampled is not None:
        target_rot_aligned = apply_alignment_to_rotations(target_rot_resampled, alignment_r)
        orientation_error_deg = compute_orientation_error_deg(robot_rotations, target_rot_aligned)
        metrics.update(compute_orientation_metrics(orientation_error_deg))
        orientation_available = True
    else:
        target_rot_aligned = None
        orientation_error_deg = None
        orientation_available = False
    sample_name = human_path.stem
    robot_label = robot_defaults.description.replace("_description", "")
    metrics.update(
        {
            "sample": sample_name,
            "human_path": str(human_path),
            "robot_traj_path": str(robot_path),
            "robot": robot_defaults.description,
            "ee_frame": ee_frame,
            "q_key": q_key,
            "requested_q_key": args.q_key,
            "target_source": target_source,
            "requested_target_source": args.target_source,
            "working_hand": hand,
            "smoothing": args.smoothing,
            "smooth_window": int(args.smooth_window),
            "smooth_polyorder": int(args.smooth_polyorder),
            "align": args.align,
            "fps": float(fps),
            "alignment_rotation": alignment_r.tolist(),
            "alignment_translation_m": alignment_t.tolist(),
            "orientation_available": orientation_available,
            "target_orientation_key": target_rot_key,
            **trajectory_metadata,
        }
    )

    paths = output_paths(out_prefix)
    plot_title = (
        f"{robot_label} tracking error | sample={sample_name} | "
        f"smoothing={args.smoothing} | align={args.align}"
    )
    time_axis = make_time_axis(len(error_mm), fps)
    x_label = "Time [s]" if fps > 0 else "Frame"
    save_error_plot(paths["error_plot"], time_axis, error_mm, metrics, plot_title, x_label)
    if orientation_available and orientation_error_deg is not None:
        save_orientation_error_plot(
            paths["orientation_error_plot"],
            time_axis,
            orientation_error_deg,
            metrics,
            f"{robot_label} orientation tracking error | sample={sample_name} | align={args.align}",
            x_label,
        )
    save_trajectory_plot(
        paths["trajectory_plot"],
        robot_positions,
        target_aligned,
        f"{robot_label} FK trajectory vs human target | {sample_name}",
    )

    with paths["metrics"].open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    write_hdf5_archive(
        paths["aligned"],
        {
            "robot_ee_pos": robot_positions,
            "robot_ee_rot": robot_rotations,
            "target_pos_raw": target_raw,
            "target_pos_resampled": target_resampled,
            "target_pos_aligned": target_aligned,
            "target_rot_raw": target_rot_raw if target_rot_raw is not None else np.empty((0, 3, 3)),
            "target_rot_resampled": target_rot_resampled if target_rot_resampled is not None else np.empty((0, 3, 3)),
            "target_rot_aligned": target_rot_aligned if target_rot_aligned is not None else np.empty((0, 3, 3)),
            "error_mm": error_mm,
            "orientation_error_deg": orientation_error_deg if orientation_error_deg is not None else np.empty((0,)),
            "time_s": time_axis,
            "alignment_rotation": alignment_r,
            "alignment_translation_m": alignment_t,
            "human_path": str(human_path),
            "robot_traj_path": str(robot_path),
            "robot": robot_defaults.description,
            "ee_frame": ee_frame,
            "q_key": q_key,
            "selected_q_key": q_key,
            "requested_q_key": args.q_key,
            "target_source": target_source,
            "requested_target_source": args.target_source,
            "smoothing": args.smoothing,
            "align": args.align,
        },
    )

    print(f"Saved error plot:      {paths['error_plot']}")
    if orientation_available:
        print(f"Saved orientation plot:{paths['orientation_error_plot']}")
    print(f"Saved trajectory plot: {paths['trajectory_plot']}")
    print(f"Saved metrics:         {paths['metrics']}")
    print(f"Saved aligned data:    {paths['aligned']}")
    print(
        "Metrics: "
        f"mean={metrics['mean_error_mm']:.1f} mm, "
        f"median={metrics['median_error_mm']:.1f} mm, "
        f"p95={metrics['p95_error_mm']:.1f} mm, "
        f"under50={metrics['frames_under_50mm_pct']:.1f}%"
    )
    if orientation_available:
        print(
            "Orientation: "
            f"mean={metrics['mean_orientation_error_deg']:.1f} deg, "
            f"median={metrics['median_orientation_error_deg']:.1f} deg, "
            f"p95={metrics['p95_orientation_error_deg']:.1f} deg"
        )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate robot FK tracking error against RGB-derived human targets.",
    )
    parser.add_argument("--human", required=True, help="Path to human landmark .npz.")
    parser.add_argument("--robot-traj", required=True, help="Path to robot trajectory .h5, or legacy .npz.")
    parser.add_argument("--robot", default=None, help="Robot alias/name, e.g. iiwa14 or ur10.")
    parser.add_argument("--ee-frame", default=None, help="Override end-effector frame name.")
    parser.add_argument(
        "--q-key",
        default="auto",
        help="Joint trajectory key in robot archive, or 'auto' for q_scene_smooth/q_scene_raw/q_approach.",
    )
    parser.add_argument(
        "--target-source",
        default="auto",
        choices=["auto", "ee_target_pos", "body_wrist"],
        help="Human target source. auto prefers ee_target_pos when present.",
    )
    parser.add_argument("--hand", default=None, choices=["right", "left"], help="Hand for body_wrist target.")
    parser.add_argument("--smoothing", default="none", choices=["none", "savgol"], help="Target smoothing.")
    parser.add_argument("--smooth-window", type=int, default=15, help="Savitzky-Golay window.")
    parser.add_argument("--smooth-polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
    parser.add_argument("--align", default="rigid", choices=["rigid", "none"], help="Target-to-robot alignment.")
    parser.add_argument("--threshold-mm", type=float, default=50.0, help="Success threshold in millimetres.")
    parser.add_argument("--out", required=True, help="Output prefix, without suffix.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        evaluate(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
