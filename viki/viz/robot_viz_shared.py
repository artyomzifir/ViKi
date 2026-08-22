"""Shared utilities for robot visualization — coordinate transforms, FK, reach.

Pure functions used by both the standalone script and the server stream generator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from viki.config import (
    MODELS_DIR,
    RETARGET_DEFAULT_ROBOT,
    RETARGET_RECENTER_TO_NEUTRAL,
    RETARGET_TRAJECTORY_SCALE,
    ROBOT_BASE_OFFSET,
    TARGET_OFFSET,
)

EXTRINSICS_FILE = Path("data/extrinsics_calibration.json")
DEBUG_FILE = Path("data/retarget_debug.json")

# Map robot alias → URDF search path relative to MODELS_DIR
ROBOT_URDF_MAP: dict[str, str] = {
    "ur10": "robot_descriptions/xacrodoc/ur10_official_description",
    "iiwa14": "robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf",
}

# Reach radii per robot alias (approximate)
ROBOT_REACH: dict[str, float] = {
    "ur10": 1.3,
    "iiwa14": 0.8,
}


@dataclass
class VizConfig:
    """Configuration for the visualization stream."""

    center_on: str = "world"  # "world" or "robot"
    axes_length: float = 2.0
    show_cameras: bool = True
    show_board: bool = True
    show_neutral_ee: bool = True
    show_human_trail: bool = True
    show_robot_trail: bool = True
    show_base_to_ee: bool = True
    show_debug_overlay: bool = True
    show_reach_sphere: bool = True
    show_fk_arm: bool = True
    show_ee_target: bool = True  # IK target dots from debug


def resolve_robot_alias(robot: str) -> str:
    """Normalise robot name to known alias (ur10 / iiwa14)."""
    r = robot.lower().replace("_description", "").replace(" ", "")
    for alias in ("ur10", "iiwa14"):
        if alias in r:
            return alias
    return robot


def get_reach_radius(robot_alias: str) -> float:
    return ROBOT_REACH.get(resolve_robot_alias(robot_alias), 1.0)


def get_robot_world_pos(p_robot: np.ndarray) -> np.ndarray:
    """Transform robot-frame point to world frame."""
    return p_robot + np.array(ROBOT_BASE_OFFSET, dtype=np.float64)


def to_robot_frame(
    p_world: np.ndarray,
    wrist_0: np.ndarray | None = None,
    p_neutral: np.ndarray | None = None,
) -> np.ndarray:
    """Transform world-frame wrist to robot-frame target (same logic as dataset IK).

    Parameters
    ----------
    p_world : (3,) array — world-frame position.
    wrist_0 : (3,) array or None — first-frame wrist (for recenter).
    p_neutral : (3,) array or None — neutral EE position (for recenter).

    Returns
    -------
    (3,) array — robot-frame target.
    """
    base = np.array(ROBOT_BASE_OFFSET, dtype=np.float64)
    nudge = np.array(TARGET_OFFSET, dtype=np.float64)
    scale = float(RETARGET_TRAJECTORY_SCALE)

    p = p_world + nudge - base

    if RETARGET_RECENTER_TO_NEUTRAL and wrist_0 is not None and p_neutral is not None:
        p_0 = wrist_0 + nudge - base
        p = p + (p_neutral - p_0)

    if abs(scale - 1.0) > 1e-12:
        anchor = wrist_0 + nudge - base
        if RETARGET_RECENTER_TO_NEUTRAL and p_neutral is not None:
            anchor = p_neutral
        p = anchor + (p - anchor) * scale

    return p


def project_to_sphere(
    point: np.ndarray, center: np.ndarray, radius: float
) -> tuple[np.ndarray, bool]:
    """Project *point* onto the surface of a sphere if outside it.

    Returns (projected_point, was_projected).
    """
    vec = point - center
    dist = np.linalg.norm(vec)
    if dist > radius and dist > 1e-12:
        return center + vec / dist * radius, True
    return point, False


def load_extrinsics(path: Path = EXTRINSICS_FILE) -> list[dict[str, Any]] | None:
    """Load extrinsics calibration JSON.

    Returns a list of dicts with keys: device_id, rvec, tvec.
    Non-dict entries are silently skipped.
    """
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return None
    return [e for e in raw if isinstance(e, dict)]


def load_debug_data(path: Path = DEBUG_FILE) -> dict[str, Any] | None:
    """Load retarget debug JSON, or None."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_skeleton_wrist(path: Path) -> tuple[np.ndarray, int]:
    """Load wrist positions from a cleaned skeleton .npz file.

    Returns (wrist_positions, frame_count).
    """
    with np.load(path, allow_pickle=True) as data:
        if "positions" in data.files:
            positions = np.asarray(data["positions"], dtype=np.float64)
        else:
            points = data["points"] if "points" in data.files else data["body"]
            wrist_idx = 16 if "right_hand" in data.files else 15
            positions = np.asarray(points[:, wrist_idx, :], dtype=np.float64)
    return positions, len(positions)


def find_urdf(robot_alias: str) -> Path | None:
    """Resolve robot alias to a URDF file path under MODELS_DIR."""
    alias = resolve_robot_alias(robot_alias)
    search_path = ROBOT_URDF_MAP.get(alias)
    if not search_path:
        return None
    full = Path(MODELS_DIR) / search_path
    if full.is_dir():
        files = [f for f in os.listdir(full) if f.endswith(".urdf")]
        if not files:
            return None
        return full / files[0]
    if full.exists():
        return full
    return None


def get_neutral_ee(robot_alias: str) -> np.ndarray:
    """Compute neutral end-effector position via Pinocchio FK.

    Returns a (3,) world-frame position.
    """
    try:
        import pinocchio as pin
    except ImportError:
        return np.array([0.4, 0.0, 0.4])

    urdf = find_urdf(robot_alias)
    if urdf is None:
        return np.array([0.4, 0.0, 0.4])

    try:
        model = pin.buildModelFromUrdf(str(urdf))
        data = pin.Data(model)
        aa = resolve_robot_alias(robot_alias)
        ee_frame = "tool0" if aa == "ur10" else "iiwa_link_ee"
        frame_id = model.getFrameId(ee_frame)

        if aa == "ur10":
            q0 = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
        else:
            q0 = pin.neutral(model)

        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)
        return np.asarray(data.oMf[frame_id].translation, dtype=np.float64)
    except Exception:
        return np.array([0.4, 0.0, 0.4])


def fk_positions(
    model: Any, data: Any, q_all: np.ndarray, ee_frame: str
) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics for a sequence of joint configurations.

    Returns (all_link_positions, ee_positions).
    """
    import pinocchio as pin

    n_frames = q_all.shape[0]
    all_positions = []
    ee_positions = np.zeros((n_frames, 3), dtype=np.float64)
    for i in range(n_frames):
        pin.forwardKinematics(model, data, q_all[i])
        pin.updateFramePlacements(model, data)
        frame_pos = []
        for name in model.names:
            if name == "universe":
                continue
            fid = model.getFrameId(name)
            frame_pos.append(data.oMf[fid].translation.copy())
        ee_fid = model.getFrameId(ee_frame)
        ee_positions[i] = data.oMf[ee_fid].translation.copy()
        frame_pos.append(ee_positions[i])
        all_positions.append(np.array(frame_pos))
    return np.array(all_positions), ee_positions


def camera_world_pos(rvec: list[float], tvec: list[float]) -> np.ndarray:
    """Compute camera world position from Rodrigues rvec + tvec."""
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    t = np.array(tvec, dtype=np.float32)
    return (-R.T @ t).flatten()


def camera_gaze_dir(rvec: list[float]) -> np.ndarray:
    """Compute camera gaze direction (unit vector) in world frame."""
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    return (R.T @ np.array([0, 0, 1], dtype=np.float32)).flatten()
