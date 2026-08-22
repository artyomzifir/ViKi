"""
viki.skeleton.hand_angles
-------------------------
World-frame hand pose from a 3-D skeleton frame.

compute_end_effector_pose returns the full world-frame pose of the wrist
end-effector: 3-D position plus a proper rotation matrix
``R_world_palm ∈ SO(3)`` from a palm-attached frame to the world.

Required landmarks:
    WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP  (MCP spread replaces thumb).

compute_palm_rotation is the palm-only rotation sub-routine used by the
optimization module.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from viki.skeleton.models import LM, EndEffectorPose

_MIN_LEN = 1e-6  # zero-length vector threshold

_NAN_VEC3 = np.full(3, np.nan, dtype=np.float32)


def _normalise(v: np.ndarray) -> np.ndarray | None:
    """
    Normalise a vector to unit length.

    Parameters
    ----------
    v : np.ndarray
        Input vector (3,).

    Returns
    -------
    np.ndarray or None
        Normalised vector, or None if norm < _MIN_LEN.
    """
    n = float(np.linalg.norm(v))
    if n < _MIN_LEN:
        return None
    return v / n


# Landmarks required to build the palm frame in the world.
# Uses MCP knuckle spread (INDEX → PINKY) for palm normal instead of THUMB_CMC,
# which moves independently and degenerates when thumb is close to fingers.
_EE_REQUIRED_LM: tuple[LM, ...] = (LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.PINKY_MCP)


def _rot_to_rpy_extrinsic_xyz(R: np.ndarray) -> np.ndarray:
    """
    Extract roll/pitch/yaw (radians) from a rotation matrix.

    Assumes extrinsic XYZ convention: R = Rz(yaw) · Ry(pitch) · Rx(roll).

    Parameters
    ----------
    R : np.ndarray
        (3,3) rotation matrix.

    Returns
    -------
    np.ndarray
        (roll, pitch, yaw) in radians.
    """
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = float(np.arcsin(sy))
    if abs(sy) > 1.0 - 1e-6:
        # Gimbal lock
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def compute_palm_rotation(
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    pinky_mcp: np.ndarray,
) -> np.ndarray | None:
    """
    Compute the 3x3 rotation matrix from palm frame to world.

    Palm frame:
        x = normalise(MIDDLE_MCP - WRIST)
        z = normalise((MIDDLE_MCP - WRIST) × (PINKY_MCP - INDEX_MCP))
        y = z × x

    The palm normal uses the MCP knuckle spread instead of the thumb,
    which is invariant to thumb pose and more stable across hand shapes.

    Parameters
    ----------
    wrist, index_mcp, middle_mcp, pinky_mcp : np.ndarray
        World‑frame positions (3,).

    Returns
    -------
    np.ndarray or None
        (3,3) rotation matrix, or None if any landmark is invalid or degenerate.
    """
    coords = [np.asarray(p, dtype=np.float64) for p in (wrist, index_mcp, middle_mcp, pinky_mcp)]
    if any(not np.all(np.isfinite(p)) for p in coords):
        return None

    fwd = coords[2] - coords[0]               # MIDDLE_MCP - WRIST
    spread = coords[3] - coords[1]             # PINKY_MCP - INDEX_MCP

    x_palm = _normalise(fwd)
    z_palm = _normalise(np.cross(fwd, spread))
    if x_palm is None or z_palm is None:
        return None

    y_palm = np.cross(z_palm, x_palm)
    y_norm = float(np.linalg.norm(y_palm))
    if y_norm < _MIN_LEN:
        return None
    y_palm = y_palm / y_norm

    return np.column_stack([x_palm, y_palm, z_palm]).astype(np.float32)


def _invalid_pose(timestamp_us: int) -> EndEffectorPose:
    """Return an invalid (NaN) EndEffectorPose."""
    return EndEffectorPose(
        position=_NAN_VEC3.copy(),
        R_world_palm=np.full((3, 3), np.nan, dtype=np.float32),
        rpy_deg=_NAN_VEC3.copy(),
        valid=False,
        timestamp_us=timestamp_us,
    )


_PALM_LM: tuple[LM, ...] = (
    LM.WRIST,
    LM.THUMB_CMC,
    LM.INDEX_MCP,
    LM.MIDDLE_MCP,
    LM.RING_MCP,
    LM.PINKY_MCP,
)


def _landmark_centroid(points: Mapping[LM, np.ndarray]) -> np.ndarray | None:
    """Compute the centroid of all finite landmark positions.

    Returns (3,) float64 or None if no finite landmarks exist.
    """
    valid = [
        p for p in (points.get(lm) for lm in _PALM_LM)
        if p is not None and np.all(np.isfinite(p))
    ]
    if not valid:
        return None
    return np.mean(valid, axis=0).astype(np.float64)


def compute_end_effector_pose(
    points: Mapping[LM, np.ndarray],
    timestamp_us: int,
) -> EndEffectorPose:
    """
    Compute the world‑frame pose of the hand from a fused skeleton.

    The primary pose uses the wrist position and palm frame orientation
    (requires WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP).

    **Fallback**: if the wrist is not available (NaN), the centroid of all
    available palm landmarks (WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP,
    RING_MCP, PINKY_MCP) is used as the position, with identity rotation.

    Parameters
    ----------
    points : Mapping[LM, np.ndarray]
        Mapping from LM enum to world‑frame position in metres.
    timestamp_us : int
        Timestamp to embed in the returned pose.

    Returns
    -------
    EndEffectorPose
        Valid pose if at least one palm landmark is finite; otherwise
        invalid with NaNs.
    """
    coords: dict[LM, np.ndarray] = {}
    for lm in _EE_REQUIRED_LM:
        p = points.get(lm)
        if p is None or not np.all(np.isfinite(p)):
            break
        coords[lm] = np.asarray(p, dtype=np.float64)
    else:
        wrist = coords[LM.WRIST]
        fwd = coords[LM.MIDDLE_MCP] - wrist
        spread = coords[LM.PINKY_MCP] - coords[LM.INDEX_MCP]

        x_palm = _normalise(fwd)
        z_palm = _normalise(np.cross(fwd, spread))
        if x_palm is not None and z_palm is not None:
            y_palm = np.cross(z_palm, x_palm)
            y_norm = float(np.linalg.norm(y_palm))
            if y_norm >= _MIN_LEN:
                y_palm = y_palm / y_norm
                R = np.column_stack([x_palm, y_palm, z_palm]).astype(np.float32)
                rpy_rad = _rot_to_rpy_extrinsic_xyz(R.astype(np.float64))
                rpy_deg = np.degrees(rpy_rad).astype(np.float32)
                return EndEffectorPose(
                    position=wrist.astype(np.float32),
                    R_world_palm=R,
                    rpy_deg=rpy_deg,
                    valid=True,
                    timestamp_us=timestamp_us,
                )

    # Fallback: centroid of available palm landmarks.
    centroid = _landmark_centroid(points)
    if centroid is not None:
        return EndEffectorPose(
            position=centroid.astype(np.float32),
            R_world_palm=np.eye(3, dtype=np.float32),
            rpy_deg=np.zeros(3, dtype=np.float32),
            valid=True,
            timestamp_us=timestamp_us,
        )

    return _invalid_pose(timestamp_us)
