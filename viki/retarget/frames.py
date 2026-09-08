"""Coordinate transforms for retargeting.

Perception artifacts stay in the camera-rig frame.  The calibration snapshot's
``T_world_display`` maps that rig into the installed calibration base used by
retargeting.  The user-provided base position and extrinsic XYZ roll/pitch/yaw
define the complete robot-base pose in that calibration frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CALIBRATION_FRAME = "calibration_base"
RIG_FRAME = "rig"
_LEGACY_RIG_LABELS = frozenset({"robot_base", "calibration_base"})

logger = logging.getLogger(__name__)


def validate_base_position(value: object) -> np.ndarray:
    """Return a finite ``(3,)`` base translation or raise a useful error."""
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError("robot base position must be a finite [x, y, z] vector")
    return out


def base_rotation(base_rpy_deg: object = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Return calibration<-robot rotation from extrinsic XYZ angles in degrees."""
    rpy = validate_base_position(base_rpy_deg)
    return Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()


def base_transform(
    base_position: object,
    base_rpy_deg: object = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the homogeneous calibration<-robot-base transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = base_rotation(base_rpy_deg)
    transform[:3, 3] = validate_base_position(base_position)
    return transform


def calibration_to_robot(
    points: np.ndarray,
    base_position: object,
    base_rpy_deg: object = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Transform calibration-frame points into the robot base frame."""
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"expected trailing xyz dimension, got {values.shape}")
    return (values - validate_base_position(base_position)) @ base_rotation(base_rpy_deg)


def robot_to_calibration(
    points: np.ndarray,
    base_position: object,
    base_rpy_deg: object = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Transform robot-frame points into the shared calibration frame."""
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"expected trailing xyz dimension, got {values.shape}")
    return values @ base_rotation(base_rpy_deg).T + validate_base_position(base_position)


def calibration_rotation_to_robot(
    rotations: np.ndarray,
    base_rpy_deg: object = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Transform calibration-frame orientation matrices into robot coordinates."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing 3x3 rotation, got {values.shape}")
    return np.einsum("ij,...jk->...ik", base_rotation(base_rpy_deg).T, values)


def robot_rotation_to_calibration(
    rotations: np.ndarray,
    base_rpy_deg: object = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Transform robot-frame orientation matrices into calibration coordinates."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing 3x3 rotation, got {values.shape}")
    return np.einsum("ij,...jk->...ik", base_rotation(base_rpy_deg), values)


def _frame_name(value: object) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value or "").strip().lower()


def require_rig_frame(value: object) -> str:
    """Validate the physical frame of a pre-retarget ``cln`` artifact.

    Historical Prepare versions wrote ``robot_base`` (and briefly
    ``calibration_base``) as metadata without ever transforming the underlying
    rig-space arrays. Those labels are migrated here; they do not select an
    alternate transform or revive the retired retarget implementation.
    """
    frame = _frame_name(value)
    if frame in _LEGACY_RIG_LABELS:
        logger.warning(
            "cln coordinate_frame=%r is a legacy metadata label; treating its "
            "unchanged values as rig-space",
            frame,
        )
        return RIG_FRAME
    if frame != RIG_FRAME:
        raise ValueError(
            f"retarget expects cln coordinate_frame={RIG_FRAME!r}, got {frame!r}; "
            "rerun Extract/Prepare so the artifact has an explicit rig-frame contract"
        )
    return RIG_FRAME


def validate_rigid_transform(value: object) -> np.ndarray:
    """Return a finite right-handed SE(3) matrix or raise."""
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_world_display must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("T_world_display must have homogeneous last row [0,0,0,1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("T_world_display rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("T_world_display rotation must be right-handed")
    return transform


def load_rig_to_calibration(raw_dir: str | Path) -> np.ndarray:
    """Load the episode's rig->calibration anchor; absent means aligned frames."""
    path = Path(raw_dir) / "world_anchor.json"
    if not path.exists():
        return np.eye(4, dtype=np.float64)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read calibration anchor {path}: {exc}") from exc
    if "T_world_display" not in payload:
        raise ValueError(f"calibration anchor {path} has no T_world_display")
    return validate_rigid_transform(payload["T_world_display"])


def rig_pose_to_calibration(
    positions: np.ndarray,
    rotations: np.ndarray,
    transform: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``T_calibration_rig`` exactly once to a pose trajectory."""
    points = np.asarray(positions, dtype=np.float64)
    orientations = np.asarray(rotations, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"positions must have shape (T, 3), got {points.shape}")
    if orientations.shape != (len(points), 3, 3):
        raise ValueError(
            f"rotations must have shape ({len(points)}, 3, 3), got {orientations.shape}"
        )
    anchor = validate_rigid_transform(transform)
    rig_to_calibration_r = anchor[:3, :3]
    calibration_positions = points @ rig_to_calibration_r.T + anchor[:3, 3]
    calibration_rotations = np.einsum(
        "ij,tjk->tik", rig_to_calibration_r, orientations
    )
    return calibration_positions, calibration_rotations
