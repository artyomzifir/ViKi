"""Small calibration-camera helpers shared by server-side visualisation."""

from __future__ import annotations

import cv2
import numpy as np


def camera_world_pos(rvec: list[float], tvec: list[float]) -> np.ndarray:
    """Compute camera position in the calibration frame."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
    translation = np.asarray(tvec, dtype=np.float32)
    return (-rotation.T @ translation).reshape(3)


def camera_gaze_dir(rvec: list[float]) -> np.ndarray:
    """Compute the camera optical-axis direction in calibration coordinates."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
    return (rotation.T @ np.array([0, 0, 1], dtype=np.float32)).reshape(3)
