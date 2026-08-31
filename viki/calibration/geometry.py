"""
viki.calibration.geometry
-------------------------
Board-pose maths for calibration workers.
"""

from __future__ import annotations

import cv2
import numpy as np


def canonical_board_extrinsics(
    rvec: np.ndarray,
    tvec: np.ndarray,
    board_size: tuple[int, int],
    square_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-centre the board origin at the board centre and flip the Z axis so the
    camera sits at +Z (board normal points toward the camera).

    Calibration (solvePnP / estimatePoseCharucoBoard) returns the pose of the
    board frame whose origin is the first corner. We shift the origin to the
    board centre and, if the solved normal points away from the camera, rotate
    the frame 180° about its X axis. Returns corrected ``(rvec, tvec)``.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(rvec)

    cx_b = (board_size[0] - 1) / 2.0 * square_size
    cy_b = (board_size[1] - 1) / 2.0 * square_size
    t = tvec + R @ np.array([cx_b, cy_b, 0.0])

    if float((-R.T @ t)[2]) < 0:
        R = R @ np.diag([1.0, -1.0, -1.0])

    rvec_corr, _ = cv2.Rodrigues(R)
    return rvec_corr.flatten(), t.flatten()
