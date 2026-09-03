"""
viki.calibration.geometry
-------------------------
Board-pose maths for calibration workers.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def robust_planar_pnp(
    obj_pts: np.ndarray, img_pts: np.ndarray,
    camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
    *, tag: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Board→camera pose from 2D↔3D correspondences, without the local-basin
    ambiguity of ``SOLVEPNP_ITERATIVE``.

    A calibration board is a plane viewed at an angle: its out-of-plane pose is
    weakly constrained and ``SOLVEPNP_ITERATIVE`` (Levenberg–Marquardt from a
    fixed init) can settle in either of two near-degenerate minima. Two cameras
    then pick *different* minima and their world frames disagree by cm / a degree
    — the "two clouds, small offset" symptom. ``SOLVEPNP_SQPNP`` is a
    globally-optimal solver (no init, no basin), then one LM step polishes it.
    Falls back to ITERATIVE only if SQPNP fails outright.
    """
    obj = np.ascontiguousarray(obj_pts, np.float32).reshape(-1, 1, 3)
    img = np.ascontiguousarray(img_pts, np.float32).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        logger.warning("robust_planar_pnp%s: SQPNP failed, using ITERATIVE",
                       f" [{tag}]" if tag else "")
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError("solvePnP failed (SQPNP and ITERATIVE)")
        return np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, camera_matrix, dist_coeffs, rvec, tvec)
    if len(obj) < 15:
        logger.warning(
            "robust_planar_pnp%s: only %d corners — out-of-plane pose is weak, "
            "vary the board angle across captures", f" [{tag}]" if tag else "", len(obj),
        )
    return np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)


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
