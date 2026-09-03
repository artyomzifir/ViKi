"""
Regression guard for :func:`viki.calibration.geometry.robust_planar_pnp`.

The rig board is a plane. Seen at an angle its out-of-plane pose is weakly
constrained, and ``cv2.SOLVEPNP_ITERATIVE`` (LM from a fixed frontal init) can
converge into the mirror-image minimum. Two cameras then pick different minima
and their solved world frames disagree by a rigid ~cm / ~degree — the
"two point clouds, small offset" symptom. ``robust_planar_pnp`` uses the
global SQPNP solver plus an LM polish; these tests pin that it recovers the
true pose (including a hard, tilted view) and is deterministic.
"""

from __future__ import annotations

import cv2
import numpy as np

from viki.calibration.geometry import robust_planar_pnp

_K = np.array([[900.0, 0.0, 640.0],
               [0.0, 900.0, 360.0],
               [0.0, 0.0, 1.0]])
_DIST = np.zeros(5)


def _charuco_like_board(cols: int = 7, rows: int = 5, square: float = 0.04) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(cols) * square, np.arange(rows) * square)
    obj = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    return obj.astype(np.float64)


def _project(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
             noise_px: float = 0.0, seed: int = 0) -> np.ndarray:
    img, _ = cv2.projectPoints(obj, rvec, tvec, _K, _DIST)
    img = img.reshape(-1, 2)
    if noise_px:
        img = img + np.random.default_rng(seed).normal(0.0, noise_px, img.shape)
    return img.astype(np.float64)


def _pose_err(rvec, tvec, rvec_true, tvec_true):
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
    Rt, _ = cv2.Rodrigues(np.asarray(rvec_true).reshape(3))
    ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ Rt) - 1) / 2, -1, 1)))
    trans = float(np.linalg.norm(np.asarray(tvec).reshape(3) - np.asarray(tvec_true).reshape(3)))
    return ang, trans


def test_recovers_a_steeply_tilted_board():
    """A ~40 deg tilt is exactly where the planar ambiguity bites."""
    obj = _charuco_like_board()
    rvec_true = np.array([0.70, -0.15, 0.05])   # ~40 deg out-of-plane
    tvec_true = np.array([0.02, -0.03, 0.55])
    img = _project(obj, rvec_true, tvec_true, noise_px=0.3, seed=1)

    rvec, tvec = robust_planar_pnp(obj, img, _K, _DIST)

    ang, trans = _pose_err(rvec, tvec, rvec_true, tvec_true)
    assert ang < 1.0, f"rotation off by {ang:.2f} deg"
    assert trans < 5e-3, f"translation off by {trans * 1e3:.1f} mm"
    # never returns the board behind the camera
    assert float(np.asarray(tvec).reshape(3)[2]) > 0


def test_is_deterministic():
    obj = _charuco_like_board()
    rvec_true = np.array([0.55, 0.25, -0.10])
    tvec_true = np.array([0.0, 0.0, 0.6])
    img = _project(obj, rvec_true, tvec_true, noise_px=0.5, seed=2)

    r1, t1 = robust_planar_pnp(obj, img, _K, _DIST)
    r2, t2 = robust_planar_pnp(obj, img, _K, _DIST)

    np.testing.assert_allclose(r1, r2)
    np.testing.assert_allclose(t1, t2)


def test_two_views_of_one_board_agree_on_a_common_frame():
    """The actual rig invariant: independent PnP from two camera viewpoints,
    composed back to the board, must land on the same world pose."""
    obj = _charuco_like_board()
    # one board pose, two cameras looking at it from different angles
    poses = [
        (np.array([0.62, -0.20, 0.03]), np.array([0.05, -0.02, 0.052e1])),
        (np.array([-0.45, 0.35, -0.08]), np.array([-0.04, 0.03, 0.061e1])),
    ]
    board_to_cam = []
    for rvec_true, tvec_true in poses:
        img = _project(obj, rvec_true, tvec_true, noise_px=0.3, seed=7)
        rvec, tvec = robust_planar_pnp(obj, img, _K, _DIST)
        R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(tvec).reshape(3)
        board_to_cam.append(T)

    # cam0 -> cam1 via the board should match the true relative pose
    rel = board_to_cam[1] @ np.linalg.inv(board_to_cam[0])
    R0, _ = cv2.Rodrigues(poses[0][0]); R1, _ = cv2.Rodrigues(poses[1][0])
    T0 = np.eye(4); T0[:3, :3] = R0; T0[:3, 3] = poses[0][1]
    T1 = np.eye(4); T1[:3, :3] = R1; T1[:3, 3] = poses[1][1]
    rel_true = T1 @ np.linalg.inv(T0)

    ang = np.degrees(np.arccos(
        np.clip((np.trace(rel[:3, :3].T @ rel_true[:3, :3]) - 1) / 2, -1, 1)))
    trans = float(np.linalg.norm(rel[:3, 3] - rel_true[:3, 3]))
    assert ang < 1.0, f"relative rotation off by {ang:.2f} deg"
    assert trans < 5e-3, f"relative translation off by {trans * 1e3:.1f} mm"
