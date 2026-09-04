"""
Capture-set diversity checks — ``viki.calibration.coverage`` (spec §4.1–4.2).
"""

from __future__ import annotations

import cv2
import numpy as np

from viki.calibration import coverage

BOARD = {"type": "aruco", "board_size": [5, 4], "square_size": 0.03,
         "marker_size": 0.022, "aruco_dict": int(cv2.aruco.DICT_4X4_50)}
KM = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
INTR = {"cam": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
                "width": 640, "height": 480, "dist_coeffs": [0, 0, 0, 0, 0]}}


def _obs(rvec, t, seed=0, noise=0.1):
    obj = coverage.board_obj_points(BOARD)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    Xc = obj @ R.T + np.asarray(t, float)
    uv = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), KM, np.zeros(5))[0].reshape(-1, 2)
    uv += np.random.default_rng(seed).normal(0, noise, uv.shape)
    return {"charuco_ids": list(range(len(obj))), "charuco_corners": uv.tolist()}


def test_tilt_deg():
    assert coverage.tilt_deg(np.eye(3)) < 1e-6
    R, _ = cv2.Rodrigues(np.array([np.radians(30.0), 0, 0]))
    assert abs(coverage.tilt_deg(R) - 30.0) < 1e-6


def test_pose_distance_and_nearest():
    obj = coverage.board_obj_points(BOARD)
    d = np.zeros(5)
    a = coverage.board_pose(**_obs([0.05, 0, 0], [0, 0, 0.6], seed=1), K=KM, dist=d, obj_all=obj)
    a_dup = coverage.board_pose(**_obs([0.05, 0, 0], [0, 0, 0.6], seed=2), K=KM, dist=d, obj_all=obj)
    far = coverage.board_pose(**_obs([0.6, 0.2, 0], [0.1, 0, 0.55], seed=3), K=KM, dist=d, obj_all=obj)

    ang, tr = coverage.pose_distance(a, a_dup)
    assert ang < 2.0 and tr < 0.01
    assert coverage.nearest_pose(a_dup, [a], min_angle_deg=8.0, min_translation_m=0.05) == 0
    assert coverage.nearest_pose(far, [a], min_angle_deg=8.0, min_translation_m=0.05) is None


def test_frame_coverage():
    # corners in one 4x4 cell -> 1/16
    one = [np.array([[10, 10], [20, 20], [30, 15]], float)]
    assert abs(coverage.frame_coverage(one, 640, 480) - 1 / 16) < 1e-9
    full = [np.array([[x, y] for x in range(0, 640, 40) for y in range(0, 480, 30)], float)]
    assert coverage.frame_coverage(full, 640, 480) == 1.0


def test_readiness_ready_on_varied_sets():
    tilts = [[0.05, 0.0, 0.0], [0.55, -0.10, 0.0], [-0.50, 0.15, 0.0], [0.10, 0.55, 0.0],
             [-0.15, -0.50, 0.0], [0.45, 0.35, 0.0], [-0.40, -0.30, 0.10], [0.20, -0.15, -0.45]]
    # centres tiling the frame so per-camera coverage clears 60%
    xs, ys = [-0.20, 0.18, -0.20, 0.18, -0.06, 0.06, -0.20, 0.18], \
             [-0.15, -0.15, 0.14, 0.14, 0.0, -0.05, 0.05, -0.02]
    sets = [
        {"observations": {"cam": _obs(tilts[i], [xs[i], ys[i], 0.42], seed=i)}}
        for i in range(8)
    ]
    r = coverage.readiness(
        sets, INTR, BOARD, reference_device="cam",
        min_sets=8, min_covisible_sets=6, min_tilted_sets=3,
        tilt_min_deg=25.0, min_frame_coverage=0.6,
    )
    assert r["n_sets"] == 8
    assert r["n_covisible"] == 8
    assert r["n_tilted"] >= 3
    assert r["ready"] is True


def test_readiness_not_ready_on_one_repeated_pose():
    sets = [{"observations": {"cam": _obs([0.04, 0.02, 0.0], [0, 0, 0.6], seed=i)}}
            for i in range(10)]
    r = coverage.readiness(
        sets, INTR, BOARD, reference_device="cam",
        min_sets=8, min_covisible_sets=6, min_tilted_sets=3,
        tilt_min_deg=25.0, min_frame_coverage=0.6,
    )
    assert r["ready"] is False
    fail = {c["name"] for c in r["criteria"] if not c["ok"]}
    assert "tilted_sets" in fail  # the whole point: no out-of-plane information
