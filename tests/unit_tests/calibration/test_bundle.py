"""
Multi-pose bundle extrinsics solver — ``viki.calibration.bundle``.

Synthetic rig: two cameras, known relative pose, a ChArUco board rendered at
several poses. The solver must recover the relative pose from noisy corner
observations, flag a single-pose input as degenerate, and cope with a set the
reference camera does not see.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from viki.calibration.bundle import solve_bundle
from viki.calibration.samples import _charuco_board

BOARD = {"type": "aruco", "board_size": [5, 4], "square_size": 0.03,
         "marker_size": 0.022, "aruco_dict": int(cv2.aruco.DICT_4X4_50)}
K = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
INTR = {"cam_ref": K, "cam_b": K}
_Kmat = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])


def _T(rvec, t):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _project(T_cam, Xb):
    Xc = Xb @ T_cam[:3, :3].T + T_cam[:3, 3]
    uv, _ = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), _Kmat, np.zeros(5))
    return uv.reshape(-1, 2)


def _make_sets(board_poses, T_b_ref, *, noise_px=0.0, drop_ref=(), seed=0):
    """board_poses: list of (rvec, t) for T_ref_board. T_b_ref: ref→cam_b."""
    rng = np.random.default_rng(seed)
    obj_all = np.asarray(_charuco_board(BOARD).getChessboardCorners(), np.float64)
    ids = np.arange(len(obj_all))
    sets = []
    for k, (rv, t) in enumerate(board_poses):
        Trb = _T(rv, t)
        Xr = obj_all @ Trb[:3, :3].T + Trb[:3, 3]
        obs = {}
        if k not in drop_ref:
            uv = _project(np.eye(4), Xr) + rng.normal(0, noise_px, (len(ids), 2))
            obs["cam_ref"] = {"charuco_ids": ids.tolist(), "charuco_corners": uv.tolist()}
        uv_b = _project(T_b_ref, Xr) + rng.normal(0, noise_px, (len(ids), 2))
        obs["cam_b"] = {"charuco_ids": ids.tolist(), "charuco_corners": uv_b.tolist()}
        sets.append({"set_id": f"s{k}", "observations": obs})
    return sets


def _pose_err(T_got, T_true):
    d_rot = np.degrees(np.arccos(np.clip(
        (np.trace(T_got[:3, :3].T @ T_true[:3, :3]) - 1) / 2, -1, 1)))
    d_t = np.linalg.norm(T_got[:3, 3] - T_true[:3, 3])
    return float(d_rot), float(d_t)


def test_recovers_relative_pose_from_varied_board_poses():
    T_b_ref = _T([0.0, 0.45, 0.0], [0.35, 0.02, 0.08])   # ref → cam_b
    poses = [
        ([0.05, 0.02, 0.0], [0.0, 0.0, 0.65]),
        ([0.55, -0.10, 0.03], [-0.03, 0.02, 0.60]),
        ([-0.45, 0.20, -0.05], [0.04, -0.03, 0.62]),
        ([0.30, 0.40, 0.10], [0.02, 0.03, 0.70]),
        ([-0.20, -0.35, 0.15], [-0.02, 0.0, 0.66]),
        ([0.10, 0.15, -0.40], [0.0, -0.02, 0.63]),
    ]
    sets = _make_sets(poses, T_b_ref, noise_px=0.2, seed=1)

    out = solve_bundle(sets, INTR, BOARD, reference_device="cam_ref")

    assert out["reference_device"] == "cam_ref"
    np.testing.assert_allclose(out["devices"]["cam_ref"], np.eye(4), atol=1e-9)
    T_ref_camb = np.asarray(out["devices"]["cam_b"])
    d_rot, d_t = _pose_err(T_ref_camb, np.linalg.inv(T_b_ref))
    assert d_rot < 0.5, f"rotation off {d_rot:.3f}°"
    assert d_t < 5e-3, f"translation off {d_t*1e3:.2f} mm"
    assert out["solve"]["degenerate"] is False
    assert out["solve"]["stereo_check"]["ran"] is True
    assert out["solve"]["stereo_check"]["agrees"] is True
    assert max(out["solve"]["rms_reproj_px"].values()) < 1.0


def test_single_board_pose_is_flagged_degenerate():
    T_b_ref = _T([0.0, 0.4, 0.0], [0.3, 0.0, 0.05])
    poses = [([0.03, 0.02, 0.0], [0.0, 0.0, 0.6])] * 4  # same pose four times
    # tiny jitter so PnP doesn't choke, but well under the 8° gate
    poses = [([0.03 + i * 1e-3, 0.02, 0.0], [0.0, 0.0, 0.6]) for i in range(4)]
    sets = _make_sets(poses, T_b_ref, noise_px=0.1, seed=2)

    out = solve_bundle(sets, INTR, BOARD, reference_device="cam_ref")
    assert out["solve"]["degenerate"] is True


def test_handles_a_set_the_reference_camera_misses():
    T_b_ref = _T([0.0, 0.45, 0.0], [0.35, 0.02, 0.08])
    poses = [
        ([0.05, 0.02, 0.0], [0.0, 0.0, 0.65]),
        ([0.55, -0.10, 0.03], [-0.03, 0.02, 0.60]),
        ([-0.45, 0.20, -0.05], [0.04, -0.03, 0.62]),
        ([0.30, 0.40, 0.10], [0.02, 0.03, 0.70]),
    ]
    sets = _make_sets(poses, T_b_ref, noise_px=0.2, drop_ref=(2,), seed=3)

    out = solve_bundle(sets, INTR, BOARD, reference_device="cam_ref")
    T_ref_camb = np.asarray(out["devices"]["cam_b"])
    d_rot, d_t = _pose_err(T_ref_camb, np.linalg.inv(T_b_ref))
    assert d_rot < 0.6 and d_t < 6e-3
    # the ref-less set still contributed (4 sets in, 4 kept)
    assert out["solve"]["n_sets"] == 4


def test_auto_picks_a_reference_when_none_given():
    T_b_ref = _T([0.0, 0.4, 0.0], [0.3, 0.0, 0.05])
    poses = [
        ([0.05, 0.0, 0.0], [0.0, 0.0, 0.65]),
        ([0.5, -0.1, 0.0], [0.0, 0.0, 0.62]),
        ([-0.4, 0.2, 0.0], [0.0, 0.0, 0.60]),
    ]
    sets = _make_sets(poses, T_b_ref, noise_px=0.15, seed=4)
    out = solve_bundle(sets, INTR, BOARD)
    assert out["reference_device"] in ("cam_ref", "cam_b")
