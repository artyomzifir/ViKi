"""
viki.calibration.samples
------------------------
Serialize the ChArUco board observations that back an extrinsics solve, and
re-solve extrinsics from a chosen subset **without live cameras**.

A "set" is one simultaneous ``capture_all`` — one board observation per camera.
Because the board is fixed to the workspace and the cameras don't move, every
kept set contributes 2D↔3D correspondences of the *same* rigid board→camera
transform, so extrinsics is one ``solvePnP`` over the concatenation.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from viki.calibration.geometry import canonical_board_extrinsics

logger = logging.getLogger(__name__)


def sample_to_dict(sample) -> dict:
    """One camera's accepted ChArUco observation → a JSON-able dict."""
    return {
        "corners": np.asarray(sample.corners, dtype=np.float64).reshape(-1, 2).tolist(),
        "c_ids": np.asarray(getattr(sample, "c_ids", []), dtype=np.int64).reshape(-1).tolist(),
        "resolution": [int(sample.resolution[0]), int(sample.resolution[1])],
    }


def board_params_to_dict(bp) -> dict:
    d = {
        "board_size": [int(bp.board_size[0]), int(bp.board_size[1])],
        "square_size": float(bp.square_size),
    }
    if hasattr(bp, "marker_size"):
        d["type"] = "aruco"
        d["marker_size"] = float(bp.marker_size)
        d["aruco_dict"] = int(bp.aruco_dict)
    else:
        d["type"] = "chess"
    return d


def _charuco_board(cfg: dict) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(int(cfg["aruco_dict"]))
    return cv2.aruco.CharucoBoard(
        tuple(cfg["board_size"]),
        float(cfg["square_size"]),
        float(cfg["marker_size"]),
        dictionary,
    )


def _K(intr: dict) -> np.ndarray:
    return np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def solve_extrinsics(
    sets_by_cam: dict[str, list[dict]],
    intrinsics_by_cam: dict[str, dict],
    board_cfg: dict,
) -> list[dict]:
    """Re-solve ``[{device_id, rvec, tvec}]`` from the kept sets.

    ``sets_by_cam[dev]`` is a list of :func:`sample_to_dict` results. Only the
    ChArUco board type is supported (the rig board is ChArUco).
    """
    if board_cfg.get("type") != "aruco":
        raise ValueError("offline extrinsics re-solve only supports ChArUco boards")

    board = _charuco_board(board_cfg)
    obj_all = np.asarray(board.getChessboardCorners(), dtype=np.float32)  # (N, 3)
    bs = tuple(board_cfg["board_size"])
    ss = float(board_cfg["square_size"])
    dist = np.zeros(5)

    out: list[dict] = []
    for dev, sets in sets_by_cam.items():
        intr = intrinsics_by_cam.get(dev)
        if not intr or not sets:
            continue
        obj_pts, img_pts = [], []
        for s in sets:
            ids = np.asarray(s.get("c_ids", []), dtype=int).reshape(-1)
            cor = np.asarray(s.get("corners", []), dtype=np.float32).reshape(-1, 2)
            if ids.size < 4 or ids.size != len(cor) or ids.max(initial=-1) >= len(obj_all):
                continue
            obj_pts.append(obj_all[ids])
            img_pts.append(cor)
        if not obj_pts:
            logger.warning("no usable sets for %s; skipping", dev)
            continue
        ok, rvec, tvec = cv2.solvePnP(
            np.vstack(obj_pts), np.vstack(img_pts), _K(intr), dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            logger.warning("solvePnP failed for %s", dev)
            continue
        rvec, tvec = canonical_board_extrinsics(rvec, tvec, bs, ss)
        out.append({
            "device_id": dev,
            "rvec": np.asarray(rvec, dtype=float).reshape(-1).tolist(),
            "tvec": np.asarray(tvec, dtype=float).reshape(-1).tolist(),
        })
    return out
