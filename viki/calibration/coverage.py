"""
viki.calibration.coverage
-------------------------
Capture-set diversity checks for the multi-pose calibration (spec §4.1–4.2).

A joint bundle solve is only well-posed when the board is seen in several
*distinct* poses, with real out-of-plane tilt, spread across each camera's
frame. These are pure functions over the stored ChArUco observations so they
can gate the capture UI and the *Solve* button without a live camera.
"""

from __future__ import annotations

import cv2
import numpy as np

from viki.calibration.geometry import robust_planar_pnp
from viki.calibration.samples import _charuco_board, _K


def board_obj_points(board_cfg: dict) -> np.ndarray:
    return np.asarray(_charuco_board(board_cfg).getChessboardCorners(), np.float64)


def board_pose(
    charuco_ids, charuco_corners, K: np.ndarray, dist: np.ndarray, obj_all: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Board→camera ``(R 3x3, t 3)`` for one observation, or ``None`` if it can't
    be solved."""
    ids = np.asarray(charuco_ids, int).reshape(-1)
    uv = np.asarray(charuco_corners, float).reshape(-1, 2)
    if ids.size < 4 or ids.size != len(uv) or int(ids.max(initial=-1)) >= len(obj_all):
        return None
    try:
        rvec, tvec = robust_planar_pnp(obj_all[ids], uv, K, dist, tag="coverage")
    except RuntimeError:
        return None
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
    return R, np.asarray(tvec).reshape(3)


def pose_distance(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    """``(angle_deg, translation_m)`` between two board→camera poses."""
    Ra, ta = a
    Rb, tb = b
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
    return ang, float(np.linalg.norm(ta - tb))


def nearest_pose(
    cand: tuple[np.ndarray, np.ndarray],
    existing: list[tuple[np.ndarray, np.ndarray]],
    *, min_angle_deg: float, min_translation_m: float,
) -> int | None:
    """Index of the first existing pose that is within *both* thresholds of
    ``cand`` (i.e. not enough new information), or ``None`` if ``cand`` is
    sufficiently different from all of them."""
    for i, e in enumerate(existing):
        ang, tr = pose_distance(cand, e)
        if ang < min_angle_deg and tr < min_translation_m:
            return i
    return None


def tilt_deg(R_cam_board: np.ndarray) -> float:
    """Angle between the board normal and the camera's optical axis. 0° = the
    board faces the camera head-on; 90° = edge-on. The bundle needs several
    sets well above ~25°."""
    n_cam = R_cam_board @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(abs(n_cam[2]), 0.0, 1.0))))


def frame_coverage(
    corner_lists: list[np.ndarray], width: int, height: int, grid: int = 4
) -> float:
    """Fraction of a ``grid×grid`` tiling of the image touched by detected
    corners, unioned over every set."""
    if width <= 0 or height <= 0:
        return 0.0
    hit = np.zeros((grid, grid), bool)
    for uv in corner_lists:
        uv = np.asarray(uv, float).reshape(-1, 2)
        if not len(uv):
            continue
        gx = np.clip((uv[:, 0] / width * grid).astype(int), 0, grid - 1)
        gy = np.clip((uv[:, 1] / height * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return float(hit.mean())


def readiness(
    sets: list[dict],
    intrinsics: dict[str, dict],
    board_cfg: dict,
    *,
    reference_device: str | None,
    min_sets: int,
    min_covisible_sets: int,
    min_tilted_sets: int,
    tilt_min_deg: float,
    min_frame_coverage: float,
) -> dict:
    """Evaluate the *Solve*-ready criteria over the collected ``sets``.

    ``sets`` items are ``{observations: {dev: {charuco_ids, charuco_corners}}}``.
    Returns per-criterion values + a single ``ready`` bool.
    """
    devs = sorted({d for s in sets for d in (s.get("observations") or {})})
    ref = reference_device if reference_device in devs else (devs[0] if devs else None)
    obj_all = board_obj_points(board_cfg) if board_cfg.get("type") == "aruco" else None

    n_sets = len(sets)
    n_covisible = sum(
        1 for s in sets if len(s.get("observations") or {}) >= len(devs) and devs
    )

    # tilt: board pose in the reference camera, per set
    n_tilted = 0
    if ref is not None and obj_all is not None and ref in intrinsics:
        Kref = _K(intrinsics[ref])
        dref = np.asarray(intrinsics[ref].get("dist_coeffs", np.zeros(5)), float).reshape(-1)
        for s in sets:
            o = (s.get("observations") or {}).get(ref)
            if not o:
                continue
            p = board_pose(o["charuco_ids"], o["charuco_corners"], Kref, dref, obj_all)
            if p and tilt_deg(p[0]) >= tilt_min_deg:
                n_tilted += 1

    # frame coverage per camera
    coverage: dict[str, float] = {}
    for d in devs:
        intr = intrinsics.get(d, {})
        w = int(intr.get("width", 0)) or int(intr.get("cx", 0) * 2) or 1280
        h = int(intr.get("height", 0)) or int(intr.get("cy", 0) * 2) or 720
        lists = [
            (s.get("observations") or {})[d]["charuco_corners"]
            for s in sets if d in (s.get("observations") or {})
        ]
        coverage[d] = frame_coverage(lists, w, h)
    min_cov = min(coverage.values()) if coverage else 0.0

    criteria = [
        {"name": "sets", "ok": n_sets >= min_sets, "value": n_sets, "need": min_sets},
        {"name": "covisible_sets", "ok": n_covisible >= min_covisible_sets,
         "value": n_covisible, "need": min_covisible_sets},
        {"name": "tilted_sets", "ok": n_tilted >= min_tilted_sets,
         "value": n_tilted, "need": min_tilted_sets},
        {"name": "frame_coverage", "ok": min_cov >= min_frame_coverage,
         "value": round(min_cov, 3), "need": min_frame_coverage},
    ]
    return {
        "reference_device": ref,
        "n_sets": n_sets,
        "n_covisible": n_covisible,
        "n_tilted": n_tilted,
        "coverage": {d: round(v, 3) for d, v in coverage.items()},
        "criteria": criteria,
        "ready": all(c["ok"] for c in criteria),
    }
