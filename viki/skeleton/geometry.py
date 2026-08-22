"""
viki.skeleton.geometry
----------------------
Lifting MediaPipe hand landmarks to 3-D camera space.

Pipeline:
  1. Project every detected landmark (pixels) to depth space and deproject it at
     its OWN measured depth -> raw3d[lm] (camera frame).
  2. Estimate the hand POSITION from a reference set of landmarks:
        - position_from_wrist=True  -> the wrist landmark only
        - position_from_wrist=False -> the palm/knuckle nodes (robust median)
     A *smart* outlier filter (robust median/MAD, capped at
     discard_outliers_max_portion) removes only genuine depth outliers, so the
     estimate is stable frame-to-frame (no fixed-fraction cut that would change
     the aggregated set and jitter).
  3. Build the hand SHAPE from MediaPipe (x, y, z) at the hand depth zd and
     translate it so the reference landmark(s) land on the estimated position.
"""

from __future__ import annotations
from typing import Any, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

from viki.skeleton.models import (
    HandDetection,
    Landmarks3D,
    LM,
    PreparedFrame,
)
from viki.calibration.models import CalibrationExtrinsics
import viki.config as config

# Palm/knuckle landmarks used to estimate the hand position when not using the
# wrist alone.  Solid skin areas that almost never project onto the inter-finger
# gaps, so they give a stable position estimate.
_PALM_LM = (
    LM.WRIST,
    LM.THUMB_CMC,
    LM.INDEX_MCP,
    LM.MIDDLE_MCP,
    LM.RING_MCP,
    LM.PINKY_MCP,
)

# Robust outlier cutoff: a point is a candidate outlier if its Mahalanobis-style
# scaled distance from the median exceeds this many robust scales.
_OUTLIER_K = 3.0


def _empty_landmarks(detection: HandDetection) -> Landmarks3D:
    return Landmarks3D(
        points={LM(idx): np.full(3, np.nan, dtype=np.float32) for idx in range(LM.N)},
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )


def lift_to_3d(
    detection: HandDetection,
    frame: PreparedFrame,
    backend: Any,
    prev_position: np.ndarray | None = None,
    discard_outliers: bool = False,
    discard_outliers_max_portion: float = 0.2,
    position_from_wrist: bool = False,
) -> Optional[Landmarks3D]:
    """
    Lift a single-camera HandDetection into 3-D camera-space landmarks.

    Parameters
    ----------
    detection : HandDetection
        MediaPipe-style 2D landmarks (pixel positions) + relative z.
    frame : PreparedFrame
        Undistorted RGB, depth (metres), and depth intrinsics K.
    backend : object
        Camera backend providing `project_color_to_depth`.
    prev_position : np.ndarray | None
        Previously estimated 3-D hand position (unused; retained for API compat).
    discard_outliers : bool
        Enable smart outlier removal of the position reference set.
    discard_outliers_max_portion : float
        Upper bound on the fraction of reference points that may be removed.
    position_from_wrist : bool
        True  -> estimate position from the wrist landmark only.
        False -> estimate position from the palm/knuckle nodes (median).

    Returns
    -------
    Landmarks3D | None
        Per-landmark 3-D camera positions, or None when the hand region has no
        usable depth (all palm nodes are depth holes) so a pose cannot be placed
        without mis-locating the hand.
    """
    depth_m = frame.depth_m
    h, w = depth_m.shape[:2]
    if h == 0 or w == 0:
        return _empty_landmarks(detection)

    K = frame.depth_K
    if K is None or K[0, 0] <= 0 or K[1, 1] <= 0:
        logger.warning(
            "lift_to_3d: invalid depth intrinsics for %s; returning NaNs",
            detection.device_id,
        )
        return _empty_landmarks(detection)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Phase 1 -- project every detected landmark (pixels) to depth space and
    # deproject at its OWN measured depth -> raw3d[lm].
    base_depth_m = getattr(frame, "base_depth_m", None)
    use_validation = bool(getattr(config, "SKELETON_ENABLE_DEPTH_VALIDATION", False))
    samp_radius = int(getattr(config, "SKELETON_DEPTH_SAMP_RADIUS", 15))
    subtract_thresh = float(getattr(config, "SKELETON_DEPTH_SUBTRACT_THRESHOLD", 0.01))

    proj_uv: dict[LM, tuple[float, float]] = {}
    raw3d: dict[LM, np.ndarray] = {}
    for lm, uv in detection.points.items():
        if np.isnan(uv[0]) or np.isnan(uv[1]):
            continue
        res = backend.project_color_to_depth(uv[0], uv[1], 1.0)
        if res is None:
            res = (uv[0], uv[1])
        ud, vd = res
        proj_uv[lm] = (ud, vd)

        ui, vi = int(round(ud)), int(round(vd))

        # When a static background depth has been captured, sample a small ROI
        # around the landmark and subtract the base: only pixels that are
        # significantly *closer* than the background (diff > threshold) belong to
        # the hand, so we take the median of those. This isolates the tracked
        # hand from the static scene and stabilises depth at sparse IR spots.
        z = np.nan
        if (
            use_validation
            and base_depth_m is not None
            and base_depth_m.shape == depth_m.shape
        ):
            r = samp_radius
            v0, v1 = max(0, vi - r), min(h, vi + r + 1)
            u0, u1 = max(0, ui - r), min(w, ui + r + 1)
            if v1 > v0 and u1 > u0:
                current_roi = depth_m[v0:v1, u0:u1]
                base_roi = base_depth_m[v0:v1, u0:u1]
                valid = (
                    ~np.isnan(current_roi) & (current_roi > 0.01) & (current_roi <= 10.0)
                )
                if valid.any():
                    if base_roi.shape == current_roi.shape:
                        diff = np.maximum(0.0, base_roi - current_roi)
                        keep = valid & (diff > subtract_thresh)
                        if keep.any():
                            z = float(np.median(current_roi[keep]))
                    if not np.isfinite(z):
                        # Fallback: median of all valid depth in the ROI.
                        z = float(np.median(current_roi[valid]))
        else:
            if 0 <= vi < h and 0 <= ui < w:
                zv = depth_m[vi, ui]
                if not np.isnan(zv) and 0.01 < zv <= 10.0:
                    z = float(zv)

        if np.isfinite(z):
            X = (ud - cx) * z / fx
            Y = (vd - cy) * z / fy
            raw3d[lm] = np.array([X, Y, z], dtype=np.float32)

    if len(proj_uv) < 3:
        logger.warning(
            "lift_to_3d: too few landmarks projected for %s; returning NaNs",
            detection.device_id,
        )
        return _empty_landmarks(detection)

    # Phase 2 -- estimate hand position from the palm/knuckle reference nodes.
    # If none of the palm nodes have depth, the hand region is a depth hole:
    # emitting any fallback depth would place the hand at a wrong position, so
    # we drop the frame (return None) and let the smooth stage interpolate
    # across the gap instead.
    palm_raw = [raw3d[lm] for lm in _PALM_LM if lm in raw3d]
    if not palm_raw:
        return None

    if position_from_wrist and LM.WRIST in raw3d:
        ref_raw = [raw3d[LM.WRIST]]
    else:
        ref_raw = palm_raw
    hand_pos = _smart_median(
        np.array(ref_raw, dtype=np.float64),
        discard_outliers,
        discard_outliers_max_portion,
    )

    zd = float(hand_pos[2])
    if not (0.05 <= zd <= 10.0):
        # Depth is present but outside the plausible hand range; do not emit a
        # mis-placed pose.
        return None
    hand_pos = np.array([float(hand_pos[0]), float(hand_pos[1]), zd], dtype=np.float32)

    # Phase 3 -- build the hand SHAPE from MediaPipe (x, y, z) at depth zd, then
    # translate so the reference landmark(s) coincide with hand_pos.
    scale = zd * w / fx
    mp_points: dict[LM, np.ndarray] = {}
    for lm, (ud, vd) in proj_uv.items():
        zm = detection.lm_z_rel[lm.value]
        X = (ud - cx) * zd / fx
        Y = (vd - cy) * zd / fy
        Z = zd + zm * scale
        mp_points[lm] = np.array([X, Y, Z], dtype=np.float32)

    if not mp_points:
        return _empty_landmarks(detection)

    # Reference point in the MediaPipe shape (the point we align to hand_pos).
    if position_from_wrist and LM.WRIST in mp_points:
        ref_mp = mp_points[LM.WRIST]
    else:
        ref_mp = np.median(
            np.array(
                [mp_points[lm] for lm in _PALM_LM if lm in mp_points],
                dtype=np.float64,
            ),
            axis=0,
        )
    shift = hand_pos - ref_mp

    points: dict[LM, np.ndarray] = {
        lm: (vec + shift).astype(np.float32) for lm, vec in mp_points.items()
    }

    nan_count = sum(1 for p in points.values() if np.isnan(p).any())
    if nan_count > 0:
        logger.warning(
            "lift_to_3d: %d/21 landmarks are NaN for %s", nan_count, detection.device_id
        )

    return Landmarks3D(
        points=points,
        device_id=detection.device_id,
        timestamp_us=detection.timestamp_us,
    )


def _smart_median(
    pts: np.ndarray,
    discard_outliers: bool,
    max_portion: float,
) -> np.ndarray:
    """
    Robust median with smart (MAD-based, capped) outlier removal.

    Returns the component-wise median of ``pts`` after dropping only points that
    are clear outliers (Mahalanobis-style scaled distance > ``_OUTLIER_K`` from
    the median), capped so at most ``max_portion`` of the points can be removed.
    When no point is a genuine outlier the input set is left intact, which keeps
    the estimate stable frame-to-frame (no jitter from a changing kept set).
    """
    if pts.shape[0] < 3 or not discard_outliers:
        return np.median(pts, axis=0).astype(np.float32)

    med = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - med), axis=0)  # per-axis robust spread
    robust_scale = 1.4826 * mad
    robust_scale = np.where(robust_scale < 1e-6, 1e-6, robust_scale)

    scaled = np.linalg.norm((pts - med) / robust_scale, axis=1)
    outlier_idx = np.where(scaled > _OUTLIER_K)[0]

    # Cap the number of removals at max_portion of the set (farthest first).
    max_remove = int(np.floor(max_portion * pts.shape[0]))
    if outlier_idx.size > max_remove:
        order = outlier_idx[np.argsort(scaled[outlier_idx])[::-1]]
        outlier_idx = order[:max_remove]

    if outlier_idx.size == 0:
        return med.astype(np.float32)

    keep = np.ones(pts.shape[0], dtype=bool)
    keep[outlier_idx] = False
    return np.median(pts[keep], axis=0).astype(np.float32)


def camera_landmarks_to_world(
    landmarks: Landmarks3D,
    extr: CalibrationExtrinsics | None,
) -> dict[LM, np.ndarray]:
    """
    Transform one camera's camera‑frame 3‑D landmarks into world coordinates.

    Uses the camera's ``transform_matrix`` (World‑to‑Camera). Landmarks with
    non‑finite coordinates are skipped so they cannot poison downstream fusion.
    This is the same transform the fusion step used; it is exposed here so the
    live pipeline can emit world‑frame per‑camera skeletons without fusing them.

    Parameters
    ----------
    landmarks : Landmarks3D
        Camera‑frame 3‑D landmarks for a single camera.
    extr : CalibrationExtrinsics | None
        Extrinsic calibration for that camera (None → identity transform).

    Returns
    -------
    dict[LM, np.ndarray]
        World‑frame landmark positions, excluding non‑finite observations.
    """
    T = extr.transform_matrix if extr is not None else np.eye(4)
    world_points: dict[LM, np.ndarray] = {}
    for index, vec in landmarks.points.items():
        if len(vec.flatten()) != 3 or np.isnan(vec).any():
            continue
        pos_mtx = np.eye(4)
        pos_mtx[:3, 3] = vec
        world_points[index] = (T @ pos_mtx)[:3, 3].flatten().astype(np.float32)
    return world_points
