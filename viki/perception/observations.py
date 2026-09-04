"""
viki.perception.observations
----------------------------
Stage 1 of the multi-view triangulation task: persist the RAW per-camera
per-frame 2-D observations a triangulator needs, **alongside** ``rec.npz`` (which
stays the legacy mono-lift baseline for A/B — this file never replaces it).

``raw/observations.npz`` — one row per accepted detection:

===================  =========  =======================================
``camera_id``        (N,)       device id
``frame_index``      (N,)       colour-frame index in that camera's mp4
``host_timestamp_us``(N,)       the synced-group tick for that frame
``uv``               (N,21,2)   landmark pixels in the FULL colour image,
                                raw / distorted (Stage 2 undistorts once)
``lm_score``         (N,21)     per-landmark detector score in [0, 1]
``lm_score_per_pt``  (N,)       False ⇒ ``lm_score`` is the hand score
                                broadcast (backend gave no per-landmark score)
``depth_m``          (N,21)     measured depth at each landmark (m; NaN if
                                unavailable) — **soft** evidence only
``depth_valid``      (N,21)     bool
``depth_spread_m``   (N,21)     local depth std in the sample window (m)
===================  =========  =======================================

``raw/observations_meta.json`` — per-camera calibration reference
(``K``, ``dist``, ``T_wc``, ``image_size``, ``calib_id``) + the sampler config.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from viki.contracts import HAND_LM_COUNT, LM

logger = logging.getLogger(__name__)

OBS_SCHEMA = 1


def calib_id(raw: Path) -> str:
    """Short hash tying an observation set to the exact calibration it used."""
    h = hashlib.sha1()
    for name in ("extrinsics.json", "intrinsics.json"):
        p = raw / name
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def sample_depth(
    uv_color, depth_m: np.ndarray, projector, radius: int
) -> tuple[float, bool, float]:
    """Measured depth near one colour-image landmark: ``(z_m, valid, spread_m)``.

    The colour pixel is mapped to the depth image with the backend projector
    (identity when depth is already colour-aligned), then a ``radius`` window is
    reduced by nanmedian. ``valid`` requires a real reading at the pixel or a
    reasonably populated window; ``spread_m`` is the window's std, a proxy for
    "am I on a depth edge / noisy patch" that Stage 2 down-weights on.
    """
    if depth_m is None or not np.isfinite(uv_color).all():
        return float("nan"), False, float("nan")
    u, v = float(uv_color[0]), float(uv_color[1])
    try:
        mapped = projector.project_color_to_depth(u, v, 1.0)
    except Exception:  # noqa: BLE001
        mapped = None
    if mapped is None:
        return float("nan"), False, float("nan")
    du, dv = int(round(mapped[0])), int(round(mapped[1]))
    h, w = depth_m.shape[:2]
    if not (0 <= du < w and 0 <= dv < h):
        return float("nan"), False, float("nan")
    u0, u1 = max(0, du - radius), min(w, du + radius + 1)
    v0, v1 = max(0, dv - radius), min(h, dv + radius + 1)
    roi = depth_m[v0:v1, u0:u1]
    fin = np.isfinite(roi) & (roi > 0)
    if not fin.any():
        return float("nan"), False, float("nan")
    vals = roi[fin]
    z = float(np.median(vals))
    spread = float(np.std(vals))
    centre_ok = np.isfinite(depth_m[dv, du]) and depth_m[dv, du] > 0
    valid = bool(centre_ok or fin.mean() > 0.25)
    return z, valid, spread


def collect_row(
    *,
    camera_id: str,
    frame_index: int,
    host_timestamp_us: int,
    detection,
    depth_m: np.ndarray,
    projector,
    depth_radius: int,
) -> dict:
    """Build one observation row from a backend ``HandDetection`` (points already
    un-mirrored into the real colour frame)."""
    uv = np.full((HAND_LM_COUNT, 2), np.nan, np.float32)
    for lm, p in detection.points.items():
        uv[int(lm)] = np.asarray(p, np.float32)[:2]

    per_pt = getattr(detection, "lm_score", None)
    if per_pt is not None and len(np.asarray(per_pt).reshape(-1)) == HAND_LM_COUNT:
        score = np.asarray(per_pt, np.float32).reshape(HAND_LM_COUNT)
        score_per_pt = True
    else:
        score = np.full(HAND_LM_COUNT, float(detection.confidence), np.float32)
        score_per_pt = False

    z = np.full(HAND_LM_COUNT, np.nan, np.float32)
    zv = np.zeros(HAND_LM_COUNT, bool)
    zs = np.full(HAND_LM_COUNT, np.nan, np.float32)
    for i in range(HAND_LM_COUNT):
        if np.isfinite(uv[i]).all():
            z[i], zv[i], zs[i] = sample_depth(uv[i], depth_m, projector, depth_radius)

    return {
        "camera_id": camera_id,
        "frame_index": int(frame_index),
        "host_timestamp_us": int(host_timestamp_us),
        "uv": uv,
        "lm_score": score,
        "lm_score_per_pt": score_per_pt,
        "depth_m": z,
        "depth_valid": zv,
        "depth_spread_m": zs,
    }


def write_observations(
    npz_path: Path, rows: list[dict], cameras: dict, sampler_cfg: dict
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        arrs = dict(
            schema=np.int32(OBS_SCHEMA),
            camera_id=np.array([r["camera_id"] for r in rows]),
            frame_index=np.array([r["frame_index"] for r in rows], np.int32),
            host_timestamp_us=np.array([r["host_timestamp_us"] for r in rows], np.int64),
            uv=np.stack([r["uv"] for r in rows]).astype(np.float32),
            lm_score=np.stack([r["lm_score"] for r in rows]).astype(np.float32),
            lm_score_per_pt=np.array([r["lm_score_per_pt"] for r in rows], bool),
            depth_m=np.stack([r["depth_m"] for r in rows]).astype(np.float32),
            depth_valid=np.stack([r["depth_valid"] for r in rows]).astype(bool),
            depth_spread_m=np.stack([r["depth_spread_m"] for r in rows]).astype(np.float32),
        )
    else:
        arrs = {"schema": np.int32(OBS_SCHEMA), "camera_id": np.array([], "<U32")}
    np.savez(npz_path, **arrs)
    (npz_path.parent / "observations_meta.json").write_text(
        json.dumps({"schema": OBS_SCHEMA, "cameras": cameras, "sampler": sampler_cfg}, indent=2)
    )
    logger.info("observations: %d rows, %d camera(s) -> %s", len(rows), len(cameras), npz_path)


def read_observations(raw: Path) -> dict | None:
    npz = raw / "observations.npz"
    if not npz.is_file():
        return None
    with np.load(npz, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    meta_p = raw / "observations_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    out["cameras"] = meta.get("cameras", {})
    out["sampler"] = meta.get("sampler", {})
    return out
