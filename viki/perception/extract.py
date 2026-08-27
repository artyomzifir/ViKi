"""
viki.perception.extract
-----------------------
Offline orchestrator: ``episodes/<id>/raw/`` -> ``rec.npz``.

Decodes the recorded colour frames + raw depth, runs the configured hand-pose
backend per camera per frame, lifts to 3-D with measured depth, transforms into
the workspace frame with the recorded extrinsics, and writes per-camera landmark
trajectories in the ``rec.npz`` schema.

Assumption: recorded depth is aligned to colour (identity colour→depth pixel
map). A backend-specific projector can be plugged in later via ``DepthProjector``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from viki.contracts import (
    HAND_LM_COUNT,
    CalibrationExtrinsics,
    LM,
    PreparedFrame,
)
from viki.episode import mark_stage
from viki.perception.backends import load_backend
from viki.perception.geometry import camera_landmarks_to_world, lift_to_3d

logger = logging.getLogger(__name__)


class _IdentityProjector:
    """Colour pixel == depth pixel (aligned depth)."""

    def project_color_to_depth(self, u: float, v: float, z: float):
        return (u, v)


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _depth_K(intr: dict) -> np.ndarray | None:
    if not intr:
        return None
    return np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
        dtype=np.float32,
    )


def _extrinsics(raw_extr: dict, dev_id: str) -> CalibrationExtrinsics | None:
    e = raw_extr.get(dev_id)
    if not e:
        return None
    return CalibrationExtrinsics(
        rvec=np.asarray(e["rvec"], dtype=np.float64),
        tvec=np.asarray(e["tvec"], dtype=np.float64),
    )


def extract_episode(
    ep,
    *,
    backend: str | None = None,
    hand: str = "right",
) -> str:
    """Run perception over an episode's raw frames; write ``rec.npz``. Returns the path."""
    from viki import config as _cfg

    raw = ep.raw_dir
    backend_name = backend or getattr(_cfg, "POSE_BACKEND", "mediapipe")
    intr_all = _read_json(raw / "intrinsics.json")
    extr_all = _read_json(raw / "extrinsics.json")
    projector = _IdentityProjector()

    device_ids: list[str] = []
    timestamps: list[int] = []
    points: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    nan3 = np.full(3, np.nan, dtype=np.float32)

    for mp4 in sorted(raw.glob("*.mp4")):
        dev_id = mp4.stem
        depth_dir = raw / f"{dev_id}_depth"
        depth_files = sorted(depth_dir.glob("*.npy")) if depth_dir.is_dir() else []
        K = _depth_K(intr_all.get(dev_id, {}))
        extr = _extrinsics(extr_all, dev_id)
        det_backend = load_backend(backend_name, mode="video")

        cap = cv2.VideoCapture(str(mp4))
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if idx < len(depth_files):
                depth_mm = np.load(depth_files[idx])
                depth_m = depth_mm.astype(np.float32) / 1000.0
                depth_m[depth_m == 0] = np.nan
            else:
                depth_m = np.full(rgb.shape[:2], np.nan, dtype=np.float32)

            prepared = PreparedFrame(
                rgb=rgb, depth_m=depth_m, depth_K=K, device_id=dev_id, timestamp_us=idx
            )
            det = det_backend.detect(prepared, hand)
            idx += 1
            if det is None:
                continue

            lms = lift_to_3d(det, prepared, projector)
            if lms is None:
                continue
            world = camera_landmarks_to_world(lms, extr)
            if not world:
                continue

            pts = np.array(
                [world.get(LM(i), nan3) for i in range(HAND_LM_COUNT)], dtype=np.float32
            )
            conf = np.full(HAND_LM_COUNT, det.confidence, dtype=np.float32)
            device_ids.append(dev_id)
            timestamps.append(int(idx))
            points.append(pts)
            confidence.append(conf)
        cap.release()
        det_backend.close()

    pts_arr = (
        np.stack(points) if points else np.empty((0, HAND_LM_COUNT, 3), dtype=np.float32)
    )
    np.savez_compressed(
        ep.rec_npz,
        device_ids=np.array(device_ids) if device_ids else np.empty((0,), dtype="<U1"),
        timestamps=np.array(timestamps, dtype=np.int64),
        points=pts_arr,
        landmark_ids=np.arange(HAND_LM_COUNT, dtype=np.int32),
        confidence=(
            np.stack(confidence)
            if confidence
            else np.empty((0, HAND_LM_COUNT), dtype=np.float32)
        ),
    )
    mark_stage(ep, "extract", frames=len(device_ids), backend=backend_name)
    logger.info("extract %s: %d frames -> %s", ep.id, len(device_ids), ep.rec_npz)
    return str(ep.rec_npz)
