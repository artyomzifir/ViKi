"""
viki.perception.extract
-----------------------
Offline orchestrator: ``episodes/<id>/raw/`` -> ``rec.npz``.

Decodes the recorded colour frames + raw depth, runs the configured hand-pose
backend per camera per frame, lifts to 3-D with measured depth, transforms into
the workspace frame with the recorded extrinsics, and writes per-camera landmark
trajectories in the ``rec.npz`` schema.

Colour→depth pixel mapping: if the episode carries a Kinect raw calibration blob
(``raw/<dev>_k4a_calib.bin``, written by the recorder) it is rebuilt offline and
used as the projector. Otherwise we fall back to the identity map
(``_IdentityProjector``) — correct only when depth is already aligned to colour
(RealSense, whose depth is colour-aligned at capture).
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


def _depth_K(entry: dict) -> np.ndarray | None:
    """Depth camera matrix from one camera's ``raw/intrinsics.json`` entry.

    Handles both layouts: the recorder's nested form
    ``{"color": {...}, "depth": {...}, "source": ...}`` and the older flat form
    ``{"fx": ..., "fy": ..., "cx": ..., "cy": ...}``.
    """
    if not entry:
        return None
    intr = entry.get("depth") or entry.get("color") or entry
    if not intr or "fx" not in intr:
        return None
    return np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
        dtype=np.float32,
    )


def _load_projector(raw: Path, dev_id: str, meta: dict):
    """Real SDK colour→depth projector, tried in order:
    the episode's ``raw/<dev>_k4a_calib.bin`` → the k4a blob stored on the
    episode's calibration preset (``meta['calibration_preset']``) → ``None``
    (caller falls back to identity)."""
    try:
        from viki.perception.k4a_offline import K4ACalibration

        cal = K4ACalibration.from_episode(raw, dev_id, meta)
        if cal is not None:
            return cal
    except Exception as exc:  # noqa: BLE001 — never let calib issues abort extract
        logger.warning("extract %s: episode k4a blob unusable (%s)", dev_id, exc)

    preset = (meta or {}).get("calibration_preset")
    if preset:
        try:
            from viki.calibration import presets as _presets

            return _presets.k4a_calibration(preset, dev_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract %s: preset %r k4a unusable (%s)", dev_id, preset, exc)
    return None


def _frame_timestamps(raw: Path) -> list[dict]:
    ts = _read_json(raw / "timestamps.json")
    return ts if isinstance(ts, list) else []


def _row_ts_us(ts_list: list[dict], idx: int, dev_id: str) -> int:
    """Real host-monotonic µs for camera ``dev_id`` at synced-group ``idx``."""
    if 0 <= idx < len(ts_list):
        e = ts_list[idx]
        return int(e.get("sync_us", idx)) + int((e.get("offsets_us") or {}).get(dev_id, 0))
    return int(idx)


def _extrinsics(raw_extr: dict, dev_id: str) -> CalibrationExtrinsics | None:
    e = raw_extr.get(dev_id)
    if not e:
        return None
    return CalibrationExtrinsics(
        rvec=np.asarray(e["rvec"], dtype=np.float64),
        tvec=np.asarray(e["tvec"], dtype=np.float64),
    )


def _noop(**_kw):
    return None


def extract_episode(
    ep,
    *,
    backend: str | None = None,
    model: str | None = None,
    hand: str = "right",
    track_lm: list[int] | None = None,
    min_confidence: float | None = None,
    flip: bool = False,
    report=None,
) -> str:
    """Run perception over an episode's raw frames; write ``rec.npz``. Returns the path.

    ``flip`` mirrors each colour frame before detection (landmark x flipped back
    before the depth lift). The MediaPipe Tasks model usually reports the correct
    anatomical side on non-mirrored Kinect frames, so the default is off; flip +
    the opposite ``hand`` is the escape hatch when a camera's handedness label
    comes out wrong. ``track_lm`` keeps only
    those landmark indices (others are left NaN). ``report(stage, camera, frame,
    total)`` is called for progress.
    """
    from viki import config as _cfg

    from viki.contracts import SkeletonFrame
    from viki.perception.recorder import write_rec

    report = report or _noop
    raw = ep.raw_dir
    backend_name = backend or getattr(_cfg, "POSE_BACKEND", "mediapipe")
    keep = set(track_lm) if track_lm else None
    intr_all = _read_json(raw / "intrinsics.json")
    extr_all = _read_json(raw / "extrinsics.json")
    meta_all = _read_json(ep.meta_path)
    ts_list = _frame_timestamps(raw)
    identity = _IdentityProjector()
    be_kw = {"mode": "video"}
    if model:
        be_kw["model"] = model
    if min_confidence is not None:
        be_kw["min_confidence"] = float(min_confidence)

    records: list[tuple[str, SkeletonFrame, dict]] = []
    mp4s = sorted(raw.glob("*.mp4"))

    for mp4 in mp4s:
        dev_id = mp4.stem
        depth_dir = raw / f"{dev_id}_depth"
        depth_files = sorted(depth_dir.glob("*.npy")) if depth_dir.is_dir() else []
        total = len(depth_files) or int(cv2.VideoCapture(str(mp4)).get(cv2.CAP_PROP_FRAME_COUNT))
        K = _depth_K(intr_all.get(dev_id, {}))
        extr = _extrinsics(extr_all, dev_id)
        projector = _load_projector(raw, dev_id, meta_all) or identity
        if projector is identity:
            logger.warning(
                "extract %s/%s: no k4a calibration — assuming depth is "
                "colour-aligned (identity colour→depth map)", ep.id, dev_id,
            )
        det_backend = load_backend(backend_name, **be_kw)

        cap = cv2.VideoCapture(str(mp4))
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            det_rgb = rgb[:, ::-1].copy() if flip else rgb
            if idx < len(depth_files):
                depth_mm = np.load(depth_files[idx])
                depth_m = depth_mm.astype(np.float32) / 1000.0
                depth_m[depth_m == 0] = np.nan
            else:
                depth_m = np.full((h, w), np.nan, dtype=np.float32)

            ts_us = _row_ts_us(ts_list, idx, dev_id)
            det = det_backend.detect(
                PreparedFrame(rgb=det_rgb, depth_m=depth_m, depth_K=K,
                              device_id=dev_id, timestamp_us=ts_us),
                hand,
            )
            idx += 1
            if idx % 30 == 0 or idx == total:
                report(stage="extract", camera=dev_id, frame=idx, total=total)
            if det is None:
                continue
            if flip:  # undo the mirror so coords are in the real colour frame
                for lm in det.points:
                    det.points[lm][0] = w - 1.0 - det.points[lm][0]

            prepared = PreparedFrame(rgb=rgb, depth_m=depth_m, depth_K=K,
                                     device_id=dev_id, timestamp_us=ts_us)
            lms = lift_to_3d(det, prepared, projector)
            if lms is None:
                continue
            world = camera_landmarks_to_world(lms, extr)
            if keep is not None:
                world = {lm: p for lm, p in world.items() if int(lm) in keep}
            if not world:
                continue

            w_cam = lms.weights or {}
            records.append((
                dev_id,
                SkeletonFrame(device_id=dev_id, points=world, timestamp_us=int(ts_us)),
                {lm: w_cam.get(lm, 0.0) for lm in world},
            ))
        cap.release()
        det_backend.close()

    write_rec(ep.rec_npz, records)
    mark_stage(
        ep, "extract", frames=len(records), backend=backend_name, model=model or "",
        track_lm=sorted(keep) if keep else "all",
    )
    logger.info("extract %s: %d frames -> %s", ep.id, len(records), ep.rec_npz)
    return str(ep.rec_npz)
