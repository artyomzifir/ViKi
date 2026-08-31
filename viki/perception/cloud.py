"""
viki.perception.cloud
---------------------
Offline stage: ``episodes/<id>/raw/`` → ``episodes/<id>/cloud/``.

Builds one **world-frame coloured point cloud per synced frame** by fusing every
camera's colour + raw depth. Purely a visualisation artifact — nothing downstream
reads it; the Viewer tab streams it a frame at a time.

Per depth pixel (stride-decimated): deproject to the *colour* camera frame with
the rebuilt k4a calibration (one SDK call, folds in the depth↔colour extrinsic),
colourise by a pinhole projection with ``K_color``, then place in the world with
the colour camera's recorded ChArUco extrinsics — the same world frame the
skeleton and camera frusta already live in.

Fallback when an episode has no k4a calibration blob (RealSense, whose depth is
colour-aligned at capture, or older Kinect recordings): plain pinhole deproject
with ``K_color`` at the depth pixel, no extrinsic.

``cloud/<i:06d>.bin`` layout (little-endian): ``int32 n`` · ``float32[n*3]`` xyz
metres · ``uint8[n*3]`` rgb.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import cv2
import numpy as np

from viki import config
from viki.contracts import CalibrationExtrinsics
from viki.episode import mark_stage

logger = logging.getLogger(__name__)


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _color_K(entry: dict) -> np.ndarray | None:
    intr = (entry or {}).get("color") or entry
    if not intr or "fx" not in intr:
        return None
    return np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )


def _fps_from_timestamps(raw: Path) -> float:
    ts = _read_json(raw / "timestamps.json")
    if not isinstance(ts, list) or len(ts) < 2:
        return 15.0
    us = np.array([e.get("sync_us", 0) for e in ts], dtype=np.float64)
    d = np.diff(us)
    d = d[d > 0]
    return float(1e6 / np.median(d)) if d.size else 15.0


def _voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, leaf: float) -> tuple[np.ndarray, np.ndarray]:
    if leaf <= 0 or len(xyz) == 0:
        return xyz, rgb
    # Pack the 3 voxel indices into one int64 and de-dup on that — a single 1-D
    # sort, ~10x faster than np.unique(axis=0)'s structured lexsort (this is the
    # per-frame hot path of the whole cloud build).
    keys = np.floor((xyz - xyz.min(axis=0)) / leaf).astype(np.int64)
    span = keys.max(axis=0) + 1
    if span[0] * span[1] * span[2] < (1 << 62):
        code = keys[:, 0] + span[0] * (keys[:, 1] + span[1] * keys[:, 2])
        _, idx = np.unique(code, return_index=True)
    else:  # workspace too large to pack — fall back
        _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return xyz[idx], rgb[idx]


def _crop_bbox(xyz: np.ndarray, rgb: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray]:
    if not bbox or len(bbox) != 6:
        return xyz, rgb
    x0, x1, y0, y1, z0, z1 = bbox
    m = (
        (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1)
        & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
        & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
    )
    return xyz[m], rgb[m]


def _camera_cloud(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    stride: int,
    K_color: np.ndarray | None,
    cal,
    T_world_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One camera, one frame → (xyz_world Nx3 metres, rgb Nx3 uint8).

    Fully vectorised. When a k4a calibration is available the depth→colour-3D
    deprojection uses a precomputed ``(A, B)`` ray map (exact SDK lens model,
    built once and cached) so every frame is pure NumPy — no per-pixel ctypes
    loop.
    """
    dh, dw = depth_mm.shape[:2]
    vs, us = np.mgrid[0:dh:stride, 0:dw:stride]
    us = us.ravel()
    vs = vs.ravel()
    z = depth_mm[vs, us].astype(np.float64)
    keep = z > 0
    us, vs, z = us[keep], vs[keep], z[keep]
    if us.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    ch, cw = color_bgr.shape[:2]

    if cal is not None:
        A, B = cal.color_deproject_maps(dh, dw)  # (dh, dw, 3) each, mm; cached
        pts = z[:, None] * A[vs, us] + B[vs, us]  # mm, colour camera frame
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite] / 1000.0  # mm → m
        us, vs = us[finite], vs[finite]
        if pts.size == 0:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
        uu = pts[:, 0] / pts[:, 2] * K_color[0, 0] + K_color[0, 2]
        vv = pts[:, 1] / pts[:, 2] * K_color[1, 1] + K_color[1, 2]
    else:
        # depth assumed colour-aligned: pinhole deproject at the depth pixel
        if K_color is None:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
        zm = z / 1000.0
        X = (us - K_color[0, 2]) * zm / K_color[0, 0]
        Y = (vs - K_color[1, 2]) * zm / K_color[1, 1]
        pts = np.stack([X, Y, zm], axis=1)
        uu, vv = us.astype(np.float64), vs.astype(np.float64)

    ui = np.clip(np.round(uu), 0, cw - 1).astype(np.int64)
    vi = np.clip(np.round(vv), 0, ch - 1).astype(np.int64)
    rgb = color_bgr[vi, ui][:, ::-1].copy()  # BGR → RGB

    world = pts @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    return world.astype(np.float32), rgb.astype(np.uint8)


def _pack(xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    n = len(xyz)
    return (
        struct.pack("<i", n)
        + np.ascontiguousarray(xyz, np.float32).tobytes()
        + np.ascontiguousarray(rgb, np.uint8).tobytes()
    )


def build_cloud(
    ep,
    stride: int | None = None,
    *,
    voxel: float | None = None,
    bbox: list[float] | None = None,
    max_points: int | None = None,
    report=None,
) -> str:
    """Write ``cloud/<i>.bin`` + ``cloud/meta.json`` for the episode. Returns the dir.

    ``stride`` / ``voxel`` / ``bbox`` / ``max_points`` default to the ``CLOUD_*``
    config keys. ``report(stage="cloud", frame=i, total=N)`` is called ~every 15
    frames so a queue job can show a progress bar.
    """
    raw = ep.raw_dir
    intr_all = _read_json(raw / "intrinsics.json")
    extr_all = _read_json(raw / "extrinsics.json")
    meta_all = _read_json(ep.meta_path)

    stride = max(1, int(stride if stride is not None else getattr(config, "CLOUD_STRIDE", 6)))
    voxel = float(voxel if voxel is not None else getattr(config, "CLOUD_VOXEL_M", 0.005))
    bbox = list(bbox if bbox is not None else (getattr(config, "CLOUD_WORKSPACE_BBOX", []) or []))
    cap = int(max_points if max_points is not None else getattr(config, "CLOUD_MAX_POINTS_PER_FRAME", 40000))

    from viki.perception.k4a_offline import K4ACalibration

    cams = []
    for mp4 in sorted(raw.glob("*.mp4")):
        dev = mp4.stem
        e = extr_all.get(dev)
        if not e:
            logger.warning("cloud %s/%s: no extrinsics, skipping camera", ep.id, dev)
            continue
        T = CalibrationExtrinsics(
            rvec=np.asarray(e["rvec"], np.float64), tvec=np.asarray(e["tvec"], np.float64)
        ).transform_matrix
        cal = None
        try:
            cal = K4ACalibration.from_episode(raw, dev, meta_all)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cloud %s/%s: k4a calib unavailable (%s)", ep.id, dev, exc)
        if cal is None:
            logger.warning(
                "cloud %s/%s: no k4a blob — pinhole fallback (assumes colour-aligned depth)",
                ep.id, dev,
            )
        cams.append({
            "dev": dev,
            "cap": cv2.VideoCapture(str(mp4)),
            "depth_dir": raw / f"{dev}_depth",
            "K": _color_K(intr_all.get(dev, {})),
            "cal": cal,
            "T": T,
        })

    out_dir = ep.cloud_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.bin"):
        old.unlink()

    total = 0
    if cams:
        total = int(cams[0]["cap"].get(cv2.CAP_PROP_FRAME_COUNT)) or len(
            list(cams[0]["depth_dir"].glob("*.npy"))
        )
    if report:
        report(stage="cloud", frame=0, total=total)

    per_frame: list[int] = []
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    i = 0
    while True:
        frames_bgr = {}
        for c in cams:
            ok, bgr = c["cap"].read()
            if ok:
                frames_bgr[c["dev"]] = bgr
        if not frames_bgr:
            break

        xyz_parts, rgb_parts = [], []
        for c in cams:
            bgr = frames_bgr.get(c["dev"])
            dpath = c["depth_dir"] / f"{i:06d}.npy"
            if bgr is None or not dpath.is_file():
                continue
            depth_mm = np.load(dpath)
            if not depth_mm.any():
                continue
            x, r = _camera_cloud(bgr, depth_mm, stride, c["K"], c["cal"], c["T"])
            if len(x):
                xyz_parts.append(x)
                rgb_parts.append(r)

        if xyz_parts:
            xyz = np.concatenate(xyz_parts)
            rgb = np.concatenate(rgb_parts)
            xyz, rgb = _crop_bbox(xyz, rgb, bbox)
            xyz, rgb = _voxel_downsample(xyz, rgb, voxel)
            if cap and len(xyz) > cap:
                # even stride to the budget — deterministic and O(1) to pick,
                # and the cloud is already voxel-uniform so it stays uniform
                sel = np.linspace(0, len(xyz) - 1, cap).astype(np.int64)
                xyz, rgb = xyz[sel], rgb[sel]
            if len(xyz):
                lo = np.minimum(lo, xyz.min(axis=0))
                hi = np.maximum(hi, xyz.max(axis=0))
        else:
            xyz = np.empty((0, 3), np.float32)
            rgb = np.empty((0, 3), np.uint8)

        (out_dir / f"{i:06d}.bin").write_bytes(_pack(xyz, rgb))
        per_frame.append(int(len(xyz)))
        i += 1
        if report and i % 15 == 0:
            report(stage="cloud", frame=i, total=max(total, i))

    for c in cams:
        c["cap"].release()

    if report:
        report(stage="cloud", frame=i, total=i)

    bounds = (
        [float(v) for v in (*lo, *hi)]
        if np.isfinite(lo).all()
        else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    (out_dir / "meta.json").write_text(json.dumps({
        "n_frames": i,
        "fps": _fps_from_timestamps(raw),
        "bounds": bounds,  # [xmin,ymin,zmin, xmax,ymax,zmax]
        "voxel": voxel,
        "stride": stride,
        "cameras": [c["dev"] for c in cams],
        "per_frame_points": per_frame,
    }, indent=2))
    mark_stage(ep, "cloud", frames=i, points=int(sum(per_frame)))
    logger.info("cloud %s: %d frames → %s", ep.id, i, out_dir)
    return str(out_dir)
