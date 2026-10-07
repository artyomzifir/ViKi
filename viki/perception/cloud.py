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


def _depth_K(entry: dict) -> np.ndarray | None:
    intr = (entry or {}).get("depth")
    if not intr or "fx" not in intr:
        return None
    return np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )


def _depth_edge_keep_mask(
    depth_mm: np.ndarray,
    K_depth: np.ndarray | None,
    *,
    radius_rad: float,
    jump_mm: float,
    jump_relative: float,
) -> np.ndarray:
    """Mark depth samples whose angular neighbourhood is one coherent surface.

    Kinect mixed pixels occur around a depth discontinuity.  A pixel radius is
    deliberately not a parameter: ``radius_rad * focal_length_px`` keeps the
    same ray-space footprint when the recorded depth resolution changes.
    Thresholds stay metric (millimetres plus a depth-relative allowance).
    Missing intrinsics disable the filter rather than inventing a resolution-
    dependent fallback.
    """
    depth = np.asarray(depth_mm)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any() or K_depth is None or radius_rad <= 0:
        return valid
    focal_px = float(np.sqrt(abs(float(K_depth[0, 0] * K_depth[1, 1]))))
    if not np.isfinite(focal_px) or focal_px <= 0:
        return valid
    radius_px = max(1, int(np.ceil(float(radius_rad) * focal_px)))
    kernel = np.ones((2 * radius_px + 1, 2 * radius_px + 1), np.uint8)
    values = depth.astype(np.float32, copy=False)
    local_min = cv2.erode(np.where(valid, values, np.float32(1e9)), kernel)
    local_max = cv2.dilate(np.where(valid, values, np.float32(0.0)), kernel)
    span = local_max - local_min
    allowed = np.maximum(float(jump_mm), float(jump_relative) * values)
    coherent = (local_min < np.float32(1e8)) & (span <= allowed)
    return valid & coherent


def _fps_from_timestamps(raw: Path) -> float:
    ts = _read_json(raw / "timestamps.json")
    if not isinstance(ts, list) or len(ts) < 2:
        return 15.0
    us = np.array([e.get("sync_us", 0) for e in ts], dtype=np.float64)
    d = np.diff(us)
    d = d[d > 0]
    return float(1e6 / np.median(d)) if d.size else 15.0


def _voxel_downsample_indices(xyz: np.ndarray, leaf: float) -> np.ndarray:
    """Return stable indices of the first point retained in every voxel."""
    if leaf <= 0 or len(xyz) == 0:
        return np.arange(len(xyz), dtype=np.int64)
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
    return idx.astype(np.int64, copy=False)


def _voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    leaf: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = _voxel_downsample_indices(xyz, leaf)
    return xyz[idx], rgb[idx]


def _bbox_to_frame(bbox, T_world_display) -> list:
    """Re-express a world-frame AABB in the rig frame. ``T_world_display`` maps
    rig→world; its inverse takes the 8 box corners back to the rig frame and we
    take their AABB (a loose bound when the anchor is rotated — fine for a viz
    crop). Identity / missing ⇒ the box is returned unchanged."""
    if not bbox or len(bbox) != 6 or not T_world_display:
        return list(bbox) if bbox else []
    T = np.asarray(T_world_display, float).reshape(4, 4)
    if np.allclose(T, np.eye(4)):
        return list(bbox)
    Tinv = np.linalg.inv(T)
    x0, x1, y0, y1, z0, z1 = bbox
    corners = np.array([[x, y, z, 1.0]
                        for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    rig = (Tinv @ corners.T).T[:, :3]
    lo, hi = rig.min(axis=0), rig.max(axis=0)
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])]


def _bbox_mask(xyz: np.ndarray, bbox) -> np.ndarray:
    if not bbox or len(bbox) != 6:
        return np.ones(len(xyz), dtype=bool)
    x0, x1, y0, y1, z0, z1 = bbox
    return (
        (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1)
        & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
        & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
    )


def _crop_bbox(xyz: np.ndarray, rgb: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray]:
    m = _bbox_mask(xyz, bbox)
    return xyz[m], rgb[m]


def _camera_samples(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    stride: int,
    K_color: np.ndarray | None,
    cal,
    T_world_cam: np.ndarray,
    bg_mm: np.ndarray | None = None,
    bg_tol_mm: float = 50.0,
    *,
    K_depth: np.ndarray | None = None,
    edge_filter: bool = False,
    edge_radius_rad: float = 0.004,
    edge_jump_mm: float = 30.0,
    edge_jump_relative: float = 0.02,
    exact_color_projection: bool = False,
    return_depth_uv: bool = False,
    depth_bias_mm: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """One camera frame → world points, RGB, and source colour pixels.

    Fully vectorised. When a k4a calibration is available the depth→colour-3D
    deprojection uses a precomputed ``(A, B)`` ray map (exact SDK lens model,
    built once and cached) so every frame is pure NumPy — no per-pixel ctypes
    loop. When ``bg_mm`` (the calibrated empty-scene depth) is given, pixels
    within ``bg_tol_mm`` of the background are dropped before deprojection —
    only the operator + objects survive.
    """
    dh, dw = depth_mm.shape[:2]
    vs, us = np.mgrid[0:dh:stride, 0:dw:stride]
    us = us.ravel()
    vs = vs.ravel()
    z = depth_mm[vs, us].astype(np.float64)
    keep = z > 0
    if edge_filter:
        edge_keep = _depth_edge_keep_mask(
            depth_mm,
            K_depth,
            radius_rad=edge_radius_rad,
            jump_mm=edge_jump_mm,
            jump_relative=edge_jump_relative,
        )
        keep &= edge_keep[vs, us]
    if bg_mm is not None and bg_mm.shape == depth_mm.shape:
        bz = bg_mm[vs, us].astype(np.float64)
        # a pixel is static scene when the background has a reading there and the
        # current depth is within tolerance of it — drop those.
        keep &= ~((bz > 0) & (np.abs(z - bz) <= float(bg_tol_mm)))
    us, vs, z = us[keep], vs[keep], z[keep]
    if depth_bias_mm:
        z -= float(depth_bias_mm)
        valid_range = z > 0
        us, vs, z = us[valid_range], vs[valid_range], z[valid_range]
    if us.size == 0:
        empty = (
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.uint8),
            np.empty((0, 2), np.int32),
        )
        return (*empty, np.empty((0, 2), np.int32)) if return_depth_uv else empty

    ch, cw = color_bgr.shape[:2]

    if cal is not None:
        A, B = cal.color_deproject_maps(dh, dw)  # (dh, dw, 3) each, mm; cached
        pts = z[:, None] * A[vs, us] + B[vs, us]  # mm, colour camera frame
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite] / 1000.0  # mm → m
        us, vs = us[finite], vs[finite]
        if pts.size == 0:
            empty = (
                np.empty((0, 3), np.float32),
                np.empty((0, 3), np.uint8),
                np.empty((0, 2), np.int32),
            )
            return (*empty, np.empty((0, 2), np.int32)) if return_depth_uv else empty
        uu = pts[:, 0] / pts[:, 2] * K_color[0, 0] + K_color[0, 2]
        vv = pts[:, 1] / pts[:, 2] * K_color[1, 1] + K_color[1, 2]
    else:
        # depth assumed colour-aligned: pinhole deproject at the depth pixel
        if K_color is None:
            return (
                np.empty((0, 3), np.float32),
                np.empty((0, 3), np.uint8),
                np.empty((0, 2), np.int32),
            )
        zm = z / 1000.0
        X = (us - K_color[0, 2]) * zm / K_color[0, 0]
        Y = (vs - K_color[1, 2]) * zm / K_color[1, 1]
        pts = np.stack([X, Y, zm], axis=1)
        uu, vv = us.astype(np.float64), vs.astype(np.float64)

    ui = np.clip(np.round(uu), 0, cw - 1).astype(np.int64)
    vi = np.clip(np.round(vv), 0, ch - 1).astype(np.int64)
    aligned_bgr = None
    if exact_color_projection and cal is not None:
        align = getattr(cal, "align_color_to_depth", None)
        if align is not None:
            aligned_bgr = align(color_bgr, depth_mm)
    if aligned_bgr is not None and aligned_bgr.shape[:2] == depth_mm.shape:
        rgb = aligned_bgr[vs, us][:, ::-1].copy()
    else:
        rgb = color_bgr[vi, ui][:, ::-1].copy()  # BGR → RGB

    world = pts @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    color_uv = np.stack([ui, vi], axis=1).astype(np.int32)
    result = (world.astype(np.float32), rgb.astype(np.uint8), color_uv)
    if return_depth_uv:
        return (*result, np.stack([us, vs], axis=1).astype(np.int32))
    return result


def _camera_cloud(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    stride: int,
    K_color: np.ndarray | None,
    cal,
    T_world_cam: np.ndarray,
    bg_mm: np.ndarray | None = None,
    bg_tol_mm: float = 50.0,
    *,
    K_depth: np.ndarray | None = None,
    edge_filter: bool = False,
    edge_radius_rad: float = 0.004,
    edge_jump_mm: float = 30.0,
    edge_jump_relative: float = 0.02,
    exact_color_projection: bool = False,
    depth_bias_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for the Viewer cloud builder."""
    xyz, rgb, _color_uv = _camera_samples(
        color_bgr,
        depth_mm,
        stride,
        K_color,
        cal,
        T_world_cam,
        bg_mm=bg_mm,
        bg_tol_mm=bg_tol_mm,
        K_depth=K_depth,
        edge_filter=edge_filter,
        edge_radius_rad=edge_radius_rad,
        edge_jump_mm=edge_jump_mm,
        edge_jump_relative=edge_jump_relative,
        exact_color_projection=exact_color_projection,
        depth_bias_mm=depth_bias_mm,
    )
    return xyz, rgb


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
    bg_subtract: bool | None = None,
    bg_tol_mm: float | None = None,
    edge_filter: bool | None = None,
    edge_radius_rad: float | None = None,
    edge_jump_mm: float | None = None,
    edge_jump_relative: float | None = None,
    exact_color_projection: bool | None = None,
    depth_biases_by_frame: dict[str, list[float]] | None = None,
    output_dir: Path | None = None,
    report=None,
) -> str:
    """Write ``cloud/<i>.bin`` + ``cloud/meta.json`` for the episode. Returns the dir.

    ``stride`` / ``voxel`` / ``bbox`` / ``max_points`` / ``bg_subtract`` /
    ``bg_tol_mm`` / edge filtering / exact colour projection default to the
    ``CLOUD_*`` config keys.
    ``bg_subtract`` drops points matching the calibration preset's empty-scene
    depth. ``report(stage="cloud", frame=i, total=N)`` drives a progress bar.
    Experimental ``depth_biases_by_frame`` requires a separate ``output_dir``;
    the raw depth and canonical cloud remain unchanged.
    """
    if depth_biases_by_frame is not None and (
        output_dir is None or Path(output_dir).resolve() == ep.cloud_dir.resolve()
    ):
        raise ValueError("depth-bias comparison must use a separate output directory")
    raw = ep.raw_dir
    intr_all = _read_json(raw / "intrinsics.json")
    extr_all = _read_json(raw / "extrinsics.json")
    meta_all = _read_json(ep.meta_path)

    stride = max(1, int(stride if stride is not None else getattr(config, "CLOUD_STRIDE", 6)))
    voxel = float(voxel if voxel is not None else getattr(config, "CLOUD_VOXEL_M", 0.005))
    bbox = list(bbox if bbox is not None else (getattr(config, "CLOUD_WORKSPACE_BBOX", []) or []))
    # The workspace AABB is a world/display-frame concept; the cloud is built in
    # the RIG (reference-camera) frame (T_world_display never touches the points,
    # spec §5). Carry the box into the rig frame for the crop test instead.
    bbox = _bbox_to_frame(bbox, _read_json(raw / "world_anchor.json").get("T_world_display"))
    cap = int(
        max_points
        if max_points is not None
        else getattr(config, "CLOUD_MAX_POINTS_PER_FRAME", 40000)
    )
    bg_subtract = bool(
        getattr(config, "CLOUD_BG_SUBTRACT", True)
        if bg_subtract is None
        else bg_subtract
    )
    bg_tol_mm = float(
        getattr(config, "CLOUD_BG_TOLERANCE_MM", 50.0)
        if bg_tol_mm is None
        else bg_tol_mm
    )
    edge_filter = bool(
        getattr(config, "CLOUD_EDGE_FILTER", True)
        if edge_filter is None
        else edge_filter
    )
    edge_radius_rad = float(
        getattr(config, "CLOUD_EDGE_RADIUS_RAD", 0.004)
        if edge_radius_rad is None else edge_radius_rad
    )
    edge_jump_mm = float(
        getattr(config, "CLOUD_EDGE_JUMP_MM", 30.0)
        if edge_jump_mm is None else edge_jump_mm
    )
    edge_jump_relative = float(
        getattr(config, "CLOUD_EDGE_JUMP_RELATIVE", 0.02)
        if edge_jump_relative is None else edge_jump_relative
    )
    exact_color_projection = bool(
        getattr(config, "CLOUD_EXACT_COLOR_PROJECTION", True)
        if exact_color_projection is None else exact_color_projection
    )

    preset = (meta_all or {}).get("calibration_preset")
    bg_by_dev: dict = {}
    if bg_subtract and preset:
        try:
            from viki.calibration import presets as _presets

            for mp4 in raw.glob("*.mp4"):
                bd = _presets.background_depth(preset, mp4.stem)
                if bd is not None:
                    bg_by_dev[mp4.stem] = bd
            if bg_by_dev:
                logger.info("cloud %s: background subtract for %s (tol %.0f mm)",
                            ep.id, sorted(bg_by_dev), bg_tol_mm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cloud %s: background load failed (%s)", ep.id, exc)

    from viki.perception.k4a_offline import K4ACalibration
    from viki.perception.rs_offline import RealSenseCalibration

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
            cal = K4ACalibration.from_episode(raw, dev, meta_all) \
                or RealSenseCalibration.from_episode(raw, dev, meta_all)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cloud %s/%s: depth calib unavailable (%s)", ep.id, dev, exc)
        if cal is None:
            logger.warning(
                "cloud %s/%s: no depth calib — pinhole fallback (assumes colour-aligned depth)",
                ep.id, dev,
            )
        cams.append({
            "dev": dev,
            "cap": cv2.VideoCapture(str(mp4)),
            "depth_dir": raw / f"{dev}_depth",
            "K": _color_K(intr_all.get(dev, {})),
            "K_depth": _depth_K(intr_all.get(dev, {})),
            "cal": cal,
            "T": T,
            "bg": bg_by_dev.get(dev),
        })

    total = 0
    if cams:
        total = int(cams[0]["cap"].get(cv2.CAP_PROP_FRAME_COUNT)) or len(
            list(cams[0]["depth_dir"].glob("*.npy"))
        )
    if depth_biases_by_frame is not None:
        for camera in cams:
            series = depth_biases_by_frame.get(camera["dev"])
            if series is None or len(series) < total or not np.isfinite(series).all():
                for opened in cams:
                    opened["cap"].release()
                raise ValueError(f"incomplete depth-bias series for {camera['dev']}")
    out_dir = Path(output_dir) if output_dir is not None else ep.cloud_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.bin"):
        old.unlink()
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
            bias_series = (depth_biases_by_frame or {}).get(c["dev"], [])
            x, r = _camera_cloud(
                bgr,
                depth_mm,
                stride,
                c["K"],
                c["cal"],
                c["T"],
                bg_mm=c["bg"],
                bg_tol_mm=bg_tol_mm,
                K_depth=c["K_depth"],
                edge_filter=edge_filter,
                edge_radius_rad=edge_radius_rad,
                edge_jump_mm=edge_jump_mm,
                edge_jump_relative=edge_jump_relative,
                exact_color_projection=exact_color_projection,
                depth_bias_mm=bias_series[i] if bias_series else 0.0,
            )
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
        "edge_filter": edge_filter,
        "edge_radius_rad": edge_radius_rad,
        "edge_jump_mm": edge_jump_mm,
        "edge_jump_relative": edge_jump_relative,
        "exact_color_projection": exact_color_projection,
        "per_frame_points": per_frame,
        "experimental_depth_bias_mm": depth_biases_by_frame,
    }, indent=2))
    if output_dir is None:
        mark_stage(ep, "cloud", frames=i, points=int(sum(per_frame)))
    logger.info("cloud %s: %d frames → %s", ep.id, i, out_dir)
    return str(out_dir)
