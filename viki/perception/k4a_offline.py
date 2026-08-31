"""
viki.perception.k4a_offline
---------------------------
Rebuild an Azure Kinect ``k4a_calibration_t`` from the raw blob captured at record
time (``raw/<dev>_k4a_calib.bin``) and expose the colour↔depth projection the
offline perception + point-cloud stages need. No device required — the k4a
calibration/transformation maths runs purely on the blob.

``K4ACalibration`` satisfies :class:`viki.contracts.DepthProjector` (it has
``project_color_to_depth``), so it can be handed straight to
``viki.perception.geometry.lift_to_3d`` in place of the identity projector.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# k4a_depth_mode_t / k4a_color_resolution_t enum ints (see viki/cameras/kinect.py).
_DEPTH_MODE_NAME_TO_INT = {
    "NFOV_2X2BINNED": 1,
    "NFOV_UNBINNED": 2,
    "WFOV_2X2BINNED": 3,
    "WFOV_UNBINNED": 4,
}
_COLOR_RES_TO_INT = {(1280, 720): 1, (1920, 1080): 2, (2048, 1536): 4}


class K4ACalibration:
    """A rebuilt ``k4a_calibration_t`` plus the three projections we use offline.

    All SDK entry points used here (``k4a_calibration_2d_to_2d`` / ``2d_to_3d`` /
    ``3d_to_2d``) are already bound in :mod:`viki.cameras.kinect`; this class only
    adds ``k4a_calibration_get_from_raw`` on top.
    """

    def __init__(self, buf: ctypes.Array, kinect_mod) -> None:
        self._buf = buf  # keep the 8 KiB struct buffer alive
        self._calib = ctypes.cast(buf, ctypes.c_void_p)
        self._k = kinect_mod
        self._lib = kinect_mod._lib

    # ------------------------------------------------------------------

    @classmethod
    def from_blob(
        cls, blob: bytes, depth_mode_int, color_res_int, tag: str = ""
    ) -> "K4ACalibration | None":
        """Rebuild ``k4a_calibration_t`` from a raw calibration blob + the target
        depth-mode / colour-resolution enum ints. ``None`` if the ints are
        missing, libk4a is unavailable, or the SDK rejects the blob."""
        if depth_mode_int is None or color_res_int is None:
            logger.warning("k4a_offline[%s]: missing depth-mode / colour-res ints", tag)
            return None
        try:
            from viki.cameras import kinect as _k
        except OSError as exc:  # libk4a not installed
            logger.warning("k4a_offline[%s]: libk4a unavailable (%s)", tag, exc)
            return None
        if not blob:
            return None
        if not blob.endswith(b"\x00"):
            blob += b"\x00"
        out = ctypes.create_string_buffer(8192)
        res = _k._lib.k4a_calibration_get_from_raw(
            blob, len(blob), int(depth_mode_int), int(color_res_int), out
        )
        if res != _k.K4A_RESULT_SUCCEEDED:
            logger.warning(
                "k4a_offline[%s]: k4a_calibration_get_from_raw failed (res=%s)", tag, res
            )
            return None
        logger.info("k4a_offline[%s]: rebuilt calibration from raw blob", tag)
        return cls(out, _k)

    @classmethod
    def from_episode(cls, raw_dir, dev_id: str, meta: dict | None) -> "K4ACalibration | None":
        """Build from ``raw/<dev>_k4a_calib.bin`` + ``meta['cameras'][dev]``.

        Returns ``None`` (caller falls back to identity / preset) when the blob is
        absent, the enum ints can't be resolved, or libk4a is unavailable.
        """
        raw_dir = Path(raw_dir)
        cam = ((meta or {}).get("cameras") or {}).get(dev_id, {}) or {}
        blob_path = raw_dir / cam.get("k4a_calib", f"{dev_id}_k4a_calib.bin")
        if not blob_path.is_file():
            return None

        depth_int = cam.get("k4a_depth_mode_int")
        color_int = cam.get("k4a_color_res_int")
        if depth_int is None or color_int is None:
            req = cam.get("requested") or {}
            depth_int = _DEPTH_MODE_NAME_TO_INT.get(req.get("depth_mode"))
            color_int = _COLOR_RES_TO_INT.get(
                (int(req.get("color_width", 0)), int(req.get("color_height", 0)))
            )
        return cls.from_blob(blob_path.read_bytes(), depth_int, color_int, tag=dev_id)

    # ── projections ───────────────────────────────────────────────────

    def project_color_to_depth(self, u: float, v: float, z: float) -> tuple[float, float] | None:
        """Colour pixel + expected depth ``z`` (metres) → depth-image pixel."""
        k = self._k
        src = k.K4AFloat2(float(u), float(v))
        dst = k.K4AFloat2()
        valid = ctypes.c_int()
        res = self._lib.k4a_calibration_2d_to_2d(
            self._calib, ctypes.byref(src), float(z) * 1000.0,
            k.K4A_CALIBRATION_TYPE_COLOR, k.K4A_CALIBRATION_TYPE_DEPTH,
            ctypes.byref(dst), ctypes.byref(valid),
        )
        if res == k.K4A_RESULT_SUCCEEDED and valid.value:
            return (dst.x, dst.y)
        return None

    def deproject_depth_px(self, u: float, v: float, z_mm: float) -> np.ndarray | None:
        """Depth-image pixel + depth ``z_mm`` → 3-D point (mm) in the depth frame."""
        k = self._k
        src = k.K4AFloat2(float(u), float(v))
        dst = k.K4AFloat3()
        valid = ctypes.c_int()
        res = self._lib.k4a_calibration_2d_to_3d(
            self._calib, ctypes.byref(src), float(z_mm),
            k.K4A_CALIBRATION_TYPE_DEPTH, k.K4A_CALIBRATION_TYPE_DEPTH,
            ctypes.byref(dst), ctypes.byref(valid),
        )
        if res == k.K4A_RESULT_SUCCEEDED and valid.value:
            return np.array([dst.x, dst.y, dst.z], dtype=np.float64)
        return None

    def deproject_depth_px_to_color3d(self, u: float, v: float, z_mm: float) -> np.ndarray | None:
        """Depth-image pixel + depth ``z_mm`` → 3-D point (mm) in the **colour**
        camera frame — one SDK call that folds in the depth↔colour extrinsic. The
        point is then colourised by a plain pinhole projection with ``K_color``
        and placed in the world with the colour camera's recorded extrinsics."""
        k = self._k
        src = k.K4AFloat2(float(u), float(v))
        dst = k.K4AFloat3()
        valid = ctypes.c_int()
        res = self._lib.k4a_calibration_2d_to_3d(
            self._calib, ctypes.byref(src), float(z_mm),
            k.K4A_CALIBRATION_TYPE_DEPTH, k.K4A_CALIBRATION_TYPE_COLOR,
            ctypes.byref(dst), ctypes.byref(valid),
        )
        if res == k.K4A_RESULT_SUCCEEDED and valid.value:
            return np.array([dst.x, dst.y, dst.z], dtype=np.float64)
        return None

    def depth_xyz_to_color_px(self, xyz_mm) -> tuple[float, float] | None:
        """3-D point (mm, depth frame) → colour-image pixel, for colourising a cloud."""
        k = self._k
        p = k.K4AFloat3(float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2]))
        pix = k.K4AFloat2()
        valid = ctypes.c_int()
        res = self._lib.k4a_calibration_3d_to_2d(
            self._calib, ctypes.byref(p),
            k.K4A_CALIBRATION_TYPE_DEPTH, k.K4A_CALIBRATION_TYPE_COLOR,
            ctypes.byref(pix), ctypes.byref(valid),
        )
        if res == k.K4A_RESULT_SUCCEEDED and valid.value:
            return (pix.x, pix.y)
        return None
