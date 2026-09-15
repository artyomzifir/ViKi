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
        self._deproj_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._transformation = None
        self._color_to_depth_fn = None

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

    def _transformation_handle(self):
        if self._transformation is None:
            self._transformation = self._lib.k4a_transformation_create(self._calib)
            if not self._transformation:
                raise RuntimeError("k4a_transformation_create failed")
        return self._transformation

    def close(self) -> None:
        handle = self._transformation
        if handle:
            self._lib.k4a_transformation_destroy(handle)
            self._transformation = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown safety
            pass

    def _align_bgra_to_depth(
        self, color_bgra: np.ndarray, depth_mm: np.ndarray
    ) -> np.ndarray | None:
        color = np.asarray(color_bgra, dtype=np.uint8)
        depth = np.ascontiguousarray(depth_mm, dtype=np.uint16)
        if color.ndim != 3 or color.shape[2] != 4 or depth.ndim != 2:
            raise ValueError("expected BGRA HxWx4 and depth HxW")
        color = np.ascontiguousarray(color)

        # Official k4a_image_format_t values.  The camera module's legacy
        # constants describe capture-format choices and must not be reused for
        # these explicitly allocated SDK images.
        image_format_bgra32 = 3
        image_format_depth16 = 4
        image_type = self._k.K4AImage
        depth_image = image_type(None)
        color_image = image_type(None)
        aligned_image = image_type(None)

        def create(fmt: int, width: int, height: int, stride: int, out) -> None:
            result = self._lib.k4a_image_create(
                fmt, int(width), int(height), int(stride), ctypes.byref(out)
            )
            if result != self._k.K4A_RESULT_SUCCEEDED or not out:
                raise RuntimeError(f"k4a_image_create failed (format={fmt}, result={result})")

        try:
            dh, dw = depth.shape
            ch, cw = color.shape[:2]
            create(image_format_depth16, dw, dh, dw * 2, depth_image)
            create(image_format_bgra32, cw, ch, cw * 4, color_image)
            create(image_format_bgra32, dw, dh, dw * 4, aligned_image)
            ctypes.memmove(
                self._lib.k4a_image_get_buffer(depth_image),
                depth.ctypes.data,
                depth.nbytes,
            )
            ctypes.memmove(
                self._lib.k4a_image_get_buffer(color_image),
                color.ctypes.data,
                color.nbytes,
            )

            if self._color_to_depth_fn is None:
                fn_type = ctypes.CFUNCTYPE(
                    ctypes.c_int,
                    self._k.K4ATransformation,
                    self._k.K4AImage,
                    self._k.K4AImage,
                    self._k.K4AImage,
                )
                self._color_to_depth_fn = fn_type(
                    ("k4a_transformation_color_image_to_depth_camera", self._lib)
                )
            result = self._color_to_depth_fn(
                self._transformation_handle(), depth_image, color_image, aligned_image
            )
            if result != self._k.K4A_RESULT_SUCCEEDED:
                logger.warning("k4a color→depth transform failed (result=%s)", result)
                return None
            size = int(self._lib.k4a_image_get_size(aligned_image))
            raw = ctypes.string_at(self._lib.k4a_image_get_buffer(aligned_image), size)
            return np.frombuffer(raw, np.uint8).reshape(dh, dw, 4).copy()
        finally:
            for image in (aligned_image, color_image, depth_image):
                if image:
                    self._lib.k4a_image_release(image)

    def align_color_to_depth(
        self, color_bgr: np.ndarray, depth_mm: np.ndarray
    ) -> np.ndarray | None:
        """Warp native-resolution BGR into the recorded depth geometry.

        Unlike the pinhole colour lookup used historically by ``cloud.py``,
        this calls libk4a's calibrated transformation and therefore preserves
        the SDK lens model. Both input images are allocated explicitly; the
        live backend's old, disabled alignment path passed an unallocated output
        handle and is intentionally not reused here.
        """
        color = np.asarray(color_bgr, dtype=np.uint8)
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("expected BGR HxWx3")
        bgra = np.empty((*color.shape[:2], 4), np.uint8)
        bgra[:, :, :3] = color
        bgra[:, :, 3] = 255
        aligned = self._align_bgra_to_depth(bgra, depth_mm)
        return None if aligned is None else aligned[:, :, :3]

    def align_masks_to_depth(
        self, masks: np.ndarray, depth_mm: np.ndarray
    ) -> np.ndarray | None:
        """Warp colour-space binary masks into depth pixels with libk4a.

        Four independent masks are packed into BGRA channels per SDK call. This
        preserves overlaps and avoids one transformation per object. A 0.5
        threshold turns the SDK's sub-pixel colour interpolation back into a
        binary membership map.
        """
        source = np.asarray(masks, dtype=bool)
        if source.ndim != 3:
            raise ValueError("expected masks MxHxW")
        result = np.zeros((len(source), *np.asarray(depth_mm).shape), dtype=bool)
        for start in range(0, len(source), 4):
            chunk = source[start:start + 4]
            packed = np.zeros((*source.shape[1:], 4), np.uint8)
            for channel, mask in enumerate(chunk):
                packed[:, :, channel] = mask.astype(np.uint8) * 255
            aligned = self._align_bgra_to_depth(packed, depth_mm)
            if aligned is None:
                return None
            for channel in range(len(chunk)):
                result[start + channel] = aligned[:, :, channel] >= 128
        return result

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

    def color_pixel_to_ray(self, u: float, v: float) -> np.ndarray | None:
        """Colour pixel → exact SDK-undistorted ray in the colour camera frame.

        The returned vector is normalised to ``z=1``.  It is used by the
        ChArUco bundle solver so calibration and depth-cloud reconstruction use
        the same factory K4A lens model, independent of output resolution.
        """
        k = self._k
        src = k.K4AFloat2(float(u), float(v))
        dst = k.K4AFloat3()
        valid = ctypes.c_int()
        res = self._lib.k4a_calibration_2d_to_3d(
            self._calib,
            ctypes.byref(src),
            1000.0,
            k.K4A_CALIBRATION_TYPE_COLOR,
            k.K4A_CALIBRATION_TYPE_COLOR,
            ctypes.byref(dst),
            ctypes.byref(valid),
        )
        if (
            res == k.K4A_RESULT_SUCCEEDED
            and valid.value
            and np.isfinite((dst.x, dst.y, dst.z)).all()
            and abs(float(dst.z)) > 1e-9
        ):
            return np.array([dst.x / dst.z, dst.y / dst.z, 1.0], dtype=np.float64)
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

    def color_deproject_maps(self, dh: int, dw: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised equivalent of :meth:`deproject_depth_px_to_color3d` over a
        whole ``dh × dw`` depth image.

        ``k4a_calibration_2d_to_3d(DEPTH→COLOR)`` is *affine* in the input depth:
        it undistorts the depth pixel to a ray, scales the ray by ``z``, then
        applies the fixed depth→colour rigid transform ``(R, t)`` — i.e.
        ``p_color(u, v, z) = z · A[v, u] + B[v, u]`` with ``A = R · ray(u, v)``
        and ``B = t`` (a constant, the ~32 mm sensor offset). So we call the SDK
        twice per pixel *once* to recover ``(A, B)`` (exact same lens model, no
        pinhole approximation), cache it per image size, and every frame after
        that is pure NumPy. Pixels the SDK rejects get NaN in both maps.

        Returns ``(A, B)`` each shape ``(dh, dw, 3)``, millimetres.
        """
        key = (int(dh), int(dw))
        hit = self._deproj_cache.get(key)
        if hit is not None:
            return hit
        A = np.full((dh, dw, 3), np.nan, dtype=np.float64)
        B = np.full((dh, dw, 3), np.nan, dtype=np.float64)
        for v in range(dh):
            for u in range(dw):
                p1 = self.deproject_depth_px_to_color3d(u, v, 1000.0)
                p2 = self.deproject_depth_px_to_color3d(u, v, 2000.0)
                if p1 is not None and p2 is not None:
                    a = (p2 - p1) / 1000.0
                    A[v, u] = a
                    B[v, u] = p1 - a * 1000.0
        self._deproj_cache[key] = (A, B)
        logger.info("k4a_offline: built %dx%d colour-deprojection maps", dw, dh)
        return A, B

    def depth3d_to_color3d(self, xyz_m) -> np.ndarray | None:
        """3-D point (metres) in the depth camera frame → the colour camera frame
        (``k4a_calibration_3d_to_3d``). The offline lift deprojects with the depth
        intrinsics but the recorded extrinsics are the colour camera's ChArUco
        pose, so points must be moved into the colour frame before the world
        transform."""
        k = self._k
        src = k.K4AFloat3(float(xyz_m[0]) * 1000.0, float(xyz_m[1]) * 1000.0,
                          float(xyz_m[2]) * 1000.0)
        dst = k.K4AFloat3()
        res = self._lib.k4a_calibration_3d_to_3d(
            self._calib, ctypes.byref(src),
            k.K4A_CALIBRATION_TYPE_DEPTH, k.K4A_CALIBRATION_TYPE_COLOR,
            ctypes.byref(dst),
        )
        if res == k.K4A_RESULT_SUCCEEDED:
            return np.array([dst.x, dst.y, dst.z], dtype=np.float64) / 1000.0
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
