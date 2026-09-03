"""
viki.cameras.realsense
----------------------
Backend for Intel RealSense D435i (and compatible D4xx models).

Like the Kinect backend it records **raw** colour + depth and leaves the
colour↔depth registration to the offline stages: ``get_rs_calibration()`` hands
the recorder both stream intrinsics + the depth→colour extrinsic, which
``viki.perception.rs_offline.RealSenseCalibration`` replays without a device.
On-device ``rs.align`` is opt-in (``align_to_color=True``) — it runs a full
depth→colour reprojection on the capture thread every frame and throttles the
stream badly at 720p+.

Frames come through librealsense's own dispatch thread via
``pipeline.start(config, callback)`` — the SDK pairs colour+depth and delivers
them, ``_on_frame`` copies the newest pair into a slot, and ``get_frame`` blocks
on it. No Python ``wait_for_frames`` poll loop fighting the GIL (which dropped
framesets and juddered the stream under recording load).

Dependency: pyrealsense2 >= 2.58
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from .base import CameraBackend, CameraIntrinsics, Frame
from .rs_math import deproject_pixel, extrinsic_matrix, project_point

logger = logging.getLogger(__name__)


def _intr_dict(intr) -> dict:
    """``rs2_intrinsics`` → the plain dict :mod:`viki.cameras.rs_math` wants."""
    return {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "ppx": float(intr.ppx), "ppy": float(intr.ppy),
        "width": int(intr.width), "height": int(intr.height),
        "model": str(intr.model).split(".")[-1].lower(),
        "coeffs": [float(c) for c in intr.coeffs],
    }

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError(
        "pyrealsense2 is not installed. Install with: uv add pyrealsense2"
    ) from e


class RealSenseBackend(CameraBackend):
    """
    Backend for Intel RealSense D435i.

    Parameters
    ----------
    serial : str | None
        Device serial number. If None, the first detected device is used.
    color_resolution : tuple[int, int]
        (width, height) for the colour stream. Default: 640x480.
    depth_resolution : tuple[int, int]
        (width, height) for the raw depth stream. Default: 848x480 — the D4xx
        depth sweet spot. Recorded as-is; the offline stages reproject it onto
        the colour plane from the stored calibration. (With ``align_to_color``
        this also bounds the on-thread ``rs.align`` cost — 1280x720 depth there
        throttles the stream to ~5 fps.)
    fps : int
        Frame rate. Default: 30.
    align_to_color : bool
        If True, resample depth onto the colour plane on the capture thread
        every frame (``project_color_to_depth`` then becomes the identity).
        Default: False — record raw and align offline, like the Kinect.
    depth_max_m : float
        Zero out depth past this range. The raw D4xx stream returns low-confidence
        estimates out to 20–30 m; ``rs.align`` used to drop them but the raw
        stream keeps them, and downstream only filters ``z > 0`` — so they become
        phantom points metres deep that swamp the workspace. The k4a NFOV depth
        engine self-limits near ~3.5 m; 6 m here is generous headroom.
    timeout_ms : int
        Frame wait timeout in milliseconds.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        color_resolution: tuple[int, int] = (640, 480),
        depth_resolution: tuple[int, int] = (848, 480),
        fps: int = 30,
        align_to_color: bool = False,
        depth_max_m: float = 6.0,
        timeout_ms: int = 5000,
    ) -> None:
        self._serial = serial
        self._color_res = color_resolution
        self._depth_res = depth_resolution
        self._fps = fps
        self._align_to_color = align_to_color
        self._depth_max_m = float(depth_max_m)
        self._timeout_ms = timeout_ms

        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._threshold: Optional["rs.threshold_filter"] = None
        self._resolved_serial: str = serial or ""
        self._running = False
        # newest (color, depth, ts_us) from the SDK callback, + a monotonically
        # rising sequence so get_frame() only returns a pair it hasn't seen.
        self._cv = threading.Condition()
        self._latest: Optional[tuple] = None
        self._latest_seq = 0
        self._last_seq = 0
        self._cb_error: Optional[str] = None
        # raw stream intrinsics + depth→colour extrinsic, captured at start().
        self._color_intr: dict | None = None
        self._depth_intr: dict | None = None
        self._color_ci: CameraIntrinsics | None = None
        self._depth_ci: CameraIntrinsics | None = None
        self._d2c_R: np.ndarray | None = None   # 3x3, depth→colour rotation
        self._d2c_t: np.ndarray | None = None   # 3, metres
        # metres per raw z16 unit, read from the device at start(). D4xx ships
        # 0.001 (1 unit == 1 mm), but the High-Accuracy preset and the D405
        # default to 0.0001 — the rest of the pipeline assumes millimetres, so
        # get_frame() rescales when this isn't 1 mm.
        self._depth_units_m: float = 0.001

    # ------------------------------------------------------------------
    # CameraBackend interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        self._pipeline = rs.pipeline()
        config = rs.config()

        if self._serial:
            config.enable_device(self._serial)

        config.enable_stream(
            rs.stream.color,
            self._color_res[0],
            self._color_res[1],
            rs.format.bgr8,
            self._fps,
        )
        dw, dh = self._depth_res
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, self._fps)

        # SDK processing blocks the callback runs — build them before start()
        # so an immediate first callback finds them.
        self._align = rs.align(rs.stream.color) if self._align_to_color else None
        self._threshold = (
            rs.threshold_filter(0.1, self._depth_max_m) if self._depth_max_m > 0 else None
        )

        try:
            profile = self._pipeline.start(config, self._on_frame)
        except RuntimeError:
            # This unit doesn't offer the requested depth profile — retry with
            # the depth stream matched to the colour resolution (always valid).
            logger.warning(
                "RealSense %s: depth %dx%d@%d unavailable, matching colour res",
                self._serial or "?", dw, dh, self._fps,
            )
            config = rs.config()
            if self._serial:
                config.enable_device(self._serial)
            config.enable_stream(
                rs.stream.color, self._color_res[0], self._color_res[1],
                rs.format.bgr8, self._fps,
            )
            config.enable_stream(
                rs.stream.depth, self._color_res[0], self._color_res[1],
                rs.format.z16, self._fps,
            )
            self._depth_res = self._color_res
            profile = self._pipeline.start(config, self._on_frame)

        if self._align_to_color and self._depth_res[0] * self._depth_res[1] > 900_000:
            logger.warning(
                "RealSense %s: depth %dx%d with align-to-colour — the reprojection "
                "runs on the capture thread and may cap the stream well below %d fps",
                self._serial or "?", self._depth_res[0], self._depth_res[1], self._fps,
            )

        dev = profile.get_device()
        self._resolved_serial = dev.get_info(rs.camera_info.serial_number)

        try:
            self._depth_units_m = float(
                dev.first_depth_sensor().get_depth_scale()
            )
        except Exception:  # noqa: BLE001 — fall back to the D4xx default
            self._depth_units_m = 0.001

        # Freeze the registration so the offline stages can replay it. Cache the
        # per-stream intrinsics as CameraIntrinsics now — rebuilding them from
        # the profile on every get_frame() is pure Python/SDK-wrapper work on the
        # capture thread, and under recording load (3× mp4 encode contending for
        # the GIL) that lag is enough to drop framesets and judder the stream.
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ci, di = cprof.get_intrinsics(), dprof.get_intrinsics()
        self._color_intr = _intr_dict(ci)
        self._depth_intr = _intr_dict(di)
        self._color_ci = self._camera_intrinsics(ci)
        self._depth_ci = self._camera_intrinsics(di)
        ext = dprof.get_extrinsics_to(cprof)
        self._d2c_R, self._d2c_t = extrinsic_matrix(ext.rotation, ext.translation)

        self._running = True

        # Block until the SDK callback has delivered the first usable pair, so a
        # caller that starts then immediately reads doesn't get a spurious
        # timeout (matches KinectBackend, which is streaming when start returns).
        with self._cv:
            if not self._cv.wait_for(
                lambda: self._latest is not None or self._cb_error,
                timeout=self._timeout_ms / 1000.0,
            ):
                raise RuntimeError(
                    f"RealSense {self._resolved_serial or '?'}: no frame within "
                    f"{self._timeout_ms} ms of start"
                )
            if self._cb_error:
                raise RuntimeError(f"RealSense callback failed: {self._cb_error}")

    def _on_frame(self, frame) -> None:
        """librealsense dispatch-thread callback: pair colour+depth, run the SDK
        filter chain, copy the newest pair into the slot. All the per-frame work
        lives here so ``get_frame`` is just a slot read."""
        try:
            fs = frame.as_frameset()
            if not fs:
                return
            if self._align is not None:
                fs = self._align.process(fs).as_frameset()
            c = fs.get_color_frame()
            d = fs.get_depth_frame()
            if not c or not d:
                return  # partial frameset — wait for a complete one
            if self._threshold is not None:
                d = self._threshold.process(d).as_depth_frame()
            # copy out of librealsense's frame pool — the frame handles die when
            # this callback returns and the buffers get recycled.
            color = np.array(c.get_data(), copy=True)   # HxWx3 BGR uint8
            depth = np.array(d.get_data(), copy=True)   # HxW uint16 z16 units
            if self._depth_units_m != 0.001:
                # normalise to millimetres so the pipeline's fixed /1000 holds
                depth = np.rint(
                    depth.astype(np.float32) * (self._depth_units_m * 1000.0)
                ).astype(np.uint16)
            ts_us = int(c.get_timestamp() * 1000)  # ms -> us
        except Exception as exc:  # noqa: BLE001 — surface it through get_frame
            with self._cv:
                self._cb_error = str(exc)
                self._cv.notify_all()
            return
        with self._cv:
            self._latest = (color, depth, ts_us)
            self._latest_seq += 1
            self._cv.notify_all()

    def stop(self) -> None:
        if self._pipeline and self._running:
            self._pipeline.stop()  # SDK joins its dispatch thread
        self._running = False
        self._pipeline = None
        self._align = None
        self._threshold = None
        with self._cv:
            self._latest = None
            self._latest_seq = self._last_seq = 0
            self._cb_error = None
            self._cv.notify_all()  # wake any get_frame() blocked on the slot

    def get_frame(self) -> Frame:
        with self._cv:
            if not self._cv.wait_for(
                lambda: (not self._running)
                or self._cb_error
                or self._latest_seq != self._last_seq,
                timeout=self._timeout_ms / 1000.0,
            ):
                raise TimeoutError("RealSense: no new frame within timeout")
            if not self._running:
                raise RuntimeError("RealSenseBackend is not started. Call start() first.")
            if self._cb_error:
                raise RuntimeError(f"RealSense callback failed: {self._cb_error}")
            color, depth, timestamp_us = self._latest
            self._last_seq = self._latest_seq

        return Frame(
            color=color,
            depth=depth,
            timestamp_us=timestamp_us,
            device_id=self._resolved_serial,
            color_intrinsics=self._color_ci,
            depth_intrinsics=self._depth_ci,
        )

    @property
    def device_id(self) -> str:
        return self._resolved_serial

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def config(self) -> dict:
        return {
            "color_width": int(self._color_res[0]),
            "color_height": int(self._color_res[1]),
            "fps": int(self._fps),
            "depth_mode": None,  # RealSense has no depth-mode enum
        }

    def project_color_to_depth(self, u: float, v: float, z: float) -> tuple[float, float] | None:
        """Colour pixel + expected range ``z`` (metres) → depth-image pixel.

        Identity when ``align_to_color`` is on (depth is already on the colour
        plane). Otherwise deproject the colour pixel at ``z``, apply the inverse
        depth→colour extrinsic, and reproject with the depth intrinsics — the
        same maths ``rs_offline.RealSenseCalibration`` uses offline.
        """
        if self._align is not None:
            return (float(u), float(v))
        if self._color_intr is None or self._d2c_R is None:
            return None
        p_col = deproject_pixel(self._color_intr, float(u), float(v), float(z))
        p_dep = self._d2c_R.T @ (p_col - self._d2c_t)
        uv = project_point(self._depth_intr, p_dep)
        return (float(uv[0]), float(uv[1]))

    def get_rs_calibration(self) -> dict | None:
        """Stream intrinsics + the depth→colour extrinsic, for the recorder to
        persist (``raw/<dev>_rs_calib.json``). ``None`` before :meth:`start`."""
        if self._color_intr is None or self._d2c_R is None:
            return None
        return {
            "color": self._color_intr,
            "depth": self._depth_intr,
            "depth_to_color": {
                "rotation": self._d2c_R.T.reshape(-1).tolist(),  # back to col-major
                "translation": self._d2c_t.tolist(),
            },
            "aligned": self._align is not None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _camera_intrinsics(intr) -> CameraIntrinsics:
        """``rs2_intrinsics`` → :class:`CameraIntrinsics`."""
        return CameraIntrinsics(
            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
            width=intr.width, height=intr.height,
            dist_coeffs=np.array(intr.coeffs),
        )

    @staticmethod
    def _get_intrinsics(stream_profile) -> CameraIntrinsics:
        return RealSenseBackend._camera_intrinsics(
            stream_profile.as_video_stream_profile().get_intrinsics()
        )

    @staticmethod
    def list_devices() -> list[str]:
        """Return serial numbers of all connected RealSense devices."""
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
