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

Dependency: pyrealsense2 >= 2.58
"""

from __future__ import annotations

import logging
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

        try:
            profile = self._pipeline.start(config)
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
            profile = self._pipeline.start(config)

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

        if self._align_to_color:
            self._align = rs.align(rs.stream.color)

        if self._depth_max_m > 0:
            self._threshold = rs.threshold_filter(0.1, self._depth_max_m)

        self._running = True

    def stop(self) -> None:
        if self._pipeline and self._running:
            self._pipeline.stop()
        self._running = False
        self._pipeline = None
        self._align = None
        self._threshold = None

    def get_frame(self) -> Frame:
        if not self._running or self._pipeline is None:
            raise RuntimeError("RealSenseBackend is not started. Call start() first.")

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)
        except RuntimeError as exc:
            # librealsense signals "no frame within timeout_ms" with a plain
            # RuntimeError; hand _CameraWorker the TimeoutError it treats as a
            # clean frame drop (same contract as KinectBackend).
            raise TimeoutError(str(exc)) from exc

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise TimeoutError("RealSense capture missing a colour or depth frame — dropped.")

        if self._threshold is not None:
            # range gate: everything outside [0.1 m, depth_max_m] → 0 (the
            # "no reading" marker the rest of the pipeline expects). Kills the
            # low-confidence 10–30 m tail the raw stream keeps.
            depth_frame = self._threshold.process(depth_frame)

        # copy out of librealsense's frame pool immediately — get_frame's frame
        # handles go out of scope on return and the buffers get recycled, so a
        # bare asanyarray view would tear once the recorder reads it later.
        color = np.array(color_frame.get_data(), copy=True)  # HxWx3 BGR uint8
        depth = np.array(depth_frame.get_data(), copy=True)  # HxW uint16 z16 units
        if self._depth_units_m != 0.001:
            # normalise to millimetres so the pipeline's fixed /1000 stays right
            depth = np.rint(
                depth.astype(np.float32) * (self._depth_units_m * 1000.0)
            ).astype(np.uint16)

        timestamp_us = int(color_frame.get_timestamp() * 1000)  # ms -> us

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
