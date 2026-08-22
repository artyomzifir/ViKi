"""
viki.capture.realsense
----------------------
Backend for Intel RealSense D435i (and compatible D4xx models).

Dependency: pyrealsense2 >= 2.58
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import CameraBackend, CameraIntrinsics, Frame

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
        (width, height) for the depth stream. Default: 640x480.
    fps : int
        Frame rate. Default: 30.
    align_to_color : bool
        If True, depth is aligned to the colour camera frame (same resolution,
        per-pixel correspondence). Default: True.
    timeout_ms : int
        Frame wait timeout in milliseconds.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        color_resolution: tuple[int, int] = (640, 480),
        depth_resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        align_to_color: bool = True,
        timeout_ms: int = 5000,
    ) -> None:
        self._serial = serial
        self._color_res = color_resolution
        self._depth_res = depth_resolution
        self._fps = fps
        self._align_to_color = align_to_color
        self._timeout_ms = timeout_ms

        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._resolved_serial: str = serial or ""
        self._running = False

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
        config.enable_stream(
            rs.stream.depth,
            self._depth_res[0],
            self._depth_res[1],
            rs.format.z16,
            self._fps,
        )

        profile = self._pipeline.start(config)

        dev = profile.get_device()
        self._resolved_serial = dev.get_info(rs.camera_info.serial_number)

        if self._align_to_color:
            self._align = rs.align(rs.stream.color)

        self._running = True

    def stop(self) -> None:
        if self._pipeline and self._running:
            self._pipeline.stop()
        self._running = False
        self._pipeline = None
        self._align = None

    def get_frame(self) -> Frame:
        if not self._running or self._pipeline is None:
            raise RuntimeError("RealSenseBackend is not started. Call start() first.")

        frames = self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to retrieve frames from RealSense.")

        color = np.asanyarray(color_frame.get_data())  # HxWx3 BGR uint8
        depth = np.asanyarray(depth_frame.get_data())  # HxW uint16 mm

        timestamp_us = int(color_frame.get_timestamp() * 1000)  # ms -> us

        color_intr = self._get_intrinsics(color_frame.profile)
        depth_intr = self._get_intrinsics(depth_frame.profile)

        return Frame(
            color=color,
            depth=depth,
            timestamp_us=timestamp_us,
            device_id=self._resolved_serial,
            color_intrinsics=color_intr,
            depth_intrinsics=depth_intr,
        )

    @property
    def device_id(self) -> str:
        return self._resolved_serial

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_intrinsics(stream_profile) -> CameraIntrinsics:
        intr = stream_profile.as_video_stream_profile().get_intrinsics()
        return CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            width=intr.width,
            height=intr.height,
            dist_coeffs=np.array(intr.coeffs),
        )

    @staticmethod
    def list_devices() -> list[str]:
        """Return serial numbers of all connected RealSense devices."""
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
