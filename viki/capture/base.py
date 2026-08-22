"""
viki.capture.base
-----------------
Abstract camera interface.

All backends (RealSense, Azure Kinect, ...) implement CameraBackend.
Consumers work only with this interface — the concrete backend is
injected from the outside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Distortion coefficients (k1,k2,p1,p2[,k3]) — optional
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))


@dataclass
class Frame:
    """
    A single synchronised frame from one camera.

    Fields
    ------
    color             : np.ndarray  HxWx3, uint8, BGR (OpenCV convention)
    depth             : np.ndarray  HxW,   uint16, millimetres
    timestamp_us      : int         capture time, microseconds (device monotonic clock)
    device_id         : str         unique device identifier (serial number or alias)
    aligned_depth     : np.ndarray | None  HxW, uint16, SDK-aligned depth
    host_timestamp_us : int         host-clock time (time.time_ns()//1000) when the frame
                                     arrived in the worker thread; set by CameraManager,
                                     not by the backend. Used for cross-camera sync.
    color_intrinsics : CameraIntrinsics | None
    depth_intrinsics : CameraIntrinsics | None
    """

    color: np.ndarray
    depth: np.ndarray
    timestamp_us: int
    device_id: str
    aligned_depth: Optional[np.ndarray] = None
    host_timestamp_us: int = 0
    color_intrinsics: Optional[CameraIntrinsics] = None
    depth_intrinsics: Optional[CameraIntrinsics] = None

    def has_depth(self) -> bool:
        return self.depth is not None and self.depth.size > 0


@dataclass
class SyncedFrameGroup:
    """
    A set of frames — one per camera — aligned to a common host-clock tick.

    offsets_us[device_id] = frame.host_timestamp_us - sync_timestamp_us
    Negative means the frame arrived before the tick (early); positive means after (late).
    """

    frames: dict
    sync_timestamp_us: int
    offsets_us: dict

    @property
    def device_ids(self) -> list:
        return list(self.frames.keys())

    def has_depth(self) -> bool:
        return all(f.has_depth() for f in self.frames.values())


class CameraBackend(ABC):
    """
    Abstract camera backend.

    Lifecycle
    ---------
    backend = SomeBackend(...)
    backend.start()
    try:
        while True:
            frame = backend.get_frame()
            ...
    finally:
        backend.stop()

    Or via context manager:
        with SomeBackend(...) as cam:
            frame = cam.get_frame()
    """

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the stream and release the device."""

    @abstractmethod
    def get_frame(self) -> Frame:
        """
        Fetch the next frame (blocking call).

        Raises
        ------
        RuntimeError  if the backend is not started or the device is lost
        TimeoutError  if no frame arrives within the configured timeout
        """

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Unique device identifier."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """True if the stream is active."""

    @abstractmethod
    def project_color_to_depth(
        self, u: float, v: float, z: float
    ) -> tuple[float, float] | None:
        return None

    def deproject_2d_to_3d(
        self, u: float, v: float, z: float
    ) -> tuple[float, float, float] | None:
        return None

    # ------------------------------------------------------------------
    # Context manager — implemented here, no need to override
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraBackend":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
