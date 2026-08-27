"""
viki.cameras.base
-----------------
Abstract camera interface. The data types (``Frame``, ``SyncedFrameGroup``,
``CameraIntrinsics``) now live in :mod:`viki.contracts` and are re-exported
here for compatibility. ``CameraBackend`` — the ABC with behaviour — stays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from viki.contracts import (  # noqa: F401
    CameraIntrinsics,
    Frame,
    SyncedFrameGroup,
)


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
