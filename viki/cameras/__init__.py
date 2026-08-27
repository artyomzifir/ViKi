"""
viki.cameras
------------
Pipeline stage 1: camera drivers, multi-camera sync, and scene recording.

The only package that talks to hardware SDKs (``pyrealsense2``, ``libk4a``).
Concrete backends (:class:`RealSenseBackend`, :class:`KinectBackend`) are
imported lazily via :class:`CameraManager` so a missing SDK does not break
imports elsewhere.
"""

from viki.cameras.base import CameraBackend  # noqa: F401
from viki.cameras.manager import CameraManager  # noqa: F401
from viki.cameras.sync import MultiCameraSync  # noqa: F401
from viki.contracts import CameraIntrinsics, Frame, SyncedFrameGroup  # noqa: F401

__all__ = [
    "CameraBackend",
    "CameraManager",
    "MultiCameraSync",
    "CameraIntrinsics",
    "Frame",
    "SyncedFrameGroup",
]
