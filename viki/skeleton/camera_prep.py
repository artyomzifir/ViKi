"""
viki.skeleton.camera_prep
-------------------------
Converts a raw capture Frame into a PreparedFrame ready for model inference.

This module has no dependency on MediaPipe or CameraManager.
"""

from __future__ import annotations

import cv2
import numpy as np

from viki.capture.base import Frame
from viki.skeleton.models import PreparedFrame


def prepare_frame(frame: Frame) -> PreparedFrame:
    """
    Convert a raw Frame into a PreparedFrame.

    The colour image is converted to RGB (no undistortion — MediaPipe
    handles lens distortion natively).
    The depth image is converted to float32 metres; invalid zeros become NaN.

    Parameters
    ----------
    frame : Frame
        Raw frame from CameraManager (colour BGR, depth uint16 mm).

    Returns
    -------
    PreparedFrame
        Ready for hand detection and geometry lifting.
    """
    # BGR → RGB
    rgb = cv2.cvtColor(frame.color, cv2.COLOR_BGR2RGB)

    # Depth: uint16 mm → float32 metres, zeros → NaN
    depth = frame.depth.astype(np.float32) / 1000.0
    depth[depth == 0] = np.nan

    # Build depth intrinsics matrix (3x3) from frame data
    depth_K: np.ndarray | None = None
    if frame.depth_intrinsics is not None:
        di = frame.depth_intrinsics
        depth_K = np.array(
            [[di.fx, 0, di.cx], [0, di.fy, di.cy], [0, 0, 1]], dtype=np.float32
        )

    return PreparedFrame(
        rgb=rgb,
        depth_m=depth,
        depth_K=depth_K,
        device_id=frame.device_id,
        timestamp_us=frame.timestamp_us,
    )
