"""
viki.viz.depth
--------------
Image preparation for the camera streams:

- ``DepthColorizer`` — turns a uint16 depth frame into a colour-mapped BGR
  image, with an EMA-smoothed display range and last-good-frame hold.
- ``Undistorter`` — applies cached intrinsic undistortion to a colour image.
- ``DepthStabilizer`` — applies temporal median and optional bilateral filtering
  to reduce depth noise.

Both hold per-stream state, so create one instance per stream.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np



class DepthColorizer:
    """
    Stateful uint16-depth → BGR turbo colour map for a single stream.

    Maintains an exponential moving average (EMA) of the depth range (min/max)
    and holds the last valid frame when a frame is mostly empty, preventing
    the stream from flickering black.

    Attributes
    ----------
    alpha : float
        Smoothing factor for the EMA range update (0..1).
    min_valid_fraction : float
        Minimum fraction of valid pixels (depth > 0) required to update the
        range and return a new frame; otherwise, the last good frame is reused.
    d_min, d_max : float
        Current EMA min and max depth values (in mm).
    _ema_initialised : bool
        Whether the EMA has been seeded.
    _last_good : Optional[np.ndarray]
        The last successfully colourised frame (BGR) to hold on empty frames.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_valid_fraction: float = 0.05,
    ) -> None:
        """
        Parameters
        ----------
        alpha : float, default=0.05
            Smoothing factor for EMA range. Higher = more responsive to changes.
        min_valid_fraction : float, default=0.05
            Minimum fraction of valid pixels (depth > 0) to update the range
            and generate a new frame. If below this, the last good frame is held.
        """
        self.alpha = alpha
        self.min_valid_fraction = min_valid_fraction
        self.d_min: float = 0.0
        self.d_max: float = 1.0
        self._ema_initialised = False
        self._last_good: Optional[np.ndarray] = None

    def colorize(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert a uint16 depth image to a colour-mapped BGR image.

        If the fraction of valid pixels is below `min_valid_fraction`, returns
        the last successfully colourised frame (or None if none exists) to avoid
        black flickering.

        The colourmap range is updated using an EMA of the 2nd and 98th percentiles
        to ignore outliers.

        Parameters
        ----------
        depth : np.ndarray
            Depth image (HxW, uint16, values in millimetres).

        Returns
        -------
        Optional[np.ndarray]
            BGR colour-mapped image (HxWx3, uint8) or None if no frame is available
            and the current frame is empty.
        """
        valid = depth[depth > 0]
        valid_fraction = valid.size / max(depth.size, 1)

        if valid_fraction < self.min_valid_fraction:
            return self._last_good  
        
        # Update EMA range using 2nd/98th percentile to ignore outliers.
        p2 = float(np.percentile(valid, 2))
        p98 = float(np.percentile(valid, 98))
        if not self._ema_initialised:
            self.d_min, self.d_max = p2, p98
            self._ema_initialised = True
        else:
            self.d_min = self.alpha * p2 + (1 - self.alpha) * self.d_min
            self.d_max = self.alpha * p98 + (1 - self.alpha) * self.d_max

        norm = np.clip(
            (depth.astype(np.float32) - self.d_min) / (self.d_max - self.d_min + 1e-6), 0, 1
        )
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        self._last_good = img
        return img


# ... existing code ...
class Undistorter:
    """
    Apply intrinsic undistortion to colour images, caching the remap tables.

    The remap tables are computed once for the given image size and reused for
    subsequent frames, improving performance.

    Attributes
    ----------
    mtx : np.ndarray
        3x3 camera matrix (intrinsics).
    dist : np.ndarray
        Distortion coefficients (vector of length 4 or 5).
    _map1, _map2 : Optional[np.ndarray]
        Cached remap tables for the current image size.
    """

    def __init__(self, mtx: np.ndarray, dist: np.ndarray) -> None:
        """
        Parameters
        ----------
        mtx : np.ndarray
            3x3 camera matrix.
        dist : np.ndarray
            Distortion coefficients.
        """
        self.mtx = mtx
        self.dist = dist
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Undistort an input image using the cached remap tables.

        Parameters
        ----------
        img : np.ndarray
            Input BGR image (HxWx3, uint8).

        Returns
        -------
        np.ndarray
            Undistorted image (same shape and type).
        """
        h, w = img.shape[:2]
        if self._map1 is None:
            # Precompute the mapping once for performance.
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.mtx, (w, h), cv2.CV_32FC1
            )
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)


class DepthStabilizer:
    """
    Reduce temporal noise in depth maps using a sliding window median filter
    and optional bilateral spatial filtering.

    Attributes
    ----------
    window_size : int
        Number of frames to keep in the buffer (temporal median window).
    use_bilateral : bool
        If True, apply a bilateral filter after the temporal median to smooth
        spatial noise while preserving edges.
    buffer : list[np.ndarray]
        Ring buffer of recent depth frames.
    """

    def __init__(
        self, 
        window_size: int = 5, 
        use_bilateral: bool = False
    ) -> None:
        """
        Parameters
        ----------
        window_size : int, default=5
            Number of recent frames to use for the temporal median.
        use_bilateral : bool, default=False
            Whether to apply a bilateral filter after temporal smoothing.
        """
        self.window_size = window_size
        self.use_bilateral = use_bilateral
        self.buffer: list[np.ndarray] = []

    def stabilize(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply temporal median filtering (and optional bilateral) to a depth frame.

        Parameters
        ----------
        depth : np.ndarray
            Depth image (HxW, uint16, values in mm).

        Returns
        -------
        np.ndarray
            Stabilised depth image (same shape and type).
        """
        if self.buffer and depth.shape != self.buffer[0].shape:
            self.buffer.clear()

        self.buffer.append(depth)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) < 2:
            return depth

        # Temporal median filter
        stack = np.stack(self.buffer, axis=0)
        median_depth = np.median(stack, axis=0).astype(np.uint16)

        if self.use_bilateral:
            # Bilateral filter expects float32 or uint8
            float_depth = median_depth.astype(np.float32)
            smoothed = cv2.bilateralFilter(float_depth, d=5, sigmaColor=50, sigmaSpace=5)
            median_depth = smoothed.astype(np.uint16)

        return median_depth

