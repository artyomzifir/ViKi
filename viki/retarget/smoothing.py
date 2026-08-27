"""Trajectory smoothing helpers for ViKi experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np


SmoothingMethod = Literal["none", "savgol"]


def adjusted_savgol_window(length: int, window: int, polyorder: int) -> int:
    """Return an odd Savitzky-Golay window that is valid for the input length."""
    length = int(length)
    window = int(window)
    polyorder = int(polyorder)

    if length <= 0:
        raise ValueError("Cannot smooth an empty trajectory.")
    if window <= 0:
        raise ValueError("Savitzky-Golay window must be positive.")
    if polyorder < 0:
        raise ValueError("Savitzky-Golay polyorder must be non-negative.")

    # scipy.signal.savgol_filter requires an odd window_length > polyorder.
    min_window = polyorder + 1
    if min_window % 2 == 0:
        min_window += 1

    max_window = length if length % 2 == 1 else length - 1
    if max_window < min_window:
        raise ValueError(
            "Trajectory is too short for Savitzky-Golay smoothing: "
            f"length={length}, polyorder={polyorder}, minimum_window={min_window}."
        )

    adjusted = min(window, max_window)
    if adjusted % 2 == 0:
        adjusted -= 1
    if adjusted < min_window:
        adjusted = min_window
    return adjusted


def smooth_none(points: np.ndarray) -> np.ndarray:
    """Return a float copy without temporal smoothing."""
    return np.asarray(points, dtype=np.float64).copy()


def smooth_savgol(
    points: np.ndarray,
    window: int = 15,
    polyorder: int = 3,
    axis: int = 0,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along the time axis."""
    arr = np.asarray(points, dtype=np.float64)
    length = arr.shape[axis]
    window_length = adjusted_savgol_window(length, window, polyorder)

    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for Savitzky-Golay smoothing. "
            "Install scipy or run with --smoothing none."
        ) from exc

    return savgol_filter(
        arr,
        window_length=window_length,
        polyorder=polyorder,
        axis=axis,
        mode="interp",
    )


def smooth_trajectory(
    points: np.ndarray,
    method: SmoothingMethod = "none",
    window: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    """Smooth a trajectory with the requested method."""
    if method == "none":
        return smooth_none(points)
    if method == "savgol":
        return smooth_savgol(points, window=window, polyorder=polyorder, axis=0)
    raise ValueError(f"Unknown smoothing method: {method}")
