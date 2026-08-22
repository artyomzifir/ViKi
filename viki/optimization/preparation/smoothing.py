"""
viki.optimization.preparation.smoothing
--------------------------------------
Temporal smoothing helpers for completed skeleton sequences.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def smooth_landmark_sequence(
    landmarks: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
    mode: str = "interp",
) -> np.ndarray:
    """
    Apply a Savitzky‑Golay filter along time for each landmark independently.

    Only contiguous valid (non‑NaN) segments are smoothed; gaps are left as NaN.

    Parameters
    ----------
    landmarks : np.ndarray
        Dense array shaped (T, L, 3). Missing landmarks must be NaN.
    window_length : int, default=7
        Preferred odd Savitzky‑Golay window length. Even values are rounded
        down to the nearest odd number.
    polyorder : int, default=2
        Polynomial order for scipy.signal.savgol_filter.
    mode : str, default="interp"
        Edge handling mode passed to savgol_filter.

    Returns
    -------
    np.ndarray
        Smoothed float64 copy with the same shape. NaN gaps remain NaN and are
        never bridged.

    Raises
    ------
    ValueError
        If shape is not (T, L, 3), or window_length <= polyorder.
    """
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("landmarks must have shape (T, L, 3)")
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")

    requested_window = _odd_at_most(int(window_length))
    if requested_window <= polyorder:
        raise ValueError("window_length must be greater than polyorder")

    out = arr.copy()
    valid = np.isfinite(arr).all(axis=2)

    for landmark_idx in range(arr.shape[1]):
        for start, stop in _true_runs(valid[:, landmark_idx]):
            effective_window = _effective_window(
                run_length=stop - start,
                requested_window=requested_window,
                polyorder=polyorder,
            )
            if effective_window is None:
                continue

            out[start:stop, landmark_idx, :] = _savgol_filter(
                arr[start:stop, landmark_idx, :],
                window_length=effective_window,
                polyorder=polyorder,
                axis=0,
                mode=mode,
            )

    return out


def _savgol_filter(values: np.ndarray, **kwargs) -> np.ndarray:
    """Wrapper around scipy.signal.savgol_filter."""
    from scipy.signal import savgol_filter

    return savgol_filter(values, **kwargs)


def _effective_window(
    run_length: int,
    requested_window: int,
    polyorder: int,
) -> int | None:
    """Return the effective window size for a run, or None if too short."""
    window = _odd_at_most(min(run_length, requested_window))
    if window <= polyorder:
        return None
    return window


def _odd_at_most(value: int) -> int:
    """Return the largest odd integer <= value."""
    if value % 2 == 0:
        value -= 1
    return value


def _true_runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield (start, stop) indices for each contiguous True segment."""
    start: int | None = None
    for idx, is_valid in enumerate(mask):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask)


def interpolate_nans(points: np.ndarray) -> np.ndarray:
    """
    Linearly fill NaNs over time for each landmark coordinate.

    Parameters
    ----------
    points : np.ndarray
        Shape (T, L, 3) with possible NaN values.

    Returns
    -------
    np.ndarray
        Copy with NaNs linearly interpolated.
    """
    out = np.asarray(points, dtype=np.float64).copy()
    frames = np.arange(out.shape[0], dtype=np.float64)
    for landmark in range(out.shape[1]):
        for dim in range(out.shape[2]):
            series = out[:, landmark, dim]
            valid = np.isfinite(series)
            if valid.all():
                continue
            if not valid.any():
                continue
            if valid.sum() == 1:
                series[~valid] = series[valid][0]
            else:
                series[~valid] = np.interp(frames[~valid], frames[valid], series[valid])
            out[:, landmark, dim] = series
    return out


__all__ = ["smooth_landmark_sequence", "interpolate_nans"]
