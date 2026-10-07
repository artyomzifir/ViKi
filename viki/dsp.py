"""
viki.dsp
--------
Shared 1-D signal helpers: Savitzky-Golay smoothing and NaN interpolation over
time. Leaf utility (numpy/scipy only) used by both :mod:`viki.prepare` and
:mod:`viki.retarget` so neither imports the other.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np

SmoothingMethod = Literal["none", "savgol"]


# ─────────────────────────── window sizing ─────────────────────────────


def _odd_at_most(value: int) -> int:
    """Largest odd integer <= value."""
    if value % 2 == 0:
        value -= 1
    return value


def adjusted_savgol_window(length: int, window: int, polyorder: int) -> int:
    """An odd Savitzky-Golay window valid for a signal of ``length`` samples."""
    length, window, polyorder = int(length), int(window), int(polyorder)
    if length <= 0:
        raise ValueError("cannot smooth an empty trajectory")
    if window <= 0:
        raise ValueError("Savitzky-Golay window must be positive")
    if polyorder < 0:
        raise ValueError("Savitzky-Golay polyorder must be non-negative")

    min_window = polyorder + 1
    if min_window % 2 == 0:
        min_window += 1
    max_window = length if length % 2 == 1 else length - 1
    if max_window < min_window:
        raise ValueError(
            "trajectory too short for Savitzky-Golay smoothing: "
            f"length={length}, polyorder={polyorder}, minimum_window={min_window}"
        )
    adjusted = _odd_at_most(min(window, max_window))
    return max(adjusted, min_window)


# ─────────────────────────── generic trajectory ───────────────────────


def smooth_none(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).copy()


def smooth_savgol(
    points: np.ndarray, window: int = 15, polyorder: int = 3, axis: int = 0
) -> np.ndarray:
    """Savitzky-Golay along ``axis``, window auto-clamped to signal length."""
    arr = np.asarray(points, dtype=np.float64)
    window_length = adjusted_savgol_window(arr.shape[axis], window, polyorder)
    from scipy.signal import savgol_filter

    return savgol_filter(
        arr, window_length=window_length, polyorder=polyorder, axis=axis, mode="interp"
    )


def smooth_trajectory(
    points: np.ndarray,
    method: SmoothingMethod = "none",
    window: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    if method == "none":
        return smooth_none(points)
    if method == "savgol":
        return smooth_savgol(points, window=window, polyorder=polyorder, axis=0)
    raise ValueError(f"unknown smoothing method: {method}")


# ───────────────────────────── rotations ─────────────────────────────


def geodesic_mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal (Frobenius) mean of a stack of rotation matrices.

    The projection of the arithmetic mean back onto SO(3). For rotations that
    stay within a fraction of a turn of one another — which is the case this is
    for — it coincides with the geodesic mean closely enough to serve as the
    linearisation point below.
    """
    arr = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    if not len(arr):
        raise ValueError("cannot average an empty rotation stack")
    u, _, vt = np.linalg.svd(arr.mean(axis=0))
    mean = u @ vt
    if np.linalg.det(mean) < 0:            # reflection — flip the weakest axis
        u[:, -1] *= -1
        mean = u @ vt
    return mean


def smooth_rotations(
    rotations: np.ndarray,
    window: int = 15,
    polyorder: int = 2,
    iterations: int = 2,
) -> np.ndarray:
    """Savitzky-Golay a rotation track without leaving SO(3).

    Rotation matrices cannot be filtered element-wise — the result is not a
    rotation. Instead each sample is expressed as a rotation *vector* relative
    to the track's mean orientation, those 3-vectors are smoothed like any other
    trajectory, and the result is mapped back. Re-centring on the smoothed mean
    (``iterations``) keeps the linearisation honest when the track does turn.

    Exact in the limit this pipeline mostly meets: an object that is carried
    rather than turned has a near-constant orientation, so every sample sits in
    one small neighbourhood of the mean and the tangent space is not an
    approximation worth worrying about.

    ``window <= 1`` returns a copy, so callers can disable smoothing by config
    without branching. Frames that are not finite are carried through untouched
    and excluded from the fit.
    """
    from scipy.spatial.transform import Rotation

    arr = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    out = arr.copy()
    if int(window) <= 1 or len(arr) < 3:
        return out
    valid = np.isfinite(arr).all(axis=(1, 2))
    if valid.sum() < 3:
        return out

    reference = geodesic_mean_rotation(arr[valid])
    for _ in range(max(1, int(iterations))):
        local = Rotation.from_matrix(reference.T @ arr[valid])
        vectors = local.as_rotvec()
        try:
            smoothed = smooth_savgol(vectors, window=window, polyorder=polyorder)
        except ValueError:
            return out                      # track too short for this window
        result = reference @ Rotation.from_rotvec(smoothed).as_matrix()
        reference = geodesic_mean_rotation(result)
    out[valid] = result
    return out


# ─────────────────────── gap-aware landmark sequence ──────────────────


def _true_runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """(start, stop) for each contiguous True segment of ``mask``."""
    start: int | None = None
    for idx, ok in enumerate(mask):
        if ok and start is None:
            start = idx
        elif not ok and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask)


def smooth_landmark_sequence(
    landmarks: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
    mode: str = "interp",
) -> np.ndarray:
    """
    Savitzky-Golay along time, per landmark, on contiguous valid segments only.
    NaN gaps are never bridged. Input/output shape ``(T, L, 3)``.
    """
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("landmarks must have shape (T, L, 3)")
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")

    requested = _odd_at_most(int(window_length))
    if requested <= polyorder:
        raise ValueError("window_length must be greater than polyorder")

    from scipy.signal import savgol_filter

    out = arr.copy()
    valid = np.isfinite(arr).all(axis=2)
    for li in range(arr.shape[1]):
        for start, stop in _true_runs(valid[:, li]):
            win = _odd_at_most(min(stop - start, requested))
            if win <= polyorder:
                continue
            out[start:stop, li, :] = savgol_filter(
                arr[start:stop, li, :],
                window_length=win,
                polyorder=polyorder,
                axis=0,
                mode=mode,
            )
    return out


def interpolate_nans(
    points: np.ndarray,
    max_gap: int = 0,
    *,
    extrapolate_edges: bool = True,
) -> np.ndarray:
    """Linearly fill NaNs over time for each landmark coordinate. ``(T, L, 3)``.

    ``max_gap`` > 0 leaves interior gaps longer than ``max_gap`` frames as NaN
    (so a long occlusion is not papered over with a straight line); ``0`` fills
    every gap. When ``extrapolate_edges`` is false, leading and trailing gaps
    remain NaN because they are not bracketed by observations.
    """
    out = np.asarray(points, dtype=np.float64).copy()
    frames = np.arange(out.shape[0], dtype=np.float64)
    for lm in range(out.shape[1]):
        for dim in range(out.shape[2]):
            series = out[:, lm, dim]
            valid = np.isfinite(series)
            if valid.all() or not valid.any():
                continue
            gap = ~valid
            first = int(np.flatnonzero(valid)[0])
            last = int(np.flatnonzero(valid)[-1])
            if valid.sum() == 1:
                series[gap] = series[valid][0]
            else:
                series[gap] = np.interp(frames[gap], frames[valid], series[valid])
            if not extrapolate_edges:
                series[:first] = np.nan
                series[last + 1:] = np.nan
            if max_gap and max_gap > 0:
                i = 0
                n = len(series)
                while i < n:
                    if not gap[i]:
                        i += 1
                        continue
                    j = i
                    while j < n and gap[j]:
                        j += 1
                    if j - i > max_gap:
                        series[i:j] = np.nan
                    i = j
            out[:, lm, dim] = series
    return out


__all__ = [
    "SmoothingMethod",
    "adjusted_savgol_window",
    "smooth_none",
    "smooth_savgol",
    "smooth_trajectory",
    "smooth_landmark_sequence",
    "interpolate_nans",
]
