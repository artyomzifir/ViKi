"""
viki.prepare.interpolate
------------------------
Gap filling for landmark trajectories over time (shape ``(T, L, 3)``).

``fill_linear``     component-wise linear interpolation (per-camera pre-fuse pass)
``fill_se3_spline`` natural cubic spline per coordinate, linear fallback when a
                    landmark has < 4 valid samples (fused-trajectory pass)

This is deliberately named after the intended preparation stage, but it is not
an SE(3) spline: every landmark coordinate is interpolated independently.  The
distinction matters because long gaps can violate bone geometry, so callers can
bound the largest gap that may be fabricated with ``max_gap``.
"""

from __future__ import annotations

import numpy as np

from viki.dsp import interpolate_nans as fill_linear  # noqa: F401

__all__ = ["fill_linear", "fill_se3_spline"]


def fill_se3_spline(points: np.ndarray, max_gap: int = 0) -> np.ndarray:
    """Fill missing coordinates with a natural cubic spline.

    ``max_gap`` has the same contract as :func:`viki.dsp.interpolate_nans`:
    positive values leave runs longer than that many frames missing, while
    zero preserves the legacy behaviour and fills every gap.
    """
    arr = np.asarray(points, dtype=np.float64).copy()
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("points must have shape (T, L, 3)")
    T = arr.shape[0]
    frames = np.arange(T, dtype=np.float64)
    from scipy.interpolate import CubicSpline

    for li in range(arr.shape[1]):
        for c in range(3):
            series = arr[:, li, c]
            valid = np.isfinite(series)
            n = int(valid.sum())
            if n == T or n == 0:
                continue
            gap = ~valid
            xv, yv = frames[valid], series[valid]
            if n >= 4:
                series[gap] = CubicSpline(xv, yv, bc_type="natural")(frames[gap])
            elif n >= 2:
                series[gap] = np.interp(frames[gap], xv, yv)
            else:
                series[gap] = yv[0]
            if max_gap > 0:
                start = 0
                while start < T:
                    if not gap[start]:
                        start += 1
                        continue
                    stop = start + 1
                    while stop < T and gap[stop]:
                        stop += 1
                    if stop - start > max_gap:
                        series[start:stop] = np.nan
                    start = stop
            arr[:, li, c] = series
    return arr
