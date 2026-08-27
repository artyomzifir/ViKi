"""
viki.prepare.interpolate
------------------------
Gap filling for landmark trajectories over time (shape ``(T, L, 3)``).

``fill_linear``     component-wise linear interpolation (per-camera pre-fuse pass)
``fill_se3_spline`` natural cubic spline per coordinate, linear fallback when a
                    landmark has < 4 valid samples (fused-trajectory pass, paper §3.7)

The paper prescribes a cubic spline over SE(3) for the final end-effector
trajectory. Here the EE ``rotations`` are re-derived from the splined *landmarks*
after smoothing, so splining the landmark positions is equivalent and avoids a
separate rotation spline.
"""

from __future__ import annotations

import numpy as np

from viki.dsp import interpolate_nans as fill_linear  # noqa: F401

__all__ = ["fill_linear", "fill_se3_spline"]


def fill_se3_spline(points: np.ndarray) -> np.ndarray:
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
            xv, yv = frames[valid], series[valid]
            if n >= 4:
                series[~valid] = CubicSpline(xv, yv, bc_type="natural")(frames[~valid])
            elif n >= 2:
                series[~valid] = np.interp(frames[~valid], xv, yv)
            else:
                series[~valid] = yv[0]
            arr[:, li, c] = series
    return arr
