"""Compatibility shim — smoothing helpers moved to :mod:`viki.dsp`."""

from viki.dsp import (  # noqa: F401
    SmoothingMethod,
    adjusted_savgol_window,
    smooth_none,
    smooth_savgol,
    smooth_trajectory,
)

__all__ = [
    "SmoothingMethod",
    "adjusted_savgol_window",
    "smooth_none",
    "smooth_savgol",
    "smooth_trajectory",
]
