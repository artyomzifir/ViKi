"""
viki.prepare.interpolate
------------------------
Gap filling for fused landmark / pose trajectories.

``fill_linear`` is the working path (component-wise linear interpolation over
time). ``fill_se3_spline`` is a STUB for the cubic-spline-over-SE(3) completion
the paper prescribes (§3.7); until it is implemented it falls back to linear and
logs.
"""

from __future__ import annotations

import logging

import numpy as np

from viki.dsp import interpolate_nans as fill_linear  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["fill_linear", "fill_se3_spline"]


def fill_se3_spline(points: np.ndarray) -> np.ndarray:
    """
    STUB: cubic-spline interpolation over SE(3) for landmark gaps (paper §3.7).

    Splining rotations properly needs a quaternion / rotation-vector spline; the
    plain linear fill below is the interim behaviour.
    """
    logger.warning(
        "fill_se3_spline is a stub (paper §3.7); falling back to linear interpolation"
    )
    return fill_linear(points)
