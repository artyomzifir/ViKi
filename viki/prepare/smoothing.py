"""Compatibility shim — smoothing helpers moved to :mod:`viki.dsp`."""

from viki.dsp import interpolate_nans, smooth_landmark_sequence  # noqa: F401

__all__ = ["smooth_landmark_sequence", "interpolate_nans"]
