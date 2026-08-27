"""Compatibility shim — PreparationPipeline moved to :mod:`viki.prepare.run`."""

from viki.prepare.run import (  # noqa: F401
    PreparationPipeline,
    estimate_fps,
    stable_palm_orientation_mask,
)

__all__ = ["PreparationPipeline", "estimate_fps", "stable_palm_orientation_mask"]
