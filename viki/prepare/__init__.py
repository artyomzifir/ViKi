"""
viki.prepare
------------
Pipeline stage 3: rec.npz -> cln.npz.

Fuse per-camera landmark trajectories onto a common time grid
(:mod:`viki.prepare.fuse`), fill gaps (:mod:`viki.prepare.interpolate`), smooth
(:mod:`viki.dsp`), derive the end-effector pose and gripper state, and — when an
object-pose track is available — the object-relative form
(:mod:`viki.prepare.represent`).
"""

from viki.prepare.run import PreparationPipeline, estimate_fps  # noqa: F401

__all__ = ["PreparationPipeline", "estimate_fps"]
