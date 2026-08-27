"""
viki.perception
---------------
Video → per-camera 3-D hand skeleton + end-effector pose.

Pipeline stage 2. Consumes recorded RGB-D frames, runs a pluggable hand-pose
backend (:mod:`viki.perception.backends`), lifts detections to 3-D with measured
depth (:mod:`viki.perception.lift`), and derives the wrist pose
(:mod:`viki.perception.end_effector`). Cross-camera fusion happens later, in
:mod:`viki.prepare`.
"""

from viki.contracts import (  # noqa: F401
    EndEffectorPose,
    HandDetection,
    Landmarks3D,
    LM,
    PreparedFrame,
    SkeletonFrame,
)
from viki.perception.backends import HandPoseBackend, load_backend  # noqa: F401

__all__ = [
    "LM",
    "PreparedFrame",
    "HandDetection",
    "Landmarks3D",
    "SkeletonFrame",
    "EndEffectorPose",
    "HandPoseBackend",
    "load_backend",
]
