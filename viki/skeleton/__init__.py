"""
viki.skeleton
-------------
Skeleton detection pipeline: SyncedFrameGroup → SkeletonFrame (23 landmarks, 3-D, metres).

Public API
----------
    from viki.skeleton import SkeletonPipeline
    from viki.skeleton.models import SkeletonFrame, LM, LandmarkSource
"""

from viki.skeleton.models import (
    PreparedFrame,
    HandDetection,
    Landmarks3D,
    SkeletonFrame,
    LM,
)

# from viki.skeleton.pipeline import SkeletonPipeline

__all__ = [
    # "SkeletonPipeline",
    "PreparedFrame",
    "HandDetection",
    "Landmarks3D",
    "SkeletonFrame",
    "LM",
]
