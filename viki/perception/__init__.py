"""
viki.perception
-------------
Skeleton detection pipeline: SyncedFrameGroup → SkeletonFrame (23 landmarks, 3-D, metres).

Public API
----------
    from viki.perception import SkeletonPipeline
    from viki.perception.models import SkeletonFrame, LM, LandmarkSource
"""

from viki.perception.models import (
    PreparedFrame,
    HandDetection,
    Landmarks3D,
    SkeletonFrame,
    LM,
)

# from viki.perception.pipeline import SkeletonPipeline

__all__ = [
    # "SkeletonPipeline",
    "PreparedFrame",
    "HandDetection",
    "Landmarks3D",
    "SkeletonFrame",
    "LM",
]
