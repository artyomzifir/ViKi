"""
viki.perception.models
--------------------
Compatibility shim. The DTOs now live in :mod:`viki.contracts`; this module
re-exports them so existing imports keep working during the refactor.
New code should import from ``viki.contracts`` directly.
"""

from __future__ import annotations

from viki.contracts import (  # noqa: F401
    HAND_LM_COUNT,
    DepthDebug,
    EndEffectorPose,
    HandDetection,
    Landmarks3D,
    LM,
    PipelineResult,
    PreparedFrame,
    SkeletonFrame,
)

__all__ = [
    "LM",
    "HAND_LM_COUNT",
    "PreparedFrame",
    "HandDetection",
    "Landmarks3D",
    "EndEffectorPose",
    "SkeletonFrame",
    "DepthDebug",
    "PipelineResult",
]
