"""
viki.skeleton.detectors
-----------------------
Modular skeleton detection: a list of PartialLandmarkDetector instances
assembled by CompositeLandmarkDetector into one HandDetection per frame.
"""

from viki.skeleton.detectors.arm_pose import MediaPipeArm
from viki.skeleton.detectors.base import (
    FusionMode,
    PartialDetection2D,
    PartialLandmarkDetector,
)
from viki.skeleton.detectors.composite import CompositeLandmarkDetector
from viki.skeleton.detectors.hand_pose import MediaPipeHand
from viki.skeleton.detectors.mediapipe_base import (
    MODELS_DIR_DEFAULT,
    MediaPipeTaskRunner,
    ensure_model,
)

__all__ = [
    "CompositeLandmarkDetector",
    "FusionMode",
    "MODELS_DIR_DEFAULT",
    "MediaPipeArm",
    "MediaPipeHand",
    "MediaPipeTaskRunner",
    "PartialDetection2D",
    "PartialLandmarkDetector",
    "ensure_model",
]
