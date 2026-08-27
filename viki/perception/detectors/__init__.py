"""
viki.perception.detectors
-----------------------
Modular skeleton detection: a list of PartialLandmarkDetector instances
assembled by CompositeLandmarkDetector into one HandDetection per frame.
"""

from viki.perception.detectors.arm_pose import MediaPipeArm
from viki.perception.detectors.base import (
    FusionMode,
    PartialDetection2D,
    PartialLandmarkDetector,
)
from viki.perception.detectors.composite import CompositeLandmarkDetector
from viki.perception.detectors.hand_pose import MediaPipeHand
from viki.perception.detectors.mediapipe_base import (
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
