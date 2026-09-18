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
from viki.perception.auto_prompts import (  # noqa: F401
    AutoPromptConfig,
    generate_auto_prompts,
)
from viki.perception.object_model import (  # noqa: F401
    ObjectModelConfig,
    build_object_models,
)
from viki.perception.scene import (  # noqa: F401
    ScenePerceptionOpts,
    build_object_models_episode,
    lift_segmented_episode,
    scene_perception_episode,
    segment_episode,
)

__all__ = [
    "LM",
    "PreparedFrame",
    "HandDetection",
    "Landmarks3D",
    "SkeletonFrame",
    "EndEffectorPose",
    "HandPoseBackend",
    "AutoPromptConfig",
    "generate_auto_prompts",
    "ObjectModelConfig",
    "build_object_models",
    "ScenePerceptionOpts",
    "segment_episode",
    "lift_segmented_episode",
    "build_object_models_episode",
    "scene_perception_episode",
    "load_backend",
]
