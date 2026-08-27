"""
viki.perception.backends.rtmpose
--------------------------------
STUB. RTMPose whole-body / hand keypoints (MMPose).

To implement:
  * add ``rtmlib`` (or mmpose + mmdeploy) and an ONNX/TensorRT RTMPose-Hand model
  * run it on ``frame.rgb``; RTMPose returns 21 hand keypoints already in the
    MediaPipe topology, so the index map to :class:`~viki.contracts.LM` is 1:1
  * there is no per-landmark z — fill ``lm_z_rel`` with zeros; depth lifting in
    ``perception.lift`` does not depend on it when measured depth is present
  * pick the hand (``left``/``right``) from the wholebody skeleton side, or run
    the hand-only model on a wrist-centred crop
"""

from __future__ import annotations

from viki.contracts import Hand, HandDetection, PreparedFrame
from viki.perception.backends.base import HandPoseBackend


class RTMPoseHandBackend(HandPoseBackend):
    name = "rtmpose"

    def __init__(self, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(
            "RTMPose backend is not implemented — see module docstring "
            "(viki/perception/backends/rtmpose.py)"
        )

    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        raise NotImplementedError
