"""
viki.perception.backends.yolo
-----------------------------
STUB. YOLO-Pose hand keypoints (Ultralytics).

To implement:
  * add ``ultralytics`` and a hand-keypoint YOLO-Pose weight (21 kpts) — or a
    hand-detection weight feeding a separate keypoint head
  * ``model(frame.rgb)`` → per-instance keypoints (x, y, visibility); map the
    keypoint order to :class:`~viki.contracts.LM` (define the permutation here)
  * use the ``visibility`` channel as ``per_index_confidence`` — it is exactly
    the ``v`` factor of the fusion weight (paper §3.5, eq. 2)
  * YOLO gives no z — fill ``lm_z_rel`` with zeros
  * handedness: infer from thumb-vs-pinky x-order, or run a small classifier
"""

from __future__ import annotations

from viki.contracts import Hand, HandDetection, PreparedFrame
from viki.perception.backends.base import HandPoseBackend


class YoloHandBackend(HandPoseBackend):
    name = "yolo"

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "YOLO backend is not implemented — see module docstring "
            "(viki/perception/backends/yolo.py)"
        )

    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        raise NotImplementedError
