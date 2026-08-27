"""
viki.perception.backends.hamer
------------------------------
STUB. HaMeR (transformer hand mesh recovery, MANO).

To implement:
  * add the ``hamer`` package + its checkpoint and a hand detector (ViTDet) for
    the bounding box
  * HaMeR outputs a MANO mesh + 3D joints in a weak-perspective camera frame;
    project the 21 joints to pixels with the predicted camera to fill
    ``HandDetection.points`` (MANO joint order maps to
    :class:`~viki.contracts.LM` with a fixed permutation — define it here)
  * ``lm_z_rel`` can carry the root-relative joint z from the mesh (already a
    real depth cue, unlike MediaPipe's heuristic z)
  * MANO is handedness-specific; instantiate for the configured ``hand``
"""

from __future__ import annotations

from viki.contracts import Hand, HandDetection, PreparedFrame
from viki.perception.backends.base import HandPoseBackend


class HaMeRHandBackend(HandPoseBackend):
    name = "hamer"

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "HaMeR backend is not implemented — see module docstring "
            "(viki/perception/backends/hamer.py)"
        )

    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        raise NotImplementedError
