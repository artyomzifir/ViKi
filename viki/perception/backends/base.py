"""
viki.perception.backends.base
-----------------------------
The pose-backend abstraction.

A backend takes one :class:`~viki.contracts.PreparedFrame` and returns pixel-space
hand landmarks (:class:`~viki.contracts.HandDetection`) for the requested hand,
or ``None`` if that hand was not found. ViKi tracks a single hand; the caller
picks which one.

Implementations: ``mediapipe`` (working) and stubs for ``rtmpose`` / ``hamer`` /
``yolo``. Select via ``cfg.POSE_BACKEND``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from viki.contracts import Hand, HandDetection, PreparedFrame

__all__ = ["HandPoseBackend", "Hand"]


class HandPoseBackend(ABC):
    """One hand-landmark model. Stateful implementations create one per camera."""

    name: str

    @abstractmethod
    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        """
        Run the model on one prepared frame.

        Returns a :class:`HandDetection` with ``points`` keyed by
        :class:`~viki.contracts.LM` (0..20) in pixel coordinates, plus
        ``lm_z_rel`` (model-relative z, not metric) and a scalar ``confidence``.
        Returns ``None`` when the requested hand is absent.
        """

    def close(self) -> None:
        """Release model resources. Default: nothing to do."""
