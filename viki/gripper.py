"""
viki.gripper
------------
Gripper-state estimation from a fused hand skeleton.

``Gripper`` is the abstraction; ``BinaryGripper`` is the only implementation —
the simplest useful one: an open/closed decision from the thumb–index fingertip
gap, normalised by palm length, with hysteresis to stop it chattering at the
threshold (paper §3.4, eq. 3).

A continuous or force-modulated gripper would be a new ``Gripper`` subclass;
the seam does not change. ``retarget`` / ``export`` only forward the estimated
state — they do not re-derive it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from viki.contracts import LM, GripperState

_MIN_LEN = 1e-6


class Gripper(ABC):
    """Maps a per-frame hand skeleton to a :class:`GripperState`."""

    name: str

    @abstractmethod
    def estimate(
        self,
        hand_points: Mapping[LM, np.ndarray],
        prev: GripperState | None,
    ) -> GripperState:
        """
        Parameters
        ----------
        hand_points : {LM: (3,) world-frame position}
            Fused hand landmarks for one frame (NaN / missing allowed).
        prev : GripperState | None
            The previous frame's state, for hysteresis. ``None`` on the first
            frame.
        """

    def reset(self) -> None:
        """Drop any internal state. Default: nothing to do."""


class BinaryGripper(Gripper):
    """
    Open/closed from the normalised thumb–index fingertip gap.

    ``d = ||THUMB_TIP - INDEX_TIP|| / ||WRIST - MIDDLE_MCP||``

    Hysteresis: an open gripper closes only once ``d < close_ratio``; a closed
    gripper opens only once ``d > open_ratio`` (``close_ratio < open_ratio``).
    When landmarks are missing the previous state is held (confidence 0).
    """

    name = "binary"

    def __init__(self, close_ratio: float = 0.55, open_ratio: float = 0.90) -> None:
        if not 0.0 < close_ratio < open_ratio:
            raise ValueError("need 0 < close_ratio < open_ratio")
        self._close = float(close_ratio)
        self._open = float(open_ratio)

    def estimate(
        self,
        hand_points: Mapping[LM, np.ndarray],
        prev: GripperState | None,
    ) -> GripperState:
        thumb = hand_points.get(LM.THUMB_TIP)
        index = hand_points.get(LM.INDEX_TIP)
        wrist = hand_points.get(LM.WRIST)
        middle = hand_points.get(LM.MIDDLE_MCP)

        vals = [thumb, index, wrist, middle]
        if any(v is None or not np.all(np.isfinite(v)) for v in vals):
            # Nothing to measure — hold the previous decision.
            held = prev.closed if prev is not None else False
            return GripperState(closed=held, width=0.0 if held else 1.0, confidence=0.0)

        palm = float(np.linalg.norm(np.asarray(wrist) - np.asarray(middle)))
        if palm < _MIN_LEN:
            held = prev.closed if prev is not None else False
            return GripperState(closed=held, width=0.0 if held else 1.0, confidence=0.0)

        d = float(np.linalg.norm(np.asarray(thumb) - np.asarray(index))) / palm

        was_closed = prev.closed if prev is not None else False
        if was_closed:
            closed = d <= self._open
        else:
            closed = d < self._close

        # width: 0 at/below close_ratio, 1 at/above open_ratio.
        width = float(np.clip((d - self._close) / (self._open - self._close), 0.0, 1.0))
        return GripperState(closed=closed, width=width, confidence=1.0)


_GRIPPERS: dict[str, type[Gripper]] = {"binary": BinaryGripper}


def load_gripper(name: str = "binary", **kwargs) -> Gripper:
    """Instantiate a gripper by name (``cfg.GRIPPER``)."""
    try:
        cls = _GRIPPERS[name]
    except KeyError:
        raise ValueError(
            f"unknown gripper {name!r}; known: {', '.join(sorted(_GRIPPERS))}"
        ) from None
    return cls(**kwargs)
