"""
viki.gripper
------------
Gripper-state estimation from a fused hand skeleton.

``Gripper`` is the abstraction. ``LinearGripper`` is the production default:
it maps the thumb–index fingertip gap, normalised by palm length, to a continuous
opening in ``[0, 1]`` and low-pass filters it over time. ``BinaryGripper`` is
retained only for protected V1 baselines and old artifacts.

``retarget`` applies the selected physical tool's velocity limit and maps this
semantic opening to its URDF joint. ``export`` forwards the profiled opening; it
does not re-derive it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from viki.contracts import LM, GripperState

_MIN_LEN = 1e-6


def _pinch_ratio(hand_points: Mapping[LM, np.ndarray]) -> float | None:
    """Thumb/index distance divided by wrist/middle-MCP palm length."""
    thumb = hand_points.get(LM.THUMB_TIP)
    index = hand_points.get(LM.INDEX_TIP)
    wrist = hand_points.get(LM.WRIST)
    middle = hand_points.get(LM.MIDDLE_MCP)
    values = [thumb, index, wrist, middle]
    if any(value is None or not np.all(np.isfinite(value)) for value in values):
        return None
    palm = float(np.linalg.norm(np.asarray(wrist) - np.asarray(middle)))
    if palm < _MIN_LEN:
        return None
    return float(np.linalg.norm(np.asarray(thumb) - np.asarray(index))) / palm


class Gripper(ABC):
    """Maps hand observations to state and state to robot command vectors."""

    name: str
    command_names: tuple[str, ...]

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

    @abstractmethod
    def command(self, state: GripperState) -> np.ndarray:
        """Encode one semantic state into this gripper's actuator space."""

    def encode(self, states: list[GripperState]) -> np.ndarray:
        """Encode a trajectory without exposing gripper-specific dimensions."""
        if not states:
            return np.empty((0, len(self.command_names)), dtype=np.float32)
        values = np.stack([self.command(state) for state in states])
        expected = (len(states), len(self.command_names))
        if values.shape != expected:
            raise ValueError(
                f"{self.name} encoded {values.shape}, expected command shape {expected}"
            )
        return values.astype(np.float32, copy=False)


class BinaryGripper(Gripper):
    """
    Open/closed from the normalised thumb–index fingertip gap.

    ``d = ||THUMB_TIP - INDEX_TIP|| / ||WRIST - MIDDLE_MCP||``

    Hysteresis: an open gripper closes only once ``d < close_ratio``; a closed
    gripper opens only once ``d > open_ratio`` (``close_ratio < open_ratio``).
    When landmarks are missing the previous state is held (confidence 0).
    """

    name = "binary"
    command_names = ("closed",)

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
        d = _pinch_ratio(hand_points)
        if d is None:
            held = prev.closed if prev is not None else False
            return GripperState(closed=held, width=0.0 if held else 1.0, confidence=0.0)

        was_closed = prev.closed if prev is not None else False
        if was_closed:
            closed = d <= self._open
        else:
            closed = d < self._close

        # width: 0 at/below close_ratio, 1 at/above open_ratio.
        width = float(np.clip((d - self._close) / (self._open - self._close), 0.0, 1.0))
        return GripperState(closed=closed, width=width, confidence=1.0)

    def command(self, state: GripperState) -> np.ndarray:
        return np.asarray([1.0 if state.closed else 0.0], dtype=np.float32)


class LinearGripper(Gripper):
    """Continuous normalised opening estimated from the human pinch distance.

    ``closed_ratio`` maps to 0 (closed), ``open_ratio`` maps to 1 (fully open).
    Values between them interpolate linearly. An exponential filter suppresses
    landmark jitter; the physical velocity limit is applied later by retarget.
    """

    name = "linear"
    command_names = ("opening",)

    def __init__(
        self,
        closed_ratio: float = 0.35,
        open_ratio: float = 1.20,
        smoothing_alpha: float = 0.35,
    ) -> None:
        if not 0.0 <= closed_ratio < open_ratio:
            raise ValueError("need 0 <= closed_ratio < open_ratio")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self._closed = float(closed_ratio)
        self._open = float(open_ratio)
        self._alpha = float(smoothing_alpha)

    def estimate(
        self,
        hand_points: Mapping[LM, np.ndarray],
        prev: GripperState | None,
    ) -> GripperState:
        ratio = _pinch_ratio(hand_points)
        if ratio is None:
            if prev is not None:
                return GripperState(prev.closed, prev.width, 0.0)
            return GripperState(closed=False, width=1.0, confidence=0.0)
        target = float(np.clip(
            (ratio - self._closed) / (self._open - self._closed), 0.0, 1.0
        ))
        opening = (
            target
            if prev is None
            else prev.width + self._alpha * (target - prev.width)
        )
        opening = float(np.clip(opening, 0.0, 1.0))
        return GripperState(closed=opening <= 0.05, width=opening, confidence=1.0)

    def command(self, state: GripperState) -> np.ndarray:
        return np.asarray([np.clip(state.width, 0.0, 1.0)], dtype=np.float32)


_GRIPPERS: dict[str, type[Gripper]] = {
    "binary": BinaryGripper,
    "linear": LinearGripper,
}


def load_gripper(name: str = "linear", **kwargs) -> Gripper:
    """Instantiate a gripper by name (``cfg.GRIPPER``)."""
    try:
        cls = _GRIPPERS[name]
    except KeyError:
        raise ValueError(
            f"unknown gripper {name!r}; known: {', '.join(sorted(_GRIPPERS))}"
        ) from None
    return cls(**kwargs)
