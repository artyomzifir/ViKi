"""Unit tests for viki.gripper.BinaryGripper."""

import numpy as np
import pytest

from viki.contracts import LM, GripperState
from viki.gripper import BinaryGripper, load_gripper


def _hand(pinch: float, palm: float = 1.0) -> dict:
    """Hand points where ||THUMB_TIP - INDEX_TIP|| / palm == pinch."""
    return {
        LM.WRIST: np.array([0.0, 0.0, 0.0]),
        LM.MIDDLE_MCP: np.array([0.0, palm, 0.0]),
        LM.THUMB_TIP: np.array([0.0, 0.0, 0.0]),
        LM.INDEX_TIP: np.array([pinch * palm, 0.0, 0.0]),
    }


def test_load_gripper_returns_binary():
    g = load_gripper("binary")
    assert isinstance(g, BinaryGripper)
    assert g.name == "binary"


def test_unknown_gripper_raises():
    with pytest.raises(ValueError):
        load_gripper("three-finger")


def test_bad_ratios_rejected():
    with pytest.raises(ValueError):
        BinaryGripper(close_ratio=0.9, open_ratio=0.5)


def test_wide_open_is_open():
    g = BinaryGripper()
    st = g.estimate(_hand(1.5), None)
    assert st.closed is False and st.confidence == 1.0


def test_tight_pinch_is_closed():
    g = BinaryGripper()
    st = g.estimate(_hand(0.2), None)
    assert st.closed is True


def test_hysteresis_open_to_closed_to_open():
    g = BinaryGripper(close_ratio=0.55, open_ratio=0.90)
    st = GripperState(closed=False, width=1.0, confidence=1.0)
    # In the dead band (0.55 < d < 0.90) while open -> stays open.
    st = g.estimate(_hand(0.70), st)
    assert st.closed is False
    # Drop below close_ratio -> closes.
    st = g.estimate(_hand(0.40), st)
    assert st.closed is True
    # Back into the dead band while closed -> stays closed.
    st = g.estimate(_hand(0.70), st)
    assert st.closed is True
    # Rise above open_ratio -> opens.
    st = g.estimate(_hand(1.10), st)
    assert st.closed is False


def test_missing_landmarks_hold_previous_state_zero_confidence():
    g = BinaryGripper()
    prev = GripperState(closed=True, width=0.0, confidence=1.0)
    st = g.estimate({LM.WRIST: np.array([np.nan, np.nan, np.nan])}, prev)
    assert st.closed is True and st.confidence == 0.0
