"""Continuous semantic gripper estimation."""

import numpy as np
import pytest

from viki.contracts import LM, GripperState
from viki.gripper import LinearGripper, load_gripper


def _hand(pinch: float, palm: float = 1.0) -> dict:
    return {
        LM.WRIST: np.array([0.0, 0.0, 0.0]),
        LM.MIDDLE_MCP: np.array([0.0, palm, 0.0]),
        LM.THUMB_TIP: np.array([0.0, 0.0, 0.0]),
        LM.INDEX_TIP: np.array([pinch * palm, 0.0, 0.0]),
    }


def test_linear_is_default_and_owns_opening_command():
    model = load_gripper()
    assert isinstance(model, LinearGripper)
    assert model.command_names == ("opening",)
    np.testing.assert_allclose(
        model.encode([GripperState(False, 0.37, 1.0)]), [[0.37]]
    )


def test_pinch_maps_linearly_between_closed_and_open():
    model = LinearGripper(closed_ratio=0.2, open_ratio=1.2, smoothing_alpha=1.0)
    assert model.estimate(_hand(0.2), None).width == pytest.approx(0.0)
    assert model.estimate(_hand(0.7), None).width == pytest.approx(0.5)
    assert model.estimate(_hand(1.2), None).width == pytest.approx(1.0)


def test_temporal_filter_removes_single_frame_jump():
    model = LinearGripper(closed_ratio=0.2, open_ratio=1.2, smoothing_alpha=0.25)
    previous = model.estimate(_hand(1.2), None)
    current = model.estimate(_hand(0.2), previous)
    assert previous.width == pytest.approx(1.0)
    assert current.width == pytest.approx(0.75)
    assert current.closed is False


def test_missing_landmarks_hold_continuous_width():
    model = LinearGripper()
    previous = GripperState(False, 0.42, 1.0)
    current = model.estimate({}, previous)
    assert current.width == pytest.approx(0.42)
    assert current.confidence == 0.0


@pytest.mark.parametrize("kwargs", [
    {"closed_ratio": 1.0, "open_ratio": 0.5},
    {"smoothing_alpha": 0.0},
    {"smoothing_alpha": 1.1},
])
def test_invalid_linear_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LinearGripper(**kwargs)
