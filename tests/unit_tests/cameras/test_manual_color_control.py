"""
Manual exposure / white balance — ``viki.cameras.kinect``.

Multi-device sync requires manual colour control: auto exposure changes frame
timing, and a Kinect timestamp marks the *centre* of exposure, so a floating
exposure walks the inter-camera offset away from ``subordinate_delay_us`` even
under a perfect hardware trigger.
"""

from __future__ import annotations

import pytest

from viki.cameras import kinect as k


@pytest.fixture()
def backend():
    return k.KinectBackend(device_index=0, color_resolution=(1280, 720), fps=30)


def _record_calls(monkeypatch, result=0):
    calls = []

    def fake(handle, command, mode, value):
        calls.append((command, mode, int(getattr(value, "value", value))))
        return result

    monkeypatch.setattr(k._lib, "k4a_device_set_color_control", fake)
    return calls


def test_pins_exposure_and_white_balance_in_manual_mode(backend, monkeypatch):
    calls = _record_calls(monkeypatch)
    backend._manual_color_control = True
    backend._exposure_time_us = 8330
    backend._whitebalance_k = 4500

    backend._apply_manual_color_control()

    assert calls == [
        (k.K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE, k.K4A_COLOR_CONTROL_MODE_MANUAL, 8330),
        (k.K4A_COLOR_CONTROL_WHITEBALANCE, k.K4A_COLOR_CONTROL_MODE_MANUAL, 4500),
    ]
    # never AUTO — that is the mode the sync guidance forbids
    assert all(mode == k.K4A_COLOR_CONTROL_MODE_MANUAL for _, mode, _ in calls)


def test_disabled_by_config_touches_nothing(backend, monkeypatch):
    calls = _record_calls(monkeypatch)
    backend._manual_color_control = False

    backend._apply_manual_color_control()

    assert calls == []


def test_a_refusal_is_tolerated_not_fatal(backend, monkeypatch):
    """A camera that will not take the setting must still stream."""
    calls = _record_calls(monkeypatch, result=1)
    backend._manual_color_control = True

    backend._apply_manual_color_control()  # must not raise

    assert len(calls) == 2  # both attempted despite the first being refused


def test_frame_wait_timeout_matches_its_docstring(backend):
    """The docstring promised 5000 ms; a subordinate waits on master pulses."""
    assert backend._timeout_ms == 5000
