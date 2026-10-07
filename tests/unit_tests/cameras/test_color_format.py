"""
Colour wire format — ``viki.cameras.kinect``.

The k4a enum is MJPG=0, NV12=1, YUY2=2, BGRA32=3, DEPTH16=4. ViKi's constants
used to read BGRA32=0 and DEPTH16=3, so the device was being asked for MJPG
while the code said BGRA32. The values are now the SDK's, and the format is
chosen by name so the wire format cannot drift silently again.
"""

from __future__ import annotations

import pytest

from viki.cameras import kinect as k


def test_constants_match_the_sdk_enum():
    assert k.K4A_IMAGE_FORMAT_COLOR_MJPG == 0
    assert k.K4A_IMAGE_FORMAT_COLOR_NV12 == 1
    assert k.K4A_IMAGE_FORMAT_COLOR_YUY2 == 2
    assert k.K4A_IMAGE_FORMAT_COLOR_BGRA32 == 3
    assert k.K4A_IMAGE_FORMAT_DEPTH16 == 4


def test_default_is_mjpg_the_format_the_rig_has_actually_been_streaming():
    b = k.KinectBackend(device_index=0)
    assert b._color_format == "mjpg"
    assert k._COLOR_FORMAT_MAP[b._color_format] == k.K4A_IMAGE_FORMAT_COLOR_MJPG


def test_explicit_choice_wins():
    b = k.KinectBackend(device_index=0, color_format="bgra32")
    assert k._COLOR_FORMAT_MAP[b._color_format] == k.K4A_IMAGE_FORMAT_COLOR_BGRA32


def test_an_unknown_format_is_refused_not_guessed():
    with pytest.raises(ValueError, match="unknown color_format"):
        k.KinectBackend(device_index=0, color_format="rgb24")
