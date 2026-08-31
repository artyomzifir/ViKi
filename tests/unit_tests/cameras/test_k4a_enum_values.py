"""The k4a depth-mode ints in kinect.py must match the installed SDK header.

They were all +1 too high once, so a "NFOV_UNBINNED" request captured 512x512
(WFOV_2X2BINNED) depth without any error.
"""

import re
from pathlib import Path

import pytest

_HDR = Path("/usr/include/k4a/k4atypes.h")


def _enum_values(name: str) -> dict[str, int]:
    if not _HDR.is_file():
        pytest.skip("k4a SDK header not installed")
    block = re.search(rf"typedef enum[^{{]*{{(.*?)}}\s*{name};", _HDR.read_text(), re.S)
    assert block, f"{name} not found in header"
    out, nxt = {}, 0
    for line in block.group(1).splitlines():
        m = re.match(r"\s*(K4A_\w+)\s*(?:=\s*(\d+))?", line)
        if not m:
            continue
        val = int(m.group(2)) if m.group(2) else nxt
        out[m.group(1)] = val
        nxt = val + 1
    return out


def test_depth_mode_ints_match_sdk():
    from viki.cameras import kinect

    sdk = _enum_values("k4a_depth_mode_t")
    for const in ("K4A_DEPTH_MODE_NFOV_2X2BINNED", "K4A_DEPTH_MODE_NFOV_UNBINNED",
                  "K4A_DEPTH_MODE_WFOV_2X2BINNED", "K4A_DEPTH_MODE_WFOV_UNBINNED"):
        assert getattr(kinect, const) == sdk[const], (
            f"{const}: kinect.py={getattr(kinect, const)} sdk={sdk[const]}")


def test_fps_and_color_ints_match_sdk():
    from viki.cameras import kinect

    fps = _enum_values("k4a_fps_t")
    assert kinect.K4A_FRAMES_PER_SECOND_5 == fps["K4A_FRAMES_PER_SECOND_5"]
    assert kinect.K4A_FRAMES_PER_SECOND_15 == fps["K4A_FRAMES_PER_SECOND_15"]
    assert kinect.K4A_FRAMES_PER_SECOND_30 == fps["K4A_FRAMES_PER_SECOND_30"]

    col = _enum_values("k4a_color_resolution_t")
    assert kinect.K4A_COLOR_RESOLUTION_720P == col["K4A_COLOR_RESOLUTION_720P"]
    assert kinect.K4A_COLOR_RESOLUTION_1080P == col["K4A_COLOR_RESOLUTION_1080P"]
    assert kinect.K4A_COLOR_RESOLUTION_1536P == col["K4A_COLOR_RESOLUTION_1536P"]
