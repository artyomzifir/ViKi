"""RealSenseBackend SDK-callback capture path: _on_frame stashes the newest
colour/depth pair, get_frame blocks for a fresh one, times out when starved, and
surfaces a callback error. No device — the rs.frame is faked."""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("pyrealsense2")

from viki.cameras.realsense import RealSenseBackend
from viki.contracts import CameraIntrinsics


class _FakeVideoFrame:
    def __init__(self, arr, ts_ms):
        self._arr = arr
        self._ts = ts_ms

    def get_data(self):
        return self._arr

    def get_timestamp(self):
        return self._ts

    # threshold_filter.process(d).as_depth_frame() is a no-op passthrough here
    def as_depth_frame(self):
        return self


class _FakeFrameset:
    def __init__(self, color, depth):
        self._c, self._d = color, depth

    def __bool__(self):
        return True

    def as_frameset(self):
        return self

    def get_color_frame(self):
        return self._c

    def get_depth_frame(self):
        return self._d


def _backend():
    b = RealSenseBackend(serial="sim", color_resolution=(8, 6), depth_resolution=(8, 6))
    # start() talks to a device; wire just the state _on_frame / get_frame need.
    b._running = True
    b._resolved_serial = "sim"
    b._align = None
    b._threshold = None
    b._depth_units_m = 0.001
    b._color_ci = CameraIntrinsics(fx=1, fy=1, cx=4, cy=3, width=8, height=6)
    b._depth_ci = CameraIntrinsics(fx=1, fy=1, cx=4, cy=3, width=8, height=6)
    return b


def _frameset(seed):
    color = np.full((6, 8, 3), seed % 256, np.uint8)
    depth = np.full((6, 8), seed * 100, np.uint16)
    return _FakeFrameset(_FakeVideoFrame(color, seed * 33.0), _FakeVideoFrame(depth, seed * 33.0))


def test_on_frame_then_get_frame_returns_it():
    b = _backend()
    b._on_frame(_frameset(1))
    f = b.get_frame()
    assert f.color.shape == (6, 8, 3) and f.depth.shape == (6, 8)
    assert int(f.depth[0, 0]) == 100
    assert f.timestamp_us == 33_000
    # the arrays are copies, not views over the fake frame's buffers
    assert f.color.flags.owndata and f.depth.flags.owndata


def test_get_frame_blocks_until_a_new_pair_arrives():
    b = _backend()
    b._on_frame(_frameset(1))
    b.get_frame()  # consume seq 1

    out = {}

    def reader():
        out["f"] = b.get_frame()

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.1)
    assert t.is_alive()  # no new frame yet → still blocked
    b._on_frame(_frameset(2))
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert int(out["f"].depth[0, 0]) == 200


def test_get_frame_times_out_when_starved():
    b = _backend()
    b._timeout_ms = 120
    b._on_frame(_frameset(1))
    b.get_frame()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        b.get_frame()
    assert time.monotonic() - t0 >= 0.1


def test_callback_error_surfaces_through_get_frame():
    b = _backend()

    class Boom:
        def as_frameset(self):
            raise RuntimeError("frame decode blew up")

    b._on_frame(Boom())
    with pytest.raises(RuntimeError, match="frame decode blew up"):
        b.get_frame()


def test_stop_wakes_a_blocked_reader():
    b = _backend()
    b._on_frame(_frameset(1))
    b.get_frame()
    b._pipeline = None  # stop() would have set this; simulate no live pipeline

    err = {}

    def reader():
        try:
            b.get_frame()
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.1)
    b.stop()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert isinstance(err.get("e"), RuntimeError)
