"""
Trigger sync for the D435i — ``viki.cameras.realsense``.

The rig runs one Kinect plus one RealSense, and their frames are currently
paired by software timestamp (measured P95 ~25 ms on 2026-09-21). Wiring the
Kinect's SYNC OUT to the D435i's trigger input is the way out of that; this is
the software half, inert until the cable exists.
"""

from __future__ import annotations

import pytest

from viki.cameras import realsense as r


class _Range:
    def __init__(self, lo, hi): self.min, self.max = lo, hi


class _Sensor:
    def __init__(self, supports=True, lo=0, hi=260, fail=False):
        self._supports, self._rng, self._fail = supports, _Range(lo, hi), fail
        self.set_calls = []

    def supports(self, _opt): return self._supports
    def get_option_range(self, _opt): return self._rng

    def set_option(self, _opt, value):
        if self._fail:
            raise RuntimeError("device refused")
        self.set_calls.append(value)


class _Dev:
    def __init__(self, *sensors): self._s = sensors
    def query_sensors(self): return list(self._s)


def _backend(mode):
    b = r.RealSenseBackend.__new__(r.RealSenseBackend)
    b._inter_cam_sync_mode = mode
    b._resolved_serial = "021222070553"
    return b


def test_free_running_by_default_touches_nothing():
    """A slave with no cable never yields a frame, so 0 must stay the default."""
    s = _Sensor()
    _backend(0)._apply_inter_cam_sync(_Dev(s))
    assert s.set_calls == []


def test_slave_mode_is_applied_to_the_supporting_sensor():
    unsupported, stereo = _Sensor(supports=False), _Sensor()
    _backend(2)._apply_inter_cam_sync(_Dev(unsupported, stereo))
    assert stereo.set_calls == [2.0]
    assert unsupported.set_calls == []


def test_a_mode_outside_the_device_range_is_refused():
    with pytest.raises(ValueError, match="out of range"):
        _backend(999)._apply_inter_cam_sync(_Dev(_Sensor(lo=0, hi=260)))


def test_a_refused_option_does_not_lose_the_camera():
    _backend(2)._apply_inter_cam_sync(_Dev(_Sensor(fail=True)))  # must not raise


def test_no_supporting_sensor_is_survivable():
    _backend(2)._apply_inter_cam_sync(_Dev(_Sensor(supports=False)))  # must not raise
