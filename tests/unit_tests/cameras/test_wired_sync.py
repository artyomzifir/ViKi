"""
Kinect wired hardware sync wiring + the stale-frame guard default.
"""

from __future__ import annotations

import pytest

from viki.cameras.sync import MultiCameraSync


class _FakeMgr:
    def active_device_ids(self):
        return []


@pytest.mark.parametrize("fps,expect_us", [(30, 25000), (15, 50000), (5, 150000)])
def test_max_offset_default_tracks_frame_period(fps, expect_us):
    s = MultiCameraSync(_FakeMgr(), sync_fps=fps)
    assert s._max_offset_us == expect_us  # 0.75 / sync_fps, a stale-frame guard


def test_explicit_max_offset_still_wins():
    s = MultiCameraSync(_FakeMgr(), sync_fps=30, max_offset_us=8000)
    assert s._max_offset_us == 8000


def test_wired_sync_for_resolves_roles(monkeypatch):
    from viki.server.routes import cameras as cam_routes

    monkeypatch.setattr(
        cam_routes.config, "KINECT_SYNC",
        {"master": "kinect_0", "subordinates": ["kinect_1"], "subordinate_delay_us": 160},
        raising=False,
    )
    assert cam_routes._wired_sync_for("kinect_0") == (cam_routes._WIRED_MASTER, 0)
    assert cam_routes._wired_sync_for("kinect_1") == (cam_routes._WIRED_SUBORDINATE, 160)
    assert cam_routes._wired_sync_for("021222070553") == (0, 0)   # RealSense: standalone


def test_wired_sync_for_empty_config_is_standalone(monkeypatch):
    from viki.server.routes import cameras as cam_routes

    monkeypatch.setattr(cam_routes.config, "KINECT_SYNC", {}, raising=False)
    assert cam_routes._wired_sync_for("kinect_0") == (0, 0)
