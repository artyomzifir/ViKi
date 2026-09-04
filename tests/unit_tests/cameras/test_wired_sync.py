"""
Kinect wired hardware sync wiring + the stale-frame guard default.
"""

from __future__ import annotations

import pytest

from viki.cameras.hw_sync import (
    HardwareSyncError,
    WIRED_MASTER,
    WIRED_STANDALONE,
    WIRED_SUBORDINATE,
    build_sync_plan,
    require_role_jack,
)
from viki.cameras.sync import MultiCameraSync


class _FakeMgr:
    def active_device_ids(self):
        return []


@pytest.mark.parametrize("fps,expect_us", [(30, 50000), (15, 100000), (5, 300000)])
def test_max_offset_default_tracks_frame_period(fps, expect_us):
    s = MultiCameraSync(_FakeMgr(), sync_fps=fps)
    # 1.5 / sync_fps — clear of the half-period phase + jitter (a threshold
    # under one frame period drops good frames), still trips on a real stall.
    assert s._max_offset_us == expect_us


def test_explicit_max_offset_still_wins():
    s = MultiCameraSync(_FakeMgr(), sync_fps=30, max_offset_us=8000)
    assert s._max_offset_us == 8000


def test_single_kinect_may_be_standalone():
    plan = build_sync_plan(["kinect_0"], {})
    assert plan["kinect_0"].mode == WIRED_STANDALONE


def test_multi_kinect_plan_covers_exact_rig():
    plan = build_sync_plan(
        ["kinect_0", "kinect_1"],
        {"master": "kinect_1", "subordinates": ["kinect_0"],
         "subordinate_delay_us": 160},
    )
    assert plan["kinect_1"].mode == WIRED_MASTER
    assert plan["kinect_1"].delay_us == 0
    assert plan["kinect_0"].mode == WIRED_SUBORDINATE
    assert plan["kinect_0"].delay_us == 160


@pytest.mark.parametrize(
    "spec,match",
    [
        ({}, "master is not set"),
        ({"master": "kinect_0", "subordinates": []}, "unassigned connected"),
        ({"master": "kinect_0", "subordinates": ["kinect_1", "kinect_1"]},
         "duplicate"),
        ({"master": "kinect_0", "subordinates": ["kinect_0", "kinect_1"]},
         "both master"),
    ],
)
def test_multi_kinect_invalid_plan_is_refused(spec, match):
    with pytest.raises(HardwareSyncError, match=match):
        build_sync_plan(["kinect_0", "kinect_1"], spec)


def test_required_physical_jack_is_fail_closed():
    plan = build_sync_plan(
        ["kinect_0", "kinect_1"],
        {"master": "kinect_1", "subordinates": ["kinect_0"]},
    )
    with pytest.raises(HardwareSyncError, match="SYNC OUT"):
        require_role_jack(
            "kinect_1", plan["kinect_1"],
            sync_in_connected=False, sync_out_connected=False,
        )
    with pytest.raises(HardwareSyncError, match="SYNC IN"):
        require_role_jack(
            "kinect_0", plan["kinect_0"],
            sync_in_connected=False, sync_out_connected=True,
        )


def test_hardware_timestamp_mismatch_is_refused():
    from viki.cameras.manager import CameraManager

    plan = build_sync_plan(
        ["kinect_0", "kinect_1"],
        {"master": "kinect_1", "subordinates": ["kinect_0"],
         "subordinate_delay_us": 160},
    )

    class Worker:
        def __init__(self, timestamp_us):
            self._frame = type("Frame", (), {"timestamp_us": timestamp_us})()

        def snapshot(self):
            return [self._frame]

    mgr = CameraManager()
    mgr._workers = {"kinect_1": Worker(1_000_000), "kinect_0": Worker(1_002_000)}
    alignment = mgr._hardware_timestamp_alignment(plan)
    assert alignment["verified"] is False
    assert alignment["offsets"]["kinect_0"]["residual_us"] == 1840


def test_manager_starts_whole_rig_subordinate_first(monkeypatch):
    from viki import config
    from viki.cameras.manager import CameraManager

    monkeypatch.setattr(
        config, "KINECT_SYNC",
        {"master": "kinect_1", "subordinates": ["kinect_0"],
         "subordinate_delay_us": 160},
        raising=False,
    )
    mgr = CameraManager()
    monkeypatch.setattr(
        mgr, "_detected_kinect_ids", lambda: ["kinect_0", "kinect_1"],
    )
    events = []

    class Backend:
        def __init__(self, device_id, cfg):
            self.device_id = device_id
            self._cfg = cfg

        @property
        def config(self):
            return self._cfg

        def start(self):
            events.append(("start", self.device_id))

        def stop(self):
            events.append(("stop", self.device_id))

    class Worker:
        def __init__(self, backend):
            self.backend = backend
            self._frames = []

        def start(self):
            self.backend.start()
            timestamp_us = (
                1_000_000 + int(self.backend.config.get("subordinate_delay_us", 0))
            )
            self._frames = [type("Frame", (), {"timestamp_us": timestamp_us})()]

        def stop(self):
            self.backend.stop()

        def join(self, timeout=8.0):
            pass

        def latest(self):
            return None

        def snapshot(self):
            return list(self._frames)

    def make_backend(device_id, fps, color_width, color_height, depth_mode, **kwargs):
        mode = int(kwargs["wired_sync_mode"])
        cfg = {
            "fps": fps, "color_width": color_width,
            "color_height": color_height, "depth_mode": depth_mode,
            **kwargs,
            "sync_in_connected": mode == WIRED_SUBORDINATE,
            "sync_out_connected": mode == WIRED_MASTER,
        }
        return Backend(device_id, cfg)

    monkeypatch.setattr(CameraManager, "_make_backend", staticmethod(make_backend))
    monkeypatch.setattr("viki.cameras.manager._CameraWorker", Worker)

    outcomes = mgr.start_configured_kinect_rig(fps=15)
    assert events[:2] == [("start", "kinect_0"), ("start", "kinect_1")]
    assert outcomes == {"kinect_0": "started", "kinect_1": "started"}
    assert mgr.get_backend("kinect_0").config["wired_sync_mode"] == WIRED_SUBORDINATE
    assert mgr.get_backend("kinect_1").config["wired_sync_mode"] == WIRED_MASTER
    assert mgr.require_hardware_sync_ready()["ready"] is True


def test_direct_manager_start_cannot_request_wrong_role(monkeypatch):
    from viki import config
    from viki.cameras.manager import CameraManager

    monkeypatch.setattr(
        config, "KINECT_SYNC",
        {"master": "kinect_1", "subordinates": ["kinect_0"]},
        raising=False,
    )
    mgr = CameraManager()
    monkeypatch.setattr(
        mgr, "_detected_kinect_ids", lambda: ["kinect_0", "kinect_1"],
    )
    with pytest.raises(HardwareSyncError, match="must run as subordinate"):
        mgr.start("kinect_0", wired_sync_mode=WIRED_MASTER)


def test_record_force_cannot_bypass_hardware_sync_gate():
    import asyncio

    from fastapi import HTTPException

    from viki.server.routes.recording import RecordRequest, start_recording

    class NotReadyManager:
        def active_device_ids(self):
            return ["kinect_0", "kinect_1"]

        def require_hardware_sync_ready(self):
            raise HardwareSyncError("SYNC IN is disconnected")

    with pytest.raises(HTTPException, match="SYNC IN is disconnected") as exc:
        asyncio.run(start_recording(RecordRequest(force=True), NotReadyManager()))
    assert exc.value.status_code == 409
