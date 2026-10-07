"""Wired Kinect pairs must use capture time, not USB arrival order."""

from types import SimpleNamespace

import pytest

from viki.cameras.hw_sync import HardwareSyncError
from viki.cameras.sync import MultiCameraSync


def _frame(device_id, device_us, host_us):
    return SimpleNamespace(
        device_id=device_id,
        timestamp_us=device_us,
        host_timestamp_us=host_us,
    )


class _Manager:
    def __init__(self, master, subordinate):
        self.frames = {"kinect_1": master, "kinect_0": subordinate}

    def active_device_ids(self):
        return list(self.frames)

    def nearest_frame(self, device_id, host_timestamp_us):
        frames = self.frames[device_id]
        return min(frames, key=lambda frame: abs(frame.host_timestamp_us - host_timestamp_us)) if frames else None

    def recent_frames(self, device_id):
        return list(self.frames[device_id])

    def hardware_sync_status(self):
        return {
            "required": True,
            "ready": True,
            "roles": {
                "kinect_1": {"role": "master"},
                "kinect_0": {"role": "subordinate"},
            },
            "timestamp_alignment": {
                "tolerance_us": 500,
                "offsets": {"kinect_0": {"actual_us": 1050}},
            },
        }


def test_wired_pair_ignores_adjacent_capture_that_arrived_nearer_tick(monkeypatch):
    monkeypatch.setattr("viki.cameras.sync.time.monotonic_ns", lambda: 1_000_000_000)
    master = [_frame("kinect_1", 2_000_000, 999_000)]
    subordinate = [
        _frame("kinect_0", 2_001_050, 978_000),
        _frame("kinect_0", 2_034_383, 999_300),
    ]
    group = MultiCameraSync(_Manager(master, subordinate), sync_fps=30).get_synced_frame()
    assert group is not None
    assert group.frames["kinect_0"] is subordinate[0]
    assert group.offsets_us["kinect_0"] == -22_000


def test_wired_pair_rejects_when_matching_capture_is_missing(monkeypatch):
    monkeypatch.setattr("viki.cameras.sync.time.monotonic_ns", lambda: 1_000_000_000)
    master = [_frame("kinect_1", 2_000_000, 999_000)]
    subordinate = [_frame("kinect_0", 2_034_383, 999_300)]
    assert MultiCameraSync(_Manager(master, subordinate), sync_fps=30).get_synced_frame() is None


def test_wired_pair_refuses_unverified_rig():
    manager = _Manager([], [])
    manager.hardware_sync_status = lambda: {
        "required": True, "ready": False, "error": "sync cable disconnected",
    }
    with pytest.raises(HardwareSyncError, match="sync cable disconnected"):
        MultiCameraSync(manager, sync_fps=30)


def test_wired_pair_rejects_stale_matching_capture(monkeypatch):
    monkeypatch.setattr("viki.cameras.sync.time.monotonic_ns", lambda: 1_000_000_000)
    master = [_frame("kinect_1", 2_000_000, 999_000)]
    subordinate = [
        _frame("kinect_0", 2_001_050, 920_000),
        _frame("kinect_0", 2_034_383, 999_300),
    ]
    assert MultiCameraSync(_Manager(master, subordinate), sync_fps=30).get_synced_frame() is None
