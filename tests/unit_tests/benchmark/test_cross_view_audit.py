"""Repeated capture timestamps must not inflate cross-view hand evidence."""

import json

import numpy as np
import pytest

from scripts.audit_cross_view_hand import audit_episode


def test_skips_ambiguous_reused_camera_frames(tmp_path):
    episode = tmp_path / "recording"
    (episode / "raw").mkdir(parents=True)
    (episode / "raw/timestamps.json").write_text(json.dumps([
        {"sync_us": 0, "offsets_us": {"a": 0, "b": 0}},
        {"sync_us": 100, "offsets_us": {"a": 0, "b": 0}},
        {"sync_us": 200, "offsets_us": {"a": -100, "b": 0}},
    ]))
    points = np.zeros((6, 21, 3), np.float32)
    points[3:, :, 0] = 0.01
    np.savez(
        episode / "rec.npz",
        device_ids=np.asarray(["a", "a", "a", "b", "b", "b"]),
        timestamps=np.asarray([0, 100, 100, 0, 100, 200], np.int64),
        points=points,
        landmark_ids=np.arange(21),
    )
    result, distances = audit_episode(episode)
    assert result["paired_frames"] == 1
    assert result["ambiguous_timestamp_frames_skipped"] == 2
    assert result["all_joints"]["joint_pairs"] == 21
    assert np.median(distances) == pytest.approx(10.0, abs=0.001)
