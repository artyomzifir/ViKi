"""Synthetic, exact-frame tests for the HO-Cap stage scorer."""

import json

import numpy as np
import pytest

from viki.benchmark.evaluate import evaluate_episode
from viki.contracts import LM


@pytest.fixture
def scored_episode(tmp_path):
    episode = tmp_path / "hocap_ep"
    (episode / "raw").mkdir(parents=True)
    (episode / "gt").mkdir()
    (episode / "meta.json").write_text(json.dumps({
        "source": "hocap",
        "timestamps": "synthetic",
        "cameras": {"cam_a": {}, "cam_b": {}},
    }))
    (episode / "raw/timestamps.json").write_text(json.dumps([
        {"sync_us": index * 100_000, "offsets_us": {"cam_a": 0, "cam_b": 0}}
        for index in range(3)
    ]))
    truth = np.zeros((3, 21, 3), np.float32)
    truth[:, :, 0] = np.arange(3, dtype=np.float32)[:, None] * 0.1
    np.savez(
        episode / "gt/joints3d_world.npz",
        joints_world=truth,
        valid=np.ones((3, 21), bool),
        frame_index=np.arange(3),
        joint_order=np.asarray([landmark.name for landmark in LM]),
        world_frame=np.asarray("hocap_tag_1"),
    )
    np.savez(
        episode / "rec.npz",
        device_ids=np.asarray(["cam_a", "cam_a"]),
        timestamps=np.asarray([0, 100_000], np.int64),
        landmark_ids=np.arange(21),
        points=truth[:2] + np.asarray([0.01, 0, 0], np.float32),
    )
    observed = np.ones((3, 21), bool)
    observed[2] = False
    np.savez(
        episode / "cln.npz",
        timestamps=np.asarray([0, 100_000, 200_000], np.int64),
        landmark_ids=np.arange(21),
        smoothed_points=truth + np.asarray([0.02, 0, 0], np.float32),
        observed_points=truth + np.asarray([0.005, 0, 0], np.float32),
        observed_mask=observed,
    )
    (episode / "status.json").write_text(json.dumps({"stages": {"prepare": {"done": True}}}))
    return episode


def test_scores_coverage_and_does_not_count_filled_points_as_observed(scored_episode):
    report = evaluate_episode(scored_episode)
    camera = report["rec_per_camera"]["cam_a"]
    assert camera["matched_joints"] == 42
    assert camera["coverage"] == pytest.approx(2 / 3)
    assert camera["mpjpe_mm"] == pytest.approx(10.0, abs=0.001)
    assert report["rec_per_camera"]["cam_b"]["matched_joints"] == 0
    assert report["rec_per_camera"]["cam_b"]["mpjpe_mm"] is None
    prepared = report["prepare"]
    assert prepared["smoothed_all"]["matched_joints"] == 63
    assert prepared["smoothed_observed"]["matched_joints"] == 42
    assert prepared["smoothed_observed"]["mpjpe_mm"] == pytest.approx(20.0, abs=0.001)
    assert prepared["fused_observed"]["mpjpe_mm"] == pytest.approx(5.0, abs=0.001)


def test_rejects_timestamp_shift_instead_of_nearest_neighbour_matching(scored_episode):
    with np.load(scored_episode / "rec.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["timestamps"] = np.asarray([1, 100_001], np.int64)
    np.savez(scored_episode / "rec.npz", **arrays)
    with pytest.raises(ValueError, match="not a capture tick"):
        evaluate_episode(scored_episode)


def test_rejects_duplicate_frame_predictions(scored_episode):
    with np.load(scored_episode / "cln.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["timestamps"] = np.asarray([0, 0, 200_000], np.int64)
    np.savez(scored_episode / "cln.npz", **arrays)
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate_episode(scored_episode)


def test_requires_imported_hocap_episode(scored_episode):
    meta_path = scored_episode / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["source"] = "local"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="HO-Cap import"):
        evaluate_episode(scored_episode)
