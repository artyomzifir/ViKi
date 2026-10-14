"""Resumable object benchmark status handling."""

import json

import pytest

from scripts.run_hocap_object_batch import run_batch


def test_reuses_compact_object_artifacts(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "revision": "pinned", "sequences": [{"id": "scene-one"}],
    }))
    dataset = tmp_path / "dataset"
    episode = dataset / "scene-one-8views"
    output = episode / "intermediates/segmentation/hocap-oracle-v1"
    output.mkdir(parents=True)
    (episode / "gt").mkdir()
    (output / "object_models.npz").touch()
    (episode / "gt/object_poses_world.npz").touch()
    (output / "evaluation.json").write_text(json.dumps({
        "objects": {"cube": {
            "coverage": 0.9, "translation_mean_mm": 5.0,
            "rotation_mean_deg": 3.0,
        }},
    }))
    (dataset / "batch_status.json").write_text(json.dumps({
        "sequences": {"scene-one": {"eight": {
            "status": "complete", "episode_name": episode.name,
        }}},
    }))
    run_batch(tmp_path, plan, dataset, poll_seconds=0)
    status = json.loads((dataset / "object_batch_status.json").read_text())
    assert status["sequences"]["scene-one"]["status"] == "complete"
    assert status["sequences"]["scene-one"]["objects"]["cube"]["coverage"] == 0.9
    assert not (output / "semantic_cloud").exists()


def test_marks_failed_hand_run_as_blocked(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "revision": "pinned", "sequences": [{"id": "scene-one"}],
    }))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "batch_status.json").write_text(json.dumps({
        "sequences": {"scene-one": {"eight": {"status": "failed"}}},
    }))
    with pytest.raises(RuntimeError, match="failed for 1 scenes"):
        run_batch(tmp_path, plan, dataset, poll_seconds=0)
    status = json.loads((dataset / "object_batch_status.json").read_text())
    assert status["sequences"]["scene-one"]["status"] == "blocked"
