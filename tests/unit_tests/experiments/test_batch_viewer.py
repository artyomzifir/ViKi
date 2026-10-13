"""Batch playback selects the latest accepted plan for each episode."""

import json

import pytest

from experiments.batch_viewer import accepted_episodes


def test_accepted_episodes_uses_latest_status_and_chronological_order(tmp_path):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    for name in ("episode-b", "episode-c"):
        (episodes_root / name).mkdir()
        (episodes_root / name / "plan.h5").touch()
    events = [
        {"episode": "episode-c", "status": "accepted"},
        {"episode": "episode-a", "status": "accepted"},
        {"episode": "episode-b", "status": "accepted"},
        {"episode": "episode-a", "status": "plan_rejected"},
    ]
    status = episodes_root / "status.jsonl"
    status.write_text("\n".join(json.dumps(event) for event in events))

    assert [episode.name for episode in accepted_episodes(status, episodes_root)] == [
        "episode-b", "episode-c",
    ]


def test_accepted_episodes_rejects_missing_plan(tmp_path):
    root = tmp_path / "episodes"
    root.mkdir()
    (root / "missing").mkdir()
    status = root / "status.jsonl"
    status.write_text(json.dumps({"episode": "missing", "status": "accepted"}))

    with pytest.raises(FileNotFoundError, match="missing"):
        accepted_episodes(status, root)
