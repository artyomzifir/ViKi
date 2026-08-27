"""viki.episode + viki.labeling round-trips and validation."""

import pytest

from viki.contracts import EpisodeLabels, Segment
from viki.episode import mark_stage, new_episode, read_status, stage_done
from viki.labeling import load_labels, save_labels, validate_labels


def test_new_episode_layout(tmp_path):
    ep = new_episode(tmp_path)
    assert ep.raw_dir.is_dir()
    assert ep.meta_path.exists() and ep.status_path.exists()
    assert read_status(ep) == {"stages": {}}


def test_mark_stage_and_stage_done(tmp_path):
    ep = new_episode(tmp_path)
    assert stage_done(ep, "prepare") is False
    mark_stage(ep, "prepare", frames=412, object_relative=False)
    assert stage_done(ep, "prepare") is True
    assert read_status(ep)["stages"]["prepare"]["frames"] == 412


def test_mark_unknown_stage_raises(tmp_path):
    ep = new_episode(tmp_path)
    with pytest.raises(ValueError):
        mark_stage(ep, "bogus")


def test_labels_roundtrip(tmp_path):
    ep = new_episode(tmp_path)
    labels = EpisodeLabels(
        task="pick up the red block",
        hand="left",
        segments=[Segment(0, 10, "approach"), Segment(10, 25, "grasp")],
        outcome="good",
        notes="clean take",
    )
    save_labels(ep, labels)
    got = load_labels(ep)
    assert got.task == labels.task and got.hand == "left" and got.outcome == "good"
    assert [s.label for s in got.segments] == ["approach", "grasp"]


def test_validate_rejects_out_of_range_segment():
    labels = EpisodeLabels(task="t", segments=[Segment(0, 50, "approach")])
    with pytest.raises(ValueError):
        validate_labels(labels, n_frames=30)


def test_validate_rejects_overlapping_segments():
    labels = EpisodeLabels(
        task="t", segments=[Segment(0, 15, "a"), Segment(10, 20, "b")]
    )
    with pytest.raises(ValueError):
        validate_labels(labels, n_frames=30)


def test_validate_requires_task_for_export():
    with pytest.raises(ValueError):
        validate_labels(EpisodeLabels(task="  "), n_frames=10, for_export=True)
