"""The CLI wires scene artifacts into the default cube-aware path."""

from types import SimpleNamespace

import pytest

from viki.cli import _build_parser
from viki.episode import new_episode


def test_run_requires_a_manipulated_object_before_work_starts(tmp_path, monkeypatch):
    ep = new_episode(tmp_path)
    monkeypatch.setattr(
        "viki.retarget.config_from_options",
        lambda *args: SimpleNamespace(object_grasp=True),
    )
    parser = _build_parser()

    args = parser.parse_args(["run", str(ep.root)])
    with pytest.raises(ValueError, match="--scene-prompts"):
        args.func(args)
    args = parser.parse_args(["run", str(ep.root), "--scene-objects", "3"])
    with pytest.raises(ValueError, match="curated --scene-prompts"):
        args.func(args)


def test_run_passes_explicit_single_object_label_to_scene(tmp_path, monkeypatch):
    ep = new_episode(tmp_path)
    calls = []
    monkeypatch.setattr(
        "viki.retarget.config_from_options",
        lambda *args: SimpleNamespace(object_grasp=True),
    )
    monkeypatch.setattr(
        "viki.perception.extract.extract_episode",
        lambda *args, **kwargs: calls.append("extract"),
    )
    monkeypatch.setattr(
        "viki.prepare.run.prepare_episode",
        lambda *args, **kwargs: calls.append("prepare"),
    )
    monkeypatch.setattr(
        "viki.perception.scene.scene_perception_episode",
        lambda episode, options, **kwargs: calls.append(("scene", options.object_label)),
    )
    monkeypatch.setattr(
        "viki.retarget.retarget_episode",
        lambda *args, **kwargs: calls.append("retarget"),
    )
    monkeypatch.setattr(
        "viki.object_centric.build_object_relative_episode",
        lambda *args, **kwargs: calls.append("object_relative"),
    )
    monkeypatch.setattr(
        "viki.replay.replay_episode",
        lambda *args, **kwargs: calls.append("replay"),
    )

    args = _build_parser().parse_args([
        "run", str(ep.root), "--scene-objects", "1",
        "--scene-object-label", "manipulated_object",
    ])
    args.func(args)
    assert calls == [
        "extract", "prepare", ("scene", "manipulated_object"),
        "retarget", "object_relative", "replay",
    ]
