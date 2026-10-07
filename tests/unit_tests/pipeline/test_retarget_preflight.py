"""Readiness checks for the default cube-aware retarget path."""

import asyncio
import json

import numpy as np
import pytest
from fastapi import HTTPException

from viki.episode import new_episode
from viki.perception import OBJECT_MODEL_SCHEMA
from viki.retarget import config_from_options, retarget_prerequisite_errors
from viki.server.routes.pipeline import _RetargetReq, retarget


def _prepared_episode(root, *, label="manipulated_object", object_frames=4):
    ep = new_episode(root)
    np.savez_compressed(ep.cln_npz, timestamps=np.arange(4))
    (ep.raw_dir / "world_anchor.json").write_text(json.dumps({"T_world_display": np.eye(4).tolist()}))
    ep.object_models_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ep.object_models_npz,
        schema=OBJECT_MODEL_SCHEMA,
        object_labels=np.asarray([label]),
        translation_world=np.zeros((object_frames, 1, 3)),
    )
    return ep


def test_default_cube_mode_requires_prepared_tracks(tmp_path):
    cfg = config_from_options("ur10", {"object_grasp": True})
    ep = new_episode(tmp_path)
    errors = retarget_prerequisite_errors(ep, cfg)
    assert any("cln.npz" in error for error in errors)
    assert any("object_models.npz" in error for error in errors)
    assert any("world_anchor.json" in error for error in errors)


def test_cube_mode_checks_label_and_alignment(tmp_path):
    cfg = config_from_options("ur10", {"object_grasp": True})
    ep = _prepared_episode(tmp_path, label="other_dynamic", object_frames=3)
    errors = retarget_prerequisite_errors(ep, cfg)
    assert any("manipulated_object" in error for error in errors)
    assert any("different lengths" in error for error in errors)


def test_cube_mode_accepts_aligned_inputs_and_hand_only_opt_out(tmp_path):
    ep = _prepared_episode(tmp_path)
    assert retarget_prerequisite_errors(ep, config_from_options("ur10", {"object_grasp": True})) == []
    ep.object_models_npz.unlink()
    assert retarget_prerequisite_errors(ep, config_from_options("ur10", {"object_grasp": False})) == []


def test_retarget_batch_rejects_all_when_one_episode_is_unready(tmp_path, monkeypatch):
    dataset = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(dataset), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    ready = _prepared_episode(dataset)
    missing = new_episode(dataset / "other")
    submitted = []
    monkeypatch.setattr(
        "viki.server.jobs.submit",
        lambda *args, **kwargs: submitted.append((args, kwargs)),
    )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(retarget(_RetargetReq(
            episodes=[str(ready.root), str(missing.root)],
            opts={"object_grasp": True},
        )))
    assert caught.value.status_code == 409
    assert missing.id in caught.value.detail
    assert submitted == []
