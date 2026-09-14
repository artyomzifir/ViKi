"""Object-model API consumed by the three.js Viewer."""

import asyncio
import json

import numpy as np
import pytest

from viki.episode import new_episode


def _episode_with_object_model(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(episodes), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    ep = new_episode(episodes)
    root = ep.intermediates_dir / "segmentation" / "sam2.1_hiera_small"
    semantic = root / "semantic_cloud"
    semantic.mkdir(parents=True)
    frame_xyz = [
        np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], np.float32),
        np.asarray([[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0]], np.float32),
    ]
    for frame, xyz in enumerate(frame_xyz):
        np.savez_compressed(
            semantic / f"{frame:06d}.npz",
            xyz=xyz,
            rgb=np.zeros((len(xyz), 3), np.uint8),
            instance_id=np.full(len(xyz), 2, np.int32),
            label_code=np.full(len(xyz), 2, np.uint8),
            source_camera=np.zeros(len(xyz), np.uint8),
        )
    rotations = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1, 1))
    translations = np.asarray([[[0, 0, 0]], [[0, 1, 0]]], np.float32)
    metric = np.asarray([[0.8], [0.9]], np.float32)
    np.savez_compressed(
        root / "object_models.npz",
        schema=np.asarray("viki_object_models_v1"),
        frame_count=np.int32(2),
        object_ids=np.asarray([2], np.int32),
        object_labels=np.asarray(["manipulated_object"]),
        model_offsets=np.asarray([0, 3], np.int64),
        model_xyz_object=np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], np.float32),
        model_kind=np.asarray([1, 1, 2], np.uint8),
        rotation_world_object=rotations,
        translation_world=translations,
        track_confidence=metric,
        residual_median_m=metric * 0.001,
        residual_p95_m=metric * 0.002,
        coverage=metric,
        retained_fraction=metric,
        rotation_information=np.ones((2, 1, 3), np.float32),
        contact=np.asarray([[False], [True]]),
        candidate_frame_offsets=np.asarray([0, 3, 5], np.int64),
        candidate_point_indices=np.asarray([0, 2, 3, 1, 3], np.int32),
        candidate_initial_id=np.asarray([2, 2, 2, 2, 2], np.int32),
        candidate_assigned_id=np.asarray([2, 0, 2, 3, 0], np.int32),
        candidate_state=np.asarray([1, 0, 3, 2, 0], np.uint8),
        candidate_residual_m=np.zeros(5, np.float32),
    )
    return ep


def test_object_model_meta_is_json_safe_and_splits_core_shell(tmp_path, monkeypatch):
    ep = _episode_with_object_model(tmp_path, monkeypatch)
    from viki.server.routes.pipeline import object_model_meta

    payload = asyncio.run(object_model_meta(ep.id))

    assert payload["schema"] == "viki_object_model_viewer_v1"
    assert payload["n_frames"] == 2
    assert len(payload["objects"]) == 1
    obj = payload["objects"][0]
    assert obj["id"] == 2
    assert len(obj["core_xyz_object"]) == 2
    assert len(obj["shell_xyz_object"]) == 1
    assert obj["translation_world"][1] == [0.0, 1.0, 0.0]
    assert obj["contact"] == [False, True]
    json.dumps(payload, allow_nan=False)


def test_object_model_frame_binary_preserves_source_indices(tmp_path, monkeypatch):
    ep = _episode_with_object_model(tmp_path, monkeypatch)
    from viki.server.routes.pipeline import object_model_frame

    response = asyncio.run(object_model_frame(ep.id, 1))
    body = response.body
    n = int(np.frombuffer(body, dtype="<i4", count=1)[0])
    xyz_offset = 4
    initial_offset = xyz_offset + n * 12
    assigned_offset = initial_offset + n * 4
    state_offset = assigned_offset + n * 4

    assert n == 2
    assert np.frombuffer(body, dtype="<f4", count=n * 3, offset=xyz_offset).reshape(n, 3).tolist() == [
        [1.0, 1.0, 0.0], [3.0, 1.0, 0.0],
    ]
    assert np.frombuffer(body, dtype="<i4", count=n, offset=initial_offset).tolist() == [2, 2]
    assert np.frombuffer(body, dtype="<i4", count=n, offset=assigned_offset).tolist() == [3, 0]
    assert np.frombuffer(body, dtype=np.uint8, count=n, offset=state_offset).tolist() == [2, 0]


def test_object_model_route_rejects_profile_path(tmp_path, monkeypatch):
    ep = _episode_with_object_model(tmp_path, monkeypatch)
    from fastapi import HTTPException
    from viki.server.routes.pipeline import object_model_meta

    with pytest.raises(HTTPException) as exc:
        asyncio.run(object_model_meta(ep.id, profile="../sam2.1_hiera_small"))
    assert exc.value.status_code == 422
