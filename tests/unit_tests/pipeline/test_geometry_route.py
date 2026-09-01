"""viki.server.routes.pipeline.geometry — the JSON shape the 3-D viewer needs.

Calls the handler coroutine directly (the test image has no httpx for TestClient).
"""

import asyncio
import json

import numpy as np
import pytest

from viki.episode import new_episode


def _synthetic(ep, T=8):
    np.savez_compressed(
        ep.cln_npz,
        positions=np.linspace([0, 0, 0.5], [0.1, 0, 0.5], T).astype(np.float32),
        rotations=np.tile(np.eye(3), (T, 1, 1)).astype(np.float32),
        valid=np.ones(T, bool),
        timestamps=(np.arange(T) * 33_000).astype(np.int64),
        omega=np.ones(T, np.float32),
        gripper=np.zeros(T, bool),
        raw_points=np.zeros((T, 21, 3), np.float32),
        smoothed_points=np.zeros((T, 21, 3), np.float32),
        landmark_ids=np.arange(21, dtype=np.int32),
        coordinate_frame="world",
    )
    (ep.raw_dir / "extrinsics.json").write_text(
        json.dumps({"cam0": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 1.5]}})
    )
    np.savez_compressed(
        ep.rec_npz,
        device_ids=np.array(["cam0"] * T),
        timestamps=(np.arange(T) * 33_000).astype(np.int64),
        points=np.zeros((T, 21, 3), np.float32),
        landmark_ids=np.arange(21, dtype=np.int32),
        confidence=np.ones((T, 21), np.float32),
    )


def test_geometry_shape(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(episodes), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    ep = new_episode(episodes)
    _synthetic(ep)

    from viki.server.routes.pipeline import geometry

    g = asyncio.run(geometry(ep.id, include_raw=1))
    assert g["n_frames"] == 8
    assert len(g["wrist_traj"]) == 8 and len(g["wrist_traj"][0]) == 3
    assert len(g["palm_rot"][0]) == 9
    assert g["cameras"]["cam0"]["pos"] == pytest.approx([0.0, 0.0, -1.5], abs=1e-4)
    assert "raw_points" in g and "cam0" in g["raw_points"]


def test_geometry_404_for_missing_episode(tmp_path, monkeypatch):
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(tmp_path / "episodes"), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    from fastapi import HTTPException

    from viki.server.routes.pipeline import geometry

    with pytest.raises(HTTPException) as exc:
        asyncio.run(geometry("nope", include_raw=0))
    assert exc.value.status_code == 404
