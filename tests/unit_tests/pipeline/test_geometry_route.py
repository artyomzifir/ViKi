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
    assert g["pose_source"] == "landmarks"
    assert g["layer_sources"] == {
        "fused": "smoothed_points",
        "hand_fit": None,
    }
    assert g["cameras"]["cam0"]["pos"] == pytest.approx([0.0, 0.0, -1.5], abs=1e-4)
    assert "raw_points" in g and "cam0" in g["raw_points"]


def test_frame_geometry_is_small_and_excludes_episode_arrays(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(episodes), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    ep = new_episode(episodes)
    _synthetic(ep, T=300)

    from viki.server.routes.pipeline import geometry

    g = asyncio.run(geometry(ep.id, frame=17))
    assert g["frame"] == 17
    assert len(g["fused_skeleton"]) == 21
    assert "cam0" in g["per_camera"]
    assert {"wrist_traj", "palm_rot", "valid", "cameras"}.isdisjoint(g)
    assert len(json.dumps(g)) < 20_000


def test_geometry_artifacts_are_decompressed_once(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(episodes), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    ep = new_episode(episodes)
    _synthetic(ep)
    with np.load(ep.cln_npz) as source:
        arrays = {key: source[key] for key in source.files}
    # More picker variants than the full-array cache can hold reproduces the
    # real episode that exposed the regression (27 preparation checkpoints).
    for index in range(12):
        np.savez_compressed(ep.root / f"cln_variant_{index}.npz", **arrays)

    from viki.server.routes import pipeline

    pipeline._load_npz_for_viewer.cache_clear()
    pipeline._inspect_viewer_variant_file.cache_clear()
    real_load = np.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(str(args[0]))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(pipeline.np, "load", counting_load)
    asyncio.run(pipeline.geometry_variants(ep.id))
    calls_after_listing = len(calls)
    asyncio.run(pipeline.geometry(ep.id))
    asyncio.run(pipeline.geometry(ep.id, frame=0))
    asyncio.run(pipeline.geometry(ep.id, frame=1))

    # Listing may inspect all variants, but playback decompresses only the
    # selected cln.npz and rec.npz once.  It must not revisit every variant on
    # every frame and evict these two arrays again.
    assert len(calls) - calls_after_listing == 2


def test_geometry_lists_and_loads_checkpoint_variant(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(episodes), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    ep = new_episode(episodes)
    _synthetic(ep)
    run = ep.intermediates_dir / "prepare" / "triangulate__gap-all__sg-7-2"
    run.mkdir(parents=True)
    variant_path = run / "10_fused_observed.npz"
    with np.load(ep.cln_npz) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["perception_fuse_mode"] = np.asarray("triangulate")
    arrays["checkpoint_stage"] = np.asarray("observed")
    arrays["positions"] = arrays["positions"].copy()
    arrays["rotations"] = arrays["rotations"].copy()
    arrays["positions"][3] = np.nan
    arrays["rotations"][3] = np.nan
    np.savez_compressed(variant_path, **arrays)

    from viki.server.routes.pipeline import geometry, geometry_variants

    listing = asyncio.run(geometry_variants(ep.id))["variants"]
    variant = next(v for v in listing if v["stage"] == "observed")
    assert variant["fusion_mode"] == "triangulate"
    assert "path" not in variant

    payload = asyncio.run(geometry(ep.id, variant=variant["id"]))
    assert payload["variant"] == variant["id"]
    assert payload["checkpoint_stage"] == "observed"
    assert payload["fusion_mode"] == "triangulate"
    assert payload["wrist_traj"][3] == [None, None, None]
    assert payload["palm_rot"][3] == [None] * 9
    json.dumps(payload, allow_nan=False)


def test_geometry_lists_protected_baseline_and_articulated_variant(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(episodes), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    ep = new_episode(episodes)
    _synthetic(ep)
    with np.load(ep.cln_npz) as source:
        arrays = {key: source[key] for key in source.files}

    baseline = ep.intermediates_dir / "baselines" / "clean-v1" / "cln.npz"
    baseline.parent.mkdir(parents=True)
    np.savez_compressed(baseline, **arrays)
    geometry_path = (
        ep.intermediates_dir / "geometry" / "articulated-v1" / "50_optimized.npz"
    )
    geometry_path.parent.mkdir(parents=True)
    arrays["checkpoint_stage"] = np.asarray("geometry_optimized")
    arrays["pose_source"] = np.asarray("articulated_landmarks")
    np.savez_compressed(geometry_path, **arrays)

    from viki.server.routes import pipeline

    pipeline._inspect_viewer_variant_file.cache_clear()
    listing = asyncio.run(pipeline.geometry_variants(ep.id))["variants"]
    ids = {row["id"] for row in listing}
    assert "baseline:clean-v1" in ids
    assert "geometry:articulated-v1/50_optimized" in ids

    payload = asyncio.run(pipeline.geometry(
        ep.id, variant="geometry:articulated-v1/50_optimized",
    ))
    assert payload["checkpoint_stage"] == "geometry_optimized"
    assert payload["pose_source"] == "articulated_landmarks"


def test_geometry_404_for_missing_episode(tmp_path, monkeypatch):
    monkeypatch.setattr("viki.config.EPISODES_DIR", str(tmp_path / "episodes"), raising=False)
    monkeypatch.setattr("viki.config.DATASETS_DIR", str(tmp_path / "datasets"), raising=False)
    from fastapi import HTTPException

    from viki.server.routes.pipeline import geometry

    with pytest.raises(HTTPException) as exc:
        asyncio.run(geometry("nope", include_raw=0))
    assert exc.value.status_code == 404
