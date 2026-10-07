"""Robustness of the SAM prompt proposer — the parts that need no hardware.

Both behaviours here were found by running the stage over ten real takes: the
clustering grid silently could not resolve a 40 mm cube, and frame zero is not
reliably a frame where the object stands apart from the hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from viki.contracts import Episode
from viki.perception import auto_prompts as AP


def _episode(tmp_path, frames: int, cameras=("kinect_0", "cam_1")) -> Episode:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for camera in cameras:
        (raw / f"{camera}.mp4").write_bytes(b"")
        depth = raw / f"{camera}_depth"
        depth.mkdir()
        for index in range(frames):
            (depth / f"{index:06d}.npy").write_bytes(b"")
    return Episode(root=tmp_path)


# ── clustering grid vs object scale ────────────────────────────────────

def test_grid_tightens_until_the_smallest_object_resolves():
    cfg = AP.AutoPromptConfig(voxel_m=0.006, min_object_extent_m=0.03)
    assert AP._effective_config(cfg).voxel_m == pytest.approx(0.00375)


def test_a_coarse_grid_is_left_alone_for_large_objects():
    cfg = AP.AutoPromptConfig(voxel_m=0.006, min_object_extent_m=0.20)
    assert AP._effective_config(cfg) is cfg


def test_an_explicitly_finer_grid_is_never_coarsened():
    cfg = AP.AutoPromptConfig(voxel_m=0.002, min_object_extent_m=0.03)
    assert AP._effective_config(cfg).voxel_m == pytest.approx(0.002)


def test_min_extent_above_max_extent_is_rejected():
    with pytest.raises(ValueError, match="min_object_extent_m"):
        AP._validate_config(AP.AutoPromptConfig(min_object_extent_m=0.4,
                                                max_object_extent_m=0.25))


# ── seed-frame search ──────────────────────────────────────────────────

def test_frame_plan_tries_the_requested_frame_first_then_spreads(tmp_path):
    ep = _episode(tmp_path, frames=100)
    plan = AP._frame_plan(ep, AP.AutoPromptConfig(frame=7, frame_attempts=5))
    assert plan[0] == 7
    assert len(plan) == len(set(plan))
    assert all(0 <= f < 100 for f in plan)
    assert max(plan) > 50  # actually reaches the far end of the take


def test_frame_plan_honours_a_single_attempt(tmp_path):
    ep = _episode(tmp_path, frames=100)
    assert AP._frame_plan(ep, AP.AutoPromptConfig(frame=3, frame_attempts=1)) == [3]


def test_frame_plan_clamps_to_the_shortest_camera(tmp_path):
    ep = _episode(tmp_path, frames=4)
    plan = AP._frame_plan(ep, AP.AutoPromptConfig(frame=0, frame_attempts=6))
    assert plan and all(0 <= f < 4 for f in plan)


def test_a_later_frame_is_used_when_the_first_has_no_candidate(tmp_path, monkeypatch):
    ep = _episode(tmp_path, frames=60)
    seen = []

    def fake_build(_ep, output, cfg):
        seen.append(cfg.frame)
        if cfg.frame == 0:
            raise ValueError("found 0 compact multi-view object candidates, need 1")
        return tmp_path / f"prompts_{cfg.frame}.json"

    monkeypatch.setattr(AP, "_build_manifest", fake_build)
    out = AP.generate_auto_prompts(ep, config=AP.AutoPromptConfig(frame_attempts=4))
    assert seen[0] == 0 and len(seen) == 2
    assert out.name == f"prompts_{seen[1]}.json"


def test_giving_up_reports_every_attempt(tmp_path, monkeypatch):
    ep = _episode(tmp_path, frames=60)

    def always_fail(_ep, _output, cfg):
        raise ValueError(f"nothing at {cfg.frame}")

    monkeypatch.setattr(AP, "_build_manifest", always_fail)
    with pytest.raises(ValueError) as excinfo:
        AP.generate_auto_prompts(ep, config=AP.AutoPromptConfig(frame_attempts=3))
    message = str(excinfo.value)
    assert "3 attempt(s)" in message
    assert message.count("nothing at") == 3


def test_a_missing_frame_is_a_retry_not_a_crash(tmp_path, monkeypatch):
    ep = _episode(tmp_path, frames=60)

    def fake_build(_ep, _output, cfg):
        if cfg.frame == 0:
            raise FileNotFoundError("raw/kinect_0_depth/000000.npy")
        return tmp_path / "prompts.json"

    monkeypatch.setattr(AP, "_build_manifest", fake_build)
    assert AP.generate_auto_prompts(ep, config=AP.AutoPromptConfig()).name == "prompts.json"


# ── colour split inside an oversized component ─────────────────────────

def _slab_with_a_cube(rng):
    """A 0.4 m tabletop slab that survived background subtraction, with a small
    saturated cube resting on it — the two are one component under any
    connectivity radius, which is the take 09-58-19 failure."""
    slab = np.c_[rng.uniform(-0.2, 0.2, 4000), rng.uniform(-0.05, 0.05, 4000),
                 np.full(4000, 0.8)]
    cube = np.c_[rng.uniform(0.00, 0.04, 400), rng.uniform(0.00, 0.04, 400),
                 rng.uniform(0.80, 0.84, 400)]
    xyz = np.vstack([slab, cube])
    rgb = np.vstack([
        np.tile([200, 198, 196], (len(slab), 1)),   # neutral tabletop
        np.tile([230, 200, 20], (len(cube), 1)),    # yellow cube
    ]).astype(np.uint8)
    camera = np.arange(len(xyz), dtype=np.int32) % 2
    return xyz, rgb, camera, len(cube)


def test_a_cube_welded_to_a_tabletop_slab_is_recovered_by_colour():
    rng = np.random.default_rng(0)
    xyz, rgb, camera, n_cube = _slab_with_a_cube(rng)
    cfg = AP.AutoPromptConfig(object_count=1, min_component_points=50,
                              component_radius_m=0.018, max_object_extent_m=0.15)

    # the slab and cube really are one component on geometry alone
    whole = AP._describe_components(xyz, rgb, camera, cfg)
    assert len(whole) == 1 and whole[0].dimensions.max() > 0.3

    operator, objects, _components, rescued = AP._select_prompt_components(
        xyz, rgb, camera, 2, cfg)
    assert len(objects) == 1
    assert id(objects[0]) in rescued
    assert objects[0].point_count == n_cube
    assert objects[0].dimensions.max() < 0.06
    assert operator.point_count > objects[0].point_count


def test_the_colour_split_only_runs_when_the_primary_pass_is_short():
    """It is a fallback, not a second opinion: a scene that separates on
    geometry must not start reporting colour sub-blobs."""
    rng = np.random.default_rng(1)
    xyz, rgb, camera, _ = _slab_with_a_cube(rng)
    far = np.c_[rng.uniform(0.9, 0.94, 400), rng.uniform(0.0, 0.04, 400),
                rng.uniform(0.80, 0.84, 400)]
    xyz = np.vstack([xyz, far])
    rgb = np.vstack([rgb, np.tile([20, 60, 220], (len(far), 1)).astype(np.uint8)])
    camera = np.arange(len(xyz), dtype=np.int32) % 2
    cfg = AP.AutoPromptConfig(object_count=1, min_component_points=50,
                              component_radius_m=0.018, max_object_extent_m=0.15)

    _operator, objects, _components, rescued = AP._select_prompt_components(
        xyz, rgb, camera, 2, cfg)
    assert len(objects) == 1 and not rescued  # the free blue cube was enough
