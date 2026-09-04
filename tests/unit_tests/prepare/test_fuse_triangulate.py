"""
Stage 3 — prepare consumes multi-view triangulation instead of averaging XYZ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from viki.contracts import HAND_LM_COUNT, LM
from viki.episode import new_episode
from viki.prepare.fuse import snap_to_grid


def test_snap_to_grid_places_and_gaps():
    arr = np.arange(9, dtype=np.float32).reshape(3, 3)   # rows at 0/100/200
    src = np.array([0, 100, 200])
    out = snap_to_grid(arr, src, np.array([0, 100, 200, 300]), tol_us=40)
    np.testing.assert_allclose(out[:3], arr)
    assert np.isnan(out[3]).all()                        # 300 has no source in tol

    out2 = snap_to_grid(arr, src, np.array([10, 90, 250]), tol_us=40)
    np.testing.assert_allclose(out2[0], arr[0])          # 10 -> 0
    np.testing.assert_allclose(out2[1], arr[1])          # 90 -> 100
    assert np.isnan(out2[2]).all()                       # 250 is 50 from 200 > tol


def _synthetic_rec(ep, n, grid_us):
    pts = np.zeros((n, HAND_LM_COUNT, 3), np.float32)
    for t in range(n):
        for i in range(HAND_LM_COUNT):
            pts[t, i] = [0.001 * t, 0.05, 0.5]
    np.savez_compressed(
        ep.rec_npz, device_ids=np.array(["cam0"] * n),
        timestamps=grid_us.astype(np.int64), points=pts,
        landmark_ids=np.arange(HAND_LM_COUNT, dtype=np.int32),
        confidence=np.ones((n, HAND_LM_COUNT), np.float32),
    )


def test_prepare_fuses_from_triangulation(tmp_path, monkeypatch):
    from viki import config
    from viki.prepare.run import prepare_episode

    monkeypatch.setattr(config, "PERCEPTION_FUSE_MODE", "triangulate", raising=False)
    monkeypatch.setattr(config, "PERCEPTION_HAND_FIT", False, raising=False)

    ep = new_episode(tmp_path)
    n = 30
    grid = (np.arange(n) * 33_000).astype(np.int64)
    _synthetic_rec(ep, n, grid)
    ep.raw_dir.mkdir(parents=True, exist_ok=True)
    (ep.raw_dir / "timestamps.json").write_text(
        json.dumps([{"sync_us": int(t), "offsets_us": {"cam0": 0}} for t in grid])
    )

    # triangulated joints: a distinctive ramp on X so we can tell them apart from
    # the rec.npz points; PINKY_TIP is a permanent gap (quality 0, NaN xyz).
    xyz = np.zeros((n, HAND_LM_COUNT, 3), np.float32)
    quality = np.full((n, HAND_LM_COUNT), 0.8, np.float32)
    for t in range(n):
        xyz[t, :, 0] = 0.10 + 0.01 * t          # != rec.npz's 0.001*t
        xyz[t, :, 1] = 0.20
        xyz[t, :, 2] = 0.60
    xyz[:, int(LM.PINKY_TIP)] = np.nan
    quality[:, int(LM.PINKY_TIP)] = 0.0
    quality[:, int(LM.WRIST)] = 0.3            # low-confidence but present
    np.savez(
        ep.raw_dir / "joints3d.npz", schema=np.int32(1), timestamps=grid,
        xyz=xyz, quality=quality,
        n_views=np.full((n, HAND_LM_COUNT), 2, np.int8),
        ray_deg=np.full((n, HAND_LM_COUNT), 25.0, np.float32),
        reproj_px=np.full((n, HAND_LM_COUNT), 0.5, np.float32),
        cameras=np.array(["cam0", "cam1"]),
    )

    prepare_episode(ep)
    with np.load(ep.cln_npz) as d:
        sp = d["smoothed_points"]              # (T, 21, 3)
        lc = d["landmark_confidence"]          # (T, 21)
        ids = d["landmark_ids"]
        assert sp.shape[0] == n
        mid = n // 2
        # came from triangulation (X ~ 0.10 + 0.01*t), not rec.npz (X ~ 0.001*t)
        assert abs(sp[mid, int(LM.MIDDLE_MCP), 0] - (0.10 + 0.01 * mid)) < 0.02
        # per-joint confidence tracks triangulation quality
        col = {int(v): k for k, v in enumerate(ids)}
        assert lc[mid, col[int(LM.MIDDLE_MCP)]] > lc[mid, col[int(LM.WRIST)]]
        assert lc[mid, col[int(LM.PINKY_TIP)]] == 0.0     # gap, not a fake weight
        assert d["perception_fuse_mode"].item() == "triangulate"
        assert d["checkpoint_stage"].item() == "smoothed"
        assert "observed_points" in d.files and "filled_points" in d.files
        assert "observed_mask" in d.files and "interpolated_mask" in d.files

    from viki.prepare.checkpoints import run_name

    checkpoints = ep.intermediates_dir / "prepare" / run_name(
        "triangulate", 0, 7, 2
    )
    expected = {
        "00_per_camera_observed.npz", "05_per_camera_filled.npz",
        "10_fused_observed.npz", "20_fused_filled.npz", "30_smoothed.npz",
        "manifest.json", "comparison.json",
    }
    assert expected <= {p.name for p in checkpoints.iterdir()}
    with np.load(checkpoints / "10_fused_observed.npz") as observed:
        assert observed["checkpoint_stage"].item() == "observed"
        assert np.isnan(observed["smoothed_points"][:, int(LM.PINKY_TIP)]).all()
    with np.load(checkpoints / "20_fused_filled.npz") as filled:
        assert filled["checkpoint_stage"].item() == "filled"

    # Explicit A/B generation is non-destructive and keeps the active CLN as a
    # control row in the cumulative episode comparison.
    active_before = ep.cln_npz.read_bytes()
    from viki.prepare.run import generate_stage_checkpoints

    outputs = generate_stage_checkpoints(
        ep, fusion_modes=("triangulate",), interp_max_gap=3
    )
    assert Path(outputs["triangulate"]).name == "cln.npz"
    assert ep.cln_npz.read_bytes() == active_before
    comparison = json.loads(
        (ep.intermediates_dir / "prepare" / "comparison.json").read_text()
    )
    assert any(Path(row["file"]) == ep.cln_npz for row in comparison["variants"])
