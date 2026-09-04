"""
Stage 1 — raw 2-D observation persistence (``viki.perception.observations``).
"""

from __future__ import annotations

import numpy as np
import pytest

from viki.contracts import HAND_LM_COUNT, HandDetection, LM
from viki.perception import observations as obs


class _Identity:
    def project_color_to_depth(self, u, v, z):
        return (u, v)


def _depth_img(h=64, w=80, patch_val=1.2):
    d = np.full((h, w), np.nan, np.float32)
    d[20:40, 30:50] = patch_val
    return d


def test_sample_depth_reads_the_patch():
    d = _depth_img(patch_val=1.2)
    z, valid, spread = obs.sample_depth(np.array([40.0, 30.0]), d, _Identity(), radius=4)
    assert valid and abs(z - 1.2) < 1e-6 and spread < 1e-6


def test_sample_depth_off_image_and_empty():
    d = _depth_img()
    assert obs.sample_depth(np.array([999.0, 5.0]), d, _Identity(), 4)[1] is False
    assert obs.sample_depth(np.array([5.0, 5.0]), d, _Identity(), 3)[1] is False   # NaN region
    z, v, s = obs.sample_depth(np.array([np.nan, 1.0]), d, _Identity(), 4)
    assert v is False and np.isnan(z)


def _det(with_scores: bool):
    pts = {LM(i): np.array([10.0 + i, 20.0 + i], np.float32) for i in range(HAND_LM_COUNT)}
    return HandDetection(
        points=pts, lm_z_rel=np.zeros(HAND_LM_COUNT, np.float32),
        confidence=0.9, device_id="k0", timestamp_us=123,
        lm_score=(np.linspace(0.5, 1.0, HAND_LM_COUNT).astype(np.float32) if with_scores else None),
    )


def test_collect_row_per_point_vs_broadcast_score():
    d = _depth_img()
    r = obs.collect_row(camera_id="k0", frame_index=7, host_timestamp_us=123,
                        detection=_det(True), depth_m=d, projector=_Identity(), depth_radius=4)
    assert r["lm_score_per_pt"] is True
    assert r["uv"].shape == (HAND_LM_COUNT, 2) and r["depth_m"].shape == (HAND_LM_COUNT,)
    assert r["frame_index"] == 7

    r2 = obs.collect_row(camera_id="k0", frame_index=0, host_timestamp_us=1,
                         detection=_det(False), depth_m=d, projector=_Identity(), depth_radius=4)
    assert r2["lm_score_per_pt"] is False
    assert np.allclose(r2["lm_score"], 0.9)


def test_write_read_round_trip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    d = _depth_img()
    rows = [
        obs.collect_row(camera_id="k0", frame_index=i, host_timestamp_us=1000 * i,
                        detection=_det(True), depth_m=d, projector=_Identity(), depth_radius=4)
        for i in range(3)
    ]
    cams = {"k0": {"K": np.eye(3).tolist(), "dist": [0] * 5,
                   "T_wc": np.eye(4).tolist(), "image_size": [80, 64], "calib_id": "abc123"}}
    obs.write_observations(raw / "observations.npz", rows, cams, {"depth_radius_px": 4})

    got = obs.read_observations(raw)
    assert got["uv"].shape == (3, HAND_LM_COUNT, 2)
    assert list(got["camera_id"]) == ["k0", "k0", "k0"]
    assert got["frame_index"].tolist() == [0, 1, 2]
    assert got["cameras"]["k0"]["calib_id"] == "abc123"
    assert got["sampler"]["depth_radius_px"] == 4


def test_write_empty_is_valid(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    obs.write_observations(raw / "observations.npz", [], {}, {})
    got = obs.read_observations(raw)
    assert got is not None and len(got["camera_id"]) == 0


def test_calib_id_changes_with_calibration(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "extrinsics.json").write_text('{"k0": {"rvec": [0,0,0], "tvec": [0,0,1]}}')
    (raw / "intrinsics.json").write_text('{"k0": {}}')
    a = obs.calib_id(raw)
    (raw / "extrinsics.json").write_text('{"k0": {"rvec": [0,0,0], "tvec": [0,0,2]}}')
    assert obs.calib_id(raw) != a and len(a) == 16
