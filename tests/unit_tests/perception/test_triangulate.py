"""
Stage 2 — multi-view joint triangulation (``viki.perception.triangulate``).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from viki.perception.triangulate import TriConfig, _Cam, triangulate_joint


def _cam_meta(pos, look_at=(0.0, 0.0, 0.6), f=650.0, w=1280, h=720, dist=None):
    pos = np.asarray(pos, float)
    z = np.asarray(look_at, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T_wc = np.eye(4)
    T_wc[:3, :3] = np.column_stack([x, y, z])   # camera axes in world
    T_wc[:3, 3] = pos
    return {
        "K": [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]],
        "dist": list(dist) if dist is not None else [0.0] * 5,
        "T_wc": T_wc.tolist(),
        "image_size": [w, h],
    }


def _project_distorted(meta, X):
    c = _Cam("x", meta)
    T_cw = np.linalg.inv(c.T_wc)
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    uv, _ = cv2.projectPoints(np.asarray(X, float).reshape(1, 3), rvec, T_cw[:3, 3], c.K, c.dist)
    return uv.reshape(2)


def _views(metas, X, noise_px=0.0, seed=0, bad=None):
    rng = np.random.default_rng(seed)
    out = []
    for cid, m in metas.items():
        uv = _project_distorted(m, X) + rng.normal(0, noise_px, 2)
        if bad and cid == bad:
            uv = uv + np.array([45.0, -30.0])
        out.append({"camera_id": cid, "uv": uv, "score": 0.9,
                    "depth_m": float("nan"), "depth_valid": False, "depth_spread_m": float("nan")})
    return out


THREE = {
    "k0": _cam_meta([0.45, -0.35, 0.05]),
    "k1": _cam_meta([-0.40, -0.30, 0.10]),
    "k2": _cam_meta([0.05, 0.55, 0.20]),
}
X_TRUE = np.array([0.04, -0.02, 0.62])


def test_triangulate_episode_end_to_end(tmp_path):
    from viki.contracts import HAND_LM_COUNT
    from viki.perception.triangulate import triangulate_episode

    raw = tmp_path / "raw"
    raw.mkdir()
    metas = THREE
    n_frames = 6
    rng = np.random.default_rng(0)
    rows = []
    for fi in range(n_frames):
        Xf = X_TRUE + np.array([0.03 * fi, 0.0, 0.0])          # slide along +x
        pts = X_TRUE[None] + (Xf - X_TRUE)[None] + rng.normal(0, 0.002, (HAND_LM_COUNT, 3))
        for cid, m in metas.items():
            c = _Cam(cid, m)
            uv = np.full((HAND_LM_COUNT, 2), np.nan, np.float32)
            for lm in range(HAND_LM_COUNT):
                uv[lm] = _project_distorted(m, pts[lm]) + rng.normal(0, 0.3, 2)
            rows.append(dict(
                camera_id=cid, frame_index=fi, host_timestamp_us=1000 * fi, uv=uv,
                lm_score=np.full(HAND_LM_COUNT, 0.9, np.float32), lm_score_per_pt=True,
                depth_m=np.full(HAND_LM_COUNT, np.nan, np.float32),
                depth_valid=np.zeros(HAND_LM_COUNT, bool),
                depth_spread_m=np.full(HAND_LM_COUNT, np.nan, np.float32),
            ))
    from viki.perception.observations import write_observations
    cams_meta = {cid: {"K": m["K"], "dist": m["dist"], "T_wc": m["T_wc"],
                       "image_size": m["image_size"], "calib_id": "test"} for cid, m in metas.items()}
    write_observations(raw / "observations.npz", rows, cams_meta, {"depth_radius_px": 8})

    summary = triangulate_episode(raw)
    assert summary["n_frames"] == n_frames
    assert summary["joints_solved"] >= 0.95 * summary["joints_total"]
    assert summary["reproj_px_median"] < 1.0
    assert summary["nviews_hist_0_1_2_3plus"][3] > 0   # most joints used all 3 views

    with np.load(raw / "joints3d.npz") as z:
        assert z["xyz"].shape == (n_frames, HAND_LM_COUNT, 3)
        assert np.isfinite(z["xyz"]).mean() > 0.9
        assert (z["quality"] > 0).mean() > 0.9


def test_recovers_a_known_point():
    cams = {c: _Cam(c, m) for c, m in THREE.items()}
    res = triangulate_joint(_views(THREE, X_TRUE, noise_px=0.3, seed=1), cams, lm=8, cfg=TriConfig())
    assert res is not None
    assert np.linalg.norm(res["xyz"] - X_TRUE) < 2e-3
    assert res["n_views"] == 3 and res["reproj_px"] < 1.0
    assert res["quality"] > 0.6


def test_one_outlier_view_is_rejected_not_blended():
    cams = {c: _Cam(c, m) for c, m in THREE.items()}
    v = _views(THREE, X_TRUE, noise_px=0.3, seed=2, bad="k2")
    res = triangulate_joint(v, cams, lm=8, cfg=TriConfig())
    assert res is not None
    assert np.linalg.norm(res["xyz"] - X_TRUE) < 3e-3   # k2 did not drag it
    assert res["n_views"] == 2


def test_near_collinear_rays_are_not_confident():
    metas = {
        "k0": _cam_meta([0.02, -0.40, 0.05]),
        "k1": _cam_meta([-0.02, -0.40, 0.05]),   # ~few deg apart, same side
    }
    cams = {c: _Cam(c, m) for c, m in metas.items()}
    res = triangulate_joint(_views(metas, X_TRUE, noise_px=0.3, seed=3), cams, lm=8, cfg=TriConfig())
    assert res is None or res["quality"] < 0.1


def test_distortion_round_trip():
    dist = [-0.12, 0.04, 0.001, -0.001, 0.0]
    metas = {c: _cam_meta(_Cam(c, m).T_wc[:3, 3], dist=dist) for c, m in THREE.items()}
    cams = {c: _Cam(c, m) for c, m in metas.items()}
    res = triangulate_joint(_views(metas, X_TRUE, noise_px=0.2, seed=4), cams, lm=8, cfg=TriConfig())
    assert res is not None
    assert np.linalg.norm(res["xyz"] - X_TRUE) < 2e-3   # undistort-once handles it


def test_depth_residual_nudges_toward_the_measurement():
    cams = {c: _Cam(c, m) for c, m in THREE.items()}
    cfg = TriConfig()
    cfg.depth_delta_m = 0.0
    v_geom = _views(THREE, X_TRUE, noise_px=0.0, seed=5)
    v_depth = [dict(vi) for vi in v_geom]
    # tell every camera the joint is 3 cm nearer than it really is
    for vi in v_depth:
        c = cams[vi["camera_id"]]
        _, z = c.project(X_TRUE)
        vi.update(depth_m=z - 0.03, depth_valid=True, depth_spread_m=0.0)

    base = triangulate_joint(v_geom, cams, lm=8, cfg=cfg)
    cfg.depth_lambda = 0.5
    pulled = triangulate_joint(v_depth, cams, lm=8, cfg=cfg)
    assert base is not None and pulled is not None
    off_base = np.linalg.norm(base["xyz"] - X_TRUE)
    off_pull = np.linalg.norm(pulled["xyz"] - X_TRUE)
    assert off_base < 1e-3                     # pure geometry nails it
    assert off_pull > 5 * off_base             # depth measurably pulled it
    assert off_pull < 0.03                     # but bounded, not a hard snap
