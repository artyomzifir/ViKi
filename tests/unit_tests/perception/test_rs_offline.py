"""viki.perception.rs_offline.RealSenseCalibration — colour↔depth reprojection
replayed from the stored JSON, no device."""

import json

import numpy as np

from viki.perception.rs_offline import RealSenseCalibration


def _pinhole(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, w=640, h=480):
    return {"fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy,
            "width": w, "height": h, "model": "none", "coeffs": [0, 0, 0, 0, 0]}


def _cal():
    # depth→colour: 15 mm baseline along -x (colour sensor sits to the right),
    # a hair of pitch so rotation isn't identity.
    ang = np.deg2rad(1.5)
    R = np.array([[1, 0, 0],
                  [0, np.cos(ang), -np.sin(ang)],
                  [0, np.sin(ang), np.cos(ang)]], dtype=np.float64)
    t = np.array([-0.015, 0.0, 0.0])
    return RealSenseCalibration(_pinhole(), _pinhole(fx=615, fy=615), R, t), R, t


def test_project_color_to_depth_round_trips():
    cal, R, t = _cal()
    depth_intr, color_intr = cal._depth, cal._color
    rng = np.random.default_rng(0)
    worst_exact = worst_near = worst_far = 0.0
    for _ in range(300):
        ud, vd = rng.uniform(40, 600), rng.uniform(40, 440)
        z = rng.uniform(0.5, 1.5)
        # depth pixel → 3D depth frame → colour frame → colour pixel
        xd = (ud - depth_intr["ppx"]) / depth_intr["fx"] * z
        yd = (vd - depth_intr["ppy"]) / depth_intr["fy"] * z
        p_dep = np.array([xd, yd, z])
        p_col = R @ p_dep + t
        uc = p_col[0] / p_col[2] * color_intr["fx"] + color_intr["ppx"]
        vc = p_col[1] / p_col[2] * color_intr["fy"] + color_intr["ppy"]

        # Exact when the hint is the true colour-frame range — this is the
        # correctness gate on the deproject → extrinsic → project maths.
        exact = cal.project_color_to_depth(uc, vc, float(p_col[2]))
        worst_exact = max(worst_exact, abs(exact[0] - ud), abs(exact[1] - vd))
        # With a fixed hint (lift_to_3d always passes 1.0) the depth-pixel error
        # grows with |z - hint|/z — same range-sensitivity as the Kinect SDK's
        # 2d_to_2d, absorbed downstream by the radius-15 depth ROI median. Small
        # when the hint is close to the true range, a few px at the extremes.
        e = cal.project_color_to_depth(uc, vc, 1.0)
        err = max(abs(e[0] - ud), abs(e[1] - vd))
        if abs(z - 1.0) <= 0.15:
            worst_near = max(worst_near, err)
        else:
            worst_far = max(worst_far, err)
    assert worst_exact < 1e-6, worst_exact
    assert worst_near < 2.5, worst_near
    assert worst_far < 12.0, worst_far


def test_depth3d_to_color3d_is_the_extrinsic():
    cal, R, t = _cal()
    p = np.array([0.05, -0.02, 0.8])
    np.testing.assert_allclose(cal.depth3d_to_color3d(p), R @ p + t, rtol=0, atol=1e-12)


def test_color_deproject_maps_match_per_pixel_transform():
    cal, R, t = _cal()
    dh, dw = 60, 80
    A, B = cal.color_deproject_maps(dh, dw)
    assert A.shape == (dh, dw, 3) and B.shape == (dh, dw, 3)
    di = cal._depth
    for (u, v) in [(0, 0), (40, 30), (79, 59), (10, 55)]:
        for z_mm in (400.0, 1200.0):
            ray = np.array([(u - di["ppx"]) / di["fx"], (v - di["ppy"]) / di["fy"], 1.0])
            want = R @ (ray * z_mm / 1000.0) * 1000.0 + t * 1000.0  # mm, colour frame
            got = z_mm * A[v, u] + B[v, u]
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-6)


def test_from_episode_reads_the_recorder_json(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _, R, t = _cal()
    payload = {
        "color": _pinhole(),
        "depth": _pinhole(fx=615, fy=615),
        "depth_to_color": {"rotation": R.T.reshape(-1).tolist(),  # col-major, as the backend writes
                           "translation": t.tolist()},
        "aligned": False,
    }
    (raw / "dev0_rs_calib.json").write_text(json.dumps(payload))
    cal = RealSenseCalibration.from_episode(raw, "dev0", {"cameras": {"dev0": {"rs_calib": "dev0_rs_calib.json"}}})
    assert cal is not None
    np.testing.assert_allclose(cal._R, R, atol=1e-12)
    np.testing.assert_allclose(cal._t, t, atol=1e-12)
    # absent file → None (caller falls back to identity)
    assert RealSenseCalibration.from_episode(raw, "missing", {}) is None
