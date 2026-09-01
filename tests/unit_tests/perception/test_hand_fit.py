"""Capsule hand → point-cloud fitting: geometry, residual assembly, convergence.

The convergence test builds a capsule hand at a known pose, samples a noisy
cloud on its capsule surfaces, warm-starts from a perturbed pose, and checks the
fit recovers the wrist pose to a few mm / a few degrees.
"""

import numpy as np
import pytest

from viki.contracts import LM
from viki.perception import hand_fit as hf


# ── pure-geometry (no pinocchio) ─────────────────────────────────────────


def test_point_segment_distance_known():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    pts = np.array([
        [0.5, 0.0, 0.0],    # on the segment
        [0.5, 0.3, 0.0],    # 0.3 above the middle
        [-1.0, 0.0, 0.0],   # 1.0 before a → clamps to a
        [2.0, 0.0, 0.0],    # 1.0 past b  → clamps to b
        [0.0, 0.0, 0.4],    # 0.4 off the a end
    ])
    d = hf.point_segment_distance(pts, a, b)
    assert np.allclose(d, [0.0, 0.3, 1.0, 1.0, 0.4], atol=1e-9)


def test_point_capsule_signed_distance():
    a, b, r = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1
    pts = np.array([[0.5, 0.1, 0.0], [0.5, 0.05, 0.0], [0.5, 0.0, 0.0]])
    d = hf.point_capsule_signed_distance(pts, a, b, r)
    assert np.allclose(d, [0.0, -0.05, -0.1], atol=1e-9)   # on / inside / centre


def test_nearest_capsule_picks_closest_surface():
    endpoints = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # capsule 0 along x
        [[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]],   # capsule 1, parallel, +0.5 in y
    ])
    radii = np.array([0.05, 0.05])
    pts = np.array([[0.5, 0.02, 0.0], [0.5, 0.48, 0.0]])
    d, idx = hf.nearest_capsule(pts, endpoints, radii)
    assert idx.tolist() == [0, 1]
    assert np.allclose(d, [0.02 - 0.05, 0.02 - 0.05], atol=1e-9)


# ── model-dependent ─────────────────────────────────────────────────────

pin = pytest.importorskip("pinocchio")
from viki.perception import hand_model as hm  # noqa: E402


def _open_hand_frame(R=None, w=None):
    """A synthetic open right hand: wrist at ``w``, fingers straight along +x."""
    R = np.eye(3) if R is None else np.asarray(R, float)
    w = np.zeros(3) if w is None else np.asarray(w, float)
    mcp_y = {"thumb": 0.045, "index": 0.03, "middle": 0.0, "ring": -0.025, "pinky": -0.045}
    seglen = {"thumb": [0.038, 0.030, 0.024]}
    pts = {LM.WRIST: w.copy()}
    for f, (a, b, c, d) in hm.FINGERS.items():
        L = seglen.get(f, [0.040, 0.026, 0.020])
        x0 = 0.02 if f == "thumb" else 0.06
        base = np.array([x0, mcp_y[f], 0.0])
        chain = [base, base + [L[0], 0, 0],
                 base + [L[0] + L[1], 0, 0], base + [L[0] + L[1] + L[2], 0, 0]]
        for lm, p in zip((a, b, c, d), chain):
            pts[lm] = R @ np.asarray(p, float) + w
    return pts


def _sample_cloud(hand, q, per_capsule=110, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    ep = hm.fk_capsule_endpoints(hand, q)
    radii = hm.capsule_radii(hand)
    out = []
    for (a, b), r in zip(ep, radii):
        axis = b - a
        n = np.linalg.norm(axis)
        u = axis / n if n > 1e-9 else np.array([1.0, 0, 0])
        for _ in range(per_capsule):
            base = a + rng.random() * axis
            perp = rng.standard_normal(3)
            perp -= perp.dot(u) * u
            perp /= np.linalg.norm(perp) + 1e-9
            out.append(base + perp * r + rng.standard_normal(3) * noise)
    return np.asarray(out)


def _wrist_errs(hand, q_fit, R_g, w_g):
    p, R = hf.wrist_pose(hand, q_fit)
    pos_mm = float(np.linalg.norm(p - w_g)) * 1e3
    rot_deg = float(np.rad2deg(np.linalg.norm(pin.log3(R.T @ R_g))))
    return pos_mm, rot_deg


def test_calibrate_and_build():
    params = hm.calibrate_from_frames([_open_hand_frame(), _open_hand_frame()])
    hand = hm.build(params)
    assert hand.nv == 26 and hand.nq == 27
    assert len(hand.capsules) == 20
    ep = hm.fk_capsule_endpoints(hand, pin.neutral(hand.model))
    assert ep.shape == (20, 2, 3) and np.isfinite(ep).all()


def test_assemble_residuals_shapes():
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    q = pin.neutral(hand.model)
    cloud = _sample_cloud(hand, q, per_capsule=20)
    w = np.ones(len(cloud))
    fc = hf.FitConfig()
    r = hf.assemble_residuals(q, hand, cloud, w, fc, q_prev=q, q_pred=q)
    # data (N) + vel (nv) + acc (nv) + limits (2*(nq-7)) + posture (nq-7)
    expect = len(cloud) + hand.nv + hand.nv + 2 * (hand.nq - 7) + (hand.nq - 7)
    assert r.shape == (expect,)
    assert np.isfinite(r).all()
    # at the sampled pose the data residuals are ~noise-sized
    assert np.median(np.abs(r[: len(cloud)])) < 0.01


def test_fit_recovers_wrist_pose():
    params = hm.calibrate_from_frames([_open_hand_frame(), _open_hand_frame()])
    hand = hm.build(params)

    R_g = pin.exp3(np.array([0.20, -0.30, 0.40]))
    w_g = np.array([0.15, -0.05, 0.90])
    q_true = pin.neutral(hand.model).copy()
    q_true[:7] = pin.SE3ToXYZQUAT(pin.SE3(R_g, w_g))
    cloud = _sample_cloud(hand, q_true, seed=1)

    q0 = q_true.copy()
    q0[:7] = pin.SE3ToXYZQUAT(pin.SE3(
        R_g @ pin.exp3(np.array([0.15, 0.10, -0.12])), w_g + np.array([0.02, -0.03, 0.025])
    ))
    p0_mm, r0_deg = _wrist_errs(hand, q0, R_g, w_g)

    q_fit, info = hf.fit_frame(hand, cloud, None, q0, hf.FitConfig())
    pf_mm, rf_deg = _wrist_errs(hand, q_fit, R_g, w_g)

    assert not info["skipped"] and info["accepted"]
    assert pf_mm < 5.0 and rf_deg < 5.0
    assert pf_mm < p0_mm and rf_deg < r0_deg     # strictly better than the warm start


def test_fit_frame_skips_sparse_cloud():
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    q0 = pin.neutral(hand.model)
    q, info = hf.fit_frame(hand, np.zeros((5, 3)), None, q0, hf.FitConfig())
    assert info["skipped"] and np.allclose(q, q0)


def test_refine_cln_noop_without_depth(tmp_path):
    """No raw depth / k4a → refine_cln must leave cln.npz untouched."""
    from viki.episode import new_episode

    ep = new_episode(tmp_path)
    T, Lc = 6, 21
    pos = np.linspace([0, 0, 0.5], [0.05, 0, 0.5], T).astype(np.float32)
    np.savez_compressed(
        ep.cln_npz,
        positions=pos,
        rotations=np.tile(np.eye(3), (T, 1, 1)).astype(np.float32),
        valid=np.ones(T, bool),
        omega=np.ones(T, np.float32),
        gripper=np.zeros(T, bool),
        timestamps=(np.arange(T) * 33_000).astype(np.int64),
        smoothed_points=np.zeros((T, Lc, 3), np.float32),
        raw_points=np.zeros((T, Lc, 3), np.float32),
        landmark_ids=np.arange(Lc, dtype=np.int32),
        coordinate_frame="robot_base",
    )
    before = np.load(ep.cln_npz)["positions"].copy()
    hf.refine_cln(ep)
    after = np.load(ep.cln_npz)
    assert np.array_equal(before, after["positions"])
    assert "hand_joint_angles" not in after.files


def test_refine_cln_writes_capsules_and_angles(tmp_path, monkeypatch):
    """With a (faked) hand-ROI cloud, refine_cln rewrites cln.npz and adds the
    capsule / joint-angle arrays the viewer needs."""
    from viki.episode import new_episode

    T = 5
    fr = _open_hand_frame(w=np.array([0.10, 0.02, 0.80]))
    lm_ids = np.arange(21, dtype=np.int32)
    sp = np.stack([[fr[LM(int(i))] for i in lm_ids] for _ in range(T)]).astype(np.float32)
    ep = new_episode(tmp_path)
    np.savez_compressed(
        ep.cln_npz,
        positions=np.tile(fr[LM.WRIST], (T, 1)).astype(np.float32),
        rotations=np.tile(np.eye(3), (T, 1, 1)).astype(np.float32),
        valid=np.ones(T, bool),
        omega=np.ones(T, np.float32),
        gripper=np.zeros(T, bool),
        timestamps=(np.arange(T) * 33_000).astype(np.int64),
        smoothed_points=sp,
        raw_points=sp,
        landmark_ids=lm_ids,
        coordinate_frame="robot_base",
    )

    # a hand model at the true (perturbed) pose → cloud sampled on its surface
    hand = hm.build(hm.calibrate_from_frames([fr]))
    q_true = pin.neutral(hand.model).copy()
    q_true[:7] = pin.SE3ToXYZQUAT(pin.SE3(
        pin.exp3(np.array([0.1, 0.15, -0.1])), fr[LM.WRIST] + np.array([0.01, -0.02, 0.015])
    ))
    cloud = _sample_cloud(hand, q_true, per_capsule=90, seed=3)

    monkeypatch.setattr(hf, "_cameras", lambda *a, **k: [{"dev": "fake"}])
    monkeypatch.setattr(hf, "hand_roi_cloud", lambda *a, **k: (cloud, np.ones(len(cloud))))

    w_true = fr[LM.WRIST] + np.array([0.01, -0.02, 0.015])
    warm_err = np.linalg.norm(fr[LM.WRIST] - w_true)   # landmark warm start

    hf.refine_cln(ep)
    d = np.load(ep.cln_npz)
    assert "hand_capsules" in d.files and "hand_capsule_radii" in d.files
    assert d["hand_capsules"].shape == (T, 20, 2, 3)
    assert d["hand_capsule_radii"].shape == (20,)
    assert np.isfinite(d["hand_capsules"][2]).all()
    assert d["hand_joint_angles"].shape == (T, hand.nq)
    assert int(d["hand_model_nq"]) == hand.nq
    # the fitted wrist is pulled toward the (faked) true pose — better than the
    # landmark warm start (the temporal / posture priors keep it from snapping
    # all the way, hence the loose bound).
    err = np.linalg.norm(d["positions"][2] - w_true)
    assert err < warm_err and err < 0.04
