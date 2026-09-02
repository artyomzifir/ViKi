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
    # one broad palm proxy + three phalanges per finger
    assert len(hand.capsules) == 16
    ep = hm.fk_capsule_endpoints(hand, pin.neutral(hand.model))
    assert ep.shape == (16, 2, 3) and np.isfinite(ep).all()


def test_right_hand_positive_flexion_curls_toward_palm_positive_z():
    """Pin the anatomical side: a right-hand grasp curls along palm +z."""
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    q = pin.neutral(hand.model).copy()
    jid = hand.model.getJointId("index_mcp")
    q[hand.model.joints[jid].idx_q] = np.deg2rad(45.0)

    index = hm.fk_landmark_positions(
        hand, q, [int(LM.INDEX_MCP), int(LM.INDEX_PIP)]
    )

    assert index[1, 2] > index[0, 2] + 0.01


def test_signed_angle_keeps_off_axis_bend_magnitude():
    """A noisy out-of-plane landmark must not collapse a real flexion angle."""
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.5, 0.8, -np.sqrt(0.11)])  # unit vector, 60 degrees from u
    angle = hm._signed_angle(u, v, np.array([0.0, 1.0, 0.0]))
    assert np.degrees(angle) == pytest.approx(60.0, abs=1e-6)


def test_calibration_frame_selection_rejects_false_giant_hand():
    """Maximum-spread landmark glitches must not define the hand geometry."""
    frames = [_open_hand_frame(w=[0.002 * t, 0.0, 0.8]) for t in range(12)]
    outlier = {lm: p.copy() for lm, p in frames[-1].items()}
    # Simulate the real failure mode: displaced pinky MCP and fingertip make
    # this frame win a naive maximum-spread ranking.
    for lm in hm.FINGERS["pinky"]:
        outlier[lm] = outlier[lm] + np.array([0.0, -0.09, 0.02])
    frames.append(outlier)

    selected = hf._calibration_frame_indices(frames, np.ones(len(frames), bool), 8)

    assert len(selected) == 8
    assert len(frames) - 1 not in selected


def test_assemble_residuals_shapes():
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    q = pin.neutral(hand.model)
    cloud = _sample_cloud(hand, q, per_capsule=20)
    w = np.ones(len(cloud))
    fc = hf.FitConfig()
    r = hf.assemble_residuals(q, hand, cloud, w, fc)
    # Frame-local assembly has data + limits + posture. Temporal blocks are
    # assembled only by the trajectory functional.
    expect = len(cloud) + 2 * (hand.nq - 7) + (hand.nq - 7)
    assert r.shape == (expect,)
    assert np.isfinite(r).all()
    # at the sampled pose the data residuals are ~noise-sized
    assert np.median(np.abs(r[: len(cloud)])) < 0.01


def test_analytic_jacobian_matches_finite_difference():
    """The hand-written Jacobian in ``residual_and_jac`` must agree with a
    central finite difference of the residual (this is the fit's speed lever —
    a sign error here silently wrecks convergence)."""
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame(), _open_hand_frame()]))
    q0 = pin.neutral(hand.model).copy()
    q0[:7] = pin.SE3ToXYZQUAT(pin.SE3(pin.exp3(np.array([0.1, -0.2, 0.15])),
                                      np.array([0.12, -0.03, 0.85])))
    cloud = _sample_cloud(hand, q0, per_capsule=25, noise=0.003, seed=5)
    w = np.linspace(0.5, 1.5, len(cloud))
    fr = _open_hand_frame(w=np.array([0.12, -0.03, 0.85]))
    order = [0, 5, 9, 17, 8]
    obs = np.array([fr[LM(i)] + 0.004 for i in order])
    fc = hf.FitConfig()

    dt0 = 0.02 * np.cos(np.arange(hand.nv))
    q_at_dt = pin.integrate(hand.model, q0, dt0)
    _, capsule_ids = hf.nearest_capsule(
        cloud, hm.fk_capsule_endpoints(hand, q_at_dt), hm.capsule_radii(hand)
    )
    # Point-to-segment distance is non-differentiable exactly where the closest
    # point switches between a capsule body and an end cap. Keep this test on
    # the smooth body branch; end-cap behaviour is covered by geometry tests.
    assigned_dist, segment_t, _ = hf._assigned_capsule_geom(
        cloud, hm.fk_capsule_endpoints(hand, q_at_dt),
        hm.capsule_radii(hand), capsule_ids,
    )
    smooth = (segment_t > 0.05) & (segment_t < 0.95) & (np.abs(assigned_dist) > 0.001)
    cloud, w, capsule_ids = cloud[smooth], w[smooth], capsule_ids[smooth]
    frame_obs = hf.FrameObservation(
        cloud, w, np.asarray(order), obs, np.ones(len(order)), capsule_ids
    )
    q_rest = pin.neutral(hand.model)

    def evaluate(delta):
        return hf._frame_geometry(delta, hand, q0, frame_obs, fc, q_rest, 1.0)[:2]

    r0, J = evaluate(dt0)

    eps = 1e-6
    Jfd = np.empty_like(J)
    for k in range(hand.nv):
        d = dt0.copy(); d[k] += eps
        rp, _ = evaluate(d)
        d = dt0.copy(); d[k] -= eps
        rm, _ = evaluate(d)
        Jfd[:, k] = (rp - rm) / (2 * eps)

    assert np.allclose(J, Jfd, atol=2e-4, rtol=2e-3)


def test_batch_jacobian_matches_finite_difference():
    """The complete T=4 CSR Jacobian, including two- and three-frame temporal
    rows, agrees with numerical differentiation."""
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    hand.capsules = hand.capsules[:2]  # deliberately tiny two-capsule case
    T = 4
    q0 = np.tile(pin.neutral(hand.model), (T, 1))
    for t in range(T):
        q0[t, :7] = pin.SE3ToXYZQUAT(pin.SE3(
            pin.exp3(np.array([0.01 * t, -0.005 * t, 0.008 * t])),
            np.array([0.01 * t, 0.002 * t, 0.75]),
        ))
    clouds = [_sample_cloud(hand, q, per_capsule=2, noise=0.001, seed=t)
              for t, q in enumerate(q0)]
    fc = hf.FitConfig(min_points=1, max_points=200, w_landmark=0.0)
    observations = hf._make_observations(
        clouds, [None] * T, None, None, hand, fc
    )
    observations = hf.freeze_correspondences(hand, q0, observations)
    dt = 0.002 * np.cos(np.arange(T * hand.nv))
    r0, J = hf.batch_residual_and_jac(dt, hand, q0, observations, fc)
    assert J.format == "csr"

    eps = 1e-6
    Jfd = np.empty(J.shape)
    for k in range(T * hand.nv):
        xp = dt.copy(); xp[k] += eps
        xm = dt.copy(); xm[k] -= eps
        rp, _ = hf.batch_residual_and_jac(xp, hand, q0, observations, fc)
        rm, _ = hf.batch_residual_and_jac(xm, hand, q0, observations, fc)
        Jfd[:, k] = (rp - rm) / (2 * eps)
    assert np.allclose(J.toarray(), Jfd, atol=3e-4, rtol=3e-3)


def _linear_q_trajectory(hand, T=5, step=0.012):
    q = np.tile(pin.neutral(hand.model), (T, 1))
    for t in range(T):
        q[t, :7] = pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), np.array([step * t, 0.0, 0.8])))
    return q


def test_batch_interpolates_completely_missing_frame():
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    true_q = _linear_q_trajectory(hand)
    clouds = [_sample_cloud(hand, q, per_capsule=5, noise=0.001, seed=t)
              for t, q in enumerate(true_q)]
    clouds[2] = np.empty((0, 3))
    q0 = true_q.copy()
    q0[2, :7] = pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), np.array([0.075, 0.02, 0.8])))
    fc = hf.FitConfig(
        min_points=10, max_points=100, outer_iterations=3, max_nfev=35,
        w_landmark=0.0, w_posture=0.0,
        w_vel_translation=20.0, w_acc_translation=1000.0,
    )
    fitted, info = hf.fit_trajectory(hand, clouds, [None] * len(clouds), q0, fc)
    p2, _ = hf.wrist_pose(hand, fitted[2])
    p1, _ = hf.wrist_pose(hand, fitted[1]); p3, _ = hf.wrist_pose(hand, fitted[3])
    assert info["empty_frame_fraction"] == pytest.approx(0.2)
    assert np.linalg.norm(p2 - 0.5 * (p1 + p3)) < 0.005
    assert np.linalg.norm(p2 - np.array([0.024, 0.0, 0.8])) < 0.012


def test_batch_rejects_outlier_and_beats_independent_jerk():
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    true_q = _linear_q_trajectory(hand, T=6, step=0.008)
    clouds = [_sample_cloud(hand, q, per_capsule=6, noise=0.0015, seed=t)
              for t, q in enumerate(true_q)]
    clouds[3] = clouds[3] + np.array([0.070, -0.025, 0.0])
    noisy_init = true_q.copy()
    jitter = np.array([0.0, 0.006, -0.005, 0.008, -0.004, 0.0])
    for t in range(len(noisy_init)):
        p, R = hf.wrist_pose(hand, noisy_init[t])
        noisy_init[t, :7] = pin.SE3ToXYZQUAT(pin.SE3(R, p + [0.0, jitter[t], 0.0]))

    greedy_fc = hf.FitConfig(
        min_points=10, max_points=120, outer_iterations=2, max_nfev=18,
        w_landmark=0.0, w_posture=0.0,
        w_vel_translation=0.0, w_vel_rotation=0.0, w_vel_joints=0.0,
        w_acc_translation=0.0, w_acc_rotation=0.0, w_acc_joints=0.0,
    )
    greedy = np.asarray([
        hf.fit_frame(hand, clouds[t], None, noisy_init[t], greedy_fc)[0]
        for t in range(len(clouds))
    ])
    batch_fc = hf.FitConfig(
        min_points=10, max_points=120, outer_iterations=3, max_nfev=20,
        w_landmark=0.0, w_posture=0.0,
        w_vel_translation=25.0, w_acc_translation=350.0,
    )
    fitted, info = hf.fit_trajectory(hand, clouds, [None] * len(clouds), noisy_init, batch_fc)

    assert hf._jerk_norm(fitted) < hf._jerk_norm(greedy)
    # Five clean frames dominate the episode median, so robustness must not buy
    # smoothness by degrading the ordinary surface fit.
    greedy_resid = []
    for q, cloud in zip(greedy, clouds):
        d, _ = hf.nearest_capsule(cloud, hm.fk_capsule_endpoints(hand, q), hm.capsule_radii(hand))
        greedy_resid.extend(np.abs(d))
    assert info["median_resid_m"] <= np.median(greedy_resid) + 0.002


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
        R_g @ pin.exp3(np.array([0.03, 0.02, -0.025])), w_g + np.array([0.005, -0.006, 0.004])
    ))
    p0_mm, r0_deg = _wrist_errs(hand, q0, R_g, w_g)

    fc = hf.FitConfig(
        w_vel_translation=0.0, w_vel_rotation=0.0, w_vel_joints=0.0,
        w_acc_translation=0.0, w_acc_rotation=0.0, w_acc_joints=0.0,
        w_posture=0.0, outer_iterations=3, max_points=1200,
    )
    anchor_xyz = hm.fk_landmark_positions(hand, q_true, list(range(21)))
    anchor = {LM(i): anchor_xyz[i] for i in range(21)}
    anchors = [anchor] * 2
    q_traj, info = hf.fit_trajectory(
        hand, [cloud, cloud], [None, None], np.stack([q0, q0]), fc,
        landmark_frames=anchors,
    )
    q_fit = q_traj[0]
    pf_mm, rf_deg = _wrist_errs(hand, q_fit, R_g, w_g)

    assert pf_mm < 8.0 and rf_deg < 5.0
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
    assert "hand_fit_joint_angles" not in after.files


def test_refine_cln_writes_new_keys_without_overwriting_pose(tmp_path, monkeypatch):
    """The fit is non-destructive and writes only the hand_fit_* estimate."""
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

    hf.refine_cln(ep)
    d = np.load(ep.cln_npz)
    assert "hand_fit_capsules" in d.files and "hand_fit_capsule_radii" in d.files
    assert d["hand_fit_capsules"].shape == (T, len(hand.capsules), 2, 3)
    assert d["hand_fit_capsule_radii"].shape == (len(hand.capsules),)
    assert np.isfinite(d["hand_fit_capsules"][2]).all()
    assert d["hand_fit_joint_angles"].shape == (T, hand.nq)
    assert int(d["hand_fit_model_nq"]) == hand.nq
    assert d["hand_fit_positions"].shape == (T, 3)
    assert np.array_equal(d["positions"], np.tile(fr[LM.WRIST], (T, 1)).astype(np.float32))
    assert "hand_fit_metrics_json" in d.files


def _bend(hand, q, joints_deg):
    q = np.asarray(q, float).copy()
    for name, deg in joints_deg.items():
        jid = hand.model.getJointId(name)
        q[hand.model.joints[jid].idx_q] = np.deg2rad(deg)
    return q


def _joint_deg(hand, q, name):
    jid = hand.model.getJointId(name)
    return float(np.degrees(q[hand.model.joints[jid].idx_q]))


def test_fit_preserves_finger_bend_against_the_posture_prior():
    """A correctly bent hand must survive the fit, not be pulled straight.

    This is the balance regression guard, and it runs the *shipped* weights on
    purpose. With a flat rest pose and a quadratic prior that ignores whether a
    joint is measured, the prior grows with the very bend it is meant to
    regularise: the same case used to converge to 31 deg and drag the wrist
    16 mm off. The depth cloud has to win on joints it actually observes.

    Note what this does *not* claim: starting from a straight hand the fit
    cannot find a 50 deg bend at all. Nearest-surface assignment binds the
    curled phalanges' points to whatever lies closest in the straight pose --
    the proximal capsules -- so the distal joints never see a gradient. That
    basin is the warm start's job; escaping it needs per-point part labels or
    a multi-hypothesis search, not a reweighting.
    """
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    bent = {f"{f}_{j}": deg for f in ("index", "middle", "ring")
            for j, deg in (("mcp", 40.0), ("pip", 50.0), ("dip", 35.0))}
    q_true = _bend(hand, pin.neutral(hand.model), bent)
    cloud = _sample_cloud(hand, q_true, per_capsule=140, noise=0.0015, seed=11)

    T = 3
    fc = hf.FitConfig(min_points=10, max_points=1600, outer_iterations=4)
    q_flat = pin.neutral(hand.model).copy()      # rest pose pulls the wrong way
    fitted, info = hf.fit_trajectory(
        hand, [cloud] * T, [None] * T, np.tile(q_true, (T, 1)), fc, q_rest=q_flat,
    )

    for name in ("index_pip", "middle_pip", "ring_pip"):
        got = _joint_deg(hand, fitted[1], name)
        assert got > 40.0, (
            f"{name} was straightened to {got:.1f}° from a correct 50°; "
            "the posture prior is outweighing the depth cloud"
        )
    p_true, _ = hf.wrist_pose(hand, q_true)
    p_fit, _ = hf.wrist_pose(hand, fitted[1])
    assert np.linalg.norm(p_fit - p_true) < 0.008   # no wrist drift to absorb the prior
    assert info["median_resid_m"] < 0.002
    assert info["energy_frac_data"] > 0.5
    assert info["energy_frac_posture"] < 0.30


def test_partially_bent_warm_start_moves_toward_the_cloud():
    """From half the true bend the fit must close the gap, not fall back."""
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    bent = {f"{f}_{j}": deg for f in ("index", "middle", "ring")
            for j, deg in (("mcp", 40.0), ("pip", 50.0), ("dip", 35.0))}
    q_true = _bend(hand, pin.neutral(hand.model), bent)
    q_half = _bend(hand, pin.neutral(hand.model), {k: v / 2 for k, v in bent.items()})
    cloud = _sample_cloud(hand, q_true, per_capsule=140, noise=0.0015, seed=11)

    T = 3
    fc = hf.FitConfig(min_points=10, max_points=1600, outer_iterations=4)
    fitted, _ = hf.fit_trajectory(
        hand, [cloud] * T, [None] * T, np.tile(q_half, (T, 1)), fc,
        q_rest=pin.neutral(hand.model).copy(),
    )
    for name in ("index_pip", "middle_pip", "ring_pip"):
        assert _joint_deg(hand, fitted[1], name) > _joint_deg(hand, q_half, name)


def test_posture_prior_is_released_by_data_support():
    """Observed joints get a weak prior, unobserved ones keep the full weight."""
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    q = pin.neutral(hand.model)
    cloud = _sample_cloud(hand, q, per_capsule=60, seed=2)
    fc = hf.FitConfig(min_points=10, max_points=2000)
    obs = hf._make_observations([cloud], [None], None, None, hand, fc)
    frozen = hf.freeze_correspondences(hand, np.stack([q]), obs, fc.inside_scale)[0]

    support = hm.joint_capsule_support(hand)
    assert support.shape == (hand.nq - 7, len(hand.capsules))
    # a DIP joint is only informed by its own distal capsule
    dip_row = hand.model.joints[hand.model.getJointId("index_dip")].idx_q - 7
    assert support[dip_row].sum() == 1
    mcp_row = hand.model.joints[hand.model.getJointId("index_mcp")].idx_q - 7
    assert support[mcp_row].sum() == 3

    assert frozen.posture_weight.shape == (hand.nq - 7,)
    # every joint is measured here, so every prior is heavily discounted
    assert (frozen.posture_weight < 0.05).all()
    # a frame with no depth at all keeps the prior at full strength
    empty = hf.freeze_correspondences(
        hand, np.stack([q]),
        hf._make_observations([np.empty((0, 3))], [None], None, None, hand, fc),
        fc.inside_scale,
    )[0]
    assert np.allclose(empty.posture_weight, 1.0)


def test_unobserved_fingers_hold_the_calibrated_rest_pose():
    """Capsules with no points follow ``q_rest``, never ``pin.neutral``.

    ``q_rest`` was accepted but never filled in by the caller for a while, so
    unobserved joints silently relaxed to the fully-extended neutral pose.
    """
    hand = hm.build(hm.calibrate_from_frames([_open_hand_frame()]))
    relaxed = {f"{f}_{j}": deg for f in hm.FINGERS
               for j, deg in (("pip" if f != "thumb" else "mcp", 32.0),
                              ("dip" if f != "thumb" else "ip", 28.0))}
    q_rest = _bend(hand, pin.neutral(hand.model), relaxed)

    # a cloud that only covers the palm proxy — every phalanx is unobserved
    palm_only = hf.deterministic_voxel_subsample(
        _sample_cloud(hand, q_rest, per_capsule=120, noise=0.001, seed=4)[:120],
        None, 0.002, 400,
    )[0]

    T = 3
    q0 = np.tile(pin.neutral(hand.model), (T, 1))    # start fully extended
    fc = hf.FitConfig(min_points=10, max_points=800, outer_iterations=3)
    fitted, _ = hf.fit_trajectory(
        hand, [palm_only] * T, [None] * T, q0, fc, q_rest=q_rest,
    )

    for name in ("index_dip", "middle_dip", "ring_dip", "pinky_dip"):
        got = _joint_deg(hand, fitted[1], name)
        assert abs(got - 28.0) < 10.0, f"{name} relaxed to {got:.1f}°, not to rest"
        assert got > 15.0, f"{name} collapsed toward pin.neutral ({got:.1f}°)"
