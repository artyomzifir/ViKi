"""
viki.perception.hand_fit
------------------------
Fit the parametric capsule hand (:mod:`viki.perception.hand_model`) to the
per-frame hand point cloud, to get a wrist pose that is more accurate and
temporally stable than one-shot triangulation of 21 sparse landmarks — and, as a
by-product, the full per-frame joint-angle vector for a future anthropomorphic
gripper.

Functional (a separate instance of the thesis eq. 4 structure — a weighted
functional, *not* the robot-IK one in ``retarget/cost.py``)::

    E(θ) = Σ_i w_i · ρ_δ( d(x_i, M(θ)) )       point → nearest capsule surface, Huber
         + λ_vel · ‖θ_t ⊖ θ_{t-1}‖²            temporal velocity   (tangent space)
         + λ_acc · ‖θ_t ⊖ θ_pred‖²             optional acceleration
         + λ_prior·( relu(θ−θ_max)+relu(θ_min−θ) ) + λ_post·‖θ_fingers − θ_rest‖²

Solved per frame with ``scipy.optimize.least_squares`` (``method='trf'``,
``loss='huber'``) over the tangent increment ``δθ ∈ R^nv`` about a warm-start
config (``θ = pin.integrate(model, θ0, δθ)``), so the free-flyer stays on
SO(3)×R³. Warm start: landmarks on frame 0, the previous frame's solution after.
Jacobian: scipy's finite-difference default for now — an analytic capsule-endpoint
Jacobian via ``pin.computeJointJacobians`` is a **TODO** (offline, so the FD cost
is acceptable).

Cloud source: a **dense hand-ROI cloud re-deprojected from raw depth** per frame
(sphere of radius ``PERCEPTION_HAND_FIT_ROI_M`` around the fused wrist estimate,
full resolution, background subtracted, *no* voxel downsample) — not the on-disk
visualisation artifact, which is voxel-5 mm + capped and too sparse across a
finger. See :func:`hand_roi_cloud`.

Integration: :func:`refine_cln` rewrites ``cln.npz`` ``positions`` / ``rotations``
**in place** (only where ``valid`` and the fit's median residual is acceptable),
adds ``hand_joint_angles`` (T, nq) + ``hand_model_nq``. It runs at the end of
``prepare_episode`` when ``PERCEPTION_HAND_FIT`` is set, and is also the CLI
entry ``viki hand-fit``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from viki.contracts import LM
from viki.perception import hand_model as hm

logger = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────


@dataclass
class FitConfig:
    roi_m: float = 0.12
    huber_delta_m: float = 0.010
    w_vel: float = 40.0
    w_acc: float = 10.0
    w_prior: float = 200.0        # joint-limit barrier
    w_posture: float = 2.0        # pull fingers toward rest
    w_landmark: float = 20.0      # anchor the model's joints to the fused landmarks
    min_points: int = 60
    max_points: int = 2500        # random-subsample the ROI cloud to this (speed)
    accept_median_resid_m: float = 0.020
    max_nfev: int = 45
    calib_frames: int = 8

    @classmethod
    def from_config(cls, cfg=None) -> "FitConfig":
        from viki import config as _c
        cfg = cfg or _c
        g = lambda k, d: float(getattr(cfg, k, d))
        return cls(
            roi_m=g("PERCEPTION_HAND_FIT_ROI_M", 0.12),
            huber_delta_m=g("PERCEPTION_HAND_FIT_HUBER_M", 0.010),
            w_vel=g("PERCEPTION_HAND_FIT_W_VEL", 40.0),
            w_acc=g("PERCEPTION_HAND_FIT_W_ACC", 10.0),
            w_prior=g("PERCEPTION_HAND_FIT_W_PRIOR", 200.0),
            w_posture=g("PERCEPTION_HAND_FIT_W_POSTURE", 2.0),
            w_landmark=g("PERCEPTION_HAND_FIT_W_LANDMARK", 20.0),
            max_points=int(g("PERCEPTION_HAND_FIT_MAX_POINTS", 2500)),
        )


# ── point → capsule geometry ─────────────────────────────────────────────


def point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of ``pts`` (N,3) to segment ``a``–``b``."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / L2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def point_capsule_signed_distance(
    pts: np.ndarray, a: np.ndarray, b: np.ndarray, r: float
) -> np.ndarray:
    """Signed distance to the capsule *surface*: negative inside, 0 on it."""
    return point_segment_distance(pts, a, b) - float(r)


def nearest_capsule_geom(
    pts: np.ndarray, endpoints: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-capsule assignment *plus* the bits the analytic Jacobian needs.

    Returns ``(dist (N,), idx (N,), t (N,), n (N,3))`` where, for each point's
    chosen capsule, ``t`` is the clamped projection parameter along the segment
    and ``n`` is the *unit* vector from the closest segment point to the point
    (``d dist / d(closest point) = -n``). Fully vectorised over points × capsules
    — the fit's hot path.
    """
    pts = np.asarray(pts, float).reshape(-1, 3)          # (N, 3)
    a = np.asarray(endpoints, float)[:, 0]               # (C, 3)
    b = np.asarray(endpoints, float)[:, 1]               # (C, 3)
    ab = b - a                                           # (C, 3)
    L2 = np.einsum("cd,cd->c", ab, ab)                   # (C,)
    L2 = np.where(L2 < 1e-12, 1.0, L2)
    ap = pts[:, None, :] - a[None, :, :]                 # (N, C, 3)
    tt = np.clip(np.einsum("ncd,cd->nc", ap, ab) / L2, 0.0, 1.0)  # (N, C)
    perp = ap - tt[..., None] * ab[None, :, :]           # (N, C, 3)
    pn = np.linalg.norm(perp, axis=2)                    # (N, C)
    D = pn - np.asarray(radii, float)[None, :]           # (N, C)
    idx = np.argmin(np.abs(D), axis=1)                   # (N,)
    rows = np.arange(len(pts))
    n = perp[rows, idx] / np.maximum(pn[rows, idx], 1e-9)[:, None]  # (N, 3) unit
    return D[rows, idx], idx, tt[rows, idx], n


def nearest_capsule(
    pts: np.ndarray, endpoints: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the min signed distance over all capsules + which capsule.

    ``endpoints`` (C, 2, 3), ``radii`` (C,). Returns ``(dist (N,), idx (N,))``.
    """
    d, idx, _t, _n = nearest_capsule_geom(pts, endpoints, radii)
    return d, idx


# ── residual assembly ────────────────────────────────────────────────────


def assemble_residuals(
    q: np.ndarray,
    hand: "hm.CapsuleHand",
    cloud: np.ndarray,
    weights: np.ndarray,
    fc: FitConfig,
    *,
    q_prev: np.ndarray | None = None,
    q_pred: np.ndarray | None = None,
    q_rest: np.ndarray | None = None,
    lm_anchor: Mapping[LM, np.ndarray] | None = None,
) -> np.ndarray:
    """Full residual vector for config ``q``: data (point→capsule) + landmark
    anchor + regularisers."""
    import pinocchio as pin

    order = []
    if lm_anchor and fc.w_landmark > 0:
        order = [int(lm) for lm, p in lm_anchor.items()
                 if int(lm) in hand.lm_frames and np.all(np.isfinite(p))]

    ep, model_p = hm.fk_capsule_and_landmarks(hand, q, order)   # one FK pass
    radii = hm.capsule_radii(hand)
    d, _ = nearest_capsule(cloud, ep, radii)
    parts = [np.sqrt(np.asarray(weights, float).reshape(-1)) * d]

    if order:
        obs = np.array([lm_anchor[LM(i)] for i in order], float)
        parts.append(np.sqrt(fc.w_landmark) * (model_p - obs).reshape(-1))

    if q_prev is not None and fc.w_vel > 0:
        parts.append(np.sqrt(fc.w_vel) * pin.difference(hand.model, q_prev, q))
    if q_pred is not None and fc.w_acc > 0:
        parts.append(np.sqrt(fc.w_acc) * pin.difference(hand.model, q_pred, q))

    qv = np.asarray(q, float)
    if fc.w_prior > 0:
        over = np.maximum(0.0, qv - hand.q_hi)
        under = np.maximum(0.0, hand.q_lo - qv)
        parts.append(np.sqrt(fc.w_prior) * np.concatenate([over[7:], under[7:]]))
    if fc.w_posture > 0:
        rest = pin.neutral(hand.model) if q_rest is None else np.asarray(q_rest, float)
        parts.append(np.sqrt(fc.w_posture) * (qv[7:] - rest[7:]))

    return np.concatenate(parts)


def residual_and_jac(
    dtheta: np.ndarray,
    hand: "hm.CapsuleHand",
    q0: np.ndarray,
    cloud: np.ndarray,
    weights: np.ndarray,
    fc: FitConfig,
    *,
    q_prev: np.ndarray | None,
    q_pred: np.ndarray | None,
    q_rest: np.ndarray | None,
    order: list[int],
    obs_lm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual vector **and** its analytic Jacobian w.r.t. the tangent
    increment ``dtheta`` (``q = integrate(q0, dtheta)``).

    Row order matches :func:`assemble_residuals` exactly. The point→capsule and
    landmark rows are differentiated through Pinocchio frame Jacobians; the
    temporal rows through ``dDifference``; the barrier / posture rows act on the
    revolute config block directly. Each geometric block is mapped from the
    tangent space at ``q`` back to ``dtheta`` via ``dIntegrate``'s ARG1 Jacobian.

    Replacing scipy's 2-point finite differences (``nv+1`` FK passes per solver
    iteration) with one FK + one Jacobian pass is the fit's main speed lever.
    """
    import pinocchio as pin

    model, data = hand.model, hand.data
    nv = model.nv
    q = pin.integrate(model, q0, np.asarray(dtheta, float))
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    Jint = pin.dIntegrate(model, q0, np.asarray(dtheta, float))[1]   # nv×nv, ARG1
    P = data.oMf
    LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

    _jc: dict[int, np.ndarray] = {}

    def frameJ(fid: int) -> np.ndarray:            # translational rows, 3×nv
        J = _jc.get(fid)
        if J is None:
            J = np.asarray(pin.getFrameJacobian(model, data, fid, LWA))[:3]
            _jc[fid] = J
        return J

    r_parts: list[np.ndarray] = []
    Jg_parts: list[np.ndarray] = []               # rows in tangent-at-q space

    # ── data term: sqrt(w) * signed point→capsule distance ──────────────
    ep = np.empty((len(hand.capsules), 2, 3), float)
    for i, (fa, fb, _r) in enumerate(hand.capsules):
        ep[i, 0], ep[i, 1] = P[fa].translation, P[fb].translation
    radii = np.array([r for _a, _b, r in hand.capsules], float)
    dist, cidx, tparam, nrm = nearest_capsule_geom(cloud, ep, radii)
    sw = np.sqrt(np.asarray(weights, float).reshape(-1))
    r_parts.append(sw * dist)
    Jd = np.zeros((len(cloud), nv), float)
    for c in range(len(hand.capsules)):
        m = cidx == c
        if not m.any():
            continue
        fa, fb, _r = hand.capsules[c]
        n_c = nrm[m]                               # (k,3) unit
        t_c = tparam[m][:, None]                   # (k,1)
        # d dist/dv = -(1-t) nᵀ Jₐ - t nᵀ J_b
        Jd[m] = -((1.0 - t_c) * (n_c @ frameJ(fa)) + t_c * (n_c @ frameJ(fb)))
    Jg_parts.append(sw[:, None] * Jd)

    # ── landmark anchor ────────────────────────────────────────────────
    if order:
        slm = np.sqrt(fc.w_landmark)
        model_p = np.array([P[hand.lm_frames[i]].translation for i in order], float)
        r_parts.append((slm * (model_p - obs_lm)).reshape(-1))
        Jg_parts.append(slm * np.concatenate([frameJ(hand.lm_frames[i]) for i in order], 0))

    r = np.concatenate(r_parts)
    J_parts = [np.vstack(Jg_parts) @ Jint]        # geometric → dtheta space

    # ── temporal (velocity / acceleration) ────────────────────────────
    if q_prev is not None and fc.w_vel > 0:
        s = np.sqrt(fc.w_vel)
        r = np.concatenate([r, s * pin.difference(model, q_prev, q)])
        J_parts.append(s * (pin.dDifference(model, q_prev, q)[1] @ Jint))
    if q_pred is not None and fc.w_acc > 0:
        s = np.sqrt(fc.w_acc)
        r = np.concatenate([r, s * pin.difference(model, q_pred, q)])
        J_parts.append(s * (pin.dDifference(model, q_pred, q)[1] @ Jint))

    # ── joint-limit barrier + posture (revolute config block) ─────────
    qv = np.asarray(q, float)
    Jq_rev = Jint[6:, :]                           # d q[7:] / d dtheta, (nq-7)×nv
    if fc.w_prior > 0:
        s = np.sqrt(fc.w_prior)
        over = np.maximum(0.0, qv - hand.q_hi)[7:]
        under = np.maximum(0.0, hand.q_lo - qv)[7:]
        r = np.concatenate([r, s * over, s * under])
        J_parts.append(s * (over > 0).astype(float)[:, None] * Jq_rev)
        J_parts.append(-s * (under > 0).astype(float)[:, None] * Jq_rev)
    if fc.w_posture > 0:
        s = np.sqrt(fc.w_posture)
        rest = pin.neutral(model) if q_rest is None else np.asarray(q_rest, float)
        r = np.concatenate([r, s * (qv[7:] - rest[7:])])
        J_parts.append(s * Jq_rev)

    return r, np.vstack(J_parts)


def fit_frame(
    hand: "hm.CapsuleHand",
    cloud: np.ndarray,
    weights: np.ndarray | None,
    q0: np.ndarray,
    fc: FitConfig,
    *,
    q_prev: np.ndarray | None = None,
    q_pred: np.ndarray | None = None,
    lm_anchor: Mapping[LM, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """One frame: least-squares fit of the tangent increment about ``q0``.

    Returns ``(q, info)`` where ``info`` has ``skipped`` / ``median_resid`` /
    ``accepted`` / ``nfev``.
    """
    import pinocchio as pin
    from scipy.optimize import least_squares

    cloud = np.asarray(cloud, float).reshape(-1, 3)
    q0 = np.asarray(q0, float)
    if len(cloud) < fc.min_points:
        return q0, {"skipped": True, "n_points": int(len(cloud))}
    w = np.ones(len(cloud)) if weights is None else np.asarray(weights, float).reshape(-1)
    if len(cloud) > fc.max_points:                       # subsample for speed
        sel = np.random.default_rng(0).choice(len(cloud), fc.max_points, replace=False)
        cloud, w = cloud[sel], w[sel]

    order: list[int] = []
    obs_lm = None
    if lm_anchor and fc.w_landmark > 0:
        order = [int(lm) for lm, p in lm_anchor.items()
                 if int(lm) in hand.lm_frames and np.all(np.isfinite(p))]
        obs_lm = np.array([lm_anchor[LM(i)] for i in order], float) if order else None

    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def _eval(dtheta):
        key = np.asarray(dtheta, float).tobytes()
        out = cache.get(key)
        if out is None:
            out = residual_and_jac(
                dtheta, hand, q0, cloud, w, fc, q_prev=q_prev, q_pred=q_pred,
                q_rest=None, order=order, obs_lm=obs_lm)
            if len(cache) > 4:
                cache.clear()
            cache[key] = out
        return out

    # Hard box on the tangent step: the revolute block can't leave the joint
    # limits, the free-flyer can't jump more than a hand's width / ~1 rad per
    # frame. Without this the analytic-Jacobian solve can walk the wrist into a
    # far-off cloud lobe through the wrist↔finch nullspace.
    lb = np.empty(hand.nv); ub = np.empty(hand.nv)
    lb[:6] = [-0.15, -0.15, -0.15, -1.0, -1.0, -1.0]
    ub[:6] = -lb[:6]
    lb[6:] = np.minimum(hand.q_lo[7:] - q0[7:], -1e-6)
    ub[6:] = np.maximum(hand.q_hi[7:] - q0[7:], 1e-6)

    res = least_squares(
        lambda d: _eval(d)[0], np.zeros(hand.nv), jac=lambda d: _eval(d)[1],
        bounds=(lb, ub), method="trf", loss="huber",
        f_scale=fc.huber_delta_m, x_scale="jac", max_nfev=fc.max_nfev,
    )
    q = pin.integrate(hand.model, q0, res.x)

    ep = hm.fk_capsule_endpoints(hand, q)
    d, _ = nearest_capsule(cloud, ep, hm.capsule_radii(hand))
    med = float(np.median(np.abs(d)))
    return q, {
        "skipped": False, "n_points": int(len(cloud)),
        "median_resid": med, "accepted": med <= fc.accept_median_resid_m,
        "nfev": int(res.nfev),
    }


# ── wrist pose out of a fitted config ────────────────────────────────────


def wrist_pose(hand: "hm.CapsuleHand", q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position (3,), R_world_palm (3,3)) from the free-flyer part of ``q``."""
    import pinocchio as pin

    se3 = pin.XYZQUATToSE3(np.asarray(q, float)[:7])
    return np.asarray(se3.translation, float), np.asarray(se3.rotation, float)


# ── dense hand-ROI cloud from raw depth ─────────────────────────────────


def _cameras(raw: Path, meta: dict, preset: str | None, bg_subtract: bool):
    """Per-camera (K4ACalibration, T_world_cam, depth_dir, bg_mm) for the episode."""
    from viki.contracts import CalibrationExtrinsics
    from viki.perception.k4a_offline import K4ACalibration

    extr = json.loads((raw / "extrinsics.json").read_text()) if (raw / "extrinsics.json").exists() else {}
    bg_by_dev: dict = {}
    if bg_subtract and preset:
        try:
            from viki.calibration import presets as _p
            for mp4 in raw.glob("*.mp4"):
                bd = _p.background_depth(preset, mp4.stem)
                if bd is not None:
                    bg_by_dev[mp4.stem] = bd
        except Exception as exc:  # noqa: BLE001
            logger.warning("hand_fit: background load failed (%s)", exc)

    cams = []
    for mp4 in sorted(raw.glob("*.mp4")):
        dev = mp4.stem
        e = extr.get(dev)
        if not e:
            continue
        T = CalibrationExtrinsics(
            rvec=np.asarray(e["rvec"], float), tvec=np.asarray(e["tvec"], float)
        ).transform_matrix
        cal = K4ACalibration.from_episode(raw, dev, meta)
        if cal is None:
            logger.warning("hand_fit %s: no k4a calib — camera skipped for fitting", dev)
            continue
        cams.append({"dev": dev, "cal": cal, "T": T,
                     "depth_dir": raw / f"{dev}_depth", "bg": bg_by_dev.get(dev)})
    return cams


def hand_roi_cloud(
    cams: list[dict], frame_i: int, wrist_world: np.ndarray, roi_m: float,
    bg_tol_mm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-res world-frame points within ``roi_m`` of ``wrist_world`` for one
    synced frame, background subtracted, no voxel. Weight = inverse range²."""
    w0 = np.asarray(wrist_world, float)
    xyz_parts, wt_parts = [], []
    for c in cams:
        dp = c["depth_dir"] / f"{frame_i:06d}.npy"
        if not dp.is_file():
            continue
        depth_mm = np.load(dp)
        if not depth_mm.any():
            continue
        dh, dw = depth_mm.shape[:2]
        A, B = c["cal"].color_deproject_maps(dh, dw)   # (dh,dw,3) mm, cached
        z = depth_mm.astype(np.float64)
        keep = z > 0
        bg = c["bg"]
        if bg is not None and bg.shape == depth_mm.shape:
            keep &= ~((bg > 0) & (np.abs(z - bg) <= float(bg_tol_mm)))
        vs, us = np.nonzero(keep)
        if us.size == 0:
            continue
        pts_mm = z[vs, us, None] * A[vs, us] + B[vs, us]      # colour-cam frame, mm
        fin = np.isfinite(pts_mm).all(axis=1)
        pts_cam = pts_mm[fin] / 1000.0
        world = pts_cam @ c["T"][:3, :3].T + c["T"][:3, 3]
        m = np.linalg.norm(world - w0, axis=1) <= roi_m
        if not m.any():
            continue
        wsel = world[m]
        rng = np.linalg.norm(pts_cam[m], axis=1)
        xyz_parts.append(wsel)
        wt_parts.append(1.0 / np.maximum(rng, 0.1) ** 2)
    if not xyz_parts:
        return np.empty((0, 3)), np.empty((0,))
    xyz = np.concatenate(xyz_parts)
    wt = np.concatenate(wt_parts)
    return xyz, wt / (wt.mean() + 1e-9)


# ── orchestration ───────────────────────────────────────────────────────


def _spread(pts: Mapping[LM, np.ndarray]) -> float:
    tips = [pts.get(t) for t in (LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP,
                                 LM.RING_TIP, LM.PINKY_TIP)]
    tips = [np.asarray(p, float) for p in tips if p is not None and np.all(np.isfinite(p))]
    if len(tips) < 3:
        return -1.0
    T = np.stack(tips)
    return float(np.mean([np.linalg.norm(T[i] - T[j])
                          for i in range(len(T)) for j in range(i + 1, len(T))]))


def refine_cln(ep, cfg=None, report=None) -> str:
    """Refine ``ep.cln_npz`` wrist poses by cloud fitting. Returns the cln path.

    No-op (returns unchanged) when the episode has no usable k4a depth or the
    cln is too short. ``report(stage="hand_fit", frame=t, total=T)`` drives a
    progress bar when driven from a job.
    """
    report = report or (lambda **_k: None)
    fc = FitConfig.from_config(cfg)
    cln_path = Path(ep.cln_npz)
    with np.load(cln_path, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}

    pos = np.asarray(data["positions"], np.float64)
    rot = np.asarray(data["rotations"], np.float64)
    valid = np.asarray(data["valid"], bool)
    lm_ids = np.asarray(data["landmark_ids"], int)
    sp = np.asarray(data["smoothed_points"], np.float64)   # (T, L, 3)
    T = len(pos)
    if T < 2:
        logger.info("hand_fit %s: <2 frames, skipping", ep.id)
        return str(cln_path)

    frames = [{LM(int(lm_ids[j])): sp[t, j] for j in range(sp.shape[1])} for t in range(T)]

    raw = Path(ep.raw_dir)
    meta = json.loads(Path(ep.meta_path).read_text()) if Path(ep.meta_path).exists() else {}
    preset = meta.get("calibration_preset")
    from viki import config as _cfg
    cams = _cameras(raw, meta, preset, bool(getattr(_cfg, "CLOUD_BG_SUBTRACT", True)))
    if not cams:
        logger.warning("hand_fit %s: no camera with k4a depth — cln unchanged", ep.id)
        return str(cln_path)

    # calibrate the hand from the most open frames
    order = np.argsort([-_spread(frames[t]) for t in range(T)])
    calib_frames = [frames[t] for t in order[: fc.calib_frames] if _spread(frames[t]) > 0]
    try:
        params = hm.calibrate_from_frames(calib_frames or frames)
        hand = hm.build(params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hand_fit %s: model calibration failed (%s) — cln unchanged", ep.id, exc)
        return str(cln_path)

    import pinocchio as pin
    nq = hand.nq
    C = len(hand.capsules)
    q_traj = np.tile(pin.neutral(hand.model), (T, 1)).astype(np.float64)
    caps = np.full((T, C, 2, 3), np.nan, np.float32)   # world capsule endpoints per frame
    cap_r = hm.capsule_radii(hand).astype(np.float32)
    q_prev = None
    n_acc = 0
    report(stage="hand_fit", frame=0, total=T)
    for t in range(T):
        if not valid[t] or not np.all(np.isfinite(pos[t])):
            q_prev = None
            continue
        cloud, wts = hand_roi_cloud(cams, t, pos[t], fc.roi_m,
                                    float(getattr(_cfg, "CLOUD_BG_TOLERANCE_MM", 50.0)))
        q0 = q_prev if q_prev is not None else hm.q_from_landmarks(hand, frames[t])
        q_pred = None
        if q_prev is not None and t >= 2 and np.all(np.isfinite(pos[t - 1])):
            # const-velocity extrapolation on the manifold (mirrors retarget.cost)
            dv = pin.difference(hand.model, q_traj[t - 2], q_prev)
            q_pred = pin.integrate(hand.model, q_prev, dv)
        q, info = fit_frame(hand, cloud, wts, q0, fc, q_prev=q_prev, q_pred=q_pred,
                            lm_anchor=frames[t])
        p, R = wrist_pose(hand, q)
        # the true wrist sits inside the ROI we cropped around the landmark wrist,
        # so measure divergence against the *landmark* wrist (pos[t] is the value
        # we're trying to correct — it can't be the reference).
        w_lm = frames[t].get(LM.WRIST)
        diverged = (w_lm is not None and np.all(np.isfinite(w_lm))
                    and float(np.linalg.norm(p - np.asarray(w_lm, float))) > 1.5 * fc.roi_m)
        good = (not info["skipped"]) and info["accepted"] and not diverged
        if good:
            q_traj[t] = q
            caps[t] = hm.fk_capsule_endpoints(hand, q)
            pos[t] = p
            rot[t] = R
            n_acc += 1
        # chain the warm start only from a clean fit — never ratchet off a bad one
        q_prev = q if good else None
        if t % 5 == 0:
            report(stage="hand_fit", frame=t, total=T)

    data["positions"] = pos.astype(np.float32)
    data["rotations"] = rot.astype(np.float32)
    data["hand_joint_angles"] = q_traj.astype(np.float32)
    data["hand_model_nq"] = np.int64(nq)
    data["hand_capsules"] = caps                       # (T, C, 2, 3) world, NaN where unfitted
    data["hand_capsule_radii"] = cap_r                 # (C,)
    np.savez_compressed(cln_path, **data)
    report(stage="hand_fit", frame=T, total=T)
    logger.info("hand_fit %s: refined %d/%d frames (nq=%d, %d capsules)",
                ep.id, n_acc, int(valid.sum()), nq, C)
    return str(cln_path)
