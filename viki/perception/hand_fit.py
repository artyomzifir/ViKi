"""Trajectory-level articulated hand fitting.

The unknown is one tangent increment for every frame and the complete episode
is solved at once.  Data rows touch one frame, velocity rows two adjacent
frames, and acceleration rows three, so the analytic Jacobian is assembled
directly as CSR and never materialised as a dense trajectory matrix.

An outer ICP loop freezes point→capsule identities before each smooth batch
solve.  Landmark anchors use per-landmark confidence and decay geometrically as
``0.35**outer_iteration``: they select the initial basin but do not remain the
final target.  One episode is one batch, including empty-data frames; temporal
edges therefore interpolate gaps using information from both sides.  A sliding
window is intentionally not implemented: the target 300-frame/26-DoF problem
has only a few million CSR nonzeros and is expected to fit in 16 GiB.  Actual
episode measurements belong in ``docs/hand_fit_batch_design.md`` once real depth
data is available on the target host.

The hand surface remains licence-free capsule geometry with its proven signed
distance and analytic endpoint Jacobian.  The palm choice is documented in
``hand_model.build``: one broad capsule replaces five overlapping ones.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from viki.contracts import LM
from viki.perception import hand_model as hm

logger = logging.getLogger(__name__)


class _FitDeadline(Exception):
    """Raised from the residual callback when a per-solve wall-clock budget is
    exceeded, so one pathological episode can't tie up a core indefinitely."""


@dataclass
class FitConfig:
    """All trajectory-fit tunables (translation is metres, angles radians)."""

    roi_margin_m: float = 0.030
    forearm_cut_m: float = 0.010
    voxel_m: float = 0.004
    huber_delta_m: float = 0.010
    w_data: float = 500.0
    w_vel_translation: float = 40.0
    w_vel_rotation: float = 4.0
    w_vel_joints: float = 0.8
    w_acc_translation: float = 120.0
    w_acc_rotation: float = 10.0
    w_acc_joints: float = 1.6
    w_prior: float = 100.0
    w_posture: float = 0.004
    w_landmark: float = 4.0
    landmark_decay: float = 0.35
    inside_scale: float = 0.15
    min_points: int = 40
    max_points: int = 400
    max_nfev: int = 35
    outer_iterations: int = 4
    outer_step_tol: float = 2e-4
    calib_frames: int = 8
    window: int = 120           # frames per sliding window; 0 = whole-episode batch
    window_overlap: int = 30    # overlapping frames blended between windows
    workers: int = 0            # window-solver threads; 0 = auto (min(4, cpu/2))
    warm_start_mad_k: float = 6.0  # wrist warm-start outlier gate (robust MAD units)
    deadline_s: float = 120.0   # wall-clock guard per fit_trajectory call; 0 = off

    @classmethod
    def from_config(cls, cfg=None) -> "FitConfig":
        from viki import config as _c

        cfg = cfg or _c
        gf = lambda k, d: float(getattr(cfg, k, d))
        gi = lambda k, d: int(getattr(cfg, k, d))
        return cls(
            roi_margin_m=gf("PERCEPTION_HAND_FIT_ROI_MARGIN_M", 0.030),
            forearm_cut_m=gf("PERCEPTION_HAND_FIT_FOREARM_CUT_M", 0.010),
            voxel_m=gf("PERCEPTION_HAND_FIT_VOXEL_M", 0.004),
            huber_delta_m=gf("PERCEPTION_HAND_FIT_HUBER_M", 0.010),
            w_data=gf("PERCEPTION_HAND_FIT_W_DATA", 500.0),
            w_vel_translation=gf("PERCEPTION_HAND_FIT_W_VEL_TRANSLATION", 40.0),
            w_vel_rotation=gf("PERCEPTION_HAND_FIT_W_VEL_ROTATION", 4.0),
            w_vel_joints=gf("PERCEPTION_HAND_FIT_W_VEL_JOINTS", 0.8),
            w_acc_translation=gf("PERCEPTION_HAND_FIT_W_ACC_TRANSLATION", 120.0),
            w_acc_rotation=gf("PERCEPTION_HAND_FIT_W_ACC_ROTATION", 10.0),
            w_acc_joints=gf("PERCEPTION_HAND_FIT_W_ACC_JOINTS", 1.6),
            w_prior=gf("PERCEPTION_HAND_FIT_W_PRIOR", 100.0),
            w_posture=gf("PERCEPTION_HAND_FIT_W_POSTURE", 0.004),
            w_landmark=gf("PERCEPTION_HAND_FIT_W_LANDMARK", 4.0),
            landmark_decay=gf("PERCEPTION_HAND_FIT_LANDMARK_DECAY", 0.35),
            inside_scale=gf("PERCEPTION_HAND_FIT_INSIDE_SCALE", 0.15),
            min_points=gi("PERCEPTION_HAND_FIT_MIN_POINTS", 40),
            max_points=gi("PERCEPTION_HAND_FIT_MAX_POINTS", 400),
            max_nfev=gi("PERCEPTION_HAND_FIT_MAX_NFEV", 35),
            outer_iterations=gi("PERCEPTION_HAND_FIT_OUTER_ITERATIONS", 4),
            window=gi("PERCEPTION_HAND_FIT_WINDOW", 120),
            window_overlap=gi("PERCEPTION_HAND_FIT_WINDOW_OVERLAP", 30),
            workers=gi("PERCEPTION_HAND_FIT_WORKERS", 0),
            warm_start_mad_k=gf("PERCEPTION_HAND_FIT_WARM_START_MAD_K", 6.0),
            deadline_s=gf("PERCEPTION_HAND_FIT_DEADLINE_S", 120.0),
        )


# ── point → capsule geometry (kept deliberately small and analytic) ─────


def point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 3)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    length2 = float(ab @ ab)
    if length2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / length2, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def point_capsule_signed_distance(
    pts: np.ndarray, a: np.ndarray, b: np.ndarray, r: float
) -> np.ndarray:
    return point_segment_distance(pts, a, b) - float(r)


def nearest_capsule_geom(
    pts: np.ndarray, endpoints: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed distance, capsule id, clamped segment t, and surface normal."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    if len(pts) == 0:
        return (np.empty(0), np.empty(0, int), np.empty(0), np.empty((0, 3)))
    a = np.asarray(endpoints, float)[:, 0]
    ab = np.asarray(endpoints, float)[:, 1] - a
    length2 = np.einsum("cd,cd->c", ab, ab)
    length2 = np.where(length2 < 1e-12, 1.0, length2)
    ap = pts[:, None, :] - a[None, :, :]
    all_t = np.clip(np.einsum("ncd,cd->nc", ap, ab) / length2, 0.0, 1.0)
    perp = ap - all_t[..., None] * ab[None, :, :]
    pn = np.linalg.norm(perp, axis=2)
    all_d = pn - np.asarray(radii, float)[None, :]
    idx = np.argmin(np.abs(all_d), axis=1)
    rows = np.arange(len(pts))
    nrm = perp[rows, idx] / np.maximum(pn[rows, idx], 1e-9)[:, None]
    return all_d[rows, idx], idx, all_t[rows, idx], nrm


def nearest_capsule(
    pts: np.ndarray, endpoints: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    d, idx, _, _ = nearest_capsule_geom(pts, endpoints, radii)
    return d, idx


def _assigned_capsule_geom(
    pts: np.ndarray, endpoints: np.ndarray, radii: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometry for frozen capsule identities; projection remains smooth."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    idx = np.asarray(idx, int)
    if len(pts) == 0:
        return np.empty(0), np.empty(0), np.empty((0, 3))
    a = endpoints[idx, 0]
    ab = endpoints[idx, 1] - a
    length2 = np.einsum("nd,nd->n", ab, ab)
    length2 = np.where(length2 < 1e-12, 1.0, length2)
    t = np.clip(np.einsum("nd,nd->n", pts - a, ab) / length2, 0.0, 1.0)
    perp = pts - (a + t[:, None] * ab)
    pn = np.linalg.norm(perp, axis=1)
    nrm = perp / np.maximum(pn, 1e-9)[:, None]
    return pn - radii[idx], t, nrm


def deterministic_voxel_subsample(
    cloud: np.ndarray, weights: np.ndarray | None, voxel_m: float, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select one stable representative per voxel, then uniformly in key order."""
    cloud = np.asarray(cloud, float).reshape(-1, 3)
    weights = np.ones(len(cloud)) if weights is None else np.asarray(weights, float).reshape(-1)
    finite = np.isfinite(cloud).all(axis=1) & np.isfinite(weights) & (weights > 0)
    cloud, weights = cloud[finite], weights[finite]
    if not len(cloud):
        return cloud, weights
    keys = np.floor(cloud / max(float(voxel_m), 1e-6)).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_s = keys[order]
    first = np.r_[True, np.any(keys_s[1:] != keys_s[:-1], axis=1)]
    chosen = order[first]
    if len(chosen) > max_points:
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_points).round().astype(int)]
    return cloud[chosen], weights[chosen]


@dataclass
class FrameObservation:
    cloud: np.ndarray
    weights: np.ndarray
    lm_order: np.ndarray
    lm_points: np.ndarray
    lm_confidence: np.ndarray
    capsule_ids: np.ndarray
    # One-sided attenuation for samples that fall inside the model surface,
    # frozen together with the correspondences. Freezing it keeps the residual
    # differentiable through the inner solve (the factor no longer jumps at
    # d = 0, where the analytic Jacobian could not see the step) and lets the
    # robust loss below use exactly the same alpha the residual was built from.
    data_scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    # Per-revolute-joint posture multiplier, frozen with the correspondences.
    posture_weight: np.ndarray = field(default_factory=lambda: np.ones(0))


def data_alpha(obs: FrameObservation, fc: FitConfig) -> np.ndarray:
    """Per-point data weight alpha, shared by the residual and the robust loss.

    ``alpha = w_data * scale^2 * w / sum(w)``. Dividing by ``sum(w)`` makes the
    block invariant to how many depth pixels survived the ROI, and ``w_data``
    restores its magnitude: without it the data block is a per-frame *mean* of
    squared distances while every other term is a *sum*, which on real episodes
    left the depth cloud contributing 0.1% of the functional.
    """
    if not len(obs.cloud):
        return np.empty(0)
    scale = obs.data_scale if len(obs.data_scale) == len(obs.cloud) else np.ones(len(obs.cloud))
    return (fc.w_data * scale ** 2 * obs.weights
            / max(float(obs.weights.sum()), 1e-12))


def _make_observations(
    clouds: Sequence[np.ndarray], weights: Sequence[np.ndarray | None],
    landmark_frames: Sequence[Mapping[LM, np.ndarray]] | None,
    landmark_confidence: np.ndarray | None, hand: "hm.CapsuleHand", fc: FitConfig,
) -> list[FrameObservation]:
    out: list[FrameObservation] = []
    for t, (cloud, w) in enumerate(zip(clouds, weights)):
        c, ww = deterministic_voxel_subsample(cloud, w, fc.voxel_m, fc.max_points)
        # An almost-empty block is treated exactly like an empty one; there is
        # no skip or state reset, only the absence of data rows for this frame.
        if len(c) < fc.min_points:
            c, ww = np.empty((0, 3)), np.empty(0)
        order: list[int] = []
        points: list[np.ndarray] = []
        conf: list[float] = []
        if landmark_frames is not None:
            fr = landmark_frames[t]
            for lm, point in fr.items():
                li = int(lm)
                if li not in hand.lm_frames or not np.all(np.isfinite(point)):
                    continue
                order.append(li); points.append(np.asarray(point, float))
                conf.append(float(landmark_confidence[t, li]) if landmark_confidence is not None else 1.0)
        out.append(FrameObservation(
            c, ww, np.asarray(order, int),
            np.asarray(points, float).reshape(-1, 3), np.clip(np.asarray(conf, float), 0.0, 1.0),
            np.zeros(len(c), dtype=int), np.ones(len(c)),
        ))
    return out


def freeze_correspondences(
    hand: "hm.CapsuleHand", q_traj: np.ndarray, observations: Sequence[FrameObservation],
    fc_inside_scale: float = 0.15,
) -> list[FrameObservation]:
    radii = hm.capsule_radii(hand)
    support_map = hm.joint_capsule_support(hand)
    n_caps = len(hand.capsules)
    frozen = []
    for q, obs in zip(q_traj, observations):
        if len(obs.cloud):
            dist, idx = nearest_capsule(obs.cloud, hm.fk_capsule_endpoints(hand, q), radii)
            scale = np.where(dist < 0.0, fc_inside_scale, 1.0)
            per_capsule = np.bincount(idx, minlength=n_caps)
        else:
            idx, scale = np.empty(0, int), np.empty(0)
            per_capsule = np.zeros(n_caps)
        # A joint measured by many samples needs no prior; an unobserved one
        # keeps the full weight. Without this the quadratic prior grows with
        # the bend it is supposed to merely regularise and straightens fingers
        # that the depth cloud has already resolved.
        support = support_map @ per_capsule
        frozen.append(replace(obs, capsule_ids=idx, data_scale=scale,
                              posture_weight=1.0 / (1.0 + support)))
    return frozen


def _component_sqrt_weights(fc: FitConfig, kind: str, nv: int) -> np.ndarray:
    if kind == "vel":
        vals = (fc.w_vel_translation, fc.w_vel_rotation, fc.w_vel_joints)
    else:
        vals = (fc.w_acc_translation, fc.w_acc_rotation, fc.w_acc_joints)
    return np.sqrt(np.r_[np.full(3, vals[0]), np.full(3, vals[1]), np.full(nv - 6, vals[2])])


def _frame_geometry(
    dtheta: np.ndarray, hand: "hm.CapsuleHand", q0: np.ndarray,
    obs: FrameObservation, fc: FitConfig, q_rest: np.ndarray, lm_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frame-local residual/Jacobian, q, dIntegrate Jacobian, wrist pos + its Jacobian."""
    import pinocchio as pin

    model, data, nv = hand.model, hand.data, hand.nv
    q = pin.integrate(model, q0, np.asarray(dtheta, float))
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    jint = np.asarray(pin.dIntegrate(model, q0, np.asarray(dtheta, float))[1])
    placements = data.oMf
    lwa = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    cache: dict[int, np.ndarray] = {}

    def frame_jac(fid: int) -> np.ndarray:
        if fid not in cache:
            cache[fid] = np.asarray(pin.getFrameJacobian(model, data, fid, lwa))[:3]
        return cache[fid]

    endpoints = np.empty((len(hand.capsules), 2, 3), float)
    for i, (fa, fb, _) in enumerate(hand.capsules):
        endpoints[i, 0] = placements[fa].translation
        endpoints[i, 1] = placements[fb].translation

    residuals: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    if len(obs.cloud):
        dist, tparam, nrm = _assigned_capsule_geom(
            obs.cloud, endpoints, hm.capsule_radii(hand), obs.capsule_ids)
        sw = np.sqrt(data_alpha(obs, fc))
        residuals.append(sw * dist)
        jd = np.zeros((len(obs.cloud), nv), float)
        for capsule_id, (fa, fb, _) in enumerate(hand.capsules):
            mask = obs.capsule_ids == capsule_id
            if not mask.any():
                continue
            tc = tparam[mask, None]; nc = nrm[mask]
            jd[mask] = -(
                (1.0 - tc) * (nc @ frame_jac(fa)) + tc * (nc @ frame_jac(fb))
            )
        jacobians.append(sw[:, None] * jd @ jint)

    if len(obs.lm_order) and fc.w_landmark > 0 and lm_scale > 0:
        slm = np.sqrt(fc.w_landmark * lm_scale * obs.lm_confidence)
        model_points = np.asarray(
            [placements[hand.lm_frames[int(i)]].translation for i in obs.lm_order]
        )
        residuals.append((slm[:, None] * (model_points - obs.lm_points)).reshape(-1))
        jlm = np.concatenate([frame_jac(hand.lm_frames[int(i)]) for i in obs.lm_order], axis=0)
        jacobians.append(np.repeat(slm, 3)[:, None] * jlm @ jint)

    qv = np.asarray(q, float)
    jq_rev = jint[6:, :]
    if fc.w_prior > 0:
        s = np.sqrt(fc.w_prior)
        over = np.maximum(0.0, qv[7:] - hand.q_hi[7:])
        under = np.maximum(0.0, hand.q_lo[7:] - qv[7:])
        residuals.extend((s * over, s * under))
        jacobians.extend((s * (over > 0)[:, None] * jq_rev,
                          -s * (under > 0)[:, None] * jq_rev))
    if fc.w_posture > 0:
        pw = obs.posture_weight
        if len(pw) != len(qv) - 7:
            pw = np.ones(len(qv) - 7)
        s = np.sqrt(fc.w_posture * pw)
        residuals.append(s * (qv[7:] - q_rest[7:]))
        jacobians.append(s[:, None] * jq_rev)

    wrist_fid = hand.lm_frames[int(LM.WRIST)]
    wrist_position = np.asarray(placements[wrist_fid].translation, float).copy()
    wrist_position_jac = frame_jac(wrist_fid) @ jint
    r = np.concatenate(residuals) if residuals else np.empty(0)
    j = np.vstack(jacobians) if jacobians else np.zeros((0, nv))
    return r, j, q, jint, wrist_position, wrist_position_jac


def batch_residual_and_jac(
    dtheta: np.ndarray, hand: "hm.CapsuleHand", q0_traj: np.ndarray,
    observations: Sequence[FrameObservation], fc: FitConfig, *,
    q_rest: np.ndarray | None = None, lm_scale: float = 1.0,
):
    """Assemble the full residual and analytic block-banded CSR Jacobian."""
    import pinocchio as pin
    from scipy.sparse import coo_matrix

    q0_traj = np.asarray(q0_traj, float)
    T, nv = len(q0_traj), hand.nv
    dt = np.asarray(dtheta, float).reshape(T, nv)
    q_rest = np.asarray(q_rest if q_rest is not None else pin.neutral(hand.model), float)
    residual_chunks: list[np.ndarray] = []
    rr: list[np.ndarray] = []; cc: list[np.ndarray] = []; vv: list[np.ndarray] = []
    row = 0
    qs: list[np.ndarray] = []; jints: list[np.ndarray] = []
    wrist_positions: list[np.ndarray] = []; wrist_jacs: list[np.ndarray] = []

    def add_block(block: np.ndarray, row0: int, col0: int) -> None:
        ri, ci = np.nonzero(block)
        if len(ri):
            rr.append(ri + row0); cc.append(ci + col0); vv.append(block[ri, ci])

    for t in range(T):
        r, j, q, jint, wrist_position, wrist_jac = _frame_geometry(
            dt[t], hand, q0_traj[t], observations[t], fc, q_rest, lm_scale
        )
        residual_chunks.append(r); add_block(j, row, t * nv); row += len(r)
        qs.append(q); jints.append(jint)
        wrist_positions.append(wrist_position); wrist_jacs.append(wrist_jac)

    sv = _component_sqrt_weights(fc, "vel", nv)
    for t in range(1, T):
        vel = np.asarray(pin.difference(hand.model, qs[t - 1], qs[t]))
        # Pinocchio expresses free-flyer translation in a moving local tangent
        # frame. Comparing those components between rotated frames creates a
        # fictitious acceleration. Use world wrist translation for the metre
        # block and manifold difference for rotation/joints.
        vel[:3] = wrist_positions[t] - wrist_positions[t - 1]
        residual_chunks.append(sv * vel)
        d0, d1 = pin.dDifference(hand.model, qs[t - 1], qs[t])
        j0 = np.asarray(d0) @ jints[t - 1]
        j1 = np.asarray(d1) @ jints[t]
        j0[:3] = -wrist_jacs[t - 1]; j1[:3] = wrist_jacs[t]
        add_block(sv[:, None] * j0, row, (t - 1) * nv)
        add_block(sv[:, None] * j1, row, t * nv)
        row += nv

    sa = _component_sqrt_weights(fc, "acc", nv)
    for t in range(2, T):
        v0 = np.asarray(pin.difference(hand.model, qs[t - 2], qs[t - 1]))
        v1 = np.asarray(pin.difference(hand.model, qs[t - 1], qs[t]))
        v0[:3] = wrist_positions[t - 1] - wrist_positions[t - 2]
        v1[:3] = wrist_positions[t] - wrist_positions[t - 1]
        residual_chunks.append(sa * (v1 - v0))
        a0, a1 = pin.dDifference(hand.model, qs[t - 2], qs[t - 1])
        b0, b1 = pin.dDifference(hand.model, qs[t - 1], qs[t])
        j2 = -np.asarray(a0) @ jints[t - 2]
        j1 = (np.asarray(b0) - np.asarray(a1)) @ jints[t - 1]
        j0 = np.asarray(b1) @ jints[t]
        j2[:3] = wrist_jacs[t - 2]
        j1[:3] = -2.0 * wrist_jacs[t - 1]
        j0[:3] = wrist_jacs[t]
        add_block(sa[:, None] * j2, row, (t - 2) * nv)
        add_block(sa[:, None] * j1, row, (t - 1) * nv)
        add_block(sa[:, None] * j0, row, t * nv)
        row += nv

    r_all = np.concatenate(residual_chunks) if residual_chunks else np.empty(0)
    rows = np.concatenate(rr) if rr else np.empty(0, int)
    cols = np.concatenate(cc) if cc else np.empty(0, int)
    vals = np.concatenate(vv) if vv else np.empty(0)
    jac = coo_matrix((vals, (rows, cols)), shape=(len(r_all), T * nv)).tocsr()
    return r_all, jac


def batch_jac_sparsity(
    hand: "hm.CapsuleHand", q0_traj: np.ndarray,
    observations: Sequence[FrameObservation], fc: FitConfig, *, q_rest=None,
    lm_scale: float = 1.0,
):
    """Boolean block-banded structure, including currently inactive barriers."""
    from scipy.sparse import lil_matrix

    T, nv = len(q0_traj), hand.nv
    frame_rows = [sum(frame_row_counts(hand, obs, fc, lm_scale)) for obs in observations]
    total_rows = sum(frame_rows) + max(T - 1, 0) * nv + max(T - 2, 0) * nv
    pattern = lil_matrix((total_rows, T * nv), dtype=bool)
    row = 0
    for t, count in enumerate(frame_rows):
        pattern[row:row + count, t * nv:(t + 1) * nv] = True
        row += count
    for t in range(1, T):
        pattern[row:row + nv, (t - 1) * nv:(t + 1) * nv] = True
        row += nv
    for t in range(2, T):
        pattern[row:row + nv, (t - 2) * nv:(t + 1) * nv] = True
        row += nv
    return pattern.tocsr()


def frame_row_counts(
    hand: "hm.CapsuleHand", obs: FrameObservation, fc: FitConfig, lm_scale: float = 1.0
) -> tuple[int, int, int, int]:
    """(data, landmark, barrier, posture) row counts emitted by ``_frame_geometry``.

    Every consumer of the residual layout derives it here. The landmark guard
    must repeat ``lm_scale > 0`` exactly: with ``landmark_decay = 0`` the anchor
    rows vanish after the first outer iteration, and a stale count would shift
    the robust-loss weights onto the wrong rows.
    """
    nrev = hand.nq - 7
    n_lm = 3 * len(obs.lm_order) if (fc.w_landmark > 0 and lm_scale > 0) else 0
    return (len(obs.cloud), n_lm,
            2 * nrev if fc.w_prior > 0 else 0,
            nrev if fc.w_posture > 0 else 0)


def _data_row_weights(
    hand: "hm.CapsuleHand", observations: Sequence[FrameObservation], fc: FitConfig,
    lm_scale: float = 1.0,
) -> np.ndarray:
    """Per-row data alpha; zero for every non-data row."""
    chunks: list[np.ndarray] = []
    for obs in observations:
        counts = frame_row_counts(hand, obs, fc, lm_scale)
        alpha = np.zeros(sum(counts), float)
        alpha[:counts[0]] = data_alpha(obs, fc)
        chunks.append(alpha)
    chunks.append(np.zeros(max(len(observations) - 1, 0) * hand.nv))
    chunks.append(np.zeros(max(len(observations) - 2, 0) * hand.nv))
    return np.concatenate(chunks)


def _data_huber_loss(alpha: np.ndarray):
    """SciPy loss callable: Huber on depth rows, exact L2 elsewhere.

    The written functional applies ``rho`` only to point→surface distances.
    Passing the string ``"huber"`` would also robustify temporal radians and
    metre translations at the same threshold, effectively turning the very
    constraints that bridge long gaps from quadratic into weak linear forces.
    """
    alpha = np.asarray(alpha, float)

    def loss(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        rho = np.empty((3, len(z)), float)
        rho[0] = z; rho[1] = 1.0; rho[2] = 0.0  # linear least squares
        # Residual rows are sqrt(alpha)*distance so their frame-wise sum of
        # squares is normalized. To keep Huber's breakpoint at |distance|=δ,
        # rather than at |sqrt(alpha)*distance|=δ, use
        # alpha * huber(z/alpha) with its exact first/second derivatives.
        robust = (alpha > 0.0) & (z > alpha)
        scaled = z[robust] / alpha[robust]
        root = np.sqrt(scaled)
        rho[0, robust] = alpha[robust] * (2.0 * root - 1.0)
        rho[1, robust] = 1.0 / root
        rho[2, robust] = -0.5 / (alpha[robust] * scaled * root)
        return rho

    return loss


ENERGY_TERMS: tuple[str, ...] = (
    "data", "landmark", "barrier", "posture",
    "vel_translation", "vel_rotation", "vel_joints",
    "acc_translation", "acc_rotation", "acc_joints",
)


def energy_split(
    hand: "hm.CapsuleHand", q_traj: np.ndarray,
    observations: Sequence[FrameObservation], fc: FitConfig, *,
    q_rest: np.ndarray, lm_scale: float = 1.0,
) -> dict[str, float]:
    """Sum of squares contributed by each term of the functional.

    Weight tuning without this is guesswork: on the target episode the depth
    cloud was contributing 0.1% of the energy while the temporal and posture
    terms carried 93%, which straightens the fingers no matter how good the
    point cloud is. ``fit_trajectory`` reports it for every episode.
    """
    import pinocchio as pin

    q_traj = np.asarray(q_traj, float)
    energy = {name: 0.0 for name in ENERGY_TERMS}
    for q, obs in zip(q_traj, observations):
        r = _frame_geometry(np.zeros(hand.nv), hand, q, obs, fc, q_rest, lm_scale)[0]
        start = 0
        for name, count in zip(("data", "landmark", "barrier", "posture"),
                               frame_row_counts(hand, obs, fc, lm_scale)):
            block = r[start:start + count]
            energy[name] += float(block @ block)
            start += count

    wrist = np.asarray([wrist_pose(hand, q)[0] for q in q_traj])
    diff = [np.asarray(pin.difference(hand.model, q_traj[t - 1], q_traj[t]))
            for t in range(1, len(q_traj))]
    for t, v in enumerate(diff):
        v[:3] = wrist[t + 1] - wrist[t]
    blocks = {"vel": np.asarray(diff) if diff else np.zeros((0, hand.nv)),
              "acc": (np.diff(np.asarray(diff), axis=0) if len(diff) > 1
                      else np.zeros((0, hand.nv)))}
    for kind, rows in blocks.items():
        scaled = _component_sqrt_weights(fc, kind, hand.nv) * rows
        for name, sl in (("translation", slice(0, 3)), ("rotation", slice(3, 6)),
                         ("joints", slice(6, None))):
            energy[f"{kind}_{name}"] = float(np.sum(scaled[:, sl] ** 2))

    total = sum(energy.values())
    out = {f"energy_{k}": v for k, v in energy.items()}
    out["energy_total"] = total
    out.update({f"energy_frac_{k}": (v / total if total > 0 else 0.0)
                for k, v in energy.items()})
    return out


def _bounds(hand: "hm.CapsuleHand", q0_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, nv = len(q0_traj), hand.nv
    lo = np.empty((T, nv)); hi = np.empty((T, nv))
    lo[:, :6] = [-0.15, -0.15, -0.15, -1.0, -1.0, -1.0]
    hi[:, :6] = -lo[:, :6]
    lo[:, 6:] = np.minimum(hand.q_lo[7:] - q0_traj[:, 7:], -1e-7)
    hi[:, 6:] = np.maximum(hand.q_hi[7:] - q0_traj[:, 7:], 1e-7)
    return lo.ravel(), hi.ravel()


def _jerk_norm(q_traj: np.ndarray) -> float:
    p = np.asarray(q_traj, float)[:, :3]
    if len(p) < 4:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.diff(p, n=3, axis=0) ** 2, axis=1))))


def fit_trajectory(
    hand: "hm.CapsuleHand", clouds: Sequence[np.ndarray],
    weights: Sequence[np.ndarray | None], q_init: np.ndarray, fc: FitConfig,
    *, landmark_frames: Sequence[Mapping[LM, np.ndarray]] | None = None,
    landmark_confidence: np.ndarray | None = None, q_rest: np.ndarray | None = None,
    report=None,
) -> tuple[np.ndarray, dict]:
    """Full-episode batch ICP. Empty frames remain variables, never skips."""
    import pinocchio as pin
    from scipy.optimize import least_squares

    started = time.perf_counter()
    report = report or (lambda **_k: None)
    q_ref = np.asarray(q_init, float).copy()
    initial_jerk = _jerk_norm(q_ref)
    T, nv = q_ref.shape[0], hand.nv
    if T < 2:
        raise ValueError("hand-fit batch requires at least two frames")
    observations = _make_observations(
        clouds, weights, landmark_frames, landmark_confidence, hand, fc
    )
    supported = np.asarray([
        len(obs.cloud) > 0
        or (fc.w_landmark > 0 and np.any(obs.lm_confidence > 0))
        for obs in observations
    ])
    q_ref = _fill_invalid_initialization(hand, q_ref, supported)
    q_rest = np.asarray(q_rest if q_rest is not None else pin.neutral(hand.model), float)
    n_outer = 0; total_nfev = 0
    for outer in range(fc.outer_iterations):
        lm_scale = fc.landmark_decay ** outer
        frozen = freeze_correspondences(hand, q_ref, observations, fc.inside_scale)
        cache: dict[bytes, tuple[np.ndarray, object]] = {}

        def evaluate(x):
            if fc.deadline_s and time.perf_counter() - started > fc.deadline_s:
                raise _FitDeadline
            key = np.asarray(x, float).tobytes()
            if key not in cache:
                if len(cache) > 3:
                    cache.clear()
                cache[key] = batch_residual_and_jac(
                    x, hand, q_ref, frozen, fc, q_rest=q_rest, lm_scale=lm_scale
                )
            return cache[key]

        lower, upper = _bounds(hand, q_ref)
        # A callable sparse Jacobian is authoritative; jac_sparsity documents
        # and protects the intended band if scipy internally finite-differences.
        sparsity = batch_jac_sparsity(hand, q_ref, frozen, fc, q_rest=q_rest,
                                      lm_scale=lm_scale)
        robust_data_weights = _data_row_weights(hand, frozen, fc, lm_scale)
        try:
            result = least_squares(
                lambda x: evaluate(x)[0], np.zeros(T * nv),
                jac=lambda x: evaluate(x)[1], jac_sparsity=sparsity,
                bounds=(lower, upper), method="trf", tr_solver="lsmr",
                loss=_data_huber_loss(robust_data_weights),
                f_scale=fc.huber_delta_m, x_scale="jac",
                max_nfev=fc.max_nfev,
            )
        except _FitDeadline:
            logger.warning(
                "hand_fit: %.0fs deadline hit at outer %d/%d — using the last iterate",
                fc.deadline_s, outer + 1, fc.outer_iterations,
            )
            break
        step = result.x.reshape(T, nv)
        q_new = np.asarray([pin.integrate(hand.model, q_ref[t], step[t]) for t in range(T)])
        step_norm = float(np.sqrt(np.mean(step ** 2)))
        q_ref = q_new; n_outer = outer + 1; total_nfev += int(result.nfev)
        report(stage="hand_fit", frame=n_outer, total=fc.outer_iterations)
        if step_norm < fc.outer_step_tol:
            break

    info = _final_metrics(
        hand, q_ref, observations, fc, q_rest=q_rest, initial_jerk=initial_jerk,
        n_outer=n_outer, total_nfev=total_nfev, elapsed_s=time.perf_counter() - started,
    )
    return q_ref, info


def _final_metrics(
    hand: "hm.CapsuleHand", q_traj: np.ndarray, observations, fc: FitConfig, *,
    q_rest: np.ndarray, initial_jerk: float, n_outer: int, total_nfev: int,
    elapsed_s: float,
) -> dict:
    """Residual / jerk / energy-split summary at the converged trajectory. Shared
    by the whole-episode and the sliding-window drivers so both report the same
    keys computed the same way."""
    radii = hm.capsule_radii(hand)
    residuals = []
    for q, obs in zip(q_traj, observations):
        if len(obs.cloud):
            d, _ = nearest_capsule(obs.cloud, hm.fk_capsule_endpoints(hand, q), radii)
            residuals.append(np.abs(d))
    all_resid = np.concatenate(residuals) if residuals else np.empty(0)
    counts = np.asarray([len(o.cloud) for o in observations], float)
    info = {
        "median_resid_m": float(np.median(all_resid)) if len(all_resid) else float("nan"),
        "p90_resid_m": float(np.percentile(all_resid, 90)) if len(all_resid) else float("nan"),
        "jerk_before_m": initial_jerk,
        "jerk_after_m": _jerk_norm(q_traj),
        "empty_frame_fraction": float(np.mean(counts == 0)) if len(counts) else 0.0,
        "mean_roi_points": float(counts.mean()) if len(counts) else 0.0,
        "outer_iterations": int(n_outer),
        "nfev": int(total_nfev),
        "elapsed_s": float(elapsed_s),
    }
    info.update(energy_split(
        hand, q_traj, freeze_correspondences(hand, q_traj, observations, fc.inside_scale),
        fc, q_rest=q_rest, lm_scale=fc.landmark_decay ** max(n_outer - 1, 0),
    ))
    return info


def _window_starts(T: int, W: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) spans covering [0, T) with `overlap` shared frames. The last
    span is pulled back so it is a full W (no runt window) when T >= W."""
    if T <= W:
        return [(0, T)]
    hop = max(1, W - overlap)
    starts = list(range(0, T - W + 1, hop))
    if starts[-1] != T - W:
        starts.append(T - W)
    return [(s, s + W) for s in starts]


def _blend_ramp(n: int) -> np.ndarray:
    """Raised-cosine 0->1 over n samples (endpoints excluded so both windows keep
    full authority at their own edge)."""
    if n <= 1:
        return np.ones(max(n, 0))
    i = np.arange(1, n + 1) / (n + 1)
    return 0.5 - 0.5 * np.cos(np.pi * i)


def fit_trajectory_windowed(
    hand: "hm.CapsuleHand", clouds: Sequence[np.ndarray],
    weights: Sequence[np.ndarray | None], q_init: np.ndarray, fc: FitConfig,
    *, landmark_frames: Sequence[Mapping[LM, np.ndarray]] | None = None,
    landmark_confidence: np.ndarray | None = None, q_rest: np.ndarray | None = None,
    report=None,
) -> tuple[np.ndarray, dict]:
    """Solve the episode as overlapping sliding windows, blended on the manifold.

    Bounds the per-solve cost regardless of recording length, and the windows are
    independent so they run on a thread pool. ``fc.window <= 0`` or a short
    episode falls straight through to :func:`fit_trajectory` (unchanged).
    """
    import os
    import pinocchio as pin

    report = report or (lambda **_k: None)
    q_init = np.asarray(q_init, float)
    T = len(q_init)
    W, ov = int(fc.window), max(0, int(fc.window_overlap))
    if W <= 0 or T <= W or T < 2:
        return fit_trajectory(
            hand, clouds, weights, q_init, fc, landmark_frames=landmark_frames,
            landmark_confidence=landmark_confidence, q_rest=q_rest, report=report,
        )

    started = time.perf_counter()
    initial_jerk = _jerk_norm(q_init)
    spans = _window_starts(T, W, ov)

    def solve(span):
        s, e = span
        lf = landmark_frames[s:e] if landmark_frames is not None else None
        lc = landmark_confidence[s:e] if landmark_confidence is not None else None
        # windows solve independently from the (already sanitised) warm start;
        # the overlap blend carries continuity across the seam.
        return fit_trajectory(
            hand, list(clouds[s:e]), list(weights[s:e]), q_init[s:e], fc,
            landmark_frames=lf, landmark_confidence=lc, q_rest=q_rest,
        )

    # Parallel windows only when BLAS can be held to one thread per worker — else
    # N workers × an unpinned OpenBLAS pegs every core (env vars are read at
    # import, too late to set here). threadpoolctl does it at runtime via the
    # BLAS C API; without it, run sequential.
    try:
        from threadpoolctl import threadpool_limits
    except Exception:  # noqa: BLE001
        threadpool_limits = None

    want = fc.workers or min(4, max(1, (os.cpu_count() or 2) // 2))
    n_workers = want if (threadpool_limits is not None and len(spans) > 1) else 1
    if n_workers == 1 and want > 1 and threadpool_limits is None:
        logger.warning("hand_fit: threadpoolctl missing — window solves run sequentially")

    limits = threadpool_limits(limits=1) if threadpool_limits is not None else None
    try:
        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(n_workers, len(spans))) as ex:
                win = list(ex.map(solve, spans))
        else:
            win = [solve(sp) for sp in spans]
    finally:
        if limits is not None:
            limits.unregister()
    for k, _ in enumerate(spans):
        report(stage="hand_fit", frame=k + 1, total=len(spans))

    # stitch on the manifold: verbatim outside overlaps, raised-cosine blend inside
    nq = q_init.shape[1]
    q_out = np.empty((T, nq), float)
    (s0, e0), (q0, _) = spans[0], win[0]
    q_out[s0:e0] = q0
    for k in range(1, len(spans)):
        (sp, ep), (qp, _) = spans[k - 1], win[k - 1]
        (sc, ec), (qc, _) = spans[k], win[k]
        lap = ep - sc                     # overlapping frame count
        if lap > 0:
            r = _blend_ramp(lap)
            for i in range(lap):
                q_out[sc + i] = pin.interpolate(hand.model, q_out[sc + i], qc[i], r[i])
            q_out[ep:ec] = qc[lap:]
        else:
            q_out[sc:ec] = qc

    observations = _make_observations(
        clouds, weights, landmark_frames, landmark_confidence, hand, fc
    )
    q_rest = np.asarray(q_rest if q_rest is not None else pin.neutral(hand.model), float)
    info = _final_metrics(
        hand, q_out, observations, fc, q_rest=q_rest, initial_jerk=initial_jerk,
        n_outer=max(int(i["outer_iterations"]) for _, i in win),
        total_nfev=sum(int(i["nfev"]) for _, i in win),
        elapsed_s=time.perf_counter() - started,
    )
    info.update({
        "n_windows": len(spans),
        "window": W,
        "window_overlap": ov,
        "workers": int(n_workers),
        "worst_window_nfev": max(int(i["nfev"]) for _, i in win),
        "worst_window_elapsed_s": max(float(i["elapsed_s"]) for _, i in win),
    })
    return q_out, info


# Single-frame compatibility helpers retain the proven analytic geometry test;
# trajectory code never calls these greedily.
def residual_and_jac(
    dtheta, hand, q0, cloud, weights, fc, *, q_rest=None, order=None,
    obs_lm=None,
):
    import pinocchio as pin

    order = np.asarray(order or [], int)
    cloud = np.asarray(cloud, float)
    obs = FrameObservation(
        cloud, np.asarray(weights, float), order,
        np.asarray(obs_lm if obs_lm is not None else [], float).reshape(-1, 3),
        np.ones(len(order)), np.zeros(len(cloud), int), np.ones(len(cloud)),
    )
    q_now = pin.integrate(hand.model, q0, np.asarray(dtheta, float))
    if len(cloud):
        dist, idx = nearest_capsule(cloud, hm.fk_capsule_endpoints(hand, q_now), hm.capsule_radii(hand))
        obs.capsule_ids = idx
        obs.data_scale = np.where(dist < 0.0, fc.inside_scale, 1.0)
    return _frame_geometry(
        dtheta, hand, q0, obs, fc,
        np.asarray(q_rest if q_rest is not None else pin.neutral(hand.model)), 1.0,
    )[:2]


def assemble_residuals(
    q, hand, cloud, weights, fc, *, q_rest=None, lm_anchor=None,
):
    order = [int(lm) for lm, p in (lm_anchor or {}).items()
             if int(lm) in hand.lm_frames and np.all(np.isfinite(p))]
    obs_lm = np.asarray([lm_anchor[LM(i)] for i in order]) if order else None
    return residual_and_jac(
        np.zeros(hand.nv), hand, q, cloud, weights, fc, q_rest=q_rest,
        order=order, obs_lm=obs_lm,
    )[0]


def fit_frame(hand, cloud, weights, q0, fc, **kwargs):
    """Compatibility fit for geometry tests; production uses ``fit_trajectory``."""
    q, info = fit_trajectory(
        hand, [cloud, cloud], [weights, weights], np.stack([q0, q0]),
        replace(fc, outer_iterations=1),
    )
    d, _ = nearest_capsule(cloud, hm.fk_capsule_endpoints(hand, q[0]), hm.capsule_radii(hand))
    return q[0], {"skipped": len(cloud) < fc.min_points, "accepted": True,
                  "median_resid": float(np.median(np.abs(d))) if len(d) else float("nan"),
                  "n_points": len(cloud), "nfev": info["nfev"]}


def wrist_pose(hand: "hm.CapsuleHand", q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import pinocchio as pin

    pose = pin.XYZQUATToSE3(np.asarray(q, float)[:7])
    return np.asarray(pose.translation, float), np.asarray(pose.rotation, float)


# ── raw depth extraction ─────────────────────────────────────────────────


def _cameras(raw: Path, meta: dict, preset: str | None, bg_subtract: bool):
    from viki.contracts import CalibrationExtrinsics
    from viki.perception.k4a_offline import K4ACalibration
    from viki.perception.rs_offline import RealSenseCalibration

    extr = json.loads((raw / "extrinsics.json").read_text()) if (raw / "extrinsics.json").exists() else {}
    bg_by_dev: dict = {}
    if bg_subtract and preset:
        try:
            from viki.calibration import presets as presets
            for mp4 in raw.glob("*.mp4"):
                bg = presets.background_depth(preset, mp4.stem)
                if bg is not None:
                    bg_by_dev[mp4.stem] = bg
        except Exception as exc:  # noqa: BLE001
            logger.warning("hand_fit: background load failed (%s)", exc)
    cams = []
    for mp4 in sorted(raw.glob("*.mp4")):
        dev = mp4.stem; extrinsic = extr.get(dev)
        if not extrinsic:
            continue
        transform = CalibrationExtrinsics(
            rvec=np.asarray(extrinsic["rvec"], float),
            tvec=np.asarray(extrinsic["tvec"], float),
        ).transform_matrix
        calibration = (K4ACalibration.from_episode(raw, dev, meta)
                       or RealSenseCalibration.from_episode(raw, dev, meta))
        if calibration is not None:
            cams.append({"dev": dev, "cal": calibration, "T": transform,
                         "depth_dir": raw / f"{dev}_depth", "bg": bg_by_dev.get(dev)})
    return cams


def _adaptive_capsule_mask(
    points: np.ndarray, endpoints: np.ndarray, radii: np.ndarray, margin_m: float,
) -> np.ndarray:
    if not len(points):
        return np.zeros(0, bool)
    # Union membership is not the same as nearest *surface*: a point deep
    # inside one primitive may have another surface closer in absolute value.
    # Accumulate capsule half-spaces one at a time to avoid an N×C×3 tensor for
    # full-resolution depth frames.
    mask = np.zeros(len(points), bool)
    for (a, b), radius in zip(endpoints, radii):
        mask |= point_segment_distance(points, a, b) <= float(radius) + margin_m
    return mask


def hand_roi_cloud(
    cams: list[dict], frame_i: int, endpoints: np.ndarray, radii: np.ndarray,
    wrist_world: np.ndarray, palm_forward_world: np.ndarray, margin_m: float,
    forearm_cut_m: float, bg_tol_mm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive capsule-union ROI with a proximal wrist/forearm half-space cut.

    A detector mask is not stored in current episodes, so it cannot be applied
    offline; background subtraction remains the additional segmentation cue.
    """
    xyz_parts: list[np.ndarray] = []; wt_parts: list[np.ndarray] = []
    forward = np.asarray(palm_forward_world, float)
    forward /= np.linalg.norm(forward) + 1e-12
    plane = np.asarray(wrist_world, float) - forearm_cut_m * forward
    for camera in cams:
        depth_path = camera["depth_dir"] / f"{frame_i:06d}.npy"
        if not depth_path.is_file():
            continue
        depth_mm = np.load(depth_path)
        if not depth_mm.any():
            continue
        h, w = depth_mm.shape[:2]
        A, B = camera["cal"].color_deproject_maps(h, w)
        z = depth_mm.astype(np.float64)
        keep = z > 0
        bg = camera["bg"]
        if bg is not None and bg.shape == depth_mm.shape:
            keep &= ~((bg > 0) & (np.abs(z - bg) <= float(bg_tol_mm)))
        vs, us = np.nonzero(keep)
        if not len(us):
            continue
        points_mm = z[vs, us, None] * A[vs, us] + B[vs, us]
        finite = np.isfinite(points_mm).all(axis=1)
        points_cam = points_mm[finite] / 1000.0
        world = points_cam @ camera["T"][:3, :3].T + camera["T"][:3, 3]
        mask = _adaptive_capsule_mask(world, endpoints, radii, margin_m)
        mask &= (world - plane) @ forward >= 0.0
        if mask.any():
            xyz_parts.append(world[mask])
            ranges = np.linalg.norm(points_cam[mask], axis=1)
            wt_parts.append(1.0 / np.maximum(ranges, 0.1) ** 2)
    if not xyz_parts:
        return np.empty((0, 3)), np.empty(0)
    xyz = np.concatenate(xyz_parts); wt = np.concatenate(wt_parts)
    return xyz, wt / (wt.mean() + 1e-12)


def _spread(points: Mapping[LM, np.ndarray]) -> float:
    tips = [points.get(lm) for lm in (LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP,
                                      LM.RING_TIP, LM.PINKY_TIP)]
    tips = [np.asarray(p, float) for p in tips if p is not None and np.all(np.isfinite(p))]
    if len(tips) < 3:
        return -1.0
    return float(np.mean([np.linalg.norm(tips[i] - tips[j]) for i in range(len(tips))
                          for j in range(i + 1, len(tips))]))


def _calibration_frame_indices(
    frames: list[Mapping[LM, np.ndarray]], valid: np.ndarray, count: int
) -> np.ndarray:
    """Choose open, geometrically plausible frames for hand calibration.

    Merely taking the frames with the largest fingertip spread strongly favours
    landmark glitches: one displaced MCP or fingertip looks like an exceptionally
    open hand and permanently distorts the calibrated palm.  First reject frames
    whose palm width or individual bone lengths disagree with the episode's
    robust median, then take the most open survivors.
    """
    requested = max(1, int(count))
    candidate = np.flatnonzero(np.asarray(valid, bool))
    if not len(candidate):
        candidate = np.arange(len(frames))

    rows: list[tuple[int, float, float, np.ndarray]] = []
    for t in candidate:
        fr = frames[int(t)]
        spread = _spread(fr)
        index = fr.get(LM.INDEX_MCP)
        pinky = fr.get(LM.PINKY_MCP)
        if spread <= 0 or index is None or pinky is None:
            continue
        palm_width = float(np.linalg.norm(np.asarray(index, float) - np.asarray(pinky, float)))
        lengths = []
        for lms in hm.FINGERS.values():
            pp = [fr.get(lm) for lm in lms]
            if any(p is None or not np.all(np.isfinite(p)) for p in pp):
                lengths = []
                break
            lengths.extend(np.linalg.norm(np.diff(np.asarray(pp, float), axis=0), axis=1))
        lengths = np.asarray(lengths, float)
        if palm_width > 1e-4 and lengths.shape == (15,) and np.all(lengths > 1e-4):
            rows.append((int(t), spread, palm_width, lengths))

    if not rows:
        ranked = sorted((int(t) for t in candidate), key=lambda t: _spread(frames[t]), reverse=True)
        return np.asarray(ranked[:requested], dtype=int)

    palm = np.asarray([row[2] for row in rows])
    bones = np.stack([row[3] for row in rows])
    spreads = np.asarray([row[1] for row in rows])
    palm_mid = max(float(np.median(palm)), 1e-6)
    bone_mid = np.maximum(np.median(bones, axis=0), 1e-6)
    palm_log_error = np.abs(np.log(palm / palm_mid))
    bone_log_error = np.abs(np.log(bones / bone_mid))

    # These multiplicative limits tolerate real articulation/depth noise while
    # rejecting the 1.5–2x palm and phalanx jumps seen in failed detections.
    plausible = (
        (palm_log_error <= np.log(1.30))
        & (np.median(bone_log_error, axis=1) <= np.log(1.22))
        & (np.max(bone_log_error, axis=1) <= np.log(1.60))
    )
    # A single bad fingertip can also manufacture a huge "spread" without
    # changing palm width. Cap the open-hand score with a robust upper fence.
    spread_mid = float(np.median(spreads))
    spread_mad = float(np.median(np.abs(spreads - spread_mid)))
    if spread_mad > 1e-9:
        plausible &= spreads <= spread_mid + 4.0 * 1.4826 * spread_mad

    kept = [rows[i] for i in np.flatnonzero(plausible)]
    if len(kept) < min(requested, len(rows)):
        # Prefer a larger sample over an over-strict bone filter, but retain the
        # palm-width guard which catches the destructive MCP outliers.
        kept = [row for row, err in zip(rows, palm_log_error) if err <= np.log(1.30)]
    if not kept:
        kept = rows
    kept.sort(key=lambda row: row[1], reverse=True)
    return np.asarray([row[0] for row in kept[:requested]], dtype=int)


def _fill_invalid_initialization(
    hand: "hm.CapsuleHand", q_traj: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Replace invalid warm starts by manifold interpolation/extrapolation.

    Prepared landmark splines can explode across a long unsupported gap. Those
    values must not become the batch reference (bounded outer increments could
    need hundreds of iterations to undo them). Interior gaps interpolate their
    valid neighbours; leading/trailing gaps hold the nearest valid pose. The
    frames remain unknowns in the solve and temporal terms can still move them.
    """
    import pinocchio as pin

    out = np.asarray(q_traj, float).copy()
    valid = np.asarray(valid, bool)
    good = np.flatnonzero(valid)
    if not len(good):
        return out
    out[:good[0]] = out[good[0]]
    out[good[-1] + 1:] = out[good[-1]]
    for left, right in zip(good[:-1], good[1:]):
        if right == left + 1:
            continue
        width = right - left
        for t in range(left + 1, right):
            out[t] = pin.interpolate(hand.model, out[left], out[right], (t - left) / width)
    return out


def _sanitize_warm_start(
    hand: "hm.CapsuleHand", q_traj: np.ndarray, valid: np.ndarray, mad_k: float = 6.0,
    hard_jump_m: float = 0.25,
) -> np.ndarray:
    """Demote ``valid`` frames whose warm-start wrist position is a local outlier.

    A frame can pass the stability mask yet carry a noisy ``q_from_landmarks``
    pose whose wrist is metres off its neighbours. ``_fill_invalid_initialization``
    only touches ``valid=False`` frames, so those spikes seed the solve and — with
    the ±0.15 m per-iteration wrist bound — survive into the fit. Mark them
    not-valid so they get re-filled by manifold interpolation from clean
    neighbours; they stay free variables in the solve.
    """
    q_traj = np.asarray(q_traj, float)
    valid = np.asarray(valid, bool).copy()
    if valid.sum() < 4:
        return valid
    p = q_traj[:, :3]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)  # (T-1,)
    med = np.median(step[np.isfinite(step)]) if np.isfinite(step).any() else 0.0
    mad = np.median(np.abs(step - med)) + 1e-9
    thr = max(med + mad_k * mad, hard_jump_m)
    # a frame is a spike if both the step into it and out of it exceed thr
    spike = np.zeros(len(q_traj), bool)
    spike[1:-1] = (step[:-1] > thr) & (step[1:] > thr)
    spike[0] = len(step) and step[0] > thr and step[min(1, len(step) - 1)] > thr
    spike[-1] = len(step) and step[-1] > thr
    n = int((spike & valid).sum())
    if n:
        logger.info("hand_fit: %d valid frame(s) demoted as warm-start wrist spikes", n)
    valid[spike] = False
    return valid


def refine_cln(ep, cfg=None, report=None) -> str:
    """Fit a complete prepared episode and append new, non-destructive keys."""
    report = report or (lambda **_k: None)
    fc = FitConfig.from_config(cfg)
    cln_path = Path(ep.cln_npz)
    with np.load(cln_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    positions = np.asarray(data["positions"], float)
    valid = np.asarray(data["valid"], bool)
    smoothed = np.asarray(data["smoothed_points"], float)
    landmark_ids = np.asarray(data["landmark_ids"], int)
    T = len(positions)
    if T < 2:
        return str(cln_path)
    frames = [{LM(int(landmark_ids[j])): smoothed[t, j] for j in range(smoothed.shape[1])}
              for t in range(T)]

    raw = Path(ep.raw_dir)
    meta = json.loads(Path(ep.meta_path).read_text()) if Path(ep.meta_path).exists() else {}
    from viki import config as global_cfg
    active_cfg = cfg or global_cfg
    cams = _cameras(raw, meta, meta.get("calibration_preset"),
                    bool(getattr(active_cfg, "CLOUD_BG_SUBTRACT", True)))
    if not cams:
        logger.warning("hand_fit %s: no camera with usable depth; cln unchanged", ep.id)
        return str(cln_path)

    calibration_indices = _calibration_frame_indices(frames, valid, fc.calib_frames)
    calibration_frames = [frames[t] for t in calibration_indices]
    try:
        hand = hm.build(hm.calibrate_from_frames(calibration_frames or frames))
    except Exception as exc:  # noqa: BLE001
        logger.warning("hand_fit %s: model calibration failed (%s)", ep.id, exc)
        return str(cln_path)

    q_landmark = np.asarray([hm.q_from_landmarks(hand, frame) for frame in frames])
    landmark_jerk = _jerk_norm(q_landmark)
    # A `valid` frame whose landmark warm start puts the wrist metres off its
    # neighbours is a spike source — demote it for the warm start and drop its
    # landmark anchor. Depth for that frame is still used (ROI built below from
    # the interpolated pose); it stays a free variable in the solve.
    valid_ws = _sanitize_warm_start(hand, q_landmark, valid, fc.warm_start_mad_k)
    q_init = _fill_invalid_initialization(hand, q_landmark, valid_ws)
    # Calibrated rest is the robust median finger posture of the open frames.
    q_rest = q_init[calibration_indices[0]].copy()
    if len(calibration_frames):
        q_cal = np.asarray([hm.q_from_landmarks(hand, frames[t]) for t in calibration_indices])
        q_rest[7:] = np.median(q_cal[:, 7:], axis=0)

    clouds: list[np.ndarray] = []; weights: list[np.ndarray] = []
    radii = hm.capsule_radii(hand)
    for t, q in enumerate(q_init):
        if not valid[t]:
            clouds.append(np.empty((0, 3))); weights.append(np.empty(0))
            continue
        endpoints = hm.fk_capsule_endpoints(hand, q)
        wrist, rotation = wrist_pose(hand, q)
        cloud, weight = hand_roi_cloud(
            cams, t, endpoints, radii, wrist, rotation[:, 0], fc.roi_margin_m,
            fc.forearm_cut_m, float(getattr(active_cfg, "CLOUD_BG_TOLERANCE_MM", 50.0)),
        )
        clouds.append(cloud); weights.append(weight)
        if t % 10 == 0:
            report(stage="hand_fit_cloud", frame=t, total=T)

    confidence = data.get("landmark_confidence")
    if confidence is not None:
        confidence = np.asarray(confidence, float)
        # Stored confidence follows landmark column order, while observation
        # construction indexes by canonical landmark id.
        canonical = np.zeros((T, 21), float)
        canonical[:, landmark_ids] = confidence
        canonical[~valid_ws] = 0.0
        confidence = canonical
    else:
        # Historical cln files only have per-frame omega. It is a less precise
        # fallback, but still avoids treating invalid frames as observations.
        frame_conf = np.asarray(data.get("omega", np.ones(T)), float)
        confidence = np.repeat(np.clip(frame_conf, 0.0, 1.0)[:, None], 21, axis=1)
        confidence[~valid_ws] = 0.0
    q_fit, info = fit_trajectory_windowed(
        hand, clouds, weights, q_init, fc, landmark_frames=frames,
        landmark_confidence=confidence, q_rest=q_rest, report=report,
    )
    info["jerk_before_m"] = landmark_jerk
    fit_positions = np.empty((T, 3), np.float32)
    fit_rotations = np.empty((T, 3, 3), np.float32)
    capsules = np.empty((T, len(hand.capsules), 2, 3), np.float32)
    for t, q in enumerate(q_fit):
        p, R = wrist_pose(hand, q)
        fit_positions[t] = p; fit_rotations[t] = R
        capsules[t] = hm.fk_capsule_endpoints(hand, q)

    # Never overwrite landmark-derived pose arrays: repeated runs always fit
    # the same input and A/B comparison remains possible.
    data.update({
        "hand_fit_positions": fit_positions,
        "hand_fit_rotations": fit_rotations,
        "hand_fit_joint_angles": q_fit.astype(np.float32),
        "hand_fit_model_nq": np.int64(hand.nq),
        "hand_fit_capsules": capsules,
        "hand_fit_capsule_radii": radii.astype(np.float32),
        "hand_fit_metrics_json": np.asarray(json.dumps(info, sort_keys=True)),
    })
    np.savez_compressed(cln_path, **data)
    logger.info(
        "hand_fit %s: median=%.4fm p90=%.4fm jerk %.6g→%.6g empty=%.1f%% "
        "roi_pts=%.0f outer=%d elapsed=%.2fs", ep.id, info["median_resid_m"],
        info["p90_resid_m"], info["jerk_before_m"], info["jerk_after_m"],
        100 * info["empty_frame_fraction"], info["mean_roi_points"],
        info["outer_iterations"], info["elapsed_s"],
    )
    logger.info(
        "hand_fit %s: energy %s", ep.id,
        "  ".join("%s=%.1f%%" % (name, 100 * info["energy_frac_" + name])
                  for name in ENERGY_TERMS),
    )
    return str(cln_path)
