"""Landmark-only articulated hand fitting for prepared episodes.

This module deliberately does *not* consume the dense RGB-D point cloud.  The
validated clean pipeline already provides useful world-space stereo anchors;
the first anatomical experiment should answer one narrow question: can a
fixed-size articulated hand remove impossible skeleton shapes without moving
the wrist trajectory that was visually validated?

Two non-destructive variants are emitted:

``40_projected.npz``
    The clean landmarks converted to joint angles, with unsupported spans
    interpolated on the hand configuration manifold, then projected through
    forward kinematics.

``50_optimized.npz``
    The projected trajectory refined against confidence-weighted 3-D
    landmarks with temporal velocity/acceleration terms.  Dense depth, object
    pixels, and background pixels cannot influence this solve.

Both variants keep the clean wrist translation exactly.  They live beside the
protected clean baseline, so running this experiment never replaces
``episode/cln.npz``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from viki.contracts import HAND_LM_COUNT, LM
from viki.perception import hand_model as hm


ARTICULATED_LANDMARKS_V1 = "articulated-landmarks-v1"

# Fixed geometry used for both robust anchor gating and structural reporting.
# Finger chains are followed by a redundant palm scaffold.  The latter is
# rigid in the articulated model and catches a folded/collapsed palm even when
# the MediaPipe drawing topology would not.
_FINGER_EDGES = tuple(
    (int(chain[i]), int(chain[i + 1]))
    for chain in hm.FINGERS.values()
    for i in range(3)
)
_PALM_EDGES = (
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (5, 9), (9, 13), (13, 17), (5, 17),
)
_GEOMETRY_EDGES = _FINGER_EDGES + _PALM_EDGES
_PALM_IDS = np.asarray([0, 5, 9, 13, 17], dtype=int)
# ``compute_palm_rotation`` specifically consumes this quartet.
_PALM_POSE_IDS = np.asarray([0, 5, 9, 17], dtype=int)


@dataclass(frozen=True)
class ArticulatedConfig:
    """Versioned recipe for the first landmark-only geometry experiment."""

    name: str = ARTICULATED_LANDMARKS_V1
    source_profile: str = "clean-triangulated-landmarks-v1"
    calibration_frames: int = 8
    min_anchor_confidence: float = 0.02
    edge_soft_ratio_min: float = 0.70
    edge_soft_ratio_max: float = 1.45
    edge_hard_ratio_min: float = 0.50
    edge_hard_ratio_max: float = 1.90
    min_supported_joints: int = 8
    min_supported_palm_joints: int = 4
    w_landmark: float = 750.0
    w_vel_translation: float = 40.0
    w_vel_rotation: float = 30.0
    w_vel_joints: float = 3.0
    w_acc_translation: float = 120.0
    w_acc_rotation: float = 150.0
    w_acc_joints: float = 15.0
    w_posture: float = 0.0005
    max_nfev: int = 20
    outer_iterations: int = 2
    window: int = 120
    window_overlap: int = 30
    workers: int = 0
    deadline_s: float = 120.0

    def manifest(self) -> dict[str, object]:
        return asdict(self)


def _canonical(
    points: np.ndarray, landmark_ids: np.ndarray, fill: float = np.nan,
) -> np.ndarray:
    """Return ``(T, 21, ...)`` in canonical landmark-id order."""
    points = np.asarray(points)
    ids = np.asarray(landmark_ids, int)
    shape = (len(points), HAND_LM_COUNT, *points.shape[2:])
    out = np.full(shape, fill, dtype=points.dtype)
    for column, landmark_id in enumerate(ids):
        if 0 <= int(landmark_id) < HAND_LM_COUNT:
            out[:, int(landmark_id)] = points[:, column]
    return out


def _frames(points: np.ndarray) -> list[Mapping[LM, np.ndarray]]:
    return [
        {LM(i): points[t, i] for i in range(HAND_LM_COUNT)}
        for t in range(len(points))
    ]


def _edge_lengths(points: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.linalg.norm(points[:, a] - points[:, b], axis=1)
         for a, b in _GEOMETRY_EDGES],
        axis=1,
    )


def _reference_edge_lengths(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    lengths = _edge_lengths(points)
    refs = np.empty(len(_GEOMETRY_EDGES), dtype=np.float64)
    for edge_index, (a, b) in enumerate(_GEOMETRY_EDGES):
        usable = (
            np.isfinite(lengths[:, edge_index])
            & (confidence[:, a] > 0.0)
            & (confidence[:, b] > 0.0)
        )
        values = lengths[usable, edge_index]
        if not len(values):
            values = lengths[np.isfinite(lengths[:, edge_index]), edge_index]
        refs[edge_index] = float(np.median(values)) if len(values) else np.nan
    return refs


def geometry_anchor_confidence(
    points: np.ndarray, confidence: np.ndarray, cfg: ArticulatedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Attenuate landmark anchors that imply implausible episode-relative bones.

    A bad internal landmark normally damages both adjacent edges, whereas each
    correct neighbour still has another sound incident edge.  Taking the best
    incident-edge score therefore localises the bad joint instead of discarding
    a whole finger.  Tips have one incident edge and are rejected directly.
    """
    points = np.asarray(points, float)
    raw = np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)
    refs = _reference_edge_lengths(points, raw)
    lengths = _edge_lengths(points)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lengths / refs[None, :]

    lo0, lo1 = cfg.edge_hard_ratio_min, cfg.edge_soft_ratio_min
    hi1, hi0 = cfg.edge_soft_ratio_max, cfg.edge_hard_ratio_max
    edge_score = np.ones_like(ratio)
    edge_score[~np.isfinite(ratio)] = 0.0
    low = (ratio > lo0) & (ratio < lo1)
    high = (ratio > hi1) & (ratio < hi0)
    edge_score[ratio <= lo0] = 0.0
    edge_score[ratio >= hi0] = 0.0
    edge_score[low] = (ratio[low] - lo0) / max(lo1 - lo0, 1e-9)
    edge_score[high] = (hi0 - ratio[high]) / max(hi0 - hi1, 1e-9)

    incident: list[list[int]] = [[] for _ in range(HAND_LM_COUNT)]
    for edge_index, (a, b) in enumerate(_GEOMETRY_EDGES):
        incident[a].append(edge_index)
        incident[b].append(edge_index)
    joint_score = np.zeros_like(raw)
    for joint, edge_indices in enumerate(incident):
        if edge_indices:
            joint_score[:, joint] = np.max(edge_score[:, edge_indices], axis=1)
    gated = raw * joint_score
    gated[gated < cfg.min_anchor_confidence] = 0.0
    return gated, refs


def _preserve_wrist(
    hand: hm.CapsuleHand, q_traj: np.ndarray, wrist: np.ndarray,
) -> np.ndarray:
    """Translate each model pose so its wrist equals the clean wrist exactly."""
    out = np.asarray(q_traj, float).copy()
    for t in range(len(out)):
        if np.all(np.isfinite(wrist[t])):
            current = hm.fk_landmark_positions(hand, out[t], [int(LM.WRIST)])[0]
            out[t, :3] += wrist[t] - current
    return out


def _fill_unreliable_finger_angles(
    hand: hm.CapsuleHand,
    q_traj: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Interpolate a finger's angles when its complete chain is not observed.

    ``q_from_landmarks`` necessarily uses all four points of a finger.  Letting
    it consume a filled or geometry-rejected tip reintroduces the exact
    coordinate-spline failure this stage is meant to remove.  Translation and
    palm orientation stay independent; only the four angles of that finger are
    filled from frames where the complete chain is supported.
    """
    out = np.asarray(q_traj, float).copy()
    time = np.arange(len(out), dtype=float)
    for finger, chain in hm.FINGERS.items():
        ids = [int(landmark) for landmark in chain]
        # q_from_landmarks dependencies: abduction/MCP use the proximal bone,
        # PIP the first two bones, and DIP the last two.  A rejected fingertip
        # must therefore suppress DIP only, not erase a well-observed MCP pose.
        required = (ids[:2], ids[:2], ids[:3], ids[1:])
        for joint_name, needed in zip(hm._JOINTS[finger], required):
            reliable = np.all(confidence[:, needed] > 0.0, axis=1)
            good = np.flatnonzero(reliable)
            if not len(good):
                continue
            joint = hand.model.joints[hand.model.getJointId(joint_name)]
            index = int(joint.idx_q)
            out[:, index] = np.interp(time, good.astype(float), out[good, index])
    return out


def _points_from_q(hand: hm.CapsuleHand, q_traj: np.ndarray) -> np.ndarray:
    order = list(range(HAND_LM_COUNT))
    return np.asarray(
        [hm.fk_landmark_positions(hand, q, order) for q in q_traj],
        dtype=np.float32,
    )


def _capsules_from_q(hand: hm.CapsuleHand, q_traj: np.ndarray) -> np.ndarray:
    return np.asarray(
        [hm.fk_capsule_endpoints(hand, q) for q in q_traj],
        dtype=np.float32,
    )


def _motion_metric(points: np.ndarray, order: int) -> float:
    if len(points) <= order:
        return 0.0
    delta = np.diff(np.asarray(points, float), n=order, axis=0)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=2))))


def _variant_metrics(
    source: np.ndarray, fitted: np.ndarray, confidence: np.ndarray,
    support: np.ndarray, refs: np.ndarray,
) -> dict[str, object]:
    residual = np.linalg.norm(fitted - source, axis=2)
    observed = (confidence > 0.0) & np.isfinite(residual)
    observed_residual = residual[observed]
    lengths = _edge_lengths(fitted)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lengths / refs[None, :]
    cv = np.nanstd(lengths, axis=0) / np.maximum(np.nanmean(lengths, axis=0), 1e-12)
    palm_columns = np.arange(len(_FINGER_EDGES), len(_GEOMETRY_EDGES))
    palm_bad = np.any(
        ~np.isfinite(ratio[:, palm_columns])
        | (ratio[:, palm_columns] < 0.80)
        | (ratio[:, palm_columns] > 1.20),
        axis=1,
    )
    wrist_error = residual[:, int(LM.WRIST)]
    finite = np.isfinite(fitted).all(axis=2)
    median_mm = (
        float(np.median(observed_residual) * 1000.0)
        if len(observed_residual) else float("nan")
    )
    p95_mm = (
        float(np.percentile(observed_residual, 95) * 1000.0)
        if len(observed_residual) else float("nan")
    )
    source_jerk_mm = _motion_metric(source, 3) * 1000.0
    fitted_jerk_mm = _motion_metric(fitted, 3) * 1000.0
    structural_pass = bool(
        finite.all()
        and not palm_bad.any()
        and float(np.nanmax(cv)) < 1e-3
        and float(np.nanmax(wrist_error)) < 1e-5
    )
    fidelity_pass = bool(
        np.isfinite(median_mm) and np.isfinite(p95_mm)
        and median_mm <= 20.0 and p95_mm <= 100.0
    )
    # A geometry stage is not allowed to buy anatomical validity by making the
    # already-good clean trajectory visibly jerkier.  Leave modest headroom for
    # the nonlinear landmarks→joint-angle→FK projection itself.
    temporal_limit_mm = max(source_jerk_mm * 1.5, source_jerk_mm + 0.75)
    temporal_pass = bool(fitted_jerk_mm <= temporal_limit_mm)
    return {
        "frames": int(len(fitted)),
        "finite_joint_fraction": float(finite.mean()),
        "supported_frame_fraction": float(np.asarray(support, bool).mean()),
        "anchor_joint_fraction": float(observed.mean()),
        "anchor_residual_median_mm": median_mm,
        "anchor_residual_p95_mm": p95_mm,
        "wrist_preservation_max_mm": float(np.nanmax(wrist_error) * 1000.0),
        "bone_length_cv_max": float(np.nanmax(cv)),
        "palm_outlier_frames": np.flatnonzero(palm_bad).tolist(),
        "source_joint_jerk_rms_mm": source_jerk_mm,
        "fitted_joint_jerk_rms_mm": fitted_jerk_mm,
        "temporal_jerk_limit_mm": temporal_limit_mm,
        "quality_gate": {
            "structural_pass": structural_pass,
            "fidelity_pass": fidelity_pass,
            "temporal_pass": temporal_pass,
            "accepted": structural_pass and fidelity_pass and temporal_pass,
        },
    }


def _json_safe(value):
    """Convert numpy/scalar non-finite values to strict JSON-compatible data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def fit_landmark_trajectory(
    points: np.ndarray,
    landmark_ids: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    cfg: ArticulatedConfig | None = None,
    *,
    optimize: bool = True,
    report=None,
) -> dict[str, object]:
    """Fit fixed hand geometry to one complete landmark trajectory."""
    cfg = cfg or ArticulatedConfig()
    report = report or (lambda **_kw: None)
    canonical_points = _canonical(np.asarray(points, np.float32), landmark_ids)
    canonical_conf = _canonical(
        np.asarray(confidence, np.float32)[..., None], landmark_ids, fill=0.0,
    )[..., 0]
    gated_conf, edge_refs = geometry_anchor_confidence(
        canonical_points, canonical_conf, cfg,
    )
    frames = _frames(canonical_points)
    reliable = gated_conf > 0.0
    source_valid = np.asarray(valid, bool)
    support = (
        source_valid
        & (reliable.sum(axis=1) >= cfg.min_supported_joints)
        & (reliable[:, _PALM_POSE_IDS].sum(axis=1) >= cfg.min_supported_palm_joints)
    )

    from viki.perception.hand_fit import (
        FitConfig,
        _calibration_frame_indices,
        _fill_invalid_initialization,
        _sanitize_warm_start,
        fit_trajectory_windowed,
    )

    calibration_indices = _calibration_frame_indices(
        frames, support, cfg.calibration_frames,
    )
    if not len(calibration_indices):
        raise ValueError("articulated fit: no usable hand-geometry calibration frames")
    hand = hm.build(hm.calibrate_from_frames([frames[t] for t in calibration_indices]))
    q_landmark = np.asarray([hm.q_from_landmarks(hand, frame) for frame in frames])
    support = _sanitize_warm_start(hand, q_landmark, support)
    q_projected = _fill_invalid_initialization(hand, q_landmark, support)
    q_projected = _fill_unreliable_finger_angles(hand, q_projected, gated_conf)
    q_projected = _preserve_wrist(
        hand, q_projected, canonical_points[:, int(LM.WRIST)],
    )

    q_rest = q_projected[int(calibration_indices[0])].copy()
    q_cal = np.asarray([q_projected[t] for t in calibration_indices])
    q_rest[7:] = np.median(q_cal[:, 7:], axis=0)
    solver_info: dict[str, object] = {"optimized": False}
    q_fitted = q_projected
    if optimize:
        fit_cfg = FitConfig(
            w_data=0.0,
            w_landmark=cfg.w_landmark,
            landmark_decay=1.0,
            w_vel_translation=cfg.w_vel_translation,
            w_vel_rotation=cfg.w_vel_rotation,
            w_vel_joints=cfg.w_vel_joints,
            w_acc_translation=cfg.w_acc_translation,
            w_acc_rotation=cfg.w_acc_rotation,
            w_acc_joints=cfg.w_acc_joints,
            w_posture=cfg.w_posture,
            min_points=1,
            max_nfev=cfg.max_nfev,
            outer_iterations=cfg.outer_iterations,
            window=cfg.window,
            window_overlap=cfg.window_overlap,
            workers=cfg.workers,
            deadline_s=cfg.deadline_s,
        )
        empty = [np.empty((0, 3), dtype=np.float64) for _ in frames]
        q_fitted, solver_info = fit_trajectory_windowed(
            hand,
            empty,
            [np.empty(0, dtype=np.float64) for _ in frames],
            q_projected,
            fit_cfg,
            landmark_frames=frames,
            landmark_confidence=gated_conf,
            q_rest=q_rest,
            report=report,
        )
        q_fitted = _preserve_wrist(
            hand, q_fitted, canonical_points[:, int(LM.WRIST)],
        )

    projected_points = _points_from_q(hand, q_projected)
    fitted_points = _points_from_q(hand, q_fitted)
    return {
        "projected_points": projected_points,
        "optimized_points": fitted_points,
        "projected_capsules": _capsules_from_q(hand, q_projected),
        "optimized_capsules": _capsules_from_q(hand, q_fitted),
        "capsule_radii": hm.capsule_radii(hand).astype(np.float32),
        "projected_q": q_projected.astype(np.float32),
        "optimized_q": np.asarray(q_fitted, np.float32),
        "anchor_confidence": gated_conf.astype(np.float32),
        "support_mask": support,
        "edge_reference_m": edge_refs.astype(np.float32),
        "calibration_indices": calibration_indices,
        "hand_params": hand.params.as_dict(),
        "solver_info": solver_info,
        "projected_metrics": _variant_metrics(
            canonical_points, projected_points, gated_conf, support, edge_refs,
        ),
        "optimized_metrics": _variant_metrics(
            canonical_points, fitted_points, gated_conf, support, edge_refs,
        ),
    }


def _verified_baseline(ep, source_profile: str) -> Path:
    root = ep.intermediates_dir / "baselines" / source_profile
    source = root / "cln.npz"
    manifest_path = root / "manifest.json"
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"no protected {source_profile!r} baseline for episode {ep.id}; "
            "run the clean perception profile first"
        )
    from viki.prepare.baseline import file_sha256

    manifest = json.loads(manifest_path.read_text())
    actual = file_sha256(source)
    if manifest.get("artifact_sha256") != actual:
        raise RuntimeError(
            f"protected baseline hash mismatch at {source}: "
            f"manifest={manifest.get('artifact_sha256')}, actual={actual}"
        )
    return source


def _variant_payload(
    source_data: dict[str, np.ndarray],
    points_canonical: np.ndarray,
    q: np.ndarray,
    capsules: np.ndarray,
    result: dict[str, object],
    cfg: ArticulatedConfig,
    stage: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    ids = np.asarray(source_data["landmark_ids"], int)
    points = np.asarray(points_canonical[:, ids], np.float32)

    # Reuse the one pose/gripper implementation used by canonical CLNs.  This
    # import is intentionally local: the clean prepare pass is complete before
    # the experimental post-process starts.
    from viki.prepare.run import _pose_and_gripper

    fit_positions, fit_rotations, _fit_rpy, _fit_valid, _fit_gripper = _pose_and_gripper(
        points,
        ids,
        np.asarray(source_data["timestamps"], np.int64),
        "binary",
    )
    payload: dict[str, object] = dict(source_data)
    payload.update({
        # Match the viewer's established hand-fit overlay contract: the fused
        # yellow skeleton remains the untouched clean trajectory, while the
        # blue ``hand fit`` marker draws this articulated candidate on top.
        "hand_fit_positions": fit_positions,
        "hand_fit_rotations": fit_rotations,
        "hand_fit_joint_angles": np.asarray(q, np.float32),
        "hand_fit_model_nq": np.int64(np.asarray(q).shape[1]),
        "hand_fit_capsules": np.asarray(capsules, np.float32),
        "hand_fit_capsule_radii": np.asarray(result["capsule_radii"], np.float32),
        "hand_fit_metrics_json": np.asarray(json.dumps(
            _json_safe(metrics), sort_keys=True, allow_nan=False,
        )),
        "geometry_source_points": np.asarray(source_data["smoothed_points"], np.float32),
        "geometry_joint_angles": np.asarray(q, np.float32),
        "geometry_anchor_confidence": np.asarray(result["anchor_confidence"], np.float32)[:, ids],
        "geometry_support_mask": np.asarray(result["support_mask"], bool),
        "geometry_edge_reference_m": np.asarray(result["edge_reference_m"], np.float32),
        "geometry_calibration_frame_indices": np.asarray(
            result["calibration_indices"], np.int32,
        ),
        "geometry_recipe": np.asarray(cfg.name),
        "geometry_source_profile": np.asarray(cfg.source_profile),
        "geometry_params_json": np.asarray(json.dumps(cfg.manifest(), sort_keys=True)),
        "geometry_hand_params_json": np.asarray(json.dumps(result["hand_params"], sort_keys=True)),
        "geometry_solver_info_json": np.asarray(json.dumps(
            _json_safe(result["solver_info"]), sort_keys=True, allow_nan=False,
        )),
        "geometry_metrics_json": np.asarray(json.dumps(
            _json_safe(metrics), sort_keys=True, allow_nan=False,
        )),
        "checkpoint_stage": np.asarray(stage),
        "pose_source": np.asarray("landmarks+hand_fit_overlay"),
        "active_variant": np.asarray(f"{cfg.name}:{stage}"),
    })
    return payload


def generate_articulated_variants(
    ep,
    *,
    cfg: ArticulatedConfig | None = None,
    report=None,
) -> dict[str, object]:
    """Generate anatomical A/B variants without replacing the active CLN."""
    cfg = cfg or ArticulatedConfig()
    source_path = _verified_baseline(ep, cfg.source_profile)
    with np.load(source_path, allow_pickle=False) as source:
        source_data = {key: source[key] for key in source.files}
    required = {"smoothed_points", "landmark_ids", "landmark_confidence", "valid", "timestamps"}
    missing = sorted(required - source_data.keys())
    if missing:
        raise ValueError(f"baseline {source_path} lacks required arrays: {missing}")

    result = fit_landmark_trajectory(
        source_data["smoothed_points"],
        source_data["landmark_ids"],
        source_data["landmark_confidence"],
        source_data["valid"],
        cfg,
        optimize=True,
        report=report,
    )
    root = ep.intermediates_dir / "geometry" / cfg.name
    projected_path = root / "40_projected.npz"
    optimized_path = root / "50_optimized.npz"
    from viki.prepare.checkpoints import atomic_savez, atomic_write_json

    projected_payload = _variant_payload(
        source_data,
        result["projected_points"],
        result["projected_q"],
        result["projected_capsules"],
        result,
        cfg,
        "geometry_projected",
        result["projected_metrics"],
    )
    optimized_payload = _variant_payload(
        source_data,
        result["optimized_points"],
        result["optimized_q"],
        result["optimized_capsules"],
        result,
        cfg,
        "geometry_optimized",
        result["optimized_metrics"],
    )
    atomic_savez(projected_path, projected_payload)
    atomic_savez(optimized_path, optimized_payload)
    report_payload = {
        "schema": 1,
        "recipe": cfg.manifest(),
        "source": str(source_path),
        "source_profile": cfg.source_profile,
        "calibration_frame_indices": np.asarray(
            result["calibration_indices"], int,
        ).tolist(),
        "projected": {"file": str(projected_path), **result["projected_metrics"]},
        "optimized": {"file": str(optimized_path), **result["optimized_metrics"]},
        "solver": result["solver_info"],
    }
    report_path = atomic_write_json(root / "report.json", _json_safe(report_payload))
    return {
        "projected": str(projected_path),
        "optimized": str(optimized_path),
        "report": str(report_path),
        "metrics": {
            "projected": result["projected_metrics"],
            "optimized": result["optimized_metrics"],
        },
    }


_OVERLAY_KEYS = (
    "hand_fit_positions",
    "hand_fit_rotations",
    "hand_fit_joint_angles",
    "hand_fit_model_nq",
    "hand_fit_capsules",
    "hand_fit_capsule_radii",
    "hand_fit_metrics_json",
)


def install_articulated_overlay(
    ep,
    *,
    cfg: ArticulatedConfig | None = None,
    variant: str = "optimized",
) -> dict[str, object]:
    """Attach one geometry candidate to active CLN as the viewer hand-fit layer.

    The canonical clean arrays are checked against their protected baseline
    before and after the write.  Only additive ``hand_fit_*`` / audit fields are
    installed, so the yellow fused skeleton remains the frozen clean result.
    """
    cfg = cfg or ArticulatedConfig()
    names = {"projected": "40_projected.npz", "optimized": "50_optimized.npz"}
    if variant not in names:
        raise ValueError(f"unknown articulated overlay {variant!r}; choose {sorted(names)}")
    baseline = _verified_baseline(ep, cfg.source_profile)
    active = Path(ep.cln_npz)
    candidate = ep.intermediates_dir / "geometry" / cfg.name / names[variant]
    if not active.is_file():
        raise FileNotFoundError(active)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"no {variant} geometry candidate at {candidate}; run geometry-fit first"
        )

    from viki.prepare.baseline import _same_core_arrays
    from viki.prepare.checkpoints import atomic_savez

    if not _same_core_arrays(baseline, active):
        raise RuntimeError(
            "active cln.npz no longer matches the protected clean trajectory; "
            "refusing to create a misleading overlay comparison"
        )
    with np.load(active, allow_pickle=False) as archive:
        payload: dict[str, object] = {key: archive[key] for key in archive.files}
    with np.load(candidate, allow_pickle=False) as archive:
        missing = [key for key in _OVERLAY_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"geometry candidate lacks overlay arrays: {missing}")
        if not np.array_equal(payload["timestamps"], archive["timestamps"]):
            raise ValueError("geometry candidate timestamps do not match active clean CLN")
        if not np.array_equal(payload["landmark_ids"], archive["landmark_ids"]):
            raise ValueError("geometry candidate landmark ids do not match active clean CLN")
        for key in _OVERLAY_KEYS:
            payload[key] = archive[key]
        for key in archive.files:
            if key.startswith("geometry_"):
                payload[key] = archive[key]
    payload["hand_fit_overlay_variant"] = np.asarray(f"{cfg.name}:{variant}")
    payload["hand_fit_overlay_source_profile"] = np.asarray(cfg.source_profile)
    atomic_savez(active, payload)
    if not _same_core_arrays(baseline, active):
        raise RuntimeError("overlay installation unexpectedly changed clean core arrays")
    return {
        "path": str(active),
        "overlay": f"{cfg.name}:{variant}",
        "clean_core_unchanged": True,
    }
