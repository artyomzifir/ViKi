"""Persistence and measurements for lossless prepare-stage checkpoints.

The canonical ``cln.npz`` is deliberately small and consumer-oriented.  These
helpers keep the evidence needed to audit it: what fusion actually observed,
what interpolation fabricated, what smoothing changed, and which parameters
produced every variant.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np


CHECKPOINT_SCHEMA = 1

# MediaPipe topology, kept local so this leaf module does not depend on the UI.
_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def run_name(fusion_mode: str, interp_max_gap: int, window: int, polyorder: int) -> str:
    gap = "all" if int(interp_max_gap) == 0 else str(int(interp_max_gap))
    return f"{fusion_mode}__gap-{gap}__sg-{int(window)}-{int(polyorder)}"


def atomic_savez(path: Path, arrays: dict[str, object]) -> Path:
    """Atomically replace one compressed NPZ so readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def save_camera_stage(
    path: Path,
    *,
    stage: str,
    trajectories: dict[str, np.ndarray],
    timestamps: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
    landmark_ids: np.ndarray,
    params: dict,
) -> Path:
    """Persist ragged per-camera trajectories without object/pickle arrays."""
    devices = sorted(trajectories)
    arrays: dict[str, object] = {
        "checkpoint_schema": np.int32(CHECKPOINT_SCHEMA),
        "checkpoint_stage": np.asarray(stage),
        "checkpoint_params_json": np.asarray(json.dumps(params, sort_keys=True)),
        "camera_ids": np.asarray(devices),
        "landmark_ids": np.asarray(landmark_ids),
    }
    for index, dev in enumerate(devices):
        arrays[f"points_{index}"] = np.asarray(trajectories[dev], np.float32)
        arrays[f"timestamps_{index}"] = np.asarray(timestamps[dev], np.int64)
        arrays[f"confidence_{index}"] = np.asarray(confidence[dev], np.float32)
    return atomic_savez(path, arrays)


def _diff_metric(points: np.ndarray, order: int) -> tuple[float, float]:
    delta = np.diff(np.asarray(points, float), n=order, axis=0)
    norm = np.linalg.norm(delta, axis=-1)
    norm = norm[np.isfinite(norm)] * 1000.0
    if not len(norm):
        return float("nan"), float("nan")
    return float(np.median(norm)), float(np.percentile(norm, 95))


def metrics(path: Path) -> dict:
    """Measure motion and anatomical stability of one viewer-compatible NPZ."""
    with np.load(path, allow_pickle=False) as data:
        observed_points = np.asarray(
            data.get("observed_points", data["smoothed_points"]), float
        )
        if "hand_fit_capsules" in data.files:
            capsules = np.asarray(data["hand_fit_capsules"], float)
            joints = [capsules[:, 0, 0]]
            for start in range(1, 16, 3):
                joints.extend([
                    capsules[:, start, 0], capsules[:, start, 1],
                    capsules[:, start + 1, 1], capsules[:, start + 2, 1],
                ])
            points = np.stack(joints, axis=1)
            ids = np.arange(points.shape[1])
        else:
            points = np.asarray(data["smoothed_points"], float)
            ids = np.asarray(data["landmark_ids"], int)
        valid = np.asarray(data.get("valid", np.zeros(len(points))), bool)
        observed = np.asarray(
            data.get("observed_mask", np.isfinite(points).all(axis=2)), bool
        )
        stage = str(np.asarray(data.get("checkpoint_stage", "cln")).item())
        mode = str(np.asarray(data.get("perception_fuse_mode", "unknown")).item())
        fitted_wrist = (
            np.asarray(data["hand_fit_positions"], float)
            if "hand_fit_positions" in data.files else None
        )
        pose_source = "hand_fit" if fitted_wrist is not None else "landmarks"

    if mode == "unknown":
        stem = Path(path).stem
        if "triangulate" in stem:
            mode = "triangulate"
        elif "xyzmean" in stem or "xyz_mean" in stem:
            mode = "xyz_mean"

    columns = {int(lm): i for i, lm in enumerate(ids)}
    wrist = fitted_wrist if fitted_wrist is not None else points[:, columns.get(0, 0)]
    step_med, step_p95 = _diff_metric(wrist, 1)
    acc_med, acc_p95 = _diff_metric(wrist, 2)
    jerk_med, jerk_p95 = _diff_metric(wrist, 3)

    edge_lengths = []
    observed_edge_lengths = []
    for a, b in _HAND_EDGES:
        if a in columns and b in columns:
            edge_lengths.append(
                np.linalg.norm(points[:, columns[a]] - points[:, columns[b]], axis=1)
            )
            observed_edge_lengths.append(np.linalg.norm(
                observed_points[:, columns[a]] - observed_points[:, columns[b]], axis=1
            ))
    lengths = np.stack(edge_lengths, axis=1) if edge_lengths else np.empty((len(points), 0))
    observed_lengths = (
        np.stack(observed_edge_lengths, axis=1)
        if observed_edge_lengths else np.empty((len(points), 0))
    )
    reference = np.asarray([
        np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
        for column in observed_lengths.T
    ]) if observed_lengths.size else np.empty(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lengths / reference
    bad_edges = np.isfinite(ratio) & ((ratio < 0.55) | (ratio > 1.8))
    anatomical_outlier = bad_edges.sum(axis=1) >= 2 if bad_edges.size else np.zeros(len(points), bool)

    palm_pairs = ((0, 5), (0, 9), (0, 17), (5, 17))
    palm_lengths = []
    observed_palm_lengths = []
    for a, b in palm_pairs:
        if a in columns and b in columns:
            palm_lengths.append(
                np.linalg.norm(points[:, columns[a]] - points[:, columns[b]], axis=1)
            )
            observed_palm_lengths.append(np.linalg.norm(
                observed_points[:, columns[a]] - observed_points[:, columns[b]], axis=1
            ))
    palm = np.stack(palm_lengths, axis=1) if palm_lengths else np.empty((len(points), 0))
    observed_palm = (
        np.stack(observed_palm_lengths, axis=1)
        if observed_palm_lengths else np.empty((len(points), 0))
    )
    palm_ref = np.asarray([
        np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
        for column in observed_palm.T
    ]) if observed_palm.size else np.empty(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        palm_ratio = palm / palm_ref
    palm_collapse = (
        np.any(np.isfinite(palm_ratio) & (palm_ratio < 0.55), axis=1)
        if palm_ratio.size else np.zeros(len(points), bool)
    )

    return {
        "file": str(Path(path)),
        "fusion_mode": mode,
        "stage": stage,
        "pose_source": pose_source,
        "frames": int(len(points)),
        "finite_joint_fraction": float(np.isfinite(points).all(axis=2).mean()),
        "direct_observation_fraction": float(observed.mean()),
        "pose_valid_fraction": float(valid.mean()),
        "wrist_step_median_mm": step_med,
        "wrist_step_p95_mm": step_p95,
        "wrist_acc_median_mm": acc_med,
        "wrist_acc_p95_mm": acc_p95,
        "wrist_jerk_median_mm": jerk_med,
        "wrist_jerk_p95_mm": jerk_p95,
        "anatomical_outlier_frame_fraction": float(anatomical_outlier.mean()),
        "anatomical_outlier_frames": np.flatnonzero(anatomical_outlier).tolist(),
        "palm_collapse_frame_fraction": float(palm_collapse.mean()),
        "palm_collapse_frames": np.flatnonzero(palm_collapse).tolist(),
    }


def write_comparison(path: Path, variants: list[Path]) -> Path:
    rows = [metrics(candidate) for candidate in variants if Path(candidate).is_file()]
    return atomic_write_json(Path(path), {
        "schema": CHECKPOINT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variants": rows,
    })
