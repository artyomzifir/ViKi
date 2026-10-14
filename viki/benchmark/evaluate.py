"""Absolute hand-joint evaluation for imported, frame-aligned HO-Cap episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from viki.contracts import HAND_LM_COUNT, LM


def _ordered_points(points: np.ndarray, landmark_ids: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    landmark_ids = np.asarray(landmark_ids, dtype=np.int64)
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] != len(landmark_ids):
        raise ValueError("landmark array must have shape (frames, ids, 3)")
    if len(set(landmark_ids.tolist())) != len(landmark_ids):
        raise ValueError("duplicate landmark ids")
    if np.any((landmark_ids < 0) | (landmark_ids >= HAND_LM_COUNT)):
        raise ValueError("landmark id outside the ViKi hand")
    ordered = np.full((len(points), HAND_LM_COUNT, 3), np.nan, dtype=np.float64)
    ordered[:, landmark_ids] = points
    return ordered


def _ordered_mask(mask: np.ndarray, landmark_ids: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != len(landmark_ids):
        raise ValueError("observed mask must have shape (frames, ids)")
    ordered = np.zeros((len(mask), HAND_LM_COUNT), dtype=bool)
    ordered[:, landmark_ids] = mask
    return ordered


def _place_rows(
    values: np.ndarray,
    timestamps: np.ndarray,
    tick_to_frame: dict[int, int],
    frame_count: int,
    *,
    fill: float | bool,
) -> np.ndarray:
    if len(values) != len(timestamps):
        raise ValueError("timestamp and prediction row counts differ")
    result = np.full((frame_count,) + values.shape[1:], fill, dtype=values.dtype)
    seen: set[int] = set()
    for value, tick in zip(values, timestamps):
        frame = tick_to_frame.get(int(tick))
        if frame is None:
            raise ValueError(f"prediction timestamp {int(tick)} is not a capture tick")
        if frame in seen:
            raise ValueError(f"duplicate prediction for capture frame {frame}")
        seen.add(frame)
        result[frame] = value
    return result


def _metrics(points: np.ndarray, ground_truth: np.ndarray, eligible: np.ndarray,
             observed: np.ndarray | None = None) -> dict:
    valid_pred = np.isfinite(points).all(axis=-1)
    matched = eligible & valid_pred
    if observed is not None:
        matched &= observed
    error = np.linalg.norm(points - ground_truth, axis=-1) * 1000.0
    distances = error[matched]
    total = int(eligible.sum())
    count = int(matched.sum())
    result = {
        "gt_joints": total,
        "matched_joints": count,
        "coverage": count / total if total else None,
        "frames_with_match": int(matched.any(axis=1).sum()),
        "mpjpe_mm": float(distances.mean()) if count else None,
        "median_mm": float(np.median(distances)) if count else None,
        "p95_mm": float(np.percentile(distances, 95)) if count else None,
        "pck_20mm": float(np.mean(distances <= 20.0)) if count else None,
        "pck_50mm": float(np.mean(distances <= 50.0)) if count else None,
    }
    for name, indices in (
        ("wrist", [int(LM.WRIST)]),
        ("fingertips", [int(value) for value in (
            LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP
        )]),
    ):
        selected = matched[:, indices]
        result[f"{name}_matched_joints"] = int(selected.sum())
        result[f"{name}_mpjpe_mm"] = (
            float(error[:, indices][selected].mean()) if selected.any() else None
        )
    eligible_steps = eligible[1:] & eligible[:-1]
    matched_steps = matched[1:] & matched[:-1]
    displacement_error = np.linalg.norm(
        (points[1:] - points[:-1]) - (ground_truth[1:] - ground_truth[:-1]),
        axis=-1,
    ) * 1000.0
    step_errors = displacement_error[matched_steps]
    result["consecutive_matched_joints"] = int(matched_steps.sum())
    result["consecutive_coverage"] = (
        float(matched_steps.sum() / eligible_steps.sum()) if eligible_steps.any() else None
    )
    result["displacement_error_mm"] = (
        float(step_errors.mean()) if len(step_errors) else None
    )
    result["displacement_p95_mm"] = (
        float(np.percentile(step_errors, 95)) if len(step_errors) else None
    )
    return result


def evaluate_episode(episode_dir: str | Path) -> dict:
    """Score raw per-camera and prepared hand points against HO-Cap MANO labels.

    Uses the importer's camera/world transform and exact capture ticks; no
    Procrustes fit, time search, root alignment, or GT-aided frame selection.
    Scores are conditional on matched joints, so coverage must be reported
    alongside MPJPE. No pipeline stage is run or modified.
    """
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / "meta.json").read_text())
    if meta.get("source") != "hocap" or meta.get("timestamps") != "synthetic":
        raise ValueError("expected a frame-aligned HO-Cap import")
    with np.load(episode_dir / "gt/joints3d_world.npz", allow_pickle=False) as data:
        ground_truth = np.asarray(data["joints_world"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        frame_index = np.asarray(data["frame_index"], dtype=np.int64)
        joint_order = np.asarray(data["joint_order"]).tolist()
        world_frame = str(np.asarray(data["world_frame"]).item())
    if world_frame != "hocap_tag_1" or joint_order != [lm.name for lm in LM]:
        raise ValueError("HO-Cap world frame or joint order does not match ViKi")
    ticks = json.loads((episode_dir / "raw/timestamps.json").read_text())
    frame_count = len(ticks)
    if ground_truth.shape != (frame_count, HAND_LM_COUNT, 3):
        raise ValueError("GT and capture frame counts differ")
    if valid.shape != (frame_count, HAND_LM_COUNT):
        raise ValueError("invalid GT validity mask")
    if not np.array_equal(frame_index, np.arange(frame_count)):
        raise ValueError("GT frame indices are not capture indices")
    eligible = valid & np.isfinite(ground_truth).all(axis=-1)
    sync_to_frame = {int(row["sync_us"]): index for index, row in enumerate(ticks)}
    if len(sync_to_frame) != frame_count:
        raise ValueError("capture sync ticks are not unique")
    cameras = list(meta["cameras"])
    camera_ticks: dict[str, dict[int, int]] = {}
    for camera in cameras:
        camera_ticks[camera] = {
            int(row["sync_us"]) + int(row.get("offsets_us", {}).get(camera, 0)): index
            for index, row in enumerate(ticks)
        }
        if len(camera_ticks[camera]) != frame_count:
            raise ValueError(f"capture ticks are not unique for {camera}")
    report: dict = {
        "source": "hocap",
        "episode": episode_dir.name,
        "ground_truth": "semi-automatic multi-view MANO annotations",
        "frame_count": frame_count,
        "gt_frames": int(eligible.any(axis=1).sum()),
        "gt_joints": int(eligible.sum()),
        "alignment": "exact synthetic capture timestamp, absolute hocap_tag_1 frame",
        "rec_per_camera": {},
        "prepare": {},
    }
    rec_path = episode_dir / "rec.npz"
    if rec_path.is_file():
        with np.load(rec_path, allow_pickle=False) as data:
            device_ids = np.asarray(data["device_ids"]).astype(str)
            timestamps = np.asarray(data["timestamps"], dtype=np.int64)
            landmark_ids = np.asarray(data["landmark_ids"], dtype=np.int64)
            points = _ordered_points(data["points"], landmark_ids)
        if len(device_ids) != len(points):
            raise ValueError("rec device-id and point row counts differ")
        if set(device_ids) - set(cameras):
            raise ValueError("rec contains a camera absent from the HO-Cap import")
        for camera in cameras:
            selected = device_ids == camera
            aligned = _place_rows(
                points[selected], timestamps[selected], camera_ticks[camera],
                frame_count, fill=np.nan,
            )
            report["rec_per_camera"][camera] = _metrics(aligned, ground_truth, eligible)
    else:
        report["rec_per_camera"] = {"status": "missing rec.npz"}

    clean_path = episode_dir / "cln.npz"
    status_path = episode_dir / "status.json"
    stages = json.loads(status_path.read_text()).get("stages", {}) if status_path.is_file() else {}
    if clean_path.is_file() and not stages.get("prepare", {}).get("done", False):
        report["prepare"] = {"status": "incomplete prepare stage; cln.npz ignored"}
    elif clean_path.is_file():
        with np.load(clean_path, allow_pickle=False) as data:
            timestamps = np.asarray(data["timestamps"], dtype=np.int64)
            landmark_ids = np.asarray(data["landmark_ids"], dtype=np.int64)
            smoothed = _ordered_points(data["smoothed_points"], landmark_ids)
            observed = (
                _ordered_mask(data["observed_mask"], landmark_ids)
                if "observed_mask" in data.files else None
            )
            if observed is None and "observed_points" in data.files:
                observed = np.isfinite(
                    _ordered_points(data["observed_points"], landmark_ids)
                ).all(axis=-1)
            fused = (
                _ordered_points(data["observed_points"], landmark_ids)
                if "observed_points" in data.files else None
            )
        aligned = _place_rows(
            smoothed, timestamps, sync_to_frame, frame_count, fill=np.nan
        )
        report["prepare"]["smoothed_all"] = _metrics(aligned, ground_truth, eligible)
        if observed is not None:
            observed_full = _place_rows(
                observed, timestamps, sync_to_frame, frame_count, fill=False
            )
            report["prepare"]["smoothed_observed"] = _metrics(
                aligned, ground_truth, eligible, observed_full
            )
            if fused is not None:
                fused_full = _place_rows(
                    fused, timestamps, sync_to_frame, frame_count, fill=np.nan
                )
                report["prepare"]["fused_observed"] = _metrics(
                    fused_full, ground_truth, eligible, observed_full
                )
        else:
            report["prepare"]["observed_status"] = "unavailable: no observation mask"
    else:
        report["prepare"] = {"status": "missing cln.npz"}
    return report
