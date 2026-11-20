"""Object-pose reference and initial-frame-aligned tracking scores for HO-Cap."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from viki.contracts import CalibrationExtrinsics


def import_object_ground_truth(
    archives: str | Path, sequence_id: str, episode_dir: str | Path
) -> Path:
    """Convert published camera-frame object fits into the tag_1 world frame."""
    archives = Path(archives)
    episode_dir = Path(episode_dir)
    episode = json.loads((episode_dir / "meta.json").read_text())
    if episode.get("source") != "hocap":
        raise ValueError("object reference requires a HO-Cap episode")
    if sequence_id != f"{episode['hocap_subject']}-{episode['hocap_sequence']}":
        raise ValueError("archive sequence does not match the episode")
    with tarfile.open(archives / "sequence_labels.tar") as source:
        member = source.extractfile(f"{sequence_id}.meta.yaml")
        if member is None:
            raise ValueError("sequence metadata missing")
        metadata = yaml.safe_load(member.read())
    object_ids = tuple(str(value) for value in metadata["object_ids"])
    frame_count = int(episode["num_frames"])
    if frame_count > int(metadata["num_frames"]):
        raise ValueError("episode exceeds source sequence length")
    camera = next(iter(episode["cameras"]))
    extrinsics = json.loads((episode_dir / "raw/extrinsics.json").read_text())[camera]
    world_from_camera = CalibrationExtrinsics(
        rvec=np.asarray(extrinsics["rvec"], dtype=np.float64),
        tvec=np.asarray(extrinsics["tvec"], dtype=np.float64),
    ).transform_matrix
    poses_world = np.full((frame_count, len(object_ids), 4, 4), np.nan, np.float32)
    valid = np.zeros((frame_count, len(object_ids)), bool)
    with tarfile.open(archives / f"{sequence_id}.tar") as source:
        for frame_index in range(frame_count):
            filename = f"{camera}-{frame_index:06d}.label.npy"
            member = source.extractfile(filename)
            if member is None:
                raise ValueError(f"missing label {filename}")
            with np.load(io.BytesIO(member.read()), allow_pickle=False) as label:
                if "obj_poses" not in label.files:
                    raise ValueError(f"missing obj_poses in {filename}")
                poses = np.asarray(label["obj_poses"], dtype=np.float64)
            if poses.shape != (len(object_ids), 4, 4):
                raise ValueError(f"unexpected obj_poses shape in {filename}: {poses.shape}")
            finite = np.isfinite(poses).all(axis=(1, 2))
            homogeneous = np.isclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5).all(axis=1)
            determinant = np.linalg.det(poses[:, :3, :3])
            good = finite & homogeneous & np.isclose(determinant, 1.0, atol=1e-3)
            valid[frame_index] = good
            poses_world[frame_index, good] = (
                world_from_camera @ poses[good]
            ).astype(np.float32)
    output = episode_dir / "gt/object_poses_world.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, object_ids=np.asarray(object_ids), poses_world=poses_world,
            valid=valid, frame_index=np.arange(frame_count, dtype=np.int32),
            world_frame=np.asarray("hocap_tag_1"), source_camera=np.asarray(camera),
            source=np.asarray("HO-Cap semi-automatic object fits"),
        )
    temporary.replace(output)
    return output


def evaluate_object_models(
    gt_path: str | Path, model_path: str | Path,
    object_name_by_id: Mapping[int, str], *, min_confidence: float = 0.35,
) -> dict:
    """Score a track after one GT-assisted SE(3) gauge alignment at its first match."""
    with np.load(gt_path, allow_pickle=False) as reference:
        if str(reference["world_frame"]) != "hocap_tag_1":
            raise ValueError("object GT is not in the HO-Cap tag_1 frame")
        names = [str(value) for value in reference["object_ids"]]
        poses = np.asarray(reference["poses_world"], dtype=np.float64)
        valid = np.asarray(reference["valid"], dtype=bool)
    with np.load(model_path, allow_pickle=False) as model:
        ids = [int(value) for value in model["object_ids"]]
        rotations = np.asarray(model["rotation_world_object"], dtype=np.float64)
        translations = np.asarray(model["translation_world"], dtype=np.float64)
        confidence = np.asarray(model["track_confidence"], dtype=np.float64)
    frames = len(poses)
    if (rotations.shape != (frames, len(ids), 3, 3)
            or translations.shape != (frames, len(ids), 3)
            or confidence.shape != (frames, len(ids))):
        raise ValueError("object model and GT frame/object dimensions differ")
    if set(ids) != set(object_name_by_id):
        raise ValueError("explicit model-id to HO-Cap-name mapping is incomplete")
    if len(set(object_name_by_id.values())) != len(ids):
        raise ValueError("object names in mapping must be unique")
    report = {
        "alignment": "one right-multiplied object-local SE(3) alignment at first confident frame",
        "absolute_pose_error": "unavailable without a calibrated object-model origin",
        "min_confidence": min_confidence,
        "objects": {},
    }
    for column, object_id in enumerate(ids):
        name = object_name_by_id[object_id]
        if name not in names:
            raise ValueError(f"object {name!r} absent from GT")
        reference_index = names.index(name)
        eligible = valid[:, reference_index]
        tracked = (
            eligible & np.isfinite(rotations[:, column]).all(axis=(1, 2))
            & np.isfinite(translations[:, column]).all(axis=1)
            & np.isfinite(confidence[:, column])
            & (confidence[:, column] >= min_confidence)
        )
        item = {
            "gt_frames": int(eligible.sum()), "tracked_frames": int(tracked.sum()),
            "coverage": float(tracked.sum() / eligible.sum()) if eligible.any() else None,
            "translation_mean_mm": None, "translation_p95_mm": None,
            "rotation_mean_deg": None, "rotation_p95_deg": None,
            "anchor_frame": None,
        }
        if tracked.any():
            first = int(np.flatnonzero(tracked)[0])
            item["anchor_frame"] = first
            initial_prediction = np.eye(4)
            initial_prediction[:3, :3] = rotations[first, column]
            initial_prediction[:3, 3] = translations[first, column]
            alignment = np.linalg.inv(initial_prediction) @ poses[first, reference_index]
            predicted = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
            predicted[:, :3, :3] = rotations[:, column]
            predicted[:, :3, 3] = translations[:, column]
            aligned = predicted[tracked] @ alignment
            truth = poses[tracked, reference_index]
            translation_error = np.linalg.norm(
                aligned[:, :3, 3] - truth[:, :3, 3], axis=1
            ) * 1000.0
            relative_rotation = (
                np.swapaxes(truth[:, :3, :3], 1, 2) @ aligned[:, :3, :3]
            )
            cosine = (np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0
            rotation_error = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
            item.update({
                "translation_mean_mm": float(translation_error.mean()),
                "translation_p95_mm": float(np.percentile(translation_error, 95)),
                "rotation_mean_deg": float(rotation_error.mean()),
                "rotation_p95_deg": float(np.percentile(rotation_error, 95)),
            })
        report["objects"][name] = item
    return report
