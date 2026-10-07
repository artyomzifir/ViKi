"""Build object-relative hand and robot trajectories after scene perception and IK.

Object poses in ``object_models.npz`` are expressed in the rig frame. The
retarget plan's TCP poses are in the calibration frame, so its recorded
rig-to-calibration anchor is applied to the object before taking the relative
transform. This artifact is diagnostic data, not an object-driven IK policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from viki.contracts import Episode, cln_pose_keys
from viki.episode import mark_stage
from viki.perception import OBJECT_MODEL_SCHEMA
from viki.prepare import object_relative


SCHEMA = "viki_object_relative_v1"


def _pose_matrices(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if positions.shape[:-1] != rotations.shape[:-2] or positions.shape[-1] != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"incompatible pose shapes: {positions.shape}, {rotations.shape}")
    poses = np.broadcast_to(np.eye(4), (*positions.shape[:-1], 4, 4)).copy()
    poses[..., :3, :3] = rotations
    poses[..., :3, 3] = positions
    return poses


def build_object_relative_episode(ep: Episode) -> Path:
    """Write per-object ``inv(T_rig_object) @ T_rig_hand/T_rig_TCP`` tracks."""
    if not ep.object_models_npz.is_file():
        raise FileNotFoundError(f"missing object models: {ep.object_models_npz}")
    if not ep.cln_npz.is_file() or not ep.plan_h5.is_file():
        raise FileNotFoundError("object-relative tracks require cln.npz and plan.h5")

    with np.load(ep.object_models_npz, allow_pickle=False) as models:
        if str(models["schema"]) != OBJECT_MODEL_SCHEMA:
            raise ValueError("unsupported object-model schema")
        object_ids = np.asarray(models["object_ids"], dtype=np.int32)
        object_labels = np.asarray(models["object_labels"]).astype(str)
        object_positions = np.asarray(models["translation_world"], dtype=np.float64)
        object_rotations = np.asarray(models["rotation_world_object"], dtype=np.float64)
        confidence = np.asarray(models["track_confidence"], dtype=np.float32)
        rotation_information = np.asarray(models["rotation_information"], dtype=np.float32)
        contact = np.asarray(models["contact"], dtype=bool)

    if len(object_ids) == 0:
        raise ValueError("object model contains no tracked objects")
    frame_count, object_count = object_positions.shape[:2]
    if (object_positions.shape != (frame_count, object_count, 3)
            or object_rotations.shape != (frame_count, object_count, 3, 3)
            or confidence.shape != (frame_count, object_count)
            or rotation_information.shape != (frame_count, object_count, 3)
            or contact.shape != (frame_count, object_count)
            or len(object_ids) != object_count or len(object_labels) != object_count):
        raise ValueError("object-model arrays have incompatible frame or object axes")
    if not (np.isfinite(object_positions).all() and np.isfinite(object_rotations).all()):
        raise ValueError("object poses must be finite")

    with h5py.File(ep.plan_h5) as plan:
        plan_timestamps = np.asarray(plan["timestamps"])
        target_pose = _pose_matrices(
            plan["target_position_calibration"], plan["target_rotation_calibration"]
        )
        achieved_pose = _pose_matrices(
            plan["achieved_position_calibration"], plan["achieved_rotation_calibration"]
        )
        gripper_opening = np.asarray(plan["gripper_opening"], dtype=np.float32)
        source_pose = plan["source_pose"][()] if "source_pose" in plan else "landmarks"
        source_pose = source_pose.decode() if isinstance(source_pose, bytes) else str(source_pose)

    with np.load(ep.cln_npz, allow_pickle=False) as hand:
        coordinate_frame = str(hand["coordinate_frame"])
        if coordinate_frame not in {"rig", "robot_base"}:
            raise ValueError(f"expected rig-frame hand pose, got {coordinate_frame!r}")
        hand_timestamps = np.asarray(hand["timestamps"])
        position_key, rotation_key = cln_pose_keys(hand.files, source_pose)
        if source_pose == "hand_fit" and position_key != "hand_fit_positions":
            raise ValueError("plan used hand-fit poses but cln.npz has no hand-fit arrays")
        hand_pose = _pose_matrices(hand[position_key], hand[rotation_key])
        hand_valid = np.asarray(hand["valid"], dtype=bool)

    if (len(plan_timestamps) != frame_count or len(hand_timestamps) != frame_count
            or not np.array_equal(plan_timestamps, hand_timestamps)
            or hand_pose.shape != (frame_count, 4, 4)
            or target_pose.shape != (frame_count, 4, 4)
            or achieved_pose.shape != (frame_count, 4, 4)
            or hand_valid.shape != (frame_count,)
            or gripper_opening.shape != (frame_count,)):
        raise ValueError("hand, object, and robot trajectories must share one frame clock")

    anchor_path = ep.raw_dir / "world_anchor.json"
    calibration_from_rig = np.asarray(
        json.loads(anchor_path.read_text())["T_world_display"], dtype=np.float64
    )
    if calibration_from_rig.shape != (4, 4) or not np.isfinite(calibration_from_rig).all():
        raise ValueError("invalid rig-to-calibration anchor")

    object_rig = _pose_matrices(object_positions, object_rotations)
    object_calibration = calibration_from_rig @ object_rig
    relative_hand = np.stack([
        object_relative(hand_pose, object_rig[:, index])
        for index in range(object_count)
    ], axis=1)
    relative_target = np.stack([
        object_relative(target_pose, object_calibration[:, index])
        for index in range(object_count)
    ], axis=1)
    relative_achieved = np.stack([
        object_relative(achieved_pose, object_calibration[:, index])
        for index in range(object_count)
    ], axis=1)
    relative_valid = (
        hand_valid[:, None]
        & (confidence > 0)
        & np.isfinite(relative_hand).all(axis=(2, 3))
        & np.isfinite(relative_target).all(axis=(2, 3))
    )
    relative_hand[~relative_valid] = np.nan
    relative_target[~relative_valid] = np.nan

    output = ep.object_relative_npz
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            schema=np.asarray(SCHEMA),
            timestamps=plan_timestamps,
            object_ids=object_ids,
            object_labels=object_labels,
            T_object_hand=relative_hand.astype(np.float32),
            T_object_target_tcp=relative_target.astype(np.float32),
            T_object_achieved_tcp=relative_achieved.astype(np.float32),
            T_rig_object=object_rig.astype(np.float32),
            relative_valid=relative_valid,
            object_track_confidence=confidence,
            object_rotation_information=rotation_information,
            object_contact=contact,
            gripper_opening=gripper_opening,
            hand_pose_source=np.asarray(source_pose),
        )
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    manipulated_ids = object_ids[object_labels == "manipulated_object"]
    mark_stage(
        ep,
        "object_relative",
        frames=frame_count,
        objects=object_count,
        manipulated_object_ids=manipulated_ids.tolist(),
        artifact=str(output.relative_to(ep.root)),
        note="Estimated object frames; not used by IK or certified as cube orientation ground truth",
    )
    return output
