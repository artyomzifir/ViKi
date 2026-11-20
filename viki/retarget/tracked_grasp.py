"""Opt-in, single-object grasp targets from an aligned rigid track."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from viki.perception import OBJECT_MODEL_SCHEMA
from viki.retarget.object_grasp import CubeGraspPlan, _first_sustained


def _jaw_rotation(
    object_rotation: np.ndarray, local_axis: int, jaw_sign: int,
    tool_axis: str,
) -> np.ndarray:
    jaw = np.asarray(object_rotation[:, local_axis], dtype=np.float64).copy()
    jaw[2] = 0.0
    jaw_length = np.linalg.norm(jaw)
    if jaw_length < 0.5:
        raise ValueError("selected object side is too vertical for a side grasp")
    jaw = jaw_sign * jaw / jaw_length
    if tool_axis == "down":
        tool = np.array([0.0, 0.0, -1.0])
    else:
        tool = np.asarray(object_rotation[:, 1 - local_axis], dtype=np.float64).copy()
        tool[2] = 0.0
        tool -= jaw * float(tool @ jaw)
        tool_length = np.linalg.norm(tool)
        if tool_length < 0.5:
            raise ValueError("selected side-approach axis is too vertical")
        tool = (1 if tool_axis == "side_positive" else -1) * tool / tool_length
    return np.column_stack((np.cross(jaw, tool), jaw, tool))


def plan_tracked_grasp(
    model_path: Path,
    object_id: int,
    hand_positions: np.ndarray,
    hand_rotations: np.ndarray,
    hand_points: np.ndarray,
    max_width_m: float,
    contact_opening: Callable[[float], float],
    contact_overlap_mm: float = 6.0,
    insertion_depth_mm: float = 0.0,
) -> CubeGraspPlan:
    """Construct a guarded pilot target; scene collision is added by the IK stage."""
    hand_positions = np.asarray(hand_positions, dtype=np.float64)
    hand_rotations = np.asarray(hand_rotations, dtype=np.float64)
    hand_points = np.asarray(hand_points, dtype=np.float64)
    if not np.isfinite(contact_overlap_mm) or not 0.0 <= contact_overlap_mm <= 12.0:
        raise ValueError("contact_overlap_mm must be finite and in [0, 12]")
    if not np.isfinite(insertion_depth_mm) or not 0.0 <= insertion_depth_mm <= 15.0:
        raise ValueError("insertion_depth_mm must be finite and in [0, 15]")
    frame_count = len(hand_positions)
    if (frame_count < 40 or hand_positions.shape != (frame_count, 3)
            or hand_rotations.shape != (frame_count, 3, 3)
            or hand_points.shape != (frame_count, 3)
            or not np.isfinite(hand_positions).all()
            or not np.isfinite(hand_rotations).all()
            or not np.isfinite(hand_points).all()):
        raise ValueError("tracked grasp needs aligned hand poses and pinch points")
    with np.load(model_path, allow_pickle=False) as models:
        if str(models["schema"]) != OBJECT_MODEL_SCHEMA:
            raise ValueError("unsupported object-model schema")
        matching = np.flatnonzero(models["object_ids"] == object_id)
        if len(matching) != 1:
            raise ValueError("selected object ID is absent or ambiguous")
        object_index = int(matching[0])
        object_positions = np.asarray(models["translation_world"][:, object_index], dtype=np.float64)
        object_rotations = np.asarray(models["rotation_world_object"][:, object_index], dtype=np.float64)
        confidence = np.asarray(models["track_confidence"][:, object_index], dtype=np.float64)
        offsets = np.asarray(models["model_offsets"], dtype=int)
        surface = np.asarray(models["model_xyz_object"][offsets[object_index]:offsets[object_index + 1]], dtype=np.float64)
    if (object_positions.shape != (frame_count, 3)
            or object_rotations.shape != (frame_count, 3, 3)
            or confidence.shape != (frame_count,) or len(surface) < 100
            or not np.isfinite(object_positions).all()
            or not np.isfinite(object_rotations).all()
            or not np.isfinite(confidence).all()
            or not np.isfinite(surface).all()):
        raise ValueError("selected object track is incomplete")
    orthogonality = object_rotations @ np.swapaxes(object_rotations, -1, -2)
    if (np.max(np.abs(orthogonality - np.eye(3))) > 0.05
            or np.min(np.linalg.det(object_rotations)) < 0.95):
        raise ValueError("selected object track has invalid rotations")
    if np.quantile(confidence, 0.1) < 0.5:
        raise ValueError("selected object track has low confidence")
    initial_height = float(np.median(object_positions[:20, 2]))
    final_height = float(np.median(object_positions[-20:, 2]))
    near = np.linalg.norm(hand_points - object_positions, axis=1) < 0.065
    close_start = _first_sustained(near, 3)
    elevated = object_positions[:, 2] > initial_height + 0.04
    carry_start = _first_sustained(elevated, 3)
    if (close_start is None or carry_start is None or close_start < 4
            or carry_start - close_start < 8):
        raise ValueError("one clear approach and pickup are required")
    settled = np.abs(object_positions[:, 2] - final_height) < 0.02
    release_offset = _first_sustained(settled[carry_start + 10:], 4)
    if release_offset is None:
        raise ValueError("object placement was not detected")
    release_start = carry_start + 10 + release_offset
    far_offset = _first_sustained(~near[release_start + 4:], 3)
    if far_offset is None:
        raise ValueError("hand withdrawal was not detected")
    open_end = release_start + 4 + far_offset
    if (open_end >= frame_count or open_end - release_start < 5
            or np.linalg.norm(object_positions[release_start] - object_positions[close_start]) < 0.02):
        raise ValueError("one substantial grasp-and-place interval is required")

    local_offsets = np.einsum(
        "tji,tj->ti", object_rotations[carry_start:release_start],
        hand_points[carry_start:release_start] - object_positions[carry_start:release_start],
    )
    local_center = np.median(local_offsets, axis=0)
    local_center = np.clip(local_center, np.quantile(surface, 0.2, axis=0),
                           np.quantile(surface, 0.8, axis=0))
    local_center[2] = max(local_center[2], float(np.median(surface[:, 2])))
    observed_height = float(local_center[2])
    candidate_heights = np.arange(observed_height, observed_height + 0.061, 0.005)
    for height in candidate_heights:
        widths = []
        face_bounds = []
        for axis in (0, 1):
            other = 1 - axis
            subset = surface[(np.abs(surface[:, 2] - height) < 0.025)
                             & (np.abs(surface[:, other] - local_center[other]) < 0.03)]
            bounds = (np.quantile(subset[:, axis], [0.05, 0.95])
                      if len(subset) >= 30 else np.array([-np.inf, np.inf]))
            face_bounds.append(bounds)
            widths.append(float(bounds[1] - bounds[0]))
        if 0.015 < min(widths) < max_width_m - 0.005:
            local_center[2] = height
            break
    jaw_axis = int(np.argmin(widths))
    object_width = widths[jaw_axis]
    if not 0.015 < object_width < max_width_m - 0.005:
        raise ValueError("object cross-section does not fit the gripper")
    local_center[jaw_axis] = float(np.mean(face_bounds[jaw_axis]))
    carry_width = object_width - contact_overlap_mm / 1000.0
    if carry_width <= 0.0:
        raise ValueError("contact overlap exceeds the selected object width")
    grasp_opening = float(contact_opening(carry_width))
    object_center = object_positions + np.einsum("tij,j->ti", object_rotations, local_center)
    candidates = []
    sample = np.arange(carry_start, release_start, max(1, (release_start - carry_start) // 12))
    for jaw_sign in (-1, 1):
        for tool_axis in ("down", "side_positive", "side_negative"):
            try:
                sampled_rotations = np.stack([
                    _jaw_rotation(object_rotations[frame], jaw_axis, jaw_sign, tool_axis)
                    for frame in sample
                ])
            except ValueError:
                continue
            differences = Rotation.from_matrix(hand_rotations[sample]).inv() * Rotation.from_matrix(sampled_rotations)
            score = float(np.median(differences.magnitude()))
            candidates.append((score, jaw_sign, tool_axis))
    if not candidates:
        raise ValueError("no valid jaw and tool orientation for the selected object")
    correction, jaw_sign, tool_axis = min(candidates)
    if np.rad2deg(correction) > 90:
        raise ValueError("object-aligned grasp is too far from the observed tool orientation")
    rotations = np.stack([
        _jaw_rotation(matrix, jaw_axis, jaw_sign, tool_axis)
        for matrix in object_rotations
    ])

    result_positions = hand_positions.copy()
    result_rotations = hand_rotations.copy()
    opening = np.ones(frame_count, dtype=np.float64)
    prealign_start = max(0, close_start - 20)
    insert_end = close_start + max(4, (carry_start - close_start) // 2)
    if carry_start - insert_end < 6:
        raise ValueError("pickup has insufficient time for open-jaw insertion and closing")
    tool_extent = float(np.quantile(np.abs(surface[:, 1 - jaw_axis] - local_center[1 - jaw_axis]), 0.95))
    approach_distance = max(0.08, tool_extent + 0.04)
    pregrasp_center = object_center - approach_distance * rotations[:, :, 2]
    if insertion_depth_mm:
        insertion_local = object_rotations[carry_start].T @ rotations[carry_start, :, 2]
        local_center = local_center + insertion_depth_mm / 1000.0 * insertion_local
        object_center = object_positions + np.einsum("tij,j->ti", object_rotations, local_center)
    for frame in range(prealign_start, close_start + 1):
        fraction = (frame - prealign_start) / max(1, close_start - prealign_start)
        result_positions[frame] = (1 - fraction) * hand_positions[frame] + fraction * pregrasp_center[frame]
    for frame in range(close_start, insert_end + 1):
        fraction = (frame - close_start) / (insert_end - close_start)
        result_positions[frame] = (1 - fraction) * pregrasp_center[frame] + fraction * object_center[frame]
    result_positions[insert_end:release_start] = object_center[insert_end:release_start]
    result_rotations[prealign_start:close_start + 1] = Slerp(
        [prealign_start, close_start],
        Rotation.from_matrix([hand_rotations[prealign_start], rotations[close_start]]),
    )(np.arange(prealign_start, close_start + 1)).as_matrix()
    result_rotations[close_start:release_start] = rotations[close_start:release_start]
    close_complete = carry_start - 2
    opening[insert_end:close_complete + 1] = np.linspace(
        1.0, grasp_opening, close_complete - insert_end + 1,
    )
    opening[close_complete:release_start] = grasp_opening
    opening[release_start:open_end + 1] = np.linspace(
        grasp_opening, 1.0, open_end - release_start + 1,
    )
    release_center = object_center[release_start].copy()
    for frame in range(release_start, open_end + 1):
        fraction = (frame - release_start) / (open_end - release_start)
        result_positions[frame] = release_center + np.array([0.0, 0.0, 0.06 * fraction])
    blend_end = min(frame_count - 1, open_end + 15)
    for frame in range(open_end + 1, blend_end + 1):
        fraction = (frame - open_end) / (blend_end - open_end)
        result_positions[frame] = (1 - fraction) * result_positions[open_end] + fraction * hand_positions[frame]
    result_rotations[release_start:open_end + 1] = rotations[release_start]
    if blend_end > open_end:
        result_rotations[open_end:blend_end + 1] = Slerp(
            [open_end, blend_end],
            Rotation.from_matrix([rotations[release_start], hand_rotations[blend_end]]),
        )(np.arange(open_end, blend_end + 1)).as_matrix()
    phases = np.zeros(frame_count, dtype=np.uint8)
    phases[close_start:carry_start] = 1
    phases[carry_start:release_start] = 2
    phases[release_start:open_end] = 3
    phases[open_end:] = 4
    return CubeGraspPlan(result_positions, result_rotations, opening, phases, {
        "mode": "experimental_tracked_side_grasp_v1",
        "object_id": int(object_id),
        "phase_bounds": [close_start, carry_start, release_start, open_end],
        "prealign_start": prealign_start,
        "grasp_start": insert_end,
        "close_complete": close_complete,
        "pregrasp_distance_mm": approach_distance * 1000.0,
        "jaw_axis_local": jaw_axis,
        "jaw_sign": jaw_sign,
        "tool_axis": tool_axis,
        "orientation_correction_deg": float(np.rad2deg(correction)),
        "object_width_mm": object_width * 1000.0,
        "carry_width_mm": carry_width * 1000.0,
        "nominal_compression_mm": contact_overlap_mm,
        "insertion_depth_mm": insertion_depth_mm,
        "grasp_offset_local_m": local_center.tolist(),
        "grasp_height_shift_mm": float((local_center[2] - observed_height) * 1000),
        "travel_mm": float(np.linalg.norm(object_positions[release_start] - object_positions[close_start]) * 1000),
        "model_path": str(model_path),
    })
