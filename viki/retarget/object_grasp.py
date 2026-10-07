"""Experimental cube-aware targets for a two-finger tabletop transfer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from viki.perception import OBJECT_MODEL_SCHEMA


@dataclass(frozen=True)
class CubeGraspPlan:
    positions: np.ndarray
    rotations: np.ndarray
    opening: np.ndarray
    phases: np.ndarray
    diagnostics: dict


def _first_sustained(mask: np.ndarray, count: int) -> int | None:
    runs = np.convolve(np.asarray(mask, dtype=np.int8), np.ones(count, dtype=np.int8), "valid")
    found = np.flatnonzero(runs == count)
    return int(found[0]) if len(found) else None


def _face_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find two well-supported, perpendicular, near-vertical cube side planes."""
    remaining = np.asarray(points, dtype=np.float64)
    if remaining.ndim != 2 or remaining.shape[1] != 3 or len(remaining) < 100:
        raise ValueError("cube model has too few surface points")
    random = np.random.default_rng(42)
    candidates: list[tuple[int, np.ndarray]] = []
    for _ in range(5):
        best = np.empty(0, dtype=np.int64)
        for _ in range(1200):
            triangle = remaining[random.choice(len(remaining), 3, replace=False)]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            magnitude = np.linalg.norm(normal)
            if magnitude < 1e-5:
                continue
            normal /= magnitude
            inliers = np.flatnonzero(np.abs((remaining - triangle[0]) @ normal) < 0.0025)
            if len(inliers) > len(best):
                best = inliers
        if len(best) < 35:
            break
        patch = remaining[best]
        _, singular, axes = np.linalg.svd(patch - patch.mean(axis=0), full_matrices=False)
        normal = axes[-1]
        if (abs(normal[2]) < 0.25 and singular[1] > 0.045
                and singular[2] / singular[1] < 0.15):
            horizontal = normal.copy()
            horizontal[2] = 0.0
            horizontal /= np.linalg.norm(horizontal)
            candidates.append((len(best), horizontal))
        remaining = np.delete(remaining, best, axis=0)
        if len(remaining) < 35:
            break
    candidates.sort(key=lambda item: item[0], reverse=True)
    for first_index, (_, first) in enumerate(candidates):
        for _, second in candidates[first_index + 1:]:
            if abs(float(first @ second)) < 0.25:
                return first, second
    raise ValueError("two supported perpendicular cube side planes were not found")


def _face_aligned_rotation(hand_rotation: np.ndarray, normals: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, float]:
    jaw = np.asarray(hand_rotation[:, 1], dtype=np.float64)
    jaw[2] = 0.0
    horizontal_length = np.linalg.norm(jaw)
    if horizontal_length < 0.2:
        raise ValueError("observed jaw axis is too vertical to select a side face")
    jaw /= horizontal_length
    candidates = [direction * normal for normal in normals for direction in (-1, 1)]
    aligned = max(candidates, key=lambda normal: float(normal @ jaw))
    down = np.array([0.0, 0.0, -1.0])
    rotation = np.column_stack((np.cross(aligned, down), aligned, down))
    angle = float(np.rad2deg(Rotation.from_matrix(hand_rotation.T @ rotation).magnitude()))
    return rotation, angle


def plan_cube_grasp(
    model_path: Path,
    positions: np.ndarray,
    rotations: np.ndarray,
    opening: np.ndarray,
    calibration_from_rig: np.ndarray,
    max_width_m: float,
    cube_side_m: float,
    grasp_opening: float,
) -> CubeGraspPlan:
    """Condition one transfer on observed cube translation and two fitted side faces."""
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    opening = np.asarray(opening, dtype=np.float64)
    frame_count = len(positions)
    if (positions.shape != (frame_count, 3) or rotations.shape != (frame_count, 3, 3)
            or opening.shape != (frame_count,) or frame_count < 30):
        raise ValueError("cube grasp needs aligned position, rotation and opening tracks")
    if not 0 < cube_side_m < max_width_m:
        raise ValueError("cube side must fit inside the selected gripper")
    if not 0 < grasp_opening < 1:
        raise ValueError("contact-calibrated grasp command must be between open and closed")
    with np.load(model_path, allow_pickle=False) as models:
        if str(models["schema"]) != OBJECT_MODEL_SCHEMA:
            raise ValueError("cube grasp needs the current object-model schema")
        indices = np.flatnonzero(models["object_labels"] == "manipulated_object")
        if len(indices) != 1:
            raise ValueError("cube grasp needs exactly one manipulated object")
        index = int(indices[0])
        object_id = int(models["object_ids"][index])
        object_positions = np.asarray(models["translation_world"][:, index], dtype=np.float64)
        object_rotations = np.asarray(models["rotation_world_object"][:, index], dtype=np.float64)
        contact = np.asarray(models["contact"][:, index], dtype=bool)
        confidence = np.asarray(models["track_confidence"][:, index], dtype=np.float64)
        offsets = np.asarray(models["model_offsets"], dtype=np.int64)
        surface = np.asarray(models["model_xyz_object"][offsets[index]:offsets[index + 1]], dtype=np.float64)
    if (object_positions.shape != positions.shape or object_rotations.shape != rotations.shape
            or contact.shape != opening.shape or confidence.shape != opening.shape
            or not np.isfinite(object_positions).all() or not np.isfinite(surface).all()):
        raise ValueError("cube object track is incomplete or not aligned with the hand")
    if np.quantile(confidence, 0.1) < 0.5:
        raise ValueError("cube track confidence is too low for object-aware IK")
    anchor = np.asarray(calibration_from_rig, dtype=np.float64)
    if anchor.shape != (4, 4) or not np.isfinite(anchor).all():
        raise ValueError("rig-to-calibration transform is invalid")
    object_calibration = object_positions @ anchor[:3, :3].T + anchor[:3, 3]
    travel = np.linalg.norm(object_calibration[-1] - object_calibration[0])
    if travel < 0.08:
        raise ValueError("no substantial cube transfer was observed")
    carry_start = _first_sustained(np.linalg.norm(object_calibration - object_calibration[0], axis=1) > 0.012, 3)
    if carry_start is None:
        raise ValueError("cube pickup could not be located")
    settled = np.linalg.norm(object_calibration - object_calibration[-1], axis=1) < 0.012
    release_offset = _first_sustained(settled[carry_start + 10:], 5)
    if release_offset is None:
        raise ValueError("cube placement could not be located")
    release_start = carry_start + 10 + release_offset
    earlier_contact = np.flatnonzero(contact[:carry_start])
    later_contact = np.flatnonzero(contact[release_start:])
    if not len(earlier_contact) or not len(later_contact):
        raise ValueError("contact track does not bracket cube transport")
    close_start = int(earlier_contact[0])
    open_end = min(frame_count - 1, release_start + int(later_contact[-1]) + 1)
    if carry_start - close_start < 4 or open_end - release_start < 4:
        raise ValueError("grasp or release phase is too short")

    surface_rig = surface @ object_rotations[close_start].T + object_positions[close_start]
    surface_calibration = surface_rig @ anchor[:3, :3].T + anchor[:3, 3]
    normals = _face_normals(surface_calibration)
    aligned_rotation, correction_deg = _face_aligned_rotation(rotations[carry_start], normals)
    if correction_deg > 100:
        raise ValueError("face-aligned grasp is too far from the observed hand pose")

    result_position = positions.copy()
    result_rotation = rotations.copy()
    result_opening = opening.copy()
    phases = np.zeros(frame_count, dtype=np.uint8)
    phases[close_start:carry_start] = 1
    phases[carry_start:release_start] = 2
    phases[release_start:open_end] = 3
    phases[open_end:] = 4
    grasp_width = cube_side_m - 0.002
    result_opening[close_start:carry_start + 1] = np.linspace(
        opening[close_start], grasp_opening, carry_start - close_start + 1
    )
    result_opening[carry_start:release_start] = grasp_opening
    result_opening[release_start:open_end + 1] = np.linspace(
        grasp_opening, 0.9, open_end - release_start + 1
    )
    result_opening[open_end:] = np.maximum(result_opening[open_end:], 0.9)

    grasp_offset = positions[carry_start] - object_calibration[carry_start]
    result_position[carry_start:release_start] = object_calibration[carry_start:release_start] + grasp_offset
    for frame in range(release_start, open_end):
        blend = (frame - release_start) / (open_end - release_start)
        rigid_position = object_calibration[frame] + grasp_offset
        result_position[frame] = (1 - blend) * rigid_position + blend * positions[frame]
    start_rotation = Rotation.from_matrix(rotations[close_start])
    face_rotation = Rotation.from_matrix(aligned_rotation)
    result_rotation[close_start:carry_start + 1] = Slerp(
        [close_start, carry_start], Rotation.concatenate([start_rotation, face_rotation])
    )(np.arange(close_start, carry_start + 1)).as_matrix()
    result_rotation[carry_start:release_start] = aligned_rotation
    result_rotation[release_start:open_end + 1] = Slerp(
        [release_start, open_end], Rotation.concatenate([
            face_rotation, Rotation.from_matrix(rotations[open_end])
        ])
    )(np.arange(release_start, open_end + 1)).as_matrix()
    return CubeGraspPlan(result_position, result_rotation, result_opening, phases, {
        "mode": "experimental_cube_face_transfer",
        "object_id": object_id,
        "phase_bounds": [close_start, carry_start, release_start, open_end],
        "cube_side_mm": cube_side_m * 1000,
        "carry_width_mm": grasp_width * 1000,
        "nominal_command_width_mm": grasp_opening * max_width_m * 1000,
        "face_orientation_correction_deg": correction_deg,
        "travel_mm": float(travel * 1000),
        "face_normals_calibration": [normal.tolist() for normal in normals],
    })
