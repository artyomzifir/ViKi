"""Align an official HO-Cap object mesh to a ViKi object-track frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def aligned_mesh(
    mesh_path: Path,
    gt_pose_path: Path,
    object_name: str,
    gt_frame: int,
    predicted_rotation: np.ndarray,
    predicted_translation: np.ndarray,
    observed_surface: np.ndarray,
) -> tuple[str, np.ndarray, dict]:
    with np.load(gt_pose_path, allow_pickle=False) as reference:
        names = [str(name) for name in reference["object_ids"]]
        if object_name not in names or gt_frame < 0 or gt_frame >= len(reference["valid"]):
            raise ValueError("HO-Cap object or anchor frame is absent")
        object_index = names.index(object_name)
        if not reference["valid"][gt_frame, object_index]:
            raise ValueError("HO-Cap object pose is invalid at the anchor frame")
        world_from_mesh = np.asarray(reference["poses_world"][gt_frame, object_index], dtype=float)
    world_from_track = np.eye(4)
    world_from_track[:3, :3] = predicted_rotation
    world_from_track[:3, 3] = predicted_translation
    track_from_mesh = np.linalg.inv(world_from_track) @ world_from_mesh
    if (not np.isfinite(track_from_mesh).all()
            or not np.isclose(np.linalg.det(track_from_mesh[:3, :3]), 1.0, atol=1e-3)):
        raise ValueError("invalid official-mesh alignment")

    vertices = []
    lines = []
    for line in mesh_path.read_text().splitlines():
        if line.startswith("v "):
            point = np.array([float(value) for value in line.split()[1:4]])
            transformed = track_from_mesh[:3, :3] @ point + track_from_mesh[:3, 3]
            vertices.append(transformed)
            lines.append("v " + " ".join(f"{value:.9g}" for value in transformed))
        elif line.startswith("vn "):
            normal = np.array([float(value) for value in line.split()[1:4]])
            transformed = track_from_mesh[:3, :3] @ normal
            lines.append("vn " + " ".join(f"{value:.9g}" for value in transformed))
        else:
            lines.append(line)
    if len(vertices) < 100:
        raise ValueError("official object mesh has too few vertices")
    vertices_array = np.asarray(vertices)
    distances = cKDTree(vertices_array).query(observed_surface)[0]
    median_mm, p95_mm = np.percentile(distances, [50, 95]) * 1000
    if median_mm > 20 or p95_mm > 30:
        raise ValueError("official mesh does not align with the observed object surface")
    return "\n".join(lines) + "\n", vertices_array, {
        "official_mesh": str(mesh_path),
        "official_pose": str(gt_pose_path),
        "official_object_name": object_name,
        "official_anchor_frame": gt_frame,
        "track_from_mesh": track_from_mesh.tolist(),
        "cloud_to_mesh_vertex_median_mm": float(median_mm),
        "cloud_to_mesh_vertex_p95_mm": float(p95_mm),
    }
