"""Make an isolated oracle-geometry HO-Cap track for a grasp ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.hocap_mesh import aligned_mesh


def replace_surface(
    track_path: Path,
    output_path: Path,
    object_id: int,
    object_name: str,
    mesh_path: Path,
    gt_pose_path: Path,
    gt_frame: int,
) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    with np.load(track_path, allow_pickle=False) as track:
        arrays = {name: np.asarray(track[name]) for name in track.files}
    matches = np.flatnonzero(arrays["object_ids"] == object_id)
    if len(matches) != 1:
        raise ValueError("selected object ID is absent or ambiguous")
    object_index = int(matches[0])
    offsets = arrays["model_offsets"]
    surface = arrays["model_xyz_object"][offsets[object_index]:offsets[object_index + 1]]
    _, vertices, alignment = aligned_mesh(
        mesh_path, gt_pose_path, object_name, gt_frame,
        arrays["rotation_world_object"][0, object_index],
        arrays["translation_world"][0, object_index], surface,
    )
    surfaces = [
        vertices if index == object_index else
        arrays["model_xyz_object"][offsets[index]:offsets[index + 1]]
        for index in range(len(arrays["object_ids"]))
    ]
    arrays["model_xyz_object"] = np.concatenate(surfaces).astype(
        arrays["model_xyz_object"].dtype, copy=False,
    )
    arrays["model_offsets"] = np.concatenate(
        ([0], np.cumsum([len(points) for points in surfaces])),
    ).astype(offsets.dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {"source_track": str(track_path), "oracle_track": str(output_path),
            "object_id": object_id, "source_points": len(surface),
            "official_mesh_points": len(vertices), **alignment}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--official-mesh", type=Path, required=True)
    parser.add_argument("--gt-object-poses", type=Path, required=True)
    parser.add_argument("--gt-anchor-frame", type=int, required=True)
    arguments = parser.parse_args()
    print(replace_surface(
        arguments.track, arguments.output, arguments.object_id,
        arguments.object_name, arguments.official_mesh,
        arguments.gt_object_poses, arguments.gt_anchor_frame,
    ))
