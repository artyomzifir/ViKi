"""Official HO-Cap mesh alignment used only by the MuJoCo pilot."""

import numpy as np
import pytest

from experiments.hocap_mesh import aligned_mesh
from experiments.hocap_mesh_track import replace_surface


def _inputs(tmp_path):
    mesh_path = tmp_path / "object.obj"
    points = np.stack(np.meshgrid(
        np.linspace(-0.04, 0.04, 5),
        np.linspace(-0.02, 0.02, 5),
        np.linspace(-0.08, 0.08, 5),
    ), axis=-1).reshape(-1, 3)
    mesh_path.write_text("".join(
        "v " + " ".join(map(str, point)) + "\n" for point in points
    ))
    pose = np.eye(4)
    pose[:3, 3] = [0.1, -0.2, 0.3]
    pose_path = tmp_path / "poses.npz"
    np.savez_compressed(
        pose_path, object_ids=np.array(["G10_4"]),
        poses_world=pose[None, None], valid=np.array([[True]]),
    )
    track_translation = np.array([0.1, -0.2, 0.2])
    return mesh_path, pose_path, track_translation, points


def test_official_mesh_is_expressed_in_tracker_object_frame(tmp_path):
    mesh_path, pose_path, track_translation, points = _inputs(tmp_path)
    text, aligned, metadata = aligned_mesh(
        mesh_path, pose_path, "G10_4", 0, np.eye(3), track_translation,
        points + [0, 0, 0.1],
    )
    np.testing.assert_allclose(aligned, points + [0, 0, 0.1])
    np.testing.assert_allclose(np.array(metadata["track_from_mesh"])[:3, 3], [0, 0, 0.1])
    assert len([line for line in text.splitlines() if line.startswith("v ")]) == 125
    assert metadata["cloud_to_mesh_vertex_p95_mm"] < 1e-9


def test_official_mesh_rejects_wrong_frame_or_alignment(tmp_path):
    mesh_path, pose_path, track_translation, points = _inputs(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        aligned_mesh(mesh_path, pose_path, "G10_4", 1, np.eye(3),
                     track_translation, points)
    with pytest.raises(ValueError, match="does not align"):
        aligned_mesh(mesh_path, pose_path, "G10_4", 0, np.eye(3),
                     track_translation, points + [0, 0, 1])


def test_oracle_geometry_variant_preserves_other_object_and_pose_track(tmp_path):
    mesh_path, pose_path, track_translation, points = _inputs(tmp_path)
    source = tmp_path / "source.npz"
    output = tmp_path / "oracle.npz"
    surfaces = np.concatenate((points[:20], points + [0, 0, 0.1]))
    np.savez_compressed(
        source, object_ids=np.array([3, 4]),
        model_offsets=np.array([0, 20, len(surfaces)]),
        model_xyz_object=surfaces,
        rotation_world_object=np.broadcast_to(np.eye(3), (2, 2, 3, 3)),
        translation_world=np.broadcast_to(track_translation, (2, 2, 3)),
        track_confidence=np.ones((2, 2)),
    )
    report = replace_surface(
        source, output, 4, "G10_4", mesh_path, pose_path, 0,
    )
    assert report["official_mesh_points"] == 125
    with np.load(source) as original, np.load(output) as variant:
        np.testing.assert_array_equal(
            variant["model_xyz_object"][:20], original["model_xyz_object"][:20],
        )
        np.testing.assert_array_equal(
            variant["translation_world"], original["translation_world"],
        )
        np.testing.assert_array_equal(variant["model_offsets"], [0, 20, 145])
