"""Cube-transfer phase and face planning without robot hardware."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viki.perception import OBJECT_MODEL_SCHEMA
from viki.retarget.object_grasp import plan_cube_grasp


def _episode_inputs(tmp_path, *, valid_faces=True):
    frame_count = 100
    grid = np.linspace(-0.018, 0.018, 20)
    first = np.array([[0.02, y, z] for y in grid for z in grid])
    second = np.array([[x, 0.02, z] for x in grid for z in grid])
    surface = np.concatenate((first, second)) if valid_faces else first
    progress = np.clip((np.arange(frame_count) - 30) / 30, 0, 1)
    object_positions = np.zeros((frame_count, 1, 3))
    object_positions[:, 0, 0] = 0.4 * progress
    object_rotations = np.broadcast_to(np.eye(3), (frame_count, 1, 3, 3))
    contact = np.zeros((frame_count, 1), bool)
    contact[20:80] = True
    path = tmp_path / "object_models.npz"
    np.savez_compressed(
        path,
        schema=OBJECT_MODEL_SCHEMA,
        object_ids=np.array([2]),
        object_labels=np.array(["manipulated_object"]),
        translation_world=object_positions,
        rotation_world_object=object_rotations,
        contact=contact,
        track_confidence=np.ones((frame_count, 1)),
        model_offsets=np.array([0, len(surface)]),
        model_xyz_object=surface,
    )
    positions = object_positions[:, 0].copy()
    positions[:, 2] = 0.01
    positions[35:70, 1] += 0.04
    rotations = np.broadcast_to(
        (Rotation.from_euler("z", 20, degrees=True)
         * Rotation.from_euler("x", 180, degrees=True)).as_matrix(),
        (frame_count, 3, 3),
    ).copy()
    opening = np.full(frame_count, 0.75)
    return path, positions, rotations, opening


def test_cube_grasp_holds_width_and_follows_object(tmp_path):
    path, positions, rotations, opening = _episode_inputs(tmp_path)
    result = plan_cube_grasp(path, positions, rotations, opening, np.eye(4), 0.085, 0.04, 0.35)
    close_start, carry_start, release_start, open_end = result.diagnostics["phase_bounds"]
    assert close_start == 20
    assert carry_start < release_start < open_end
    np.testing.assert_allclose(result.opening[carry_start:release_start], 0.35)
    np.testing.assert_allclose(result.positions[carry_start:release_start, 1], positions[carry_start, 1])
    np.testing.assert_allclose(result.opening[open_end:], 0.9)
    assert np.all(result.phases[carry_start:release_start] == 2)
    np.testing.assert_allclose(result.rotations[carry_start:release_start, 2, 1], 0, atol=1e-8)
    np.testing.assert_allclose(result.rotations[carry_start:release_start, 2, 2], -1, atol=1e-8)
    np.testing.assert_allclose(opening, 0.75)


def test_cube_grasp_rejects_unverified_faces(tmp_path):
    path, positions, rotations, opening = _episode_inputs(tmp_path, valid_faces=False)
    with pytest.raises(ValueError, match="perpendicular cube side planes"):
        plan_cube_grasp(path, positions, rotations, opening, np.eye(4), 0.085, 0.04, 0.35)


def test_cube_grasp_rejects_missing_transfer(tmp_path):
    path, positions, rotations, opening = _episode_inputs(tmp_path)
    with np.load(path) as model:
        arrays = {key: model[key] for key in model.files}
    arrays["translation_world"] = np.zeros_like(arrays["translation_world"])
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="substantial cube transfer"):
        plan_cube_grasp(path, positions, rotations, opening, np.eye(4), 0.085, 0.04, 0.35)
