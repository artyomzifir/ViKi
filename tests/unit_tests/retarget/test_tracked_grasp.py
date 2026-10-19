"""Guardrails for the opt-in rigid-object grasp pilot."""

import numpy as np
import pytest

from viki.perception import OBJECT_MODEL_SCHEMA
from viki.retarget.tracked_grasp import plan_tracked_grasp
from viki.retarget.run import config_from_options


def _example(tmp_path):
    frames = 100
    centers = np.zeros((frames, 1, 3), dtype=np.float64)
    centers[:, 0, 2] = 0.12
    centers[35:45, 0, 2] = np.linspace(0.12, 0.20, 10)
    centers[45:65, 0, 2] = 0.20
    centers[40:65, 0, 0] = np.linspace(0, 0.15, 25)
    centers[65:75, 0, 2] = np.linspace(0.20, 0.12, 10)
    centers[65:, 0, 0] = 0.15
    surface = np.stack(np.meshgrid(
        np.linspace(-0.04, 0.04, 11),
        np.linspace(-0.03, 0.03, 11),
        np.linspace(-0.07, 0.07, 11),
    ), axis=-1).reshape(-1, 3)
    model_path = tmp_path / "object_models.npz"
    np.savez_compressed(
        model_path, schema=OBJECT_MODEL_SCHEMA, object_ids=np.array([4]),
        translation_world=centers,
        rotation_world_object=np.broadcast_to(np.eye(3), (frames, 1, 3, 3)),
        track_confidence=np.ones((frames, 1)),
        model_offsets=np.array([0, len(surface)]), model_xyz_object=surface,
    )
    hand_points = centers[:, 0].copy()
    hand_points[:25, 0] = -0.2
    hand_points[80:, 0] = 0.4
    hand_points[:, 2] += 0.02
    return model_path, hand_points


def test_tracked_grasp_carries_relative_to_object_and_sets_contact_width(tmp_path):
    model_path, hand_points = _example(tmp_path)
    rotations = np.broadcast_to(np.eye(3), (len(hand_points), 3, 3))
    result = plan_tracked_grasp(
        model_path, 4, hand_points, rotations, hand_points, 0.085,
        lambda width: width / 0.085,
    )
    close_start, carry_start, release_start, open_end = result.diagnostics["phase_bounds"]
    assert close_start < carry_start < release_start < open_end
    grasp_start = result.diagnostics["grasp_start"]
    assert close_start < grasp_start < carry_start
    assert result.diagnostics["pregrasp_distance_mm"] >= 80
    assert result.opening[grasp_start] == 1.0
    assert result.diagnostics["jaw_axis_local"] == 1
    assert result.diagnostics["tool_axis"].startswith("side_")
    assert 0.05 < result.diagnostics["carry_width_mm"] / 1000 < 0.07
    assert np.all(result.phases[carry_start:release_start] == 2)
    assert np.ptp(result.positions[carry_start:release_start, 0]) > 0.1
    assert np.allclose(result.opening[carry_start:release_start], result.opening[carry_start])
    assert result.opening[0] == result.opening[-1] == 1.0


def test_tracked_grasp_rejects_wrong_object_or_width(tmp_path):
    model_path, hand_points = _example(tmp_path)
    rotations = np.broadcast_to(np.eye(3), (len(hand_points), 3, 3))
    with pytest.raises(ValueError, match="absent"):
        plan_tracked_grasp(model_path, 9, hand_points, rotations, hand_points,
                           0.085, lambda width: width / 0.085)
    with pytest.raises(ValueError, match="fit"):
        plan_tracked_grasp(model_path, 4, hand_points, rotations, hand_points,
                           0.05, lambda width: width / 0.05)


def test_tracked_grasp_accepts_short_placement_but_rejects_negligible_travel(tmp_path):
    model_path, hand_points = _example(tmp_path)
    rotations = np.broadcast_to(np.eye(3), (len(hand_points), 3, 3))
    with np.load(model_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    centers = arrays["translation_world"].copy()
    centers[:, 0, 0] *= 0.2
    arrays["translation_world"] = centers
    np.savez_compressed(model_path, **arrays)
    hand_points[25:80, 0] = centers[25:80, 0, 0]
    result = plan_tracked_grasp(
        model_path, 4, hand_points, rotations, hand_points, 0.085,
        lambda width: width / 0.085,
    )
    assert 20 < result.diagnostics["travel_mm"] < 40

    arrays["translation_world"][:, 0, 0] *= 0.1
    np.savez_compressed(model_path, **arrays)
    hand_points[25:80, 0] = arrays["translation_world"][25:80, 0, 0]
    with pytest.raises(ValueError, match="substantial grasp-and-place"):
        plan_tracked_grasp(
            model_path, 4, hand_points, rotations, hand_points, 0.085,
            lambda width: width / 0.085,
        )


def test_tracked_grasp_uses_narrower_supported_upper_cross_section(tmp_path):
    model_path, hand_points = _example(tmp_path)
    with np.load(model_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    surface = arrays["model_xyz_object"].copy()
    upper = surface[:, 2] > 0.035
    surface[:, 0] *= np.where(upper, 0.8, 1.4)
    surface[:, 1] *= np.where(upper, 1.0, 1.7)
    arrays["model_xyz_object"] = surface
    np.savez_compressed(model_path, **arrays)
    rotations = np.broadcast_to(np.eye(3), (len(hand_points), 3, 3))
    result = plan_tracked_grasp(
        model_path, 4, hand_points, rotations, hand_points, 0.085,
        lambda width: width / 0.085,
    )
    assert result.diagnostics["grasp_height_shift_mm"] >= 15
    assert result.diagnostics["object_width_mm"] < 80


def test_tracked_grasp_contact_overlap_is_explicit_and_bounded(tmp_path):
    model_path, hand_points = _example(tmp_path)
    rotations = np.broadcast_to(np.eye(3), (len(hand_points), 3, 3))
    baseline = plan_tracked_grasp(
        model_path, 4, hand_points, rotations, hand_points, 0.085,
        lambda width: width / 0.085,
    )
    tighter = plan_tracked_grasp(
        model_path, 4, hand_points, rotations, hand_points, 0.085,
        lambda width: width / 0.085, 8.0,
    )
    np.testing.assert_allclose(baseline.positions, tighter.positions)
    assert tighter.diagnostics["nominal_compression_mm"] == 8.0
    assert tighter.diagnostics["carry_width_mm"] == pytest.approx(
        baseline.diagnostics["carry_width_mm"] - 2.0
    )
    assert tighter.opening[tighter.diagnostics["phase_bounds"][1]] < baseline.opening[
        baseline.diagnostics["phase_bounds"][1]]
    for overlap in (float("nan"), -1.0, 13.0):
        with pytest.raises(ValueError, match="contact_overlap_mm"):
            plan_tracked_grasp(model_path, 4, hand_points, rotations, hand_points,
                               0.085, lambda width: width / 0.085, overlap)


def test_insertion_depth_moves_grasp_into_object_without_changing_jaw_gap(tmp_path):
    model_path, hand_points = _example(tmp_path)
    rotations = np.broadcast_to(np.eye(3), (len(hand_points), 3, 3))
    baseline = plan_tracked_grasp(
        model_path, 4, hand_points, rotations, hand_points, 0.085,
        lambda width: width / 0.085,
    )
    deeper = plan_tracked_grasp(
        model_path, 4, hand_points, rotations, hand_points, 0.085,
        lambda width: width / 0.085, insertion_depth_mm=10.0,
    )
    close_start, carry_start, _, _ = baseline.diagnostics["phase_bounds"]
    np.testing.assert_allclose(deeper.positions[close_start], baseline.positions[close_start])
    tool = baseline.rotations[carry_start, :, 2]
    np.testing.assert_allclose(
        deeper.positions[carry_start] - baseline.positions[carry_start],
        0.01 * tool, atol=1e-6,
    )
    np.testing.assert_allclose(deeper.opening, baseline.opening)
    np.testing.assert_allclose(deeper.rotations, baseline.rotations)
    assert deeper.diagnostics["insertion_depth_mm"] == 10.0
    for depth in (float("nan"), -1.0, 16.0):
        with pytest.raises(ValueError, match="insertion_depth_mm"):
            plan_tracked_grasp(model_path, 4, hand_points, rotations, hand_points,
                               0.085, lambda width: width / 0.085,
                               insertion_depth_mm=depth)


def test_insertion_depth_option_is_track_only():
    assert config_from_options("ur10").object_insertion_depth_mm == 0.0
    track = {"object_grasp": True, "object_track_path": "/tmp/selected.npz", "object_id": 4}
    assert config_from_options("ur10", {**track, "object_insertion_depth_mm": 10.0}).object_insertion_depth_mm == 10.0
    for invalid in (float("nan"), -1.0, 16.0):
        with pytest.raises(ValueError, match="object_insertion_depth_mm"):
            config_from_options("ur10", {**track, "object_insertion_depth_mm": invalid})
    with pytest.raises(ValueError, match="requires an explicit object track"):
        config_from_options("ur10", {"object_insertion_depth_mm": 10.0})


def test_overlap_option_is_track_only_and_keeps_cube_default():
    assert config_from_options("ur10").object_contact_overlap_mm == 6.0
    track = {"object_grasp": True, "object_track_path": "/tmp/selected.npz", "object_id": 4}
    assert config_from_options("ur10", {**track, "object_contact_overlap_mm": 4.0}).object_contact_overlap_mm == 4.0
    for invalid in (float("nan"), -1.0, 13.0):
        with pytest.raises(ValueError, match="object_contact_overlap_mm"):
            config_from_options("ur10", {**track, "object_contact_overlap_mm": invalid})
    with pytest.raises(ValueError, match="requires an explicit object track"):
        config_from_options("ur10", {"object_contact_overlap_mm": 4.0})
