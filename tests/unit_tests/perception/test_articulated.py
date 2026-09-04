"""Landmark-only articulated geometry must repair gaps without moving wrist."""

from __future__ import annotations

import json
import numpy as np
import pytest

from viki.contracts import HAND_LM_COUNT, LM
from viki.perception.articulated import ArticulatedConfig, fit_landmark_trajectory


pytest.importorskip("pinocchio")


def _open_hand(wrist: np.ndarray) -> np.ndarray:
    points = np.empty((HAND_LM_COUNT, 3), np.float32)
    points[int(LM.WRIST)] = wrist
    roots = {
        "thumb": np.array([0.020, 0.045, 0.0]),
        "index": np.array([0.060, 0.030, 0.0]),
        "middle": np.array([0.065, 0.000, 0.0]),
        "ring": np.array([0.060, -0.025, 0.0]),
        "pinky": np.array([0.052, -0.045, 0.0]),
    }
    lengths = {
        "thumb": np.array([0.038, 0.030, 0.024]),
        "index": np.array([0.043, 0.027, 0.020]),
        "middle": np.array([0.046, 0.029, 0.022]),
        "ring": np.array([0.042, 0.027, 0.020]),
        "pinky": np.array([0.034, 0.022, 0.017]),
    }
    from viki.perception.hand_model import FINGERS

    for finger, chain in FINGERS.items():
        point = roots[finger].copy()
        points[int(chain[0])] = wrist + point
        for edge, length in enumerate(lengths[finger]):
            point = point + np.array([length, 0.0, 0.0])
            points[int(chain[edge + 1])] = wrist + point
    return points


def test_projection_repairs_zero_confidence_collapse_and_preserves_wrist():
    T = 14
    source = np.stack([
        _open_hand(np.array([0.004 * t, 0.0, 0.75], np.float32))
        for t in range(T)
    ])
    confidence = np.ones((T, HAND_LM_COUNT), np.float32)
    # Reproduce the real failure: a long interpolated span has no triangulated
    # evidence and all fingers fold into the wrist, while wrist motion is good.
    confidence[5:10] = 0.0
    for t in range(5, 10):
        source[t, 1:] = source[t, 0] + 0.002
    original = source.copy()

    result = fit_landmark_trajectory(
        source,
        np.arange(HAND_LM_COUNT, dtype=np.int32),
        confidence,
        np.ones(T, bool),
        ArticulatedConfig(calibration_frames=4),
        optimize=False,
    )
    fitted = np.asarray(result["projected_points"])

    np.testing.assert_array_equal(source, original)  # input not mutated
    np.testing.assert_allclose(fitted[:, 0], source[:, 0], atol=1e-7)
    assert not np.asarray(result["support_mask"])[5:10].any()
    # Every model edge is fixed across the episode, including the repaired gap.
    for a, b in ((0, 5), (5, 6), (6, 7), (7, 8), (0, 17), (17, 18)):
        length = np.linalg.norm(fitted[:, a] - fitted[:, b], axis=1)
        assert float(np.ptp(length)) < 1e-6
        assert float(length[7]) > 0.015
    assert result["projected_metrics"]["quality_gate"]["structural_pass"] is True


def test_geometry_confidence_rejects_a_single_giant_tip():
    from viki.perception.articulated import geometry_anchor_confidence

    points = np.stack([_open_hand(np.array([0.0, 0.0, 0.75])) for _ in range(10)])
    confidence = np.ones((10, HAND_LM_COUNT), np.float32)
    points[4, int(LM.INDEX_TIP)] += np.array([0.30, 0.0, 0.0])

    gated, _ = geometry_anchor_confidence(points, confidence, ArticulatedConfig())

    assert gated[4, int(LM.INDEX_TIP)] == 0.0
    assert gated[4, int(LM.INDEX_DIP)] > 0.0
    assert gated[4, int(LM.MIDDLE_TIP)] > 0.0


def test_install_overlay_keeps_active_clean_arrays(tmp_path):
    from viki.episode import new_episode
    from viki.perception.articulated import install_articulated_overlay
    from viki.prepare.baseline import file_sha256

    ep = new_episode(tmp_path)
    T = 4
    points = np.stack([
        _open_hand(np.array([0.01 * t, 0.0, 0.75])) for t in range(T)
    ])
    clean = {
        "positions": points[:, 0].copy(),
        "rotations": np.tile(np.eye(3), (T, 1, 1)),
        "valid": np.ones(T, bool),
        "omega": np.ones(T, np.float32),
        "gripper": np.zeros(T, bool),
        "timestamps": np.arange(T, dtype=np.int64),
        "raw_points": points.copy(),
        "smoothed_points": points.copy(),
        "landmark_confidence": np.ones((T, 21), np.float32),
        "landmark_ids": np.arange(21, dtype=np.int32),
        "coordinate_frame": np.asarray("robot_base"),
        "perception_fuse_mode": np.asarray("triangulate"),
        "pose_source": np.asarray("landmarks"),
    }
    np.savez_compressed(ep.cln_npz, **clean)
    baseline = (
        ep.intermediates_dir / "baselines" /
        "clean-triangulated-landmarks-v1" / "cln.npz"
    )
    baseline.parent.mkdir(parents=True)
    np.savez_compressed(baseline, **clean)
    baseline.with_name("manifest.json").write_text(json.dumps({
        "artifact_sha256": file_sha256(baseline),
    }))

    candidate = (
        ep.intermediates_dir / "geometry" / "articulated-landmarks-v1" /
        "50_optimized.npz"
    )
    candidate.parent.mkdir(parents=True)
    overlay = dict(clean)
    overlay.update({
        "hand_fit_positions": points[:, 0] + 0.001,
        "hand_fit_rotations": np.tile(np.eye(3), (T, 1, 1)),
        "hand_fit_joint_angles": np.zeros((T, 27), np.float32),
        "hand_fit_model_nq": np.int64(27),
        "hand_fit_capsules": np.zeros((T, 16, 2, 3), np.float32),
        "hand_fit_capsule_radii": np.ones(16, np.float32) * 0.01,
        "hand_fit_metrics_json": np.asarray("{}"),
        "geometry_recipe": np.asarray("articulated-landmarks-v1"),
    })
    np.savez_compressed(candidate, **overlay)

    result = install_articulated_overlay(ep)

    assert result["clean_core_unchanged"] is True
    with np.load(ep.cln_npz, allow_pickle=False) as active:
        np.testing.assert_array_equal(active["smoothed_points"], clean["smoothed_points"])
        np.testing.assert_array_equal(active["positions"], clean["positions"])
        np.testing.assert_array_equal(
            active["hand_fit_positions"], overlay["hand_fit_positions"],
        )
        assert active["hand_fit_overlay_variant"].item() == (
            "articulated-landmarks-v1:optimized"
        )
