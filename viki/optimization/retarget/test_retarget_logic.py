"""Retargeting tests that do not require PINK/Pinocchio."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import viki.config as viki_config

from viki.optimization.retarget.retarget_rgb_only import (
    R_DEFAULT,
    align_rotations_to_initial,
    build_direct_rotation_targets,
    effective_orientation_cost,
    fill_invalid_rotations,
    load_retarget_input,
    load_smoothed_targets,
    normalize_robot,
    output_traj_path,
    resolve_trajectory_scale_origin,
    should_apply_legacy_transform,
    transform_points,
    transform_rotations_to_robot,
)


class FakePin:
    class SE3:
        def __init__(self, rotation, translation):
            self.rotation = np.asarray(rotation, dtype=np.float64)
            self.translation = np.asarray(translation, dtype=np.float64)


# Calibration shift applied by the smooth-stage loader for non-legacy frames.
OFFSET = np.asarray(viki_config.TARGET_OFFSET, dtype=np.float64) - np.asarray(
    viki_config.ROBOT_BASE_OFFSET, dtype=np.float64
)


class RetargetLogicTests(unittest.TestCase):
    def test_coordinate_frame_controls_legacy_transform(self) -> None:
        self.assertFalse(should_apply_legacy_transform("robot_base"))
        self.assertTrue(should_apply_legacy_transform("viki_world_or_camera"))

    def test_auto_scale_origin_uses_robot_base_for_calibrated_input(self) -> None:
        self.assertEqual(
            resolve_trajectory_scale_origin("auto", "robot_base"),
            "robot_base",
        )
        self.assertEqual(
            resolve_trajectory_scale_origin("auto", "viki_world_or_camera"),
            "initial_wrist",
        )

    def test_output_trajectory_path_uses_hdf5(self) -> None:
        robot = normalize_robot("ur10")
        sample = Path("sample.npz")
        self.assertEqual(output_traj_path(Path("out"), sample, robot).name, "out_traj.h5")
        self.assertEqual(output_traj_path(Path("out.npz"), sample, robot).name, "out.h5")
        self.assertEqual(output_traj_path(Path("out.hdf5"), sample, robot).name, "out.hdf5")

    def test_invalid_rotations_hold_single_valid_frame(self) -> None:
        valid_rotation = np.eye(3)
        filled, valid = fill_invalid_rotations([None, valid_rotation, None])
        np.testing.assert_allclose(filled[0], valid_rotation)
        np.testing.assert_allclose(filled[1], valid_rotation)
        np.testing.assert_allclose(filled[2], valid_rotation)
        np.testing.assert_array_equal(valid, [False, True, False])

    def test_invalid_rotations_are_slerp_interpolated(self) -> None:
        angle = np.pi / 2.0
        end = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        filled, valid = fill_invalid_rotations([np.eye(3), None, end])
        middle_angle = np.pi / 4.0
        expected_middle = np.array(
            [
                [np.cos(middle_angle), -np.sin(middle_angle), 0.0],
                [np.sin(middle_angle), np.cos(middle_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(filled[1], expected_middle, atol=1e-12)
        np.testing.assert_array_equal(valid, [True, False, True])

    def test_align_rotations_to_initial_maps_first_frame_to_target(self) -> None:
        angle = np.pi / 2.0
        first = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        second = np.eye(3, dtype=np.float64)
        initial_target = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        aligned = align_rotations_to_initial(np.stack([first, second]), initial_target)
        np.testing.assert_allclose(aligned[0], initial_target, atol=1e-12)
        np.testing.assert_allclose(aligned[1], second @ first.T @ initial_target, atol=1e-12)

    def test_smoothed_targets_load_positions_rotations_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-test.npz"
            positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            rotations = np.stack([np.eye(3), np.full((3, 3), np.nan)]).astype(np.float32)
            valid = np.array([True, True])
            timestamps = np.array([1_000_000, 1_100_000], dtype=np.int64)
            np.savez(
                path,
                positions=positions,
                rotations=rotations,
                valid=valid,
                timestamps=timestamps,
                coordinate_frame="robot_base",
            )
            loaded = load_smoothed_targets(path, "right", limit_frames=None)
            self.assertEqual(loaded.source_format, "smoothed_targets")
            self.assertEqual(loaded.coordinate_frame, "robot_base")
            self.assertIsNone(loaded.hand)
            self.assertEqual(loaded.body.shape, (2, 33, 3))
            np.testing.assert_allclose(loaded.body[:, 16, :], positions + OFFSET)
            np.testing.assert_allclose(loaded.target_rotations, rotations)
            np.testing.assert_array_equal(loaded.orientation_valid, [True, False])
            np.testing.assert_array_equal(loaded.timestamps_us, timestamps)
            self.assertAlmostEqual(loaded.fps, 10.0)

    def test_smoothed_targets_interpolate_missing_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-gap.npz"
            np.savez(
                path,
                positions=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [np.nan, np.nan, np.nan],
                        [2.0, 4.0, 6.0],
                    ],
                    dtype=np.float64,
                ),
                rotations=np.tile(np.eye(3), (3, 1, 1)),
                valid=np.array([True, True, True]),
                timestamps=np.array([0, 100_000, 200_000], dtype=np.int64),
                coordinate_frame="robot_base",
            )
            loaded = load_smoothed_targets(path, "right", None)
            np.testing.assert_allclose(loaded.body[:, 16, :], [[0, 0, 0], [1, 2, 3], [2, 4, 6]] + OFFSET)
            np.testing.assert_array_equal(loaded.orientation_valid, [True, True, True])

    def test_smoothed_targets_apply_legacy_transform_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-legacy.npz"
            positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
            rotations = np.tile(np.eye(3), (1, 1, 1))
            np.savez(
                path,
                positions=positions,
                rotations=rotations,
                valid=np.array([True]),
                timestamps=np.array([0], dtype=np.int64),
            )
            loaded = load_smoothed_targets(path, "right", None)
            np.testing.assert_allclose(loaded.body[:, 16, :], transform_points(positions))
            np.testing.assert_allclose(loaded.target_rotations, transform_rotations_to_robot(rotations))
            np.testing.assert_array_equal(loaded.orientation_valid, [True])

    def test_smoothed_targets_reject_malformed_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-bad.npz"
            np.savez(
                path,
                positions=np.ones((2, 3), dtype=np.float64),
                valid=np.array([True, True]),
                timestamps=np.array([0, 100_000], dtype=np.int64),
            )
            with self.assertRaises(KeyError):
                load_smoothed_targets(path, "right", None)

    def test_processor_smoothed_archive_routes_to_smoothed_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-processor-output.npz"
            positions = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32)
            rotations = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
            valid = np.array([True, False])
            timestamps = np.array([1_000_000, 1_033_333], dtype=np.int64)
            np.savez(
                path,
                positions=positions,
                rotations=rotations,
                rpy=np.zeros((2, 3), dtype=np.float32),
                valid=valid,
                timestamps=timestamps,
            )
            loaded = load_retarget_input(path, "right", 99, 3, None)
            self.assertEqual(loaded.source_format, "smoothed_targets")
            self.assertIsNone(loaded.hand)
            np.testing.assert_allclose(loaded.body[:, 16, :], transform_points(positions))
            np.testing.assert_allclose(loaded.target_rotations, transform_rotations_to_robot(rotations))
            np.testing.assert_array_equal(loaded.orientation_valid, valid)
            np.testing.assert_array_equal(loaded.timestamps_us, timestamps)

    def test_direct_rotation_targets_use_positions_and_valid_mask(self) -> None:
        body = np.zeros((2, 33, 3), dtype=np.float64)
        body[:, 16, :] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        rotations = np.stack([np.eye(3), np.diag([1.0, -1.0, -1.0])])
        targets, valid, filled = build_direct_rotation_targets(
            FakePin, body, rotations, "right", np.array([True, False]),
        )
        np.testing.assert_array_equal(valid, [True, False])
        np.testing.assert_allclose(targets[0].translation, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(targets[1].translation, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(filled[0], np.eye(3))
        np.testing.assert_allclose(filled[1], np.eye(3))

    def test_wrist_position_forces_zero_orientation_cost(self) -> None:
        robot = normalize_robot("ur10")
        from viki.optimization.retarget.retarget_rgb_only import RunConfig

        cfg = RunConfig(
            robot=robot,
            working_hand="right",
            landmark_sg_window=0,
            landmark_sg_polyorder=0,
            ik_position_cost=1.0,
            ik_orientation_cost=0.5,
            ik_posture_cost=1e-3,
            target_mode="wrist_position",
            ik_substeps=20,
            ik_solver="quadprog",
            approach_sec=5.0,
            joint_sg_window=0,
            joint_sg_polyorder=0,
            limit_frames=None,
            recenter_to_neutral=False,
            trajectory_scale=1.0,
            align_initial_orientation=False,
        )
        self.assertEqual(effective_orientation_cost(cfg), 0.0)


if __name__ == "__main__":
    unittest.main()
