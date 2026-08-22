"""Unit tests for offline evaluation helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from viki.optimization.retarget.eval_tracking_error import (  # noqa: E402
    compute_error_metrics,
    compute_error_mm,
    evaluate,
    resample_trajectory,
    rigid_align_points,
    select_q_key,
    select_target_source,
)
from viki.optimization.retarget.smoothing import adjusted_savgol_window, smooth_none  # noqa: E402
from viki.optimization.retarget.archive_io import write_hdf5_archive  # noqa: E402


class EvalTrackingErrorTests(unittest.TestCase):
    def test_select_q_key_prefers_smooth_raw_approach(self) -> None:
        class Archive:
            files = ["q_approach", "q_scene_raw", "q_scene_smooth"]

        self.assertEqual(select_q_key(Archive()), "q_scene_smooth")
        self.assertEqual(select_q_key(Archive(), "q_scene_raw"), "q_scene_raw")

    def test_select_target_source_prefers_ee_target_pos(self) -> None:
        class WithTarget:
            files = ["ee_target_pos", "q_scene_smooth"]

        class WithoutTarget:
            files = ["q_scene_smooth"]

        self.assertEqual(select_target_source(WithTarget()), "ee_target_pos")
        self.assertEqual(select_target_source(WithoutTarget()), "body_wrist")

    def test_rigid_align_points_recovers_rotation_and_translation(self) -> None:
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        theta = np.deg2rad(35.0)
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([0.4, -1.2, 2.0])
        target = source @ rotation.T + translation

        aligned, estimated_rotation, estimated_translation = rigid_align_points(source, target)

        np.testing.assert_allclose(aligned, target, atol=1e-10)
        np.testing.assert_allclose(estimated_rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(estimated_translation, translation, atol=1e-10)
        self.assertGreater(np.linalg.det(estimated_rotation), 0.999)

    def test_resample_trajectory_interpolates_linearly(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]])
        out = resample_trajectory(points, 5)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.5, 1.0],
                [0.5, 1.0, 2.0],
                [0.75, 1.5, 3.0],
                [1.0, 2.0, 4.0],
            ]
        )
        np.testing.assert_allclose(out, expected)

    def test_compute_error_metrics_uses_millimetres(self) -> None:
        robot = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.10, 0.0, 0.0]])
        target = np.zeros((3, 3))
        error_mm = compute_error_mm(robot, target)
        metrics = compute_error_metrics(error_mm, threshold_mm=50.0)

        np.testing.assert_allclose(error_mm, [0.0, 30.0, 100.0])
        self.assertAlmostEqual(metrics["mean_error_mm"], 130.0 / 3.0)
        self.assertAlmostEqual(metrics["median_error_mm"], 30.0)
        self.assertEqual(metrics["num_frames"], 3)
        self.assertAlmostEqual(metrics["frames_under_50mm_pct"], 200.0 / 3.0)

    def test_adjusted_savgol_window_is_odd_and_bounded(self) -> None:
        self.assertEqual(adjusted_savgol_window(length=10, window=20, polyorder=3), 9)
        self.assertEqual(adjusted_savgol_window(length=10, window=4, polyorder=2), 3)
        self.assertEqual(adjusted_savgol_window(length=5, window=1, polyorder=3), 5)

    def test_smooth_none_returns_float_copy(self) -> None:
        points = np.array([[1, 2, 3]], dtype=np.int32)
        out = smooth_none(points)
        self.assertEqual(out.dtype, np.float64)
        np.testing.assert_allclose(out, points)
        self.assertIsNot(out, points)

    def test_evaluate_smoke_with_mocked_fk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            human = root / "human.npz"
            robot_traj = root / "robot_traj.h5"
            out = root / "eval"

            body = np.zeros((5, 33, 3), dtype=np.float64)
            body[:, 16, 0] = np.linspace(0.0, 0.04, len(body))
            np.savez(human, body=body, fps=10.0, working_hand="right")
            write_hdf5_archive(
                robot_traj,
                {
                    "q_scene_smooth": np.zeros((5, 6), dtype=np.float64),
                    "robot": "ur10_official_description",
                    "ee_frame": "tool0",
                    "working_hand": "right",
                },
            )

            def fake_fk(_robot_description, q_traj, _ee_frame):
                positions = body[:, 16, :].copy()
                rotations = np.repeat(np.eye(3)[None, :, :], len(q_traj), axis=0)
                return positions, rotations

            args = type(
                "Args",
                (),
                {
                    "human": str(human),
                    "robot_traj": str(robot_traj),
                    "robot": None,
                    "ee_frame": None,
                    "q_key": "auto",
                    "target_source": "body_wrist",
                    "hand": None,
                    "smoothing": "none",
                    "smooth_window": 15,
                    "smooth_polyorder": 3,
                    "align": "none",
                    "threshold_mm": 50.0,
                    "out": str(out),
                },
            )()

            with (
                patch("viki.optimization.retarget.eval_tracking_error.load_robot_poses", side_effect=fake_fk),
                patch("viki.optimization.retarget.eval_tracking_error.save_error_plot"),
                patch("viki.optimization.retarget.eval_tracking_error.save_trajectory_plot"),
            ):
                metrics = evaluate(args)

            self.assertEqual(metrics["q_key"], "q_scene_smooth")
            self.assertEqual(metrics["target_source"], "body_wrist")
            self.assertEqual(metrics["num_frames"], 5)
            self.assertAlmostEqual(metrics["mean_error_mm"], 0.0)
            self.assertTrue(out.with_name(out.name + "_metrics.json").exists())
            self.assertTrue(out.with_name(out.name + "_aligned.h5").exists())


if __name__ == "__main__":
    unittest.main()
