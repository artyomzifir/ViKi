"""Tests for skeleton hand pose."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from viki.skeleton.detectors import MediaPipeHand
from viki.skeleton.hand_angles import compute_end_effector_pose, compute_palm_rotation
from viki.skeleton.models import LM, SkeletonFrame
from viki.skeleton.recorder import SkeletonRecorder


def synthetic_points() -> dict[LM, np.ndarray]:
    points = {LM(i): np.zeros(3, dtype=np.float32) for i in range(LM.N)}
    points[LM.WRIST] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    points[LM.MIDDLE_MCP] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    points[LM.PINKY_MCP] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return points


class HandPoseTests(unittest.TestCase):
    def test_palm_rotation_is_identity_for_axis_aligned_hand(self) -> None:
        points = synthetic_points()
        rotation = compute_palm_rotation(
            points[LM.WRIST],
            points[LM.INDEX_MCP],
            points[LM.MIDDLE_MCP],
            points[LM.PINKY_MCP],
        )
        self.assertIsNotNone(rotation)
        assert rotation is not None
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-6)

    def test_palm_rotation_rejects_invalid_inputs(self) -> None:
        points = synthetic_points()
        wrist = points[LM.WRIST]
        index = points[LM.INDEX_MCP]
        middle = points[LM.MIDDLE_MCP]
        pinky = points[LM.PINKY_MCP]

        # Degenerate forward axis (wrist == middle).
        self.assertIsNone(compute_palm_rotation(middle, index, middle, pinky))
        # Degenerate spread (index == pinky).
        self.assertIsNone(compute_palm_rotation(wrist, pinky, middle, pinky))
        # Non-finite landmark.
        bad = middle.copy()
        bad[0] = np.nan
        self.assertIsNone(compute_palm_rotation(wrist, index, bad, pinky))

    def test_end_effector_pose_and_recorder_npz(self) -> None:
        points = synthetic_points()
        pose = compute_end_effector_pose(points, timestamp_us=123)
        self.assertTrue(pose.valid)

        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkeletonRecorder(base_dir=tmp)
            filename = recorder.start()
            recorder.record(
                SkeletonFrame(
                    device_id="cam0",
                    points=points,
                    timestamp_us=123,
                    end_effector=pose,
                )
            )
            saved = recorder.stop()

            self.assertIsNotNone(saved)
            self.assertTrue(filename.startswith("rec-"))
            assert saved is not None
            self.assertTrue(str(saved).endswith(".npz"))

            with np.load(saved) as data:
                self.assertIn("timestamps", data)
                self.assertIn("points", data)
                self.assertIn("landmark_ids", data)
                self.assertEqual(data["points"].shape, (1, LM.N, 3))
                self.assertEqual(data["landmark_ids"].tolist(), list(range(LM.N)))


if __name__ == "__main__":
    unittest.main()
