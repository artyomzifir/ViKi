"""Tests for batch palm-orientation validity checks."""

from __future__ import annotations

import unittest

import numpy as np

from viki.skeleton.models import LM
from viki.optimization.preparation.processor import stable_palm_orientation_mask


class ProcessorOrientationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.zeros((5, LM.N, 3), dtype=np.float64)
        self.points[:, LM.MIDDLE_MCP] = [1.0, 0.0, 0.0]
        self.points[:, LM.THUMB_CMC] = [0.0, 1.0, 0.0]
        self.ids = np.arange(LM.N, dtype=np.int32)
        self.rotations = np.tile(np.eye(3), (5, 1, 1))
        self.pose_valid = np.ones(5, dtype=bool)

    def test_rejects_implausible_palm_bone_length(self) -> None:
        self.points[2, LM.THUMB_CMC] = [0.0, 10.0, 0.0]

        valid = stable_palm_orientation_mask(
            self.points,
            self.ids,
            self.rotations,
            self.pose_valid,
        )

        np.testing.assert_array_equal(valid, [True, True, False, True, True])

    def test_rejects_large_adjacent_rotation_jump(self) -> None:
        angle = np.radians(120.0)
        self.rotations[2] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]

        valid = stable_palm_orientation_mask(
            self.points,
            self.ids,
            self.rotations,
            self.pose_valid,
        )

        np.testing.assert_array_equal(valid, [True, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
