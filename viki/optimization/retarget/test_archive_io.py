"""Tests for HDF5 archive helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from viki.optimization.retarget.archive_io import load_archive, write_hdf5_archive


class ArchiveIoTests(unittest.TestCase):
    def test_hdf5_archive_round_trips_arrays_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj.h5"
            q = np.arange(12, dtype=np.float64).reshape(3, 4)
            write_hdf5_archive(
                path,
                {
                    "q_scene_smooth": q,
                    "robot": "ur10_official_description",
                    "fps": 30.0,
                    "recenter_to_neutral": True,
                    "ee_target_rot": np.repeat(np.eye(3)[None, :, :], 3, axis=0),
                    "orientation_valid": np.array([True, False, True]),
                },
            )

            with load_archive(path) as archive:
                self.assertIn("q_scene_smooth", archive.files)
                np.testing.assert_allclose(archive["q_scene_smooth"], q)
                self.assertEqual(archive["robot"], "ur10_official_description")
                self.assertEqual(float(archive["fps"]), 30.0)
                self.assertTrue(bool(archive["recenter_to_neutral"]))
                np.testing.assert_allclose(
                    archive["ee_target_rot"],
                    np.repeat(np.eye(3)[None, :, :], 3, axis=0),
                )
                np.testing.assert_array_equal(archive["orientation_valid"], [True, False, True])


if __name__ == "__main__":
    unittest.main()
