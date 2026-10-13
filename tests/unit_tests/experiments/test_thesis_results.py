"""Thesis summary helpers should keep their units and selection rules."""

import numpy as np
import pytest

from scripts.build_thesis_results import blue_extent_mm, longest_run, wilson


def test_wilson_interval_for_all_successes_and_failures():
    success_lower, success_upper = wilson(30, 30)
    failure_lower, failure_upper = wilson(0, 30)
    assert success_lower == pytest.approx(0.88649, abs=1e-5)
    assert success_upper == pytest.approx(1.0)
    assert failure_lower == pytest.approx(0.0)
    assert failure_upper == pytest.approx(1 - success_lower)


def test_longest_run_counts_consecutive_unsupported_frames():
    assert longest_run(np.array([], dtype=bool)) == 0
    assert longest_run(np.array([False, True, True, False, True])) == 2
    assert longest_run(np.array([True, True, True])) == 3


def test_blue_extent_selects_manipulated_object_and_quantiles(tmp_path):
    path = tmp_path / "objects.npz"
    background = np.full((10, 3), 10.0)
    blue = np.column_stack((np.linspace(0, 0.04, 101),
                            np.linspace(0, 0.05, 101),
                            np.linspace(0, 0.06, 101)))
    np.savez(path, object_labels=np.array(["background", "manipulated_object"]),
             model_offsets=np.array([0, len(background), len(background) + len(blue)]),
             model_xyz_object=np.vstack((background, blue)))
    np.testing.assert_allclose(blue_extent_mm(path), [38.4, 48.0, 57.6])
