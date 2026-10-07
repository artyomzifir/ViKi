"""Rotation smoothing — ``viki.dsp.smooth_rotations``.

Written against a measured failure: on the 40mm-cubes-play set the tracked pose
travelled 7-28x further than the object actually turned, and the operator
confirms the cube was lifted, carried and set down, never turned. A near-
constant orientation is therefore the right answer, and the filter has to reach
it without leaving SO(3) or flattening a rotation that is real.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viki.dsp import geodesic_mean_rotation, smooth_rotations


def _angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("fij,fkj->fik", a, b)
    return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def _steps(track: np.ndarray) -> np.ndarray:
    return _angles(track[1:], track[:-1])


def _jittered(truth: np.ndarray, sigma_deg: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = Rotation.from_rotvec(np.radians(rng.normal(0, sigma_deg, (len(truth), 3))))
    return np.einsum("fij,fjk->fik", truth, noise.as_matrix())


def test_output_is_still_a_rotation():
    truth = np.tile(np.eye(3), (120, 1, 1))
    out = smooth_rotations(_jittered(truth, 3.0, 0))
    np.testing.assert_allclose(
        np.einsum("fij,fkj->fik", out, out), np.tile(np.eye(3), (120, 1, 1)), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(out), 1.0, atol=1e-9)


def test_a_carried_object_comes_out_nearly_still():
    """The measured case: no real rotation, 2-3 deg of per-frame jitter."""
    truth = np.tile(Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix(), (200, 1, 1))
    noisy = _jittered(truth, 2.5, 1)
    out = smooth_rotations(noisy, window=15, polyorder=2)

    # a 15/2 Savitzky-Golay does not erase white noise, it attenuates it; the
    # claim is a large reduction, not silence
    assert np.median(_steps(noisy)) > 2.0
    assert np.median(_steps(out)) < np.median(_steps(noisy)) / 3.0
    assert np.median(_steps(out)) < 0.8
    # and it converged on the true orientation, not just on something smooth
    assert np.median(_angles(out, truth)) < np.median(_angles(noisy, truth)) / 2
    # a 0.5 s window still follows low-frequency noise, so this is not exact
    # recovery of a constant — see the window sweep in
    # docs/scene_object_extraction_eval.md
    assert np.median(_angles(out, truth)) < 2.0


def test_a_real_rotation_survives():
    angle = np.linspace(0, np.radians(90), 200)
    truth = Rotation.from_rotvec(np.c_[angle, np.zeros(200), np.zeros(200)]).as_matrix()
    out = smooth_rotations(_jittered(truth, 2.0, 2), window=15, polyorder=2)
    swept = _angles(out[-1][None], out[0][None])[0]
    assert 85.0 < swept < 95.0                    # the 90 deg turn is still there
    assert np.median(_angles(out, truth)) < 1.5


def test_smoothing_is_disabled_by_a_degenerate_window():
    noisy = _jittered(np.tile(np.eye(3), (50, 1, 1)), 3.0, 3)
    np.testing.assert_array_equal(smooth_rotations(noisy, window=1), noisy)
    np.testing.assert_array_equal(smooth_rotations(noisy, window=0), noisy)


def test_non_finite_frames_are_carried_through():
    noisy = _jittered(np.tile(np.eye(3), (80, 1, 1)), 2.0, 4)
    noisy[10] = np.nan
    out = smooth_rotations(noisy, window=15, polyorder=2)
    assert np.isnan(out[10]).all()
    assert np.isfinite(out[np.arange(80) != 10]).all()
    assert np.median(_steps(out[np.arange(80) != 10])) < 0.5


def test_a_track_too_short_to_filter_is_returned_unchanged():
    noisy = _jittered(np.tile(np.eye(3), (2, 1, 1)), 2.0, 5)
    np.testing.assert_array_equal(smooth_rotations(noisy, window=15), noisy)


def test_geodesic_mean_of_one_rotation_is_itself():
    R = Rotation.from_rotvec([0.4, 0.1, -0.2]).as_matrix()
    np.testing.assert_allclose(geodesic_mean_rotation(R[None]), R, atol=1e-12)


def test_geodesic_mean_rejects_an_empty_stack():
    with pytest.raises(ValueError, match="empty"):
        geodesic_mean_rotation(np.empty((0, 3, 3)))
